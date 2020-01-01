import tensorflow as tf
import tensorflow.keras as KK
import sys
import numpy as np
from . import struct

class Model():

    def __init__(self, args):

        self.args = args

        inputs = KK.layers.Input(shape=(args.rows,args.cols,args.baseinfo), name="inputs") # [None, 16, 640, 4] The MSA
        inputsCallProp = KK.layers.Input(shape=(args.cols,1), name="inputsCallProp") # [None, 640, 1] Where POA (not true) made a call.
        readNumber = KK.layers.Input(shape=(1,), name="readNumber") # single read row to examine. TODO is this the only way to feed parameter?
        inputsCallTrue = KK.layers.Input(shape=(args.cols,1), name="inputsCallTrue") # [None, 640, 1] The true call optionally used only in loss

        readNumberHack = KK.layers.Lambda( lambda x: x)(readNumber) # fixes error cannot be fed and fetched
        inputsCallTrueHack = KK.layers.Lambda( lambda x: x)(inputsCallTrue) # fixes error cannot be fed and fetched

        #### take the baseint and embed for all. full rank as baseint is different everywhere
        baseint = KK.layers.Lambda( lambda xx: xx[:,:,:,0], name="baseint" )(inputs) # [None, 16, 640]
        embed =   KK.layers.Embedding(input_dim=1, output_dim=8, name="embed")(baseint) # [None, 16, 640, 8]
        merged =  KK.layers.Concatenate(axis= -1,name="merged")([inputs,embed]) # [None, 16, 640, 12] 12=4+8

        #### pick out one read that is specified in readNumber
        oneread = KK.layers.Lambda( lambda xx: KK.backend.mean(xx))(readNumber)
        onereadIndex = KK.layers.Lambda( lambda xx: KK.backend.cast(xx,dtype=tf.int32))(oneread)
        readdat = KK.layers.Lambda( lambda xx: xx[0][:, xx[1] ,:,:], name="readdat")( (merged,onereadIndex) ) # (None,640,12)

        #### single readNumber in so which read can influence model readNumber(None,1)->(None,640,1)
        readNumberExp = KK.layers.Lambda( lambda xx: KK.backend.expand_dims(xx, axis=1), name="readNumberExp")( readNumber ) # (None,1,1)
        readNumberTile = KK.layers.Lambda( lambda xx: KK.backend.tile(xx, (1,args.cols,1)), name="readNumberTile")( readNumber ) # (None,640,1)
 
        #### column coverage, so the model can know approx how many reads went into the POA, therefore POA accuracy
        colcoverage = KK.layers.Lambda( lambda xx: KK.backend.sum(xx!=0,axis=1), name="colcoverage")(baseint) # [None,640]

        #### concat all together [None, 640, 12 +1+1+1] = [None, 640, 15]
        readdatConcat = KK.layers.Concatenate(axis= -1,name="readdatConcat")([readdat, readNumberTile, colcoverage, inputsCallProp])

        #### forwardsense RNN (?, 640, rnn_hidden_size)
        rnn_hidden_size= 256 # 256 does not seem to help

        #### try GRU->LSTM or SimpleRNN
        # 0th RNN layer
        forwardNotBack0 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=False, name="forwardNotBack0")(readdatConcat)
        forwardYesBack0 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=True, name="forwardYesBack0")(readdatConcat) 

        # 1st RNN layer
        forwardNotBack1 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=False, name="forwardNotBack1")(forwardNotBack0)
        forwardYesBack1 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=True, name="forwardYesBack1")(forwardYesBack0)

        # 2nd RNN layer
        forwardNotBack = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=False, name="forwardNotBack")(forwardNotBack1)
        forwardYesBack = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=True, name="forwardYesBack")(forwardYesBack1)

        rnnconcat = KK.layers.Concatenate(axis= -1,name="rnnconcat")([forwardNotBack, forwardYesBack]) # [None, 640, 512]

        #### make predications on HP base, len, call
        predHP = KK.layers.Dense(133, activation='softmax',name="predHPBase")(rnnconcat) #  [None, 640, 133] windowAlignment.py 0:132
        predHPCall =  KK.layers.Dense(2, activation='softmax',name="predHPCall")(rnnconcat) # [None, 640, 2]

        ################################
        self.model = KK.models.Model(inputs=[inputs,inputsCallProp,inputsCallTrue,readNumber],
                                     outputs=[predHP,predHPCall,inputsCallTrueHack,readNumberHack]) 
        
        # hack: inputsCallTrue passed back out for model saving
        # This is what happens if you bring in input that is not used:
        # model.model.save(args.modelsave)
        # ValueError: Could not pack sequence. Structure had 3 elements, but
        # flat_sequence had 2 elements.  Structure: 
        # [<tf.Tensor 'inputs:0' shape=(?, 16, 640, 4) dtype=float32>, 
        # <tf.Tensor 'inputsCallProp:0' shape=(?, 640, 1) dtype=float32>, 
        # <tf.Tensor 'inputsCallTrue:0' shape=(?, 640, 1) dtype=float32>], 
        # flat_sequence: [<tensorflow.python.keras.utils.tf_utils.ListWrapper object at
        # 0x7f924475c160>, <tensorflow.python.keras.utils.tf_utils.ListWrapper object
        # at 0x7f924475cb00>].

        #self.model.summary()
        print("================================")
        for layer in self.model.layers:
            print("layer: name",layer.name, "outputshape", layer.get_output_at(0).get_shape().as_list())
        print("================================")

        #myopt = KK.optimizers.SGD()
        myopt = KK.optimizers.Adam()

        #### compute loss only where true call is made. kl only at calltrue==1. don't penalize outside real calls
        def loss_sparse_kl_closure( callTrue,baseint ):
            # compute exponential weights over 33 possibilities, kmers drop off exponentially
            #expweights = np.power(2.0, range(0,33))
            def sparse_kl(y_pred,y_true):
                eps = 1.0E-9 # 0.0 or too small causes NAN loss!
                #kl = (y_true+eps)*(tf.math.log(y_true+eps)-tf.math.log(y_pred+eps))
                kl = (y_true)*(-tf.math.log(y_pred+eps))
                klsum = tf.reduce_sum(kl,axis=-1, keepdims=True) # sum up kl components, only 1 non-zero

                # only take loss where there is a true call and there is a base
                sparsekl1 = tf.multiply(klsum, callTrue) # callTrue is 0/1
                sparsekl = tf.multiply(sparsekl1, readdat[:,:,0]!=0) # only where baseint!=0

                print("*kl.shape",kl.shape)
                print("*klsum.shape",klsum.shape)
                print("*callTrue.shape",callTrue.shape)
                print("*sparsekl.shape",sparsekl.shape)
                return(sparsekl)
            return(sparse_kl)
        #myloss = loss_sparse_kl_closure(inputsCallTrue,readdat)

        def my_sparse_categorical_crossentropy( y_pred,y_true ):
            """Compute sparse_categorical_crossentropy: y_pred = probDistribution, y_true=Integer

            Only where the read is covered: basedat!=0
            """
            pass

        def zero_loss(y_true, y_pred):
            # print("zero_loss y_pred.shape", y_pred.shape)
            # print("zero_loss y_true.shape", y_true.shape)
            #return(KK.backend.zeros_like(y_pred))
            return(KK.backend.zeros_like(y_true))

        #### different losses:

        # "kullback_leibler_divergence"
        # loss_weights=[0.0,1.0,0.0,0.0])
        self.model.compile(optimizer=myopt, 
                           loss=["sparse_categorical_crossentropy","sparse_categorical_crossentropy",zero_loss,zero_loss])
                           
