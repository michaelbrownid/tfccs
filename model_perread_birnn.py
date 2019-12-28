import tensorflow as tf
import tensorflow.keras as KK
import sys
import numpy as np
from . import struct

# TypeError: ('Not JSON Serializable:', <function fubarClosure.<locals>.func at 0x7fd4c6109c80>)
def fubarClosure(fubarIndex):
    def func( xx ):
        return( xx[:, fubarIndex ,:,:] )
    return(func)


class Model():
    """ Per read model:

Input:
- aln : the alignment information
- callProp: where the proposed reference changes HP. MAYBE?
- propHP,propKmer: context in the proposed reference. MAYBE?

Output:
- directHP: the correct HP (identity+len)
- callTrue: when the model should truly call a new HP


- Take input aln.

- embed first integer in per-base info into 8-D and concatenate snr,
ipd, pw to get 11 vector for every base.

(Note there is a forward/RC for the strand AND a forward/backward for
the bidirectional RNN)

- For each forwardSense read [0:7]:
  - Apply forwardSense model not-Backwards to get per column predictions
  - Apply forwardSense model go-Backwards  to get per column predictions

- For each reverseSense read [8:15]:
  - Apply RCSense model not-Backwards to get per column predictions
  - Apply RCSense model go-Backwards  to get per column predictions

- At this point I have 4 column-prediction vectors per MSA column.

- Concatenate the vectors and feed into 3 softmax's to predict
[hpBase,hpLength,notcall/call]

- The loss is the sum of KLs when "callTrue"==1 AND baseint!=26 (the
alignment says make call and the read has a base there. This might
have issues at ends of reads).

- This will produce HP+call prediction vector for every subread.

- First see if this does well. Then take all the calls, combine them,
and make a consensus call.

    """
    #readdat = KK.layers.Lambda( lambda xx: xx[:,1,:,:],name="readdat")(merged) # [None, 640, 12]
    # np.choose???
    #readdat = KK.layers.Lambda( lambda xx: np.take(xx, 1, axis=1 ), name="readdat")(merged) # [None, 640, 12] )
    #fubarIndex = tf.cast(tf.reduce_mean(readNumber),dtype=tf.int32) # tf.reduce_mean(readNumber).shape=()

    def __init__(self, args):

        self.args = args

        inputs = KK.layers.Input(shape=(args.rows,args.cols,args.baseinfo), name="inputs") # [None, 16, 640, 4]

        inputsCallProp = KK.layers.Input(shape=(args.cols,1), name="inputsCallProp") # [None, 640, 1]

        inputsCallTrue = KK.layers.Input(shape=(args.cols,1), name="inputsCallTrue") # [None, 640, 1]
        inputsCallTrueHack = KK.layers.Lambda( lambda x: x)(inputsCallTrue) # fixes error cannot be fed and fetched

        readNumber = KK.layers.Input(shape=(1,), name="readNumber") #TODO is this the only way to feed parameter?
        readNumberHack = KK.layers.Lambda( lambda x: x)(readNumber) # fixes error cannot be fed and fetched

        #### take the baseint and embed for all
        baseint = KK.layers.Lambda( lambda xx: xx[:,:,:,0], name="baseint" )(inputs) # [None, 16, 640]
        embed = KK.layers.Embedding(input_dim=1, output_dim=8, name="embed")(baseint) # [None, 16, 640, 8]
        merged     = KK.layers.Concatenate(axis= -1,name="merged")([inputs,embed]) # [None, 16, 640, 12] 12=4+8

        #### one forward read. TODO: compute for all; different loss functions for each read if base!=0 used in loss!
        fubarIndex1 = KK.layers.Lambda( lambda xx: KK.backend.mean(xx))(readNumber) # tf.reduce_mean
        fubarIndex2 = KK.layers.Lambda( lambda xx: KK.backend.cast(xx,dtype=tf.int32))(fubarIndex1)
        #fu = fubarClosure( fubarIndex2 )

        #readdat = KK.layers.Lambda( lambda xx: xx[:, fubarIndex2 ,:,:],name="readdat")(merged) # [None, 640, 12]
        #readdat = KK.layers.Lambda( lambda xx: fu(xx), name="readdat")(merged)
        readdat = KK.layers.Lambda( lambda xx: xx[0][:, xx[1] ,:,:], name="readdat")( (merged,fubarIndex2) )

        #### merge inputsCallProp
        readdatmerge = KK.layers.Concatenate(axis= -1,name="readdatmerge")([readdat, inputsCallProp]) # [None, 640, 13]

        #### forwardsense RNN (?, 640, rnn_hidden_size)
        rnn_hidden_size= 256 # 256 does not seem to help

        #### try GRU->LSTM or SimpleRNN
        # 1st RNN layer
        forwardNotBack0 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=False, name="forwardNotBack0")(readdatmerge) # [None, 640, 64]
        forwardYesBack0 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=True, name="forwardYesBack0")(readdatmerge) # [None, 640, 64]

        # 2nd RNN layer
        forwardNotBack = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=False, name="forwardNotBack")(forwardNotBack0) # [None, 640, 64]
        forwardYesBack = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=True, name="forwardYesBack")(forwardYesBack0) # [None, 640, 64]

        rnnconcat = KK.layers.Concatenate(axis= -1,name="rnnconcat")([forwardNotBack, forwardYesBack]) # [None, 640, 128]

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

        #### compute loss only where true call is made
        def loss_sparse_kl_closure( callTrue,baseint ):
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

        #### compute loss only where true call is made exponential weighting for hplen
        def loss_EXP_sparse_kl_closure( callTrue,baseint ):
            # compute exponential weights over 33 possibilities, kmers drop off exponentially
            #expweights = np.power(2.0, range(0,33))
            expweights = np.array( [1.0]*33, dtype=np.float32)

            def sparse_kl(y_pred,y_true):
                eps = 1.0E-9 # 0.0 or too small causes NAN loss!
                #kl = (y_true+eps)*(tf.math.log(y_true+eps)-tf.math.log(y_pred+eps))
                kl = (y_true*expweights)*(-tf.math.log(y_pred+eps))
                klsum = tf.reduce_sum(kl,axis=-1, keepdims=True) # sum up kl components, only 1 non-zero

                # only take loss where there is a true call and there is a base
                sparsekl1 = tf.multiply(klsum, callTrue) # callTrue is 0/1
                sparsekl = tf.multiply(sparsekl1, readdat[:,:,0]!=0) # only where baseint!=0

                print("**kl.shape",kl.shape)
                print("**klsum.shape",klsum.shape)
                print("**callTrue.shape",callTrue.shape)
                print("**sparsekl.shape",sparsekl.shape)
                return(sparsekl)
            return(sparse_kl)
        #mylossEXP = loss_EXP_sparse_kl_closure(inputsCallTrue,readdat)

        def zero_loss(y_true, y_pred):
            # print("zero_loss y_pred.shape", y_pred.shape)
            # print("zero_loss y_true.shape", y_true.shape)
            #return(KK.backend.zeros_like(y_pred))
            return(KK.backend.zeros_like(y_true))

        #### different losses:
        # predHPBase: kl only at calltrue==1. don't penalize outside real calls
        # predHPLen:  kl only at calltrue==1. don't penalize outside real calls
        # predHPCall: kl everywhere. Must get call right everywhere
        # inputsCallTrue: no loss, hack to get model save

        # "kullback_leibler_divergence", myloss, mylossEXP
        # loss_weights=[0.0,1.0,0.0,0.0])
        self.model.compile(optimizer=myopt, 
                           loss=["sparse_categorical_crossentropy","categorical_crossentropy",zero_loss,zero_loss])
                           
