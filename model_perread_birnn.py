import tensorflow as tf
import tensorflow.keras as KK
import sys
import numpy as np

class Model():

    def __init__(self, args):

        self.args = args

        inputs = KK.layers.Input(shape=(args.rows,args.cols,args.baseinfo), name="inputs") # [None, 16, 640, 4] The MSA
        inputsCallProp = KK.layers.Input(shape=(args.cols,1), name="inputsCallProp") # [None, 640, 1] Where POA (not true) made a call.
        readNumber = KK.layers.Input(shape=(1,), name="readNumber") # single read row to examine. TODO is this the only way to feed parameter?
        inputsCallTrue = KK.layers.Input(shape=(args.cols,1), name="inputsCallTrue") # [None, 640, 1] The true call optionally used only in loss

        readNumberHack = KK.layers.Lambda( lambda x: x)(readNumber) # fixes error cannot be fed and fetched
        inputsCallTrueHack = KK.layers.Lambda( lambda x: x)(inputsCallTrue) # fixes error cannot be fed and fetched

        #### pick out one read that is specified in readNumber
        oneread = KK.layers.Lambda( lambda xx: KK.backend.mean(xx))(readNumber)
        onereadIndex = KK.layers.Lambda( lambda xx: KK.backend.cast(xx,dtype=tf.int32))(oneread)
        readdat = KK.layers.Lambda( lambda xx: xx[0][:, xx[1] ,:,:], name="readdat")( (inputs,onereadIndex) ) # (None,640,4)

        #### take the baseint and embed.
        readbaseint = KK.layers.Lambda( lambda xx: xx[:,:,0], name="baseint" )(readdat) # [None, 640]
        embedBase =   KK.layers.Embedding(input_dim=1, output_dim=8, name="embed")(readbaseint) # [None, 640, 8]

        #### single readNumber in so which read can influence model readNumber(None,1)->(None,640,1)
        readNumberExp = KK.layers.Lambda( lambda xx: KK.backend.expand_dims(xx, axis=1), name="readNumberExp")( readNumber ) # (None,1,1)
        readNumberTile = KK.layers.Lambda( lambda xx: KK.backend.tile(xx, (1, 640 ,1) ), name="readNumberTile")( readNumberExp ) # (None,640,1) AttributeError: 'str' object has no attribute 'cols'
 
        #### coverage over all input subreads at each column, so the model can know approx how many reads went into the POA, therefore POA accuracy
        colNotEmpty = KK.layers.Lambda( lambda xx: KK.backend.cast( KK.backend.not_equal( xx[:,:,:,0],0.0 ), dtype=tf.float32), name="colNotEmpty")(inputs) # [None,16,640]
        colcoverage = KK.layers.Lambda( lambda xx: KK.backend.expand_dims( KK.backend.sum( xx, axis=1) ), name="colcoverage")(colNotEmpty) # [None,640,1]

        #### concat all together [None, 640, 4+8+1+1+1] = [None, 640, 15]
        readdatConcat = KK.layers.Concatenate(axis= -1,name="readdatConcat")([readdat, embedBase, readNumberTile, colcoverage, inputsCallProp])

        #### forwardsense RNN (?, 640, rnn_hidden_size)
        rnn_hidden_size= 256

        #### try GRU->LSTM or SimpleRNN
        # 0th RNN layer
        forwardNotBack0 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=False, name="forwardNotBack0")(readdatConcat)
        forwardYesBack0 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=True, name="forwardYesBack0")(readdatConcat) 

        # 1st RNN layer
        #forwardNotBack1 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=False, name="forwardNotBack1")(forwardNotBack0)
        #forwardYesBack1 = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=True, name="forwardYesBack1")(forwardYesBack0)

        # last RNN layer
        forwardNotBack = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=False, name="forwardNotBack")(forwardNotBack0)
        forwardYesBack = KK.layers.GRU( rnn_hidden_size, return_sequences=True, go_backwards=True, name="forwardYesBack")(forwardYesBack0)

        rnnconcat = KK.layers.Concatenate(axis= -1,name="rnnconcat")([forwardNotBack, forwardYesBack]) # [None, 640, 512]

        #### make predications on HP base, len, call
        predHP0 = KK.layers.Dense(194, activation='elu',name="predHP0")(rnnconcat) 
        predHP  = KK.layers.Dense(133, activation='softmax',name="predHP")(predHP0) #  [None, 640, 133] windowAlignment.py 0:132
        
        predHPCall0=  KK.layers.Dense(128, activation='elu',name="predHPCall0")(rnnconcat)
        predHPCall =  KK.layers.Dense(2, activation='softmax',name="predHPCall")(predHPCall0) # [None, 640, 2]

        if False:
            #### make uniform loss everywhere there is no base covering
            basecovered    = KK.layers.Lambda( lambda xx: KK.backend.expand_dims( KK.backend.cast( KK.backend.not_equal( xx[:,:,0], 0.0 ), dtype=tf.float32), axis=-1 ), name="basecovered")(   readdat)
            notbasecovered = KK.layers.Lambda( lambda xx: 1.0-xx, name="notbasecovered")(basecovered)

            #### !!! Not KK.layers.Multiply() and Add() !!!
            #predHPUniform = KK.backend.constant( 1.0/133.0, dtype='float32', shape=[batchsize,args.cols,133], name="predHPUniform" )
            predHPUniform = KK.backend.ones_like(predHP)*(1.0/133.0)
            ma = tf.multiply( basecovered,predHP)
            mb = tf.multiply( notbasecovered,predHPUniform)
            predHPCovered =  tf.add( ma, mb) 

            #predHPCallUniform = KK.backend.constant( 1.0/2.0, dtype='float32',   shape=(batchsize,args.cols,2),  name="predHPCallUniform" )
            predHPCallUniform = KK.backend.ones_like(predHPCall)*(1.0/2.0)
            mc = tf.multiply(basecovered,predHPCall)
            md = tf.multiply(notbasecovered,predHPCallUniform)
            predHPCallCovered = tf.add( mc, md )
            
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

        #### TODO: BUG!!! OLD work has y_pred and y_true SWITCHED!!!!!
        #### compute loss only where true call is made. kl only at calltrue==1. don't penalize outside real calls

        # def loss_sparse_kl_closure( callTrue,baseint ):
        #     # compute exponential weights over 33 possibilities, kmers drop off exponentially
        #     #expweights = np.power(2.0, range(0,33))
        #     def sparse_kl(y_true,y_pred):
        #         eps = 1.0E-9 # 0.0 or too small causes NAN loss!
        #         #kl = (y_true+eps)*(tf.math.log(y_true+eps)-tf.math.log(y_pred+eps))
        #         kl = (y_true)*(-tf.math.log(y_pred+eps))
        #         klsum = tf.reduce_sum(kl,axis=-1, keepdims=True) # sum up kl components, only 1 non-zero

        #         # only take loss where there is a true call and there is a base
        #         sparsekl1 = tf.multiply(klsum, callTrue) # callTrue is 0/1
        #         sparsekl = tf.multiply(sparsekl1, KK.backend.cast(readdat[:,:,0]!=0.0,'float32')) # only where baseint!=0

        #         print("*kl.shape",kl.shape)
        #         print("*klsum.shape",klsum.shape)
        #         print("*callTrue.shape",callTrue.shape)
        #         print("*sparsekl.shape",sparsekl.shape)
        #         return(sparsekl)
        #     return(sparse_kl)
        # #myloss = loss_sparse_kl_closure(inputsCallTrue,readdat)

        def myclosure( numonehot ):
            def my_sparse_categorical_crossentropy( y_true, y_pred ):
                """Compute sparse_categorical_crossentropy: y_pred = probDistribution, y_true=Integer

                Only where the read is covered: basedat!=0
                """

                y_true= KK.backend.cast( y_true[:,:,0],'int32') # so must take the integer at [0] AND feed as [batch,col,1]!

                print("*y_true.shape",y_true.shape)
                print("*y_true.dtype",y_true.dtype)
                print("*y_pred.shape",y_pred.shape)
                print("*y_pred.dtype",y_pred.dtype)
                # *y_true.shape (?, ?, ?)
                # *y_true.dtype <dtype: 'int32'>
                # *y_pred.shape (?, 640, 133)
                # *y_pred.dtype <dtype: 'float32'>
                # Traceback (most recent call last):

                eps = 1.0E-9 # 0.0 or too small causes NAN loss!

                #EXPAND onehot
                onehot = KK.backend.one_hot(y_true,numonehot) # onehot.shape (?, ?, 133)
                print("onehot.shape",onehot.shape)
                logloss = -tf.math.log(y_pred+eps) # logloss.shape (?, 640, 133)
                print("logloss.shape",logloss.shape)
                klall = tf.multiply(logloss,onehot)
                kl = tf.reduce_sum(klall,axis=-1, keepdims=True) # sum up kl components, only 1 non-zero
                print("*kl.shape",kl.shape)

                # only take loss where there is a true call and there is a base
                # this is a closure because it refers to readbaseint outside
                #sparsekl1 = tf.multiply(kl, inputsCallTrue) # callTrue is 1 when true call is made
                present = KK.backend.expand_dims(KK.backend.cast( KK.backend.not_equal( readdat[:,:,0], 0.0 ), dtype=tf.float32),axis= -1)
                print("present.shape",present.shape)
                sparsekl = tf.multiply(kl, present ) # only where readbaseint!=0
                print("*sparsekl.shape",sparsekl.shape)

                #print("*inputsCallTrue.shape",inputsCallTrue.shape)
                #print("*sparsekl1.shape",sparsekl1.shape)

                return( tf.reduce_mean(sparsekl))

            return(my_sparse_categorical_crossentropy) # closure


        def zero_loss(y_true, y_pred):
            # print("zero_loss y_pred.shape", y_pred.shape)
            # print("zero_loss y_true.shape", y_true.shape)
            #return(KK.backend.zeros_like(y_pred))
            return(KK.backend.zeros_like(y_true))

        # "kullback_leibler_divergence" "sparse_categorical_crossentropy"
        # loss_weights=[0.0,1.0,0.0,0.0])
        self.model.compile(optimizer=myopt, 
                           loss=[myclosure(133), myclosure(2), zero_loss,zero_loss])
                           
