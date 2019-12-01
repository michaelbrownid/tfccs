import tensorflow as tf
import tensorflow.keras as KK
import sys
import numpy as np
from . import struct

class Model():
    def __init__(self, args):

        self.args = args
        """class args:
            rows=16
            cols=640
            baseinfo=10
            hps = 128
            hpdist = 33
        """

        # inputs = KK.layers.Input(shape=(args.rows,args.cols,args.baseinfo))
        # x = KK.layers.Conv2D(8, kernel_size= (1, 6), strides=(1,5), activation='relu')(inputs)
        # x = KK.layers.Conv2D(8, (16, 1), activation='relu')(x)
        # bottle = KK.layers.Flatten()(x) # This is now the bottleneck
        # predictionsHPLEN = KK.layers.Dense(args.hpdist, activation='softmax')(bottle)
        # predictionsHPID = KK.layers.Dense(4, activation='softmax')(bottle)

        mystruct = struct.struct()
        myparam = mystruct.fixed2()
        myparamhash = hash( frozenset(myparam.items()))
        print(">>>>myparam",myparamhash)
        print(myparam)
        print(">>>>myparam",myparamhash)

        inputs = KK.layers.Input(shape=(args.rows,args.cols,args.baseinfo))
        baseadj = KK.layers.Conv2D(myparam["P_0NumKern"], kernel_size= (1, 6), strides=(1,5), activation='relu')(inputs)
        #majority = KK.layers.average([baseadj[:,xx] for xx in range(16)])
        majority = KK.backend.mean(baseadj, axis=[1])
        lengths = KK.layers.Conv1D(myparam["LengthsNumKern"], kernel_size= myparam["LengthsYKern"], activation='relu')(majority)
        #bottle = KK.layers.Flatten()(majority)
        predictionsHPLEN = KK.layers.Dense(33, activation='softmax')(lengths[:,0,:]) # take only 0th position
        #predictionsHPID = KK.layers.Dense(4, activation='softmax')(bottle)

        ################################
        #self.model = KK.models.Model(inputs=inputs, outputs=[predictionsHPID,predictionsHPLEN])
        self.model = KK.models.Model(inputs=inputs, outputs=[predictionsHPLEN])
        self.model.summary()
        self.model.compile(optimizer="adam", loss="categorical_crossentropy") #, metrics=["categorical_accuracy","kullback_leibler_divergence"])

        # # instrument tensorboard
        # tf.summary.histogram('output', self.output)
        # tf.summary.histogram('softmax_w', softmax_w)
        # tf.summary.histogram('logits', self.logits)
        # tf.summary.histogram('probs', self.probs)
        # tf.summary.scalar('train_loss', self.cost)
