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

        inputs = KK.layers.Input(shape=(args.rows,args.cols,args.baseinfo))

        baseadj = KK.layers.Conv2D(128, kernel_size= (1, 6), strides=(1,5), activation='relu', name="baseadj")(inputs)

        perm = KK.layers.Permute((2,3,1),name="perm")(baseadj) # [None, 16, 127, 64] -> [None, 127, 64, 16]

        majorityIn = KK.layers.Conv2D(64, kernel_size= (1, 1), activation='relu', name="majorityIn")(perm)

        majority = KK.backend.mean(majorityIn, axis=[1])

        lengths = KK.layers.Conv1D(33, kernel_size= 33, activation='relu', name="lengths")(majority)

        #bottle = KK.layers.Flatten()(majority)

        lengthszero = lengths[:,0,:] # take only 0th position
        
        predictionsHPLEN = KK.layers.Dense(33, activation='softmax', name="predictionsHPLEN")(lengthszero)
        #predictionsHPID = KK.layers.Dense(4, activation='softmax')(bottle)

        ################################
        #self.model = KK.models.Model(inputs=inputs, outputs=[predictionsHPID,predictionsHPLEN])
        self.model = KK.models.Model(inputs=inputs, outputs=[predictionsHPLEN])

        #self.model.summary()
        print("================================")
        for layer in self.model.layers:
            print("layer: name",layer.name, "outputshape", layer.get_output_at(0).get_shape().as_list())
        print("================================")

        self.model.compile(optimizer="adam", loss="categorical_crossentropy") #, metrics=["categorical_accuracy","kullback_leibler_divergence"])

        # # instrument tensorboard
        # tf.summary.histogram('output', self.output)
        # tf.summary.histogram('softmax_w', softmax_w)
        # tf.summary.histogram('logits', self.logits)
        # tf.summary.histogram('probs', self.probs)
        # tf.summary.scalar('train_loss', self.cost)
