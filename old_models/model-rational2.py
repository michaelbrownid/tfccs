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

        baseadj = KK.layers.Conv2D(64, kernel_size= (1, 6), strides=(1,5), activation='relu')(inputs)

        #majority = KK.layers.average([baseadj[:,xx] for xx in range(16)])
        majority = KK.backend.mean(baseadj, axis=[1])

        lengths = KK.layers.Conv1D(33, kernel_size= 33, activation='relu')(majority)

        #bottle = KK.layers.Flatten()(majority)

        lengthszero = lengths[:,0,:] # take only 0th position
        layerA = KK.layers.Dense(128, activation='relu')(lengthszero)
        
        predictionsHPLEN = KK.layers.Dense(33, activation='softmax')(layerA) # take only 0th position
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
