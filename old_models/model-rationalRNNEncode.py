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

        majority = KK.backend.mean(baseadj, axis=[1])
        # layer: name Mean outputshape [None, 127, 64]

        hidden_size= 128
        #KK.layers.TimeDistributed(majority)
        rnn1 = KK.layers.LSTM( hidden_size, return_sequences=True)(majority) # input_shape=(1,64), 

        predictionsHPLEN = KK.layers.Dense(33, activation='softmax')(rnn1)
        predictionsHPID = KK.layers.Dense(4, activation='softmax')(rnn1)
        predictionsCALL = KK.layers.Dense(2, activation='softmax')(rnn1)

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
