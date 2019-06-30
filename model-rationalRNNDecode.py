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
        baseadj = KK.layers.Conv2D(128, kernel_size= (1, 6), strides=(1,5), activation='relu')(inputs)
        majority = KK.backend.mean(baseadj, axis=[1])
        #layer: name Mean outputshape [None, 127, 128]

        lengths = KK.layers.Conv1D(33, kernel_size= 33, activation='relu')(majority)
        print("lengths.shape",lengths.shape)

        #bottle = KK.layers.Flatten(lengths)
        bottle1 = KK.layers.Reshape((lengths.shape[1]*lengths.shape[2],1))(lengths)
        bottle = KK.backend.squeeze(bottle1,2)
        print("bottle.shape",bottle.shape)

        bottleConstantTime = KK.layers.RepeatVector(args.hps)(bottle) # repeat the bottle for each HP in output

        hidden_size= 256
        #KK.layers.TimeDistributed(majority)
        rnn1 = KK.layers.GRU( hidden_size, return_sequences=True)(bottleConstantTime)
        rnn2 = KK.layers.GRU( hidden_size, return_sequences=True)(rnn1)
        rnn3 = KK.layers.GRU( hidden_size, return_sequences=True)(rnn2)

        predictionsHPLEN = KK.layers.Dense(33, activation='softmax')(rnn3)
        predictionsHPID = KK.layers.Dense(4, activation='softmax')(rnn3)

        ################################
        self.model = KK.models.Model(inputs=inputs, outputs=[predictionsHPID,predictionsHPLEN])

        #self.model.summary()
        print("================================")
        for layer in self.model.layers:
            print("layer: name",layer.name, "outputshape", layer.get_output_at(0).get_shape().as_list())
        print("================================")

        myopt = KK.optimizers.SGD()
        #myopt = KK.optimizers.Adam()
        self.model.compile(optimizer=myopt, loss="kullback_leibler_divergence") #, metrics=["categorical_accuracy",])"categorical_crossentropy"

        # # instrument tensorboard
        # tf.summary.histogram('output', self.output)
        # tf.summary.histogram('softmax_w', softmax_w)
        # tf.summary.histogram('logits', self.logits)
        # tf.summary.histogram('probs', self.probs)
        # tf.summary.scalar('train_loss', self.cost)
