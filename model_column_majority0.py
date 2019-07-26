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

        # call each position by base^Present, X^Missing. Could be 5 but using 16. Kernel=10-vector
        baseadj = KK.layers.Conv2D(16, kernel_size= (1, 1), activation='relu')(inputs)

        majority = KK.backend.mean(baseadj, axis=[1])

        predBase = KK.layers.TimeDistributed( KK.layers.Dense(5, activation='softmax'))(majority)

        # this should be (+ (* 10 16) (* 16 5))=240 parameters

        ################################
        self.model = KK.models.Model(inputs=inputs, outputs=[predBase])

        #self.model.summary()
        print("================================")
        for layer in self.model.layers:
            print("layer: name",layer.name, "outputshape", layer.get_output_at(0).get_shape().as_list())
        print("================================")

        #myopt = KK.optimizers.SGD()
        myopt = KK.optimizers.Adam()
        self.model.compile(optimizer=myopt, loss="kullback_leibler_divergence") #, metrics=["categorical_accuracy",])"categorical_crossentropy"

        # # instrument tensorboard
        # tf.summary.histogram('output', self.output)
        # tf.summary.histogram('softmax_w', softmax_w)
        # tf.summary.histogram('logits', self.logits)
        # tf.summary.histogram('probs', self.probs)
        # tf.summary.scalar('train_loss', self.cost)
