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

        baseadj = KK.layers.Conv2D(256, kernel_size= (21, 16), strides=(16,1),activation='relu', padding="same")(inputs)
        baseadj2 = KK.backend.squeeze(baseadj,1) 

        pred0 = KK.layers.TimeDistributed( KK.layers.Dense(128, activation='softmax'))(baseadj2)
        pred1 = KK.layers.TimeDistributed( KK.layers.Dense(64, activation='softmax'))(pred0)

        predictionsHPLEN = KK.layers.Dense(33, activation='softmax')(pred1)
        predictionsHPID = KK.layers.Dense(4, activation='softmax')(pred1)

        ################################
        self.model = KK.models.Model(inputs=inputs, outputs=[predictionsHPID,predictionsHPLEN])

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
