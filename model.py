import tensorflow as tf
import tensorflow.keras as KK
import sys
import numpy as np

class Model():
    def __init__(self, args):

        """in -> convolutions - > reduced -> out
        """

        self.args = args

        """class args:
            rows=16
            cols=640
            baseinfo=10
            hps = 128
            hpdist = 33
        """

        inputs = KK.layers.Input(shape=(args.rows,args.cols,args.baseinfo))
        x = KK.layers.Conv2D(8, kernel_size= (1, 6), strides=(1,5), activation='relu')(inputs)
        x = KK.layers.Conv2D(8, (16, 1), activation='relu')(x)
        bottle = KK.layers.Flatten()(x) # This is now the bottleneck

        self.model = KK.models.Model(inputs=inputs, outputs=[bottle])
        self.model.summary()

        ################################
        # predict hp length
        predictionsHPLEN = KK.layers.Dense(args.hpdist, activation='softmax')(bottle)

        ################################
        # predict hp base IDentity
        predictionsHPID = KK.layers.Dense(4, activation='softmax')(bottle)

        ################################
        self.model = KK.models.Model(inputs=inputs, outputs=[predictionsHPID,predictionsHPLEN])

        self.model.summary()

        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy","kullback_leibler_divergence"])

        # # instrument tensorboard
        # tf.summary.histogram('output', self.output)
        # tf.summary.histogram('softmax_w', softmax_w)
        # tf.summary.histogram('logits', self.logits)
        # tf.summary.histogram('probs', self.probs)
        # tf.summary.scalar('train_loss', self.cost)
