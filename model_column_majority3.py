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

        baseadj = KK.layers.Conv2D(args.modelNumConv2d, kernel_size= (args.modelKernelRows,args.modelKernelCols), activation='relu', padding="same")(inputs)
        # layer: name conv2d outputshape [None, 16, 640, args.modelNumConv2d]

        #baseadjperm = KK.backend.permute_dimensions(baseadj,(0,2,1,3))
        baseadjperm = KK.layers.Lambda( lambda xx: KK.backend.permute_dimensions( xx, (0,2,1,3)), name="baseadjperm")(baseadj)

        #baseadjpermreshape = KK.layers.Reshape((args.cols, args.rows*args.modelNumConv2d))(baseadjperm)
        baseadjpermreshape = KK.layers.Lambda( lambda xx: KK.layers.Reshape((args.cols, args.rows*args.modelNumConv2d))(xx), name="baseadjpermrehape")(baseadjperm)

        predBase0 = KK.layers.TimeDistributed( KK.layers.Dense(args.modelExtraLayerN, activation='relu'))(baseadjpermreshape)
        predBase = KK.layers.TimeDistributed( KK.layers.Dense(5, activation='softmax'))(predBase0)

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
