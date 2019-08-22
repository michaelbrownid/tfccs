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

        inputmsa = KK.layers.Input(shape=(args.rows,args.cols,args.baseinfo), name="inputmsa")
        inputctx = KK.layers.Input(shape=(args.cols,), name="inputctx")

        baseadj = KK.layers.Conv2D(256, kernel_size= (21, 16), strides=(16,1),activation='relu', padding="same", name="baseadj")(inputmsa)

        #### you can't use KK.backend.squeeze if you want to use save/load!
        # baseadj2 = KK.backend.squeeze(baseadj,1) # [None, 1, 640, 256] -> [None, 640, 256]
        baseadj2 = KK.layers.Lambda( lambda xx: tf.squeeze( xx, 1), name="baseadj2")(baseadj)

        print("baseadj2.shape",baseadj2.shape)

        # merge the msa features with the read ctx embedding (int(0:1024)=10-vect
        embed = KK.layers.Embedding(input_dim=1024, output_dim=10,name="embed")(inputctx)
        print("embed.shape",embed.shape)

        merged = KK.layers.Concatenate(axis=2,name="merged")([baseadj2,embed])

        predBase = KK.layers.TimeDistributed( KK.layers.Dense(5, activation='softmax',name="predBase"))(merged)

        ################################
        self.model = KK.models.Model(inputs=[inputmsa,inputctx], outputs=[predBase])

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
