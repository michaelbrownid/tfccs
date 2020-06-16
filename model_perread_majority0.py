import tensorflow as tf
import tensorflow.keras as KK
import sys
import numpy as np
from . import struct

class Model():
    def __init__(self, args):

        self.args = args

        inputs = KK.layers.Input(shape=(args.rows,args.cols,args.baseinfo))

        #### take the baseint and embed
        # non-serializable: baseint = inputs[:,:,:,0] # the 0th out of 4 of base info
        baseint = KK.layers.Lambda( lambda xx: xx[:,:,:,0] )(inputs)
        embed = KK.layers.Embedding(input_dim=26+1, output_dim=4, name="embed")(baseint)
        print("embed.shape",embed.shape)

        #### merged is input plus the 4 embed for a total of 4+4 =8
        merged = KK.layers.Concatenate(axis=3,name="merged")([inputs,embed])

        # call each position by base^Present, X^Missing. Could be 5 but using 16. Kernel=10-vector
        baseadj = KK.layers.Conv2D(16, kernel_size= (1, 1), activation='relu')(merged)

        #majority = KK.backend.mean(baseadj, axis=[1])
        majority = KK.layers.Lambda( lambda xx: KK.backend.mean( xx, axis=[1]), name="majority")(baseadj)

        predBase = KK.layers.Dense(5, activation='softmax')(majority)

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
