import tensorflow as tf
import tensorflow.keras as KK
import sys
import numpy as np
from . import struct

class Model():
    def __init__(self, args):

        self.args = args

        inputs = KK.layers.Input(shape=(118,))
        lr = KK.layers.Dense(8, activation='softmax')(inputs)

        self.model = KK.models.Model(inputs=inputs, outputs=[lr])

        self.model.summary()
        print("================================")
        for layer in self.model.layers:
            print("layer: name",layer.name, "outputshape", layer.get_output_at(0).get_shape().as_list())
        print("================================")

        #myopt = KK.optimizers.SGD()
        myopt = KK.optimizers.Adam()
        self.model.compile(optimizer=myopt, loss="categorical_crossentropy") # loss="kullback_leibler_divergence")
