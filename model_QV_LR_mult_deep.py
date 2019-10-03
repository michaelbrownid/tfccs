import tensorflow as tf
import tensorflow.keras as KK
import sys
import numpy as np
from . import struct

class Model():
    def __init__(self, args):

        self.args = args

        inputs = KK.layers.Input(shape=(118,))

        hidden1= KK.layers.Dense(120, activation='relu')(inputs)
        hidden2= KK.layers.Dense(100, activation='relu')(hidden1)
        hidden3= KK.layers.Dense(80, activation='relu')(hidden2)
        hidden4= KK.layers.Dense(60, activation='relu')(hidden3)
        hidden5= KK.layers.Dense(40, activation='relu')(hidden4)
        hidden6= KK.layers.Dense(20, activation='relu')(hidden5)
        out = KK.layers.Dense(8, activation='softmax')(hidden6)

        self.model = KK.models.Model(inputs=inputs, outputs=[out])

        self.model.summary()
        print("================================")
        for layer in self.model.layers:
            print("layer: name",layer.name, "outputshape", layer.get_output_at(0).get_shape().as_list())
        print("================================")

        #myopt = KK.optimizers.SGD()
        myopt = KK.optimizers.Adam()
        self.model.compile(optimizer=myopt, loss="categorical_crossentropy") # loss="kullback_leibler_divergence")
