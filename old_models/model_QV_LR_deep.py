import tensorflow as tf
import tensorflow.keras as KK
import sys
import numpy as np
from . import struct

class Model():
    def __init__(self, args):

        self.args = args

        inputs = KK.layers.Input(shape=(120,))

        hidden1= KK.layers.Dense(120, activation='relu')(inputs)
        hidden2= KK.layers.Dense(60, activation='relu')(hidden1)
        hidden3= KK.layers.Dense(30, activation='relu')(hidden2)
        out = KK.layers.Dense(1, activation='linear')(hidden3)

        self.model = KK.models.Model(inputs=inputs, outputs=[out])

        self.model.summary()
        print("================================")
        for layer in self.model.layers:
            print("layer: name",layer.name, "outputshape", layer.get_output_at(0).get_shape().as_list())
        print("================================")

        #myopt = KK.optimizers.SGD()
        myopt = KK.optimizers.Adam()
        self.model.compile(optimizer=myopt, loss="mean_squared_error")
