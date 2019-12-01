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

        baseadj = KK.layers.Conv2D(128, kernel_size= (16, 1), strides=(16,1),activation='relu', padding="same")(inputs)
        baseadj2 = KK.layers.Lambda( lambda xx: tf.squeeze( xx, 1), name="baseadj2")(baseadj) # [None, 1, 640, 128] -> [None, 640, 128]

        predBase = KK.layers.Dense(64, activation='softmax')(baseadj2)

        #### now take predBase and run rnn to predict correct base that
        #### is labeled at every column of the input MSA
        
        rnn_hidden_size= 64
        bidi = KK.layers.Bidirectional( KK.layers.LSTM( rnn_hidden_size, return_sequences=True))(predBase)
        #rnn1 = KK.layers.LSTM( rnn_hidden_size, return_sequences=True)(bidi)

        #predictionsHPLEN = KK.layers.Dense(33, activation='softmax')(rnn1)
        #predictionsHPID = KK.layers.Dense(4, activation='softmax')(rnn1)
        predictBase = KK.layers.Dense(5, activation='softmax')(bidi)

        ################################
        self.model = KK.models.Model(inputs=inputs, outputs=[predictBase])

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
