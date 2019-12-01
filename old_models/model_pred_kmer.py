import tensorflow as tf
import tensorflow.keras as KK
import sys
import numpy as np
from . import struct

# return function with from_logits=True
def crossEntropySparseLoss(Truth, Pred):
    print("KK.backend.int_shape(Truth)",KK.backend.int_shape(Truth))
    print("KK.backend.int_shape(Pred)",KK.backend.int_shape(Pred))
    return(tf.keras.losses.sparse_categorical_crossentropy(Truth, Pred, from_logits=True))
    #return(tf.nn.sparse_softmax_cross_entropy_with_logits( labels=Truth, logits=Pred ))

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

        convNum = 256
        baseadj = KK.layers.Conv2D(convNum, kernel_size= (16, 61), strides=(16,1),activation='relu', padding="same", name="baseadj")(inputs) 
        baseadj2 = KK.layers.Lambda( lambda xx: tf.squeeze( xx, 1), name="baseadj2")(baseadj)

        #baseadjperm = KK.layers.Lambda( lambda xx: KK.backend.permute_dimensions(xx,(0,2,1,3)), name="baseadjperm")(baseadj)
        #baseadjpermreshape = KK.layers.Reshape((args.cols, args.rows*convNum),name="baseadjpermreshape")(baseadjperm)

        predBase0 = KK.layers.TimeDistributed( KK.layers.Dense(1024, activation='softmax'), name="out1024")(baseadj2)

        ################################
        self.model = KK.models.Model(inputs=inputs, outputs=predBase0)

        #self.model.summary()
        print("================================")
        for layer in self.model.layers:
            print("layer: name",layer.name, "outputshape", layer.get_output_at(0).get_shape().as_list())
        print("================================")

        #myopt = KK.optimizers.SGD()
        myopt = KK.optimizers.Adam()
        self.model.compile(optimizer=myopt, loss="sparse_categorical_crossentropy") # loss="sparse_categorical_crossentropy") #, metrics=["categorical_accuracy",])"" "kullback_leibler_divergence" crossEntropySparseLoss

        # # instrument tensorboard
        # tf.summary.histogram('output', self.output)
        # tf.summary.histogram('softmax_w', softmax_w)
        # tf.summary.histogram('logits', self.logits)
        # tf.summary.histogram('probs', self.probs)
        # tf.summary.scalar('train_loss', self.cost)
