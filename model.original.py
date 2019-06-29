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
        x = KK.layers.Conv2D(64, kernel_size= (1, 6), strides=(1,5), activation='relu')(inputs)
        x = KK.layers.Conv2D(64, (16, 1), activation='relu')(x)
        bottle = KK.layers.Flatten()(x) # This is now the bottleneck

        self.model = KK.models.Model(inputs=inputs, outputs=[bottle])
        self.model.summary()

        ################################
        # predict hp lengths
        predsHPLEN = []
        for ii in range( args.hps ):
            tmp = KK.layers.Dense(args.hpdist, activation='softmax')(bottle)
            # print("tmp",tmp.shape) # tmp (N, 33)
            predsHPLEN.append(tmp)

        # # stack into one array yielding (128,N,33)
        predsHPLEN2 = KK.backend.stack(predsHPLEN)
        print("predsHPLEN2.shape", predsHPLEN2.shape)

        # # permute to yield (N,128,33) to compare against truth
        predictionsHPLEN = KK.backend.permute_dimensions(predsHPLEN2,(1,0,2))
        print("predictionsHPLEN.shape",predictionsHPLEN.shape) # predictions.shape (?, 128, 33)

        ################################
        # predict hp base IDentity
        predsHPID = []
        for ii in range( args.hps ):
            tmp = KK.layers.Dense(4, activation='softmax')(bottle)
            # print("tmp",tmp.shape) # tmp (N, 4)
            predsHPID.append(tmp)

        # # stack into one array yielding (128,N,4)
        predsHPID2 = KK.backend.stack(predsHPID)
        print("predsHPID2.shape", predsHPID2.shape)

        # # permute to yield (N,128,4) to compare against truth
        predictionsHPID = KK.backend.permute_dimensions(predsHPID2,(1,0,2))
        print("predictionsHPID.shape",predictionsHPID.shape) # predictions.shape (?, 128, 4)

        ################################
        self.model = KK.models.Model(inputs=inputs, outputs=[predictionsHPID,predictionsHPLEN])

        #self.model.summary()
        # # ValueError: You tried to call `count_params` on stack, but the layer isn't built. You can build it manually via: `stack.build(batch_input_shape)`.

        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy","kullback_leibler_divergence"])

        # # instrument tensorboard
        # tf.summary.histogram('output', self.output)
        # tf.summary.histogram('softmax_w', softmax_w)
        # tf.summary.histogram('logits', self.logits)
        # tf.summary.histogram('probs', self.probs)
        # tf.summary.scalar('train_loss', self.cost)
