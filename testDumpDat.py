#!/usr/bin/env python

from __future__ import print_function
import argparse
import time
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as KK
from .model import Model
from . import data

################################
def test(args):


    #with tf.device("/gpu:2"):
    if True:
        # check compatibility if training is continued from previously saved model
        if args.init_from is not None:
            # check if all necessary files exist
            assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
            ckpt = tf.train.latest_checkpoint(args.init_from)
            assert ckpt, "No checkpoint found"
            print("ckpt", ckpt, file=sys.stderr)

        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)

        model = Model(args)

        # make it appear as though there is only one gpu and use it
        os.environ["CUDA_VISIBLE_DEVICES"]="2"

        tfconfig=tf.ConfigProto()
        # tfconfig.allow_soft_placement=True
        # tfconfig.log_device_placement=True
        # tfconfig.gpu_options.allow_growth=True

        with tf.Session( config=tfconfig ) as sess:

            #sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            # restore model
            # if args.init_from is not None:
            #     saver.restore(sess, ckpt)

            model.model = KK.models.load_model("my_model_FULL.h5")
            #model.model.load_weights("my_model.h5")

            ################################
            for layer in model.model.layers:
                print("layer: name",layer.name, "outputshape", layer.get_output_at(0).get_shape().as_list())
                myweights = layer.get_weights()
                for xx in myweights:
                    print("xx.shape",xx.shape)
                # dump 
                if "embed" in layer.name:
                    np.savetxt("embedding.numpy.txt",myweights[0],delimiter='\t')


            print("================================")

if __name__ == '__main__':
    exec(open(sys.argv[1]).read())
    for aa in sys.argv:
        if "EXEC:" in aa:
            toexec = aa.replace("EXEC:","")
            print("toexec",toexec)
            exec(toexec)
            
    print("-------")
    print(help(args))
    print("-------")

    test(args)
