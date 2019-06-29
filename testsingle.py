#!/usr/bin/env python

from __future__ import print_function
import argparse
import time
import os
import sys
import numpy as np
import tensorflow as tf
from .model import Model
from . import data

################################
def test(args):

    data_loader = data.data( args.batch_size, sys.argv[2],shortcut=True)

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

        num=0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            # restore model
            if args.init_from is not None:
                saver.restore(sess, ckpt)

            ################################
            countErr=0
            countTotal=0

            for b in range(data_loader.num_batches):

                start = time.time()
                x, y = data_loader.next_batch()
                #### Try to predict only the first full HP at [0]
                #yid = y[:,0,0:4]
                ylen = y[:,0,4:]

                predictions = model.model.predict(x)

                # print("len(predictions)",len(predictions))
                # print("predictions[0].shape",predictions[0].shape)
                # print("predictions[1].shape",predictions[1].shape)
                # print("len(ylen)",len(ylen))
                # print("ylen[0].shape",ylen[0].shape)

                #np.save("test.0.predictions",predictions[0])
                #np.save("test.1.predictions",predictions[1])


                ################################
                # take max for error rate

                ####
                if True:
                    for ii in range(len(predictions)):
                            truth = ylen[ii,:]
                            countTotal+=1
                            estimate = predictions[ii]
                            truemax = np.argmax(truth)
                            estmax = np.argmax(estimate)
                            # # estsort = np.sort(-estimate,1)
                            if truemax!=estmax:
                                print("err1 %d %d true est prob" % (ii, 0), truemax,estmax,estimate[estmax])
                                countErr+=1
                            else:
                                print("correct1 %d %d true est prob" % (ii, 0), truemax,estmax,estimate[estmax])
            print("#error rate %f = %d / %d" % (float(countErr)/countTotal,countErr,countTotal))


if __name__ == '__main__':
    exec(open(sys.argv[1]).read())

    test(args)
