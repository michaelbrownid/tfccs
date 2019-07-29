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

    data_loader = data.data( args.batch_size, sys.argv[2],
                                 inputdatName=args.inputdatName,
                                 outputdatName=args.outputdatName)

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
            for b in range(data_loader.num_batches):

                start = time.time()
                x, y = data_loader.next_batch()
                
                yid = y[:,:,0:4]
                ylen = y[:,:,4:]

                predictions = model.model.predict(x)

                print("predictions[0].shape",predictions[0].shape)
                print("predictions[1].shape",predictions[1].shape)

                #np.save("test.0.predictions",predictions[0])
                #np.save("test.1.predictions",predictions[1])

                #### SINGLE HP
                if False:
                    fp = open("test.0.preds.txt","w")
                    for ii in range(predictions[0].shape[0]):
                        print("0\t%d\t%d\t%s\t-1\t%s" % (ii,
                                                         0,
                                                         "\t".join([str(xx) for xx in predictions[0][ii,:]]),
                                                         "\t".join([str(xx) for xx in yid[ii,:]])
                                                     ), file=fp)
                    fp.close()

                    fp = open("test.1.preds.txt","w")
                    for ii in range(predictions[1].shape[0]):
                        print("0\t%d\t%d\t%s\t-1\t%s" % (ii,
                                                         0,
                                                         "\t".join([str(xx) for xx in predictions[1][ii,:]]),
                                                         "\t".join([str(xx) for xx in ylen[ii,:]])
                                                     ), file=fp)
                    fp.close()

                #### All 128 HPs
                if False:
                    fp = open("test.0.preds.txt","w")
                    for ii in range(predictions[0].shape[0]):
                        for jj in range(predictions[0].shape[1]):
                            print("0\t%d\t%d\t%s\t-1\t%s" % (ii,
                                                            jj,
                                                            "\t".join([str(xx) for xx in predictions[0][ii,jj]]),
                                                            "\t".join([str(xx) for xx in yid[ii,jj]])
                                                        ), file=fp)
                    fp.close()

                    fp = open("test.1.preds.txt","w")
                    for ii in range(predictions[1].shape[0]):
                        for jj in range(predictions[1].shape[1]):
                            print("0\t%d\t%d\t%s\t-1\t%s" % (ii,
                                                            jj,
                                                            "\t".join([str(xx) for xx in predictions[1][ii,jj]]),
                                                            "\t".join([str(xx) for xx in ylen[ii,jj]])
                                                        ), file=fp)
                    fp.close()

                end = time.time()

                ################################
                # take max for error rate

                #### SINGLE HP
                if False:
                    numerr = 0
                    total = 0
                    for ii in range(predictions[1].shape[0]):
                            truth = ylen[ii,:]
                            estimate = predictions[1][ii,:]
                            truemax = np.argmax(truth)
                            estmax = np.argmax(estimate)
                            # # estsort = np.sort(-estimate,1)
                            if truemax>3:
                                print("long1 %d %d true est prob" % (ii, 0), truemax,estmax,estimate[estmax])
                            if truemax!=estmax:
                                numerr+=1
                                print("err1 %d %d true est prob" % (ii, 0), truemax,estmax,estimate[estmax])
                            total+=1
                    print("error rate 1 %f = %d / %d" % (float(numerr)/total,numerr,total))

                    numerr = 0
                    total = 0
                    for ii in range(predictions[0].shape[0]):
                            truth = yid[ii,:]
                            # skip null
                            # if truth[0]==0.25 and truth[1]==0.25 and truth[2]==0.25 and truth[3]==0.25: continue
                            estimate = predictions[0][ii,:]
                            truemax = np.argmax(truth)
                            estmax = np.argmax(estimate)
                            # # estsort = np.sort(-estimate,1)
                            if truemax!=estmax:
                                numerr+=1
                                print("err0 %d %d true est prob" % (ii, 0), truemax,estmax,estimate[estmax])
                            total+=1
                    print("error rate 0 %f = %d / %d" % (float(numerr)/total,numerr,total))

                #### All 128 HPs
                if True:
                    numerr = 0
                    total = 0
                    for ii in range(predictions[1].shape[0]):
                        for jj in range(predictions[1].shape[1]):
                            truth = ylen[ii,jj]
                            estimate = predictions[1][ii,jj]
                            truemax = np.argmax(truth)
                            estmax = np.argmax(estimate)
                            # # estsort = np.sort(-estimate,1)
                            if truemax>3:
                                print("long1 %d %d true est prob" % (ii, jj), truemax,estmax,estimate[estmax])
                            if truemax!=estmax:
                                numerr+=1
                                print("err1 %d %d true est prob" % (ii, jj), truemax,estmax,estimate[estmax])
                            total+=1
                    print("error rate 1 %f = %d / %d" % (float(numerr)/total,numerr,total))

                    numerr = 0
                    total = 0
                    for ii in range(predictions[0].shape[0]):
                        for jj in range(predictions[0].shape[1]):
                            truth = yid[ii,jj]
                            # skip null
                            if truth[0]==0.25 and truth[1]==0.25 and truth[2]==0.25 and truth[3]==0.25:
                                continue
                            estimate = predictions[0][ii,jj]
                            truemax = np.argmax(truth)
                            estmax = np.argmax(estimate)
                            # # estsort = np.sort(-estimate,1)
                            if truemax!=estmax:
                                numerr+=1
                                print("err0 %d %d true est prob" % (ii, jj), truemax,estmax,estimate[estmax])
                            total+=1
                    print("error rate 0 %f = %d / %d" % (float(numerr)/total,numerr,total))

                break # only look at 0th batch for time

if __name__ == '__main__':
    exec(open(sys.argv[1]).read())
    for aa in sys.argv:
        if "EXEC:" in aa:
            toexec = aa.replace("EXEC:","")
            print("toexec",toexec)
            exec(toexec)
            
    print("-------")
    print(args)
    print("-------")

    test(args)
