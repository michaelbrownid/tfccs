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

        # make it appear as though there is only one gpu and use it
        os.environ["CUDA_VISIBLE_DEVICES"]="2"

        tfconfig=tf.ConfigProto()
        # tfconfig.allow_soft_placement=True
        # tfconfig.log_device_placement=True
        # tfconfig.gpu_options.allow_growth=True

        with tf.Session( config=tfconfig ) as sess:

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

                start = time.time()
                predictions = model.model.predict(x)
                end = time.time()
                print("time batch",b,"x.shape[0]",x.shape[0],"duration",end-start)

                print("predictions[0].shape",predictions[0].shape)
                print("predictions[1].shape",predictions[1].shape)
                print("yid.shape",yid.shape)
                print("ylen.shape",ylen.shape)

                # predictions[0].shape (256, 640, 4)
                # predictions[1].shape (256, 640, 33)
                # yid.shape (256, 640, 4)
                # ylen.shape (256, 640, 29)

                np.save("test.0.estimate",predictions[0])
                np.save("test.1.estimate",predictions[1])

                np.save("test.0.truth",yid)
                np.save("test.1.truth",ylen)

                #### dump the predictions with true position index
                for myclass in [1,0]:
                    print("------",myclass)

                    numerr = 0
                    total = 0
                    fp = open("test.%d.preds.txt" % myclass,"w")
                    print("#myclass\tobject\tposition\ttruecallIdx\tdata",file=fp)
                    for ii in range(predictions[myclass].shape[0]):
                        for jj in range(predictions[myclass].shape[1]):

                            if myclass == 1:
                                truth = ylen[ii,jj]
                            else:
                                truth = yid[ii,jj]
                            estimate = predictions[myclass][ii,jj]

                            # skip null
                            # if truth[0]==0.25 and truth[1]==0.25 and truth[2]==0.25 and truth[3]==0.25:
                            #     continue

                            truemax = np.argmax(truth)
                            estmax = np.argmax(estimate)

                            if truemax!=estmax:
                                numerr+=1
                                print("err %d %d %d true est prob" % (myclass, ii, jj), truemax,estmax,estimate[estmax])
                            total+=1

                            print("%d\t%d\t%d\t%d\t%d\t%s" % (myclass,
                                                              ii,
                                                              jj,
                                                              truemax,
                                                              estmax,
                                                              "\t".join([str(xx) for xx in estimate])
                                                          ), file=fp)
                    fp.close()
                    print("error rate 1 %f = %d / %d" % (float(numerr)/total,numerr,total))

                break # only look at 0th batch for time

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
