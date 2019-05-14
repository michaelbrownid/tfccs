#!/usr/bin/env python

from __future__ import print_function
import argparse
import time
import datetime
import os
import sys
import tensorflow as tf
import tensorflow.keras
import numpy as np
from .model import Model
from . import data

################################
def train(args):

    data_loader = data.data( args.batch_size, sys.argv[2])

    # get test if there
    data_loader_test = None
    if len(sys.argv)>3:
        data_loader_test = data.data( args.batch_size, sys.argv[3])

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        ckpt = tf.train.latest_checkpoint(args.init_from)
        assert ckpt, "No checkpoint found"

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    #with tf.device("/cpu:0"):
    #with tf.device("/gpu:3"):
    if True:
        model = Model(args)

        with tf.Session() as sess:
            # instrument for tensorboard
            summaries = tf.summary.merge_all()
            writer = tf.summary.FileWriter(os.path.join(args.log_dir, time.strftime("%Y-%m-%d-%H-%M-%S")))
            writer.add_graph(sess.graph)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            # restore model
            if args.init_from is not None:
                saver.restore(sess, ckpt)

            print("# args.num_epochs", args.num_epochs, "args.batch_size", args.batch_size, "num_batches", data_loader.num_batches)

            for e in range(args.num_epochs):
                storeloss = []
                data_loader.reset_batch_pointer()
                for b in range(data_loader.num_batches):
                    start = time.time()
                    x, y = data_loader.next_batch()
                    yid = y[:,:,0:4]
                    ylen = y[:,:,4:]

                    #myfit=model.model.fit( x, [yid,ylen], epochs=1, batch_size=1,verbose=2)
                    myfit = model.model.train_on_batch( x, [yid,ylen])
                    end = time.time()
                    print("epoch %d batch %d time %f" % (e, b, end-start))
                    for (kk,vv) in zip(model.model.metrics_names,myfit):
                          print("epoch %d batch %d trainMetric %s %f batchsize %d" % (e ,b ,kk,vv, x.shape[0]))
                          if kk=="loss":
                              storeloss.append( (vv,x.shape[0]) )

                #writer.add_summary(summ, e * data_loader.num_batches + b)
                # compute average loss across all batches
                trainnum = 0
                trainsum = 0.0
                for xx in storeloss:
                    trainsum += xx[0]*xx[1]
                    trainnum += xx[1]
                train_loss = trainsum/float(trainnum)
                print("epoch %d trainLossAvg %f" % (e , train_loss))

                #### Training ran through all the batches

                # if save_every or at tend then save and run validation test
                if (e % args.save_every == 0) or (e == args.num_epochs-1):
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
                    # TODO: tried keras model saving to json and hd5
                    # but got not serialable error.
                    # https://github.com/tensorflow/tensorflow/issues/27112

                    # run the test set if there
                    if data_loader_test is not None:
                        storeloss = []
                        data_loader_test.reset_batch_pointer()
                        for b in range(data_loader_test.num_batches):
                            x, y = data_loader_test.next_batch()
                            yid = y[:,:,0:4]
                            ylen = y[:,:,4:]
                            #mytest=model.model.evaluate( x, [yid,ylen],verbose=0)
                            mytest=model.model.test_on_batch( x, [yid,ylen])
                            for (kk,vv) in zip(model.model.metrics_names,mytest):
                                print("epoch %d batch %d testMetric %s %f batchsize %d" % (e ,b ,kk,vv, x.shape[0]))
                                if kk=="loss":
                                    storeloss.append( (vv,x.shape[0]) )

                        # compute average loss across all batches
                        testnum = 0
                        testsum = 0.0
                        for xx in storeloss:
                            testsum += xx[0]*xx[1]
                            testnum += xx[1]
                        test_loss = testsum/float(testnum)
                        print("epoch %d testLossAvg %f" % (e , test_loss))

if __name__ == '__main__':

    print("time begin",str(datetime.datetime.now()))

    exec(open(sys.argv[1]).read())
    args.init_from = None

    print("-------")
    train(args)

    print("time end",str(datetime.datetime.now()))
