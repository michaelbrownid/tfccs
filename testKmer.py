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

from .model import crossEntropySparseLoss # for model restore

################################
def test(args):


    data_loader = data.data( args.batch_size, sys.argv[2],
                             inputdatName=args.inputdatName,
                             outputdatName=args.outputdatName)

    #with tf.device("/gpu:2"):
    if True:

        model = Model(args)

        # make it appear as though there is only one gpu and use it
        os.environ["CUDA_VISIBLE_DEVICES"]="2"

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # restore model
            model.model = KK.models.load_model("my_model_FULL.h5", custom_objects={"crossEntropySparseLoss": crossEntropySparseLoss}) # loss is custom: otherwise ValueError: Unknown loss function:crossEntropySparseLoss

            ################################
            for b in range(data_loader.num_batches):

                start = time.time()
                x, y = data_loader.next_batch()
                
                predictions = model.model.predict(x)

                print("predictions.shape",predictions.shape)
                print("y.shape",y.shape)

                np.save("test.kmer.predictions",predictions)
                np.save("test.kmer.truth",y)

                if True:
                    numerr = 0
                    total = 0
                    for obj in range(y.shape[0]):
                        for objelement in range(y.shape[1]):
                            truemax = y[obj,objelement]
                            estimate = predictions[obj,objelement]
                            estmax = np.argmax(estimate)
                            if truemax!=estmax:
                                numerr+=1
                                print("err1 %d %d true est prob" % (obj, objelement), truemax,estmax,estimate[estmax])
                            total+=1
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
