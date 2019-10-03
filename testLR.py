#!/usr/bin/env python

from __future__ import print_function
import argparse
import time
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as KK
#from .model import Model
from . import data
import math

################################
def test(args):

    data_loader = data.data( args.batch_size, sys.argv[2],
                             inputdatName=args.inputdatName,
                             outputdatName=args.outputdatName)

    if True:
        # load the model based on name and access as Model: from .model import args.model
        myimport = importlib.import_module("tfccs.%s" % args.model)
        Model = myimport.Model

        model = Model(args)

        # make it appear as though there is only one gpu and use it
        os.environ["CUDA_VISIBLE_DEVICES"]=args.CUDA_VISIBLE_DEVICES

        num=0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # restore model
            model.model = KK.models.load_model(args.modelsave)

            ################################
            sse = 0.0
            total = 0
            count = 0
            for b in range(data_loader.num_batches):

                start = time.time()
                x, y = data_loader.next_batch()
                
                predictions = model.model.predict(x)

                print("predictions.shape",predictions.shape)
                print("y.shape",y.shape)
                print("x.shape",x.shape)


                #### print out all predictions
                if True:
                    #fp = open("test.LR.preds.txt","w")
                    for ii in range(predictions.shape[0]):
                        mytruth = y[ii]
                        mypred = predictions[ii,0]
                        myx = x[ii]
                        se = (mytruth-mypred)*(mytruth-mypred)
                        print(ii,mytruth,mypred,se)
                        sse += se
                        total +=1
                end = time.time()



                #break # only look at 0th batch for time
            print("rms %f sse %f total %d" % (math.sqrt(float(sse)/total),sse,total))

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
