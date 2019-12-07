#!/usr/bin/env python

from __future__ import print_function
import argparse
import time
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as KK
from . import data
import importlib

################################
def test(args):

    data_loader = data.data( args.batch_size, sys.argv[2],
                             inputdatName=args.inputdatName,
                             outputdatName=args.outputdatName)

    if True:
        # make it appear as though there is only one gpu and use it
        os.environ["CUDA_VISIBLE_DEVICES"]=args.CUDA_VISIBLE_DEVICES

        # load the model based on name and access as Model: from .model import args.model
        myimport = importlib.import_module("tfccs.%s" % args.model)
        Model = myimport.Model
        model = Model(args)

        objnum=0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # restore model. Defined losses aren't used in testing
            def nullloss(xx,yy): return(KK.backend.mean(yy,axis= -1))

            model.model = KK.models.load_model(args.modelsave, custom_objects={"KK":KK, 
                                                                               "zero_loss": nullloss,
                                                                               "myloss": nullloss,
                                                                               "sparse_kl": nullloss})
            ################################
            numerr = 0
            total = 0
            first=True
            for b in range(data_loader.num_batches):

                start = time.time()
                x, y = data_loader.next_batch()

                # self.model = KK.models.Model(inputs=[inputs,inputsCallProp,inputsCallTrue],
                #                              outputs=[rnnconcat,predHPBase,predHPLen,predHPCall,inputsCallTrue])
                # remove rnnconcat from first output # ii 0 predictions[ii].shape (1000, 640, 128)
                # ii 1 predictions[ii].shape (1000, 640, 4)
                # ii 2 predictions[ii].shape (1000, 640, 33)
                # ii 3 predictions[ii].shape (1000, 640, 2)
                # ii 4 predictions[ii].shape (1000, 640, 1)

                predictions = model.model.predict(x)

                ######## Dump the predictions and truth
                """ output - predHPBase, predHPlen, predHPCall in test.dump.predictions.<model>.dat
                           - truth sets                        in test.dump.truth.<model>.dat
                """
                headerDone = False
                headerCols = ["hpbase","hplen","hpcall","inputscalltrue"]
                fppred = open("test.dump.predictions.%s.dat" % args.modelsave,"w")
                fptruth = open("test.dump.truth.%s.dat" % args.modelsave,"w")
                for obj in range(predictions[0].shape[0]): # [0,256,512,768]:
                    for column in range(predictions[0].shape[1]):
                        datapred = []
                        datatruth = []
                        header = ["obj","column"]
                        for predii in range(0,3):
                            for datii in range(predictions[predii].shape[2]):
                                datapred.append(predictions[predii][obj][column][datii])
                                datatruth.append(y[predii][obj][column][datii])
                                if not headerDone:
                                    header.append("%s-%d" % (headerCols[predii],datii))

                        if not headerDone:
                            fppred.write("%s\n" % "\t".join(header))
                            fptruth.write("%s\n" % "\t".join(header))
                            headerDone=True

                        fppred.write("%d\t%d\t%s" % (obj,column,"\t".join([str(xx) for xx in datapred]))+"\n")
                        fptruth.write("%d\t%d\t%s" % (obj,column,"\t".join([str(xx) for xx in datatruth]))+"\n")

                fppred.close()
                fptruth.close()

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
