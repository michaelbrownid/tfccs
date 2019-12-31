#!/usr/bin/env python

"""
export PYTHONPATH=/home/UNIXHOME/mbrown/mbrown/workspace2019Q2/ccs-ml-model/; \
python3 -m tfccs.test_combine_subreads perread_birnn /home/UNIXHOME/mbrown/mbrown/workspace2019Q2/ccs-ml-model/zip1Embed.class \
<(echo 'zip1.VAL.data.poa.perread.npz') \
EXEC:args.inputdatName=\'windowinputPlusCallProp\' \
EXEC:args.outputdatName=\'windowoutputDirectHPPlusCallTrue\' \
EXEC:args.model=\'model_perread_birnn\' \
EXEC:args.modelsave=\'model.perread.birnn.readnumber.1.h5.best\' \
EXEC:args.CUDA_VISIBLE_DEVICES=\'2\' \
EXEC:args.batch_size=1000 \
EXEC:args.batch_number=0 \
EXEC:args.saveFile=\'test.combine.perread.birnn.readnumber.1.npz\'


I want to dump for the i'th batch ( batch=1000, column=640, readNum=16):
predhp: (batch, readNum, column, 133)
predcall: (batch, readNum, column, 2)

In a separate file is corresponding truth :
truehp: (batch, column, 4)
truecall: (batch, column, 2)
"""

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
            for b in [ args.batch_number ]:

                start = time.time()

                if b==-1:
                    # special dump only truth
                    data_loader.set_batch_pointer(0)
                    x, y = data_loader.next_batch()
                    np.savez_compressed(args.saveFile, truehp=y[0], truecall=y[1])
                    return()

                data_loader.set_batch_pointer( b )
                x, y = data_loader.next_batch()

                # set up space for readnumber
                x.append(None)
                y.append(None)

                # set up space for all objects and readNumbers
                outPredhp = np.zeros( (args.batch_size, args.rows, args.cols, 133), dtype=np.float32)
                outPredcall = np.zeros( (args.batch_size, args.rows, args.cols, 2), dtype=np.float32)
                outInBaseint = np.zeros( (args.batch_size, args.rows, args.cols), dtype=np.float32) # the input base. 0=empty

                for readNumber in range(args.rows):
                    print("readNumber:",readNumber)
                    readNumberArray = np.full( (x[0].shape[0],1), readNumber, dtype=np.float32)
                    x[-1]=readNumberArray
                    y[-1]=readNumberArray

                    # get the model predications
                    predictions = model.model.predict(x)
                    for ii in range(len(predictions)):
                      print("ii %d predictions[ii].shape" %ii, predictions[ii].shape)

                    #### place predictions in correct places in output
                    for obj in range(predictions[0].shape[0]):
                        #for column in range(predictions[0].shape[1]):
                            outPredhp[obj, readNumber,:,:] = predictions[0][obj,:,:]
                            outPredcall[obj, readNumber,:,:] = predictions[1][obj,:,:]
                            outInBaseint[obj, readNumber,:] = x[0][obj,readNumber,:,0]
                            
            # Now I have all the data. Write it out
            np.savez_compressed(args.saveFile, predhp=outPredhp, predcall=outPredcall, inBasein=outInBaseint)

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
