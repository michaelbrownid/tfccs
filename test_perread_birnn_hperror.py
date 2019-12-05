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

        # load the model based on name and access as Model: from .model import args.model
        myimport = importlib.import_module("tfccs.%s" % args.model)
        Model = myimport.Model

        model = Model(args)

        # make it appear as though there is only one gpu and use it
        os.environ["CUDA_VISIBLE_DEVICES"]=args.CUDA_VISIBLE_DEVICES

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # restore model. Defined losses aren't used in testing
            def nullloss(xx,yy): return(KK.backend.mean(yy,axis= -1))

            model.model = KK.models.load_model(args.modelsave, custom_objects={"KK":KK, 
                                                                               "zero_loss": nullloss,
                                                                               "myloss": nullloss,
                                                                               "sparse_kl": nullloss})
            ################################
            numerrcall = 0
            totalcall = 0
            numerrhpbase = 0
            numerrhplength = 0
            total = 0
            objnum = 0
            first=True
            idx2Base = ["A","C","G","T",""]

            for b in range(data_loader.num_batches):

                start = time.time()
                x, y = data_loader.next_batch()

                # calltruedat = np.expand_dims(self.dat["windowHPCallTrue"][mystart:myend,],axis= -1)
                # callpropdat = np.expand_dims(self.dat["windowHPCallProp"][mystart:myend,],axis= -1)
                # inputs = [ self.dat["windowinput"][mystart:myend,],
                #            callpropdat,
                #            calltruedat ]

                # calltruedat = np.expand_dims(self.dat["windowHPCallTrue"][mystart:myend,],axis= -1)
                # # make p=0/1 into binary p,(1-p)
                # calltruedatbinary = np.concatenate((calltruedat, 1.0-calltruedat), axis= -1)
                # #hackshape = calltruedat.shape
                # #hackshape = (hackshape[0],hackshape[1],128)
                # #np.zeros( hackshape ), # for RNN output, zero_loss (keras doesn't have optional in/out!)
                # outputs = [ 
                #             self.dat["windowoutputDirectHP"][mystart:myend,:,0:4],
                #             self.dat["windowoutputDirectHP"][mystart:myend,:,4:],
                #             calltruedatbinary,
                #             np.zeros_like(calltruedat) ] # hack, zeros for batchsize, zero_loss (keras doesn't have optional in/out!)

                # self.model = KK.models.Model(inputs=[inputs,inputsCallProp,inputsCallTrue],
                #                              outputs=[rnnconcat,predHPBase,predHPLen,predHPCall,inputsCallTrue])
                # ii 0 predictions[ii].shape (1000, 640, 4)
                # ii 1 predictions[ii].shape (1000, 640, 33)
                # ii 2 predictions[ii].shape (1000, 640, 2)
                # ii 3 predictions[ii].shape (1000, 640, 1) # this is just inputscalltrue for hack

                predictions = model.model.predict(x)

                ######## Dump the predictions and truth
                """ 
                - Count number of times that call is correct tp, fp, fn

                - For correct counts: count number of times hpbase and hplength is correct

                """

                consensusSeq = []
                trueSeq = []
                for obj in range(predictions[2].shape[0]):

                    consensusSeq.append( [] )
                    trueSeq.append( [] )

                    for column in range(predictions[2].shape[1]):
                        truebase = np.argmax(y[0][obj][column])
                        truelength = np.argmax(y[1][obj][column])
                        predbase = np.argmax(predictions[0][obj][column])
                        predlength = np.argmax(predictions[1][obj][column])

                        if predictions[3][obj][column]>=0.5:
                            # true call
                            totalcall += 1
                            trueSeq[obj].append(idx2Base[truebase]*truelength)

                            if predictions[2][obj][column][0]<0.5:
                                # FN call
                                numerrcall += 1
                                print("objnum", objnum, "column", column, "err fncall", predictions[2][obj][column][0], truebase,truelength, predbase,predlength)
                            else:
                                # TP call
                                total += 1
                                consensusSeq[obj].append(idx2Base[predbase]*predlength)

                                # check calls
                                if truebase != predbase:
                                    numerrhpbase += 1
                                    print("objnum", objnum, "column", column, "err hpbase", predictions[0][obj][column][predbase], "true", truebase, "pred", predbase)

                                if truelength != predlength:
                                    numerrhplength += 1
                                    print("objnum", objnum, "column", column, "err hplength", predictions[1][obj][column][predlength], "true", truelength, "pred", predlength)

                        else:
                            # non call
                            if predictions[2][obj][column][0]<0.5:
                                # TN call
                                pass
                            else:
                                # FP call
                                numerrcall += 1
                                print("objnum", objnum, "column", column, "err fpcall", predictions[2][obj][column][0], truebase,truelength, predbase,predlength)
                                consensusSeq[obj].append(idx2Base[predbase]*predlength)

                    objnum+=1 # for obj
                    
                #### evaluate full seqs
                fptrue = open("seqtrue.fa","a")
                fpest = open("seqest.fa","a")
                for obj in range(len(trueSeq)):
                    fptrue.write(">true-%d-%d\n" % (b,obj))
                    fptrue.write("".join(trueSeq[obj]))
                    fptrue.write("\n")

                    fpest.write(">est-%d-%d\n" % (b,obj))
                    fpest.write("".join(consensusSeq[obj]))
                    fpest.write("\n")
                fptrue.close()
                fpest.close()

                print("BATCH %d call %f = %d / %d" % (b,float(numerrcall)/totalcall,numerrcall,totalcall))
                print("BATCH %d hpbase %f = %d / %d" % (b,float(numerrhpbase)/total,numerrhpbase,total))
                print("BATCH %d hplength %f = %d / %d" % (b,float(numerrhplength)/total,numerrhplength,total))

                #break # only look at 0th batch for time

            print("FINAL %d call %f = %d / %d" % (b,float(numerrcall)/totalcall,numerrcall,totalcall))
            print("FINAL %d hpbase %f = %d / %d" % (b,float(numerrhpbase)/total,numerrhpbase,total))
            print("FINAL %d hplength %f = %d / %d" % (b,float(numerrhplength)/total,numerrhplength,total))

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
