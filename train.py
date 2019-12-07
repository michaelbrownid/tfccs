#!/usr/bin/env python

from __future__ import print_function
import argparse
import time
import os
import sys
import tensorflow as tf
import tensorflow.keras
import numpy as np
#from .model import Model
from . import data
import importlib
import tensorflow.keras as KK

################################
def train(args):

    #with tf.device("/cpu:0"):
    #with tf.device("/gpu:3"):
    if True:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.CUDA_VISIBLE_DEVICES

        # load the model based on name and access as Model: from .model import args.model
        myimport = importlib.import_module("tfccs.%s" % args.model)
        Model = myimport.Model

        model = Model(args)

        #model.model.save(args.modelsave)
        # TypeError: ('Not JSON Serializable:', <function fubarClosure.<locals>.func at 0x7fd4c6109c80>)
        #with open("%s.model.json" % args.modelsave,"w") as f:
        #    f.write(model.model.to_json())
        #model.model.save_weights("%s.model.h5" % args.modelsave)

        t0=time.time()
        data_loader = data.data( args.batch_size, sys.argv[2],
                                 inputdatName=args.inputdatName,
                                 outputdatName=args.outputdatName)
        t1=time.time()
        print("time data_loader",str(t1-t0))

        # get test if there
        data_loader_test = None
        if len(sys.argv)>3:
            t0=time.time()
            data_loader_test = data.data( args.batch_size, sys.argv[3],
                                          inputdatName=args.inputdatName,
                                          outputdatName=args.outputdatName)
            t1=time.time()
            print("time data_loader test",str(t1-t0))

        # # check compatibility if training is continued from previously saved model
        # if args.init_from is not None:
        #     # check if all necessary files exist
        #     assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        #     ckpt = tf.train.latest_checkpoint(args.init_from)
        #     assert ckpt, "No checkpoint found"

        # if not os.path.isdir(args.save_dir):
        #     os.makedirs(args.save_dir)
            
        tfconfig=tf.ConfigProto()
        # tfconfig.allow_soft_placement=True
        # tfconfig.log_device_placement=True
        # tfconfig.gpu_options.allow_growth=True

        with tf.Session( config=tfconfig ) as sess:
            sess.run(tf.global_variables_initializer())

            if hasattr(args, "modelrestore"):
                model.model = KK.models.load_model(args.modelrestore)
                print("restored model", args.modelrestore)

            print("# args.num_epochs", args.num_epochs, "args.batch_size", args.batch_size, "num_batches", data_loader.num_batches)

            testLossAvgMIN = 999.9E+99
            first=True
            for e in range(args.num_epochs):
                storeloss = []
                data_loader.reset_batch_pointer()
                for b in range(data_loader.num_batches):
                    start = time.time()
                    x, y = data_loader.next_batch()

                    # train only on readNumber=1 for now. can cycle through them all.
                    # must be on output as hack to get keras to write model!
                    readNumber = args.readNumber
                    readNumberArray = np.full( (x[0].shape[0],1), readNumber, dtype=np.float32)
                    x.append(readNumberArray)
                    y.append(readNumberArray)

                    if first:
                        first=False
                        # print("x.shape",x.shape)
                        # print("y.shape",y.shape)
                        # print("x[4]",x[4])
                        # print("y[4]",y[4])
                        for ii in range(len(x)):
                            print("ii",ii,"x[ii].shape",x[ii].shape)
                        for ii in range(len(y)):
                            print("ii",ii,"y[ii].shape",y[ii].shape)
                            
                    # print("========")
                    # print("y.shape",y.shape)
                    # print("y[0,0]",y[0,0])
                    # predictions = model.model.predict(x[0:2,])
                    # print("predictions.shape",predictions.shape)
                    # for oo in range(predictions.shape[0]):
                    #     for cc in range(predictions.shape[1]):
                    #         print("meanStdArgmax",oo,cc,np.mean(predictions[oo,cc]),np.std(predictions[oo,cc]),np.argmax(predictions[oo,cc]), np.max(predictions[oo,cc]), y[oo,cc], predictions[oo,cc,y[oo,cc]])

                    #myfit=model.model.fit( x, [yid,ylen], epochs=1, batch_size=1,verbose=2)

                    myfit = model.model.train_on_batch( x, y )

                    end = time.time()
                    #print("epoch %d batch %d time %f" % (e, b, end-start))
                    for (kk,vv) in zip(model.model.metrics_names,[myfit]):
                          print("epoch",e,"batch",b,"trainMetric",kk," ".join([str(xx) for xx in vv]))
                          if kk=="loss":
                              if not isinstance(vv,list): vv = [vv] # only single loss
                              # handle multiple inputs
                              if isinstance(x,list):
                                  myx = x[0]
                              else:
                                  myx=x
                              storeloss.append( (vv[0],myx.shape[0]) )

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

                    # checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    # saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    # print("model saved to {}".format(checkpoint_path))

                    model.model.save(args.modelsave)
                    # TypeError: ('Not JSON Serializable:', <function fubarClosure.<locals>.func at 0x7fd4c6109c80>)
                    #with open("%s.model.json" % args.modelsave,"w") as f:
                    #    f.write(model.model.to_json())
                    #model.model.save_weights("%s.model.h5" % args.modelsave)

                    # run the test set if there
                    if data_loader_test is not None:
                        storeloss = []
                        data_loader_test.reset_batch_pointer()
                        for b in range(data_loader_test.num_batches):
                            x, y = data_loader_test.next_batch()

                            # train only on readNumber=1 for now. can cycle through them all.
                            # must be on output as hack to get keras to write model!
                            readNumber = args.readNumber
                            readNumberArray = np.full( (x[0].shape[0],1), readNumber, dtype=np.float32)
                            x.append(readNumberArray)
                            y.append(readNumberArray)

                            #mytest=model.model.evaluate( x, [yid,ylen],verbose=0)
                            mytest = model.model.test_on_batch( x, y )

                            for (kk,vv) in zip(model.model.metrics_names,[mytest]):
                                #print("epoch",e,"batch",b,"trainMetric",kk,"=",vv,"batchsize",x.shape[0])
                                if kk=="loss":
                                    if not isinstance(vv,list): vv = [vv] # only single loss
                                    # handle multiple inputs
                                    if isinstance(x,list):
                                        myx = x[0]
                                    else:
                                        myx=x
                                    storeloss.append( (vv[0],myx.shape[0]) ) # vv[0] for multiple losses

                        # compute average loss across all batches
                        testnum = 0
                        testsum = 0.0
                        for xx in storeloss:
                            testsum += xx[0]*xx[1]
                            testnum += xx[1]
                        testLossAvg = testsum/float(testnum)
                        print("epoch %d testLossAvg %f" % (e , testLossAvg))
                        if testLossAvg < testLossAvgMIN:
                            testLossAvgMIN = testLossAvg
                            cmd = "mv %s %s.best" % (args.modelsave, args.modelsave)
                            print(cmd)
                            os.system(cmd)
                            cmd = "mv %s.model.json %s.model.json.best" % (args.modelsave, args.modelsave)
                            print(cmd)
                            os.system(cmd)
                            cmd = "mv %s.model.h5 %s.model.h5.best" % (args.modelsave, args.modelsave)
                            print(cmd)
                            os.system(cmd)
                        if testLossAvg > 2.0*testLossAvgMIN:
                            print("EARLY STOPPING:",testLossAvg,testLossAvgMIN)
                            return()

                #sys.exit(1)

if __name__ == '__main__':

    print("time begin",str(time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(time.time()))))

    exec(open(sys.argv[1]).read())
    args.init_from = None

    for aa in sys.argv:
        if "EXEC:" in aa:
            toexec = aa.replace("EXEC:","")
            print("toexec",toexec)
            exec(toexec)
            
    print("-------")
    train(args)

    print("time end",str(time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(time.time()))))
