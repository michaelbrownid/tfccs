import numpy as np
import random
import sys
import zipfile
import os
import multiprocessing as mp

################################
def loadonefile( filename, myzfname=None ):
    print("loadonefile",  filename, "from", myzfname)
    if myzfname is not None: 
        zf = zipfile.ZipFile(myzfname, mode='r') # can't pass zf without corruption so open here!
        zf.extract(filename) # ,path=workdir)
        zf.close()
    tmp = np.load(filename)
    print("loadonefile",  filename, "tmp['windowinput'].shape", tmp['windowinput'].shape)
    if myzfname is not None: os.remove(filename)
    return(tmp)

################################
class data:
    ################################
    def __init__(self, batch_size, datafile, shortcut=False):

        self.batch_size = batch_size
        self.datafile = datafile

        mydat = self.loaddata( datafile, isZip=True, shortcut=shortcut)
        self.inputdat = mydat["inputdat"]
        self.outputdat = mydat["outputdat"]

        self.create_batches()
        self.reset_batch_pointer()

    ################################
    def loaddata( self, datafile, isZip=False, shortcut=False ):
        # shortcut only loads first of first

        ####
        def loadfilenames( filenames, zfname=None ):
            # first
            for ii in [0]:
                df = filenames[ii]
                tmp = loadonefile(df, myzfname=zfname)
                inputdat = tmp['windowinput']
                outputdat = tmp['windowoutput']
            # rest

            if not shortcut:
                #### try loading in parallel
                if False:
                    # mp.Pool can't pickle and mp appears to use pickle to transfer objects=bad for BIG
                    #pool = mp.Pool(processes=2)
                    pool = mp.pool.ThreadPool(processes=32)
                    # fixes transfer problem: multiprocessing.pool.MaybeEncodingError: Error sending result:
                    # 'TypeError(“cannot serialize '_io.BufferedReader' object”,)'
                    manyresults = [pool.apply_async(loadonefile, (ff,), {'myzfname':zfname}) for ff in filenames]
                    for res in manyresults:
                        tmp = res.get()
                        tmprs = tmp['windowinput']
                        inputdat = np.concatenate( (inputdat, tmprs), axis=0)
                        tmprs = tmp['windowoutput']
                        outputdat = np.concatenate( (outputdat, tmprs), axis=0)
                if False:
                    # mp.Pool can't pickle and mp appears to use pickle to transfer objects=bad for BIG
                    pool = mp.Pool(processes=8)
                    for tmp in pool.imap_unordered( mapcall, filenames ):
                        tmprs = tmp['windowinput']
                        inputdat = np.concatenate( (inputdat, tmprs), axis=0)
                        tmprs = tmp['windowoutput']
                        outputdat = np.concatenate( (outputdat, tmprs), axis=0)
                if True:
                    for ii in range(1,len(filenames)):
                        df = filenames[ii]
                        tmp = loadonefile(df, zf)
                        tmprs = tmp['windowinput']
                        inputdat = np.concatenate( (inputdat, tmprs), axis=0)
                        tmprs = tmp['windowoutput']
                        outputdat = np.concatenate( (outputdat, tmprs), axis=0)

            print("loadfilenames final inputdat.shape", inputdat.shape)
            print("loadfilenames final outputdat.shape", outputdat.shape)

            return( { "inputdat": inputdat, "outputdat": outputdat} )
        ####

        #### filenames can be a list of zip files with ~200 numpy arrays.
        #### load all numpy arrays in each zip file
        filenames=open(datafile).read().splitlines()
        if not isZip:
            mydat = loadfilenames( filenames )
            print("loaddata final inputdat.shape", inputdat.shape)
            print("loaddata final outputdat.shape", outputdat.shape)
            return( { "inputdat": mydat["inputdat"], "outputdat": mydat["outputdat"]} )
        else:
            first=True
            for ff in filenames:
                print("loaddata zipfile",ff)
                zf = zipfile.ZipFile(ff, mode='r')
                filenames = zf.namelist()
                zf.close()
                mydat = loadfilenames(filenames, ff)
                if first:
                    inputdat = mydat["inputdat"]
                    outputdat = mydat["outputdat"]
                    first=False
                    if shortcut: break
                else:
                    inputdat = np.concatenate( (inputdat, mydat["inputdat"]),axis=0)
                    outputdat = np.concatenate( (outputdat, mydat["outputdat"]),axis=0)
            print("loaddata zip final inputdat.shape", inputdat.shape)
            print("loaddata zip final outputdat.shape", outputdat.shape)
            return( { "inputdat": inputdat, "outputdat": outputdat} )

    ################################
    # seperate the whole data into different batches.
    def create_batches(self):
        if (self.inputdat.shape[0] % self.batch_size) == 0:
            extra = 0
        else:
            extra = 1
        self.num_batches = int(self.inputdat.shape[0]/self.batch_size)+extra

    def reset_batch_pointer(self):
        self.pointer = 0

    def next_batch(self):
        mystart = self.pointer * self.batch_size
        myend = (self.pointer+1) * self.batch_size
        inputs = self.inputdat[mystart:myend,]
        outputs = self.outputdat[mystart:myend,]
        self.pointer += 1
        return( inputs, outputs )

def printit( x ):
    print("----")
    print(x.shape)
    for bb in range(x.shape[0]):
      feats = []    
      for ff in x[bb]:
        feats.append("%s" % str(ff))
      print("%d: %s" % (bb," ".join(feats)))
