import numpy as np
import random
import sys
import zipfile
import os
import multiprocessing as mp
import time

################################
def timeit( func, *args, **kwargs ):
    ts = time.time()
    tmp = func(*args, **kwargs)
    te = time.time()
    print("timeit", (te-ts))
    return(tmp)

#def mult(x,y):
#    return(x*y)
#timeit(mult, 5,3)
#inputdat = timeit(np.concatenate, (inputdat, tmprs), axis=0)

################################
def loadonefile( filename, myzfname=None ):
    print("loadonefile",  filename, "from", myzfname, file=sys.stderr)
    prefix = ""
    if myzfname is not None: 
        zf = zipfile.ZipFile(myzfname, mode='r') # parallel: can't pass zf without corruption so open here!
        print("loadonefile extract")
        prefix=""
        timeit( zf.extract, filename) #, path=prefix )
        zf.close()
    print("loadonefile np.load")
    tmp = timeit( np.load, prefix+filename )
    tmpdat = dict(tmp) # force lazy load of all data!!!!
    #print("loadonefile",  filename, "tmp['windowinput'].shape", tmp['windowinput'].shape)
    if myzfname is not None: os.remove(prefix+filename)
    return(tmpdat)

################################
class data:
    ################################
    def __init__(self, batch_size, datafile, inputdatName="input", outputdatName="output", isZip=False, shortcut=False):

        self.batch_size = batch_size
        self.datafile = datafile
        self.inputdatName = inputdatName
        self.outputdatName = outputdatName

        self.loaddata( datafile, isZip=isZip, shortcut=shortcut)

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
                mydat = tmp
            # rest
            if not shortcut:
                    for ii in range(1,len(filenames)):
                        df = filenames[ii]
                        tmp = loadonefile(df, myzfname=zfname)
                        for kk in tmp.keys():
                            mydat[kk] = np.concatenate( (mydat[kk], tmp[kk]), axis=0 )
            return(mydat)
        ####

        #### filenames can be a list of zip files with ~200 numpy arrays.
        #### load all numpy arrays in each zip file
        filenames=open(datafile).read().splitlines()
        if not isZip:
            self.dat = loadfilenames( filenames )
        else:
            first=True
            for ff in filenames:
                print("loaddata zipfile",ff)
                zf = zipfile.ZipFile(ff, mode='r')
                zipfilenames = zf.namelist()
                zf.close()
                mydat = loadfilenames(zipfilenames, ff)
                if first:
                    self.dat = mydat
                    first=False
                    if shortcut: break
                else:
                    for kk in mydat.keys():
                        self.dat[kk] = np.concatenate( (self.dat[kk],mydat[kk]), axis=0 )

        for kk in self.dat.keys():
            print("loaddata final shape", kk, self.dat[kk].shape)

    ################################
    # seperate the whole data into different batches.
    def create_batches(self):
        numpoints = self.dat[ list(self.dat.keys())[0] ].shape[0]
        if (numpoints % self.batch_size) == 0:
            extra = 0
        else:
            extra = 1
        self.num_batches = int(numpoints/self.batch_size)+extra
        #print("create_batches inputdat  shape", self.inputdatName, self.dat[self.inputdatName].shape)
        #print("create_batches outputdat shape", self.outputdatName, self.dat[self.outputdatName].shape)

    def reset_batch_pointer(self):
        self.pointer = 0

    def set_batch_pointer(self, batch):
        self.pointer = batch

    def next_batch(self):
        # TODO: compute begin/end to get it right at last batch
        mystart = self.pointer * self.batch_size
        myend = (self.pointer+1) * self.batch_size

        if self.inputdatName=="windowinputCTX":
            # provide both windowinput and windowinputKmer
            inputs = [ self.dat["windowinput"][mystart:myend,],
                       self.dat["windowinputKmer"][mystart:myend,] ]
        elif self.inputdatName=="windowinputPlusCallProp":
            callpropdat = np.expand_dims(self.dat["windowHPCallProp"][mystart:myend,],axis= -1)
            calltruedat = np.expand_dims(self.dat["windowHPCallTrue"][mystart:myend,],axis= -1)
            inputs = [ self.dat["windowinput"][mystart:myend,],
                       callpropdat.astype('int32'),
                       calltruedat.astype('int32') ]
        else:
            fullinputdat = self.dat[self.inputdatName]
            inputs = fullinputdat[mystart:myend,]

        if "windowoutputDirectHPPlusCallTrue" in self.outputdatName:
            calltruedat = np.expand_dims(self.dat["windowHPCallTrue"][mystart:myend,],axis= -1)
            outputs = [ 
                        np.expand_dims(self.dat["windowoutputDirectHP"][mystart:myend,:],axis= -1),
                        calltruedat,
                        np.zeros_like(calltruedat) ] # hack, zeros for batchsize, zero_loss (keras doesn't have optional in/out!)
        elif "HP" in self.outputdatName:
            # provide both HP identity and length
            outputs = [ self.dat[self.outputdatName][mystart:myend,:,0:4],
                        self.dat[self.outputdatName][mystart:myend,:,4:] ]
        elif "Kmer" in self.outputdatName:
            fulloutputdat = self.dat[self.outputdatName]
            #outputs = np.expand_dims(fulloutputdat[mystart:myend,],2) # (14986, 640) -> (14986, 640, 1)
            outputs = fulloutputdat[mystart:myend,].astype("int32") # float->int
            outputs = np.abs(outputs) # edge case -1 at end gets replaced with 1
        else:
            fulloutputdat = self.dat[self.outputdatName]
            outputs = fulloutputdat[mystart:myend,]

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

################################

if __name__ == "__main__":

    ### load the data from sys.argv[1]
    t0=time.time()
    data_loader = data( 128, sys.argv[1], isZip=True)
    t1=time.time()
    print("time data_loader",str(t1-t0))

    ### save the data as npz in sys.argv[2]
    t0=time.time()
    name2dat= {}
    for kk in data_loader.dat.keys():
        name2dat[kk] = data_loader.dat[kk]
    np.savez_compressed(sys.argv[2], **name2dat) # this will save each of the data under the name
    t1=time.time()
    print("time savez_compressed",str(t1-t0))

