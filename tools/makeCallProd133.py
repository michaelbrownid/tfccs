import numpy as np
import sys

outputprefix = sys.argv[2]

ofpseq = open("%s.fasta" % outputprefix, "w")

################################
def idx2hp( idx ):
    idx = int(idx)
    if idx == 0: return("X") # no call, shouldn't happen the way I'm using it!

    idx2Base = ["A","C","G","T",""]

    idx = idx-1 # now 0:132

    mybase = idx2Base[ int(idx/33) ]
    mylen = idx % 33
    return( (mybase, mylen) )

################################
# load dataset
dat = np.load(sys.argv[1])

# list(dat.keys())
# ['predhp', 'predlen']

#### DEBUG
if False:
    predhp = dat["predhp"]
    predcall = dat["predcall"]

    obj=0
    read=0
    for col in range(640):
        hp=predhp[obj,read,col,:]
        predidx = np.argmax(hp) 
        predprob = np.max(hp)
        (predbase,predlength) = idx2hp( predidx )
        print("***",col,predcall[obj,read,col,0],predcall[obj,read,col,1],"=",predidx,predprob,predbase,predlength,hp.shape)
    sys.exit(1)


################################

predhp = dat["predhp"] # (1000, 16, 640, 133)
predcall = dat["predcall"] # (1000, 16, 640, 2)
inBaseint = dat["inBaseint"] # (1000, 16, 640)


#### take product (mean) only where inBaseint is not 0 meaning there was a base there.
#### this can be done by setting predX=1.0 everywhere inBaseint==0

predcall[ inBaseint==0.0 ] = 1.0
predhp[ inBaseint==0.0 ] = 1.0

predcallprod = np.prod(predcall,axis=1)
predhpprod = np.prod(predhp,axis=1)
inBaseintprod = np.any(inBaseint!=0,axis=1) # if any not 0 then 1, else all 0 so 0

#print("predhpprod.shape",predhpprod.shape)
#print("predcallprod.shape",predcallprod.shape)
#predhpprod.shape (1000, 640, 133)
#predcallprod.shape (1000, 640, 2)

#### for product, must renormalize to compare to 0.5
#renormpredcall = predcallprod[:,:,1] / (predcallprod[:,:,0]+predcallprod[:,:,1])
z=np.sum(predcallprod, axis= -1, keepdims=True)
z2 = np.repeat(z, repeats=2, axis=-1)
predcall = predcallprod / z2

z=np.sum(predhpprod, axis= -1, keepdims=True)
z2 = np.repeat(z, repeats=133, axis=-1)
predhp = predhpprod / z2

# save the normalized distributions
outfilenpz = "%s.npz" % outputprefix
np.savez_compressed(outfilenpz, predhp=predhp, predcall=predcall, inBaseint=inBaseintprod)

################################
#### Now I have all the calls, write them to a fasta file so I can compute an error rate

for obj in range(predcall.shape[0]):
    seq = []
    for col in range(predcall.shape[1]):
        # make call and add HP
        if predcall[obj,col,1]>0.5:
            # pull hp data vector over 133 possibilities
            dat = predhp[obj,col]
            predidx = np.argmax(dat)
            predprob = np.max(dat)
            (predbase,predlength) = idx2hp( predidx )
            #if obj==112: print("***",predcall[obj,col,1],predidx, predprob,predbase,predlength)
            seq.append(predbase*predlength)

    # output fasta after going through all columns
    print(">obj-%d" % obj, file=ofpseq)
    print("".join(seq), file=ofpseq)
