import numpy as np
import sys

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

#list(dat.keys())
#['truehplen','truebase']

################################

# just propagate single truth forward

predcall = dat["truecall"]
predhp = dat["truehp"]

#predhp.shape (1000, 640, 1)
#predcall.shape (1000, 640, 1)

################################
#### Now I have all the calls, write them to a fasta file so I can compute an error rate

for obj in range(predcall.shape[0]):
    seq = []
    for col in range(predcall.shape[1]):
        # make call and add HP
        if predcall[obj,col]>0.5: # values are 0/1
            predidx = predhp[obj,col] # true is single integer
            predprob = 1.0
            (predbase,predlength) = idx2hp( predidx )
            #if obj==112: print("***",predcall[obj,col,1],predidx, predprob,predbase,predlength)
            seq.append(predbase*predlength)

    # output fasta after going through all columns
    print(">obj-%d" % obj)
    print("".join(seq))
