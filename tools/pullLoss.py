import sys

def pullSpecs( dat ):
    """>>>>myparam -4431890095789198542
{'P_1XKern': 16, 'P_0NumKern': 32, 'P_1NumKern': 4, 'P_1YKern': 7}
>>>>myparam -4431890095789198542
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 16, 640, 10) 0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 16, 127, 32)  1952        input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 1, 121, 4)    14340       conv2d[0][0]                     
__________________________________________________________________________________________________
flatten (Flatten)               (None, 484)          0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 4)            1940        flatten[0][0]                    
__________________________________________________________________________________________________
dense (Dense)                   (None, 33)           16005       flatten[0][0]                    
==================================================================================================
Total params: 34,237
Trainable params: 34,237
Non-trainable params: 0
    """

    struct = {}
    for ii in range(len(dat)):
        ll = dat[ii]
        if "XKern" in ll:
            for (kk,vv) in eval(ll).items():
                struct[kk]=vv

        if "conv2d" in ll or "flatten" in ll or "dense" in ll:
            ll2 = ll.replace(", ",",")
            ff = ll2.split()
            struct[ff[0]+"SHA"] = ff[2]
            struct[ff[0]+"NUM"] = ff[3]
            
    tmp = []
    for kk in sorted(struct.keys()):
        tmp.append("%s\t%s" % (kk, struct[kk]))
    return("\t".join(tmp))
                     
def update( old, new ):
    tmp = old
    if float(new[-1]) <= old[0]:
        tmp = ( float(new[-1]), new[1] )
    return(tmp)

for ii in range(1,len(sys.argv)):
    file=sys.argv[ii]


    #myspec = pullSpecs( open(file).read().splitlines() )
    myspec="myspec"

    minTrain = (999.999,"na")
    minTest = (999.999,"na")

    for ll in open(file):
        if "LossAvg" in ll:
            ff = ll.strip().split()
            if "train" in ll:
                minTrain = update(minTrain,ff)
            else:
                minTest = update(minTest,ff)

    print("%s\tminTrain\t%f\t%s\tminTest\t%f\t%s\t%s" % (file,minTrain[0],minTrain[1],minTest[0],minTest[1],myspec))
