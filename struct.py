import random

class struct:
    ################################
    def __init__(self):
        #random.seed(42)
        pass

    ################################
    def old(self):
        tmp = {}
        tmp["P_0NumKern"] = 8
        tmp["P_1NumKern"] = 8
        tmp["P_1XKern"] = 16
        tmp["P_1YKern"] = 1
        return(tmp)

    ################################
    def rnd1(self):
        tmp = {}
        tmp["P_0NumKern"] = random.randint(8,32)
        tmp["P_1NumKern"] = random.randint(2,8)
        tmp["P_1XKern"] = random.randint(8,16)
        tmp["P_1YKern"] = random.randint(1,8)
        return(tmp)

    ################################
    def fixed2(self):
        tmp = {}
        tmp["P_0NumKern"] = 25
        tmp["LengthsNumKern"] = 33
        tmp["LengthsXKern"] = 1
        tmp["LengthsYKern"] = 33
        return(tmp)

    ################################
    def oldfoobar(self):
        """Wanted to exec(myexec, globals(),locals())
        """
        return("""
global inputs
global x
global bottle
global predictionsHPLEN
global predictionsHPID
inputs = KK.layers.Input(shape=(args.rows,args.cols,args.baseinfo))
x = KK.layers.Conv2D(8, kernel_size= (1, 6), strides=(1,5), activation='relu')(inputs)
x = KK.layers.Conv2D(8, (16, 1), activation='relu')(x)
bottle = KK.layers.Flatten()(x) # This is now the bottleneck
predictionsHPLEN = KK.layers.Dense(args.hpdist, activation='softmax')(bottle)
predictionsHPID = KK.layers.Dense(4, activation='softmax')(bottle)
        """)

