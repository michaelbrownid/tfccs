"""
Take npz prediction dump and make an image of the model probabilities.

python3 -m tfccs.tools.predictionsToImage \
output_model1ondata1 \
windownum 33 \
begin 240 \
end 381 \

"""

import sys
import math
import numpy as np
import os

################################
def probToUnit( prob ):
  eps=1E-6
  lp = math.log2(prob+eps)
  # lp ranges from log2(1E-6)=-19.93157 to log2(1+1E-6)=1.442694e-06
  mymin=math.log2(eps)
  mymax=math.log2(1+eps)
  myval = (lp-mymin)/(mymax-mymin)
  return(myval)

def unitTo255( xx ):
    return( int(255*xx + 0.5))

def matrixToPPM(matrix, outfile):
    ofp = open(outfile,"w")
    mydim = matrix.shape
    ofp.write("P3\n")
    ofp.write(str(mydim[1])) # X = columns
    ofp.write(" ")
    ofp.write(str(mydim[0])) # Y = rows
    ofp.write(" ")
    ofp.write("255")
    ofp.write("\n")
    for yy in range(mydim[0]):
        for xx in range(mydim[1]):
            for pp in range(3):
                ofp.write(str(matrix[yy,xx,pp])) # rows,cols
                ofp.write("\n")
    ofp.close()

################################
def predictionsToImage( datpredhp, dathpredcall,datinbaseint, pngfile ):

    """predictionsToImage:

    call0
    call1
    xxxx (L3)
    baseint==0
    xxxx (L2)
    hp0
    Ahp
    Chp
    Ghp
    Thp
    xxx (+ 1 33 33 33 33 1)=134

    height = (+ 3 2 134)=139
    """

    print("datpredhp.shape",datpredhp.shape)
    print("datpredcall.shape",datpredcall.shape)
    print("datinbaseint.shape",datinbaseint.shape)
    
    mywidth = datpredhp.shape[0]
    myheight = 139

    mymatrix = np.zeros( (myheight,mywidth,3), dtype="int32")

    cursor = 0

    #### draw first row in white
    for ii in range(mywidth): mymatrix[cursor,ii] = np.array([255,255,255])

    cursor += 1

    ######## call data 0/1
    
    #### draw pred,1-pred in red
    for ii in range(mywidth):
        if datpredcall[ii,0] > datpredcall[ii,1]:
            # 0 is max
            fill0 = np.array( [ 0,unitTo255(datpredcall[ii,0]),0]) # green for max
            fill1 = np.array( [ unitTo255(datpredcall[ii,1]),0,0]) # red for non-max
        else:
            # 1 is max
            fill0 = np.array( [ unitTo255(datpredcall[ii,0]),0,0]) # red for non-max
            fill1 = np.array( [ 0,unitTo255(datpredcall[ii,1]),0]) # green for max

        mymatrix[cursor,ii] = fill0
        mymatrix[cursor+1,ii] = fill1
    cursor+=2

    #### draw  row in white
    for ii in range(mywidth): mymatrix[cursor,ii] = np.array([255,255,255])
    cursor += 1

    #### baseint==0
    for ii in range(mywidth):
        if datinbaseint[ii] == 0:
            fill = np.array( [ 0,0,0]) # black for no base
        else:
            fill = np.array( [ 128,128,128]) # grey for no base
        mymatrix[cursor,ii] = fill
    cursor+=1

    #### draw  row in white
    for ii in range(mywidth): mymatrix[cursor,ii] = np.array([255,255,255])
    cursor += 1

    #### do all HP
    for ii in range(mywidth):
        # draw all in red
        for jj in range(1+4*33):
            fill = np.array( [ unitTo255(datpredhp[ii,jj]),0,0]) # red for non-max
            mymatrix[cursor+jj,ii] = fill
        # draw max in green
        for jj in [np.argmax(datpredhp[ii])]:
            fill = np.array( [ 0,unitTo255(datpredhp[ii,jj]),0]) # green for max
            mymatrix[cursor+jj,ii] = fill
    cursor+=(1+4*33)

    #### draw  row in white
    #for ii in range(mywidth): mymatrix[cursor,ii] = np.array([255,255,255])
    #cursor += 1

    #### Now mymatrix is filled out. Write PPM and convert it
    matrixToPPM(mymatrix, pngfile+".ppm")
    cmd = "convert %s.ppm -scale 800%% %s" % (pngfile,pngfile)
    print("cmd",cmd)
    os.system(cmd)

if __name__ == "__main__":
  
  npzprefix    =sys.argv[1]      
  windownum  =int(sys.argv[3]) 
  begin      =int(sys.argv[5]) 
  end        =int(sys.argv[7]) 

  ################################
  # get the product. special because there are no subreads only the product over subread from makeCallProd133.py.
  dat = np.load("%s.callprod.subreads.perread.birnn.npz" % npzprefix)
  datpredhp = dat["predhp"][windownum][begin:end]
  datpredcall = dat["predcall"][windownum][begin:end]
  datinbaseint = dat["inBaseint"][windownum][begin:end]
  predictionsToImage( datpredhp, datpredcall,datinbaseint, "%s_product.png" % npzprefix )

  ################################
  # get all the subread images

  dat = np.load("%s.subreads.perread.birnn.npz" % npzprefix)
  for readnum in range(16):
    datpredhp = dat["predhp"][windownum][readnum][begin:end]
    datpredcall = dat["predcall"][windownum][readnum][begin:end]
    datinbaseint = dat["inBaseint"][windownum][readnum][begin:end]
    predictionsToImage( datpredhp, datpredcall,datinbaseint, "%s_%d.png" % (npzprefix,readnum) )

