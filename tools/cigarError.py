import sys

totalerrs = 0
totalbases = 0

for line in sys.stdin:
  ff = line.split()
  readid = ff[0]
  refid = ff[2]
  cigar = ff[5]

  oplen =[]
  op = []
  tmp  = []

  for ii in range(len(cigar)):
    if cigar[ii] in "=XDIS":
        op.append(cigar[ii])
        oplen.append(int("".join(tmp)))
        tmp=[]
    else:
        tmp.append(cigar[ii])

  tlen = 0
  numdel = 0
  numins = 0
  nummm = 0
  for ii in range(len(op)):
    if op[ii]!="I" and op[ii]!="S":
      tlen+=oplen[ii]
    if op[ii]=="D":
      numdel+=oplen[ii]
    if op[ii]=="I":
      numins+=oplen[ii]
    if op[ii]=="X":
      nummm+=oplen[ii]

  err = float(numdel+numins+nummm)/tlen

  totalerrs += numdel+numins+nummm
  totalbases += tlen
  print("%s\t%s\t%f\t%d\t%d" % (readid, refid, err, (numdel+numins+nummm), tlen))
      
print("# overallerror = %f = %d / %d" % ( float(totalerrs)/totalbases, totalerrs, totalbases))
