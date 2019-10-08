import sys

for line in sys.stdin.read().splitlines():
  oplen =[]
  op = []
  tmp  = []

  for ii in range(len(line)):
    if line[ii] in "=XDIS":
        op.append(line[ii])
        oplen.append(int("".join(tmp)))
        tmp=[]
    else:
        tmp.append(line[ii])

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

  print("%f\t%d\t%d" % (err, (numdel+numins+nummm), tlen))
      
