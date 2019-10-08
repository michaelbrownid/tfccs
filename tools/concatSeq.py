import sys

dat = open(sys.argv[1]).read().splitlines()
print(">concatseq/1111")
ii=0
start=0
while ii<len(dat):
    myid = dat[ii]
    ii+=1
    myseq = dat[ii]
    ii+=1
    print>>sys.stderr, myid, start, start+len(myseq)
    print myseq
    start = start+len(myseq)

    
