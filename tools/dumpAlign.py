import pysam
import sys
import datetime

################################
def readref( reffile ):
    first = True
    storage=[]
    fp = open(reffile)
    for line in fp:
        if first:
            # skip >id line
            first=False
        else:
            if ">" in line: break
            storage.append(line.strip())
    fp.close()
    return("".join(storage))

################################
def hpify( ref ):
    """return list of ranges showing the HP regions in the ref
    HPID = {50149542:("T",19),50149561:("N",0) }
    Base 50149542 is the start of a T19 HP.

hpify("AGGT")
{1: ('G', 2)}

hpify("AGGACCT")
{1: ('G', 2), 3: ('A', 1), 4: ('C', 2)}

"""

    # make sure begin and end aren't HPs
    # begin=0
    # while ref[begin]==ref[begin+1]:
    #     begin+=1
    # end=len(ref)-1
    # while ref[end]==ref[end-1]:
    #     end-=1

    # assume single base on begin and end that are the begin/end of HP region

    HPID={}
    begin=1
    end=len(ref)-1

    rangebegin = begin
    hpbase = ref[rangebegin]
    hplen=1
    this=rangebegin
    nextloc=this+1
    while (nextloc<end):
        while (nextloc<end) and (ref[nextloc]==ref[this]):
            this+=1
            nextloc+=1
            hplen+=1

        HPID[rangebegin] = (hpbase,hplen)
        #print >>sys.stderr, rangebegin,hpbase,hplen

        rangebegin=nextloc
        if rangebegin<end:
            hpbase = ref[rangebegin]
            hplen=1
            this=rangebegin
            nextloc=this+1

    # add last base placeholder to force output
    HPID[end] = (ref[end],0)

    return(HPID)

################################
def startdat():
    return( {"visited":0, "hplen":0, "numdel":0, "numins":0, "listins":[], "sumins":0, "numsub":0, "listsub":[]})

################################
def processBase(store, colpos, refbase, qn, readbase, readindel, insertion):

        store["visited"]+=1

        if readbase=="-":
            store["numdel"]+=1
        else:
            store["hplen"]+=1

        if readbase!="-" and readbase!=refbase:
            store["numsub"]+=1
            store["listsub"].append(readbase)

        if len(insertion)>0:
            store["numins"]+=1
            store["sumins"]+=len(insertion)
            store["listins"].append(insertion)
            store["hplen"]+=len(insertion)

################################
def outputReadStorage( mychr, ref, rs ):

    output = []
    correct = rs["hplen"]

    numcorrect = 0
    numtotal = 0

    for (kk,vv) in rs.items():
        #print "kk,vv", kk, vv
        if not("/" in kk or "-" in kk): continue # look only at read data

        if vv["visited"] != correct:
            # this read did not fully span the HP region so don't output it
            #print >>sys.stderr, kk, "did not span hp at", rs["refpos"],vv["visited"],"!=",correct
            continue
            
        numtotal += 1

        errhp = vv["hplen"] != correct
        errins = vv["numins"] > 0
        errdel = vv["numdel"] > 0
        errsub = vv["numsub"] > 0

        if errhp or errins or errsub or errdel:
            lineoutput=[]
            refpos = rs["refpos"]
            lineoutput.append(mychr)
            lineoutput.append(refpos)
            lineoutput.append(rs["hpbase"])
            lineoutput.append(rs["hplen"])

            leftctx = ref[(refpos-3):(refpos)]
            idx = (refpos+rs["hplen"])
            rightctx = ref[(idx):(idx+3)]
            lineoutput.append(leftctx)
            lineoutput.append(rightctx)

            lineoutput.append(kk)
            lineoutput.append(vv["hplen"])
            lineoutput.append(vv["numins"])
            lineoutput.append(vv["sumins"])
            lineoutput.append(vv["numdel"])
            lineoutput.append(vv["numsub"])
            lineoutput.append(",".join(vv["listins"]))
            lineoutput.append(",".join(vv["listsub"]))
            output.append("\t".join([ str(xx) for xx in lineoutput]))
        else:
            numcorrect += 1

    # add dummy num correct
    if True:
        lineoutput=[]
        refpos = rs["refpos"]
        lineoutput.append(mychr)
        lineoutput.append(refpos)
        lineoutput.append(rs["hpbase"])
        lineoutput.append(rs["hplen"])

        leftctx = ref[(refpos-3):(refpos)]
        idx = (refpos+rs["hplen"])
        rightctx = ref[(idx):(idx+3)]
        lineoutput.append(leftctx)
        lineoutput.append(rightctx)

        lineoutput.append("correct")
        lineoutput.append(numcorrect)
        lineoutput.append(numtotal)
        if numtotal>0:
            myerr=(1.0-float(numcorrect)/numtotal)
        else:
            myerr=0.0
        lineoutput.append(myerr)
        lineoutput.append(-1)
        lineoutput.append(-1)
        lineoutput.append("")
        lineoutput.append("")

        output.append("\t".join([ str(xx) for xx in lineoutput]))

    return(output)

################################
def printHeader():
    print "chr	refpos	truebase	truelen	leftctx	rightctx	zmw	hplen	numins	sumins	numdel	numsub	listins	listsub"

################################
def main():
    print >>sys.stderr, "RUNNING", datetime.datetime.now(), "\t".join(sys.argv)

    DEBUG=False

    fullref = readref(sys.argv[1])
    print >>sys.stderr, "len(fullrun)", len(fullref)

    chr = sys.argv[3]
    mybegin = int(sys.argv[4])
    myend = int(sys.argv[5])

    # NOTE: mybegin and myend should be single different bases that end the interior HP

    samfile = pysam.AlignmentFile(sys.argv[2], "rb")

    readstorage={"refpos":-1, "hpbase":"N", "hplen":0}

    ref=fullref[mybegin:myend]
    print >>sys.stderr, "hpify start"
    HPID = hpify(ref)
    print >>sys.stderr, "hpify end"
    #print "ref", ref
    #print "HPID", HPID

    printHeader()

    #HPID = {50149542:("T",19),50149561:("N",0) }

    count = 0
    for col in samfile.pileup(chr, mybegin, myend, trucate=True): # truncate: only return inside
        print >>sys.stderr, "count", count
        count += 1

        if col.pos<mybegin or col.pos>=myend: continue

        # no qualities when pbmm2 two fastas
        #col.set_min_base_quality(0) # !!!! THIS CAUSED HUGE PROBLEMS and simply skips bases below with default settings

        if DEBUG: print ("\ncoverage at base %s = %s" % (col.pos, col.n))

        #### Start of new HP?
        if (col.pos-mybegin) in HPID:
            if readstorage["refpos"]!= -1: # skips dummy first
                print "\n".join( outputReadStorage( chr, fullref, readstorage ))
            readstorage = {"refpos":col.pos, "hpbase":HPID[col.pos-mybegin][0], "hplen":HPID[col.pos-mybegin][1]}

        for read in col.pileups:

            ### collect the info for this base alignemt
            qs = read.alignment.query_sequence
            qp = read.query_position
            qn = read.alignment.query_name

            refbase = ref[col.pos-mybegin]

            if read.is_del:
                readbase = "-"
            elif read.is_refskip:
                readbase = "_"
            else:
                readbase = qs[qp]

            if read.indel>0:
                if qp is None:
                    # this happens with deletion followed by insertion
                    qp = read.query_position_or_next
                    hpend=qp
                    hpbegin=qp-read.indel
                else:
                    hpbegin=qp+1
                    hpend=qp+1+read.indel
                insertion = qs[hpbegin:hpend]
            else:
                insertion=""

            ### process the information
            if not qn in readstorage:
                readstorage[qn] = startdat()
            if DEBUG: print(col.pos, refbase, qn, readbase, read.indel, insertion)
            processBase(readstorage[qn], col.pos, refbase, qn, readbase, read.indel, insertion)

    samfile.close()

    print >>sys.stderr, "!!!DONE!!!", datetime.datetime.now(), "\t".join(sys.argv)

if __name__=="__main__":
    main()
