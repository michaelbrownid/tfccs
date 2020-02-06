"""combineMakecallsAlignerror.py

After a trained model:

- combine the network posteriors

- make call using product

- construct sequences for calls

- compute alignment error
"""

import sys
import subprocess
import argparse

def combineMakecallsAlignerror( outputprefix, datanpzfile, modelname, cudavisdev ):

    ################################
    # dump the data for the subreads. pull truth to be consistent

    #### truth
    cmd = """
    export PYTHONPATH=/home/UNIXHOME/mbrown/mbrown/workspace2019Q2/ccs-ml-model/; \
    python3 -m tfccs.test_combine_subreads_perread_birnn /home/UNIXHOME/mbrown/mbrown/workspace2019Q2/ccs-ml-model/zip1Embed.class \
    <(echo '%s') \
    EXEC:args.inputdatName=\\'windowinputPlusCallProp\\' \
    EXEC:args.outputdatName=\\'windowoutputDirectHPPlusCallTrue\\' \
    EXEC:args.model=\\'model_perread_birnn\\' \
    EXEC:args.modelsave=\\'%s\\' \
    EXEC:args.CUDA_VISIBLE_DEVICES=\\'%s\\' \
    EXEC:args.batch_size=1000 \
    EXEC:args.batch_number=-1 \
    EXEC:args.saveFile=\\'%s.truth.perread.birnn.npz\\'
    """ % (datanpzfile, modelname, cudavisdev, outputprefix)
    print("cmd", cmd)
    subprocess.call(cmd, shell=True, executable="/bin/bash")

    #### dump the predictions
    cmd = """
    export PYTHONPATH=/home/UNIXHOME/mbrown/mbrown/workspace2019Q2/ccs-ml-model/; \
    python3 -m tfccs.test_combine_subreads_perread_birnn /home/UNIXHOME/mbrown/mbrown/workspace2019Q2/ccs-ml-model/zip1Embed.class \
    <(echo '%s') \
    EXEC:args.inputdatName=\\'windowinputPlusCallProp\\' \
    EXEC:args.outputdatName=\\'windowoutputDirectHPPlusCallTrue\\' \
    EXEC:args.model=\\'model_perread_birnn\\' \
    EXEC:args.modelsave=\\'%s\\' \
    EXEC:args.CUDA_VISIBLE_DEVICES=\\'%s\\' \
    EXEC:args.batch_size=1000 \
    EXEC:args.batch_number=0 \
    EXEC:args.saveFile=\\'%s.subreads.perread.birnn.npz\\'
    """ % ( datanpzfile, modelname, cudavisdev, outputprefix)
    print("cmd", cmd)
    subprocess.call(cmd, shell=True, executable="/bin/bash")

    # construct sequence calls by taking product

    cmd = """
    export PYTHONPATH=/home/UNIXHOME/mbrown/mbrown/workspace2019Q2/ccs-ml-model/; \
    python3 -m tfccs.tools.makeCallProd133Truth %s.truth.perread.birnn.npz > %s.callprod.truth.perread.birnn.fasta
    """ % (outputprefix, outputprefix)
    print("cmd", cmd)
    subprocess.call(cmd, shell=True, executable="/bin/bash")

    cmd = """
    export PYTHONPATH=/home/UNIXHOME/mbrown/mbrown/workspace2019Q2/ccs-ml-model/; \
    python3 -m tfccs.tools.makeCallProd133 %s.subreads.perread.birnn.npz %s.callprod.subreads.perread.birnn
    """ % (outputprefix, outputprefix)
    print("cmd", cmd)
    subprocess.call(cmd, shell=True, executable="/bin/bash")

    # compute alignment error

    #
    cmd = """
    # /bin/bash does not get module system 
    source /mnt/software/Modules/current/init/bash
    module add smrtanalysis/mainline

    # which pbmm2; /pbi/dept/secondary/builds/links/current_develop_smrttools-cleanbuild_installdir/private/otherbins/all/bin/pbmm2
    pbmm2 align \
    %s.callprod.truth.perread.birnn.fasta \
    %s.callprod.subreads.perread.birnn.fasta \
    %s.callprod.truth.pred.align.bam --sort -u

    #TODO: python3 -m does not work here for some reason... so can't use -m tfccs.
    export PYTHONPATH=/home/UNIXHOME/mbrown/mbrown/workspace2019Q2/ccs-ml-model/; \
    samtools view %s.callprod.truth.pred.align.bam | python3 $PYTHONPATH/tfccs/tools/cigarError.py > %s.prod.true2pred.align.err
    """ % (outputprefix, outputprefix, outputprefix, outputprefix, outputprefix)
    print("cmd", cmd)
    subprocess.call(cmd, shell=True, executable="/bin/bash")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputprefix',help="output prefix for outputfiles, outputmodel1")
    parser.add_argument('--datanpzfile',help="npz data file name of data vectors to operate on, data.001.npz")
    parser.add_argument('--modelname',help="npz data file name of data vectors to operate on, model.perread.birnn_1.h5.best")
    parser.add_argument('--cudavisdev',help="gpu number to run on, 0")
    args=parser.parse_args()

    combineMakecallsAlignerror( args.outputprefix, args.datanpzfile, args.modelname, args.cudavisdev )
