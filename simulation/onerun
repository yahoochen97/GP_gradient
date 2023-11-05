#!/bin/bash
#BSUB -n 1
#BSUB -R "span[hosts=1]"

source /project/engineering/conda/pytorch-mpi-23.setup.sh
mkdir -p /export/cluster-tmp/chenyehu
export TMPDIR=/export/cluster-tmp/chenyehu

module add seas-lab-vorobeychik

SEED=$1
N=$2
T=$3
MODEL=$4

if [ $MODEL = "2FE" ]
then
    python 2FE.py -N ${N} -T ${T} -s ${SEED}
elif [ $MODEL = "2RE" ]
then
    python 2RE.py -N ${N} -T ${T} -s ${SEED}
elif [ $MODEL = "FULLBAYES" ]
then
    python gpr_fullbayes.py -N ${N} -T ${T} -s ${SEED}
elif [ $MODEL = "GPR" ]
then
    python gpr.py -N ${N} -T ${T} -s ${SEED}
fi