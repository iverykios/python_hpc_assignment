#!/bin/sh

#BSUB -q c02613
#BSUB -J cupy

#BSUB -n 4
#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30

#BSUB -o cupy_%J.out
#BSUB -e cupy_%J.err

#BSUB -B
#BSUB -N


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

time python task_9.py 10