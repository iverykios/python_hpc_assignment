#!/bin/sh

#BSUB -q c02613
#BSUB -J cuda

#BSUB -n 4
#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30

#BSUB -o cuda_%J.out
#BSUB -e cuda_%J.err

#BSUB -B
#BSUB -N

nvidia-smi

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python ../task_8.py 100 >> results_task8.dat