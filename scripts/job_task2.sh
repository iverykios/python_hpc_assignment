#!/bin/sh

#BSUB -q c02613
#BSUB -J ref_mes

#BSUB -n 1
#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=2GB]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -W 00:30

#BSUB -o ref_mes_%J.out
#BSUB -e ref_mes_%J.err

#BSUB -B
#BSUB -N

lscpu -C

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python ../task_2.py 10 >> results_task2.dat