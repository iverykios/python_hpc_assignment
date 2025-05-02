#!/bin/sh

#BSUB -q c02613
#BSUB -J all_bids

#BSUB -n 4
#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=2GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30

#BSUB -o all_bids_%J.out
#BSUB -e all_bids_%J.err

#BSUB -B
#BSUB -N

nvidia-smi

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

TOTAL=4571
CHUNK=1500

for START in $(seq 0 $CHUNK $((TOTAL - 1))); do
    echo "Processing buildings $START to $((START + CHUNK - 1))"
    python ../cuda_simulation.py $CHUNK $START >> all_results.csv
done
