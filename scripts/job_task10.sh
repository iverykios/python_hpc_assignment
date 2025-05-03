#!/bin/sh

#BSUB -q c02613
#BSUB -J profiling_gpu

#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"

#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 00:30

#BSUB -o profiling_gpu_%J.out
#BSUB -e profiling_gpu_%J.err

#BSUB -B
#BSUB -N

export TMPDIR=$__LSF_JOB_TMPDIR__

lscpu 
nvidia-smi

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

nsys profile -o jacobi_prof python ../task_9.py 50
nsys stats jacobi_prof.nsys-rep >> prof_data.txt

nsys profile -o jacobi_prof_fix python ../task_10.py 50
nsys stats jacobi_prof_fix.nsys-rep >> prof_data_fix.txt