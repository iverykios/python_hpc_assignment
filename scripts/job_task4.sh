#!/bin/bash
#BSUB -J models
#BSUB -q hpc
#BSUB -R "rusage[mem=16GB]"
#BSUB -o outputv_task_4.out  
#BSUB -e errorv_task_4.err  
#BSUB -W 24:00    
#BSUB -n 16
#BSUB -R "span[hosts=1]"

# Initialize and activate environment
source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

module swap cuda/12.3.2
module swap cudnn/v9.3.0.75-prod-cuda-12.X

export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_ROOT"
kernprof -l -v HPC_project/task4.py 1