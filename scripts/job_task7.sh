#!/bin/bash
#BSUB -J numba_jit_task7
#BSUB -q hpc
#BSUB -R "rusage[mem=16GB]"
#BSUB -o output_task7_%J.out  
#BSUB -e error_task7_%J.err  
#BSUB -W 1:00    
#BSUB -n 1
#BSUB -R "span[hosts=1]"

# Initialize and activate environment
source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

# Number of floorplans to test with
N=10

# Run the Numba JIT implementation
echo "Running Numba JIT implementation on $N floorplans..."
time python HPC_project/task7.py $N

echo ""
echo "For comparison, running reference implementation (Task 2) on $N floorplans..."
time python HPC_project/task2.py $N

echo ""
echo "Speedup calculation:"
echo "Check the output above to calculate the speedup ratio between Task 7 and Task 2"