#!/bin/bash
#BSUB -J wall_heating_task5
#BSUB -q hpc
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "select[model==XeonGold6226R]"  
#BSUB -o output_task5_%J.out  
#BSUB -e error_task5_%J.err  
#BSUB -W 12:00    
#BSUB -n 16
#BSUB -R "span[hosts=1]"

# Initialize and activate environment
source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

# Directory for results
RESULTS_DIR="HPC_project/results/task5"
mkdir -p $RESULTS_DIR

# Number of floorplans to process
N=100

# Different worker counts to test
WORKER_COUNTS=(1 2 4 8 16)

# Number of repetitions for each configuration
REPETITIONS=3

# Function to run experiment with given worker count
run_experiment() {
    local workers=$1
    local rep=$2
    
    echo "Running with $workers workers (repetition $rep)..."
    
    # Run the simulation and capture the output
    output=$(python HPC_project/task5.py simulate $N $workers)
    
    # Extract the execution time
    time_taken=$(echo "$output" | grep "Total processing time" | awk '{print $4}')
    
    # Save the time to a file
    echo "$workers,$time_taken" >> $RESULTS_DIR/timing_rep${rep}.csv
    
    echo "Done. Time taken: $time_taken seconds"
}

# Create header for timing files
for rep in $(seq 1 $REPETITIONS); do
    echo "workers,time" > $RESULTS_DIR/timing_rep${rep}.csv
done

# Run experiments for each worker count and repetition
for workers in "${WORKER_COUNTS[@]}"; do
    for rep in $(seq 1 $REPETITIONS); do
        run_experiment $workers $rep
    done
done

# Run the analysis part of the script
python HPC_project/task5.py analyze $RESULTS_DIR