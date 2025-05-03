#!/usr/bin/env python3
"""
task6.py

Combined script for Task 6:
- Mode 1: Parallel floorplan simulation with DYNAMIC scheduling
- Mode 2: Analysis of timing results and speedup calculations

Usage:
  For simulation:
    python task6.py simulate <N> <num_workers>
    
  For analysis:
    python task6.py analyze <results_dir>
"""

import os
import sys
import time
from os.path import join
import numpy as np
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Directory where the modified Swiss dwellings dataset is stored
LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

# Simulation parameters
MAX_ITER = 20_000
ABS_TOL = 1e-4
SIZE = 512  # Domain data is 512x512

#------------------------------------------------------------------------------
# Simulation functions
#------------------------------------------------------------------------------

def load_data(bid):
    """
    Load the simulation grid and interior mask for a given building ID.
    """
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(LOAD_DIR, f"{bid}_domain.npy"))
    interior_mask = np.load(join(LOAD_DIR, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    """
    Runs the Jacobi iteration until convergence or max_iter is reached.
    """
    u = np.copy(u)
    for i in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] +
                        u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u


def summary_stats(u, interior_mask):
    """
    Computes summary statistics for the interior of a floorplan.
    """
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }


def process_floorplan(bid):
    """
    Process a single floorplan: load data, run simulation, compute statistics.
    """
    try:
        u, interior_mask = load_data(bid)
        u_final = jacobi(u, interior_mask, MAX_ITER, ABS_TOL)
        stats = summary_stats(u_final, interior_mask)
        return (bid, stats)
    except Exception as e:
        print(f"Error processing floorplan {bid}: {e}")
        return (bid, {"error": str(e)})


def run_simulation(N, num_workers):
    """
    Run the parallel simulation with N floorplans and num_workers workers
    using DYNAMIC scheduling.
    """
    # Read building IDs from file
    with open(join(LOAD_DIR, "building_ids.txt"), 'r') as f:
        building_ids = f.read().splitlines()
    
    # Limit to first N floorplans
    building_ids = building_ids[:N]
    
    print(f"Processing {N} floorplans using {num_workers} workers with DYNAMIC scheduling...")
    
    # Record start time
    start_time = time.time()
    
    # Process floorplans in parallel using a Pool of workers with dynamic scheduling
    results = []
    with mp.Pool(processes=num_workers) as pool:
        # Use smaller chunks for dynamic scheduling (as mentioned in the slides)
        # chunksize=1 ensures maximum dynamism - each worker gets a new task as soon as it finishes
        for result in pool.imap_unordered(process_floorplan, building_ids, chunksize=1):
            results.append(result)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    
    # Print results in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print("building_id," + ",".join(stat_keys))
    for bid, stats in results:
        if "error" in stats:
            print(f"{bid},error: {stats['error']}")
        else:
            print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))
    
    return elapsed_time

#------------------------------------------------------------------------------
# Analysis functions
#------------------------------------------------------------------------------

def amdahl_law(n, p):
    """
    Amdahl's law: S(n) = 1 / ((1-p) + p/n)
    where:
        n is the number of processors
        p is the parallel fraction
        S(n) is the speedup with n processors
    """
    return 1 / ((1 - p) + p / n)


def analyze_results(results_dir):
    """
    Analyze timing results from multiple runs and calculate speedups.
    """
    # Load timing data from all repetitions
    dfs = []
    for file in os.listdir(results_dir):
        if file.startswith("timing_rep") and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(results_dir, file))
            dfs.append(df)
    
    if not dfs:
        print("No timing data found!")
        return
    
    # Combine all dataframes and compute mean time for each worker count
    all_times = pd.concat(dfs)
    mean_times = all_times.groupby('workers')['time'].mean().reset_index()
    
    # Calculate speedups relative to single worker
    base_time = mean_times.loc[mean_times['workers'] == 1, 'time'].values[0]
    mean_times['speedup'] = base_time / mean_times['time']
    
    # Fit to Amdahl's law to estimate parallel fraction
    workers = mean_times['workers'].values
    speedups = mean_times['speedup'].values
    
    # Initial guess for parallel fraction
    p0 = [0.95]
    
    # Fit the curve
    params, _ = curve_fit(amdahl_law, workers, speedups, p0=p0)
    parallel_fraction = params[0]
    
    # Calculate theoretical maximum speedup
    max_speedup = 1 / (1 - parallel_fraction)
    
    # Create a table with results
    results_table = pd.DataFrame({
        'Workers': workers,
        'Mean Time (s)': mean_times['time'].values,
        'Speedup': speedups
    })
    
    # Save results to file
    results_table.to_csv(os.path.join(results_dir, 'speedup_results.csv'), index=False)
    
    # Plot the speedup curve
    plt.figure(figsize=(10, 6))
    
    # Plot measured speedups
    plt.plot(workers, speedups, 'o-', label='Measured Speedup')
    
    # Plot Amdahl's law prediction
    x = np.linspace(1, max(workers)*1.5, 100)
    plt.plot(x, amdahl_law(x, parallel_fraction), '--', 
             label=f"Amdahl's Law (p={parallel_fraction:.4f})")
    
    # Plot theoretical maximum speedup
    plt.axhline(y=max_speedup, color='r', linestyle='--', 
                label=f'Theoretical Maximum ({max_speedup:.2f})')
    
    plt.xlabel('Number of Workers')
    plt.ylabel('Speedup')
    plt.title('Speedup vs. Number of Workers (Dynamic Scheduling)')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(os.path.join(results_dir, 'speedup_plot.png'))
    
    # Estimate time for all floorplans
    total_floorplans = 4571
    test_floorplans = 100
    fastest_time = mean_times['time'].min()
    fastest_workers = mean_times.loc[mean_times['time'] == fastest_time, 'workers'].values[0]
    estimated_time_all = (total_floorplans / test_floorplans) * fastest_time
    
    # Write summary to file
    with open(os.path.join(results_dir, 'summary.txt'), 'w') as f:
        f.write("Task 6 Results Summary (Dynamic Scheduling)\n")
        f.write("==========================================\n\n")
        
        f.write("a) Speedup measurements:\n")
        f.write(results_table.to_string(index=False))
        f.write("\n\n")
        
        f.write("b) Parallel fraction (Amdahl's Law estimate):\n")
        f.write(f"   p = {parallel_fraction:.4f}\n\n")
        
        f.write("c) Theoretical maximum speedup:\n")
        f.write(f"   S(∞) = 1/(1-p) = {max_speedup:.2f}\n")
        f.write(f"   Best achieved speedup: {max(speedups):.2f} with {fastest_workers} workers\n\n")
        
        f.write("d) Estimated time for all floorplans:\n")
        f.write(f"   Time for {test_floorplans} floorplans with {fastest_workers} workers: {fastest_time:.2f} seconds\n")
        f.write(f"   Estimated time for all {total_floorplans} floorplans: {estimated_time_all:.2f} seconds")
        f.write(f"   ({estimated_time_all/3600:.2f} hours)\n")
    
    # Print summary to console
    print("\nTask 6 Analysis Results (Dynamic Scheduling):")
    print("============================================")
    print(f"a) Speedup measurements saved to {results_dir}/speedup_results.csv")
    print(f"b) Parallel fraction (Amdahl's Law): p = {parallel_fraction:.4f}")
    print(f"c) Theoretical maximum speedup: {max_speedup:.2f}")
    print(f"   Best achieved: {max(speedups):.2f} with {fastest_workers} workers")
    print(f"d) Estimated time for all {total_floorplans} floorplans: {estimated_time_all:.2f} seconds ({estimated_time_all/3600:.2f} hours)")
    print(f"\nDetailed summary saved to {results_dir}/summary.txt")
    print(f"Speedup plot saved to {results_dir}/speedup_plot.png")


#------------------------------------------------------------------------------
# Main function
#------------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  For simulation: python task6.py simulate <N> <num_workers>")
        print("  For analysis:   python task6.py analyze <results_dir>")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "simulate":
        if len(sys.argv) < 4:
            print("Usage for simulation: python task6.py simulate <N> <num_workers>")
            sys.exit(1)
        
        try:
            N = int(sys.argv[2])
            num_workers = int(sys.argv[3])
        except ValueError:
            print("Error: Both <N> and <num_workers> must be integers.")
            sys.exit(1)
        
        run_simulation(N, num_workers)
    
    elif mode == "analyze":
        if len(sys.argv) < 3:
            print("Usage for analysis: python task6.py analyze <results_dir>")
            sys.exit(1)
        
        results_dir = sys.argv[2]
        analyze_results(results_dir)
    
    else:
        print("Unknown mode. Use 'simulate' or 'analyze'.")
        print("Usage:")
        print("  For simulation: python task6.py simulate <N> <num_workers>")
        print("  For analysis:   python task6.py analyze <results_dir>")
        sys.exit(1)


if __name__ == "__main__":
    main()