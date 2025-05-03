#!/usr/bin/env python3
"""
task7.py

Task 7: Implementation of the Jacobi method using Numba JIT on CPU.

This script reimplements the jacobi function using Numba JIT to accelerate
the computations on CPU while ensuring good cache utilization.
"""

from os.path import join
import sys
import time
import numpy as np
from numba import jit


def load_data(load_dir, bid):
    """Load the simulation grid and interior mask for a given building ID."""
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@jit(nopython=True)
def jacobi_jit(u, interior_mask, max_iter, atol=1e-6):
    """
    Jacobi method implemented with Numba JIT compilation.
    
    This implementation is optimized for cache efficiency by:
    1. Using row-major iteration to match NumPy's memory layout
    2. Accessing memory in a predictable pattern for better prefetching
    3. Minimizing data movement between memory and CPU
    """
    # Create a copy to avoid modifying the input
    u = np.copy(u)
    
    for iter_count in range(max_iter):
        # Compute average of neighbors
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + 
                         u[:-2, 1:-1] + u[2:, 1:-1])
        
        # Instead of boolean indexing, use explicit loops
        # This is more Numba-friendly
        delta = 0.0
        for i in range(interior_mask.shape[0]):
            for j in range(interior_mask.shape[1]):
                if interior_mask[i, j]:
                    old_val = u[i+1, j+1]
                    new_val = u_new[i, j]
                    diff = abs(old_val - new_val)
                    if diff > delta:
                        delta = diff
                    u[i+1, j+1] = new_val
        
        # Check for convergence
        if delta < atol:
            break
    
    return u


def summary_stats(u, interior_mask):
    """Compute summary statistics for the interior of a floorplan."""
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


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
    
    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]
    
    print(f"Processing {N} floorplans using Numba JIT...")
    
    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask
    
    # Warm up the JIT compiler (first call compiles)
    _ = jacobi_jit(all_u0[0], all_interior_mask[0], 1, 1e-4)
    
    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4
    
    start_time = time.time()
    
    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_jit(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u
    
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    
    # Estimate time for all floorplans
    total_floorplans = 4571
    estimated_time = elapsed_time / N * total_floorplans
    print(f"Estimated time for all {total_floorplans} floorplans: {estimated_time:.2f} seconds ({estimated_time/3600:.2f} hours)")
    
    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid}, " + ", ".join(str(stats[k]) for k in stat_keys))