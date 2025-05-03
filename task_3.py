#!/usr/bin/env python3
"""
task3.py

Task 3: Visualize the simulation results for a few floorplans.

This script performs the following steps:
  1. Loads a subset of floorplan files from the provided data directory.
  2. Runs the Jacobi solver to simulate the steady-state heat distribution.
  3. Plots the final temperature distribution (simulation result) and saves the images
     to the folder HPC_project/visualizations/task3.
"""

import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

def load_data(load_dir, bid):
    """
    Load the temperature grid (domain) and interior mask for a given building ID.
    
    Parameters:
      load_dir (str): Directory where the floorplan data is stored.
      bid (str): Building ID.
      
    Returns:
      u (ndarray): A padded (514×514) array with the domain data.
      interior_mask (ndarray): A binary mask (512×512) indicating interior points.
    """
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    # Load the domain file and insert it into the padded region of u.
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi(u, interior_mask, max_iter, atol=1e-6):
    """
    Perform the Jacobi iterations to compute the steady-state heat distribution.
    
    For each interior grid point in u (as indicated by the interior_mask), its new value is
    computed as the average of the four neighboring grid points.
    
    Parameters:
      u (ndarray): The padded simulation grid.
      interior_mask (ndarray): Boolean mask (512x512) for points to be updated.
      max_iter (int): Maximum number of iterations.
      atol (float): Tolerance for convergence.
      
    Returns:
      u (ndarray): Updated padded simulation grid after convergence or max_iter iterations.
    """
    u = np.copy(u)
    for i in range(max_iter):
        # Compute the average of the four neighbors (using slicing on the padded grid)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        # Determine the maximum difference for convergence checking
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        # Update only the interior points
        u[1:-1, 1:-1][interior_mask] = u_new_interior
        if delta < atol:
            break
    return u

def main():
    # Define the input data directory and output folder for visualizations.
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings'
    OUTPUT_DIR = 'HPC_project/visualizations/task3'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load building IDs from file.
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    # For demonstration, we select the first 3 building floorplans.
    num_buildings = 3
    selected_ids = building_ids[:num_buildings]

    # Set parameters for the simulation (Jacobi) solver.
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    # Process each selected building:
    for bid in selected_ids:
        # Load the floorplan data.
        u, interior_mask = load_data(LOAD_DIR, bid)
        # Run the Jacobi simulation.
        u_sim = jacobi(u, interior_mask, MAX_ITER, ABS_TOL)

        # Visualize the simulation result (display only the inner 512x512 part).
        plt.figure(figsize=(6, 6))
        plt.imshow(u_sim[1:-1, 1:-1], cmap="hot", origin="upper")
        plt.title(f"Simulation Result for Building {bid}")
        plt.colorbar(label="Temperature [ºC]")

        # Save the plot.
        result_file = join(OUTPUT_DIR, f"{bid}_simulation_result.png")
        plt.savefig(result_file)
        plt.close()
        print(f"Saved simulation result for building {bid} to {result_file}")

    print("Task 3 completed. Check the simulation result images in:", OUTPUT_DIR)

if __name__ == '__main__':
    main()
