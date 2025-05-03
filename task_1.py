#!/usr/bin/env python3
"""
task1.py

Task 1: Familiarize yourself with the data.

This script:
    - Loads a few floorplan input files (domain and interior mask) from the path
      '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'.
    - Visualizes the temperature grid and interior mask.
    - Saves the plots in the folder 'HPC_project/visualizations/task1'.
"""

import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

def load_data(load_dir, bid):
    """
    Load the simulation grid and interior mask for a given building id.
    
    Parameters:
      load_dir (str): Directory where the floorplan data is stored.
      bid (str): The building ID.
      
    Returns:
      u (ndarray): A padded (514x514) simulation grid loaded from "{bid}_domain.npy".
      interior_mask (ndarray): A binary mask (512x512) loaded from "{bid}_interior.npy".
    """
    SIZE = 512
    # Create a padded array to hold the domain data.
    u = np.zeros((SIZE + 2, SIZE + 2))
    # Load the domain file and place it in the inner region of u.
    domain_file = join(load_dir, f"{bid}_domain.npy")
    u[1:-1, 1:-1] = np.load(domain_file)
    # Load the interior mask.
    interior_file = join(load_dir, f"{bid}_interior.npy")
    interior_mask = np.load(interior_file)
    return u, interior_mask

def main():
    # Define the input and output directories.
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings'
    OUTPUT_DIR = 'HPC_project/visualizations/task1'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load building IDs from file.
    building_ids_path = join(LOAD_DIR, 'building_ids.txt')
    with open(building_ids_path, 'r') as f:
        building_ids = f.read().splitlines()

    # For demonstration, we visualize the first 3 floorplans.
    num_buildings = 3
    selected_ids = building_ids[:num_buildings]

    for bid in selected_ids:
        u, interior_mask = load_data(LOAD_DIR, bid)
        
        # Visualize the temperature grid.
        plt.figure(figsize=(4, 4))
        # Exclude the padded border (show only u[1:-1,1:-1]).
        plt.imshow(u[1:-1, 1:-1], cmap="hot", origin='upper')
        plt.title(f"Temperature Grid for Building {bid}")
        plt.colorbar(label="Temperature [ºC]")
        temp_plot_file = join(OUTPUT_DIR, f"{bid}_temperature_grid.png")
        plt.savefig(temp_plot_file)
        plt.close()
        
        # Visualize the interior mask.
        plt.figure(figsize=(4, 4))
        plt.imshow(interior_mask, cmap="gray", origin='upper')
        plt.title(f"Interior Mask for Building {bid}")
        plt.colorbar(label="Interior Mask (1: interior, 0: wall/outside)")
        mask_plot_file = join(OUTPUT_DIR, f"{bid}_interior_mask.png")
        plt.savefig(mask_plot_file)
        plt.close()
        
        print(f"Visualizations for building {bid} saved.")

    print("Task 1 completed. Visualizations are stored in:", OUTPUT_DIR)

if __name__ == '__main__':
    main()
