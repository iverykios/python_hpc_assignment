from os.path import join
import sys
import time
import numpy as np
from numba import cuda

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):
    i, j = cuda.grid(2)
    
    if 1 <= i < u.shape[0] - 1 and 1 <= j < u.shape[1] - 1:
        if interior_mask[i - 1, j - 1]: 
            u_new[i, j] = 0.25 * (
                u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1]
            )

def helper_function(u, interior_mask, max_iter):
    u_d = cuda.to_device(u)
    u_new_d = cuda.to_device(u.copy())
    mask_d = cuda.to_device(interior_mask)

    threads_per_block = (32, 32)
    blocks_per_grid_x = (u.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (u.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    for _ in range(max_iter):
        jacobi_kernel[blocks_per_grid, threads_per_block](u_d, u_new_d, mask_d)
        u_d, u_new_d = u_new_d, u_d 

    return u_d.copy_to_host()


if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000

    start = time.time()
    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = helper_function(u0, interior_mask, MAX_ITER)
        all_u[i] = u
        cuda.synchronize()

    elapsed_time_ms = time.time() - start

    print(f"GPU Time for {N} floorplans: {elapsed_time_ms:.2f} ms")
    print(f"Avg time per floorplan: {elapsed_time_ms/N:.2f} ms")

    TOTAL_FLOORPLANS = 4571

    estimated_total_ms = (elapsed_time_ms / N) * TOTAL_FLOORPLANS
    estimated_total_sec = estimated_total_ms / 1000

    print(f"\nEstimated total time for processing {TOTAL_FLOORPLANS} floorplans: {estimated_total_sec:.2f} seconds")