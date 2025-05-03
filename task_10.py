from os.path import join
import sys
import time

import cupy as cp

def load_data(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = cp.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = cp.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi_batch(u, interior_mask, max_iter, atol=1e-6):
    u = cp.copy(u)
    
    for _ in range(max_iter):
        
        u_new = 0.25 * (
            u[:, 1:-1, :-2] +  
            u[:, 1:-1, 2:] +   
            u[:, :-2, 1:-1] +  
            u[:, 2:, 1:-1]     
        )
        delta = cp.abs(u[:, 1:-1, 1:-1] - u_new)
        delta_masked = cp.where(interior_mask, delta, 0.0)
        max_delta = delta_masked.max()
        cp.copyto(u[:, 1:-1, 1:-1], u_new, where=interior_mask)

        if max_delta < atol:
            break

    return u

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
    
    all_u0 = cp.empty((N, 514, 514))
    all_interior_mask = cp.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = jacobi_batch(all_u0, all_interior_mask, MAX_ITER, ABS_TOL)