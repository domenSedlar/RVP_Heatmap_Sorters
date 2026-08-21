from Heuristics.hclust import hclust_with_olo
from Heuristics.bae import bae
from Heuristics.rand_swap import rand_swaps, rand_block_swaps, randomly_mirror
from Heuristics.tsp_heur import tsp_lk

from OptimalSorters.tsp_solver import tsp_reorder_matrix_opt

def hclust_wrapper(H, metric):
    return hclust_with_olo(H, metric=metric, method='average')

def rand_swaps_wrapper(H, metric):
    return rand_swaps(H, metric=metric, tries=30)
def rand_blocks_wrapper(H, metric):
    return rand_block_swaps(H, metric=metric, tries=30)
def rand_mirror_wrapper(H, metric):
    return randomly_mirror(H, metric=metric, tries=30)

def tsp_lk_wrapper(H, metric):
    return tsp_lk(H, metric=metric, runs=10)

def tsp_opt_wrapper(H, metric):
    _, row_order, col_order = tsp_reorder_matrix_opt(H, 0, 0, metric=metric, time_limit_seconds=60)
    return row_order, col_order

def chain(ls, H, metric):
    for f in ls:
        rows, cols = f(H, metric)
        H = H[rows][:, cols]

    return H