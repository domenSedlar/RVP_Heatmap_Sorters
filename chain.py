from Heuristics.hclust import hclust_with_olo
from Heuristics.bae import bae
from Heuristics.rand_swap import rand_swaps, rand_block_swaps, randomly_mirror
from Heuristics.tsp_heur import tsp_lk

from OptimalSorters.tsp_solver import tsp_reorder_matrix_opt
from RVP_Metrics.metrics import Metric

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

def chain(ls, H):
    for f in ls:
        H = f(H)

    return H


class Chain:
    def __init__(self, algos, metric=Metric.NS):
        self.metric = metric
        self.algos = algos

    def get_name(self):
        return '->'.join([a.get_name() for a in self.algos])
    
    def __call__(self, H, metric=Metric.NS, *args, **kwds):
        return chain(self.algos, H)