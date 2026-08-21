import numpy as np
from RVP_Metrics.metrics import moore_stress4, me4
from OptimalSorters.tsp_solver import tsp_reorder_matrix_opt
from OptimalSorters.brute_force import brue_force
import random
import time

from Heuristics.hclust import hclust_with_olo
from Heuristics.bae import bae
from Heuristics.rand_swap import rand_swaps, rand_block_swaps, randomly_mirror
from Heuristics.tsp_heur import tsp_lk

def test(n_tests=25):
    sizes = [3,4,5]

    for i in range(n_tests):
        n = random.choice(sizes)
        m = random.choice(sizes)
#        A = np.random.rand(n,m)
        A = np.array([[random.randint(0,1) for _ in range(m)] for _ in range(n)], dtype=np.float64)

        B, _, _ = tsp_reorder_matrix_opt(A, n, m)
        true_score = moore_stress4(B)
        rows, cols = tsp_lk(A)
        score = moore_stress4(A[rows][:, cols])
        print(true_score, score)

if __name__ == "__main__":
    test()