import numpy as np
from RVP_Metrics.metrics import moore_stress4, me4
from OptimalSorters.tsp_solver import tsp_reorder_matrix_opt
from OptimalSorters.brute_force import brue_force
import random

def test(n_tests=25):
    sizes = [3,4,5]

    for i in range(n_tests):
        n = random.choice(sizes)
        m = random.choice(sizes)
#        A = np.random.rand(n,m)
        A = np.array([[random.randint(0,1) for _ in range(m)] for _ in range(n)], dtype=np.float64)

        C, true_score, _, _ = brue_force(A, moore_stress4, bigger_better=False)
        B, _, _, _ = tsp_reorder_matrix_opt(A, n, m)
        score = moore_stress4(B)

        if score != true_score:
            print(B)
            print()
            print(C)
            print(score, true_score)
        else:
            print("yay")


if __name__ == "__main__":
    test()