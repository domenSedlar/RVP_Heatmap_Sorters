import numpy as np
from RVP_Metrics.metrics import moore_stress4, me4
from OptimalSorters.tsp_solver import tsp_reorder_matrix_opt
from OptimalSorters.brute_force import brue_force
import random
import time

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

def test_sizes(): # it can solve 27x27 size matrices in under 2 mins conssitently
    for n in range(3, 100):
        A = np.random.rand(n,n)
        _, res1, res2, _, _, _ = tsp_reorder_matrix_opt(A, n, n, True)
        print(n)
        if res1.status == 1 or res2.status == 1:
            print("timeout on size", n)
            break

def test_size(size=34, n_times=50):
    timer_start = time.time()
    for n in range(n_times):
        A = np.random.rand(size,size)
        _, res1, res2, _, _, _ = tsp_reorder_matrix_opt(A, n, n, True)
        #print(n)
        if res1.status == 1 or res2.status == 1:
            #print("timeout on attempt", n)
            timer_end = time.time()
            print(f"size: {size} - {(timer_end - timer_start)/(n+1)}")
            return False
    #print("success")
    timer_end = time.time()
    print(f"size: {size} - {(timer_end - timer_start)/(n_times+1)}")
    return True

def find_breaking_point(start=34, n_times=25):
    res = False
    while not res:
        start -= 1
        res = test_size(start, n_times)

    print("didn't break on ", start)

if __name__ == "__main__":
    test()