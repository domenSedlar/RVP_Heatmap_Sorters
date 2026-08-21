import numpy as np

from MyUtils.distance_matrix import create_matrices_ns, create_matrices_me
from MyUtils.helpers import extract_tour_order
from RVP_Metrics.metrics import Metric

def greedy_tsp(A):
    (n, m) = A.shape
    tour = [0]
    v = 0

    m = np.max(A) + 1

    while len(tour) < n:
        A[:, v] = m # we cant select the same town twice
        next_city = np.argmin(A[v]) 
        tour.append(next_city)
        v = next_city

    return tour

def _bae(A_rows, A_cols):
    rows = greedy_tsp(A_rows)
    cols = greedy_tsp(A_cols)

    return rows, cols

def bae_ns(A):
    (n,m) = A.shape
    cr, cc = create_matrices_ns(A)
    cr = cr[:n, :n]
    cc = cc[:m, :m]

    rows, cols = _bae(cr, cc)
    return rows, cols

def bae_me(A):
    (n,m) = A.shape
    cr, cc = create_matrices_me(A)
    cr = cr[:n, :n]
    cc = cc[:m, :m]

    rows, cols = _bae(cr, cc)
    return rows, cols
    
def bae(H, metric=Metric.NS):
    (n,m) = H.shape
    if metric == Metric.NS:
        rows, cols =  bae_ns(H)
    elif metric == Metric.ME4:
        rows, cols = bae_me(H)

    return rows, cols
