import elkai
from MyUtils.distance_matrix import create_matrices_ns, create_matrices_me
from RVP_Metrics.metrics import Metric
from MyUtils.helpers import extract_tour_order
import numpy as np

def tsp_lk(H, metric : Metric = Metric.NS, runs=10):
    if metric == Metric.NS:
        c_rows, c_cols = create_matrices_ns(H)
    elif metric == Metric.ME4:
        c_rows, c_cols = create_matrices_me(H)
    else:
        return None

    (n,m) = H.shape

    c_rows = elkai.DistanceMatrix(c_rows)
    c_cols = elkai.DistanceMatrix(c_cols)

    rows = c_rows.solve_tsp()
    cols = c_cols.solve_tsp()

    rows = extract_tour_order(rows, n)
    cols = extract_tour_order(cols, m)
    print(rows, cols)

    return rows, cols