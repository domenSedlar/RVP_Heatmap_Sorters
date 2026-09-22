import elkai
from MyUtils.distance_matrix import create_matrices_ns, create_matrices_me
from RVP_Metrics.metrics import Metric
import numpy as np


def rm_dummy(tour):
    if tour is None:
        return None
    dummy = max(tour)
    order = []
    for i, v in enumerate(tour):
        if v == dummy:
            order += tour[(i+1):]
            order += tour[:i]

    return order


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

    rows = c_rows.solve_tsp()[1:]
    cols = c_cols.solve_tsp()[1:]
    rows = rm_dummy(rows)
    cols = rm_dummy(cols)

    return H[rows][:, cols]

class TSP_LK:
    def __init__(self, metric=Metric.NS, runs = 10):
        self.runs = runs
        self.metric = metric

    def get_name(self):
        return f'TSP_LK_runs={self.runs}'

    def __call__(self, H, metric=None, *args, **kwds):
        if metric is not None:
            self.metric = metric
        return tsp_lk(H, metric=self.metric, runs=self.runs), False