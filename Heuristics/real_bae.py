import numpy as np

from MyUtils.distance_matrix import create_matrices_ns, create_matrices_me
from MyUtils.helpers import extract_tour_order
from RVP_Metrics.metrics import Metric

def bae(A):
    (n,m) = A.shape

    leftmost = 0
    rightmost = 0

    tour = [0]

    m = np.max(A) + 1

    A[:, 0] = m

    while len(tour) < n:
        cand_left = np.argmin(A[leftmost])
        cand_right = np.argmin(A[rightmost])

        # Pick the overall minimum between the two
        if A[leftmost, cand_left] <= A[rightmost, cand_right]:
            tour = [cand_left] + tour
            leftmost = cand_left
            A[:, cand_left] = m

        else:
            tour.append(cand_right)
            rightmost = cand_right
            A[:, cand_right] = m
        
    return tour

class real_BAE:
    def get_name(self):
        return 'realBAE'
    
    def __call__(self, H, metric=Metric.NS, *args, **kwds):
        if metric == Metric.NS:
            cr, cc = create_matrices_ns(H, padding=False)
        elif metric == Metric.ME4:
            raise NotImplementedError
        (rows, cols) = (bae(cr), bae(cc))

        return H[rows][:, cols]