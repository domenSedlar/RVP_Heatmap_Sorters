import numpy as np

from MyUtils.distance_matrix import create_matrices_ns, create_matrices_me
from MyUtils.helpers import extract_tour_order
from RVP_Metrics.metrics import Metric

def bea(A):
    (n, _) = A.shape
    n = n - 1 # so n is the dummy city now
    max_val = np.max(A) + 1
    tour_linked_list_dictionary = np.ones((n+1, 2), dtype=np.int64)*n
    A[:, 0] = max_val

    placed_distances = np.ones((n+1, 2, n+1))*max_val
    placed_distances[0,0,:] = A[0] + A[n]
    placed_distances[0,1,:] = A[0] + A[n]
    i = 1
    while i < n:
        i += 1
        flat_idx = np.argmin(placed_distances[:, :, :n])
        (root, direction_of_neighbor, new_v) = np.unravel_index(flat_idx, placed_distances[:n, :, :n].shape)        
        direction_of_root = 0 if direction_of_neighbor == 1 else 1
        neighbor = tour_linked_list_dictionary[root][direction_of_neighbor]

        A[:, new_v] = max_val
        placed_distances[:, :, new_v] = max_val*2

        tour_linked_list_dictionary[new_v] = [root, neighbor] if direction_of_root == 0 else [neighbor, root]
        tour_linked_list_dictionary[neighbor][direction_of_root] = new_v
        tour_linked_list_dictionary[root][direction_of_neighbor] = new_v

        placed_distances[new_v, direction_of_root, :] = A[new_v] + A[root]
        placed_distances[new_v, direction_of_neighbor, :] = A[new_v] + A[neighbor]

        placed_distances[root, direction_of_neighbor, :] = A[root] + A[new_v]
        placed_distances[neighbor, direction_of_root, :] = A[neighbor] + A[new_v]
        
    tour_left = []
    tour_right = [0]
    v = tour_linked_list_dictionary[0][1]
    while v != 0 and v != n:
        tour_right.append(v)
        v = tour_linked_list_dictionary[v][1]

    v = tour_linked_list_dictionary[0][0]
    while v != 0 and v != n:
        tour_left.append(v)
        v = tour_linked_list_dictionary[v][0]

    list.reverse(tour_left)
    tour = tour_left + tour_right
    #print(tour_linked_list_dictionary)
    return tour

class real_BAE:
    def get_name(self):
        return 'BEA'
    
    def __call__(self, H, metric=Metric.NS, *args, **kwds):
        if metric == Metric.NS:
            cr, cc = create_matrices_ns(H, padding=True)
        elif metric == Metric.ME4:
            raise NotImplementedError
        (rows, cols) = (bea(cr), bea(cc))

        return H[rows][:, cols], False