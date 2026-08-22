from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import pdist, squareform

from MyUtils.distance_matrix import create_matrices_ns, create_matrices_me


def _hclust_with_olo(D, method='average'):
    deandogram = linkage(D, method=method)

    ordered = optimal_leaf_ordering(deandogram, D)
    leaves = leaves_list(ordered)

    return leaves

def hclust_euclidian(A, method = 'average'):
    dist_condensed = pdist(A, metric='euclidean')

    dist_matrix = squareform(dist_condensed)
    return _hclust_with_olo(dist_condensed, method=method,)

def hclust_with_olo(H, metric='NS', method='average'):
    """
        H ~ heatmap
        metric : string - 'NS', 'ME', 'EC'
    """
    if metric == 'EC':
        return hclust_euclidian(H), hclust_euclidian(H.T) #TODO test!
    elif metric == 'NS':
        c_row, c_col = create_matrices_ns(H, padding=False)
    else:
        c_row, c_col = create_matrices_me(H, padding=False)

    rows = _hclust_with_olo(c_row, method=method)
    cols = _hclust_with_olo(c_col, method=method)

    return H[rows][:, cols]