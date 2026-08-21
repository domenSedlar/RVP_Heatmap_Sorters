import numpy as np

def create_matrices_me(heatmap, padding=True):
    padd = 0
    if padding:
        padd = 1
    A = heatmap
    (n,m) = A.shape

    c_row = np.zeros(shape=(n+padd,n+padd), dtype=np.float64)
    c_row[:n,:n] = - 2 * A @ A.T

    c_col = np.zeros(shape=(m+padd,m+padd), dtype=np.float64)
    c_col[:m,:m] = - 2*A.T@A

    return c_row, c_col

def create_matrices_ns(heatmap, padding=True):
    padd = 0
    if padding:
        padd = 1
    A = heatmap
    (n,m) = A.shape
    A2 = A*A

    row_norms = np.sum(A2, axis=1)

    c_row = np.zeros(shape=(n+padd,n+padd), dtype=np.float64)
    c_row[:n,:n] = 2*(row_norms[:, None] + row_norms[None, :] - 2 * A @ A.T)

    col_norms = np.sum(A2, axis=0)
    c_col = np.zeros(shape=(m+padd,m+padd), dtype=np.float64)

    c_col[:m,:m] = 2*(col_norms[:, None] + col_norms[None, :] - 2*A.T@A)

    return c_row, c_col