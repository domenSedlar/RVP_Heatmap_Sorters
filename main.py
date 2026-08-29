import random
import numpy as np
from scipy.linalg import issymmetric

from MyUtils.distance_matrix import create_matrices_ns

m = 5
n = 5
A = np.array([[random.randint(0,1) for _ in range(m)] for _ in range(n)], dtype=np.float64)
r, c = create_matrices_ns(A)
print(r)
print(issymmetric(r))