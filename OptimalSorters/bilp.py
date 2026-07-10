# Binary Integer linear program

import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import LinearConstraint, Bounds, milp
import random

n = 5
m = 5

heatmap = np.array([[random.randint(0,1) for i in range(m)] for _ in range(n)])
H = heatmap

# Number of vars
#---------------
num_x = n*n
num_y = m*m
num_wyp = m*m*(m-1)
num_wym = m*m*(m-1)
num_wxp = n*n*(n-1)
num_wxm = n*n*(n-1)

num_alpha = num_x*num_wyp
num_beta = num_x*num_wym
num_gama = num_y*num_wxp
num_delta = num_y*num_wxm

# defining c
#-------------
H2 = heatmap * heatmap

row_sums_H2 = np.sum(H2, axis=1)

cx = np.zeros((n,n))
cx[0] = row_sums_H2
cx[n-1] = row_sums_H2

cx = cx.reshape(-1)

row_sums_H2 = np.sum(H2, axis=0)
cy = np.zeros((m,m))
cy[0] = row_sums_H2
cy[m-1] = row_sums_H2

cy = cy.reshape(-1)

cw = np.zeros(shape=(num_wyp+num_wym+num_wxp+num_wxm))

c_alpha = []

for i in range(n):
    for k in range(n):
        for j in range(m):
            for j_ in range(m):
                if j == j_:
                    continue
                for l in range(m-1):
                    c_alpha.append(H[i,j]*H[i, j_])

c_beta = []

for i in range(n):
    for k in range(n):
        for j in range(m):
            for j_ in range(m):
                if j == j_:
                    continue
                for l in range(1,m):
                    c_beta.append(H[i,j]*H[i, j_])

c_gamma = []

for i in range(n):
    for i_ in range(n-1):
        if i_ == i:
            continue
        for k in range(n):
            for j in range(m):
                for l in range(m):
                    c_gamma.append(H[i,j]*H[i_, j])

c_delta = []

for i in range(n):
    for i_ in range(1,n):
        if i_ == i:
            continue
        for k in range(n):
            for j in range(m):
                for l in range(m):
                    c_delta.append(H[i,j]*H[i_, j])

c_last = c_alpha + c_beta + c_gamma + c_delta
c_last = np.array(c_last)

c = np.concatenate([cx, cy, c_last])

# Constraints
# ===========

lower_bound = []
upper_bound = []
cols = []
data = []
row = []
row_counter = 0


# X sums to 1 in each row
# 1 <= X_row_sum <= 1
X_ids = np.arange(n*n).reshape((n,n))
for _ in range(n):
    row.extend([row_counter for _ in range(n)])
    row_counter += 1

cols = [X_ids[i,j] for i in range(n) for j in range(n)]
data = [1 for _ in range(n) for _ in range(n)]
lower_bound.extend([1 for i in range(n)])
upper_bound.extend([1 for i in range(n)])

# X sums to 1 in each column
# 1 <= X_column_sum <= 1
for _ in range(n):
    row.extend([row_counter for _ in range(n)])
    row_counter += 1

cols.extend([X_ids[i,j] for j in range(n) for i in range(n)])
data.extend([1 for _ in range(n) for _ in range(n)])
lower_bound.extend([1 for i in range(n)])
upper_bound.extend([1 for i in range(n)])


# Y sums to 1 in each row
# 1 <= Y_row_sum <= 1
Y_ids = np.arange(n*n,n*n+m*m).reshape((m,m))
for _ in range(m):
    row.extend([row_counter for _ in range(m)])
    row_counter += 1

cols.extend([Y_ids[i,j] for i in range(m) for j in range(m)])
data.extend([1 for _ in range(m) for _ in range(m)])
lower_bound.extend([1 for i in range(m)])
upper_bound.extend([1 for i in range(m)])

# Y sums to 1 in each column
# 1 <= Y_column_sum <= 1

for _ in range(m):
    row.extend([row_counter for _ in range(m)])
    row_counter += 1

cols.extend([Y_ids[i,j] for j in range(m) for i in range(m)])
data.extend([1 for _ in range(m) for _ in range(m)])
lower_bound.extend([1 for i in range(m)])
upper_bound.extend([1 for i in range(m)])

num_x_y = num_x + num_y
counter = num_x_y
# alpha equals x*y*y
for i in range(n):
    for k in range(n):
        for j in range(m):
            for j_ in range(m):
                if j == j_:
                    continue
                for l in range(m-1):
                    row.extend([row_counter for _ in range(4)])
                    row_counter += 1
                    cols.extend([counter, X_ids[i,k], Y_ids[j, l], Y_ids[j, l+1]])
                    data.extend([1, -1, -1, -1])
                    lower_bound.extend([-2])
                    upper_bound.extend([np.inf])

                    row.extend([row_counter, row_counter, row_counter+1, row_counter+1, row_counter+2, row_counter+2])
                    row_counter += 3
                    cols.extend([counter, X_ids[i,k], counter, Y_ids[j, l], counter, Y_ids[j, l+1]])
                    data.extend([1, -1, 1, -1, 1, -1])
                    lower_bound.extend([0-np.inf, 0-np.inf, 0-np.inf])
                    upper_bound.extend([0,0,0])

                    counter += 1


# beta constraints
for i in range(n):
    for k in range(n):
        for j in range(m):
            for j_ in range(m):
                if j == j_:
                    continue
                for l in range(1,m):
                    row.extend([row_counter for _ in range(4)])
                    row_counter += 1
                    cols.extend([counter, X_ids[i,k], Y_ids[j, l], Y_ids[j, l-1]])
                    data.extend([1, -1, -1, -1])
                    lower_bound.extend([-2])
                    upper_bound.extend([np.inf])

                    row.extend([row_counter, row_counter, row_counter+1, row_counter+1, row_counter+2, row_counter+2])
                    row_counter += 3
                    cols.extend([counter, X_ids[i,k], counter, Y_ids[j, l], counter, Y_ids[j, l-1]])
                    data.extend([1, -1, 1, -1, 1, -1])
                    lower_bound.extend([0-np.inf, 0-np.inf, 0-np.inf])
                    upper_bound.extend([0,0,0])

                    counter += 1


for i in range(n):
    for i_ in range(n):
        if i_ == i:
            continue
        for k in range(n-1):
            for j in range(m):
                for l in range(m):
                    row.extend([row_counter for _ in range(4)])
                    row_counter += 1
                    cols.extend([counter, X_ids[i,k], Y_ids[j, l], X_ids[i_, k+1]])
                    data.extend([1, -1, -1, -1])
                    lower_bound.extend([-2])
                    upper_bound.extend([np.inf])

                    row.extend([row_counter, row_counter, row_counter+1, row_counter+1, row_counter+2, row_counter+2])
                    row_counter += 3
                    cols.extend([counter, X_ids[i,k], counter, Y_ids[j, l], counter, X_ids[i_, k+1]])
                    data.extend([1, -1, 1, -1, 1, -1])
                    lower_bound.extend([0-np.inf, 0-np.inf, 0-np.inf])
                    upper_bound.extend([0,0,0])

                    counter += 1

for i in range(n):
    for i_ in range(n):
        if i_ == i:
            continue
        for k in range(1,n):
            for j in range(m):
                for l in range(m):
                    row.extend([row_counter for _ in range(4)])
                    row_counter += 1
                    cols.extend([counter, X_ids[i,k], Y_ids[j, l], X_ids[i_, k-1]])
                    data.extend([1, -1, -1, -1])
                    lower_bound.extend([-2])
                    upper_bound.extend([np.inf])

                    row.extend([row_counter, row_counter, row_counter+1, row_counter+1, row_counter+2, row_counter+2])
                    row_counter += 3
                    cols.extend([counter, X_ids[i,k], counter, Y_ids[j, l], counter, X_ids[i_, k-1]])
                    data.extend([1, -1, 1, -1, 1, -1])
                    lower_bound.extend([0-np.inf, 0-np.inf, 0-np.inf])
                    upper_bound.extend([0,0,0])

                    counter += 1


A = coo_matrix((data, (row, cols)), shape=(row_counter, counter))
constraints = LinearConstraint(A, lb=lower_bound, ub=upper_bound)

bounds = Bounds(lb=0, ub=1)
integrality = np.ones_like(c)
options = {"time_limit": 60.0}
res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality, options=options)

# Print the results
if res.success:
    print("Optimization successful!")
    print(f"Optimal decision variables (x): {res.x}")
    print(f"Optimal objective value: {res.fun}")
elif res.status == 1:
    print(f"Time limit reached! Best solution found so far: {res.x}")
else:
    print(f"Optimization failed: {res.message}")