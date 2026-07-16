import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import LinearConstraint, Bounds, milp
import itertools
import random

def tsp_solver_mtz(c, n):
    c = c.reshape(-1)
    lower_bound = []
    upper_bound = []
    cols = []
    data = []
    row = []
    row_counter = 0

    c = np.concatenate([c, np.zeros(n-1)])

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

    u_ids = np.array(range(n*n, n*n + n-1))

    for (i,j) in itertools.permutations(range(1,n), 2):
        cols.extend([u_ids[i-1], u_ids[j-1], X_ids[i,j]])
        data.extend([1, -1, n-1])
        row.extend([row_counter for _ in range(3)])
        row_counter += 1
        upper_bound.append(n-2)
        lower_bound.append(-np.inf)

    A = coo_matrix((data, (row, cols)), shape=(row_counter, n*n+n-1))
    constraints = LinearConstraint(A, lb=lower_bound, ub=upper_bound)

    ub = np.ones(n*n + n - 1)
    lb = np.zeros(n*n + n - 1)
    for i in range(n):
        ub[X_ids[i,i]] = 0

    for i in range(n*n, n*n + n - 1):
        ub[i] = n
        lb[i] = 2    

    bounds = Bounds(lb=lb, ub=ub)

    integrality = np.zeros(n*n+n-1, dtype=int)
    integrality[:n*n] = 1
    options = {"time_limit": 60.0}
    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality, options=options)

    # #print the results
    if res.success:
        print("Optimization successful!")
        ##print(f"Optimal decision variables (x): {res.x}")
        #print(f"Optimal objective value: {res.fun}")
    elif res.status == 1:
        print(f"Time limit reached! Best solution found so far: {res.x}")
    #else:
        #print(f"Optimization failed: {res.message}")
    
    return res

def tsp_solver_dfj(c, n):
    c = c.reshape(-1)
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

    for q in range(2, n):
        for Q in itertools.combinations(range(n), q):
            for (i,j) in itertools.permutations(Q,2):
                cols.extend([X_ids[i,j]])
                data.append(1)
                row.append(row_counter)
            

            lower_bound.append(0)
            upper_bound.append(q-1)
            row_counter += 1
    
    A = coo_matrix((data, (row, cols)), shape=(row_counter, n*n))
    constraints = LinearConstraint(A, lb=lower_bound, ub=upper_bound)

    ub = np.ones(n*n)
    for i in range(n): # x_ii is a self connection aka a cycle of size 1, which we never want as the hamiltonian path does not contain smaller cycles
        ub[X_ids[i,i]] = 0

    bounds = Bounds(lb=0, ub=ub)

    integrality = np.ones_like(c)
    options = {"time_limit": 60.0}
    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality, options=options)
    return res
    # #print the results
    #if res.success:
        #print("Optimization successful!")
        #print(f"Optimal decision variables (x): {res.x}")
        #print(f"Optimal objective value: {res.fun}")
    #elif res.status == 1:
        #print(f"Time limit reached! Best solution found so far: {res.x}")
    #else:
        #print(f"Optimization failed: {res.message}")

def extract_tour_order(res, n):
    x = res.x[:(n+1)*(n+1)].reshape((n+1,n+1))
    #print(x)
    tour = []
    current = n

    while len(tour) < n:
        next_city = np.argmax(x[current])
        if next_city == n:
            current = next_city
            continue
        tour.append(next_city)
        current = next_city

    return tour


def tsp_reorder_matrix_opt(heatmap, n, m):
    A = heatmap
    (n,m) = A.shape
    A2 = A*A
    A2_col_sum = np.sum(A2, axis=1)
    c_row = np.zeros(shape=(n+1,n+1), dtype=np.float64)

    for h in range(n):
        for i in range(n):
            #c_row[h,i] += 2*A2_col_sum[i]
            for j in range(m):
                #c_row[h,i] += - 4*A[h,j]*A[i,j]
                c_row[h, i] += - 4*A[h,j]*A[i,j] + 2*A2[i, j] + 2*A2[h,j]
                #if i == 0 or i == n-1:
                 #   c_row[h, i] += - A2[h,j]

    res_row = tsp_solver_mtz(c_row, n+1)
    row_order = extract_tour_order(res_row, n)

    A2_row_sum = np.sum(A2, axis=0)
    c_col = np.zeros(shape=(m+1,m+1), dtype=np.float64)

    for j in range(m):
        for k in range(m):
            #c_col[j,k] += 2*A2_row_sum[k]
            for i in range(n):
                #c_col[j,k] += - 4*A[i,j]*A[i,k]
                c_col[j, k] += - 4*A[i,j]*A[i,k] + 2*A2[i,k] + 2*A2[i,j]
                #if k == 0 or k == m-1:
                  #  c_col[j,k] += - A2[i,j]

    res_col = tsp_solver_mtz(c_col, m+1)
    col_order = extract_tour_order(res_col, m)

    score = None # ill do this later
    return heatmap[row_order][:, col_order], score, row_order, col_order

def tsp_reorder_matrix_opt_me(heatmap, n, m): # confirmed that it works
    A = heatmap
    (n,m) = A.shape
    A2 = A*A
    A2_col_sum = np.sum(A2, axis=1)
    c_row = np.zeros(shape=(n+1,n+1), dtype=np.float64)

    for h in range(n):
        for i in range(n):
            #c_row[h,i] += 2*A2_col_sum[i]
            for j in range(m):
                #c_row[h,i] += - 4*A[h,j]*A[i,j]
                c_row[h, i] += - 2*A[h,j]*A[i,j]
                #if i == 0 or i == n-1:
                #    c_row[h, i] += - A2[h,j]

    res_row = tsp_solver_mtz(c_row, n+1)
    row_order = extract_tour_order(res_row, n)

    A2_row_sum = np.sum(A2, axis=0)
    c_col = np.zeros(shape=(m+1,m+1), dtype=np.float64)

    for j in range(m):
        for k in range(m):
            #c_col[j,k] += 2*A2_row_sum[k]
            for i in range(n):
                #c_col[j,k] += - 4*A[i,j]*A[i,k]
                c_col[j, k] += - 2*A[i,j]*A[i,k]
                #if k == 0 or k == m-1:
                #    c_col[j,k] += - A2[i,j]

    res_col = tsp_solver_mtz(c_col, m+1)
    col_order = extract_tour_order(res_col, m)

    score = None # ill do this later
    return heatmap[row_order][:, col_order], score, row_order, col_order

def test_reordering():
    n = 5
    m = 5

    heatmap = np.array([[random.randint(0,1) for i in range(m)] for _ in range(n)])
    heatmap, _,_,_ = tsp_reorder_matrix_opt(heatmap, n, m)


if __name__ == "__main__":
    test_reordering()