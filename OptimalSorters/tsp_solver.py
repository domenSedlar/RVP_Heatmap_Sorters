import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import LinearConstraint, Bounds, milp
import itertools

def tsp_solver_mtz(c, n):
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

    # Print the results
    if res.success:
        print("Optimization successful!")
        print(f"Optimal decision variables (x): {res.x}")
        print(f"Optimal objective value: {res.fun}")
    elif res.status == 1:
        print(f"Time limit reached! Best solution found so far: {res.x}")
    else:
        print(f"Optimization failed: {res.message}")

def tsp_solver_dfj(c, n):
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

    # Print the results
    if res.success:
        print("Optimization successful!")
        print(f"Optimal decision variables (x): {res.x}")
        print(f"Optimal objective value: {res.fun}")
    elif res.status == 1:
        print(f"Time limit reached! Best solution found so far: {res.x}")
    else:
        print(f"Optimization failed: {res.message}")