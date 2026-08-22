import numpy as np
import random

from MyUtils.distance_matrix import create_matrices_ns, create_matrices_me


def rand_block_swaps(A, metric, tries=30, temperature=0.1, cooling_rate=0.005, num_of_iterations=1000):
    #A =  A[row_order][:, col_order]
    (n,m) = A.shape
    rows = np.arange(n)
    cols = np.arange(m)
    score_curr = metric(A)
    score_new = metric(A)
    temp = temperature

    for j in range(num_of_iterations):
        up = False
        for i in range(tries):
            a = random.randint(0,n-1)
            b = random.randint(a, n) # size b - a
            if b == n:
                continue
            to = random.choice([i for i in range(n - b + a+1)])

            block = rows[a:b]

            new_rows = []
            l = np.delete(rows, slice(a,b))
            for i in range(to):
                new_rows.append(l[i])
            for i in block:
                new_rows.append(i)
            for i in range(to, len(l)):
                new_rows.append(l[i])


            score_new = metric(A[new_rows][:, cols])
            p = random.random()
            if score_new > score_curr or p < temp:
                score_curr = score_new
                rows = new_rows
                up = True
                break
        temp -= cooling_rate
        temp = max(temp, 0)   
                
        if not up:
            break
    temp = temperature

    for j in range(num_of_iterations):
        up = False
        for i in range(tries):
            a = random.randint(0,m-1)
            b = random.randint(a, m) # size b - a
            if b == n:
                continue
            to = random.choice([i for i in range(m - b + a + 1)])

            block = cols[a:b]

            new_cols = []
            l = np.delete(cols, slice(a,b))
            for i in range(to):
                new_cols.append(l[i])
            for i in block:
                new_cols.append(i)
            for i in range(to, len(l)):
                new_cols.append(l[i])


            score_new = metric(A[rows][:, new_cols])
            p = random.random()
            if score_new > score_curr or p < temp:
                score_curr = score_new
                cols = new_cols
                up = True
                break
        temp -= cooling_rate
        temp = max(temp, 0)       
        if not up:
            break
    return A[rows][:, cols]

def randomly_mirror(A, metric, tries=30, temperature=0.1, cooling_rate=0.005, num_of_iterations=1000):
    #A =  A[row_order][:, col_order]
    (n,m) = A.shape
    rows = np.arange(n)
    cols = np.arange(m)
    score_curr = metric(A)
    score_new = metric(A)
    temp = temperature

    for j in range(num_of_iterations):
        up = False
        for i in range(tries):
            a = random.randint(0,n-1)
            b = random.randint(a+1, n) # size b - a
            if b == n:
                continue

            new_rows = rows.copy()
            new_rows[a:b] = rows[a:b][::-1]

            score_new = metric(A[new_rows][:, cols])
            p = random.random()
            if score_new > score_curr or p < temp:
                score_curr = score_new
                rows = new_rows
                up = True
                break
        temp -= cooling_rate
        temp = max(temp, 0)        
        if not up:
            break
    temp = temperature

    for j in range(num_of_iterations):
        up = False
        for i in range(tries):
            a = random.randint(0,m-1)
            b = random.randint(a+1, m) # size b - a
            if b == n:
                continue

            new_cols = cols.copy()
            new_cols[a:b] = cols[a:b][::-1]

            score_new = metric(A[rows][:, new_cols])
            p = random.random()
            if score_new > score_curr or p < temp:
                score_curr = score_new
                cols = new_cols
                up = True
                break
        temp -= cooling_rate
        temp = max(temp, 0)
        if not up:
            break
    return A[rows][:, cols]

def rand_swaps(A, metric, tries=30, temperature=0.1, cooling_rate=0.005, num_of_iterations=1000):
    #A =  A[row_order][:, col_order]
    (n,m) = A.shape
    rows = np.arange(n)
    cols = np.arange(m)
    score_curr = metric(A)
    score_new = metric(A)

    temp = temperature

    for j in range(num_of_iterations):
        up = False
        for i in range(tries):
            a = random.randint(0,n-1)
            b = random.choice([i for i in range(n) if i != a])

            t = rows[a]
            rows[a] = rows[b]
            rows[b] = t
            score_new = metric(A[rows][:, cols])
            p = random.random()
            if score_new > score_curr or p < temp:
                score_curr = score_new
                up = True
                break
            else:
                t = rows[a]
                rows[a] = rows[b]
                rows[b] = t

        temp -= cooling_rate
        temp = max(temp, 0)
        if not up:
            break

    temp = temperature

    for j in range(num_of_iterations):
        up = False
        for i in range(tries):
            a = random.randint(0,m-1)
            b = random.choice([i for i in range(m) if i != a])

            t = cols[a]
            cols[a] = cols[b]
            cols[b] = t
            score_new = metric(A[rows][:, cols])
            p = random.random()
            if score_new > score_curr or p < temp:
                score_curr = score_new
                up = True
                break
            else:
                t = cols[a]
                cols[a] = cols[b]
                cols[b] = t
        temp -= cooling_rate
        temp = max(temp, 0)
        if not up:
            break

    return A[rows][:, cols]

class RandomSorter:
    def __init__(self, func, nm, metric, tries=30, temperature=0.1, cooling_rate=0.005, num_of_iterations=1000):
        self.func = func
        self.nm = '_'.join([nm, 'Tries='+str(tries), 'Temp='+str(temperature),'Cooling='+str(cooling_rate), 'NumIter='+str(num_of_iterations)])
        self.metric = metric
        self.tries = tries
        self.temperature = temperature
        self.cooling_rate = cooling_rate,
        self.num_of_iter = num_of_iterations

    def get_name(self):
        return self.nm

    def __call__(self, H, metric=None, *args, **kwds):
        if metric is not None:
            raise NotImplementedError

        return self.func(H, metric=self.metric, tries=self.tries, temperature=self.temperature, cooling_rate=self.cooling_rate, num_of_iterations=self.num_of_iter)