import itertools
import numpy as np

def brue_force(heatmap, criteria_function, bigger_better=False):
    (n,m) = heatmap.shape
    best_row = [i for i in range(n)]
    best_col = [i for i in range(m)]
    best_score = criteria_function(heatmap)

    for row_p in itertools.permutations(range(n)):
        for col_p in itertools.permutations(range(m)):
            s = criteria_function(heatmap[np.ix_(row_p, col_p)])
            if (s < best_score and not bigger_better) or (s > best_score and bigger_better):
                best_score = s
                best_row = row_p
                best_col = col_p

    return heatmap[np.ix_(best_row, best_col)], best_score, best_row, best_col