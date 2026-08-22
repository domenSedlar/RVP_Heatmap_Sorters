import numpy as np
import pandas as pd

import os
import random
import time

from RVP_Metrics.metrics import moore_stress4, me4, full_eval

from OptimalSorters.tsp_solver import tsp_reorder_matrix_opt
from OptimalSorters.brute_force import brue_force

from Heuristics.hclust import hclust_with_olo
from Heuristics.bae import bae
from Heuristics.rand_swap import rand_swaps, rand_block_swaps, randomly_mirror
from Heuristics.tsp_heur import tsp_lk

def get_matrix(filepath):
    df = pd.read_csv(filepath, delimiter='\t')
    df = df.drop(columns=['ID_REF', 'IDENTIFIER'], errors='ignore')
    A = df.to_numpy()

    return A

def ev(filepath, algo):
    H = get_matrix(filepath)

    sorted = algo(H)

    res = full_eval(sorted)

    return res

def get_size(filepath):
    df = pd.read_csv(filepath, delimiter='\t')
    df = df.drop(columns=['ID_REF', 'IDENTIFIER'], errors='ignore')
    A = df.to_numpy()

    return A.shape

def get_metadata(filenm, dirpath, algo_nm, tm, dataset, metric):
    (n,m) = get_size(os.path.join(dirpath, filenm))
    row = {
        'file_name': filenm,
        'dir' : dirpath,
        'algo': algo_nm,
        'time':tm,
        'size': n*m,
        'row_size': n,
        'col_size': m,
        'dataset' : dataset,
        'optimizing': metric,
    }

    return row


def run(algo, in_dir, algo_nm, dataset, metric, output_path='results.parquet', only_small=True):
    if only_small:
        in_dir = os.path.join(in_dir, 'Small')
    tm = 0
    df = []
    for root, dirs, files in os.walk(in_dir):
        for f in files:
            if '.tsv' in f:
                start_time = time.perf_counter()
                res = ev(os.path.join(root, f), algo)
                end_time = time.perf_counter()
                tm = end_time - start_time

                row = get_metadata(f, root, algo_nm, tm, dataset, metric) | res
                df.append(row)

    df = pd.DataFrame(df)
    
    df.to_parquet(
    output_path, engine="fastparquet", append=os.path.exists(output_path), index=False
        )

if __name__ == "__main__":
    run(
        algo = bae,
        in_dir = 'Data',
        algo_nm = 'BAE',
        dataset='test',
        metric='NS',
        output_path='test_results.parquet',
        only_small=True
    )