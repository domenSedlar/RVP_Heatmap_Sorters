import numpy as np
import pandas as pd

import os
import random
import time
import tarfile

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

def ev(H, algo):
    sorted = algo(H)

    res = full_eval(sorted)

    return res

def get_size(filepath):
    df = pd.read_csv(filepath, delimiter='\t')
    df = df.drop(columns=['ID_REF', 'IDENTIFIER'], errors='ignore')
    A = df.to_numpy()

    return A.shape


def get_metadata(filenm, dirpath, algo_nm, tm, dataset, metric, n=None, m=None):
    if n is None or m is None:
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


def open_tar_mem(f):
    data = f.read()
    text = data.decode().splitlines()

    start = 0
    for i, line in enumerate(text):
        if line[0] != '%':
            start = i+1
            break

    text = text[start:]
    
    parsed_data = [line.strip().split() for line in text if line.strip()]
    edges = [(int(src), int(dst), float(w)) for src, dst, w in parsed_data]
    max_node = max(max(src, dst) for src, dst, _ in edges)
    num_nodes = max_node + 1
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for src, dst, weight in edges:
        adj_matrix[src, dst] = weight

    return adj_matrix

def run_on_tar_gz(algo, in_dir, algo_nm, dataset, metric, output_path='results.parquet', only_small=True, csv_file='Data/sparse_matrix_list_new.csv', size_lim=200):
    table = pd.read_csv(csv_file, sep=';')
    table = table[table['height'] < size_lim]
    table = table[table['width'] < size_lim]
    df = []
    for i, tr in table.iterrows():
        tar_pth = tr.loc['new_path']

        with tarfile.open(tar_pth, "r:gz") as src:
            for member in src.getmembers():
                if member.name != tr['file_name']:
                    continue
                if not (".mtx" in member.name):
                    continue

                f = src.extractfile(member)
                H = open_tar_mem(f)

                start_time = time.perf_counter()
                res = ev(H, algo)
                end_time = time.perf_counter()
                tm = end_time - start_time

                row = get_metadata(member.name, tar_pth, algo_nm, tm, dataset, metric, tr.loc['height'], tr.loc['width']) | res
                df.append(row)

    df = pd.DataFrame(df)
    
    df.to_parquet(
    output_path, engine="fastparquet", append=os.path.exists(output_path), index=False
        )

def run(algo, in_dir, algo_nm, dataset, metric, output_path='results.parquet', only_small=True):
    if only_small:
        in_dir = os.path.join(in_dir, 'Small')
    tm = 0
    df = []
    for root, dirs, files in os.walk(in_dir):
        for f in files:
            if '.tsv' in f:
                H = get_matrix(os.path.join(root, f))

                start_time = time.perf_counter()
                res = ev(H, algo)
                end_time = time.perf_counter()
                tm = end_time - start_time

                row = get_metadata(f, root, algo_nm, tm, dataset, metric) | res
                df.append(row)

    df = pd.DataFrame(df)
    
    df.to_parquet(
    output_path, engine="fastparquet", append=os.path.exists(output_path), index=False
        )

if __name__ == "__main__":
    run_on_tar_gz(
        algo = bae,
        in_dir = 'Data',
        algo_nm = 'BAE',
        dataset='test',
        metric='NS',
        output_path='test_results.parquet',
        only_small=True
    )