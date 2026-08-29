import numpy as np
import pandas as pd

import os
import random
import time
import tarfile

from RVP_Metrics.metrics import moore_stress4, me4, full_eval

from OptimalSorters.tsp_solver import TSP_LIN
from OptimalSorters.brute_force import brue_force

from Heuristics.hclust import Hclust
from Heuristics.bae import BAE
from Heuristics.rand_swap import rand_swaps, rand_block_swaps, randomly_mirror, RandomSorter
from Heuristics.tsp_heur import TSP_LK

def get_matrix(filepath):
    df = pd.read_csv(filepath, delimiter='\t')
    df = df.drop(columns=['ID_REF', 'IDENTIFIER'], errors='ignore')

    df = df.select_dtypes(include=[np.number])
    
    df = df.fillna(df.mean())
    
    if len(df) <= 2:
        return None
        
    A = df.to_numpy()
    return A

def ev(H, algo):
    sorted = algo(H)
    if sorted is None:
        return None
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

def run_on_tar_gz(algo, in_dir, dataset_nm, metric, output_path='results.parquet', only_small=True, csv_file='Data/sparse_matrix_list.csv', size_lim=200):
    table = pd.read_csv(csv_file, sep=';')
    table = table[table['height'] < size_lim]
    table = table[table['width'] < size_lim]
    df = []
    algo_nm = algo.get_name()
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
                if res is None:
                    continue
                end_time = time.perf_counter()
                tm = end_time - start_time

                row = get_metadata(member.name, tar_pth, algo_nm, tm, dataset_nm, metric, tr.loc['height'], tr.loc['width']) | res
                df.append(row)

    df = pd.DataFrame(df)
    return df
    df.to_parquet(
    output_path, engine="fastparquet", append=os.path.exists(output_path), index=False
        )

def run(algo, in_dir, dataset_nm, metric, output_path='results.parquet', only_small=True):
    print("starting ", algo.get_name())
    if only_small:
        in_dir = os.path.join(in_dir, 'Small')
    tm = 0
    algo_nm = algo.get_name()
    df = []
    for root, dirs, files in os.walk(in_dir):
        for f in files:
            if '.tsv' in f:
                H = get_matrix(os.path.join(root, f))
                if H is None:
                    continue

                start_time = time.perf_counter()
                res = ev(H, algo)
                if res is None:
                    continue
                end_time = time.perf_counter()
                tm = end_time - start_time

                row = get_metadata(f, root, algo_nm, tm, dataset_nm, metric) | res
                df.append(row)

    df = pd.DataFrame(df)

    return df
    df.to_parquet(
    output_path, engine="fastparquet", append=os.path.exists(output_path), index=False
        )

def save(df, output_path='results.parquet'):
    df.to_parquet(
        output_path, engine="fastparquet", append=os.path.exists(output_path), index=False
        )

if __name__ == "__main__":
    datasets = [
        ('SparseMatrixSuite', 'Data/Random'),
    ]
    dataset_nm = datasets[0][0]
    dataset = datasets[0][1]

    opt = TSP_LIN()
    save(
        run_on_tar_gz(
        algo = opt,
        in_dir = dataset,
        dataset_nm=dataset_nm,
        metric='NS',
        output_path='results.parquet',
        only_small=True,
        size_lim=31
    ))

    method = lambda algo : save(
        run_on_tar_gz(
        algo = algo,
        in_dir = dataset,
        dataset_nm=dataset_nm,
        metric='NS',
        output_path='results.parquet',
        only_small=False
    ))

    bae = BAE()
    method(bae)

    hclust = Hclust()
    method(hclust)

    tsp_lk = TSP_LK()
    method(tsp_lk)

    rnd_swp = RandomSorter(rand_swaps, "Random_swaps", moore_stress4)
    method(rnd_swp)

    rnd_mir = RandomSorter(randomly_mirror, "Mirror", moore_stress4)
    method(rnd_mir)
    
    rnd_blk = RandomSorter(rand_block_swaps, "Block_Swaps", moore_stress4)
    method(rnd_blk)