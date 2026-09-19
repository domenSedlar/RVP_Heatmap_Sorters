import numpy as np
import pandas as pd
from numpy.linalg import det

import os
import random
import time
import tarfile

from RVP_Metrics.metrics import moore_stress4, me4, full_eval

from OptimalSorters.tsp_solver import TSP_LIN
from OptimalSorters.brute_force import brue_force
from OptimalSorters.tsp_gurobi import TSP_gurobi

from Heuristics.hclust import Hclust
from Heuristics.bae import BAE
from Heuristics.rand_swap import rand_swaps, rand_block_swaps, randomly_mirror, RandomSorter
from Heuristics.tsp_heur import TSP_LK

from chain import Chain

def get_matrix(filepath, header=0):
    df = pd.read_csv(filepath, delimiter='\t', header=header)
    df = df.drop(columns=['ID_REF', 'IDENTIFIER'], errors='ignore')

    df = df.select_dtypes(include=[np.number])
    
    df = df.fillna(df.mean())
    
    if len(df) <= 2:
        return None
        
    A = df.to_numpy()
    return A

def ev(H, algo):
    is_square = H.shape[0] == H.shape[1]
    if is_square:
        detH = int(abs(det(H.copy()))) # i round it to an int, so we can ignore minor float errors (this is here only to make sure we don't seriously destroy the matrix)
    ordered = algo(H)
    if ordered is None:
        return None
    print(ordered.shape)

    if is_square:
        if int(abs(det(ordered.copy()))) != detH: # This checks that the sorters don't mess up the heatmap(row and column permutations dont affect the absolute value of the determinante)
            print(detH)
            print(abs(det(ordered.copy())))
            print(H)
            print(ordered)
            raise ValueError("Returned matrix determinant doesn't match the initial determinant!")

    res = full_eval(ordered)
    
    return res

def get_size(filepath):
    df = pd.read_csv(filepath, delimiter='\t')
    df = df.drop(columns=['ID_REF', 'IDENTIFIER'], errors='ignore')
    A = df.to_numpy()

    return A.shape


def get_metadata(filenm, dirpath, algo_nm, tm, dataset, metric, n=None, m=None, is_opt=False):
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
        'opt': is_opt
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

def run(algo, in_dir, dataset_nm, metric, output_path='results.parquet', only_small=True, header=0):
    print("starting ", algo.get_name())
    if only_small:
        in_dir = os.path.join(in_dir, 'Small')
    tm = 0
    algo_nm = algo.get_name()
    df = []
    for root, dirs, files in os.walk(in_dir):
        for f in files:
            if '.tsv' in f:
                H = get_matrix(os.path.join(root, f), header=header)
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


def save(df, output_path='results.parquet'):
    df.to_parquet(
        output_path, engine="fastparquet", append=os.path.exists(output_path), index=False
        )

def old_code():
    datasets = [
        ('SparseMatrixSuite', ''),
        ('GDS_rand', 'Data/GDS_Random'),
        ('Random', 'Data/Random'),

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

if __name__ == "__main__":
    datasets = [
        ('Random', 'Data/Random', None),
        #('GDS_rand', 'Data/GDS_Random', 0),
        ]
    dataset_nm = datasets[0][0]
    dataset = datasets[0][1]
    header = datasets[0][2]
    a = BAE()

    save(
        run(
        algo = a,
        in_dir = dataset,
        dataset_nm=dataset_nm,
        metric='NS',
        output_path='results.parquet',
        only_small=True,
        header= header
        )
    )