import numpy as np
from RVP_Metrics.metrics import moore_stress4, me4
from OptimalSorters.tsp_solver import tsp_reorder_matrix_opt
from OptimalSorters.brute_force import brue_force
import random
import time

from Heuristics.hclust import hclust_with_olo
from Heuristics.bae import bae
from Heuristics.rand_swap import rand_swaps, rand_block_swaps, randomly_mirror
from Heuristics.tsp_heur import tsp_lk