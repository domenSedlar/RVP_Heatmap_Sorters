from scipy.optimize import differential_evolution

from Heuristics.rand_swap import RandomSorter, rand_swaps, rand_block_swaps, randomly_mirror
from RVP_Metrics.metrics import moore_stress4, Metric

from eval import run

def ev(param1, param2, param3, func=rand_swaps):
    algo = RandomSorter(func, "rand_swap", moore_stress4, param3, param1, param2, 1000)

    df = run(
        algo,
        in_dir = 'Data/Random_Train',
        dataset_nm='test',
        metric='NS',
        output_path='test_results.parquet',
        only_small=False
    )
    return df['NS4'].mean()

def objective(params):
    p1, p2, p3 = params
    p3 = int(p3)
    return ev(p1, p2, p3)


bounds = [(0.0, 0.1), (0.0, 0.1), (10,100)]
x0 = [0.0, 0.0, 30]

result = differential_evolution(objective, bounds, integrality=[False, False, True], seed=42)

print(f"Optimal continuous (p1, p2): {result.x[:2]}")
print(f"Optimal integer (p3): {int(result.x[2])}")