import os
import gzip
import time
import numpy as np
import tsplib95

from tsp_solver import tsp_solver_mtz


TSP_DIR = "ALL_tsp"


def load_tsp(path):
    """
    Load a TSPLIB .tsp.gz file and return the distance matrix.
    """
    with gzip.open(path, "rt") as f:
        problem = tsplib95.read(f)

    n = problem.dimension
    if n > 50:
        return None, n, problem

    C = np.zeros((n, n), dtype=float)

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            C[i - 1, j - 1] = problem.get_weight(i, j)

    return C, n, problem


def load_optimal_tour_length(tsp_problem, opt_path):
    """
    Load .opt.tour.gz and calculate the optimal tour length.
    """

    with gzip.open(opt_path, "rt") as f:
        opt_tour = tsplib95.read(f)

    tour = opt_tour.tours[0]

    length = 0

    for i in range(len(tour)):
        a = tour[i]
        b = tour[(i + 1) % len(tour)]
        length += tsp_problem.get_weight(a, b)

    return length


def test_all_tsp():
    results = []

    files = sorted(os.listdir(TSP_DIR))

    tsp_files = [
        f for f in files
        if f.endswith(".tsp.gz")
    ]

    for tsp_file in tsp_files:

        name = tsp_file.replace(".tsp.gz", "")
        tsp_path = os.path.join(TSP_DIR, tsp_file)

        opt_file = name + ".opt.tour.gz"
        opt_path = os.path.join(TSP_DIR, opt_file)

        print("\n==============================")
        print("Testing:", name)

        try:
            # Load problem
            C, n, problem = load_tsp(tsp_path)

            # Skip very large problems
            if n > 50:
                print(f"Skipping n={n} (too large)")
                continue

            # Load known optimum
            if os.path.exists(opt_path):
                optimum = load_optimal_tour_length(problem, opt_path)
            else:
                optimum = None
                print("No optimal tour file found")

            print("Cities:", n)
            if optimum:
                print("Known optimum:", optimum)

            # Solve
            start = time.time()

            result = tsp_solver_mtz(
                C.ravel(),
                n
            )

            elapsed = time.time() - start

            if result.success:
                found = result.fun

                print("Solver result:", found)
                print("Time:", elapsed)

                if optimum is not None:
                    correct = abs(found - optimum) < 1e-6
                    print("Correct:", correct)

                results.append(
                    {
                        "name": name,
                        "n": n,
                        "found": found,
                        "optimal": optimum,
                        "correct": (
                            abs(found - optimum) < 1e-6
                            if optimum is not None
                            else None
                        ),
                        "time": elapsed
                    }
                )

            else:
                print("Solver failed:", result.message)

        except Exception as e:
            print("ERROR:", e)

            results.append(
                {
                    "name": name,
                    "error": str(e)
                }
            )

    return results


if __name__ == "__main__":

    results = test_all_tsp()

    print("\n\n========== SUMMARY ==========")

    for r in results:
        print(r)