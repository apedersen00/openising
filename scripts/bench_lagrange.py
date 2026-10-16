"""
Benchmark the Lagrange solver on QKP instances.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from lagrange import BENCH, knapsack_to_qubo, load_optimum, load_qkp, multiplicative_lagrange, qkp_profit, qubo_to_ising, paper_rule

WORKERS = 30
EXPERIMENT = 'qkp_rule'
N_ITER = 200
SEEDS = 200

# Common for all runs
COMMON = {
    'init_frac' : [0.4, 0.8],   # fraction of spins in initial cluster size
    'end_frac'  : [0.0625],     # fraction of spins in final cluster size
    'exponent'  : [0.0],        # 0 = linear decay
    'beta1'     : [0.9],        # used by momentum and adam
    'beta2'     : [0.999],      # used by adam
}

# Parametric, all possible combinations will be run
GRIDS = [
    {
        'update'    : ['plain'],
        'P_factor'  : [0.25, 0.5, 1, 2, 4, 8, 16, 32],
        'n_factor'  : [0.25, 0.5, 1, 2, 4, 8, 16, 64, 256]
    },
    {
        'update'    : ['plain'],
        'P_factor'  : [64, 256, 1024, 4096, 16384],
        'n_factor'  : [0]
    },
]


def settings():
    """
    I have no idea what's going on and don't use itertools but it is correct.
    """
    for grid in GRIDS:
        grid = COMMON | grid
        for values in itertools.product(*grid.values()):
            yield dict(zip(grid, values))


def run_instance(name):
    w, P, C = load_qkp(name)
    best = load_optimum(name)
    Q_obj, Q_pen, _, a = knapsack_to_qubo(w, P, C)
    J_obj, h_obj, _ = qubo_to_ising(Q_obj)
    J_pen, h_pen, _ = qubo_to_ising(Q_pen)
    P_rule, n_rule = paper_rule(P, a, C)

    results = []
    for setting in settings():
        for seed in range(SEEDS):
            states, lambdas = multiplicative_lagrange(
                J_obj,
                h_obj,
                J_pen,
                h_pen,
                a,
                C,
                P=setting['P_factor'] * P_rule,
                eta=setting['n_factor'] * n_rule,
                n_iter=N_ITER,
                init_size=int(setting['init_frac'] * len(a)),
                end_size=int(setting['end_frac'] * len(a)),
                seed=seed,
                exponent=setting['exponent'],
                update=setting['update'],
                beta1=setting['beta1'],
                beta2=setting['beta2'],
            )

            z = (1 + states) / 2    # convert [-1, +1] to [0, 1]
            x = z[:, :len(w)]       # slice off the slack bits
            profit = np.array([qkp_profit(xi, P) for xi in x])
            feasible = x @ w <= C

            results.append(pd.DataFrame({
                'instance': name,
                **setting,
                'P': setting['P_factor'] * P_rule,
                'n': setting['n_factor'] * n_rule,
                'seed': seed,
                'run': np.arange(N_ITER),
                'gap': np.where(feasible, 1 - profit / best, 1.0),
                'feasible': feasible,
                'lambda': lambdas,
                'profit': profit,
                'weight': x @ w,
                'penalty': z @ a - C,
            }))

    out = Path(__file__).parent.parent / 'data' / EXPERIMENT
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat(results, ignore_index=True)
    df.to_csv(out / f'{name}.csv.gz', index=False)
    print(f'{name}: {len(df)} rows')


if __name__ == '__main__':
    names = sys.argv[1:] or sorted((p.stem for p in BENCH.glob('jeu_*.txt')), reverse=True)
    with Pool(WORKERS) as pool:
        pool.map(run_instance, names)
