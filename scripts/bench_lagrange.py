"""
Benchmark the Lagrange solver on QKP instances.

Output in data/<EXPERIMENT>/:
    meta.json               commit hash, date, seeds etc.
    settings.csv            one row per setting, the row number is the setting's index in the arrays
    n<budget>/<name>.npz    gap, feasible, lam, penalty: arrays of shape (setting, seed, run)
                            P, eta: the P and η each setting used on this instance
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import itertools
import json
import subprocess
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from lagrange import BENCH, knapsack_to_qubo, load_optimum, load_qkp, multiplicative_lagrange, qkp_profit, qubo_to_ising, paper_rule

WORKERS = 30                                                # number of cores
EXPERIMENT = 'qkp_sweep'
BUDGETS = [50, 200]                                         # number of iterations
SEEDS = 200                                                 # Number of seeds per run
BAD = {'jeu_100_100_4', 'jeu_200_25_3', 'jeu_300_25_3'}     # Bad benchmarks to skip

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'data' / EXPERIMENT

# Common for all runs
COMMON = {
    'init_frac' : [0.8],        # fraction of spins in initial cluster size
    'end_frac'  : [0.0625],     # fraction of spins in final cluster size
    'exponent'  : [0.0],        # 0 = linear decay
    'beta2'     : [0.999],      # used by adam
}

# P = P_factor * 2dN (paper).
# n can be set with one of three keys:
#   - 'r'       : η = r*P, r is a ratio
#   - 'n_factor': η = n_factor * 20, some scalar of the papers n = 20
#   - 'eta'     : η as an absolute value
P_FACTORS = [1, 2, 4, 8, 16, 32, 64]
RATIOS    = [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]

# Parametric, all possible combinations will be run
GRIDS = [
    {   # penalty only
        'update'    : ['plain'],
        'beta1'     : [0.0],
        'P_factor'  : [64, 256, 1024, 4096, 16384],
        'eta'       : [0.0],
    },
    {   # lagrange, scale n by ratio
        'update'    : ['plain'],
        'beta1'     : [0.0],
        'P_factor'  : P_FACTORS,
        'r'         : RATIOS,
    },
    {   # lagrange w/ momentun, scale n by ratio
        'update'    : ['momentum', 'momentum_bc'],
        'beta1'     : [0.5, 0.75, 0.875],
        'P_factor'  : P_FACTORS,
        'r'         : RATIOS,
    },
    {   # lagrange w/ adam
        'update'    : ['adam'],
        'beta1'     : [0.5, 0.9],
        'P_factor'  : P_FACTORS,
        'eta'       : [0.3, 1, 3, 10, 30],
    },
    # Some fixed runs that gave good results previously
    {'update': ['plain'],    'beta1': [0.0], 'P_factor': [1],  'n_factor': [1]},
    {'update': ['plain'],    'beta1': [0.0], 'P_factor': [4],  'n_factor': [4]},
    {'update': ['plain'],    'beta1': [0.0], 'P_factor': [8],  'n_factor': [64]},
    {'update': ['momentum'], 'beta1': [0.5], 'P_factor': [16], 'n_factor': [256]},
]


def settings():
    """
    I have no idea what's going on and don't use itertools but it is correct.
    """
    for grid in GRIDS:
        grid = COMMON | grid
        for values in itertools.product(*grid.values()):
            yield dict(zip(grid, values))


SETTINGS = list(settings())


def step_size(setting, P, n_rule):
    """
    η of a setting, from whichever of 'r', 'n_factor' or 'eta' it has.
    """
    if 'r' in setting:
        return setting['r'] * P
    if 'n_factor' in setting:
        return setting['n_factor'] * n_rule
    return setting['eta']


def run_instance(name, n_iter):
    w, P, C = load_qkp(name)
    best = load_optimum(name)
    Q_obj, Q_pen, _, a = knapsack_to_qubo(w, P, C)
    J_obj, h_obj, _ = qubo_to_ising(Q_obj)
    J_pen, h_pen, _ = qubo_to_ising(Q_pen)
    P_rule, n_rule = paper_rule(P, a, C)

    shape = (len(SETTINGS), SEEDS, n_iter)
    gap      = np.zeros(shape, dtype=np.float32)
    feasible = np.zeros(shape, dtype=bool)
    lam      = np.zeros(shape, dtype=np.float32)
    penalty  = np.zeros(shape, dtype=np.float32)
    Ps       = np.zeros(len(SETTINGS))
    etas     = np.zeros(len(SETTINGS))

    for i, setting in enumerate(SETTINGS):
        Ps[i] = setting['P_factor'] * P_rule
        etas[i] = step_size(setting, Ps[i], n_rule)
        for seed in range(SEEDS):
            states, lambdas = multiplicative_lagrange(
                J_obj,
                h_obj,
                J_pen,
                h_pen,
                a,
                C,
                P=Ps[i],
                eta=etas[i],
                n_iter=n_iter,
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

            feasible[i, seed] = x @ w <= C
            gap[i, seed] = np.where(feasible[i, seed], 1 - profit / best, 1.0)
            lam[i, seed] = lambdas
            penalty[i, seed] = z @ a - C

    np.savez(OUT / f'n{n_iter}' / f'{name}.npz', gap=gap, feasible=feasible, lam=lam, penalty=penalty, P=Ps, eta=etas)
    print(f'{name} n{n_iter}: done', flush=True)


if __name__ == '__main__':
    names = sys.argv[1:] or sorted((p.stem for p in BENCH.glob('jeu_*.txt') if p.stem not in BAD), reverse=True)

    for n_iter in BUDGETS:
        (OUT / f'n{n_iter}').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(SETTINGS).to_csv(OUT / 'settings.csv', index_label='setting')
    meta = {
        'commit'    : subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        'dirty'     : subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=ROOT).returncode != 0,    # uncommitted changes to tracked files
        'date'      : datetime.now().isoformat(timespec='seconds'),
        'seeds'     : SEEDS,
        'budgets'   : BUDGETS,
        'instances' : names,
    }
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=2))

    tasks = [(name, n_iter) for n_iter in sorted(BUDGETS, reverse=True) for name in names]     # biggest first
    with Pool(WORKERS) as pool:
        pool.starmap(run_instance, tasks)
