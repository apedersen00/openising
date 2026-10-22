"""
Benchmark the Lagrange solver on MKP instances.

Output in data/<EXPERIMENT>/:
    meta.json                   commit hash, date, seeds etc.
    settings.csv                one row per setting, the row number is the setting's index in the arrays
    n<budget>/<name>_<k>.npz    gap, feasible, penalty: arrays of shape (setting, seed, run)
                                lam: mean λ over the last quarter of the runs, shape (setting, seed, constraint)
                                P, eta: the P and η each setting used on this instance
                                lp_price: dual prices of the LP relaxation, one per constraint
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
from scipy.optimize import linprog

from lagrange import multiplicative_lagrange, penalty_qubo, qubo_to_ising
from mkp import load_best, load_mkp, mkp_rule, mkp_to_qubo

WORKERS = 30                                                # number of cores
EXPERIMENT = 'mkp_sweep'
BUDGETS = [50, 200]                                         # number of iterations
SEEDS = 50                                                  # Number of seeds per run
FILES = ['mknap1'] + [f'mknapcb{i}' for i in range(1, 10)]

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'data' / EXPERIMENT

# Common for all runs
COMMON = {
    'init_frac' : [0.8],        # fraction of spins in initial cluster size
    'end_frac'  : [0.0625],     # fraction of spins in final cluster size
    'exponent'  : [0.0],        # 0 = linear decay
    'beta2'     : [0.999],      # used by adam
}

# P = P_factor * 5dN (paper, MKP).
# n can be set with one of three keys:
#   - 'r'       : η = r*P, r is a ratio
#   - 'n_factor': η = n_factor * 0.05, some scalar of the papers n = 0.05
#   - 'eta'     : η as an absolute value
P_FACTORS = [1/4, 1/2, 1, 2, 4, 8, 16]
RATIOS    = [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]

# Parametric, all possible combinations will be run
GRIDS = [
    {   # penalty only
        'update'    : ['plain'],
        'beta1'     : [0.0],
        'P_factor'  : [1, 4, 16, 64, 256, 1024],
        'eta'       : [0.0],
    },
    {   # lagrange, scale n by ratio
        'update'    : ['plain'],
        'beta1'     : [0.0],
        'P_factor'  : P_FACTORS,
        'r'         : RATIOS,
    },
    {   # lagrange w/ momentun, scale n by ratio
        'update'    : ['momentum'],
        'beta1'     : [0.875],
        'P_factor'  : P_FACTORS,
        'r'         : RATIOS,
    },
    # The paper's setting
    {'update': ['plain'], 'beta1': [0.0], 'P_factor': [1], 'n_factor': [1]},
]


def settings():
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


def load_instance(name):
    """
    p, W, C and the best known value of an instance named <file>_<k>, e.g. mknapcb1_0.
    """
    file, k = name.rsplit('_', 1)
    return *load_mkp(file, int(k)), load_best(file, int(k))


def spins(name):
    p, W, C, _ = load_instance(name)
    return mkp_to_qubo(p, W, C)[1].shape[1]


def run_instance(name, n_iter):
    p, W, C, best = load_instance(name)
    Q_obj, A, b = mkp_to_qubo(p, W, C)
    J_obj, h_obj, _ = qubo_to_ising(Q_obj)
    J_pen, h_pen, _ = qubo_to_ising(penalty_qubo(A, b))
    P_rule, n_rule = mkp_rule(p, A, b)
    lp_price = -linprog(-p, A_ub=W, b_ub=C, bounds=(0, 1)).ineqlin.marginals   # profit per extra unit of capacity, items fractional

    shape = (len(SETTINGS), SEEDS, n_iter)
    gap      = np.zeros(shape, dtype=np.float32)
    feasible = np.zeros(shape, dtype=bool)
    penalty  = np.zeros(shape, dtype=np.float32)
    lam      = np.zeros((len(SETTINGS), SEEDS, len(C)), dtype=np.float32)
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
                A,
                b,
                P=Ps[i],
                eta=etas[i],
                n_iter=n_iter,
                init_size=int(setting['init_frac'] * A.shape[1]),
                end_size=int(setting['end_frac'] * A.shape[1]),
                seed=seed,
                exponent=setting['exponent'],
                update=setting['update'],
                beta1=setting['beta1'],
                beta2=setting['beta2'],
            )

            z = (1 + states) / 2    # convert [-1, +1] to [0, 1]
            x = z[:, :len(p)]       # slice off the slack bits

            feasible[i, seed] = np.all(x @ W.T <= C, axis=1)
            gap[i, seed] = np.where(feasible[i, seed], 1 - x @ p / best, 1.0)
            penalty[i, seed] = np.abs(z @ A.T - b).sum(axis=1)     # 0 = items and slack both right
            lam[i, seed] = lambdas[-(n_iter // 4):].mean(axis=0)

    np.savez(OUT / f'n{n_iter}' / f'{name}.npz', gap=gap, feasible=feasible, penalty=penalty, lam=lam, P=Ps, eta=etas, lp_price=lp_price)
    print(f'{name} n{n_iter}: done', flush=True)


if __name__ == '__main__':
    names = sys.argv[1:] or [f'{file}_{k}' for file in FILES for k in range(7 if file == 'mknap1' else 30)]
    names = sorted(names, key=spins, reverse=True)

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
