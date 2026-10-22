from pathlib import Path

import numpy as np
from numba import njit

from scipy.optimize import Bounds, LinearConstraint, milp

BENCH = Path(__file__).parent.parent / 'ising/benchmarks/MKP'


def load_mkp(name, k):
    """
    Instance k of an OR-Library file.
    """
    numbers = np.array((BENCH / f'{name}.txt').read_text().split(), dtype=float)
    i = 1                                           # numbers[0] is the number of instances in the file
    for _ in range(k + 1):
        n, m = int(numbers[i]), int(numbers[i + 1]) # items, constraints (numbers[i + 2] is the optimum, 0 = not given)
        i += 3
        p = numbers[i:i + n]                        # profit per item
        i += n
        W = numbers[i:i + m * n].reshape(m, n)      # weight of every item in every constraint
        i += m * n
        C = numbers[i:i + m]                        # capacity per constraint
        i += m
    return p, W, C


def mkp_to_qubo(p, W, C):
    n_bits = [int(np.log2(c)) + 1 for c in C]       # slack bits per constraint
    S = np.zeros((len(C), sum(n_bits)))
    start = 0
    for i, k in enumerate(n_bits):
        S[i, start:start + k] = 2.0 ** np.arange(k) # each constraint's slack in its own row
        start += k
    A = np.hstack([W, S])                           # items | slack
    Q_obj = np.zeros((A.shape[1], A.shape[1]))
    Q_obj[:len(p), :len(p)] = np.diag(-p)           # profit only on items, slack earns nothing
    return Q_obj, A, C.astype(float)


def mkp_optimum(p, W, C, time_limit=None):
    """
    Get exact optimum with scipy's solver.
    """
    res = milp(-p, constraints=LinearConstraint(W, -np.inf, C), integrality=np.ones(len(p)), bounds=Bounds(0, 1),
               options={'time_limit': time_limit, 'mip_rel_gap': 0})
    return -res.fun, res.mip_gap


def load_best(name, k):
    """
    Best value of an instance from best_values.txt (see the gap column there: 0 = proven optimal).
    """
    for line in (BENCH / 'best_values.txt').read_text().splitlines():
        parts = line.split()
        if parts[:2] == [name, str(k)]:
            return float(parts[2])


def mkp_rule(p, A, b):
    """
    P = 5dN and η = 0.05 from Delacour 2025 (Table I), in our units.
    MKP has no couplings, so the paper sets d = 2/(N+1), which makes P ~= 10 before scaling.
    """
    N = A.shape[1]
    scale = np.abs(p).max() / max(np.abs(A).max(), np.abs(b).max())**2
    return 5 * 2 / (N + 1) * N * scale, 0.05 * scale
