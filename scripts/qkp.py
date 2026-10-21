from pathlib import Path

import numpy as np
from numba import njit

BENCH = Path(__file__).parent.parent / 'ising/benchmarks/Knapsack'


def load_qkp(name):
    lines = (BENCH / f'{name}.txt').read_text().splitlines()
    n = int(lines[1])
    P = np.diag(np.array(lines[2].split(), dtype=float))
    for i in range(n - 1):
        row = np.array(lines[3 + i].split(), dtype=float)
        P[i, i+1:] = row
        P[i+1:, i] = row
    C = int(lines[n + 4])
    w = np.array(lines[n + 5].split(), dtype=float)
    return w, P, C


def load_optimum(name):
    for line in (BENCH / 'optimal_energy.txt').read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == name:
            return float(parts[1])


def knapsack_to_qubo(w, P, C):
    n_items = len(w)                                            # Number of items
    B = int(np.floor(np.log2(C)) + 1)                           # Number of slack bits
    a = np.concatenate([w, 2.0 ** np.arange(B)])                # Slack bit vector e.g. [1, 2, 4, 8] appended to w
                                                                # sum(w_i, x_i) + y = C -> sum(a_i, x_i) = C
    n = len(a)
    Q_obj = np.zeros((n, n))                                    # Define Q, we want x^T Q x = -profit for minimization
    Q_obj[:n_items, :n_items] = -P / 2                          # Half all values
    np.fill_diagonal(Q_obj[:n_items, :n_items], -np.diag(P))    # Put back the original diagonal
    return Q_obj, a[None, :], np.array([float(C)])              # one constraint row: a z = C


@njit
def qkp_profit(x, P):
    profit = 0.0
    for i in range(len(x)):                         # same as x @ np.triu(P) @ x, without building np.triu(P)
        for j in range(i, len(x)):
            profit += P[i, j] * x[i] * x[j]
    return profit
