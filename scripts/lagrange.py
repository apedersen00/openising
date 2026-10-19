from pathlib import Path

import numpy as np
from numba import njit

BENCH = Path(__file__).parent.parent / 'ising/benchmarks/Knapsack'


# ---------- QKP problem ----------

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
    Q_pen = np.outer(a, a) - 2 * C * np.diag(a)
    return Q_obj, Q_pen, C**2, a


def qubo_to_ising(Q):
    J = -0.5 * Q
    np.fill_diagonal(J, 0)
    h = -0.5 * Q.sum(axis=1)
    c = 0.25 * Q.sum() + 0.25 * np.trace(Q)
    return J, h, c


def max_field(J, h):
    return np.max(np.abs(J).sum(axis=1) + np.abs(h))


@njit
def qkp_profit(x, P):
    profit = 0.0
    for i in range(len(x)):                         # same as x @ np.triu(P) @ x, without building np.triu(P)
        for j in range(i, len(x)):
            profit += P[i, j] * x[i] * x[j]
    return profit


def paper_rule(P, a, C):
    """
    P and n from Delacour 2025: P = 2dN and n = 20 after dividing the profits by their max
    and the weights by max(a, C). Returned in our (not divided) units.
    """
    d = np.mean(P[~np.eye(len(P), dtype=bool)] != 0)          # density of the pair profits
    scale = np.abs(P).max() / max(np.abs(a).max(), C)**2
    return 2 * d * len(a) * scale, 20 * scale


# ---------- Cluster sizes ----------

def cluster_size(it, n_iter, init_size, end_size):
    """
    Linear decay from init_size to end_size over n_iter.
    """
    steps = np.linspace(init_size, end_size, n_iter)
    return round(steps[it])


def cluster_size_exp(it, n_iter, init_size, end_size, exponent=3.0):
    """
    Exponential decay from init_size to end_size over n_iter.
    """
    decay = ((end_size - 1) / init_size) ** (it * exponent / (n_iter - 1))
    return int(np.floor(decay * (init_size - end_size) + end_size))


# ---------- Event solver ----------

@njit
def analog_event(J, h, v, max_events):
    """
    Args:
        - J: coupling
        - h: biases
        - v: continuous voltages per spin
        - max_events: maximum number of flips per run
    """
    n = len(h)                                                          # Number of spins
    s = np.sign(v)
    for i in range(n):                                                  # If a spin is exactly at the threshold, make it +1
        if s[i] == 0:
            s[i] = 1.0

    f = J @ s + h                                                       # The local fields
    t = 0.0                                                             # time since start
    for event in range(max_events):                                     # iterate till either conergence or max_events
        t_next = np.inf                                                 # time till next spin crosses the threshold
        k = -1                                                          # Spin index that has has the next crossing at t_next
        for i in range(n):                                              # Check every spin
            if v[i] * f[i] < 0 or (v[i] == 0.0 and s[i] * f[i] < 0):    # Check if a pin moves torwards the threshold (0)
                t_i = -v[i] / f[i]                                      # Time till the spin crosses the threshold
                if t_i < t_next:                                        # log the spin if it has the earliest crossing time found so far
                    t_next = t_i
                    k = i

        if k < 0:                                                       # If no spin heads for the crossing: convergence
            return s, t, event

        for i in range(n):                                              # Update all spin voltages
            reached = v[i] * f[i] < 0 and -v[i] / f[i] <= t_next        # Which spins reach the threshold at t_next
            v[i] = min(1.0, max(-1.0, v[i] + t_next * f[i]))            # Updated the voltages
            if reached:                                                 # Place the crossing spins at exactly 0
                v[i] = 0.0

        d = np.sign(f[k]) - s[k]                                        # change in spin state
        s[k] = np.sign(f[k])                                            # Update spin signs
        for j in range(n):                                              # Update the field for every spin
            f[j] += J[k, j] * d
        t += t_next
    return s, t, max_events


@njit
def solver_event(J, h, v, n_iter, sizes, clusters, max_events):
    """
    Outer iteration loop for the event solver.

    Run the event solver, update the best state and perturb with a decreasing cluster size.
    """
    history = np.empty((n_iter, 3))                     # energy, best energy, settle time
    states = np.zeros((n_iter, len(h)))                 # end state of every run
    best_sigma = np.zeros(len(h))
    best_E = np.inf
    for it in range(n_iter):
        sigma, T, _ = analog_event(J, h, v, max_events)
        states[it] = sigma
        E = -0.5 * sigma @ J @ sigma - h @ sigma
        if E < best_E:
            best_sigma, best_E = sigma.copy(), E
        history[it, 0] = E
        history[it, 1] = best_E
        history[it, 2] = T
        v = best_sigma.copy()
        for c in clusters[it, :sizes[it]]:
            v[c] *= -1
    return best_sigma, best_E, history, states


def multiplicative_event(J, h, n_iter, init_size, end_size, seed, max_events=10**6):
    rng = np.random.default_rng(seed)
    n = len(h)
    v = rng.uniform(-1, 1, n)
    sizes = np.array([cluster_size(it, n_iter, init_size, end_size) for it in range(n_iter)])
    clusters = np.zeros((n_iter, n), dtype=np.int64)
    for it in range(n_iter):
        clusters[it, :sizes[it]] = rng.choice(n, size=sizes[it], replace=False)
    return solver_event(J, h, v, n_iter, sizes, clusters, max_events)


# ---------- Lagrange solver ----------

@njit
def energy(s, J, h):
    return -0.5 * s.T @ J @ s - h @ s


@njit
def qkp_penalty(a, sigma, C):
    return a.T @ ((1 + sigma) / 2) - C


@njit
def solver_lagrange(J_obj, h_obj, J_pen, h_pen, a, C, P, eta, v, sizes, clusters, max_events, update, beta1, beta2):
    """
    Args:
        - J_obj, h_obj: f in Ising form
        - J_pen, h_pen: ||g||^2 in Ising form
        - a, C: the extended A and C
        - P, eta: P and n of the algorithm
        - v, sizes, cluster: multiplicative solver's start voltages and cluster size
    """
    n_iter = len(sizes)                                 # the K iterations of the algorithm
    n = len(h_obj)
    J = J_obj + P * J_pen                               # Compute the J
    states = np.zeros((n_iter, n))                      # end state of every run
    lams = np.zeros(n_iter)                             # lambdas for every run
    winner = np.zeros(n)                                # winner: the state we perturb on next iteration
    lam = 0.0                                           # Current lambda (lambda_0 = 0 from algorithm)
    m = 0.0                                             # running average of g (momentum)
    s2 = 0.0                                            # running average of g**2 (adam)
    for it in range(n_iter):                            # Do K iterations
        h = h_obj + P * h_pen - lam * a / 2             # Add the Lagrange term (only affects bias)
        sigma, T, _ = analog_event(J, h, v, max_events) # Run the typical solver
        states[it] = sigma
        lams[it] = lam

        # winner = sigma.copy()
        if it == 0:
            winner = sigma.copy()
        elif energy(sigma, J, h) < energy(winner, J, h):
            winner = sigma.copy()

        g = qkp_penalty(a, sigma, C)
        if update == 'momentum':
            m = beta1 * m + (1 - beta1) * g
            lam += eta * m
        elif update == 'momentum_bc':
            m = beta1 * m + (1 - beta1) * g
            lam += eta * m / (1 - beta1**(it + 1))
        elif update == 'adam':
            m = beta1 * m + (1 - beta1) * g
            s2 = beta2 * s2 + (1 - beta2) * g**2
            lam += eta * (m / (1 - beta1**(it + 1))) / (np.sqrt(s2 / (1 - beta2**(it + 1))) + 1e-8)
        else:
            lam += eta * g

        v = winner.copy()
        for c in clusters[it, :sizes[it]]:
            v[c] *= -1
    return states, lams


def multiplicative_lagrange(J_obj, h_obj, J_pen, h_pen, a, C, P, eta, n_iter, init_size, end_size, seed, max_events=10**6, exponent=0.0, update='plain', beta1=0.9, beta2=0.999):
    rng = np.random.default_rng(seed)
    n = len(h_obj)
    v = rng.uniform(-1, 1, n)

    if exponent != 0.0:
        sizes = np.array([cluster_size_exp(it, n_iter, init_size, end_size, exponent) for it in range(n_iter)])
    else:
        sizes = np.array([cluster_size(it, n_iter, init_size, end_size) for it in range(n_iter)])   # List of cluster sizes e.g. [10, 5, 1]

    clusters = np.zeros((n_iter, n), dtype=np.int64)                                            # Array of which spins in each cluster
    for it in range(n_iter):
        clusters[it, :sizes[it]] = rng.choice(n, size=sizes[it], replace=False)
    return solver_lagrange(J_obj, h_obj, J_pen, h_pen, a, C, P, eta, v, sizes, clusters, max_events, update, beta1, beta2)
