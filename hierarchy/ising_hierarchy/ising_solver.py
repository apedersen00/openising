"""
Event-based multiplicative solver, all compiled with numba.

Refactored from Andreas Pedersen's scripts/lagrange.py (analog_event, solver_event, multiplicative_event)
and ising/solvers/Multiplicative_event.py (openising fork).

Convention: J symmetric with zero diagonal, both float64. E(s) = -1/2 s^T J s - h^T s, local fields f = J s + h.

- settle: one iteration, the analog run from some voltages until no spin heads for 0, or until max_time
- run_iterations: n_iter runs of settle. Keep the best spins, kick a random cluster of them, settle again
- solve: seeds numba's random generator, draws the start voltages, then run_iterations
- sweep: solve for several seeds (plain Python)
"""
import numpy as np
from numba import njit


@njit(cache=True)
def energy(s, J, h):
    return -0.5 * s @ J @ s - h @ s


@njit(cache=True)
def cluster_size(it, n_iter, init_size, end_size, exponent=3.0):
    """
    Number of spins kicked after run it: a linear decay from init_size to end_size if exponent = 0,
    exponential otherwise (needs end_size >= 1). Same formulas as cluster_size and cluster_size_exp of lagrange.py.
    """
    if n_iter < 2:
        return init_size
    if exponent == 0.0:
        return int(np.rint(init_size + (end_size - init_size) * it / (n_iter - 1)))
    decay = ((end_size - 1) / init_size) ** (it * exponent / (n_iter - 1))
    return int(np.floor(decay * (init_size - end_size) + end_size))


@njit(cache=True)
def _time_to_zero(v, s, f):
    """
    Time until a voltage reaches 0, np.inf if it does not head there.

    A spin heads for 0 if its voltage moves towards it (v*f < 0), or if it sits at 0 with its field against its sign.
    """
    if v * f < 0 or (v == 0.0 and s * f < 0):
        return -v / f
    return np.inf


@njit(cache=True)
def _events(J, v, s, f, max_time):
    """
    The event loop, from voltages v, their signs s and the fields f of those signs. Changes v, s and f in place.

    Returns:
        time (max_time if the run was cut), number of events
    """
    n = len(v)
    t_next = np.inf                                                     # time until the next crossing
    k = -1                                                              # spin of the next crossing, -1 if none
    for i in range(n):
        t_i = _time_to_zero(v[i], s[i], f[i])
        if t_i < t_next:
            t_next, k = t_i, i

    t = 0.0
    n_events = 0
    while k >= 0:                                                       # some spin still heads for 0
        if t + t_next > max_time:                                       # time is up before the next flip
            return max_time, n_events
        d = np.sign(f[k]) - s[k]                                        # change of spin k: +-2
        s[k] = np.sign(f[k])
        t += t_next
        n_events += 1

        dt = t_next                                                     # in one pass: move every voltage to the
        k_flip = k                                                      # crossing, update the fields, and find
        t_next, k = np.inf, -1                                          # the next crossing
        for i in range(n):
            reached = v[i] * f[i] < 0 and -v[i] / f[i] <= dt            # crosses 0 within dt, ties included
            v[i] = min(1.0, max(-1.0, v[i] + dt * f[i]))                # straight line, clipped at the rails
            if reached:
                v[i] = 0.0                                              # exactly 0, not 1e-17 on either side
            f[i] += J[k_flip, i] * d                                    # only row k_flip of J changes the fields
            t_i = _time_to_zero(v[i], s[i], f[i])
            if t_i < t_next:
                t_next, k = t_i, i
    return t, n_events


@njit(cache=True)
def settle(J, h, v0, max_time=np.inf):
    """
    One iteration: the event-based analog run from voltages v0, until no spin heads for 0 or until max_time.

    Args:
        - J, h: the Ising model
        - v0: start voltages in [-1, 1], one per spin (not changed)
        - max_time: time budget of the run, np.inf to run until it settles

    Returns:
        spins s, time t (max_time if the run was cut), number of events
    """
    v = v0.astype(np.float64)                                           # astype copies
    s = np.where(v >= 0, 1.0, -1.0)                                     # a voltage at exactly 0 counts as +1
    f = J @ s + h
    t, n_events = _events(J, v, s, f, max_time)
    return s, t, n_events


@njit(cache=True)
def run_iterations(J, h, v0, n_iter, init_size, end_size, exponent=0.0, max_time=np.inf):
    """
    Multiple runs: settle from v0, keep the best spins, kick a random cluster of them, settle again, n_iter times.

    The kicks come from numba's random generator, seed it first (solve does).
    Only the first run computes J @ s: after a kick of size m the fields are updated in m * n operations.

    Args:
        - J, h: the Ising model
        - v0: start voltages of the first run (not changed)
        - n_iter: number of runs
        - init_size, end_size: number of spins kicked after the first and the last run
        - exponent: 0 for a linear decay of the cluster size, > 0 for exponential
        - max_time: time budget of every run

    Returns:
        best spins, best energy, history (n_iter, 4): energy, best energy, time and number of events of every run
    """
    n = len(h)
    history = np.zeros((n_iter, 4))
    v = v0.astype(np.float64)
    s = np.where(v >= 0, 1.0, -1.0)
    f = J @ s + h
    best_s, best_f, best_E = s.copy(), f.copy(), np.inf
    for it in range(n_iter):
        t, n_events = _events(J, v, s, f, max_time)
        E = -0.5 * (s @ f + h @ s)                                      # = -1/2 s J s - h s, from the fields
        if E < best_E:
            best_s, best_f, best_E = s.copy(), f.copy(), E
        history[it, 0] = E
        history[it, 1] = best_E
        history[it, 2] = t
        history[it, 3] = n_events
        if it == n_iter - 1:
            break

        size = min(max(cluster_size(it, n_iter, init_size, end_size, exponent), 0), n)
        v, s, f = best_s.copy(), best_s.copy(), best_f.copy()          # next run starts from the best, at the rails
        for c in np.random.choice(n, size, replace=False):              # kick: flip a random cluster
            v[c] = -v[c]
            s[c] = -s[c]
            for j in range(n):
                f[j] += 2.0 * s[c] * J[c, j]
    return best_s, best_E, history


@njit(cache=True)
def solve(J, h, n_iter, init_size=0.9, end_size=0.1, seed=42, v0=None, exponent=0.0, max_time=np.inf):
    """
    Settle, keep the best, kick: n_iter runs from random start voltages, or from v0.

    Args:
        - J, h: the Ising model
        - n_iter: number of runs
        - init_size, end_size: fraction of the spins kicked after the first and the last run (at least 1 spin)
        - seed: seeds the start voltages and the kicks
        - v0: start voltages, e.g. a previous state. Random in [-1, 1] if None
        - exponent: 0 for a linear decay of the cluster size, > 0 for exponential
        - max_time: time budget of every run

    Returns:
        best spins, best energy, history (n_iter, 4): energy, best energy, time and number of events of every run
    """
    np.random.seed(seed)                                                # numba's generator, not numpy's
    v = np.random.uniform(-1.0, 1.0, len(h))                            # always drawn, so v0 does not change the kicks
    if v0 is not None:
        v = v0.astype(np.float64)
    n = len(h)
    init_size_abs = max(1, int(init_size * n))                          # fractions -> numbers of spins
    end_size_abs = max(1, int(end_size * n))
    return run_iterations(J, h, v, n_iter, init_size_abs, end_size_abs, exponent, max_time)


def sweep(J, h, seeds, n_iter, v0=None, max_time=np.inf):
    """
    Run solve for every seed, with the default cluster sizes of solve.

    Args:
        - seeds: list of seeds

    Returns:
        best spins (seed, n), best energies (seed,), histories (seed, n_iter, 4)
    """
    J = np.ascontiguousarray(J, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    best_s = np.zeros((len(seeds), len(h)))
    best_E = np.zeros(len(seeds))
    history = np.zeros((len(seeds), n_iter, 4))
    for i, seed in enumerate(seeds):
        best_s[i], best_E[i], history[i] = solve(J, h, n_iter, seed=seed, v0=v0, max_time=max_time)
    return best_s, best_E, history
