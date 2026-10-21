import numpy as np
from numba import njit


def qubo_to_ising(Q):
    J = -0.5 * Q
    np.fill_diagonal(J, 0)
    h = -0.5 * Q.sum(axis=1)
    c = 0.25 * Q.sum() + 0.25 * np.trace(Q)
    return J, h, c


def penalty_qubo(A, b):
    """
    ||Az - b||^2 as a QUBO, without the constant b^T b. z is binary, so z_i^2 = z_i puts the linear term on the diagonal.
    """
    return A.T @ A - 2 * np.diag(A.T @ b)


def max_field(J, h):
    return np.max(np.abs(J).sum(axis=1) + np.abs(h))


def paper_rule(W, A, b):
    """
    P and n from Delacour 2025: P = 2dN and n = 20.

    The paper rescales the objective so its largest value is 1,
    and the constraints so the largest of A and b is 1.

    Args:
        - W: objective matrix (the profit matrix for QKP)
        - A, b: constraints A z = b, slack bits included
    """
    d = np.mean(W[~np.eye(len(W), dtype=bool)] != 0)  # density of all couplings
    scale = np.abs(W).max() / max(np.abs(A).max(), np.abs(b).max())**2   # Rescale
    return 2 * d * A.shape[1] * scale, 20 * scale


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
def solver_lagrange(J_obj, h_obj, J_pen, h_pen, A, b, P, eta, v, sizes, clusters, max_events, update, beta1, beta2):
    """
    Args:
        - J_obj, h_obj: f in Ising form
        - J_pen, h_pen: ||g||^2 in Ising form
        - A, b: constraints A z = b, slack bits included
        - P, eta: P and n of the algorithm
        - v, sizes, cluster: multiplicative solver's start voltages and cluster size
    """
    n_iter = len(sizes)                                 # the K iterations of the algorithm
    n = len(h_obj)
    J = J_obj + P * J_pen                               # Compute the J
    states = np.zeros((n_iter, n))                      # end state of every run
    lams = np.zeros((n_iter, len(b)))                   # lambdas for every run, one per constraint
    winner = np.zeros(n)                                # winner: the state we perturb on next iteration
    lam = np.zeros(len(b))                              # Current lambdas (lambda_0 = 0 from algorithm)
    m = np.zeros(len(b))                                # running average of g (momentum)
    s2 = np.zeros(len(b))                               # running average of g**2 (adam)
    for it in range(n_iter):                            # Do K iterations
        h = h_obj + P * h_pen - lam @ A / 2             # Add the Lagrange term (only affects bias)
        sigma, T, _ = analog_event(J, h, v, max_events) # Run the typical solver
        states[it] = sigma
        lams[it] = lam

        # winner = sigma.copy()
        if it == 0:
            winner = sigma.copy()
        elif energy(sigma, J, h) < energy(winner, J, h):
            winner = sigma.copy()

        g = A @ ((1 + sigma) / 2) - b                   # constraint violation, one per constraint
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


def multiplicative_lagrange(J_obj, h_obj, J_pen, h_pen, A, b, P, eta, n_iter, init_size, end_size, seed, max_events=10**6, exponent=0.0, update='plain', beta1=0.9, beta2=0.999):
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
    return solver_lagrange(J_obj, h_obj, J_pen, h_pen, A, b, P, eta, v, sizes, clusters, max_events, update, beta1, beta2)
