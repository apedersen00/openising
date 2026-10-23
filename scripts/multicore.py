import numpy as np
import networkx as nx
from numba import njit


# ---------- Problem split ----------

def split_couplings(J, labels, near_cores):
    """
    Split J by where each coupling lives in hardware.

    Args:
        - J: symmetric coupling matrix, zero diagonal
        - labels: core of every spin
        - near_cores: near_cores[p, q] is True when cores p and q sit side by side

    Returns J_in (inside a core, the crossbar), J_near (neighbouring cores, edge couplers) and
    J_far (all other cores, SRAM + bias accumulator). The three add up to J.
    """
    same = labels[:, None] == labels[None, :]
    near = near_cores[labels[:, None], labels[None, :]] & ~same
    far = ~same & ~near
    return np.where(same, J, 0.0), np.where(near, J, 0.0), np.where(far, J, 0.0)


def kl_partition(J, K, seed=0):
    """
    K balanced cores by recursive Kernighan-Lin bisection on |J| (keeps strongly coupled spins together).
    """
    rng = np.random.default_rng(seed)
    A = np.abs(J)
    labels = np.zeros(len(J), dtype=np.int64)
    stack = [(np.arange(len(J)), K, 0)]                                 # (spins, number of cores to make, first label)
    while stack:
        nodes, k, base = stack.pop()
        if k == 1:
            labels[nodes] = base
            continue
        k1 = k // 2
        n1 = int(round(len(nodes) * k1 / k))                            # uneven splits, so any K works
        perm = rng.permutation(len(nodes))
        a, b = nx.algorithms.community.kernighan_lin_bisection(
            nx.from_numpy_array(A[np.ix_(nodes, nodes)]),
            partition=(set(perm[:n1].tolist()), set(perm[n1:].tolist())), weight='weight',
            seed=int(rng.integers(1 << 30)))
        if len(a) != n1:                                                # networkx may return the sides swapped
            a, b = b, a
        stack.append((nodes[sorted(a)], k1, base))
        stack.append((nodes[sorted(b)], k - k1, base + k1))
    return labels


def grid_near(rows, cols):
    """
    near[p, q] for cores on a rows x cols torus grid (core p at row p // cols, column p % cols).
    """
    r, c = np.divmod(np.arange(rows * cols), cols)
    dr = np.abs(r[:, None] - r[None, :])
    dc = np.abs(c[:, None] - c[None, :])
    dr = np.minimum(dr, rows - dr)                                      # distances wrap around
    dc = np.minimum(dc, cols - dc)
    return (dr + dc) == 1


def place_cores(J, labels, near, seed=0, iters=20000):
    """
    Put the cores on the grid slots so that strongly coupled cores sit side by side (simulated annealing on swaps).

    Args:
        - near: slot adjacency from grid_near (at least as many slots as cores)

    Returns new labels where core p sits in grid slot p, so near can be used directly.
    """
    rng = np.random.default_rng(seed)
    K, S = labels.max() + 1, len(near)
    A = np.abs(J)
    M = np.zeros((K, K))                                                # coupling weight between every pair of cores
    for p in range(K):
        for q in range(K):
            if p != q:
                M[p, q] = A[np.ix_(labels == p, labels == q)].sum()
    slot = rng.permutation(S)[:K]                                       # slot of every core
    occ = -np.ones(S, dtype=np.int64)                                   # core in every slot, -1 = empty
    occ[slot] = np.arange(K)

    def score():
        return sum(M[p, occ[t]] for p in range(K) for t in np.flatnonzero(near[slot[p]]) if occ[t] >= 0) / 2

    cur = score()
    T0 = M.max() + 1e-9
    for it in range(iters):
        T = T0 * (1 - it / iters) + 1e-9
        a, b = rng.integers(S, size=2)
        p, q = occ[a], occ[b]
        if p < 0 and q < 0:
            continue
        occ[a], occ[b] = q, p
        if p >= 0:
            slot[p] = b
        if q >= 0:
            slot[q] = a
        new = score()
        if new >= cur or rng.random() < np.exp((new - cur) / T):
            cur = new
        else:                                                           # undo the swap
            occ[a], occ[b] = p, q
            if p >= 0:
                slot[p] = a
            if q >= 0:
                slot[q] = b
    return slot[labels]


# ---------- Event solver with delays ----------

@njit
def heap_push(ht, hk, hd, hc, size, t, k, d, c):
    """
    Add a message (arrival time t, spin k, change d, class c: 0 near, 1 far) to the binary heap; earliest on top.
    """
    i = size
    ht[i] = t; hk[i] = k; hd[i] = d; hc[i] = c
    while i > 0:                                                        # move up while earlier than the parent
        p = (i - 1) // 2
        if ht[p] <= ht[i]:
            break
        ht[p], ht[i] = ht[i], ht[p]; hk[p], hk[i] = hk[i], hk[p]; hd[p], hd[i] = hd[i], hd[p]; hc[p], hc[i] = hc[i], hc[p]
        i = p
    return size + 1


@njit
def heap_pop(ht, hk, hd, hc, size):
    """
    Remove and return the earliest message.
    """
    k, d, c = hk[0], hd[0], hc[0]
    size -= 1
    ht[0] = ht[size]; hk[0] = hk[size]; hd[0] = hd[size]; hc[0] = hc[size]
    i = 0
    while True:                                                         # move the last one down to its place
        l, r, m = 2 * i + 1, 2 * i + 2, i
        if l < size and ht[l] < ht[m]:
            m = l
        if r < size and ht[r] < ht[m]:
            m = r
        if m == i:
            break
        ht[m], ht[i] = ht[i], ht[m]; hk[m], hk[i] = hk[i], hk[m]; hd[m], hd[i] = hd[i], hd[m]; hc[m], hc[i] = hc[i], hc[m]
        i = m
    return k, d, c, size


@njit
def analog_event_delayed(J_in, J_near, J_far, h, v, tau_near, tau_far, deg_near, deg_far, max_events, jitter, t_max):
    """
    Exact event-driven multiplicative solver where couplings between cores arrive late.

    Inside a core a flip changes the partners' fields at once (crossbar). To a neighbouring core it arrives
    tau_near later (edge couplers), to every other core tau_far later (flip message + SRAM read + add + bias DAC).
    A delay of 0 is applied at once, so tau_near = tau_far = 0 is scripts/lagrange.analog_event exactly.

    Args:
        - J_in, J_near, J_far: the split couplings (split_couplings)
        - h: biases
        - v: start voltages (changed in place)
        - tau_near, tau_far: delays
        - deg_near, deg_far: number of near / far partners of every spin (for the counters)
        - max_events: cap on flips + arrivals (also the queue length)
        - jitter: every message waits tau * (1 + jitter * U(-1, 1)); breaks the lock-step that makes pairs oscillate
        - t_max: readout time; the run stops there even if it has not settled (np.inf = no timeout)

    Returns the spins, the end time, whether the run settled, and the counters
    [flips, near messages, far messages, near cell updates, far accumulator adds].
    """
    n = len(h)
    s = np.sign(v)
    for i in range(n):                                                  # a spin exactly at the threshold counts as +1
        if s[i] == 0:
            s[i] = 1.0
    f = J_in @ s + J_near @ s + J_far @ s + h                           # every core knows the start state (shared kicks)

    ht = np.empty(max_events); hk = np.empty(max_events, np.int64)      # messages in flight: arrival time, spin,
    hd = np.empty(max_events); hc = np.empty(max_events, np.int64)      # change, class (0 near, 1 far)
    size = 0
    counts = np.zeros(5)                                                # flips, near msgs, far msgs, near updates, far adds

    t = 0.0
    for event in range(max_events):
        dt_cross = np.inf                                               # time until the next spin reaches zero
        k = -1
        for i in range(n):
            if v[i] * f[i] < 0 or (v[i] == 0.0 and s[i] * f[i] < 0):
                t_i = -v[i] / f[i]
                if t_i < dt_cross:
                    dt_cross = t_i
                    k = i
        dt_msg = ht[0] - t if size > 0 else np.inf                      # time until the next message arrives
        if dt_cross == np.inf and dt_msg == np.inf:                     # nothing moves and nothing in flight: settled
            return s, t, True, counts

        dt = min(dt_cross, dt_msg)
        readout = t_max - t <= dt                                       # the readout comes first: stop there
        if readout:
            dt = t_max - t
        for i in range(n):                                              # move every voltage up to the next event
            reached = v[i] * f[i] < 0 and -v[i] / f[i] <= dt
            v[i] = min(1.0, max(-1.0, v[i] + dt * f[i]))
            if reached:
                v[i] = 0.0
        t += dt
        if readout:
            return s, t, False, counts

        if dt_msg <= dt_cross:                                          # a message arrives (messages win ties)
            kk, d, c, size = heap_pop(ht, hk, hd, hc, size)
            if c == 0:                                                  # edge couplers see the new state
                for j in range(n):
                    f[j] += J_near[kk, j] * d
                counts[1] += 1
                counts[3] += deg_near[kk]
            else:                                                       # the accumulators add
                for j in range(n):
                    f[j] += J_far[kk, j] * d
                counts[2] += 1
                counts[4] += deg_far[kk]
        else:                                                           # spin k flips
            d = np.sign(f[k]) - s[k]
            s[k] = np.sign(f[k])
            counts[0] += 1
            for j in range(n):                                          # inside the core: at once (J symmetric, so row k = column k)
                f[j] += J_in[k, j] * d
            if deg_near[k] > 0:
                if tau_near == 0.0:
                    for j in range(n):
                        f[j] += J_near[k, j] * d
                    counts[1] += 1
                    counts[3] += deg_near[k]
                else:
                    delay = tau_near * (1 + jitter * (2 * np.random.random() - 1))
                    size = heap_push(ht, hk, hd, hc, size, t + delay, k, d, 0)
            if deg_far[k] > 0:
                if tau_far == 0.0:
                    for j in range(n):
                        f[j] += J_far[k, j] * d
                    counts[2] += 1
                    counts[4] += deg_far[k]
                else:
                    delay = tau_far * (1 + jitter * (2 * np.random.random() - 1))
                    size = heap_push(ht, hk, hd, hc, size, t + delay, k, d, 1)
    return s, t, False, counts


@njit
def solver_delayed(J_in, J_near, J_far, h, v, tau_near, tau_far, sizes, clusters, max_events, jitter, t_max, seed):
    """
    Kicked restarts around analog_event_delayed: keep the best state, flip a cluster of it, run again.

    Returns the best state and energy, and per run: energy, best energy, end time, settled, and the 5 counters.
    """
    np.random.seed(seed)                                                # numba's own generator, for the jitter
    n_iter = len(sizes)
    J = J_in + J_near + J_far
    deg_near = (J_near != 0).sum(axis=1).astype(np.float64)
    deg_far = (J_far != 0).sum(axis=1).astype(np.float64)
    history = np.zeros((n_iter, 9))                                     # E, best E, time, settled, counters
    best_s = np.zeros(len(h))
    best_E = np.inf
    for it in range(n_iter):
        s, T, settled, counts = analog_event_delayed(J_in, J_near, J_far, h, v, tau_near, tau_far, deg_near, deg_far,
                                                   max_events, jitter, t_max)
        E = -0.5 * s @ J @ s - h @ s                                    # always the full problem
        if E < best_E:
            best_s, best_E = s.copy(), E
        history[it, 0] = E
        history[it, 1] = best_E
        history[it, 2] = T
        history[it, 3] = settled
        history[it, 4:] = counts
        v = best_s.copy()
        for c in clusters[it, :sizes[it]]:
            v[c] *= -1
    return best_s, best_E, history


def multicore_delayed(J_in, J_near, J_far, h, tau_near, tau_far, n_iter, kick0, seed, jitter=0.5, t_max=np.inf,
                      max_events=None):
    """
    Args:
        - kick0: first kick as a fraction of the spins, shrinking linearly to 1 spin
        - jitter: random spread of every message delay (0.5: between 0.5 tau and 1.5 tau)
        - t_max: readout time per run (np.inf: wait until settled)
    """
    rng = np.random.default_rng(seed)
    n = len(h)
    max_events = 100 * n if max_events is None else max_events          # flips + messages per run
    v = rng.uniform(-1, 1, n)
    sizes = np.round(np.linspace(max(1, kick0 * n), 1, n_iter)).astype(np.int64)
    clusters = np.zeros((n_iter, n), dtype=np.int64)
    for it in range(n_iter):
        clusters[it, :sizes[it]] = rng.choice(n, size=sizes[it], replace=False)
    return solver_delayed(J_in, J_near, J_far, h, v, tau_near, tau_far, sizes, clusters, max_events, jitter, t_max,
                          seed)
