import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from numba import njit
    import marimo as mo

    return Path, mo, njit, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    System energy:

    $H(\sigma) = - \frac{1}{2} \sigma^T J \sigma - h^T \sigma + c$

    Derivative:

    $-\frac{\partial H}{\partial \sigma} = J \sigma + h$

    Sign intuition

    - $J_{ij} > 0$: the two spins want to align
    - $J_{ij} < 0$: the two spins want to be opposite
    - $h_i > 0$: $\sigma_i$ wants to be +1

    ## Max Cut

    ```
      2
    /   \
    1---3
    |   |
    0---4
    ```
    """)
    return


@app.cell
def _(np):
    N = 5
    edges = [
        (0, 1, 1.0),
        (0, 4, 1.0),
        (1, 2, 1.0),
        (1, 3, 1.0),
        (2, 3, 1.0),
        (3, 4, 1.0),
    ]

    # Generate weight matrix
    W = np.zeros((N, N))
    for _i, _j, _w in edges:
        W[_i, _j] = _w
        W[_j, _i] = _w

    print('Weight Matrix')
    for _row in range(N):
        print(W[_row, :])
    return N, W, edges


@app.cell
def _(N, W, np):
    def maxcut_to_ising(W):
        J = -W / 2
        h = np.zeros(len(W))
        return J, h

    def energy(s, J, h):
        return -0.5 * s.T @ J @ s - h @ s

    def cut_value(s, edges):
        tot = 0
        for _i, _j, _w in edges:
            if s[_i] != s[_j]:
                tot += _w
        return tot

    def lower_bound(J, h):
        return -0.5 * np.abs(J).sum() - np.abs(h).sum()

    J, h = maxcut_to_ising(W)
    print('J Matrix')
    for _row in range(N):
        print(J[_row, :])

    print('\nBias vector (h)')
    print(h)

    print('\nLower bound for H')
    print(lower_bound(J, h))
    return J, cut_value, energy, h, lower_bound


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Brute force all possible solutions
    """)
    return


@app.cell
def _(J, N, cut_value, edges, energy, h, np, pd):
    # There are 5 spins for a total of 2^5 possible combinations
    # Iterate stupidly over the array and fill with binary counting
    # Iterate col again to replace 0 with -1

    possible_states = np.zeros((2**N, N))
    for _row in range(2**N):
        bin_val = format(_row, f'0{N}b')
        for _col in range(N):
            possible_states[_row, _col] = bin_val[_col]
        for _col in range(N):
            if possible_states[_row, _col] == 0:
                possible_states[_row, _col] = -1

    # Compute energy for every single possible combination
    possible_energies = np.zeros((2**N, 1))
    for _i in range(2**N):
        possible_energies[_i] = energy(possible_states[_i, :], J, h)

    # Compute cut value for every single possible combination
    possible_cuts = np.zeros((2**N, 1))
    for _i in range(2**N):
        possible_cuts[_i] = cut_value(possible_states[_i, :], edges)

    pd.DataFrame({
        "state": list(possible_states),
        "energy": possible_energies[:, 0],
        "cut": possible_cuts[:, 0],
    })
    return possible_cuts, possible_energies, possible_states


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Local Field (gradient) and Minima

    For each spin, the "local field" is the total effect from its neighbors on it:

    $f_i = \sum_j J_{ij} \sigma_j + h_i$

    The above is the same as:

    $-\frac{\partial H}{\partial \sigma} = J \sigma + h$

    Flipping $\sigma_i$ changes the energy by:

    $\Delta H_i = 2 \sigma_i f_i$
    """)
    return


@app.cell
def _(J, N, h, np, pd, possible_cuts, possible_energies, possible_states):
    def field(J, s, h):
        """
        Returns the local field that pushes on a node i.e. the gradient.
        """
        return J @ s + h

    local_min = np.zeros(2**N, dtype=bool)
    for _row in range(2**N):
        _s = possible_states[_row]                      # one possible state
        grad = field(J, _s, h)                          # local field for each spin
        local_min[_row] = np.all(2 * _s * grad >= 0)    # energy gradient by flipping the spin


    print('There exists minima (grad = 0) for non-optimal solutions.')
    pd.DataFrame({
        "state": list(possible_states),
        "energy": possible_energies[:, 0],
        "cut": possible_cuts[:, 0],
        "local_min": local_min 
    })
    return (field,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Analog Solvers

    Sigmas are continuous voltages. They are updated based on the local field affecting them.

    Below is a "dumb" solver i.e. it settles into the nearest minimum with a gradient of 0.
    """)
    return


@app.cell
def _(J, N, cut_value, edges, field, h, np, plt):
    def analog_run(J, h, v0, dt, n_steps):
        v = v0.copy()
        trajectory = [v.copy()]             # Store initial states
        for _ in range(n_steps):            # Update voltages
            f = field(J, np.sign(v), h)     # Change in state depends on local field
            v = np.clip(v + dt * f, -1, 1)  # Update the voltage and clip to -1, +1
            trajectory.append(v.copy())     # Append updated voltages
        return np.array(trajectory)

    SEED = 0
    v0 = np.random.default_rng(SEED).uniform(-1, 1, N)  # Starting voltages
    dt = 0.05                                           # Voltage change rate
    trajectory = analog_run(J, h, v0, dt, 200)
    sigma = np.sign(trajectory[-1])
    sigma, cut_value(sigma, edges)

    _fig, _ax = plt.subplots()
    for _k in range(N):
        _ax.plot(trajectory[:, _k], label=f's{_k} = {sigma[_k]}', linewidth=2)
    _ax.set_xlabel('step')
    _ax.set_ylabel('voltage')
    _ax.legend(loc='upper right')
    _fig
    return SEED, analog_run, dt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Iterative Analog Solver

    1. **Settle:** a single run of a dumb solver
    2. **Keep the best:** is these $\sigma$ has the lowest energy seen, keep them
    3. **Kick:** start the next run from the best known start with a random _cluster_ of spins flipped
    """)
    return


@app.cell
def _(J, N, SEED, analog_run, dt, energy, h, lower_bound, np, plt):
    def cluster_size(it, n_iter, init_size, end_size):
        """
        Linear decay from init_size to end_size over n_iter.
        """
        steps = np.linspace(init_size, end_size, n_iter)
        return round(steps[it])

    def multiplicative_solver(J, h, n_iter, n_steps, dt, init_size, end_size, seed):
        rng = np.random.default_rng(seed)       # Setup RNG
        n = len(h)                              # Number of spins
        v = rng.uniform(-1, 1, n)               # Initial voltages for the analog run
        best_sigma, best_E = None, np.inf       # Store best sigmas and their corresponding energy
        history = []
        traj_hist = []

        for it in range(n_iter):
            traj = analog_run(J, h, v, dt, n_steps)                 # Analog run
            traj_hist.append(traj)                                  # Store trajectories for plotting
            sigma = np.sign(traj[-1])                               # Store final sigmas from the analog run
            E = energy(sigma, J, h)                                 # Compute the energy for this run
            if E < best_E:                                          # If the new run has better energy, save it
                best_sigma, best_E = sigma.copy(), E
            history.append((E, best_E))
            k = cluster_size(it, n_iter, init_size, end_size)       # Get cluster size i.e. number of spins to flip
            cluster = rng.choice(n, size=k, replace=False)          # Choose k random spins
            v = best_sigma.copy()                                   # Copy best sigmas so far
            v[cluster] *= -1                                        # Flip k random spins
        return best_sigma, best_E, np.array(history), np.array(traj_hist)

    best_sigma, best_E, history, traj_hist = multiplicative_solver(
        J, h, n_iter=3, n_steps=200, dt=dt, init_size=2, end_size=1, seed=SEED
    )

    _lower_bound = lower_bound(J, h)

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4))

    _ax[0].plot(history[:, 0], label='energy of this run', linewidth=2)
    _ax[0].plot(history[:, 1], label='best energy', linewidth=2)
    _ax[0].axhline(_lower_bound, label='lower bound', linewidth=2, color='black', linestyle='--')
    _ax[0].set_xlabel('iteration')
    _ax[0].set_ylabel('energy')
    _ax[0].legend()

    traj_hist = traj_hist.reshape(-1, N)
    for _k in range(N):
        _ax[1].plot(traj_hist[:, _k], label=f's{_k}', linewidth=2)
    _ax[1].set_xlabel('step')
    _ax[1].set_ylabel('voltage')
    _ax[1].legend(loc='upper right')
    _fig.suptitle('5-node Max-Cut')

    _fig
    return cluster_size, multiplicative_solver


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## QUBO

    1. Formulate the COP as a *Quadratic Unconstrained Binary Optimiation* problem:

    $f(\boldsymbol{x}) = \sum_{i,j} a_{ij}x_ix_j + \sum_i b_i x_i + c$

    - $x_i \in \{0, 1\}$: the binary variables
    - $a_{ij}, b_i, c \in \mathbb{R}$: the weight matrix elements, the weight vector elements and a constant, which is usually dropped

    2. Substitute $\boldsymbol{x} = (1 + \boldsymbol{\sigma})/2$, which turns the QUBO into the Ising Hamiltonian with $J_{ij} = -a_{ij}/2$ and $h_i = -\left(b_i/2 + \sum_j a_{ij}/2\right)$

    ### Max-Cut

    An edge is cut when exactly one end is 1 i.e. when $x_i \oplus x_j = x_i + x_j - 2 x_i x_j = 1$:

    $-cut = \sum_{i<j} w_{ij} (2x_i x_j - x_i - x_j) = x^T W x - \sum_i d_i x_i$

    - $d_i$ = `W.sum(axis=1)` i.e. the weighte degree of node.
    """)
    return


@app.cell
def _(J, W, np):
    def qubo_to_ising(Q):
        J = -0.5 * Q
        np.fill_diagonal(J, 0)
        h = -0.5 * Q.sum(axis=1)
        c = 0.25 * Q.sum() + 0.25 * np.trace(Q)
        return J, h, c

    def maxcut_to_qubo(W):
        return W - np.diag(W.sum(axis=1))

    Q_mc = maxcut_to_qubo(W)
    J_q, h_q, c_q = qubo_to_ising(Q_mc)

    print("Is the Max-Cut QUBO the same as previously derived?")
    np.allclose(J_q, J), h_q, c_q
    return Q_mc, qubo_to_ising


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Penalties

    The solver has no concept of valid or invalid solutions, it just minimizes energy.

    A penalty for "choose exactly one node that can have value $x_i = 1$ for max-cut is defined.

    The penalty:

    $P(x) = (\sum_i x_i - k)^2$

    - Exactly 0 when the constraint is true
    - Is $\geq 1$ otherwise
    - Grows larger the further from the valid solution
    - $k$ is the number of nodes that must have $x_i = 1$

    Expanding:

    $$
    \begin{align*}
    P(x) = (\sum x_i - k)^2 &= (\sum x_i)^2 + k^2 - 2k \sum x_i \\
    &= \sum_i x_i^2 + 2 \sum_{i<j} x_i x_j + k^2 - 2k \sum x_i  \\
    &= \sum_i x_i + 2 \sum_{i<j} x_i x_j + k^2 - 2k \sum x_i    \\
    &= 2\sum_{i<j} x_i x_j + (1 - 2k) \sum_i x_i + k^2
    \end{align*}
    $$

    The penalty is of the form: $\text{number} \cdot x_i x_j + \text{number} \cdot x_i + \text{constant}$
    """)
    return


@app.cell
def _(
    N,
    Q_mc,
    SEED,
    cut_value,
    dt,
    edges,
    multiplicative_solver,
    np,
    plt,
    qubo_to_ising,
):
    k_target = 1
    Q_pen = np.ones((N, N)) - 2 * k_target * np.eye(N)  # Q_ij = 1, Q_ii = 1 - 2k

    J_pen, h_pen, c_pen = qubo_to_ising(Q_pen)
    c_pen = c_pen + k_target**2                         # Add k^2 to c

    lam = 3.0 # Penalty weight

    J_pq, h_pq, c_pq = qubo_to_ising(Q_mc + lam * Q_pen)
    c_pq = c_pq + lam * k_target ** 2

    # Run solver
    _best_sigma, _best_E, _history, _traj_hist = multiplicative_solver(
        J_pq, h_pq, n_iter=10, n_steps=200, dt=dt, init_size=2, end_size=1, seed=SEED
    )

    _x = (1 + _best_sigma) / 2
    print(f"x = {_x.astype(int)}, ones = {_x.sum():.0f} (target {k_target}), "
          f"cut = {cut_value(_best_sigma, edges)}, H = {_best_E + c_pq}")

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4))

    _ax[0].plot(_history[:, 0] + c_pq, label='energy of this run')
    _ax[0].plot(_history[:, 1] + c_pq, label='best energy')
    _ax[0].set_xlabel('iteration')
    _ax[0].set_ylabel('energy')
    _ax[0].legend()

    _traj_hist = _traj_hist.reshape(-1, N)
    for _k in range(N):
        _ax[1].plot(_traj_hist[:, _k], label=f's{_k}', linewidth=2)
    _ax[1].set_xlabel('step')
    _ax[1].set_ylabel('voltage')
    _ax[1].legend(loc='upper right')

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quadratic Knapsack

    $x_i = 1$ means "item $i$ goes in the knapsack". With $N$ items there are $N$ binary variables. The goal is maximum profit:

    $$
    \max_{x_i \in \{0, 1\}^N} \sum_{i < j} P_{ij} x_i x_j + \sum_{i=1}^N P_{ii} x_i \hspace{20pt} \text{s.t.} \sum_{i=1}^N w_i x_i \leq C
    $$

    - $P_{ii}$: the profit of item $i$ on its own. Only counts if the item is taken, $x_i = 1$.
    - $P_{ij}$: the extra profit for having both item $i$ and $j$. Only counts if both items are packed, $x_i = x_j = 1$.

    Each item has a weight $w_i$. To keep the items under the knapsack's capacity:
    $$
    \sum_{i=1}^N w_i x_i \leq C
    $$
    Using this in the same kind of penalty as the [[Traveling salesman problem|TSP]], $(\sum_i w_i x_i - C)^2$, ==would also penalize being under capacity==. The fix is to introduce the unused capacity:
    $$
    y = C - \sum_i w_i x_i
    $$
    The inequality then becomes an equality:
    $$
    \sum_i w_i x_i + y = C, \hspace{20pt} y \geq 0
    $$

    The slack is binarized as
    $$
    y = \sum_{k=0}^{B-1} 2^k y_k, \hspace{20pt} B = \lfloor \log_2 (C) + 1 \rfloor
    $$

    The expression in matrix form:

    $$
    \max_{x_i \in \{0, 1\}^N} x^T P x + Px
    $$

    The Q is then built:

    - $b \cdot x_i$ -> $Q_{ii} = b$ on the diagonal
    - $c \cdot x_i x_j$ counted once (i < j) -> $c/2$ in $Q_{ij}$ and $c/2$ in $Q_{ji}$
    """)
    return


@app.cell
def _(np):
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
        return Q_obj, Q_pen, C**2

    kp_w = np.array([3, 4, 5, 6, 2, 7]) # weights
    kp_p = np.array([4, 5, 6, 8, 2, 9]) # profits P_ii
    kp_C = 12                           # capacity
    kp_pairs = {                        # pair profits P_ij
        (0, 4): 6,
        (1, 2): 3,
        (3, 5): 10
    }

    kp_P = np.diag(kp_p).astype(float)                              # Put item prices on the diagonal (counted once)
    for (_i, _j), _v in kp_pairs.items():                           # Fill the matrix with pair prices
        kp_P[_i, _j] = kp_P[_j, _i] = _v

    kp_Q_obj, kp_Q_pen, kp_c_pen = knapsack_to_qubo(kp_w, kp_P, kp_C)
    return knapsack_to_qubo, kp_C, kp_P, kp_Q_obj, kp_Q_pen, kp_c_pen, kp_w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## QKP Brute Force
    """)
    return


@app.cell
def _(kp_C, kp_P, kp_w, np):
    def qkp_profit(x, P):
        return x @ np.triu(P) @ x

    def qkp_brute_force(w, P, C):
        n = len(w)
        bext_x, best_profit = None, -np.inf
        for row in range(2**n):
            x = np.array([int(b) for b in format(row, f'0{n}b')])
            if w @ x <= C and qkp_profit(x, P) > best_profit:
                best_x, best_profit = x, qkp_profit(x, P)
        return best_x, best_profit

    kp_best_x, kp_best_profit = qkp_brute_force(kp_w, kp_P, kp_C)
    print(f'Best x: {kp_best_x}, Profit: {kp_best_profit}')
    return kp_best_profit, qkp_profit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## QKP Solver
    """)
    return


@app.cell
def _(
    SEED,
    kp_P,
    kp_Q_obj,
    kp_Q_pen,
    kp_best_profit,
    kp_c_pen,
    kp_w,
    multiplicative_solver,
    np,
    plt,
    qkp_profit,
    qubo_to_ising,
):
    kp_lam = 10.0
    kp_J, kp_h, kp_c = qubo_to_ising(kp_Q_obj + kp_lam * kp_Q_pen)
    kp_c = kp_c + kp_lam * kp_c_pen

    def max_field(J, h):
        return np.max(np.abs(J).sum(axis=1) + np.abs(h))

    _dt = 1 / max_field(kp_J, kp_h)

    _best_sigma, _best_E, _history, _traj_hist = multiplicative_solver(
        kp_J, kp_h, n_iter=50, n_steps=200, dt=_dt, init_size=3, end_size=1, seed=SEED
    )

    _n = len(kp_h)
    _z = (1 + _best_sigma) / 2
    _x, _y = _z[:len(kp_w)], _z[len(kp_w):]
    print(f"items {_x.astype(int)}, profit {qkp_profit(_x, kp_P):.0f} (optimum {kp_best_profit:.0f})")

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4))

    _ax[0].plot(_history[:, 0] + kp_c, label='energy of this run')
    _ax[0].plot(_history[:, 1] + kp_c, label='best energy')
    _ax[0].set_xlabel('iteration')
    _ax[0].set_ylabel('energy')
    _ax[0].legend()

    _traj_hist = _traj_hist.reshape(-1, _n)
    for _k in range(_n):
        _ax[1].plot(_traj_hist[:, _k], label=f's{_k}', linewidth=2)
    _ax[1].set_xlabel('step')
    _ax[1].set_ylabel('voltage')
    _ax[1].legend(loc='upper right')

    _ax[0].set_xlim(0, 5)
    _ax[1].set_xlim(0, 1000)

    _fig
    return (max_field,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fast Solver

    This is AI-generated for now but gives the exact same results on below problem. Be somewhat careful.
    """)
    return


@app.cell
def _(cluster_size, njit, np):
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
        s = np.sign(v)                                                      # sign values of spins
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

            for i in range(n):                                              # Update all spins
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

    return


@app.cell
def _(
    Path,
    SEED,
    knapsack_to_qubo,
    max_field,
    multiplicative_solver,
    np,
    plt,
    qkp_profit,
    qubo_to_ising,
):
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


    bm_name = "jeu_100_25_1"
    bm_w, bm_P, bm_C = load_qkp(bm_name)
    bm_best = load_optimum(bm_name)

    bm_Q_obj, bm_Q_pen, bm_c_pen = knapsack_to_qubo(bm_w, bm_P, bm_C)
    bm_lam = bm_P.max()
    bm_J, bm_h, bm_c = qubo_to_ising(bm_Q_obj + bm_lam * bm_Q_pen)
    bm_c = bm_c + bm_lam * bm_c_pen

    _dt = 1 / max_field(bm_J, bm_h)

    N_STEPS = 10000

    _best_sigma, _best_E, _history, _states = multiplicative_solver(
        bm_J, bm_h, n_steps=N_STEPS, dt=_dt, n_iter=50, init_size=20, end_size=1, seed=SEED
    )

    _z = (1 + _best_sigma) / 2
    _x = _z[:len(bm_w)]
    _a = np.concatenate([bm_w, 2.0 ** np.arange(len(_z) - len(bm_w))])
    print(f"weight {bm_w @ _x:.0f}/{bm_C}, feasible {bm_w @ _x <= bm_C}, "
          f"profit {qkp_profit(_x, bm_P):.0f} / {bm_best:.0f}, gap {1 - qkp_profit(_x, bm_P) / bm_best:.1%}")

    _fig, _ax = plt.subplots()

    _ax.plot(_history[:, 0] + bm_c, label='energy of this run')
    _ax.plot(_history[:, 1] + bm_c, label='best energy')
    _ax.axhline(-18000, color='black', linestyle='--')
    _ax.set_xlabel('iteration')
    _ax.set_ylabel('energy')
    _ax.legend()
    _ax.set_title(f'QKP: {bm_name}')

    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
