import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import itertools

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from numba import njit
    import marimo as mo

    return Path, itertools, mo, njit, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Basic multiplicative solver
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

    return analog_event, multiplicative_event


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper functions
    """)
    return


@app.cell
def _(Path, np):
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

    def max_field(J, h):
        return np.max(np.abs(J).sum(axis=1) + np.abs(h))

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

    def cluster_size(it, n_iter, init_size, end_size):
        """
        Linear decay from init_size to end_size over n_iter.
        """
        steps = np.linspace(init_size, end_size, n_iter)
        return round(steps[it])

    def qkp_profit(x, P):
        return x @ np.triu(P) @ x

    return (
        cluster_size,
        knapsack_to_qubo,
        load_optimum,
        load_qkp,
        qkp_profit,
        qubo_to_ising,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline
    """)
    return


@app.cell
def _(
    knapsack_to_qubo,
    load_optimum,
    load_qkp,
    multiplicative_event,
    plt,
    qkp_profit,
    qubo_to_ising,
):
    SEED = 0

    bm_name = "jeu_100_25_1"
    bm_w, bm_P, bm_C = load_qkp(bm_name)
    bm_best = load_optimum(bm_name)

    bm_Q_obj, bm_Q_pen, bm_c_pen, bm_a = knapsack_to_qubo(bm_w, bm_P, bm_C)
    bm_lam = bm_P.max()
    bm_J, bm_h, bm_c = qubo_to_ising(bm_Q_obj + bm_lam * bm_Q_pen)
    bm_c = bm_c + bm_lam * bm_c_pen

    _best_sigma, _best_E, _history, _states = multiplicative_event(
        bm_J, bm_h, n_iter=50, init_size=20, end_size=1, seed=SEED
    )

    _z = (1 + _best_sigma) / 2
    _x = _z[:len(bm_w)]
    print(f"weight {bm_w @ _x:.0f}/{bm_C}, feasible {bm_w @ _x <= bm_C}, "
          f"profit {qkp_profit(_x, bm_P):.0f} / {bm_best:.0f}, gap {1 - qkp_profit(_x, bm_P) / bm_best:.1%}")

    _fig, _ax = plt.subplots()

    _ax.plot(_history[:, 0] + bm_c, label='energy of this run')
    _ax.plot(_history[:, 1] + bm_c, label='best energy')
    _ax.set_xlabel('iteration')
    _ax.set_ylabel('energy')
    _ax.legend()
    _ax.set_title(f'QKP: {bm_name}')

    _fig
    return SEED, bm_C, bm_P, bm_Q_obj, bm_Q_pen, bm_a, bm_best, bm_lam, bm_w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solver that can update weights every iteration
    """)
    return


@app.cell
def _(analog_event, cluster_size, njit, np):
    @njit
    def solver_schedule(J_obj, h_obj, J_pen, h_pen, lambdas, v, sizes, clusters, max_events):
        n_iter = len(lambdas)
        states = np.zeros((n_iter, len(h_obj)))             # end state of every run
        times = np.zeros(n_iter)                            # settle time, largest current scaled to 1
        inc = np.zeros(len(h_obj))                          # incumbent: the state we kick from
        for it in range(n_iter):
            J = J_obj + lambdas[it] * J_pen
            h = h_obj + lambdas[it] * h_pen
            sigma, T, _ = analog_event(J, h, v, max_events)
            states[it] = sigma
            times[it] = T * np.max(np.abs(J).sum(axis=1) + np.abs(h))
            if it == 0 or -0.5 * sigma @ J @ sigma - h @ sigma < -0.5 * inc @ J @ inc - h @ inc:   # both under this lam
                inc = sigma.copy()
            v = inc.copy()
            for c in clusters[it, :sizes[it]]:
                v[c] *= -1
        return states, times

    def multiplicative_schedule(J_obj, h_obj, J_pen, h_pen, lambdas, init_size, end_size, seed, max_events=10**6):
        rng = np.random.default_rng(seed)
        n = len(h_obj)
        n_iter = len(lambdas)
        v = rng.uniform(-1, 1, n)
        sizes = np.array([cluster_size(it, n_iter, init_size, end_size) for it in range(n_iter)])
        clusters = np.zeros((n_iter, n), dtype=np.int64)
        for it in range(n_iter):
            clusters[it, :sizes[it]] = rng.choice(n, size=sizes[it], replace=False)
        return solver_schedule(J_obj, h_obj, J_pen, h_pen, lambdas, v, sizes, clusters, max_events)

    return (multiplicative_schedule,)


@app.cell
def _(
    bm_C,
    bm_P,
    bm_Q_obj,
    bm_Q_pen,
    bm_best,
    bm_lam,
    bm_w,
    multiplicative_schedule,
    np,
    pd,
    qkp_profit,
    qubo_to_ising,
):
    def evaluate_qkp(states, w, P, C, best):
        x = (1 + states[:, :len(w)]) / 2
        fits = x @ w <= C
        profit = np.array([qkp_profit(xi, P) for xi in x])
        best_profit = profit[fits].max() if fits.any() else 0.0
        return 1 - best_profit / best, fits.mean()

    def schedule(shape, lam_start, lam_end, n_iter, ramp_frac=0.7):
        """Penalty weight of every run: go from lam_start to lam_end over the first ramp_frac of the runs, then hold."""
        n_ramp = int(ramp_frac * n_iter)
        p = np.minimum(np.arange(n_iter) / (n_ramp - 1), 1.0)          # progress: 0 → 1 during the ramp, then stays 1
        if shape == 'constant':
            return np.full(n_iter, float(lam_end))
        if shape == 'linear':                                           # same step every run
            return lam_start + (lam_end - lam_start) * p
        if shape == 'exponential':                                      # same factor every run
            return lam_start * (lam_end / lam_start) ** p
        if shape == 'quadratic':                                        # slow start, fast end
            return lam_start + (lam_end - lam_start) * p**2
        if shape == 'step':                                             # lam_start for the whole ramp, then jump
            return np.where(p < 1, float(lam_start), float(lam_end))

    bm_J_obj, bm_h_obj, _ = qubo_to_ising(bm_Q_obj)             # objective and penalty kept apart
    bm_J_pen, bm_h_pen, _ = qubo_to_ising(bm_Q_pen)             # so each run can weight the penalty differently

    N_ITER = 50
    SHAPES = ['constant', 'linear', 'exponential', 'quadratic', 'step']
    SCHEDULES = {shape: schedule(shape, 1, bm_lam, N_ITER) for shape in SHAPES}

    _rows = []
    for _name, _lams in SCHEDULES.items():
        for _seed in range(10):
            _states, _times = multiplicative_schedule(bm_J_obj, bm_h_obj, bm_J_pen, bm_h_pen, _lams,
                                                      init_size=10, end_size=1, seed=_seed)
            _gap, _feas = evaluate_qkp(_states, bm_w, bm_P, bm_C, bm_best)
            _rows.append(dict(schedule=_name, seed=_seed, gap=_gap, feasible=_feas, settle=np.median(_times)))

    results = pd.DataFrame(_rows)
    results.groupby('schedule')[['gap', 'feasible', 'settle']].mean()
    return (
        N_ITER,
        SCHEDULES,
        bm_J_obj,
        bm_J_pen,
        bm_h_obj,
        bm_h_pen,
        evaluate_qkp,
    )


@app.cell
def _(
    SCHEDULES,
    SEED,
    bm_C,
    bm_J_obj,
    bm_J_pen,
    bm_P,
    bm_best,
    bm_h_obj,
    bm_h_pen,
    bm_w,
    multiplicative_schedule,
    np,
    plt,
    qkp_profit,
):
    _lam_lim = (min(l.min() for l in SCHEDULES.values()) / 1.5, max(l.max() for l in SCHEDULES.values()) * 1.5)

    _fig, _ax = plt.subplots(1, len(SCHEDULES), figsize=(20, 4), sharey=True)
    for _a, (_name, _lams) in zip(_ax, SCHEDULES.items()):
        _states, _ = multiplicative_schedule(bm_J_obj, bm_h_obj, bm_J_pen, bm_h_pen, _lams,
                                             init_size=10, end_size=1, seed=SEED)
        _x = (1 + _states[:, :len(bm_w)]) / 2
        _fits = _x @ bm_w <= bm_C
        _profit = np.array([qkp_profit(_xi, bm_P) for _xi in _x])
        _a.plot(np.where(_fits, _profit, np.nan), 'o', label='items fit')
        _a.plot(np.where(~_fits, _profit, np.nan), 'x', label='overweight')
        _a.axhline(bm_best, color='black', linestyle='--', label='optimum')
        _a.set_title(_name)
        _a.set_xlabel('run')
        _a2 = _a.twinx()
        _a2.plot(_lams, color='gray', alpha=0.5)
        _a2.set_ylim(_lam_lim)
    _a2.set_ylabel('λ')
    _ax[0].set_ylabel('profit')
    _ax[0].legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Lagrange

    Follows the paper: delacour2025selfadaptive

    - $f(x)$ is the cost to minimize. For QKP f = -profit
    - $g(x)$ is the constraint, satisfied when $g(x) = 0$. For a knapsack its $g = w \cdot x - C$
    - The penalty if $E = f + P \cdot g^2$. The paper's P is $\lambda$

    The minimum of $E$ is only feasible if $P \geq P_C$, some critical value that depends on the problem. A large $P$ makes $f$ negligible and the landscape difficult.

    Instead of the penalty, Lagrange relaxtion linearly weights constraints by Lagrange multipliers $\lambda$ and adds them to the energy function:

    $L = E(x) + \lambda^T g(x) = -\text{profit} + \lambda (\text{weight} - C)$

    We don't know the right $\lambda$ in advance, so it's updated based on solver output. In continuous time this is:

    $$
        \tau_\lambda \frac{d\lambda}{dt} = g(\overline{x}) = +\nabla_\lambda LB_L
    $$

    The algorithm from the paper:

    ```
    (lambda_0, P) <- (0, adN)
    for K iterations:
        minimize L_k:   x_k = argmin_x L_k                      > Ising machine
        store feasible  x_k                                     > CPU
        update:         lambda_{k+1} <- lamdda_k + n * g(x_k)   > CPU
    x <- argmin_k f(x_k)
    ```
    """)
    return


@app.cell
def _(analog_event, cluster_size, njit, np):
    @njit
    def energy(s, J, h):
        return -0.5 * s.T @ J @ s - h @ s

    @njit
    def qkp_penalty(a, sigma, C):
        return a.T @ ((1 + sigma) / 2) - C

    @njit
    def solver_lagrange(J_obj, h_obj, J_pen, h_pen, a, C, P, eta, v, sizes, clusters, max_events):
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
        for it in range(n_iter):                            # Do K iterations
            h = h_obj + P * h_pen - lam * a / 2             # Add the Lagrange term (only affects bias)
            sigma, T, _ = analog_event(J, h, v, max_events) # Run the typical solver
            states[it] = sigma
            lams[it] = lam

            if it == 0:
                winner = sigma.copy()
            elif energy(sigma, J, h) < energy(winner, J, h):
                winner = sigma.copy()

            lam += eta * qkp_penalty(a, sigma, C)
            v = winner.copy()
            for c in clusters[it, :sizes[it]]:
                v[c] *= -1
        return states, lams

    def multiplicative_lagrange(J_obj, h_obj, J_pen, h_pen, a, C, P, eta, n_iter, init_size, end_size, seed, max_events=10**6):
        rng = np.random.default_rng(seed)
        n = len(h_obj)
        v = rng.uniform(-1, 1, n)
        sizes = np.array([cluster_size(it, n_iter, init_size, end_size) for it in range(n_iter)])   # List of cluster sizes e.g. [10, 5, 1]
        clusters = np.zeros((n_iter, n), dtype=np.int64)                                            # Array of which spins in each cluster
        for it in range(n_iter):
            clusters[it, :sizes[it]] = rng.choice(n, size=sizes[it], replace=False)
        return solver_lagrange(J_obj, h_obj, J_pen, h_pen, a, C, P, eta, v, sizes, clusters, max_events)

    return (multiplicative_lagrange,)


@app.cell
def _(
    bm_C,
    bm_J_obj,
    bm_J_pen,
    bm_P,
    bm_a,
    bm_best,
    bm_h_obj,
    bm_h_pen,
    bm_w,
    evaluate_qkp,
    itertools,
    multiplicative_lagrange,
    pd,
):
    # Parameters to sweep. All combinations will be tested
    RUNS = {
        'P': [0.1, 0.5, 1.0, 2.0],
        'n': [0.1, 0.5, 1.0, 2.0]
    }

    _N_ITER = 50
    INIT_SIZE = 1
    END_SIZE = 1
    SEEDS = 10

    lagrange_results = []

    for _P, _n in itertools.product(RUNS['P'], RUNS['n']):
      for _seed in range(SEEDS):
        _states, _lambdas = multiplicative_lagrange(
            bm_J_obj,
            bm_h_obj,
            bm_J_pen,
            bm_h_pen,
            bm_a,
            bm_C,
            _P,
            _n,
            n_iter=_N_ITER,
            init_size=INIT_SIZE,
            end_size=END_SIZE,
            seed=_seed,
        )

        _gap, _feas = evaluate_qkp(_states, bm_w, bm_P, bm_C, bm_best)

        lagrange_results.append({
            'P': _P,
            'n': _n,
            'seed': _seed,
            'gap': _gap,
            'feasible': _feas,
            'lambda': _lambdas[-1],
            'lambdas': _lambdas,
            'states': _states,
        })

    df = pd.DataFrame(lagrange_results)
    df
    return


@app.cell
def _(
    LAGRANGE,
    N_ITER,
    SEED,
    bm_C,
    bm_J_obj,
    bm_J_pen,
    bm_P,
    bm_a,
    bm_best,
    bm_h_obj,
    bm_h_pen,
    bm_w,
    multiplicative_lagrange,
    np,
    plt,
    qkp_profit,
):
    _runs = {_name: multiplicative_lagrange(bm_J_obj, bm_h_obj, bm_J_pen, bm_h_pen, bm_a, bm_C, n_iter=N_ITER,
                                            init_size=10, end_size=1, seed=SEED, **_kw)
            for _name, _kw in LAGRANGE.items()}
    _lam_max = max(_lams.max() for _, _lams in _runs.values()) * 1.1          # same price axis on every panel

    _fig, _ax = plt.subplots(2, len(LAGRANGE), figsize=(6 * len(LAGRANGE), 7), sharex=True, sharey='row')
    for _col, (_name, (_states, _lams)) in enumerate(_runs.items()):
        _x = (1 + _states[:, :len(bm_w)]) / 2
        _fits = _x @ bm_w <= bm_C
        _profit = np.array([qkp_profit(_xi, bm_P) for _xi in _x])
        _over = _x @ bm_w - bm_C                                                  # > 0: overweight

        _a = _ax[0, _col]                                                         # top: profit of every run
        _a.plot(np.where(_fits, _profit, np.nan), 'o', label='items fit')
        _a.plot(np.where(~_fits, _profit, np.nan), 'x', label='overweight')
        _a.axhline(bm_best, color='black', linestyle='--', label='optimum')
        _a.set_title(_name)

        _b = _ax[1, _col]                                                         # bottom: over/under capacity and the price
        _b.bar(range(N_ITER), _over, color=np.where(_over > 0, 'tab:orange', 'tab:blue'))
        _b.axhline(0, color='black', linewidth=0.8)
        _b.set_xlabel('run')
        _b2 = _b.twinx()
        _b2.step(range(N_ITER), _lams, where='post', color='gray')
        _b2.set_ylim(0, _lam_max)
    _b2.set_ylabel('price λ')
    _ax[0, 0].set_ylabel('profit')
    _ax[0, 0].legend()
    _ax[1, 0].set_ylabel('items weight − capacity')
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
