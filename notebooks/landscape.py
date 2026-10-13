import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from numba import njit
    import marimo as mo

    return Path, njit, np, plt


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

    return (multiplicative_event,)


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
        return Q_obj, Q_pen, C**2

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
        max_field,
        qkp_profit,
        qubo_to_ising,
    )


@app.cell
def _(
    knapsack_to_qubo,
    load_optimum,
    load_qkp,
    max_field,
    multiplicative_event,
    np,
    plt,
    qkp_profit,
    qubo_to_ising,
):
    SEED = 0

    bm_name = "jeu_100_25_1"
    bm_w, bm_P, bm_C = load_qkp(bm_name)
    bm_best = load_optimum(bm_name)

    bm_Q_obj, bm_Q_pen, bm_c_pen = knapsack_to_qubo(bm_w, bm_P, bm_C)
    bm_lam = bm_P.max()
    bm_J, bm_h, bm_c = qubo_to_ising(bm_Q_obj + bm_lam * bm_Q_pen)
    bm_c = bm_c + bm_lam * bm_c_pen

    _dt = 1 / max_field(bm_J, bm_h)

    N_STEPS = 1000000

    _best_sigma, _best_E, _history, _states = multiplicative_event(
        bm_J, bm_h, n_iter=50, init_size=20, end_size=1, seed=SEED
    )

    _z = (1 + _best_sigma) / 2
    _x = _z[:len(bm_w)]
    _a = np.concatenate([bm_w, 2.0 ** np.arange(len(_z) - len(bm_w))])
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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
