import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import networkx as nx
    import marimo as mo

    return mo, np, nx, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solver
    """)
    return


@app.cell
def _(np):
    def energy(s, J, h):
        return -0.5 * s.T @ J @ s - h @ s

    def field(J, s, h):
        """
        Returns the local field that pushes on a node i.e. the gradient.
        """
        return J @ s + h

    def analog_run(J, h, v0, dt, n_steps):
        v = v0.copy()
        trajectory = [v.copy()]             # Store initial states
        for _ in range(n_steps):            # Update voltages
            f = field(J, np.sign(v), h)     # Change in state depends on local field
            v = np.clip(v + dt * f, -1, 1)  # Update the voltage and clip to -1, +1
            trajectory.append(v.copy())     # Append updated voltages
        return np.array(trajectory)

    def cluster_size(it, n_iter, init_size, end_size):
        """
        Linear decay from init_size to end_size over n_iter.
        """
        steps = np.linspace(init_size, end_size, n_iter)
        return round(steps[it])

    def multiplicative_solver(J, h, n_iter, n_steps, dt, init_size, end_size, seed, v0=None):
        rng = np.random.default_rng(seed)       # Setup RNG
        n = len(h)                              # Number of spins
        if v0 is None:
            v = rng.uniform(-1, 1, n)           # Initial voltages for the analog run
        else:
            v = v0.copy()
        best_sigma, best_E = None, np.inf       # Store best sigmas and their corresponding energy
        history = []
        traj_hist = []
        steps = np.linspace(init_size, end_size, n_iter)

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

    return (multiplicative_solver,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Max-Cut Problem

    Generate random max-cut problems
    """)
    return


@app.cell
def _(np, nx, plt):
    def community_graph(n_comm, size, p_in, p_out, seed):
        rng = np.random.default_rng(seed)
        N = n_comm * size
        labels = np.repeat(np.arange(n_comm), size)
        W = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                p = p_in if labels[i] == labels[j] else p_out
                if rng.random() < p:
                    W[i, j] = W[j, i] = rng.integers(1, 4)
        return W, labels

    W, labels = community_graph(n_comm=4, size=4, p_in=0.8, p_out=0.1, seed=1)
    N = len(W)
    J = -W / 2
    h = np.zeros(N)

    _fig, _ax = plt.subplots(1, 2, figsize=(10, 4))
    _ax[0].imshow(W, cmap='Greys')
    _ax[0].set_title('W (darker = larger edge)')
    _G = nx.from_numpy_array(W)
    nx.draw(_G, pos=nx.spring_layout(_G, seed=0), node_color=labels, cmap='tab10', with_labels=True, ax=_ax[1])
    _fig
    return J, N, h


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Brute Force
    """)
    return


@app.cell
def _(J, h, np):
    def all_states(n):
        states = np.zeros((2**n, n))
        for row in range(2**n):
            bits = format(row, f'0{n}b')
            for col in range(n):
                states[row, col] = 1 if bits[col] == "1" else -1
        return states

    def brute_force(J, h):
        states = all_states(len(h))
        energies = -0.5 * ((states @ J) * states).sum(axis=1) - states @ h
        return states[energies.argmin()], energies.min()

    opt_sigma, opt_E = brute_force(J, h)
    print(f'Optimal sigma: {opt_sigma}')
    print(f'Optimal energy: {opt_E}')
    return (opt_E,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solver
    """)
    return


@app.cell
def _(J, N, h, multiplicative_solver, opt_E, plt):
    # Run solver
    _best_sigma, _best_E, _history, _traj_hist = multiplicative_solver(
        J, h, n_iter=80, n_steps=200, dt=0.01, init_size=3, end_size=1, seed=0
    )

    print(f"flat solver E = {_best_E}, optimum E = {opt_E}")

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4))

    _ax[0].plot(_history[:, 0], label='energy of this run')
    _ax[0].plot(_history[:, 1], label='best energy')
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
    ## Sub-problems
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
