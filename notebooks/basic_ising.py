import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    return np, pd, plt


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

    J, h = maxcut_to_ising(W)
    print('J Matrix')
    for _row in range(N):
        print(J[_row, :])

    print('\nBias vector (h)')
    print(h)
    return J, cut_value, energy, h


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
    return SEED, analog_run, dt, sigma


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
def _(J, N, SEED, analog_run, dt, energy, h, np, plt, sigma):
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

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4))

    _ax[0].plot(history[:, 0], label='energy of this run')
    _ax[0].plot(history[:, 1], label='best energy')
    _ax[0].set_xlabel('iteration')
    _ax[0].set_ylabel('energy')
    _ax[0].legend()

    traj_hist = traj_hist.reshape(-1, N)
    for _k in range(N):
        _ax[1].plot(traj_hist[:, _k], label=f's{_k} = {sigma[_k]}', linewidth=2)
    _ax[1].set_xlabel('step')
    _ax[1].set_ylabel('voltage')

    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
