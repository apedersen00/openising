import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Event Solver

    The multiplicative solver computes a dt based on the fasted dynamic and then performs Forward Euler for `n_steps`.

    For penalty problems the dt becomes incredibly small requiring high amounts of simulation time.

    For the simplified model of the analog solver, all sigma voltages move in a straight line. Thus the solver can be made "event-based" where we simply compute the intersect of the sigmas with the threshold.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from numba import njit
    import marimo as mo

    return mo, njit, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simple Problem Definition

    A simple max-cut problem is defined for testing.
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

    def maxcut_to_ising(W):
        J = -W / 2
        h = np.zeros(len(W))
        return J, h

    def energy(s, J, h):
        return -0.5 * s.T @ J @ s - h @ s

    def field(J, s, h):
        """
        Returns the local field that pushes on a node i.e. the gradient.
        """
        return J @ s + h

    def max_field(J, h):
        return np.max(np.abs(J).sum(axis=1) + np.abs(h))

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

    return (
        N,
        W,
        cluster_size,
        energy,
        field,
        max_field,
        maxcut_to_ising,
        multiplicative_solver,
    )


@app.cell
def _(N, W, max_field, maxcut_to_ising, multiplicative_solver, plt):
    SEED = 0

    J, h = maxcut_to_ising(W)
    _dt = 1 / max_field(J, h)

    best_sigma, best_E, history, traj_hist = multiplicative_solver(
        J, h, n_iter=3, n_steps=200, dt=0.05, init_size=2, end_size=1, seed=SEED
    )

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4))

    _ax[0].plot(history[:, 0], label='energy of this run', linewidth=2)
    _ax[0].plot(history[:, 1], label='best energy', linewidth=2)
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
    return J, SEED, h, history


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fast Solver
    """)
    return


@app.cell
def _(energy, field, np):
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

    return cluster_size, multiplicative_solver


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
def _(J, SEED, h, history, multiplicative_event, plt):
    _best_sigma, _best_E, _history, _ = multiplicative_event(
        J, h, n_iter=5, init_size=2, end_size=1, seed=SEED
    )

    _fig, _ax = plt.subplots()

    _ax.plot(history[:, 0], label='energy of this run', linewidth=2)
    _ax.plot(history[:, 1], label='best energy', linewidth=2)
    _ax.set_xlabel('iteration')
    _ax.set_ylabel('energy')
    _ax.legend()

    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
