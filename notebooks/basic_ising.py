import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    return np, pd


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
    return (possible_states,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Local Field (gradient)
    """)
    return


@app.cell
def _(J, N, h, np, possible_states):
    def field(J, s, h):
        """
        Returns the local field that pushes on a node i.e. the gradient.
        """
        return J @ s + h

    local_min = np.zeros(2**N, dtype=bool)
    for _row in range(2**N):
        _s = possible_states[_row]
        grad = field(J, _s, h)
        print(grad)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
