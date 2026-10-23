import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo

    return mo, np, plt


@app.cell
def _(np):
    def spin_glass_2d(L, seed=0):
        """+-1 couplings between nearest neighbours on an L x L torus. Spin index = row * L + col."""
        rng = np.random.default_rng(seed)
        N = L * L
        J = np.zeros((N, N))
        for r in range(L):
            for c in range(L):
                i = r * L + c
                for j in (((r + 1) % L) * L + c, r * L + (c + 1) % L):   # neighbour below and to the right
                    J[i, j] = J[j, i] = rng.choice([-1.0, 1.0])
        return J

    def energy(J, h, s):
        return -0.5 * s @ J @ s - h @ s

    L = 8
    J = spin_glass_2d(L)
    h = np.zeros(L * L)
    N = len(h)

    print(f'Spins: {N}')
    print(f'Couplings: {np.count_nonzero(J) // 2}')
    return J, L, N, energy, h


@app.cell
def _(J, L, N, np, plt):
    core_size = 4
    rows, cols = np.divmod(np.arange(N), L)
    labels = (rows // core_size) * (L // core_size) + cols // core_size
    same = labels[:, None] == labels[None, :]

    _fig, _ax = plt.subplots(figsize=(3, 3))
    _ax.imshow(labels.reshape(L, L), cmap='tab10')
    _ax.set_title('core of every spin')
    print('couplings inside cores:', np.count_nonzero(J * same) // 2, ' between cores:', np.count_nonzero(J * ~same) // 2)
    _fig
    return (labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Core
    """)
    return


@app.cell
def _(np):
    class Core:
        """
        One fully connected crossbar: c spins, c*c coupler cells and one bias per row.
        """
        def __init__(self, idx, J, h):
            self.idx = np.asarray(idx)              # which spins of the full problem are in this core
            self.J = J[np.ix_(self.idx, self.idx)]  # coupling in this core
            self.h = h[self.idx].copy()             # bias of the rows
            self.v = np.zeros(len(self.idx))        # spin voltages

        @property
        def sigma(self):
            return np.where(self.v >= 0, 1.0, -1.0)
    
        def field(self, h_ext):
            return self.J @ self.sigma + self.h + h_ext
    
        def step(self, dt, h_ext):
            self.v = np.clip(self.v + dt * self.field(h_ext), -1, 1)

    return (Core,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the core on the 64-spin problem and see that it settles.
    """)
    return


@app.cell
def _(Core, J, N, energy, h, np):
    big = Core(np.arange(N), J, h)
    big.v = np.random.default_rng(1).uniform(-1, 1, N)
    big_E = []
    for _ in range(1000):
        big.step(0.01, 0.0)
        big_E.append(energy(J, h, big.sigma))
    big_settled = np.all(big.sigma * big.field(0.0) >= 0)

    print(f'Initial energy: {big_E[0]}')
    print(f'Final energy: {big_E[-1]}')
    print(f'Settled? {big_settled}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-core
    """)
    return


@app.cell
def _(Core, np):
    class MultiCore:
        """
        Several cores on one problem. A core sees the spins of other cores through a copy
        that is refreshed every T time units.
        """
        def __init__(self, J, h, labels, T):
            self.N = len(h)
            self.cores = [Core(np.flatnonzero(labels == k), J, h) for k in range(labels.max() + 1)]
            same = labels[:, None] == labels[None, :]
            self.J_ext = np.where(same, 0.0, J)     # Couplings between cores
            self.T = T
            self.sigma_seen = np.ones(self.N)
            self.t = 0.0
            self.t_refresh = 0.0

        @property
        def sigma(self):
            """
            Get the sigmas from all cores i.e. the global state.
            """
            s = np.empty(self.N)
            for core in self.cores:
                s[core.idx] = core.sigma
            return s

        def start(self, v):
            """
            Initialize all cores to the starting voltages.
            """
            for core in self.cores:
                core.v = v[core.idx].copy()
            self.sigma_seen = self.sigma
            self.t = self.t_refresh = 0.0
    
        def step(self, dt):
            if self.t - self.t_refresh >= self.T:
                self.sigma_seen = self.sigma
                self.t_refresh = self.t
            h_ext = self.J_ext @ self.sigma_seen
            for core in self.cores:
                core.step(dt, h_ext[core.idx])
            self.t += dt

    return (MultiCore,)


@app.cell
def _(Core, J, MultiCore, N, energy, h, labels, np):
    _v0 = np.random.default_rng(1).uniform(-1, 1, N)
    _big = Core(np.arange(N), J, h)
    _big.v = _v0.copy()
    _mc = MultiCore(J, h, labels, T=0)
    _mc.start(_v0)
    for _ in range(1000):
        _big.step(0.01, 0.0)
        _mc.step(0.01)
    np.array_equal(_big.sigma, _mc.sigma), energy(J, h, _mc.sigma)
    return


@app.cell
def _(Core, J, MultiCore, N, energy, h, labels, np):
    for _T in (0, 0.1, 1.0, 3.0, np.inf):
        _differs, _dE, _not_min = 0, [], 0
        for _seed in range(200):
            _v0 = np.random.default_rng(_seed).uniform(-1, 1, N)
            _big = Core(np.arange(N), J, h)
            _big.v = _v0.copy()
            _mc = MultiCore(J, h, labels, _T)
            _mc.start(_v0)
            for _ in range(1000):
                _big.step(0.01, 0.0)
                _mc.step(0.01)
            _s = _mc.sigma
            _differs += not np.array_equal(_s, _big.sigma)
            _dE.append(energy(J, h, _s) - energy(J, h, _big.sigma))
            _not_min += not np.all(_s * (J @ _s + h) >= 0)      # judged with the TRUE fields, not the copy
        print(f'T = {_T}: differs from the big core in {_differs}/200 runs, mean energy difference {np.mean(_dE):+.2f}, not a true local minimum in {_not_min}/200')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
