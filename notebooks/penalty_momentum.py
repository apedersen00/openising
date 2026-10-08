import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Penalty ramps and momentum for the Multiplicative solver

    Two ideas to help the multiplicative Ising machine traverse the energy landscape:

    1. **Penalty ramps** for constrained problems (TSP, QKP). With a strong penalty, feasible solutions are isolated:
       swapping two TSP cities takes 4 spin flips through states that each cost about one penalty unit. Starting with a
       weak penalty lowers those barriers; ramping it up makes the final state feasible. Related ideas tried as well:
       an *adaptive* penalty (strategic oscillation around the feasibility boundary) and an *augmented Lagrangian*
       (one multiplier per constraint, which only changes the biases $h$).
    2. **Momentum / Adam.** The capacitor voltage $v$ already integrates the field (like the latent weights of a
       binarised neural network). Momentum low-pass filters the field before it reaches $v$ (heavy-ball dynamics, one
       extra integrator per spin). Adam additionally divides by the running rms of the field, per spin.

    Both are tested on the flat Multiplicative solver and on the hierarchical solver. The solvers are re-implemented
    below in a compact form (one numba kernel and a Python outer loop) and checked against the repo's solvers.

    **Running.** Set `QUICK` in the settings cell: `True` is a smoke test (well under a minute), `False` the full
    study (about 2 CPU-hours; runs are spread over all cores with threads). To run headless and keep all outputs:

    ```bash
    uv run marimo export html notebooks/penalty_momentum.py -o penalty_momentum.html
    ```

    All runs are also written to `data/penalty_momentum.csv`.
    """)
    return


@app.cell
def _():
    import os

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # we parallelise over runs instead
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    import time
    import warnings
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import dataclass

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    from cycler import cycler
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator, NullFormatter
    from numba import njit

    from ising.generators.TSP import TSP, get_index
    from ising.solvers.Hierarchical_solver import HierarchicalSolver
    from ising.solvers.Multiplicative import Multiplicative
    from ising.stages import TOP
    from ising.stages.maxcut_parser_stage import MaxcutParserStage
    from ising.stages.model.ising import IsingModel
    from ising.stages.qkp_parser_stage import QKPParserStage
    from ising.stages.tsp_parser_stage import TSPParserStage
    from ising.utils.numpy import triu_to_symm
    from ising.utils.partitioning_schemes import make_modularity_partitioning

    warnings.filterwarnings("ignore", category=FutureWarning)  # networkx adjacency_matrix in the TSP generator
    return (
        HierarchicalSolver,
        IsingModel,
        LinearSegmentedColormap,
        MaxNLocator,
        MaxcutParserStage,
        Multiplicative,
        NullFormatter,
        QKPParserStage,
        TOP,
        TSP,
        TSPParserStage,
        ThreadPoolExecutor,
        cycler,
        dataclass,
        get_index,
        make_modularity_partitioning,
        mo,
        njit,
        np,
        nx,
        os,
        pd,
        plt,
        time,
        triu_to_symm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Settings
    """)
    return


@app.cell
def _(TOP, os):
    QUICK = False  # True: smoke test with tiny budgets. False: the full study.

    N_RUNS = 3 if QUICK else 20  # runs per (problem, variant), seeds 0..N_RUNS-1 for every variant
    WORKERS = os.cpu_count()

    TSP_NAMES = ["burma14"] if QUICK else ["burma14", "ulysses16", "ulysses22", "bayg29"]
    QKP_NAMES = ["jeu_100_25_1"] if QUICK else ["jeu_100_25_1", "jeu_200_25_1", "jeu_300_50_1"]
    MAXCUT_NAMES = ["G1"]
    CONSTRAINED = TSP_NAMES + QKP_NAMES

    # Budget of the flat solver: outer iterations (analog run + cluster kick) and max Forward-Euler steps per run
    FLAT = dict(nb_flips=20, max_steps=2000) if QUICK else dict(nb_flips=200, max_steps=10000)

    # Hierarchical solver: problems, sweeps, partitions, meta-nodes per spin and the budget of every core call
    HIER_NAMES = ["burma14", "jeu_100_25_1", "G1"] if QUICK else ["ulysses16", "jeu_100_25_1", "G1"]
    HIER_BASE = dict(
        nb_sweeps=3 if QUICK else 10,
        nb_partitions=4,
        meta_frac=0.25,
        influence="upper",
        sweep_ramp=False,
        core=dict(nb_flips=10, max_steps=2000) if QUICK else dict(nb_flips=50, max_steps=10000),
    )

    SAVE_TO = TOP / "data/penalty_momentum.csv"
    return (
        CONSTRAINED,
        FLAT,
        HIER_BASE,
        HIER_NAMES,
        MAXCUT_NAMES,
        N_RUNS,
        QKP_NAMES,
        QUICK,
        SAVE_TO,
        TSP_NAMES,
        WORKERS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problems

    Every problem is written as

    $$E(s;\lambda) = E_\text{obj}(s) + \lambda\,\kappa\,\lVert A x - b\rVert^2, \qquad x = (1+s)/2,$$

    with $\kappa$ the penalty weight the repo uses, so $\lambda = 1$ is exactly the repo's model (checked below).
    TSP has $2N$ one-hot constraints (one city per time step, every city once); QKP has one (weights plus a binary
    slack variable equal the capacity). Max-Cut has no constraints. The matrices are built in float64; the repo's
    `IsingModel` stores float32, which costs QKP a few energy units of rounding noise.
    """)
    return


@app.cell
def _(
    IsingModel,
    MaxcutParserStage,
    QKPParserStage,
    TOP,
    TSP,
    TSPParserStage,
    dataclass,
    get_index,
    np,
    nx,
    triu_to_symm,
):
    BENCH = TOP / "ising/benchmarks"


    @dataclass
    class Problem:
        """E(s; lam) = E_obj(s) + lam * E_pen(s), each as -1/2 s.C.s - h.s + c with C symmetric, zero diagonal."""

        name: str
        C_obj: np.ndarray
        h_obj: np.ndarray
        C_pen: np.ndarray
        h_pen: np.ndarray
        c_obj: float = 0.0
        c_pen: float = 0.0
        A: np.ndarray | None = None  # constraints A x = b, only used for feasibility and the Lagrangian
        b: np.ndarray | None = None
        best_found: float | None = None
        model: IsingModel | None = None  # the repo's model (lam = 1), for partitioning and the repo solvers

        @property
        def n(self):
            return len(self.h_obj)

        def obj(self, s):
            return -0.5 * s @ self.C_obj @ s - self.h_obj @ s + self.c_obj

        def pen(self, s):
            return -0.5 * s @ self.C_pen @ s - self.h_pen @ s + self.c_pen

        def energy(self, s, lam=1.0):
            return self.obj(s) + lam * self.pen(s)

        def violation(self, s):
            return self.A @ ((1 + s) / 2) - self.b

        def feasible(self, s):
            return self.A is None or not np.any(self.violation(s))


    def qubo_to_ising(Q, const=0.0):
        """x.Q.x + const with x = (1 + s) / 2   ->   C, h, c of  -1/2 s.C.s - h.s + c."""
        Q = (Q + Q.T) / 2
        C = -0.5 * Q
        np.fill_diagonal(C, 0)
        return C, -0.5 * Q.sum(1), 0.25 * (Q.sum() + np.trace(Q)) + const


    def constrained_problem(name, Q_obj, A, b, kappa, best_found, model) -> Problem:
        """E = x.Q_obj.x + lam * kappa * |A x - b|^2."""
        C_obj, h_obj, c_obj = qubo_to_ising(Q_obj)
        C_pen, h_pen, c_pen = qubo_to_ising(kappa * (A.T @ A - 2 * np.diag(A.T @ b)), kappa * b @ b)  # x_i^2 = x_i
        return Problem(name, C_obj, h_obj, C_pen, h_pen, c_obj, c_pen, A, b, best_found, model)


    def load_tsp(name: str) -> Problem:
        graph, best = TSPParserStage.TSP_parser(BENCH / f"TSP/{name}.tsp")
        N = graph.number_of_nodes()
        W = nx.to_numpy_array(graph)
        Q = np.zeros((N * N, N * N))  # spin get_index(t, u) = +1  <=>  city u is visited at time t
        A = np.zeros((2 * N, N * N))
        for t in range(N):
            for u in range(N):
                A[t, get_index(t, u, N)] = 1  # one city per time step
                A[N + u, get_index(t, u, N)] = 1  # every city once
                for v in range(N):
                    Q[get_index(t, u, N), get_index(t + 1, v, N)] += W[u, v]  # tour length
        return constrained_problem(name, Q, A, np.ones(2 * N), W.max(), best, TSP(graph, weight_constant=1.0))


    def load_qkp(name: str) -> Problem:
        graph, best = QKPParserStage.QKP_parser(BENCH / f"Knapsack/{name}.txt")
        N = graph.number_of_nodes()
        P = np.zeros((N, N))
        for (i, j), value in nx.get_edge_attributes(graph, "profit").items():
            P[i, j] = P[j, i] = value
        weights = np.array([graph.edges[i, i]["weight"] for i in range(N)])
        capacity = graph.graph["capacity"]
        nb_bits = int(np.floor(np.log2(capacity) + 1))  # binary slack variable, as in knapsack_to_ising
        Q = np.zeros((N + nb_bits, N + nb_bits))
        Q[:N, :N] = -(P + np.diag(np.diag(P))) / 2  # x.Q.x = -(sum_i P_ii x_i + sum_i<j P_ij x_i x_j) = -profit
        A = np.concatenate([weights, 2.0 ** np.arange(nb_bits)])[None, :]  # sum(w x) + slack = capacity
        model = QKPParserStage.knapsack_to_ising(P, capacity, weights, 1.0)
        return constrained_problem(name, Q, A, np.array([capacity], dtype=float), P.max(), best, model)


    def load_maxcut(name: str) -> Problem:
        graph, best = MaxcutParserStage.G_parser(BENCH / f"G/{name}.txt")
        m = MaxcutParserStage.generate_maxcut(graph)
        C = triu_to_symm(m.J).astype(float)
        zeros = np.zeros(m.num_variables)
        return Problem(name, C, m.h.astype(float), np.zeros_like(C), zeros, float(m.c), best_found=best, model=m)

    return Problem, load_maxcut, load_qkp, load_tsp


@app.cell
def _(
    HIER_NAMES,
    MAXCUT_NAMES,
    QKP_NAMES,
    TSP_NAMES,
    load_maxcut,
    load_qkp,
    load_tsp,
    np,
    pd,
):
    PROBLEMS = (
        {name: load_tsp(name) for name in TSP_NAMES}
        | {name: load_qkp(name) for name in QKP_NAMES}
        | {name: load_maxcut(name) for name in MAXCUT_NAMES}
    )
    assert set(HIER_NAMES) <= set(PROBLEMS)

    # Check: lam = 1 reproduces the repo's model on random states
    _rng = np.random.default_rng(0)
    _rows = []
    for _p in PROBLEMS.values():
        _states = _rng.choice([-1.0, 1.0], (5, _p.n))
        _diff = max(abs(_p.energy(s) - _p.model.evaluate(s.astype(np.float32))) / abs(_p.energy(s)) for s in _states)
        _rows.append(
            dict(
                problem=_p.name,
                spins=_p.n,
                constraints=0 if _p.A is None else len(_p.b),
                best_found=_p.best_found,
                rel_diff_to_repo_model=_diff,
            )
        )
    pd.DataFrame(_rows)
    return (PROBLEMS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The solver

    **Analog kernel.** Forward Euler of the multiplicative dynamics, with the field split into objective and penalty:

    $$\dot v_i = f_i,\qquad f = (C_\text{obj}\,s + h_\text{obj}) + \lambda(t)\,(C_\text{pen}\,s + h_\text{pen}),
    \qquad s = \operatorname{sign}(v),\quad v \in [-1, 1].$$

    Both parts of the field are only updated when a spin changes sign, as in the repo's fast path, so a time-varying
    $\lambda(t)$ costs $O(N)$ per step. Fields are scaled so the largest possible field is 1 and $dt = 0.1$.
    The options:

    | option | dynamics | hardware analogue |
    |---|---|---|
    | `inner_ramp` | $\lambda$ ramps linearly inside every analog run | global gain on the penalty couplers |
    | `tau_m` | heavy ball: $\tau_m \dot m = f - m$, $\dot v = m$ | low-pass filter (integrator) per spin |
    | `tau_r` | Adam: $\dot v = m / \mathrm{rms}(f)$ per spin; `tau_r = 0`: $\mathrm{sign}(f)$ | gain control per spin |
    | `global_norm` | one rms over all spins: an adaptive *global* time step | one global gain control |

    With momentum, outward momentum is dropped at the walls $|v| = 1$ (inelastic walls, as in bSB).

    A run stops when no spin moves any more (every spin at a wall with its field pointing outwards, or zero field),
    i.e. in a single-flip local minimum.
    """)
    return


@app.cell
def _(njit, np):
    @njit(nogil=True)
    def analog_run(C_obj, h_obj, C_pen, h_pen, v, lam_start, lam_end, ramp_steps, max_steps, tau_m, tau_r, global_norm):
        """Simulate one analog run from voltages v. Returns the final spins and the number of steps."""
        dt = 0.1
        n = v.size
        v = v.copy()
        s = np.where(v >= 0, 1.0, -1.0)
        f_obj = C_obj @ s + h_obj  # fields, updated incrementally when a spin flips
        f_pen = C_pen @ s + h_pen
        beta_m = np.exp(-1.0 / tau_m) if tau_m > 0 else 0.0
        beta_r = np.exp(-1.0 / tau_r) if tau_r > 0 else 0.0
        m = np.zeros(n)  # momentum: low-pass filtered field
        r = (f_obj + lam_start * f_pen) ** 2  # running mean of field^2 (Adam's second moment)
        step = 0
        while step < max_steps:
            lam = lam_end if step >= ramp_steps else lam_start + (lam_end - lam_start) * step / ramp_steps
            rms = np.sqrt(r.mean()) + 1e-12
            settled = True
            for k in range(n):
                f = f_obj[k] + lam * f_pen[k]
                m[k] = beta_m * m[k] + (1 - beta_m) * f
                dv = m[k]
                if tau_r >= 0:
                    r[k] = beta_r * r[k] + (1 - beta_r) * f * f
                    dv /= rms if global_norm else np.sqrt(r[k]) + 1e-12
                v[k] += dt * dv
                if v[k] >= 1.0:
                    v[k] = 1.0
                    m[k] = min(m[k], 0.0)  # inelastic wall: drop outward momentum
                elif v[k] <= -1.0:
                    v[k] = -1.0
                    m[k] = max(m[k], 0.0)
                elif abs(dv) > 1e-9:
                    settled = False
            for k in range(n):  # sign changes are applied after the step: all spins saw the same state
                if (v[k] >= 0) != (s[k] > 0):
                    s[k] = -s[k]
                    for j in range(n):  # C is symmetric, so row k is column k
                        f_obj[j] += 2 * s[k] * C_obj[k, j]
                        f_pen[j] += 2 * s[k] * C_pen[k, j]
            step += 1
            if settled and step >= ramp_steps:
                break
        return s, step

    return (analog_run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Outer loop.** As in the repo: analog run, keep the best state, flip a random cluster of the incumbent (the
    cluster shrinks over the iterations, same schedule as `Multiplicative.size_function`), repeat.
    The returned state is the best under the *original* model ($\lambda = 1$). The incumbent we kick from is compared
    under the energy the dynamics currently see ($\lambda_k$ and the multipliers), re-evaluated every iteration, so
    energies at different $\lambda$ are never compared.

    Penalty schedules over the outer iterations $k$:

    - `const`: $\lambda = $ `lam_end` (the baseline, `lam_end = 1`)
    - `lin` / `geo`: ramp `lam_start` → `lam_end` over the first `ramp_frac` of the iterations, then hold
    - `cyclic`: the geometric ramp repeated `cycles` times (re-annealing the penalty)
    - `adaptive`: ×1.25 after an infeasible analog result, ×0.9 after a feasible one, within [`lam_start`, 2 `lam_end`]
    - `alm_rho > 0`: augmented Lagrangian on top of any schedule, $\mu \mathrel{+}= \rho\,\kappa\,(Ax - b)$ after each
      run, which adds $-\tfrac12 A^\top\mu$ to the biases
    """)
    return


@app.cell
def _(Multiplicative, analog_run, np):
    DEFAULTS = dict(
        nb_flips=100,  # outer iterations (nb_flipping in the repo)
        max_steps=5000,  # max Forward-Euler steps per analog run
        init_cluster=0.8,  # cluster-kick sizes, as in the repo's configs
        end_cluster=0.0625,
        exponent=3.0,
        schedule="const",  # const | lin | geo | cyclic | adaptive
        lam_start=0.1,
        lam_end=1.0,
        ramp_frac=0.7,
        cycles=3,
        inner_ramp=0,  # > 0: ramp lam_start -> lam inside every analog run, over this many steps
        alm_rho=0.0,  # > 0: augmented Lagrangian step
        tau_m=0.0,  # momentum time constant [steps], 0 = off
        tau_r=-1.0,  # rms time constant [steps], -1 = off, 0 = sign(field)
        global_norm=False,  # one rms for all spins instead of one per spin
    )

    _cluster_size = Multiplicative().size_function


    def max_field(p):
        """Largest field any spin can feel at lam = 1."""
        return np.max(np.abs(p.C_obj + p.C_pen).sum(1) + np.abs(p.h_obj + p.h_pen))


    def penalty_at(cfg, k):
        """Penalty weight of outer iteration k for the fixed schedules."""
        K, lo, hi = cfg["nb_flips"], cfg["lam_start"], cfg["lam_end"]
        if cfg["schedule"] == "const":
            return hi
        x = k / max(K - 1, 1)  # progress 0 -> 1
        if cfg["schedule"] == "cyclic":
            x = (x * cfg["cycles"]) % 1.0 if k < K - 1 else 1.0
        ramp = min(1.0, x / cfg["ramp_frac"])
        if cfg["schedule"] == "lin":
            return lo + (hi - lo) * ramp
        return lo * (hi / lo) ** ramp  # geo, cyclic


    def solve(p, cfg, seed, v0=None):
        """Multiplicative solver. Returns the best state under lam = 1 and statistics."""
        cfg = {**DEFAULTS, **cfg}
        rng = np.random.default_rng(seed)
        n, K = p.n, cfg["nb_flips"]
        scale = 1.0 / max_field(p)
        C_obj, h_obj, C_pen, h_pen = p.C_obj * scale, p.h_obj * scale, p.C_pen * scale, p.h_pen * scale
        init_size = int(cfg["init_cluster"] * n)
        end_size = max(1, int(cfg["end_cluster"] * n))

        v = rng.uniform(-1, 1, n) if v0 is None else np.asarray(v0, dtype=float)
        lam = cfg["lam_start"] if cfg["schedule"] == "adaptive" else penalty_at(cfg, 0)
        mu = np.zeros(len(p.b)) if cfg["alm_rho"] > 0 else None
        kappa = penalty_per_violation(p) if mu is not None else 0.0

        def dynamics_energy(s):  # the energy the dynamics currently minimise
            e = p.obj(s) + lam * p.pen(s)
            return e if mu is None else e + mu @ p.violation(s)

        best, best_energy, best_feasible_obj = None, np.inf, np.inf
        incumbent, steps, history = None, 0, []
        for k in range(K):
            if cfg["schedule"] != "adaptive":
                lam = penalty_at(cfg, k)
            h_run = h_obj if mu is None else h_obj - 0.5 * scale * (p.A.T @ mu)
            lam_first = cfg["lam_start"] if cfg["inner_ramp"] else lam
            s, nb_steps = analog_run(
                C_obj, h_run, C_pen, h_pen, v, lam_first, lam, cfg["inner_ramp"], cfg["max_steps"],
                cfg["tau_m"], cfg["tau_r"], cfg["global_norm"],
            )  # fmt: skip
            steps += nb_steps

            feasible = p.feasible(s)
            if p.energy(s) < best_energy:
                best, best_energy = s.copy(), p.energy(s)
            if feasible:
                best_feasible_obj = min(best_feasible_obj, p.obj(s))
            if incumbent is None or dynamics_energy(s) < dynamics_energy(incumbent):
                incumbent = s.copy()
            history.append((k, lam, feasible, best_feasible_obj))

            if cfg["schedule"] == "adaptive":  # strategic oscillation around the feasibility boundary
                lam = max(cfg["lam_start"], lam * 0.9) if feasible else min(2 * cfg["lam_end"], lam * 1.25)
            if mu is not None:
                mu += cfg["alm_rho"] * kappa * p.violation(s)

            size = _cluster_size(
                iteration=k, total_iterations=K + int(K == 1), init_size=init_size, end_size=end_size,
                exponent=cfg["exponent"],
            )  # fmt: skip
            v = incumbent.copy()
            v[rng.choice(n, size, replace=False)] *= -1
        return best, dict(steps=steps, best_feasible_obj=best_feasible_obj, history=history)


    def penalty_per_violation(p):
        """kappa in E_pen = kappa * |A x - b|^2, measured on a random state."""
        s = np.where(np.random.default_rng(0).random(p.n) < 0.5, -1.0, 1.0)
        return p.pen(s) / np.sum(p.violation(s) ** 2)

    return DEFAULTS, max_field, penalty_at, solve


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Hierarchical solver.** The sweep of `HierarchicalSolver.solve`: solve the upper model (couplings averaged over
    meta-nodes), then every subproblem with the other spins fixed, then vote the subproblems into the upper state.
    The partitioning, meta-nodes and voting come from the repo's `HierarchicalSolver`. Because the upper model and the
    subproblems are linear in $(C, h)$, the objective/penalty split carries through to every core call, so the core
    solver can ramp the penalty. Extra options:

    - `influence`: where the fixed spins of the other subproblems come from. `"upper"` is the repo's scheme (the
      expanded upper state); `"lower"` uses the lower states of the previous sweep (block Jacobi, still parallel);
      `"sequential"` solves the subproblems one after another with the latest states (block Gauss-Seidel, not parallel).
    - `sweep_ramp`: constant penalty within a sweep, geometric ramp over the sweeps.

    Analog steps count the upper solve plus the slowest subproblem (parallel cores), or the sum for `"sequential"`.
    """)
    return


@app.cell
def _(HierarchicalSolver, Problem, make_modularity_partitioning, np, penalty_at, solve):
    def restrict(p, idx, x):
        """Subproblem on spins idx; the other spins are fixed to x and act as a bias (as get_influence)."""
        x = x.astype(float).copy()
        x[idx] = 0
        ix = np.ix_(idx, idx)
        h_obj, h_pen = p.h_obj[idx] + p.C_obj[idx] @ x, p.h_pen[idx] + p.C_pen[idx] @ x
        return Problem(p.name, p.C_obj[ix], h_obj, p.C_pen[ix], h_pen)


    def coarse_grain(p, labels):
        """Upper model: couplings and biases averaged over the meta-nodes (as make_hierarchy)."""
        P = np.zeros((p.n, labels.max() + 1))
        P[np.arange(p.n), labels] = 1
        P /= P.sum(0)

        def average(C):
            U = P.T @ C @ P
            np.fill_diagonal(U, 0)
            return U

        return Problem(p.name, average(p.C_obj), P.T @ p.h_obj, average(p.C_pen), P.T @ p.h_pen)


    def solve_hierarchical(p, cfg, seed):
        rng = np.random.default_rng(seed)
        hs = HierarchicalSolver()  # used for its partitioning, meta-nodes and voting
        partitioning = make_modularity_partitioning(p.model, cfg["nb_partitions"])
        hs.make_hierarchy(p.model, partitioning, max(cfg["nb_partitions"], int(cfg["meta_frac"] * p.n)))
        hs.upper_state = rng.choice([-1.0, 1.0], hs.upper_model.num_variables)  # as hs.initialize, with our rng
        for sp in hs.subproblems.values():
            sp.initialize(rng.choice([-1.0, 1.0], p.n))
            sp.vote_into(hs.upper_state)
        upper = coarse_grain(p, hs.original_to_upper)

        best_feasible_obj, steps = np.inf, 0
        for sweep in range(cfg["nb_sweeps"]):
            core = cfg["core"]
            if cfg["sweep_ramp"]:
                lam = penalty_at({"schedule": "geo", "lam_start": 0.1, "lam_end": 1.0, "ramp_frac": 0.7,
                                  "nb_flips": cfg["nb_sweeps"]}, sweep)  # fmt: skip
                core = {**core, "schedule": "const", "lam_end": lam}
            seed_sweep = seed + 1000 * (sweep + 1)
            hs.upper_state, info = solve(upper, core, seed_sweep, v0=hs.upper_state)
            core_steps = []
            fixed = hs.expand_upper() if cfg["influence"] == "upper" else hs.assemble_state()
            for i, sp in enumerate(hs.subproblems.values()):
                if cfg["influence"] == "sequential":
                    fixed = hs.assemble_state()
                sp.state, sub_info = solve(restrict(p, sp.original_nodes, fixed), core, seed_sweep + i + 1, sp.state)
                core_steps.append(sub_info["steps"])
            for sp in hs.subproblems.values():
                sp.vote_into(hs.upper_state)
            parallel = cfg["influence"] != "sequential"
            steps += info["steps"] + (max(core_steps) if parallel else sum(core_steps))

            s = hs.assemble_state().astype(float)
            if p.feasible(s):
                best_feasible_obj = min(best_feasible_obj, p.obj(s))
        return s, dict(steps=steps, best_feasible_obj=best_feasible_obj)

    return (solve_hierarchical,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment harness and plots

    Every variant runs with the same seeds. **gap** is the distance of the best feasible objective to the best known
    value, $(E_\text{obj} - E_\text{best}) / |E_\text{best}|$, averaged over the runs that found a feasible solution;
    **feasible** is the fraction of runs that did. **analog steps** is the total number of Forward-Euler steps, i.e.
    analog time, which is what matters for the hardware.
    """)
    return


@app.cell
def _(PROBLEMS, N_RUNS, ThreadPoolExecutor, WORKERS, mo, np, pd, solve, time):
    def run_experiment(name, problem_names, variants, solver=solve, base=None, n_runs=N_RUNS):
        """Run every variant on every problem n_runs times. One row per run."""
        jobs = [
            (PROBLEMS[problem], variant, {**(base or {}), **cfg}, seed)
            for problem in problem_names
            for variant, cfg in variants.items()
            for seed in range(n_runs)
        ]

        def run_one(job):
            p, variant, cfg, seed = job
            start = time.perf_counter()
            _, info = solver(p, cfg, seed)
            gap = (info["best_feasible_obj"] - p.best_found) / abs(p.best_found)
            return dict(
                experiment=name,
                problem=p.name,
                variant=variant,
                seed=seed,
                feasible=bool(np.isfinite(gap)),
                gap=gap if np.isfinite(gap) else np.nan,
                steps=info["steps"],
                time_s=time.perf_counter() - start,
            )

        with ThreadPoolExecutor(WORKERS) as pool:
            rows = list(mo.status.progress_bar(pool.map(run_one, jobs), total=len(jobs), title=name))
        return pd.DataFrame(rows)


    def summarize(df):
        """Per experiment, problem and variant: feasible fraction, gap over the feasible runs, mean analog steps."""
        return (
            df.groupby(["experiment", "problem", "variant"], sort=False)
            .agg(
                runs=("seed", "size"),
                feasible=("feasible", "mean"),
                gap_mean=("gap", "mean"),
                gap_median=("gap", "median"),
                gap_best=("gap", "min"),
                steps=("steps", "mean"),
                time_s=("time_s", "mean"),
            )
            .reset_index()
        )


    def show(df):
        """Summary table with readable numbers."""
        table = summarize(df).drop(columns="experiment")
        formats = {"feasible": "{:.0%}", "gap_mean": "{:.2%}", "gap_median": "{:.2%}", "gap_best": "{:.2%}",
                   "steps": "{:,.0f}", "time_s": "{:.2f}"}  # fmt: skip
        for column, fmt in formats.items():
            table[column] = [fmt.format(x) if pd.notna(x) else "–" for x in table[column]]
        return table

    return run_experiment, show, summarize


@app.cell
def _(LinearSegmentedColormap, NullFormatter, cycler, np, plt):
    # Reference categorical palette (fixed order) and recessive text/grid colours
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    INK, INK_2, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#f1f0ec"
    DIVERGING = LinearSegmentedColormap.from_list("better_worse", ["#7fb0ea", "#f3f2ef", "#f3a07c"])  # blue = better
    SEQUENTIAL = LinearSegmentedColormap.from_list("feasible", ["#f3f2ef", "#7fb0ea"])
    plt.rcParams.update(
        {
            "axes.prop_cycle": cycler(color=SERIES),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 10,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
        }
    )


    def plot_experiment(df, title):
        """One row per problem: gap of every run (dots) and its mean (tick), feasible runs, mean analog steps.
        The first variant (the baseline) is shaded and its mean gap is the dashed line."""
        problems = list(dict.fromkeys(df.problem))
        variants = list(dict.fromkeys(df.variant))
        y = np.arange(len(variants))[::-1]
        jitter = np.random.default_rng(0).uniform(-0.18, 0.18, df.seed.max() + 1)
        fig, axes = plt.subplots(
            len(problems), 3, figsize=(13, (0.3 * len(variants) + 0.9) * len(problems)), sharey=True, squeeze=False,
            gridspec_kw=dict(width_ratios=[3, 1, 1.4]),
        )  # fmt: skip
        for row, problem in zip(axes, problems):
            d = df[df.problem == problem]
            for yi, variant in zip(y, variants):
                g = d[d.variant == variant]
                row[0].scatter(100 * g.gap, yi + jitter[g.seed], s=14, color=SERIES[0], alpha=0.35, linewidths=0)
                row[0].plot(100 * g.gap.mean(), yi, marker="|", ms=14, mew=2.5, color=INK)
                row[1].barh(yi, 100 * g.feasible.mean(), height=0.6, color=SERIES[0])
                row[2].plot(g.steps.mean(), yi, "o", ms=6, color=SERIES[0])
            base_gap = d[d.variant == variants[0]].gap.mean()
            if np.isfinite(base_gap):
                row[0].axvline(100 * base_gap, color=MUTED, lw=1, ls="--")
            for ax in row:
                ax.axhspan(y[0] - 0.5, y[0] + 0.5, color=BAND, zorder=0)
                ax.grid(axis="y", visible=False)
            row[0].set_title(f"{problem}: gap to best known [%]", loc="left")
            row[1].set_title("feasible runs [%]", loc="left")
            row[1].set_xlim(0, 100)
            row[2].set_title("analog steps per run", loc="left")
            if d.steps.notna().any():  # log axis over whole decades
                steps = d.groupby("variant").steps.mean()
                row[2].set_xscale("log")
                row[2].set_xlim(10 ** np.floor(np.log10(steps.min())), 10 ** np.ceil(np.log10(steps.max())))
                row[2].xaxis.set_minor_formatter(NullFormatter())
        axes[0, 0].set_yticks(y, variants)
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
        fig.tight_layout()
        return fig

    return BAND, DIVERGING, INK, MUTED, SEQUENTIAL, SERIES, plot_experiment


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check: our baseline against the repo's Multiplicative solver

    The baseline below is the repo's Multiplicative dynamics with one difference: the time step. The repo sets
    `dt = 0.1 C / (I max|row sum|)` over the coupling matrix *including the frozen bias node's row*, whose sum is
    $\sum_i h_i$. For TSP that row is about $N^2$ times larger than any field a spin can feel, so the repo's `dt` is
    much too small and analog runs time out unless `num_iterations` is huge (`dt_ratio` below). Here
    `dt = 0.1 / max field`. The repo solver gets the same analog time per run (`num_iterations = max_steps × dt_ratio`),
    so both should land in the same range. The repo solver reports only its final state, so its gap is that state's.
    """)
    return


@app.cell
def _(FLAT, MAXCUT_NAMES, Multiplicative, QUICK, max_field, np, pd, run_experiment, show, triu_to_symm):
    def dt_ratio(p):
        """Our time step over the repo's (Multiplicative.solve, line `dtMult = ...`)."""
        model = p.model.transform_to_no_h() if np.linalg.norm(p.model.h) >= 1e-10 else p.model
        dt_repo = 0.1 / np.max(np.abs(np.sum(triu_to_symm(model.J), axis=1)))
        return (0.1 / max_field(p)) / dt_repo


    def repo_multiplicative(p, cfg, seed):
        init = np.random.default_rng(seed).uniform(-1, 1, p.n)
        s, *_ = Multiplicative().solve(
            p.model, init, num_iterations=int(cfg["max_steps"] * dt_ratio(p)), nb_flipping=cfg["nb_flips"],
            cluster_threshold=0.3, init_cluster_size=0.8, end_cluster_size=0.0625, seed=seed + 1,
        )  # fmt: skip
        s = np.asarray(s, dtype=float)
        return s, dict(steps=np.nan, best_feasible_obj=p.obj(s) if p.feasible(s) else np.inf)


    _check_names = ["burma14", "jeu_100_25_1", *MAXCUT_NAMES]
    _budget = {**FLAT, "nb_flips": min(FLAT["nb_flips"], 50)}
    _n = 2 if QUICK else 5
    check_df = pd.concat(
        [
            run_experiment("check", _check_names, {"ours": {}}, base=_budget, n_runs=_n),
            run_experiment("check", _check_names, {"repo": {}}, solver=repo_multiplicative, base=_budget, n_runs=_n),
        ]
    )
    show(check_df)
    return (dt_ratio,)


@app.cell
def _(PROBLEMS, dt_ratio, pd):
    pd.DataFrame({"problem": list(PROBLEMS), "dt_ratio (ours / repo)": [dt_ratio(p) for p in PROBLEMS.values()]})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 1: what the penalty weight does

    Constant $\lambda$ over the whole run. Expected: a lower $\lambda$ gives lower barriers and better objectives, but
    fewer feasible runs; a higher $\lambda$ the opposite. This trade-off is what a ramp tries to get around.
    """)
    return


@app.cell
def _(CONSTRAINED, FLAT, run_experiment):
    exp1 = run_experiment(
        "1: constant penalty",
        CONSTRAINED,
        {
            "baseline": {},  # lam = 1, the repo's model
            "λ = 0.25": {"lam_end": 0.25},
            "λ = 0.5": {"lam_end": 0.5},
            "λ = 2": {"lam_end": 2.0},
            "λ = 4": {"lam_end": 4.0},
        },
        base=FLAT,
    )
    return (exp1,)


@app.cell
def _(exp1, plot_experiment):
    plot_experiment(exp1, "Experiment 1: constant penalty weight")
    return


@app.cell
def _(exp1, show):
    show(exp1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 2: penalty schedules (flat solver)

    All variants end at the repo's penalty ($\lambda = 1$), except the ALM variant that stays at $\lambda = 0.5$ and
    relies on the multipliers for feasibility.

    - **linear / geometric ramp**: $\lambda$ 0.1 → 1 over the first 70% of the outer iterations, then hold
    - **cyclic ramp**: three geometric ramps (re-annealing the penalty after it has frozen the state)
    - **in-run ramp**: $\lambda$ 0.1 → 1 over the first 500 steps of *every* analog run (the hardware-native version)
    - **adaptive**: raise $\lambda$ after an infeasible result, lower it after a feasible one
    - **ALM**: augmented Lagrangian multipliers, at $\lambda = 1$ and at $\lambda = 0.5$
    """)
    return


@app.cell
def _(CONSTRAINED, FLAT, run_experiment):
    SCHEDULES = {
        "baseline": {},
        "linear ramp": {"schedule": "lin"},
        "geometric ramp": {"schedule": "geo"},
        "cyclic ramp (3x)": {"schedule": "cyclic"},
        "in-run ramp": {"inner_ramp": 500},
        "geometric + in-run ramp": {"schedule": "geo", "inner_ramp": 500},
        "adaptive": {"schedule": "adaptive"},
        "ALM (λ = 1)": {"alm_rho": 0.5},
        "ALM (λ = 0.5)": {"alm_rho": 0.5, "lam_end": 0.5},
    }
    exp2 = run_experiment("2: penalty schedules", CONSTRAINED, SCHEDULES, base=FLAT)
    return SCHEDULES, exp2


@app.cell
def _(exp2, plot_experiment):
    plot_experiment(exp2, "Experiment 2: penalty schedules")
    return


@app.cell
def _(exp2, show):
    show(exp2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How the schedules evolve.** One problem, a few seeds: the penalty over the outer iterations (left) and the best
    feasible gap found so far (right, median over the seeds; missing until more than half of the seeds found a
    feasible state).
    """)
    return


@app.cell
def _(
    FLAT,
    INK,
    MaxNLocator,
    PROBLEMS,
    QUICK,
    SCHEDULES,
    SERIES,
    TSP_NAMES,
    ThreadPoolExecutor,
    WORKERS,
    np,
    plt,
    solve,
):
    _problem = PROBLEMS[TSP_NAMES[min(1, len(TSP_NAMES) - 1)]]
    _shown = ["baseline", "linear ramp", "geometric ramp", "cyclic ramp (3x)", "adaptive", "ALM (λ = 0.5)"]
    _seeds = range(2 if QUICK else 8)
    _jobs = [(name, seed) for name in _shown for seed in _seeds]
    with ThreadPoolExecutor(WORKERS) as _pool:
        _results = list(_pool.map(lambda job: solve(_problem, {**FLAT, **SCHEDULES[job[0]]}, job[1]), _jobs))
    _histories = [info["history"] for _, info in _results]

    _fig, (_ax_lam, _ax_gap) = plt.subplots(1, 2, figsize=(13, 4))
    for _color, _name in zip(SERIES, _shown):
        _h = np.array([h for (name, _), h in zip(_jobs, _histories) if name == _name], dtype=float)  # seed, k, field
        _gap = (_h[:, :, 3] - _problem.best_found) / abs(_problem.best_found)  # inf until a feasible state is found
        _median = np.median(_gap, 0)
        _ax_lam.plot(_h[0, :, 0], _h[:, :, 1].mean(0), color=_color, label=_name)
        _ax_gap.plot(_h[0, :, 0], 100 * np.where(np.isfinite(_median), _median, np.nan), color=_color, label=_name)
    _ax_lam.set(title=f"{_problem.name}: penalty weight λ", xlabel="outer iteration", yscale="log")
    _ax_gap.set(title=f"{_problem.name}: best feasible gap so far [%]", xlabel="outer iteration")
    for _ax in (_ax_lam, _ax_gap):
        _ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _ax_gap.legend(loc="upper right", labelcolor=INK)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 3: momentum and normalisation (flat solver)

    - **momentum τ**: heavy-ball dynamics with time constant τ steps. At τ = 0 this is the baseline.
    - **sign(field)**: every spin moves at the same speed (Adam's limit). Expected to oscillate: the
      strongest-field-first ordering of the baseline (a greedy steepest descent) is lost.
    - **rms per spin**: Adam's second moment without momentum; **Adam**: both.
    - **global rms**: divides by one rms over all spins. This is an adaptive *global* time step, without equalising the
      spin speeds, and separates the two effects of Adam's normalisation.

    Max-Cut (G1) is included as an unconstrained reference. Watch the analog steps as well as the gap: QKP has a
    large dynamic range in its fields (binary slack weights up to $2^{12}$), so in the baseline the item spins are
    thousands of times slower than the largest slack spins and the analog runs time out.
    """)
    return


@app.cell
def _(CONSTRAINED, FLAT, MAXCUT_NAMES, run_experiment):
    ADAM = {"tau_m": 10, "tau_r": 30}
    MOMENTUM = {
        "baseline": {},
        "momentum τ = 3": {"tau_m": 3},
        "momentum τ = 10": {"tau_m": 10},
        "momentum τ = 30": {"tau_m": 30},
        "momentum τ = 100": {"tau_m": 100},
        "sign(field)": {"tau_r": 0},
        "rms per spin": {"tau_r": 30},
        "Adam (τm = 10, τr = 30)": ADAM,
        "global rms": {"tau_r": 30, "global_norm": True},
        "global rms + momentum": {**ADAM, "global_norm": True},
    }
    exp3 = run_experiment("3: momentum and normalisation", CONSTRAINED + MAXCUT_NAMES, MOMENTUM, base=FLAT)
    return ADAM, exp3


@app.cell
def _(exp3, plot_experiment):
    plot_experiment(exp3, "Experiment 3: momentum and normalisation")
    return


@app.cell
def _(exp3, show):
    show(exp3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 4: combinations

    Penalty schedules on top of Adam and momentum. Adam runs are much shorter, so the in-run ramp is 100 steps here.
    Edit the dictionary to combine the winners of experiments 2 and 3.
    """)
    return


@app.cell
def _(ADAM, CONSTRAINED, FLAT, run_experiment):
    exp4 = run_experiment(
        "4: combinations",
        CONSTRAINED,
        {
            "baseline": {},
            "Adam": ADAM,
            "geometric ramp + Adam": {"schedule": "geo", **ADAM},
            "in-run ramp + Adam": {"inner_ramp": 100, **ADAM},
            "adaptive + Adam": {"schedule": "adaptive", **ADAM},
            "ALM (λ = 0.5) + Adam": {"alm_rho": 0.5, "lam_end": 0.5, **ADAM},
            "geometric ramp + momentum τ = 10": {"schedule": "geo", "tau_m": 10},
        },
        base=FLAT,
    )
    return (exp4,)


@app.cell
def _(exp4, plot_experiment):
    plot_experiment(exp4, "Experiment 4: combinations")
    return


@app.cell
def _(exp4, show):
    show(exp4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment 5: hierarchical solver

    - **baseline**: the repo's scheme (subproblems see the expanded upper state), plain core solver
    - **core: …**: the penalty ramp / Adam inside every core call
    - **sweeps: geometric ramp**: constant penalty within a sweep, ramped over the sweeps
    - **Jacobi / Gauss-Seidel**: the subproblems see the actual lower states of the other subproblems instead of the
      upper state, from the previous sweep (Jacobi, parallel) or the latest (Gauss-Seidel, sequential). This tests
      whether the coarse upper state is what keeps the hierarchy from producing feasible solutions.
    - **flat solver**: the flat baseline with as many outer iterations as the hierarchy's sweeps × core iterations

    On G1 the penalty variants equal the baseline (no constraints), which doubles as a check.
    """)
    return


@app.cell
def _(ADAM, HIER_BASE, HIER_NAMES, pd, run_experiment, solve_hierarchical):
    def hier(influence="upper", sweep_ramp=False, **core):
        return {**HIER_BASE, "influence": influence, "sweep_ramp": sweep_ramp, "core": {**HIER_BASE["core"], **core}}


    _flat_reference = {
        **HIER_BASE["core"],
        "nb_flips": HIER_BASE["nb_sweeps"] * HIER_BASE["core"]["nb_flips"],
    }
    exp5 = pd.concat(
        [
            run_experiment(
                "5: hierarchical",
                HIER_NAMES,
                {
                    "baseline": hier(),
                    "core: geometric ramp": hier(schedule="geo"),
                    "sweeps: geometric ramp": hier(sweep_ramp=True),
                    "core: Adam": hier(**ADAM),
                    "Jacobi (lower states)": hier("lower"),
                    "Gauss-Seidel": hier("sequential"),
                    "Gauss-Seidel + geometric ramp": hier("sequential", schedule="geo"),
                    "Gauss-Seidel + Adam": hier("sequential", **ADAM),
                },
                solver=solve_hierarchical,
            ),
            run_experiment("5: hierarchical", HIER_NAMES, {"flat solver": _flat_reference}),
        ]
    )
    return (exp5,)


@app.cell
def _(exp5, plot_experiment):
    plot_experiment(exp5, "Experiment 5: hierarchical solver")
    return


@app.cell
def _(exp5, show):
    show(exp5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Overview

    Every variant against the baseline of its own experiment. Left: change in mean gap in percentage points (blue is
    better). Middle: feasible runs. Right: analog time relative to the baseline (blue is faster). Empty cells: the
    variant did not run on that problem, or the variant or its baseline never found a feasible state.
    """)
    return


@app.cell
def _(DIVERGING, INK, SEQUENTIAL, exp1, exp2, exp3, exp4, exp5, np, pd, plt, summarize):
    all_runs = pd.concat([exp1, exp2, exp3, exp4, exp5], ignore_index=True)
    overview = summarize(all_runs)
    _base = overview[overview.variant == "baseline"].set_index(["experiment", "problem"])
    _keys = pd.MultiIndex.from_frame(overview[["experiment", "problem"]])
    overview["d_gap"] = overview.gap_mean.values - _base.gap_mean.reindex(_keys).values
    overview["steps_ratio"] = overview.steps.values / _base.steps.reindex(_keys).values


    def _heatmap(ax, column, title, cmap, vmin, vmax, transform, label):
        table = overview.set_index(["experiment", "variant", "problem"])[column].unstack("problem")
        table = table.reindex(index=pd.MultiIndex.from_frame(_rows), columns=overview.problem.unique())
        values = table.to_numpy(dtype=float)
        ax.imshow(transform(values), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for (i, j), value in np.ndenumerate(values):
            if np.isfinite(value):
                ax.text(j, i, label(value), ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(table.shape[1]), table.columns, rotation=40, ha="right")
        ax.set_yticks(range(table.shape[0]), [f"{e.split(':')[0]} · {v}" for e, v in table.index])
        ax.set_title(title, loc="left")
        ax.grid(False)


    _limit = 100 * max(0.01, np.nanquantile(np.abs(overview.d_gap), 0.95))
    _rows = overview[["experiment", "variant"]].drop_duplicates()
    _fig, _axes = plt.subplots(1, 3, figsize=(17, 0.26 * len(_rows) + 2), sharey=True)
    _heatmap(_axes[0], "d_gap", "Δ mean gap vs baseline [pp]", DIVERGING, -_limit, _limit,
             lambda x: 100 * x, lambda x: f"{100 * x:+.1f}")  # fmt: skip
    _heatmap(_axes[1], "feasible", "feasible runs [%]", SEQUENTIAL, 0, 100,
             lambda x: 100 * x, lambda x: f"{100 * x:.0f}")  # fmt: skip
    _heatmap(_axes[2], "steps_ratio", "analog steps / baseline", DIVERGING, -4, 4, np.log2, lambda x: f"{x:.2g}×")
    _fig.tight_layout()
    _fig
    return all_runs, overview


@app.cell
def _(mo, overview):
    def _line(problem, d):
        base = d[(d.variant == "baseline") & d.experiment.str.startswith(("1", "2", "3", "4"))].iloc[0]
        ok = d[(d.feasible >= 0.8) & d.gap_mean.notna() & ~d.experiment.str.startswith("5")]
        if ok.empty:
            return f"- **{problem}**: no flat variant was feasible in at least 80% of the runs"
        top = ok.sort_values("gap_mean").iloc[0]
        return (
            f"- **{problem}**: best is *{top.variant}* (experiment {top.experiment[0]}): "
            f"mean gap {top.gap_mean:.1%} vs {base.gap_mean:.1%} for the baseline, "
            f"feasible {top.feasible:.0%} vs {base.feasible:.0%}, "
            f"analog time {top.steps / base.steps:.2f}× the baseline"
        )


    def _hier_line(problem, d):
        d = d[d.experiment.str.startswith("5")]
        if d.empty:
            return None
        top = d.sort_values(["feasible", "gap_mean"], ascending=[False, True]).iloc[0]
        return f"- **{problem}**: *{top.variant}*, feasible {top.feasible:.0%}, mean gap {top.gap_mean:.1%}"


    _groups = list(overview.groupby("problem", sort=False))
    _hier = [line for line in (_hier_line(problem, d) for problem, d in _groups) if line]
    mo.md(
        "### Best flat variant per problem\n(feasible in ≥ 80% of the runs, lowest mean gap)\n\n"
        + "\n".join(_line(problem, d) for problem, d in _groups)
        + "\n\n### Best hierarchical variant per problem\n(most feasible, then lowest mean gap)\n\n"
        + "\n".join(_hier)
    )
    return


@app.cell
def _(SAVE_TO, all_runs, mo):
    SAVE_TO.parent.mkdir(parents=True, exist_ok=True)
    all_runs.to_csv(SAVE_TO, index=False)
    mo.md(f"Saved {len(all_runs)} runs to `{SAVE_TO}`.")
    return


if __name__ == "__main__":
    app.run()
