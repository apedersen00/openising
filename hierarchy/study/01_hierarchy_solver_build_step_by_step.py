"""
hierarchy solver
input: J and h of the original problem
steps:
1. create partitions
2. create metanodes
3. create upper model
4. solve
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # hierarchy solver
    input: J and h of the original problem
    steps:
    1. create partitions
    2. create metanodes
    3. create upper model
    4. solve
    """)
    return


@app.cell
def _():
    import pathlib
    import sys

    import marimo as mo
    import numpy as np

    # add the ising hierarchy to path
    PATH = pathlib.Path(__file__).parent.parent.parent / "hierarchy" / "ising_hierarchy"
    sys.path.append(str(PATH))

    from core import Core
    from logger import configure_logger
    from node import Node
    from problems import hand_dummy, load_maxcut, generate_nodes
    from metanode import group_nodes, make_metanodes, connect_metanodes, vote

    return (
        Core,
        Node,
        configure_logger,
        connect_metanodes,
        generate_nodes,
        group_nodes,
        hand_dummy,
        load_maxcut,
        make_metanodes,
        mo,
        np,
        vote,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A. get the original problem
    """)
    return


@app.cell
def _(np):
    # For now only a dummy model
    def energy(J, h, s):
        """
        Compute the energy of a configuration s for the Ising model defined by J and h.
        E(s) = -1/2 * sum_{i,j} J_{ij} s_i s_j - sum_i h_i s_i
        """
        return -0.5 * np.sum(J * np.outer(s, s)) - np.dot(h, s)

    def brute_force(J, h):
        """
        Test all the possible spin configurations and return the best energy and all the states that reach it.
        """
        best, states = +np.inf, []
        n = len(h)
        for s in range(2**n):
            # convert s in a list repr
            s_bin = np.array([(s >> i) & 1 for i in range(n)])
            s_bin = 2 * s_bin - 1  # convert to -1, 1
            e = energy(J, h, s_bin)
            if e < best:
                best = e
                states = [s_bin]
            elif e == best:
                states.append(s_bin)

        return best, states

    return brute_force, energy


@app.cell
def _(brute_force, energy, hand_dummy, np):
    J, h = hand_dummy()

    brute_force_energy, brute_force_states = brute_force(J, h)
    print("Energy of all ones:", energy(J, h, np.ones(len(h))))
    print("Energy of all minus ones:", energy(J, h, -np.ones(len(h))))
    print("Energy of random configuration:", energy(J, h, np.random.choice([-1, 1], size=len(h))))
    print("Brute force solution:", brute_force_energy)
    for state in brute_force_states:
        print("Brute force state:", state, "Energy:", energy(J, h, state))
    return J, h


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # B. one core: the whole problem
    """)
    return


@app.cell
def _(Core, J, Node, configure_logger, h):
    configure_logger()

    # nodes and core in the same cell: a re-run rebuilds both, so no node is added twice to a core
    N = len(h)
    nodes = [Node(i, h[i]) for i in range(N)]
    for i in range(N):
        for j in range(N):
            if J[i, j] != 0:
                nodes[i].add_neighbor(nodes[j], J[i, j])


    core = Core(0, nodes)
    core.init_J_h()
    print(core)
    print(nodes[0])
    return core, nodes


@app.cell
def _(core):
    # marimo does not see changes inside core: return the results, so the cells using them re-run after solve
    core.solve()
    history, best_spins, best_energy = core.history, core.best_spins, core.best_energy
    return


@app.cell
def _(core):
    core.best_spins, core.best_energy, core.best_history
    return


@app.cell
def _(nodes):
    for node in nodes:
        print(node)
    return


@app.cell
def _(Core, generate_nodes, load_maxcut):
    PROBLEMS = [
        "G/regenerated/G32_2000_4000_toroidal_1410",
        "G/regenerated/G29_2000_19990_random_3405",
        "G/regenerated/G40_2000_11766_planar_2400",
        (1024, 0.02)
    ]

    J_p, h_p = load_maxcut(PROBLEMS[0])
    nodes_p = generate_nodes(J_p, h_p)
    core_p = Core(0, nodes_p)
    return (core_p,)


@app.cell
def _(core_p):
    core_p.solve(nb_iterations=200, nb_seeds=80)
    print("Best energy:", core_p.best_energy)
    return


@app.cell
def _(core_p):
    iterations = [50, 100, 200, 400, 800]
    nb_seeds   = [10,  20,  40,  80, 160]
    energies   = {}
    for n_iter in iterations:
        for n_seeds in nb_seeds:
            core_p.solve(nb_iterations=n_iter, nb_seeds=n_seeds)
            energies[(n_iter, n_seeds)] = core_p.best_energy
            print(f"Best energy for {n_iter} iterations and {n_seeds} seeds: {core_p.best_energy}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Partitions
    start simple
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What i want to do
    1. Make "no top level, one core after the other" the baseline in your notebook.
    2. Add the parallel version, since that is what the hardware does.
    3. Judge any hierarchy by whether it gets the parallel version back to the baseline at the same cost.
    """)
    return


@app.cell
def _(Core, J, generate_nodes, h, np):
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    nodes_2 = generate_nodes(J, h)
    cores = [Core(c, [nodes_2[i] for i in np.flatnonzero(labels == c)]) for c in np.unique(labels)]
    for core_id in cores:
        core_id.init_J_h()

    return (cores,)


@app.cell
def _(J, cores, energy, h, np):
    _state = np.random.default_rng(0).choice([-1.0, 1.0], len(h))   # every spin of the problem, by node.row
    next_state = _state.copy()
    for _round in range(3):
        for _core in cores:
            _ids = [_node.row for _node in _core.nodes]
            _core.field_from_outside(_state)                         # sets _core.f from the current _state
            _core.solve(nb_iterations=20, nb_seeds=4)
            next_state[_ids] = _core.best_spins                          # write back: the next core sees it
        _state = next_state
        print(_round, energy(J, h, _state))
    return


@app.cell
def _(mo):
    # now lets create some metanodes
    nodes_per_metanode = mo.ui.slider(1, 8, value=3, label="nodes per metanode")
    method = mo.ui.dropdown(["linkage", "original"], value="linkage", label="method")
    mo.hstack([nodes_per_metanode, method])
    return method, nodes_per_metanode


@app.cell
def _(
    Core,
    J,
    connect_metanodes,
    generate_nodes,
    group_nodes,
    h,
    make_metanodes,
    method,
    nodes_per_metanode,
):
    nodes_m = generate_nodes(J, h)
    groups = group_nodes(J, nodes_per_metanode.value, method.value)
    metanodes = make_metanodes(nodes_m, groups)
    connect_metanodes(metanodes)
    upper_core = Core(0, metanodes)
    upper_core.init_J_h()
    return (metanodes,)


@app.cell
def _(cores, h, metanodes, np, vote):
    lower_state = np.zeros(len(h))                     # spins of level 0, by node.row
    for lower_core in cores:
        lower_core.init_J_h()                          # f = 0: each core alone, no field from the rest
        lower_core.solve()
        lower_state[[_node.row for _node in lower_core.nodes]] = lower_core.best_spins
    upper_state = vote(metanodes, lower_state, np.ones(len(metanodes)))
    print(lower_state, upper_state)
    return lower_state, upper_state


@app.cell
def _(np):
    def expand(nodes, upper_state, state):
        """
        Majority vote, upper -> lower: each node takes the sign of the sum of the spins of its metanodes, and keeps its
        spin in state on a tie (or with no metanode). With one metanode per node: the spin of its metanode.

        - nodes: the nodes of the level below
        - upper_state: spins of the metanodes, by metanode.row
        - state: current spins of the level below, by node.row
        Returns the spins of the level below as the level above sees them, by node.row.
        """
        expanded = state.copy()
        for node in nodes:
            total = sum(upper_state[metanode.row] for metanode in node.in_metanode)
            if total != 0: # keep previous spin value
                expanded[node.row] = np.sign(total)
        return expanded

    return (expand,)


@app.cell
def _(cores, expand, lower_state, nodes, upper_state):
    seen = expand(nodes, upper_state, lower_state)
    for lower_core_2 in cores:
        lower_core_2.field_from_outside(seen)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
