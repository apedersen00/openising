import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pegasus Graphs

    ## Imports
    """)
    return


@app.cell
def _():
    import networkx as nx
    import dwave_networkx as dnx
    import minorminer
    import matplotlib.pyplot as plt
    from collections import Counter
    import numpy as np

    return Counter, dnx, minorminer, np, nx, plt


@app.cell
def _(np, nx):
    # Three spin graph:
    # 1--2
    # \
    #  0

    J_small = np.array([[0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.5],
                        [0.0, 0.5, 0.0]])
    G_small = nx.from_numpy_array(J_small)

    print(f'Num spins: {G_small.nodes}')
    print(f'Num edges: {G_small.edges}')
    print('Degrees:')
    for _i in range(len(G_small.degree())):
        print(G_small.degree()[_i])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embedding
    """)
    return


@app.cell
def _(minorminer, nx, plt):
    triangle = nx.complete_graph(3) # problem graph: spins 0, 1, 2 all coupled
    ring = nx.cycle_graph(4)        # hardware graph: spins 0-1-2-3-0
    emb_tri = minorminer.find_embedding(triangle.edges, ring.edges, random_seed=0)

    print(f'Problem spin 0 = HW spin: {emb_tri[0]}')
    print(f'Problem spin 1 = HW spin: {emb_tri[1]}')
    print(f'Problem spin 2 = HW spin: {emb_tri[2]}')

    _colors = ['tab:red', 'tab:blue', 'tab:green']                       # one colour per problem spin
    _owner = {hw: p for p, chain in emb_tri.items() for hw in chain}     # hardware spin -> problem spin
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(8, 4))
    nx.draw(triangle, ax=_a1, with_labels=True, node_color=_colors, node_size=600)
    _a1.set_title('problem: 3 spins, all coupled')
    _pos = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (0, 0)}                 # draw the ring as a square
    nx.draw(ring, _pos, ax=_a2, with_labels=True, node_color=[_colors[_owner[h]] for h in ring], node_size=600)
    _a2.set_title('hardware: 4 spins in a ring')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pegasus

    - Problem graph: the $J_{ij}$ the problem needs
    - Hardware graph: the $J_{ij}$ the chip can use

    Pegasus is the hardware graph of D-wave. Each spin has edges (couplers) to at most 15 others.
    """)
    return


@app.cell
def _(Counter, dnx):
    peg6        = dnx.pegasus_graph(6)                                  # build hardware pegasus graph of size 6
    n_spins6    = peg6.number_of_nodes()
    n_couplers6 = peg6.number_of_edges()
    degrees6    = sorted(Counter(d for _, d in peg6.degree()).items())

    print(f'Num spins: {n_spins6}')
    print(f'Num edges: {n_couplers6}')
    print('Degrees (num spins, edges):')
    for _i in range(len(degrees6)):
        print(degrees6[_i])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Create Pegasus HW gtaph of size 3 (128 spins)
    - Define a complete graph problem with 6 spins
    - Embed that graph to the HW graph
    """)
    return


@app.cell
def _(dnx, minorminer, nx, plt):
    problem = nx.complete_graph(24)
    peg3 = dnx.pegasus_graph(3)
    emb = minorminer.find_embedding(problem.edges, peg3.edges, random_seed=1)

    hw_spins = 0
    for _i in range(len(emb)):
        hw_spins += len(emb[_i])
        print(f'Problem spin {_i} = HW spin: {emb[_i]}')

    print(f'\nProblem spins: {len(emb)}\nHardware spins: {hw_spins}')

    spin_colors = {p: plt.cm.tab10(p) for p in problem}     # one colour per problem spin
    _fig, _ax = plt.subplots(figsize=(4, 4))
    nx.draw(problem, nx.circular_layout(problem), ax=_ax, with_labels=True,
            node_color=[spin_colors[p] for p in problem], node_size=600)
    _ax.set_title('problem: 6 spins, all coupled')
    _fig
    return emb, peg3, problem, spin_colors


app._unparsable_cell(
    r"""
    plt.figure(figsize=(6, 6))
    dnx.draw_pegasus_embedding(peg3, emb, node_size=20)
    plt.gca()n_spins6    = peg6.number_of_nodes()
    n_couplers6 = peg6.number_of_edges()
    degrees6    = sorted(Counter(d for _, d in peg6.degree()).items())

    print(f'Num spins: {n_spins6}')
    """,
    name="_"
)


@app.cell
def _(dnx, emb, nx, peg3, plt, problem, spin_colors):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(15, 7.5), width_ratios=[1.4, 1.4])

    # left: the problem graph, spins on a circle
    nx.draw(problem, nx.circular_layout(problem), ax=_a1, with_labels=True,
            node_color=[spin_colors[p] for p in problem], node_size=600)
    _a1.set_title(f'problem: {problem.number_of_nodes()} spins')

    # right: the same spins mapped onto Pegasus, same colours
    dnx.draw_pegasus_embedding(peg3, emb, chain_color=spin_colors, node_size=30, ax=_a2)
    _a2.set_title(f'Pegasus: {sum(len(chain) for chain in emb.values())} hardware spins')
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How does it scale?
    """)
    return


@app.cell
def _(dnx, np, plt):
    from minorminer import busclique

    peg16 = dnx.pegasus_graph(16)                    # 5,640 hardware spins
    clique_cache = busclique.busgraph_cache(peg16)

    _n_spins    = peg16.number_of_nodes()
    print(f'Num spins: {_n_spins}')

    n_problem = np.arange(2, 256)                    # 180 is the largest fully connected problem P16 can hold
    n_hardware = []
    for _n in n_problem:
        _emb = clique_cache.find_clique_embedding(_n)                       # same dict as find_embedding
        n_hardware.append(sum(len(_chain) for _chain in _emb.values()))
    n_hardware = np.array(n_hardware)

    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.set_title(f'Pegasus HW graph with {_n_spins} spins')
    _ax.plot(n_problem, n_hardware / n_problem, label='Pegasus')
    _ax.axhline(1, color='k', ls='--', label='galena')
    _ax.set_xlabel('problem spins (fully connected)')
    _ax.set_ylabel('hardware spins per problem spin')
    _ax.legend()
    _fig
    return (busclique,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Real problems mapped to Pegasus-15
    """)
    return


@app.cell
def _(dnx):
    _big = dnx.pegasus_graph(49, nice_coordinates=True)     # big enough to cut both shapes out of
    wu_chip = _big.subgraph([s for s in _big if s[1] < 12 and s[2] < 24]).copy()   # 12 x 24 cells
    wu_four = _big.subgraph([s for s in _big if s[1] < 48 and s[2] < 24]).copy()   # 48 x 24 cells (4 chips)
    wu_chip.number_of_nodes(), wu_four.number_of_nodes()
    return wu_chip, wu_four


@app.cell
def _(busclique, wu_chip, wu_four):
    wu_chip_clique = busclique.busgraph_cache(wu_chip).largest_clique()
    wu_four_clique = busclique.busgraph_cache(wu_four).largest_clique()
    len(wu_chip_clique), len(wu_four_clique)
    return


@app.cell
def _(np):
    from ising.stages import TOP                                   # path to the openising folder
    from ising.stages.maxcut_parser_stage import MaxcutParserStage

    maxcut_file = TOP / 'ising/benchmarks/biqmac/rudy/g05_100.0'
    maxcut_graph, _ = MaxcutParserStage.G_parser(maxcut_file)       # the problem graph: spins and coupled pairs
    maxcut_model = MaxcutParserStage.generate_maxcut(maxcut_graph)   # the Ising model openising builds from it
    maxcut_graph.number_of_nodes(), maxcut_graph.number_of_edges(), np.count_nonzero(maxcut_model.J)
    return (maxcut_graph,)


@app.cell
def _(maxcut_graph, minorminer, wu_chip):
    maxcut_emb = minorminer.find_embedding(maxcut_graph.edges, wu_chip.edges, random_seed=1)
    maxcut_hw = sum(len(chain) for chain in maxcut_emb.values())
    maxcut_hw, maxcut_hw / maxcut_graph.number_of_nodes(), max(len(chain) for chain in maxcut_emb.values())
    return


if __name__ == "__main__":
    app.run()
