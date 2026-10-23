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
def _(busclique, minorminer, nx, wu_chip, wu_four):
    import sys
    import pandas as pd
    from ising.stages.tsp_parser_stage import TSPParserStage
    from ising.stages.mimo_parser_stage import MIMOParserStage
    from ising.stages.maxcut_parser_stage import MaxcutParserStage
    from ising.generators.TSP import TSP
    from ising.stages import TOP
    sys.path.insert(0, str(TOP / 'scripts'))
    from qkp import load_qkp, knapsack_to_qubo

    # ---------- the problems, each as its J matrix (only which entries are non-zero matters) ----------
    def _maxcut(file):
        return MaxcutParserStage.generate_maxcut(MaxcutParserStage.G_parser(file)[0]).J

    def _mimo(file):
        _H, _x, _M, _ant, _users = MIMOParserStage.parse_MIMO(TOP / 'ising/benchmarks/MIMO' / file)
        return MIMOParserStage.MIMO_to_Ising(_H, _x[:, 0], 10, _users, _ant, _M, seed=1)[0].J   # first received signal

    _rudy = TOP / 'ising/benchmarks/biqmac/rudy'
    _w, _P, _C = load_qkp('jeu_100_25_1')
    _Q, _A, _b = knapsack_to_qubo(_w, _P, _C)

    bench_J = {
        'Max-Cut pm1s_100': _maxcut(_rudy / 'pm1s_100.0'),
        'Max-Cut g05_100': _maxcut(_rudy / 'g05_100.0'),
        'Max-Cut pm1d_100': _maxcut(_rudy / 'pm1d_100.0'),
        'Max-Cut G14': _maxcut(TOP / 'ising/benchmarks/G/G14.txt'),
        'QKP jeu_100_25_1 penalty': _Q + _A.T @ _A,         # constraint as squared penalty, slack bits included
        'QKP jeu_100_25_1 objective': _Q[:100, :100],       # option B: constraint outside the grid, items only
        'TSP burma14': TSP(TSPParserStage.TSP_parser(TOP / 'ising/benchmarks/TSP/burma14.tsp')[0]).J,
        'MIMO 16 users 4-QAM': _mimo('user16_ant32_M4.txt'),
        'MIMO 64 users 16-QAM': _mimo('user64_ant256_M16.txt'),
    }

    # ---------- embed every problem and count hardware spins ----------
    def _count(emb):
        return sum(len(chain) for chain in emb.values()) if emb else None    # None = doesn't fit

    _cache_chip = busclique.busgraph_cache(wu_chip)       # loads from disk, built in the cell above
    _cache_four = busclique.busgraph_cache(wu_four)

    _rows = []
    for _name, _J in bench_J.items():
        _G = nx.from_numpy_array(_J != 0)                 # problem graph: which pairs have a non-zero J
        _G.remove_edges_from(nx.selfloop_edges(_G))       # the diagonal holds biases, not couplings
        _n, _e = _G.number_of_nodes(), _G.number_of_edges()
        _density = _e / (_n * (_n - 1) / 2)

        # worst case: treat the problem as fully connected (busclique, instant)
        _fc_chip = _count(_cache_chip.find_clique_embedding(_n))
        _fc_four = _count(_cache_four.find_clique_embedding(_n))

        # problem-specific (minorminer, slow): try one chip, then four; skipped when (almost) fully connected
        _mm, _mm_on = None, 'skipped (dense)'
        if _density < 0.9:
            for _hw_name, _hw in [('1 chip', wu_chip), ('4 chips', wu_four)]:
                _mm = _count(minorminer.find_embedding(_G.edges, _hw.edges, random_seed=1, tries=3, timeout=300))
                _mm_on = _hw_name if _mm else 'failed'
                if _mm:
                    break

        _found = [_x for _x in (_fc_chip, _fc_four, _mm) if _x]
        _rows.append({
            'problem': _name,
            'spins': _n,
            'couplings': _e,
            '% pairs coupled': round(100 * _density, 1),
            'fully connected, 1 chip': _fc_chip,
            'fully connected, 4 chips': _fc_four,
            'minorminer': _mm,
            'minorminer on': _mm_on,
            'HW spins per problem spin': round(min(_found) / _n, 1) if _found else None,
        })
        print(f'done: {_name}')

    bench_table = pd.DataFrame(_rows)
    bench_table.to_csv(TOP / 'data/pegasus_embedding.csv', index=False)
    bench_table
    return TOP, pd


@app.cell
def _(TOP, pd, plt):
    from pathlib import Path

    bench_table = pd.read_csv(TOP / 'data/pegasus_embedding.csv')       # the table from the cell above
    _t = bench_table.sort_values('HW spins per problem spin', ascending=False, na_position='first')
    _labels = [f'{p}  ({n} spins, {c:.0f}% coupled)' for p, n, c in zip(_t['problem'], _t['spins'], _t['% pairs coupled'])]
    _ratio = _t['HW spins per problem spin']

    _fig, _ax = plt.subplots(figsize=(9, 5))
    _ax.barh(_labels, _ratio.fillna(0), color='tab:blue')
    for _y, _r in enumerate(_ratio):                                     # write the number next to each bar
        _ax.text(_r + 0.2 if pd.notna(_r) else 1.3, _y, f'{_r:.1f}' if pd.notna(_r) else "doesn't fit", va='center')
    _ax.axvline(1, color='k', ls='--', label='crossbar: 1 per problem spin')
    _ax.set_xlabel('Pegasus hardware spins per problem spin')
    _ax.set_title("Real problems on Wu et al.'s Pegasus chips")
    _ax.legend(loc='lower right')
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
