"""
Benchmark the multi-core machine (hwmodel/fast.py) on openising's problems.

Every problem runs on one big crossbar, on 4 cores on one chip and on 4 cores on 4 chiplets, while R is swept.
R is the number of clock cycles per full-scale ramp: how fast the digital side is compared to the analog. Small R =
slow digital or fast analog. The clock stays at 1 GHz, so R sets I_unit: the largest coupling gives 20 uA / R.

    python scripts/bench_hwmodel.py                 # every problem
    python scripts/bench_hwmodel.py G14 burma14     # some problems

Quantization is off (32-bit cells, exact h and far DAC). TSP and QKP use openising's default penalty weight (1).
Every setting uses the same seeds.

Output in data/hwmodel/bench/:
    meta.json                       commit, date and the constants below
    problems.csv                    name, family, kind, n, density (of J), best (best known cost)
    <problem>/<config>_R<R>.csv     one row per run: seed, run, E (openising's energy), cost, t_end (ns from the
                                    start to the end of the run), flips (during the run), t_max (the run hit t_max)

cost is the objective, lower is better: -cut (Max-Cut), x Q x (QUBO), tour length (TSP) and -profit (QKP).
NaN = the state breaks a constraint. Time in full-scale ramps is t_end / R.
Finished files are skipped, so a stopped benchmark can be restarted.

*ai-generated*
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import json
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hwmodel import fast
from hwmodel.spec import MachineSpec
from ising.generators.TSP import TSP
from ising.stages.biqmac_parser_stage import BiqMacParserStage
from ising.stages.maxcut_parser_stage import MaxcutParserStage
from ising.stages.qkp_parser_stage import QKPParserStage
from ising.stages.tsp_parser_stage import TSPParserStage

WORKERS = min(30, os.cpu_count())
SEEDS = 10
RUNS = 300                              # restarts per solve
R_LIST = [4, 8, 16, 32, 64, 128, 256]   # clock cycles per full-scale ramp
CELL_BITS = 32                          # quantization off
T_QUIET = 7.0                           # read out after this many full-scale ramps without activity
T_MAX = 52.5                            # read out at the latest after this many full-scale ramps
BENCH = ROOT / 'ising' / 'benchmarks'
OUT = ROOT / 'data' / 'hwmodel' / 'bench'

CONFIGS = {                             # cores, mesh (tiles per side), chip (tiles per chip side)
    'crossbar': (1, 1, 1),              # one core with every spin: no mesh, no far units
    '4core':    (4, 2, 2),              # 2 x 2 cores on one chip
    '4chiplet': (4, 2, 1),              # 2 x 2 cores, every core on its own chip (t_chip per chip edge)
}

PROBLEMS = {                            # name: family, kind, file in ising/benchmarks
    'G1':                   ('maxcut', 'random 6% +1',          'G/G1.txt'),
    'G6':                   ('maxcut', 'random 6% ±1',          'G/G6.txt'),
    'G11':                  ('maxcut', 'toroidal ±1',           'G/G11.txt'),
    'G14':                  ('maxcut', 'planar +1',             'G/G14.txt'),
    'G18':                  ('maxcut', 'planar ±1',             'G/G18.txt'),
    'G22':                  ('maxcut', 'random 1% +1',          'G/G22.txt'),
    'G32':                  ('maxcut', 'toroidal ±1',           'G/G32.txt'),
    'G35':                  ('maxcut', 'planar +1',             'G/G35.txt'),
    'K2000':                ('maxcut', 'complete ±1',           'G/K2000.txt'),
    'w05_100.0':            ('maxcut', 'random 50% -10..10',    'biqmac/rudy/w05_100.0'),
    't2g20_5555':           ('maxcut', '2D toroidal gaussian',  'biqmac/ising/t2g20_5555'),
    't3g7_5555':            ('maxcut', '3D toroidal gaussian',  'biqmac/ising/t3g7_5555'),
    'ising2.5-300_5555':    ('maxcut', '1D chain long range',   'biqmac/ising/ising2.5-300_5555'),
    'bqp250-1':             ('qubo',   'beasley 10%',           'biqmac/beasley/bqp250-1.sparse'),
    'bqp500-1':             ('qubo',   'beasley 10%',           'biqmac/beasley/bqp500-1.sparse'),
    'gka3f':                ('qubo',   'gka 50%',               'biqmac/gka/gka3f.sparse'),
    'be200.8.1':            ('qubo',   'be 80%',                'biqmac/be/be200.8.1.sparse'),
    'burma14':              ('tsp',    'geographic',            'TSP/burma14.tsp'),
    'gr21':                 ('tsp',    'distance matrix',       'TSP/gr21.tsp'),
    'fri26':                ('tsp',    'distance matrix',       'TSP/fri26.tsp'),
    'bayg29':               ('tsp',    'distance matrix',       'TSP/bayg29.tsp'),
    'jeu_100_25_1':         ('qkp',    'profits 25%',           'Knapsack/jeu_100_25_1.txt'),
    'jeu_100_100_1':        ('qkp',    'profits 100%',          'Knapsack/jeu_100_100_1.txt'),
    'jeu_200_50_1':         ('qkp',    'profits 50%',           'Knapsack/jeu_200_50_1.txt'),
    'jeu_300_25_1':         ('qkp',    'profits 25%',           'Knapsack/jeu_300_25_1.txt'),
}


def tour(S, W):
    """
    Tour length of every state (row of S), NaN when it is not a tour. Spin city * N + time is +1 when the city is
    visited at that time.
    """
    N = len(W)
    X = S.reshape(len(S), N, N) > 0
    ok = (X.sum(axis=1) == 1).all(axis=1) & (X.sum(axis=2) == 1).all(axis=1)
    order = X.argmax(axis=1)                                            # city at every time
    return np.where(ok, W[order, np.roll(order, -1, axis=1)].sum(axis=1), np.nan)


def knapsack(S, P, w, C):
    """
    -profit of every state, NaN when the items are over capacity. The first len(w) spins are the items, the rest slack.
    """
    X = (S[:, :len(w)] > 0).astype(float)
    return np.where(X @ w <= C, -((X @ np.triu(P)) * X).sum(axis=1), np.nan)


def load(name):
    """
    openising's Ising model as hwmodel takes it (J symmetric, E = -1/2 s J s - h s), the constant c of openising's
    energy, the best known cost and cost(S). cost None: the cost is the energy.
    """
    family, _, file = PROBLEMS[name]
    path = BENCH / file
    cost = None
    if family == 'maxcut':
        graph, best = MaxcutParserStage.G_parser(path)
        model = MaxcutParserStage.generate_maxcut(graph)
    elif family == 'qubo':
        graph, best = BiqMacParserStage.biqmac_parser(path)
        model = BiqMacParserStage.generate_biqmac(graph)
    elif family == 'tsp':
        graph, best = TSPParserStage.TSP_parser(path)
        model = TSP(graph)
        cost = partial(tour, W=nx.adjacency_matrix(graph).toarray())
    else:
        graph, best = QKPParserStage.QKP_parser(path)                  # as QKPParserStage.generate_knapsack
        N = len(graph.nodes)
        P = np.zeros((N, N))
        for (i, j), value in nx.get_edge_attributes(graph, 'profit').items():
            P[i, j] = P[j, i] = value
        w = np.array([graph.edges[i, i]['weight'] for i in range(N)], dtype=float)
        C = graph.graph['capacity']
        model = QKPParserStage.knapsack_to_ising(P, C, w, 1.0)
        cost = partial(knapsack, P=P, w=w, C=C)
    return (model.J + model.J.T).astype(float), model.h.astype(float), float(model.c), best, cost


def make_spec(config, R, n, seed):
    """
    The default spec with the config's cores, R cycles per full-scale ramp (sets I_unit) and quantization off.
    """
    K, mesh, chip = CONFIGS[config]
    s = MachineSpec()
    full = 2**CELL_BITS - 1                                             # full-scale cell current in I_unit
    I_unit = s.node.C * (s.node.V_high - s.node.V_th) * s.clock.f_core / (R * full)
    ramp = s.node.ramp_time(I_unit * full)                              # ps
    return replace(s, node=replace(s.node, I_unit=I_unit),
                   core=replace(s.core, c=-(-n // K), cell_bits=CELL_BITS, h_bits=None),
                   net=replace(s.net, mesh=mesh, chip=chip, dither=2),
                   control=replace(s.control, n_runs=RUNS, kick_mode='local', t_quiet=round(T_QUIET * ramp),
                                   t_max=round(T_MAX * ramp), seed=seed))


def task(args):
    """
    Every seed of one problem, config and R. Writes <problem>/<config>_R<R>.csv.
    """
    name, config, R = args
    J, h, c, _, cost = load(name)
    rows = []
    for seed in range(SEEDS):
        ctl = fast.Controller(J, h, make_spec(config, R, len(h), seed))
        ctl.solve()
        S = np.array(ctl.end, dtype=float)
        E = -0.5 * ((S @ J) * S).sum(axis=1) - S @ h + c
        t_end, t_run = np.array(ctl.t_end), np.array(ctl.t_run)
        t_flip = ctl.machine.flips_t[:ctl.machine.n_flips]
        rows.append(pd.DataFrame(dict(
            seed=seed, run=np.arange(RUNS), E=E, cost=E if cost is None else cost(S), t_end=t_end / 1e3,
            flips=np.searchsorted(t_flip, t_end, 'right') - np.searchsorted(t_flip, t_end - t_run, 'left'),
            t_max=np.array(ctl.why) == 't_max')))
    file = OUT / name / f'{config}_R{R}.csv'
    pd.concat(rows).to_csv(file.with_suffix('.tmp'), index=False)
    file.with_suffix('.tmp').rename(file)
    return name, config, R


def main(names):
    OUT.mkdir(parents=True, exist_ok=True)
    problems = []
    for name in PROBLEMS:
        J, h, _, best, _ = load(name)
        n = len(h)
        problems.append(dict(name=name, family=PROBLEMS[name][0], kind=PROBLEMS[name][1], n=n,
                             density=np.count_nonzero(J) / (n * (n - 1)), best=best))
        (OUT / name).mkdir(exist_ok=True)
    problems = pd.DataFrame(problems)
    problems.to_csv(OUT / 'problems.csv', index=False)
    meta = dict(commit=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
                date=datetime.now().isoformat(), seeds=SEEDS, runs=RUNS, R=R_LIST, cell_bits=CELL_BITS,
                t_quiet=T_QUIET, t_max=T_MAX, configs=CONFIGS)
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=2))

    n = dict(zip(problems.name, problems.n))
    tasks = [(name, config, R) for name in sorted(names, key=lambda x: -n[x]) for config in CONFIGS for R in R_LIST
             if not (OUT / name / f'{config}_R{R}.csv').exists()]
    print(f'{len(tasks)} tasks on {WORKERS} workers', flush=True)
    t0 = time.perf_counter()
    with Pool(WORKERS) as pool:
        for i, (name, config, R) in enumerate(pool.imap_unordered(task, tasks)):
            print(f'{i + 1}/{len(tasks)}  {name} {config} R={R}  {time.perf_counter() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:] or list(PROBLEMS))
