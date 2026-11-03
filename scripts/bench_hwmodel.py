"""
Benchmark the multi-core machine (hwmodel/fast.py) against the ideal single crossbar (solver_event, no digital).
Every setting uses the same seeds, so settings can be compared seed by seed.

    python scripts/bench_hwmodel.py E0              # every problem
    python scripts/bench_hwmodel.py E0 G14 SK512    # some problems

R = full-scale ramp / clock period (cycles per ramp). Digital times stay in cycles, t_quiet and t_max are set in ramps.

Output in data/hwmodel/<experiment>/:
    meta.json                   commit hash, date, seeds, base spec, problems (n, best known cut, J sum, t_quiet)
    settings.csv                one row per setting: its overrides, R and the full-scale ramp in ps
    <problem>/<setting>.npz     per (seed, run): E (original J), t_end (ps, end of the run), t_run (ps), quiet, flips,
                                and during the run: deliveries, link_waits, far_reads (rows), sent (packets)
                                per seed: best_s, events, wall (s, a worker's first solve includes ~15 s of numba
                                compile), cycles (whole solve), link_wait_max (cycles),
                                link_load_max and far_read_max (per cycle, busiest link / far unit over the solve),
                                far_pending_max, port_pending_max, cancelled (fraction of far packets),
                                load (mesh, mesh, 4: packets per link, ways E W S N), far_reads_core, far_packets_core,
                                sent_core (per core, whole solve), tile (per core, row-major in the mesh)
                                mapping (the same for every seed): K, mesh, ramp_ps, t_quiet, t_max (ps), local_frac
                                (couplings inside a core), rows_mean, rows_max (far rows per core), lat0_mean, lat0_max
                                (zero-load far latency per (spin, destination core) pair in cycles, mean sync and dither)
    <problem>/ideal.npz         the event solver (one crossbar, no digital) with the same start voltages and kicks:
                                per (seed, run): E, t_settle (ramps), flips; per seed: best_s

A Max-Cut cut is -E/2 - J_sum/4. Kick time after run i is t_end[i + 1] - t_run[i + 1] - t_end[i], and the traffic
during kicks is the per-core total minus the sum over runs. Finished files are skipped, so a stopped run can be
restarted.

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
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hwmodel import fast
from hwmodel.dac import CurrentDAC
from hwmodel.spec import MachineSpec
from hwmodel.speed import gset, sk
from lagrange import analog_event

WORKERS = min(30, os.cpu_count())
SEEDS = 10
OUT = ROOT / 'data' / 'hwmodel'

BASE = MachineSpec()
BASE = replace(BASE, net=replace(BASE.net, dither=2), control=replace(BASE.control, kick_mode='local'))
QUIET = {'unit': 2.1, 'weighted': 7.0}      # t_quiet in ramps: 60 ns and 200 ns (C dV / I_unit) at R = 28.6
T_MAX = 52.5                                # t_max in ramps: 1.5 us at R = 28.6

PROBLEMS = ['G14', 'G15', 'G16', 'G22', 'G23', 'G24', 'G1', 'G2', 'G3', 'G48', 'G49', 'G50', 'K2000',
            'SK256', 'SK512', 'SK1024', 'SK2048']

# setting name: overrides as 'part.field': value, or 'R': cycles per ramp (sets I_unit)
EXPERIMENTS = {
    'E0': {                                 # the same R through I_unit and through f_core: must agree in ramps
        'I=200nA':  {'node.I_unit': 200e-9},    # R = 14.3
        'f=500MHz': {'clock.f_core': 0.5e9},    # R = 14.3
        'I=50nA':   {'node.I_unit': 50e-9},     # R = 57.1
        'f=2GHz':   {'clock.f_core': 2e9},      # R = 57.1
    },
}


def load(name):
    """
    J, best known cut (None if unknown) and t_quiet in ramps.
    """
    if name.startswith('SK'):
        return sk(int(name[2:])), None, QUIET['weighted']
    lines = (ROOT / 'ising/benchmarks/G/optimal_energy.txt').read_text().split('\n')
    best = dict(line.split() for line in lines if line.strip())
    return gset(name), int(best[name]), QUIET['unit']


def make_spec(setting, n, quiet, seed):
    """
    BASE with the overrides, t_quiet and t_max in ramps and the smallest mesh (a multiple of chip) for n spins.
    Returns the spec and the full-scale ramp in ps.
    """
    s = BASE
    for key, value in setting.items():
        if key != 'R':
            part, name = key.split('.')
            s = replace(s, **{part: replace(getattr(s, part), **{name: value})})
    full = 2**s.core.cell_bits - 1                                  # full-scale cell current in I_unit
    if 'R' in setting:
        I_unit = s.node.C * (s.node.V_high - s.node.V_th) * s.clock.f_core / (setting['R'] * full)
        s = replace(s, node=replace(s.node, I_unit=I_unit))
    ramp = s.node.ramp_time(s.node.I_unit * full)
    K = -(-n // s.core.c)
    mesh = s.net.chip * -(-int(np.ceil(np.sqrt(K))) // s.net.chip)
    s = replace(s, net=replace(s.net, mesh=mesh),
                control=replace(s.control, t_quiet=round(quiet * ramp), t_max=round(T_MAX * ramp), seed=seed))
    return s, ramp


def energy(S, J):
    """
    E = -1/2 s J s of every row of S, with the original J.
    """
    return -0.5 * ((S @ J) * S).sum(axis=1)


def run_machine(J, quiet, setting, seed):
    spec, ramp = make_spec(setting, len(J), quiet, seed)
    t0 = time.perf_counter()
    ctl = fast.Controller(J, np.zeros(len(J)), spec)
    ctl.solve()
    wall = time.perf_counter() - t0
    m = ctl.machine
    S = np.array(ctl.end, dtype=float)
    E = energy(S, J)
    t_end, t_run = np.array(ctl.t_end), np.array(ctl.t_run)
    t_flip = m.flips_t[:m.n_flips]                                  # in time order
    cycles = m.now / m.period
    traffic = np.array(ctl.traffic)
    mp, net = ctl.map, spec.net
    src = np.repeat(mp.core, [len(f) for f in mp.fanout])           # every (spin's core, destination core) pair
    dst = np.concatenate(mp.fanout)
    lat0 = (max(spec.clock.sync - 0.5, 0) + net.dither / 2 + net.t_inj + spec.far.t_pipe
            + net.t_hop * mp.hops[src, dst] + net.t_chip * mp.chips[src, dst])
    rows = [len(r) for r in mp.rows]
    return dict(
        E=E, t_end=t_end, t_run=t_run, quiet=np.array(ctl.why) == 'quiet',
        flips=np.searchsorted(t_flip, t_end, 'right') - np.searchsorted(t_flip, t_end - t_run, 'left'),
        deliveries=traffic[:, 0], link_waits=traffic[:, 1], far_reads=traffic[:, 2], sent=traffic[:, 3],
        best_s=S[np.argmin(E)].astype(np.int8), events=m.n_events, wall=wall, cycles=cycles,
        link_wait_max=m.link_wait_max, link_load_max=m.load.max() / cycles, far_read_max=m.far_applied.max() / cycles,
        far_pending_max=m.far_pending_max.max(), port_pending_max=m.port_pending_max.max(),
        cancelled=m.far_cancelled.sum() / m.far_packets.sum(),
        load=m.load.copy(), far_reads_core=m.far_applied.copy(), far_packets_core=m.far_packets.copy(),
        sent_core=m.port_sent.copy(), tile=mp.tile,
        K=mp.K, mesh=net.mesh, ramp_ps=ramp, t_quiet=spec.control.t_quiet, t_max=spec.control.t_max,
        local_frac=sum(np.count_nonzero(x) for x in mp.local) / np.count_nonzero(mp.J),
        rows_mean=np.mean(rows), rows_max=np.max(rows), lat0_mean=lat0.mean(), lat0_max=lat0.max(),
    )


def run_ideal(J, seed):
    """
    lagrange.solver_event (kept here to count flips) on the machine's codes with fast.Controller's start voltages
    and kicks.
    """
    c, n = BASE.control, len(J)
    r_v0, _, r_kick = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))
    v0 = r_v0.uniform(-1, 1, n)
    sizes = np.round(np.linspace(max(1, c.kick_start * n), max(1, c.kick_end * n), c.n_runs)).astype(int)
    clusters = np.zeros((c.n_runs, n), dtype=np.int64)
    for i in range(c.n_runs - 1):                                   # the controller kicks after every run but the last
        clusters[i, :sizes[i]] = r_kick.choice(n, sizes[i], replace=False)
    cell = CurrentDAC(BASE.core.cell_bits, BASE.node.I_unit)
    Jq = cell.code(J, np.abs(J).max() / cell.max_code).astype(float)  # as Mapping
    h = np.zeros(n)
    v, best_s, best_E = v0, None, np.inf
    states, t_settle, flips = np.zeros((c.n_runs, n)), np.zeros(c.n_runs), np.zeros(c.n_runs, dtype=np.int64)
    for i in range(c.n_runs):
        states[i], t_settle[i], flips[i] = analog_event(Jq, h, v, 10**6)
        E_q = -0.5 * states[i] @ Jq @ states[i]                     # the winner is picked on the codes, as in the machine
        if E_q < best_E:
            best_s, best_E = states[i].copy(), E_q
        v = best_s.copy()
        v[clusters[i, :sizes[i]]] *= -1
    E = energy(states, J)
    return dict(E=E, t_settle=t_settle * cell.max_code,             # in units of C dV / I_unit -> full-scale ramps
                flips=flips, best_s=states[np.argmin(E)].astype(np.int8))


def task(args):
    experiment, name, label, setting = args
    J, _, quiet = load(name)
    rows = [run_ideal(J, seed) if setting is None else run_machine(J, quiet, setting, seed) for seed in range(SEEDS)]
    folder = OUT / experiment / name
    np.savez(folder / f'{label}.tmp.npz', **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    (folder / f'{label}.tmp.npz').rename(folder / f'{label}.npz')
    return name, label


def main(experiment, names):
    out = OUT / experiment
    grid = {'ideal': None} | EXPERIMENTS[experiment]
    problems = {}
    for name in names:
        J, best, quiet = load(name)
        problems[name] = dict(n=len(J), best=best, J_sum=float(J.sum()), quiet_ramps=quiet)
        (out / name).mkdir(parents=True, exist_ok=True)
    meta = dict(commit=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
                date=datetime.now().isoformat(), seeds=SEEDS, n_runs=BASE.control.n_runs, T_MAX=T_MAX,
                base=str(BASE), problems=problems)
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    rows = []
    for label, setting in grid.items():
        if setting is None:
            rows.append(dict(setting=label, R=np.inf))
        else:
            s, ramp = make_spec(setting, 1, 1, 0)
            rows.append(dict(setting=label, **setting, R=ramp * s.clock.f_core / 1e12, ramp_ps=ramp))
    pd.DataFrame(rows).to_csv(out / 'settings.csv', index=False)

    tasks = [(experiment, name, label, setting) for name in sorted(names, key=lambda x: -problems[x]['n'])
             for label, setting in grid.items() if not (out / name / f'{label}.npz').exists()]
    print(f'{len(tasks)} tasks on {WORKERS} workers', flush=True)
    t0 = time.perf_counter()
    with Pool(WORKERS) as pool:
        for i, (name, label) in enumerate(pool.imap_unordered(task, tasks)):
            print(f'{i + 1}/{len(tasks)}  {name} {label}  {time.perf_counter() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:] or PROBLEMS)
