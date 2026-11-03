"""
speed.py

Speed harness: fixed problems with fixed seeds. Prints wall time, events per second and a fingerprint, a hash of
every core's flips, the energy and end state of every run, and the event count. A change that should not change
the model must leave every fingerprint unchanged.

    python -m hwmodel.speed                 # all cases
    python -m hwmodel.speed G14 SK512       # some cases
    python -m hwmodel.speed --fast          # the numba machine (fast.py): must print the same fingerprints

*ai-generated*
"""

import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from hwmodel import fast
from hwmodel.controller import Controller
from hwmodel.spec import MachineSpec

ROOT = Path(__file__).parent.parent


def gset(name):
    """
    Max-Cut instance from ising/benchmarks/G as J = -W.
    """
    lines = (ROOT / f'ising/benchmarks/G/{name}.txt').read_text().split('\n')
    n = int(lines[0].split()[0])
    J = np.zeros((n, n))
    for line in lines[1:]:
        if line.strip():
            i, j, w = line.split()
            J[int(i) - 1, int(j) - 1] = J[int(j) - 1, int(i) - 1] = -float(w)
    return J


def sk(n, seed=0):
    """
    Dense Gaussian couplings.
    """
    J = np.triu(np.random.default_rng(seed).standard_normal((n, n)), 1)
    return J + J.T


def random160():
    """
    The Test A/B problem: 160 spins, 5 % of pairs coupled, codes -7..7.
    """
    rng = np.random.default_rng(42)
    J = np.triu(rng.integers(-7, 8, (160, 160)) * (rng.random((160, 160)) < 0.05), 1)
    return J + J.T


def cases():
    """
    name: (problem, spec). R160 is Test B; the others use dither 2.
    """
    s = MachineSpec()
    runs = lambda n, **kw: replace(s.control, n_runs=n, **kw)
    return {
        'R160':  (random160, replace(s, core=replace(s.core, c=16), control=runs(30, t_quiet=200_000))),
        'G14':   (lambda: gset('G14'), replace(s, net=replace(s.net, dither=2), control=runs(10))),
        'SK512': (lambda: sk(512), replace(s, net=replace(s.net, dither=2), control=runs(10, t_quiet=200_000))),
        'G22':   (lambda: gset('G22'), replace(s, net=replace(s.net, mesh=8, dither=2), control=runs(3))),
    }


def flips(ctl):
    """
    (time, spin) of every flip, per core.
    """
    if isinstance(ctl, fast.Controller):
        return ctl.flips()
    return [np.array(core.flips, dtype=np.int64) for core in ctl.machine.cores]


def fingerprint(ctl):
    h = hashlib.sha256()
    for f in flips(ctl):
        h.update(f.tobytes())
    h.update(np.array(ctl.E).tobytes())
    h.update(np.array(ctl.end).tobytes())
    h.update(np.array([ctl.sim.n_events, ctl.sim.now]).tobytes())
    return h.hexdigest()[:12]


def main(names, ctl_class=Controller):
    if ctl_class is fast.Controller:                                # compile once, outside the timing
        t0 = time.perf_counter()
        make, spec = cases()['R160']
        fast.Controller(make(), np.zeros(160), replace(spec, control=replace(spec.control, n_runs=2))).solve()
        print(f'compiled in {time.perf_counter() - t0:.1f} s')
    for name in names:
        make, spec = cases()[name]
        J = make()
        t0 = time.perf_counter()
        ctl = ctl_class(J, np.zeros(len(J)), spec)
        ctl.solve()
        wall = time.perf_counter() - t0
        m = ctl.machine
        deliveries = m.deliveries if ctl_class is fast.Controller else m.mesh.stats['deliveries']
        print(f'{name:6s} {fingerprint(ctl)}  {wall:6.2f} s  {ctl.n_runs / wall:6.1f} runs/s  '
              f'{ctl.sim.n_events / wall / 1e3:7.1f} k events/s  events {ctl.sim.n_events}  '
              f'flips {sum(len(f) for f in flips(ctl))}  deliveries {deliveries}')


if __name__ == '__main__':
    args = sys.argv[1:]
    use_fast = '--fast' in args
    names = [a for a in args if a != '--fast'] or list(cases())
    main(names, fast.Controller if use_fast else Controller)
