"""
controller.py

Runs the machine. Restarts, kicks, computes energy.
"""

import numpy as np

from hwmodel.kernel import CONTROL
from hwmodel.machine import Machine


class Controller:
    """
    A run ends when nothing has happened for t_quiet and nothing is busy, or after t_max.
    Sigma is then read out and its energy is computed and compared (winner is kept).
    All cores are forzen and kicked.
    """
    def __init__(self, J, h, spec):
        c = spec.control
        n = len(h)
        r_v0, r_net, self.rng = (np.random.default_rng(s) for s in np.random.SeedSequence(c.seed).spawn(3))
        self.last = 0
        self.v0 = r_v0.uniform(-1, 1, n)
        self.machine = Machine(J, h, spec, self.v0, r_net, touch=self.touch)
        self.sim = self.machine.sim
        self.n_runs = c.n_runs
        self.sizes = np.round(np.linspace(max(1, c.kick_start * n), max(1, c.kick_end * n), c.n_runs)).astype(int)
        self.t_quiet = c.t_quiet
        self.t_max = c.t_max
        self.t_restart = self.machine.clock.cycles(c.t_restart)
        self.check = None
        self.best_s, self.best_E = None, np.inf
        self.start, self.end, self.E, self.t_run, self.why = [], [], [], [], []

    def touch(self):
        """
        Something happened. Restart the timer.
        """
        self.last = self.sim.now

    def watch(self):
        """
        The timer ran out. check for activity or if something is busy.
        """
        if self.sim.now - self.last < self.t_quiet:
            self.check = self.sim.schedule(self.last + self.t_quiet, self.watch, priority=CONTROL)
        elif self.machine.busy:
            self.check = self.sim.after(self.machine.clock.period, self.watch, priority=CONTROL)
        else:
            self.sim.stop()

    def run(self):
        """
        One run until quiet or t_max.
        """
        t0 = self.sim.now
        self.last = t0
        self.check = self.sim.schedule(t0 + self.t_quiet, self.watch, priority=CONTROL)
        why = self.sim.run(until=t0 + self.t_max)
        self.sim.cancel(self.check)
        self.t_run.append(self.sim.now - t0)
        return 'quiet' if why == 'stopped' else 't_max'

    def kick(self, s):
        """
        Freeze all cores. t_restart later force s on the rails.
        """
        for core in self.machine.cores:
            core.hold()
        self.sim.after(self.t_restart, self.preset, s, priority=CONTROL)
        self.sim.run()
        for core in self.machine.cores:
            core.release()

    def preset(self, s):
        for p, core in enumerate(self.machine.cores):
            core.preset(s[self.machine.map.spins[p]])

    def solve(self):
        """
        All runs. Return best state and energy.
        """
        n = len(self.v0)
        for i in range(self.n_runs):
            self.start.append(self.machine.s)
            self.why.append(self.run())
            s, E = self.machine.s, self.machine.energy()
            self.end.append(s)
            self.E.append(E)
            if E < self.best_E:
                self.best_s, self.best_E = s, E
            if i < self.n_runs - 1:
                s = self.best_s.copy()
                s[self.rng.choice(n, self.sizes[i], replace=False)] *= -1
                self.kick(s)
        return self.best_s, self.best_E