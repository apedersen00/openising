"""
farunit.py

Computes equivalent bias from far spins (in other cores).
"""

import numpy as np


class FarUnit:
    """
    A packet is received "far spin k has flipped". The SRAM row is put in a FIFO in a waiting state.
    A waiting row "swallows" later packets of the same spin, merging them. Each cycle, rows are read
    and acc += 2 * sigma * row. The new F bias is put in the core t_pipe cycles later.
    """
    def __init__(self, sim, core, rows, sram, s_far, fdac, spec, clock, touch):
        self.sim = sim                                          # simulator
        self.core = core                                        # the core of this far unit
        self.row = {k: r for r, k in enumerate(rows)}           # (addr, spin) pairs for SRAM
        self.sram = sram                                        # Row r has J[rows[r], spin i]
        self.known = np.array(s_far)                            # bit 0, the sigma the accumulators currently include
        self.pending = np.zeros(len(rows), dtype=bool)          # pending far spins
        self.n_pending = 0
        self.last = -1
        self.acc = sram.T @ self.known                          # load accumulator with starting field
        self.fdac = fdac                                        # far spin current dac
        self.lanes = spec.lanes                                 # rows read per cycle
        self.t_pipe = clock.cycles(spec.t_pipe)                 # delay
        self.period = clock.period
        self.touch = touch                                      # function in the controller
        self.running = False                                    # a step is already being performed (forces it to be sequential)
        self.free = 0
        self.writes = 0                                         # count far spin bias writess
        self.stats = dict(packets=0, cancelled=0, applied=0, pending_max=0)
        core.set_far(self.code())

    def code(self):
        """
        F codes from the accumulators.
        """
        return self.fdac.code(self.acc, 1)

    @property
    def busy(self):
        return self.running or self.writes > 0

    def receive(self, k):
        """
        A packet is received from the mesh network. A far spin k has flipped.
        """
        r = self.row[k]                         # look up the row address
        self.pending[r] = not self.pending[r]   # flip the pending bit (two flips cancel)
        self.n_pending += 1 if self.pending[r] else - 1
        self.stats['packets'] += 1
        self.stats['cancelled'] += not self.pending[r]
        self.stats['pending_max'] = max(self.stats['pending_max'], self.n_pending)
        if self.pending[r] and not self.running:
            self.running = True
            self.sim.schedule(max(self.sim.now, self.free), self.step)

    def step(self):
        """
        One cycle.
        """
        self.free = self.sim.now + self.period
        # round robin style processing of pending spins
        idx = np.flatnonzero(self.pending)
        first = np.searchsorted(idx, self.last + 1)
        take = np.concatenate((idx[first:], idx[:first]))[:self.lanes]
        for r in take:
            self.pending[r] = False
            self.known[r] = - self.known[r]
            self.acc += 2 * self.known[r] * self.sram[r]
        self.n_pending -= len(take)
        self.stats['applied'] += len(take)
        if len(take):
            self.last = take[-1]
            self.writes += 1
            self.sim.after(self.t_pipe, self.write, self.code())
        if self.n_pending:
            self.sim.after(self.period, self.step)
        else:
            self.running = False
        self.touch()

    def write(self, F):
        """
        End of pipeline. The far spin bias regsiters are updated. 
        """
        self.writes -= 1
        self.core.set_far(F)
        self.touch()

    def kick(self, s_win, s):
        """
        Local kick, no packets. Restore the winner's far spins and accumulators (snapshot), then read the rows
        of the far spins the kick flips (every core knows the cluster from the same generator).
        """
        self.known = np.array(s_win)
        self.acc = self.sram.T @ self.known
        self.core.set_far(self.code())
        self.pending = np.array(s) != self.known
        self.n_pending = int(self.pending.sum())
        if self.n_pending:
            self.running = True
            self.sim.schedule(max(self.sim.now, self.free), self.step)
