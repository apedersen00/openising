"""
farunit.py

Computes equivalent bias from far spins (in other cores).
"""

from collections import deque

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
        self.target = self.known.copy()                         # bit 1, the sigma affter every packet received
        self.queued = np.zeros(len(rows), dtype=bool)           # bit 2, this row is waiting in the FIFO 
        self.fifo = deque()                                     # FIFO of row addresses
        self.acc = sram.T @ self.known                          # load accumulator with starting field
        self.fdac = fdac                                        # far spin current dac
        self.lanes = spec.lanes                                 # rows read per cycle
        self.t_pipe = clock.cycles(spec.t_pipe)                 # delay
        self.period = clock.period
        self.touch = touch                                      # function in the controller
        self.running = False                                    # a step is already being performed (forces it to be sequential)
        self.writes = 0                                         # count far spin bias writess
        self.stats = dict(packets=0, merged=0, applied=0, fifo_max=0)
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
        r = self.row[k]                     # look up the row address
        self.target[r] = -self.target[r]    # that row's target bit is flipped
        self.stats['packets'] += 1
        if self.queued[r]:                  # if already in the queue, merge the events
            self.stats['merged'] += 1
            return
        self.queued[r] = True               # queue the event
        self.fifo.append(r)
        self.stats['fifo_max'] = max(self.stats['fifo_max'], len(self.fifo))
        if not self.running:
            self.running = True
            self.sim.after(0, self.step)

    def step(self):
        """
        One cycle.
        """
        changed = False
        for _ in range(min(self.lanes, len(self.fifo))):            # use all lanes
            r = self.fifo.popleft()                                 # pop the fifo for the oldest row
            self.queued[r] = False
            if self.target[r] != self.known[r]:                     # an odd number of packets has arrived: flip
                self.known[r] = self.target[r]
                self.acc += 2 * self.known[r] * self.sram[r]        # compute new local field
                self.stats['applied'] += 1
                changed = True
        if changed:
            self.writes += 1
            self.sim.after(self.t_pipe, self.write, self.code())
        if self.fifo:
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
