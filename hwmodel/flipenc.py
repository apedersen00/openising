"""
flipenc.py

The send port of one analog core.
Comparator flip -> digital -> merging queue -> packet to mesh network.
"""

import numpy as np


class SendPort:
    """
    A flip is seen on the n'th sync clock edge after it happens.
    Each spin has a pending bit. A flip and a flip back before the
    packet leaves cancels it. One packet leaves every t_link cycles.
    """
    def __init__(self, sim, clock, sync, spec, mesh, ids):
        self.sim = sim
        self.clock = clock
        self.sync = sync
        self.mesh = mesh
        self.ids = ids
        self.out = np.array([len(mesh.fanout[k]) > 0 for k in ids])
        self.pending = np.zeros(len(ids), dtype=bool)
        self.last = -1
        self.t_link = clock.cycles(spec.t_link)
        self.free = 0
        self.syncing = 0
        self.sending = False
        self.stats = dict(flips=0, sent=0, pending_max=0)

    @property
    def busy(self):
        return self.syncing > 0 or self.sending

    def flip(self, k):
        """
        A comparator of skin k has flipped.
        """
        if not self.out[k]:
            return
        self.syncing += 1
        t = self.sim.now if self.sync == 0 else self.clock.edge_after(self.sim.now, self.sync)
        self.sim.schedule(t, self.seen, k)

    def seen(self, k):
        """
        The flip comes out of the synchronizer.
        """
        self.syncing -= 1
        self.stats['flips'] += 1
        self.pending[k] = not self.pending[k]
        self.stats['pending_max'] = max(self.stats['pending_max'], int(self.pending.sum()))
        if self.pending[k] and not self.sending:
            self.sending = True
            self.sim.schedule(max(self.sim.now, self.free), self.send)

    def send(self):
        """
        The mesh is free. Send the oldest spin.
        """
        idx = np.flatnonzero(self.pending)
        if len(idx) == 0:
            self.sending = False
            return
        k = idx[np.searchsorted(idx, self.last + 1) % len(idx)]
        self.pending[k] = False
        self.last = k
        self.mesh.send(self.ids[k])
        self.stats['sent'] += 1
        self.free = self.sim.now + self.t_link
        if self.pending.any():
            self.sim.schedule(self.free, self.send)
        else:
            self.sending = False
