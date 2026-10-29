"""
trace.py

Records what a machine does, for animations and debugging. It wraps methods of the built blocks,
so the model itself is unchanged. Attach it after the controller is built and before it solves.

*ai-generated*
"""

from bisect import bisect_right

from hwmodel.timebase import seconds


def wrap(obj, name, before=None, after=None):
    """
    Replace obj.name by a function that calls before(*args), the original, then after(*args).
    """
    original = getattr(obj, name)

    def wrapped(*args):
        if before:
            before(*args)
        out = original(*args)
        if after:
            after(*args)
        return out
    setattr(obj, name, wrapped)


class Trace:
    """
    Every entry starts with its time in ps.
        cores[p]    (t, V, I, held)  core p after every change: V moves at I/C until its next entry
        flips       (t, p, k)        comparator k of core p switched
        packets     (t, spin, p)     a packet for spin leaves core p
        arrivals    (t, q, spin)     a copy reaches the router of core q
        deliveries  (t, q, spin)     it enters the far unit of core q
        writes      (t, q)           new F codes in core q
        phases      (t, name)        'run', 'end', 'hold', 'preset', 'release'
    """
    def __init__(self, ctl):
        m = ctl.machine
        self.map = m.map
        self.node = m.cores[0].node.spec
        self.s0 = m.s                                               # sigma when the trace starts
        self.cores = [[] for _ in m.cores]
        self.times = [[] for _ in m.cores]                          # the times of cores[p], for searching
        self.flips, self.packets, self.arrivals, self.deliveries, self.writes, self.phases = [], [], [], [], [], []
        sim = m.sim
        log = lambda entries, *entry: entries.append((sim.now, *entry))

        for p, core in enumerate(m.cores):
            state = lambda *_, p=p, core=core: (self.times[p].append(sim.now),
                                                log(self.cores[p], core.node.V.copy(), core.xb.current(), core.held))
            state()
            wrap(core, 'schedule', after=state)
            wrap(core, 'preset', after=state)
            wrap(core, 'on_flip', after=lambda k, p=p: log(self.flips, p, k))
        wrap(m.mesh, 'send', before=lambda k: log(self.packets, k, int(m.map.core[k])))
        wrap(m.mesh, 'arrive', before=lambda q, k: log(self.arrivals, q, k))
        wrap(m.mesh, 'deliver', before=lambda q, k: log(self.deliveries, q, k))
        for q, far in enumerate(m.fars):
            wrap(far, 'write', after=lambda F, q=q: log(self.writes, q))
        wrap(ctl, 'run', before=lambda: log(self.phases, 'run'), after=lambda: log(self.phases, 'end'))
        wrap(ctl, 'kick', before=lambda s: log(self.phases, 'hold'), after=lambda s: log(self.phases, 'release'))
        wrap(ctl, 'preset', before=lambda s: log(self.phases, 'preset'))

    def voltages(self, p, t):
        """
        Node voltages of core p at time t (ps), from its last entry at or before t.
        """
        t0, V, I, held = self.cores[p][bisect_right(self.times[p], t) - 1]
        if held:
            return V
        return (V + I / self.node.C * seconds(t - t0)).clip(self.node.V_low, self.node.V_high)
