"""
core.py

One analog core. node + crossbar.
"""

import numpy as np

from hwmodel.kernel import ANALOG


class AnalogCore:
    """
    The crossbar defines the currents and couplings and the nodes say when a spin reaches V_th.
    """
    def __init__(self, sim, node, xb, t_cmp, on_flip=None):
        self.sim = sim          # simulator kernel
        self.node = node        # analog node
        self.xb = xb            # analog crossbar
        self.t_cmp = t_cmp      # comparator delay
        self.on_flip = on_flip  # when comparator k flips
        self.t = sim.now
        self.held = False       # True: voltages are frozen
        self.pending = None     # None when no spin is heading for V_th
        self.flips = []         # (time, spin) for every flip
        self.schedule()

    def sync(self):
        """
        Move voltages up to now.
        Has to always be called first before changing something. 
        """
        if not self.held:
            self.node.advance(self.xb.current(), self.sim.now - self.t)
        self.t = self.sim.now

    def schedule(self):
        """
        Replace pending event by the next crossing at the current currents (hehe)
        """
        self.sim.cancel(self.pending)   # something has changed, so the pending event may no longer be correct
        dt, k = (None, -1) if self.held else self.node.next_crossing(self.xb.current())
        self.pending = None if k < 0 else self.sim.after(dt, self.cross, k, priority=ANALOG)

    def cross(self, k):
        """
        Spin k reaches V_th. Cells see the new sigma t_cmp later (comparator delay).
        """
        self.sync()
        self.node.flip(k)
        self.sim.after(self.t_cmp, self.switch, k, priority=ANALOG)
        self.schedule()

    def switch(self, k):
        """
        Comparator output of spin k changes t_cmp later.
        """
        if self.xb.s[k] == self.node.s[k]:
            return
        self.sync()
        self.xb.flip(k)
        self.flips.append((self.sim.now, k))
        if self.on_flip:
            self.on_flip(k)
        self.schedule()

    def set_far(self, F):
        """
        The far unit/coupler writes a new bias code for all lines.
        """
        self.sync()
        self.xb.F[:] = F
        self.schedule()

    def hold(self):
        """
        Freeze voltages.
        """
        self.sync()
        self.held = True
        self.schedule()

    def preset(self, s):
        """
        While frozen, set all nodes to the rail of its sigma.
        """
        self.node.V = np.where(s > 0, self.node.spec.V_high, self.node.spec.V_low)
        self.node.s = np.array(s)
        for k in np.flatnonzero(self.xb.s != s):
            self.xb.flip(k)
            if self.on_flip:
                self.on_flip(k)

    def release(self):
        """
        Voltages can move again.
        """
        self.sync()
        self.held = False
        self.schedule()
