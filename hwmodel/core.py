"""
core.py

One analog core. node + crossbar.
"""

from hwmodel.kernel import ANALOG


class AnalogCore:
    """
    The crossbar defines the currents and couplings and the nodes say when a spin reaches V_th.
    """
    def __init__(self, sim, node, xb, t_cmp):
        self.sim = sim          # simulator kernel
        self.node = node        # analog node
        self.xb = xb            # analog crossbar
        self.t_cmp = t_cmp      # comparator delay
        self.t = sim.now
        self.pending = None     # None when no spin is heading for V_th
        self.flips = []         # (time, spin) for every flip
        self.schedule()

    def sync(self):
        """
        Move voltages up to now.
        Has to always be called first before changing something. 
        """
        self.node.advance(self.xb.current(), self.sim.now - self.t)
        self.t = self.sim.now

    def schedule(self):
        """
        Replace pending event by the next crossing at the current currents (hehe)
        """
        self.sim.cancel(self.pending)   # something has changed, so the pending event may no longer be correct
        dt, k = self.node.next_crossing(self.xb.current())
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
        Comparator output of spin k changes.
        """
        self.sync()
        self.xb.flip(k)
        self.flips.append((self.sim.now, k))
        self.schedule()

    def set_far(self, i, code):
        """
        The far unit/coupler writes a new bias code for line i.
        """
        self.sync()
        self.xb.F[i] = code
        self.schedule()
