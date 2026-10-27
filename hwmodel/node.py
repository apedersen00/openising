"""
node.py

The rows of a core. C dV/dt = I, comparator with V_th.
"""

import numpy as np

from hwmodel.timebase import ps, seconds


class LinearNode:
    """
    Integrator with rails.
    """
    def __init__(self, spec, V):
        self.spec = spec
        self.V = np.array(V, dtype=float)
        self.s = np.where(self.V >= spec.V_th, 1, -1)

    def next_crossing(self, I):
        """
        (dt, k) ps until the first sigma reaches V_th at constant currents I (A), and which line.
        (None, -1) no sigma is heading for the threshold.
        """
        heading = self.s * I < 0    # is the spin heading for the threshold?
        if not heading.any():
            return None, -1
        t = np.full(len(self.V), np.inf)
        t[heading] = (self.V[heading] - self.spec.V_th) * self.spec.C / -I[heading]
        k = int(np.argmin(t))
        return ps(t[k]), k

    def advance(self, I, dt):
        """
        Move every sigma dt ps.
        Sigmas that reach V_th are put at exactly V_th.
        """
        heading = self.s * I < 0
        self.V = np.clip(self.V + I / self.spec.C * seconds(dt), self.spec.V_low, self.spec.V_high)
        reached = heading & (self.s * (self.V - self.spec.V_th) <= 0)
        self.V[reached] = self.spec.V_th

    def flip(self, k):
        """
        Sigma k has reached V_th.
        """
        self.s[k] = -self.s[k]
