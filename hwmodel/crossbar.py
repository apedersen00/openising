"""
crossbar.py

Analog core: coupling cells, h row and far row of a core.
Comparators are NOT in this module.
"""

import numpy as np


class Crossbar:
    """
    Line i gets J[i, j] * s[j] from every spin j, plus h[i] and F[i].
    s is the spins. F is written by the far unit.
    """
    def __init__(self, J, h, s, cell, hdac, fdac):
        self.J = np.asarray(J, dtype=np.int64)
        self.h = np.asarray(h, dtype=np.int64)
        self.F = np.zeros(len(self.h), dtype=np.int64)
        self.s = np.array(s, dtype=np.int64)
        self.cell, self.hdac, self.fdac = cell, hdac, fdac
        self.Js = self.J @ self.s

    def flip(self, k):
        """
        Spin k flips: all lines change by 2 J[:, k] s_k.
        """
        self.s[k] = -self.s[k]
        self.Js += 2 * self.J[:, k] * self.s[k]

    def current(self):
        """
        Current into every line (A).
        """
        return self.cell.current(self.Js) + self.hdac.current(self.h) + self.fdac.current(self.F)
