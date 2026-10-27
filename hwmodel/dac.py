"""
dac.py

The current DAC for coupling cells, h cells and the far couplings.
"""

import numpy as np


class CurrentDAC:
    """
    A sign bit and magnitude bits of binar weighted pairs (1x, 2x, 4x, ...).
    bits = None is exact.
    """
    def __init__(self, bits, I_lsb):
        self.bits = bits
        self.I_lsb = I_lsb
        self.max_code = np.inf if bits is None else 2**bits - 1

    def code(self, x, lsb):
        """
        Nearest code for the value x.
        """
        return np.clip(np.round(np.asarray(x) / lsb), -self.max_code, self.max_code).astype(np.int64)

    def current(self, code):
        """
        Output current (A).
        """
        return code * self.I_lsb


