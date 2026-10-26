"""
timebase.py

Time is tracked in integer picoseconds.
"""

import math

PS = 1
NS = 1_000
US = 1_000_000


def ps(s):
    """
    Convert seconds to integer ps. Rounds up.
    """
    x = s * 1e12
    r = round(x)
    if abs(x - r) >= 1e-6:
        return math.ceil(x)
    return r


def seconds(t):
    return t * 1e-12


class Clock:
    """
    A clock domain. Period is rounded to integer ps.
    """
    def __init__(self, name, freq, phase=0):
        self.name = name
        self.period = round(1e12 / freq)
        self.phase = phase % self.period
        self.freq = 1e12 / self.period

    def next_edge(self, t, after=False):
        """
        Returns edge at or after t (always after t if True).
        """
        k = -(-(t - self.phase) // self.period)
        e = self.phase + k * self.period
        if after and e == t:
            return e + self.period
        return e

    def edge_after(self, t, n=1):
        """
        Returns the n-th edge after t.
        """
        return self.next_edge(t, after=True) + (n - 1) * self.period

    def cycles(self, n):
        return n * self.period

    def __repr__(self):
        return f'Clock({self.name}, {self.freq / 1e9:.4g} GHz, period {self.period} ps)'
