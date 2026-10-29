"""
mapping.py


Converts a problem to fit the multi-core machine (which spin where).
"""

import numpy as np


def grow(A, K, c):
    """
    Greedy growth. Which spin goes in which core?
    Core p starts at the first free spin and keeps taking the spins with largest share of |J|.

    Args:
        - A: |J|
        - K: number of cores
    """
    n = len(A)                                          # number of spins
    deg = A.sum(axis=1)                                 # each spins total coupling
    deg[deg == 0] = 1                                   # remove zeros in deg as we get 0/0 otherwisse
    core = np.full(n, -1)                               # an array of all spins, -1 = not placed in a core
    for p in range(K):                                  # iterate all cores
        pull = np.zeros(n)                              # total |J| between spin j and spins in core p
        since = np.full(n, np.inf)                      # step where spin j got a nonzero pull
        for step in range(min(c, n - p * c)):           # how many spins go in the core, except the last which gets the remainder
            score = np.where(core < 0, pull / deg, -1)  # free spins get pull / deg, the weight that would be inside the core
            top = np.flatnonzero(score == score.max())  # find the best scoring spins
            i = top[np.argmin(since[top])]              # get the best scoring spin
            core[i] = p                                 # spin i joins core p
            pull += A[:, i]
            since[(A[:, i] > 0) & (since == np.inf)] = step
    return core


class Mapping:
    """
    Partition, place and quantize for every core.
    J is symmetric.
    """
    def __init__(self, J, h, spec, cell, hdac):
        n, c, mesh, chip = len(h), spec.core.c, spec.net.mesh, spec.net.chip
        self.K = -(-n // c)
        if self.K > mesh**2:
            raise ValueError(f'{n} spins need {self.K} cores, the mesh has {mesh**2} tiles')

        # partition spins to cores
        self.core = grow(np.abs(J), self.K, c)
        self.spins = [np.flatnonzero(self.core == p) for p in range(self.K)]

        # place cores in a tile.
        r, q = np.divmod(np.arange(mesh**2), mesh)
        self.tile = np.lexsort((q % chip, r % chip, q // chip, r // chip))[:self.K]
        R, Q = r[self.tile], q[self.tile]
        self.hops = np.abs(R[:, None] - R) + np.abs(Q[:, None] - Q)
        self.chips = np.abs(R[:, None] // chip - R // chip) + np.abs(Q[:, None] // chip - Q // chip)

        # Quantize
        self.lsb = np.abs(J).max() / cell.max_code
        self.J = cell.code(J, self.lsb)
        self.h = hdac.code(h, self.lsb)

        # Fill
        self.local = [self.J[np.ix_(s, s)] for s in self.spins]
        need = np.array([(self.J[s] != 0).any(axis=0) for s in self.spins])
        need[self.core, np.arange(n)] = False
        self.rows = [np.flatnonzero(x) for x in need]
        self.sram = [self.J[np.ix_(r, s)] for r, s in zip(self.rows, self.spins)]
        self.fanout = [np.flatnonzero(x) for x in need.T]
