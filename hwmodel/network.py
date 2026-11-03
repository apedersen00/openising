"""
network.py

Router mesh between cores.
"""

import numpy as np

STEP = {
    'E': ( 0,  1),
    'W': ( 0, -1),
    'S': ( 1,  0),
    'N': (-1,  0),
}


class Mesh:
    """
    One router per core. A packet for spin k goes to every core with a row for k.
    Packets are sent in a "XY" tree. First along X and then Y. The router copies
    the packet where destinations split. Each link has width lanes, each carries one packet
    per t_link cycles. If all lanes are busy it waits in the router.
    """
    def __init__(self, sim, clock, spec, mapping, rng):
        self.sim = sim
        self.core = mapping.core
        self.fanout = mapping.fanout
        self.chip = spec.chip
        self.R, self.Q = np.divmod(mapping.tile, spec.mesh) # row and column of all cores
        self.t_inj = clock.cycles(spec.t_inj)
        self.t_hop = clock.cycles(spec.t_hop)
        self.t_chip = clock.cycles(spec.t_chip)
        self.t_link = clock.cycles(spec.t_link)
        self.width = spec.width
        self.link_free = {}                                 # (row, col, way)
        self.load = {}                                      # (row, col, way)
        self.dither = spec.dither
        self.t_rx = clock.cycles(spec.t_rx)
        self.rx_free = [0] * mapping.K
        self.period = clock.period
        self.rng = rng
        self.far = None
        self.in_flight = 0
        self.stats = dict(packets=0, deliveries=0, rx_waits=0, link_waits=0, link_wait_max=0)

    def send(self, k):
        """
        Spin k has flipped. The packet reaches the router of its core t_inj later.
        """
        p = self.core[k]
        self.stats['packets'] += 1
        d = int(self.rng.integers(0, self.dither + 1)) * self.period if self.dither else 0
        self.in_flight += len(self.fanout[k])
        self.sim.after(self.t_inj + d, self.hop, self.R[p], self.Q[p], self.fanout[k], k)

    def hop(self, r, q, dests, k):
        """
        Packet for spin k, going to the destinations, is in the router ar row r, col q.
        """
        dr, dq = self.R[dests] - r, self.Q[dests] - q
        way = np.where(dq > 0, 'E', np.where(dq < 0, 'W', np.where(dr > 0, 'S', np.where(dr < 0, 'N', '-'))))
        for p in dests[way == '-']:
            self.arrive(p, k)
        for w in 'EWSN':
            if (way == w).any():
                self.link(r, q, w, dests[way == w], k)

    def link(self, r, q, w, dests, k):
        """
        A copy goes out the router (r, q) in direction w. Takes the first free lane, waits if all are busy.
        """
        free = self.link_free.setdefault((r, q, w), [0] * self.width)  # when each lane is free
        lane = int(np.argmin(free))                         # the first free lane
        t = max(self.sim.now, free[lane])
        free[lane] = t + self.t_link
        self.load[(r, q, w)] = self.load.get((r, q, w), 0) + 1
        self.stats['link_waits'] += t > self.sim.now
        self.stats['link_wait_max'] = max(self.stats['link_wait_max'], (t - self.sim.now) // self.period)
        r2, q2 = r + STEP[w][0], q + STEP[w][1]
        edge = r // self.chip != r2 // self.chip or q // self.chip != q2 // self.chip
        self.sim.schedule(t + self.t_hop + edge * self.t_chip, self.hop, r2, q2, dests, k)

    def arrive(self, q, k):
        """
        Packet reaches router of core q. Waits for the port to the far unit.
        """
        t = max(self.sim.now, self.rx_free[q])
        self.stats['rx_waits'] += t > self.sim.now
        self.rx_free[q] = t + self.t_rx
        self.sim.schedule(t, self.deliver, q, k)

    def deliver(self, q, k):
        """
        Packet reaches core q.
        """
        self.in_flight -= 1
        self.stats['deliveries'] += 1
        self.far[q].receive(k)
