"""
network.py

Router mesh between cores.
"""


class Mesh:
    """
    One router per core. A packet for spin k goes to every core with a row for k.
    Packet jumps core to core and arrives in t_inj + hops * t_hop + chip edges * t_chip.
    """
    def __init__(self, sim, clock, spec, mapping, rng):
        self.sim = sim
        self.core = mapping.core
        self.fanout = mapping.fanout
        self.delay = clock.cycles(spec.t_inj + spec.t_hop * mapping.hops + spec.t_chip * mapping.chips)
        self.dither = spec.dither
        self.period = clock.period
        self.rng = rng
        self.far = None
        self.in_flight = 0
        self.stats = dict(packets=0, deliveries=0)

    def send(self, k):
        """
        Spin k has flipped.
        """
        p = self.core[k]
        self.stats['packets'] += 1
        for q in self.fanout[k]:
            d = int(self.delay[p, q])
            if self.dither:
                d += int(self.rng.integers(0, self.dither + 1)) * self.period
            self.in_flight += 1
            self.sim.after(d, self.deliver, q, k)

    def deliver(self, q, k):
        """
        Packet reaches core q.
        """
        self.in_flight -= 1
        self.stats['deliveries'] += 1
        self.far[q].receive(k)