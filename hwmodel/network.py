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
        self.t_rx = clock.cycles(spec.t_rx)
        self.rx_free = [0] * mapping.K                                  # time each far unit takes the next packet
        self.period = clock.period
        self.rng = rng
        self.far = None
        self.in_flight = 0
        self.stats = dict(packets=0, deliveries=0, rx_waits=0)

    def send(self, k):
        """
        Spin k has flipped.
        """
        p = self.core[k]
        self.stats['packets'] += 1
        d = int(self.rng.integers(0, self.dither + 1)) * self.period if self.dither else 0
        for q in self.fanout[k]:
            self.in_flight += 1
            self.sim.after(int(self.delay[p, q]) + d, self.arrive, q, k)

    def arrive(self, q, k):
        """
        Packet reaches the router of core q. It waits for the router's port to the far unit.
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