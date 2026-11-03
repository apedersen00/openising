"""
fast.py

The same machine as machine.py and controller.py, compiled with numba. Every block keeps its methods and names:
    <block>_<method>(p, ...) here  =  <block>.<method>(...) of block p in the reference
e.g. core_cross(p, k) is machine.cores[p].cross(k) and far_step(q) is machine.fars[q].step().
State is stored for all blocks at once, so node_V[p] is machine.cores[p].node.V (padded to c spins).
Events are integers (kind, a, b) on one heap ordered by (time, priority, sequence) like kernel.py, so ties
resolve in the same order and every flip is the same as in the reference.

    from hwmodel.fast import Controller      # same interface as hwmodel.controller.Controller

*ai-generated*
"""

import numba
import numpy as np
from numba import boolean, float64, int64, njit
from numba.experimental import jitclass

from hwmodel.dac import CurrentDAC
from hwmodel.kernel import ANALOG, CONTROL, EDGE
from hwmodel.mapping import Mapping
from hwmodel.timebase import Clock

# event kinds: the reference method each one calls
CROSS, SWITCH, SEEN, SEND, HOP, DELIVER, STEP, WRITE, WATCH, PRESET = range(10)
# Simulator.run return values: 'empty', 'until', 'stopped', 'max_events'
EMPTY, UNTIL, STOPPED, MAX_EVENTS = range(4)
# network.py STEP for the ways E, W, S, N
STEP_R = np.array([0, 0, 1, -1])
STEP_Q = np.array([1, -1, 0, 0])


@njit
def ps(s):
    """
    timebase.ps: seconds to integer ps, rounds up.
    """
    x = s * 1e12
    r = np.floor(x + 0.5)
    if abs(x - r) >= 1e-6:
        return int(np.ceil(x))
    return int(r)


@njit
def edge_after(t, n, period):
    """
    Clock.edge_after with phase 0: the n-th edge after t.
    """
    return (t // period + 1) * period + (n - 1) * period


spec = [
    # kernel.py: Simulator
    ('now', int64), ('n_events', int64), ('seq', int64), ('stop', boolean),
    ('heap', int64[:]), ('n_heap', int64), ('free', int64[:]), ('n_free', int64),
    ('ev_t', int64[:]), ('ev_priority', int64[:]), ('ev_seq', int64[:]), ('ev_kind', int64[:]),
    ('ev_a', int64[:]), ('ev_b', int64[:]), ('ev_cancelled', boolean[:]),
    # timebase.py: Clock
    ('period', int64),
    # node.py: LinearNode
    ('C', float64), ('V_low', float64), ('V_high', float64), ('V_th', float64),
    ('node_V', float64[:, :]), ('node_s', int64[:, :]),
    # crossbar.py: Crossbar (cell, h and far DACs share one LSB current)
    ('I_lsb', float64), ('xb_J', int64[:, :, :]), ('xb_h', int64[:, :]), ('xb_F', int64[:, :]),
    ('xb_s', int64[:, :]), ('xb_Js', int64[:, :]), ('I', float64[:]),
    # core.py: AnalogCore
    ('K', int64), ('n_spin', int64[:]), ('ids', int64[:, :]), ('t_cmp', int64),
    ('core_t', int64[:]), ('core_held', boolean[:]), ('core_pending', int64[:, :]),
    ('flips_t', int64[:]), ('flips_p', int64[:]), ('flips_k', int64[:]), ('n_flips', int64),
    # flipenc.py: SendPort
    ('sync', int64), ('t_link', int64), ('port_out', boolean[:, :]), ('port_pending', boolean[:, :]),
    ('port_n_pending', int64[:]), ('port_last', int64[:]), ('port_free', int64[:]), ('port_syncing', int64[:]),
    ('port_sending', boolean[:]), ('port_flips', int64[:]), ('port_sent', int64[:]), ('port_pending_max', int64[:]),
    # network.py: Mesh
    ('mesh', int64), ('chip', int64), ('mesh_core', int64[:]), ('fanout_ptr', int64[:]), ('fanout', int64[:]),
    ('mesh_R', int64[:]), ('mesh_Q', int64[:]), ('t_inj', int64), ('t_hop', int64), ('t_chip', int64),
    ('link_free', int64[:, :, :]), ('load', int64[:, :, :]), ('dither', int64), ('t_rx', int64),
    ('rx_free', int64[:]), ('rng', numba.typeof(np.random.default_rng(0))), ('in_flight', int64),
    ('mesh_packets', int64), ('deliveries', int64), ('rx_waits', int64), ('link_waits', int64), ('link_wait_max', int64),
    # farunit.py: FarUnit (rows of all far units in one array, far unit q has rows far_off[q] ...)
    ('far_row', int64[:, :]), ('far_off', int64[:]), ('far_n_rows', int64[:]), ('sram', int64[:, :]),
    ('known', int64[:]), ('far_pending', boolean[:]), ('far_n_pending', int64[:]), ('far_last', int64[:]),
    ('acc', int64[:, :]), ('fdac_max', int64), ('lanes', int64), ('t_pipe', int64),
    ('far_running', boolean[:]), ('far_writes', int64[:]), ('far_free', int64[:]),
    ('far_F', int64[:, :, :]), ('far_head', int64[:]), ('far_tail', int64[:]),   # F codes on their way to the core
    ('far_packets', int64[:]), ('far_cancelled', int64[:]), ('far_applied', int64[:]), ('far_pending_max', int64[:]),
    # controller.py: Controller
    ('last', int64), ('t_quiet', int64), ('t_max', int64), ('t_restart', int64), ('check', int64[:]),
    ('kick_s', int64[:]),
]


@jitclass(spec)
class Machine:
    """
    machine.Machine plus the parts of kernel.Simulator and controller.Controller that act during a run.
    Built by make_machine().
    """
    def __init__(self, K, c, n, n_rows, mesh, n_F, rng):
        self.now, self.n_events, self.seq, self.stop = 0, 0, 0, False
        self.heap = np.zeros(1024, np.int64)
        self.n_heap = 0
        self.free = np.arange(1024)[::-1].copy()
        self.n_free = 1024
        self.ev_t, self.ev_priority, self.ev_seq = np.zeros(1024, np.int64), np.zeros(1024, np.int64), np.zeros(1024, np.int64)
        self.ev_kind, self.ev_a, self.ev_b = np.zeros(1024, np.int64), np.zeros(1024, np.int64), np.zeros(1024, np.int64)
        self.ev_cancelled = np.zeros(1024, np.bool_)
        self.node_V, self.node_s = np.zeros((K, c)), np.zeros((K, c), np.int64)
        self.xb_J = np.zeros((K, c, c), np.int64)
        self.xb_h, self.xb_F = np.zeros((K, c), np.int64), np.zeros((K, c), np.int64)
        self.xb_s, self.xb_Js = np.zeros((K, c), np.int64), np.zeros((K, c), np.int64)
        self.I = np.zeros(c)
        self.K = K
        self.n_spin, self.ids = np.zeros(K, np.int64), np.full((K, c), -1, np.int64)
        self.core_t, self.core_held = np.zeros(K, np.int64), np.zeros(K, np.bool_)
        self.core_pending = np.full((K, 2), -1, np.int64)                             # handle: (slot, seq)
        self.flips_t, self.flips_p, self.flips_k = np.zeros(1024, np.int64), np.zeros(1024, np.int64), np.zeros(1024, np.int64)
        self.n_flips = 0
        self.port_out, self.port_pending = np.zeros((K, c), np.bool_), np.zeros((K, c), np.bool_)
        self.port_n_pending, self.port_last = np.zeros(K, np.int64), np.full(K, -1, np.int64)
        self.port_free, self.port_syncing = np.zeros(K, np.int64), np.zeros(K, np.int64)
        self.port_sending = np.zeros(K, np.bool_)
        self.port_flips, self.port_sent, self.port_pending_max = np.zeros(K, np.int64), np.zeros(K, np.int64), np.zeros(K, np.int64)
        self.mesh = mesh
        self.mesh_core, self.fanout_ptr, self.fanout = np.zeros(n, np.int64), np.zeros(n + 1, np.int64), np.zeros(0, np.int64)
        self.mesh_R, self.mesh_Q = np.zeros(K, np.int64), np.zeros(K, np.int64)
        self.link_free, self.load = np.zeros((mesh, mesh, 4), np.int64), np.zeros((mesh, mesh, 4), np.int64)
        self.rx_free = np.zeros(K, np.int64)
        self.rng = rng
        self.in_flight = 0
        self.mesh_packets, self.deliveries, self.rx_waits, self.link_waits, self.link_wait_max = 0, 0, 0, 0, 0
        self.far_row = np.full((K, n), -1, np.int64)
        self.far_off, self.far_n_rows = np.zeros(K, np.int64), np.zeros(K, np.int64)
        self.sram, self.known = np.zeros((n_rows, c), np.int64), np.zeros(n_rows, np.int64)
        self.far_pending = np.zeros(n_rows, np.bool_)
        self.far_n_pending, self.far_last = np.zeros(K, np.int64), np.full(K, -1, np.int64)
        self.acc = np.zeros((K, c), np.int64)
        self.far_running, self.far_writes, self.far_free = np.zeros(K, np.bool_), np.zeros(K, np.int64), np.zeros(K, np.int64)
        self.far_F = np.zeros((K, n_F, c), np.int64)
        self.far_head, self.far_tail = np.zeros(K, np.int64), np.zeros(K, np.int64)
        self.far_packets, self.far_cancelled = np.zeros(K, np.int64), np.zeros(K, np.int64)
        self.far_applied, self.far_pending_max = np.zeros(K, np.int64), np.zeros(K, np.int64)
        self.last = 0
        self.check = np.full(2, -1, np.int64)
        self.kick_s = np.zeros(n, np.int64)

    # ------------------------------------------------------------------ kernel.py: Simulator

    def sim_less(self, i, j):
        """
        Heap order of slots i and j: (t, priority, seq).
        """
        if self.ev_t[i] != self.ev_t[j]:
            return self.ev_t[i] < self.ev_t[j]
        if self.ev_priority[i] != self.ev_priority[j]:
            return self.ev_priority[i] < self.ev_priority[j]
        return self.ev_seq[i] < self.ev_seq[j]

    def sim_grow(self):
        """
        Double the number of event slots.
        """
        n = len(self.ev_t)
        self.ev_t = np.concatenate((self.ev_t, np.zeros(n, np.int64)))
        self.ev_priority = np.concatenate((self.ev_priority, np.zeros(n, np.int64)))
        self.ev_seq = np.concatenate((self.ev_seq, np.zeros(n, np.int64)))
        self.ev_kind = np.concatenate((self.ev_kind, np.zeros(n, np.int64)))
        self.ev_a = np.concatenate((self.ev_a, np.zeros(n, np.int64)))
        self.ev_b = np.concatenate((self.ev_b, np.zeros(n, np.int64)))
        self.ev_cancelled = np.concatenate((self.ev_cancelled, np.zeros(n, np.bool_)))
        self.heap = np.concatenate((self.heap, np.zeros(n, np.int64)))
        self.free = np.concatenate((np.arange(n, 2 * n)[::-1], self.free))
        self.n_free += n

    def sim_schedule(self, t, priority, kind, a, b):
        """
        At time t, call event kind with (a, b). Returns the handle (slot, seq).
        """
        if self.n_free == 0:
            self.sim_grow()
        self.n_free -= 1
        slot = self.free[self.n_free]
        self.ev_t[slot], self.ev_priority[slot], self.ev_seq[slot] = t, priority, self.seq
        self.ev_kind[slot], self.ev_a[slot], self.ev_b[slot] = kind, a, b
        self.ev_cancelled[slot] = False
        self.seq += 1
        i = self.n_heap                                         # sift up
        self.heap[i] = slot
        self.n_heap += 1
        while i > 0:
            parent = (i - 1) // 2
            if not self.sim_less(self.heap[i], self.heap[parent]):
                break
            self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
            i = parent
        return slot, self.ev_seq[slot]

    def sim_cancel(self, slot, seq):
        """
        Cancelling marks the entry, which the heap then skips. A slot reused by a later event has another seq.
        """
        if slot >= 0 and self.ev_seq[slot] == seq:
            self.ev_cancelled[slot] = True

    def sim_pop(self):
        """
        Remove the first slot from the heap and free it.
        """
        slot = self.heap[0]
        self.n_heap -= 1
        self.heap[0] = self.heap[self.n_heap]
        i = 0                                                   # sift down
        while True:
            child = 2 * i + 1
            if child >= self.n_heap:
                break
            if child + 1 < self.n_heap and self.sim_less(self.heap[child + 1], self.heap[child]):
                child += 1
            if not self.sim_less(self.heap[child], self.heap[i]):
                break
            self.heap[i], self.heap[child] = self.heap[child], self.heap[i]
            i = child
        self.free[self.n_free] = slot
        self.n_free += 1
        return slot

    def sim_peek(self):
        """
        Time of the next event (or -1).
        """
        while self.n_heap > 0 and self.ev_cancelled[self.heap[0]]:
            self.sim_pop()
        return self.ev_t[self.heap[0]] if self.n_heap > 0 else -1

    def sim_run(self, until, max_events):
        """
        Run events with time <= until (-1 = no limit). Returns EMPTY, UNTIL, STOPPED or MAX_EVENTS.
        """
        n = 0
        while True:
            t = self.sim_peek()
            if t < 0:
                return EMPTY
            if until >= 0 and t > until:
                self.now = until
                return UNTIL
            if max_events >= 0 and n >= max_events:
                return MAX_EVENTS
            slot = self.sim_pop()
            kind, a, b = self.ev_kind[slot], self.ev_a[slot], self.ev_b[slot]
            self.now = t
            self.n_events += 1
            n += 1
            self.sim_call(kind, a, b)
            if self.stop:
                self.stop = False
                return STOPPED

    def sim_call(self, kind, a, b):
        """
        fn(*args) of the reference.
        """
        if kind == CROSS:
            self.core_cross(a, b)
        elif kind == SWITCH:
            self.core_switch(a, b)
        elif kind == SEEN:
            self.port_seen(a, b)
        elif kind == SEND:
            self.port_send(a)
        elif kind == HOP:
            self.mesh_hop(a // self.mesh, a % self.mesh, b)
        elif kind == DELIVER:
            self.mesh_deliver(a, b)
        elif kind == STEP:
            self.far_step(a)
        elif kind == WRITE:
            self.far_write(a)
        elif kind == WATCH:
            self.ctl_watch()
        elif kind == PRESET:
            self.ctl_preset()

    # ------------------------------------------------------------------ node.py: LinearNode

    def node_next_crossing(self, p, I):
        """
        (dt, k) ps until the first sigma of core p reaches V_th at constant currents I (A), and which line.
        (-1, -1): no sigma is heading for the threshold.
        """
        k, t_k = -1, np.inf
        for i in range(self.n_spin[p]):
            if self.node_s[p, i] * I[i] < 0:                    # is the spin heading for the threshold?
                t = (self.node_V[p, i] - self.V_th) * self.C / -I[i]
                if t < t_k:
                    k, t_k = i, t
        if k < 0:
            return -1, -1
        return ps(t_k), k

    def node_advance(self, p, I, dt):
        """
        Move every sigma of core p dt ps. Sigmas that reach V_th are put at exactly V_th.
        """
        sdt = dt * 1e-12                                        # seconds(dt)
        for i in range(self.n_spin[p]):
            heading = self.node_s[p, i] * I[i] < 0
            V = min(max(self.node_V[p, i] + I[i] / self.C * sdt, self.V_low), self.V_high)
            if heading and self.node_s[p, i] * (V - self.V_th) <= 0:
                V = self.V_th
            self.node_V[p, i] = V

    # ------------------------------------------------------------------ crossbar.py: Crossbar

    def xb_flip(self, p, k):
        """
        Spin k flips: all lines change by 2 J[:, k] s_k.
        """
        self.xb_s[p, k] = -self.xb_s[p, k]
        for i in range(self.n_spin[p]):
            self.xb_Js[p, i] += 2 * self.xb_J[p, i, k] * self.xb_s[p, k]

    def xb_current(self, p):
        """
        Current into every line (A).
        """
        for i in range(self.n_spin[p]):
            self.I[i] = self.xb_Js[p, i] * self.I_lsb + self.xb_h[p, i] * self.I_lsb + self.xb_F[p, i] * self.I_lsb
        return self.I

    # ------------------------------------------------------------------ core.py: AnalogCore

    def core_sync(self, p):
        """
        Move voltages up to now.
        """
        if not self.core_held[p]:
            self.node_advance(p, self.xb_current(p), self.now - self.core_t[p])
        self.core_t[p] = self.now

    def core_schedule(self, p):
        """
        Replace pending event by the next crossing at the current currents.
        """
        self.sim_cancel(self.core_pending[p, 0], self.core_pending[p, 1])
        dt, k = (-1, -1) if self.core_held[p] else self.node_next_crossing(p, self.xb_current(p))
        if k < 0:
            self.core_pending[p, 0], self.core_pending[p, 1] = -1, -1
        else:
            self.core_pending[p, 0], self.core_pending[p, 1] = self.sim_schedule(self.now + dt, ANALOG, CROSS, p, k)

    def core_cross(self, p, k):
        """
        Spin k reaches V_th. Cells see the new sigma t_cmp later (comparator delay).
        """
        self.core_sync(p)
        self.node_s[p, k] = -self.node_s[p, k]                  # node.flip(k)
        self.sim_schedule(self.now + self.t_cmp, ANALOG, SWITCH, p, k)
        self.core_schedule(p)

    def core_switch(self, p, k):
        """
        Comparator output of spin k changes t_cmp later.
        """
        if self.xb_s[p, k] == self.node_s[p, k]:
            return
        self.core_sync(p)
        self.xb_flip(p, k)
        if self.n_flips == len(self.flips_t):
            self.flips_t = np.concatenate((self.flips_t, np.zeros(self.n_flips, np.int64)))
            self.flips_p = np.concatenate((self.flips_p, np.zeros(self.n_flips, np.int64)))
            self.flips_k = np.concatenate((self.flips_k, np.zeros(self.n_flips, np.int64)))
        self.flips_t[self.n_flips], self.flips_p[self.n_flips], self.flips_k[self.n_flips] = self.now, p, k
        self.n_flips += 1
        self.port_flip(p, k)                                    # on_flip
        self.core_schedule(p)

    def core_set_far(self, p, F):
        """
        The far unit writes a new bias code for all lines.
        """
        self.core_sync(p)
        self.xb_F[p, :self.n_spin[p]] = F[:self.n_spin[p]]
        self.core_schedule(p)

    def core_hold(self, p):
        """
        Freeze voltages.
        """
        self.core_sync(p)
        self.core_held[p] = True
        self.core_schedule(p)

    def core_preset(self, p, s):
        """
        While frozen, set all nodes to the rail of its sigma.
        """
        for i in range(self.n_spin[p]):
            self.node_V[p, i] = self.V_high if s[i] > 0 else self.V_low
            self.node_s[p, i] = s[i]
        for k in range(self.n_spin[p]):
            if self.xb_s[p, k] != s[k]:
                self.xb_flip(p, k)
                self.port_flip(p, k)                            # on_flip

    def core_release(self, p):
        """
        Voltages can move again.
        """
        self.core_sync(p)
        self.core_held[p] = False
        self.core_schedule(p)

    # ------------------------------------------------------------------ flipenc.py: SendPort

    def port_busy(self, p):
        return self.port_syncing[p] > 0 or self.port_sending[p]

    def port_flip(self, p, k):
        """
        A comparator of spin k has flipped.
        """
        self.ctl_touch()
        if not self.port_out[p, k]:
            return
        self.port_syncing[p] += 1
        t = self.now if self.sync == 0 else edge_after(self.now, self.sync, self.period)
        self.sim_schedule(t, EDGE, SEEN, p, k)

    def port_seen(self, p, k):
        """
        The flip comes out of the synchronizer.
        """
        self.port_syncing[p] -= 1
        self.port_flips[p] += 1
        self.port_pending[p, k] = not self.port_pending[p, k]
        self.port_n_pending[p] += 1 if self.port_pending[p, k] else -1
        self.port_pending_max[p] = max(self.port_pending_max[p], self.port_n_pending[p])
        if self.port_pending[p, k] and not self.port_sending[p]:
            self.port_sending[p] = True
            self.sim_schedule(max(self.now, self.port_free[p]), EDGE, SEND, p, 0)

    def port_send(self, p):
        """
        The mesh is free. Send the next pending spin after the last one sent.
        """
        k = -1
        for j in range(self.n_spin[p]):
            i = (self.port_last[p] + 1 + j) % self.n_spin[p]
            if self.port_pending[p, i]:
                k = i
                break
        if k < 0:
            self.port_sending[p] = False
            return
        self.port_pending[p, k] = False
        self.port_n_pending[p] -= 1
        self.port_last[p] = k
        self.mesh_send(self.ids[p, k])
        self.port_sent[p] += 1
        self.port_free[p] = self.now + self.t_link
        if self.port_n_pending[p] > 0:
            self.sim_schedule(self.port_free[p], EDGE, SEND, p, 0)
        else:
            self.port_sending[p] = False

    # ------------------------------------------------------------------ network.py: Mesh

    def mesh_send(self, k):
        """
        Spin k has flipped. The packet reaches the router of its core t_inj later.
        """
        p = self.mesh_core[k]
        self.mesh_packets += 1
        d = self.rng.integers(0, self.dither + 1) * self.period if self.dither else 0
        self.in_flight += self.fanout_ptr[k + 1] - self.fanout_ptr[k]
        self.sim_schedule(self.now + self.t_inj + d, EDGE, HOP, self.mesh_R[p] * self.mesh + self.mesh_Q[p], k)

    def mesh_hop(self, r, q, k):
        """
        The packet for spin k is in the router at row r, column q. Its copy here carries the destinations whose
        XY route from the source passes this router (the dests of the reference).
        """
        p0 = self.mesh_core[k]
        R0, Q0 = self.mesh_R[p0], self.mesh_Q[p0]
        go = [False, False, False, False]                       # E, W, S, N
        for j in range(self.fanout_ptr[k], self.fanout_ptr[k + 1]):
            p = self.fanout[j]
            R1, Q1 = self.mesh_R[p], self.mesh_Q[p]
            if not ((r == R0 and min(Q0, Q1) <= q <= max(Q0, Q1)) or (q == Q1 and min(R0, R1) <= r <= max(R0, R1))):
                continue                                        # another branch of the tree
            dr, dq = R1 - r, Q1 - q
            if dq > 0:
                go[0] = True
            elif dq < 0:
                go[1] = True
            elif dr > 0:
                go[2] = True
            elif dr < 0:
                go[3] = True
            else:
                self.mesh_arrive(p, k)
        for w in range(4):
            if go[w]:
                self.mesh_link(r, q, w, k)

    def mesh_link(self, r, q, w, k):
        """
        A copy goes out of router (r, q) in direction w. It waits until the link is free.
        """
        t = max(self.now, self.link_free[r, q, w])
        self.link_free[r, q, w] = t + self.t_link
        self.load[r, q, w] += 1
        self.link_waits += t > self.now
        self.link_wait_max = max(self.link_wait_max, (t - self.now) // self.period)
        r2, q2 = r + STEP_R[w], q + STEP_Q[w]
        edge = r // self.chip != r2 // self.chip or q // self.chip != q2 // self.chip
        self.sim_schedule(t + self.t_hop + edge * self.t_chip, EDGE, HOP, r2 * self.mesh + q2, k)

    def mesh_arrive(self, q, k):
        """
        Packet reaches the router of core q. It waits for the router's port to the far unit.
        """
        t = max(self.now, self.rx_free[q])
        self.rx_waits += t > self.now
        self.rx_free[q] = t + self.t_rx
        self.sim_schedule(t, EDGE, DELIVER, q, k)

    def mesh_deliver(self, q, k):
        """
        Packet reaches core q.
        """
        self.in_flight -= 1
        self.deliveries += 1
        self.far_receive(q, k)

    # ------------------------------------------------------------------ farunit.py: FarUnit

    def far_busy(self, q):
        return self.far_running[q] or self.far_writes[q] > 0

    def far_code(self, q):
        """
        F codes from the accumulators.
        """
        return np.minimum(np.maximum(self.acc[q], -self.fdac_max), self.fdac_max)

    def far_receive(self, q, k):
        """
        A packet is received from the mesh network. A far spin k has flipped.
        """
        r = self.far_off[q] + self.far_row[q, k]                # look up the row address
        self.far_pending[r] = not self.far_pending[r]           # flip the pending bit (two flips cancel)
        self.far_n_pending[q] += 1 if self.far_pending[r] else -1
        self.far_packets[q] += 1
        self.far_cancelled[q] += not self.far_pending[r]
        self.far_pending_max[q] = max(self.far_pending_max[q], self.far_n_pending[q])
        if self.far_pending[r] and not self.far_running[q]:
            self.far_running[q] = True
            self.sim_schedule(max(self.now, self.far_free[q]), EDGE, STEP, q, 0)

    def far_step(self, q):
        """
        One cycle. Round robin: the first `lanes` pending rows after the last row taken.
        """
        self.far_free[q] = self.now + self.period
        n_take = min(self.lanes, self.far_n_pending[q])
        taken = 0
        for j in range(self.far_n_rows[q]):
            if taken == n_take:
                break
            r = (self.far_last[q] + 1 + j) % self.far_n_rows[q]
            i = self.far_off[q] + r
            if self.far_pending[i]:
                self.far_pending[i] = False
                self.known[i] = -self.known[i]
                for c in range(self.n_spin[q]):
                    self.acc[q, c] += 2 * self.known[i] * self.sram[i, c]
                taken += 1
                last = r
        self.far_n_pending[q] -= taken
        self.far_applied[q] += taken
        if taken:
            self.far_last[q] = last
            self.far_writes[q] += 1
            self.far_F[q, self.far_tail[q]] = self.far_code(q)  # self.code() travels with the write event
            self.far_tail[q] = (self.far_tail[q] + 1) % self.far_F.shape[1]
            self.sim_schedule(self.now + self.t_pipe, EDGE, WRITE, q, 0)
        if self.far_n_pending[q]:
            self.sim_schedule(self.now + self.period, EDGE, STEP, q, 0)
        else:
            self.far_running[q] = False
        self.ctl_touch()

    def far_write(self, q):
        """
        End of pipeline. The far spin bias registers are updated.
        """
        self.far_writes[q] -= 1
        F = self.far_F[q, self.far_head[q]]
        self.far_head[q] = (self.far_head[q] + 1) % self.far_F.shape[1]
        self.core_set_far(q, F)
        self.ctl_touch()

    # ------------------------------------------------------------------ machine.py: Machine

    def start(self):
        """
        The events Machine.__init__ schedules: every core schedules, then its far unit writes the starting F.
        """
        for p in range(self.K):
            self.core_t[p] = self.now
            self.core_schedule(p)
            self.core_set_far(p, self.far_code(p))

    def busy(self):
        """
        A flip is still on its way.
        """
        if self.in_flight > 0:
            return True
        for p in range(self.K):
            if self.port_busy(p) or self.far_busy(p):
                return True
        return False

    @property
    def s(self):
        """
        Sigma of every spin.
        """
        s = np.zeros(len(self.kick_s), np.int64)
        for p in range(self.K):
            for i in range(self.n_spin[p]):
                s[self.ids[p, i]] = self.xb_s[p, i]
        return s

    # ------------------------------------------------------------------ controller.py: Controller

    def ctl_touch(self):
        """
        Something happened. Restart the timer.
        """
        self.last = self.now

    def ctl_watch(self):
        """
        The timer ran out. check for activity or if something is busy.
        """
        if self.now - self.last < self.t_quiet:
            self.check[0], self.check[1] = self.sim_schedule(self.last + self.t_quiet, CONTROL, WATCH, 0, 0)
        elif self.busy():
            self.check[0], self.check[1] = self.sim_schedule(self.now + self.period, CONTROL, WATCH, 0, 0)
        else:
            self.stop = True

    def ctl_run(self):
        """
        One run until quiet or t_max. Returns (quiet, run time in ps).
        """
        t0 = self.now
        self.last = t0
        self.check[0], self.check[1] = self.sim_schedule(t0 + self.t_quiet, CONTROL, WATCH, 0, 0)
        why = self.sim_run(t0 + self.t_max, -1)
        self.sim_cancel(self.check[0], self.check[1])
        return why == STOPPED, self.now - t0

    def ctl_kick(self, s):
        """
        Freeze all cores. t_restart later force s on the rails.
        """
        for p in range(self.K):
            self.core_hold(p)
        self.kick_s[:] = s
        self.sim_schedule(self.now + self.t_restart, CONTROL, PRESET, 0, 0)
        self.sim_run(-1, -1)
        for p in range(self.K):
            self.core_release(p)

    def ctl_preset(self):
        for p in range(self.K):
            self.core_preset(p, self.kick_s[self.ids[p, :self.n_spin[p]]])


def make_machine(J, h, spec, v0, rng):
    """
    Machine.__init__: builds every block from J, h and the spec. Returns the numba machine and the mapping.
    """
    ns = spec.node
    clock = Clock('core', spec.clock.f_core)
    cell = CurrentDAC(spec.core.cell_bits, ns.I_unit)
    hdac = CurrentDAC(spec.core.h_bits, ns.I_unit)
    fdac = CurrentDAC(spec.far.dac_bits, ns.I_unit)
    m = Mapping(J, h, spec, cell, hdac)
    n, K, c = len(h), m.K, spec.core.c
    s = np.where(v0 >= 0, 1, -1)
    n_rows = np.array([len(r) for r in m.rows])
    mach = Machine(K, c, n, max(1, n_rows.sum()), spec.net.mesh, clock.cycles(spec.far.t_pipe) // clock.period + 2, rng)

    # timebase, node, crossbar, core
    mach.period = clock.period
    mach.C, mach.V_low, mach.V_high, mach.V_th = ns.C, ns.V_low, ns.V_high, ns.V_th
    mach.I_lsb = ns.I_unit
    mach.t_cmp = ns.t_cmp
    for p in range(K):
        ids = m.spins[p]
        node_V = ns.V_th + v0[ids] * (ns.V_high - ns.V_th)
        node_s = np.where(node_V >= ns.V_th, 1, -1)
        mach.n_spin[p] = len(ids)
        mach.ids[p, :len(ids)] = ids
        mach.node_V[p, :len(ids)] = node_V
        mach.node_s[p, :len(ids)] = node_s
        mach.xb_J[p, :len(ids), :len(ids)] = m.local[p]
        mach.xb_h[p, :len(ids)] = m.h[ids]
        mach.xb_s[p, :len(ids)] = node_s
        mach.xb_Js[p, :len(ids)] = np.asarray(m.local[p], dtype=np.int64) @ node_s

    # flipenc, network
    net = spec.net
    mach.sync, mach.t_link = spec.clock.sync, clock.cycles(net.t_link)
    mach.chip = net.chip
    mach.mesh_core[:] = m.core
    mach.fanout_ptr[:] = np.concatenate(([0], np.cumsum([len(f) for f in m.fanout])))
    mach.fanout = np.concatenate(m.fanout).astype(np.int64)
    mach.mesh_R[:], mach.mesh_Q[:] = np.divmod(m.tile, net.mesh)
    mach.t_inj, mach.t_hop, mach.t_chip = clock.cycles(net.t_inj), clock.cycles(net.t_hop), clock.cycles(net.t_chip)
    mach.dither, mach.t_rx = net.dither, clock.cycles(net.t_rx)
    for p in range(K):
        mach.port_out[p, :len(m.spins[p])] = [len(m.fanout[k]) > 0 for k in m.spins[p]]

    # far unit
    mach.far_off[:] = np.concatenate(([0], np.cumsum(n_rows)[:-1]))
    mach.far_n_rows[:] = n_rows
    for q in range(K):
        rows, o = m.rows[q], mach.far_off[q]
        mach.far_row[q, rows] = np.arange(len(rows))
        mach.sram[o:o + len(rows), :len(m.spins[q])] = m.sram[q]
        mach.known[o:o + len(rows)] = s[rows]
        mach.acc[q, :len(m.spins[q])] = m.sram[q].T @ s[rows]
    mach.fdac_max = 2**62 if fdac.bits is None else fdac.max_code
    mach.lanes, mach.t_pipe = spec.far.lanes, clock.cycles(spec.far.t_pipe)

    # controller
    ctl = spec.control
    mach.t_quiet, mach.t_max, mach.t_restart = ctl.t_quiet, ctl.t_max, clock.cycles(ctl.t_restart)
    mach.start()
    return mach, m


class Controller:
    """
    controller.Controller with the machine in numba: the same seeds, kicks and results.
    """
    def __init__(self, J, h, spec):
        c = spec.control
        n = len(h)
        r_v0, r_net, self.rng = (np.random.default_rng(s) for s in np.random.SeedSequence(c.seed).spawn(3))
        self.v0 = r_v0.uniform(-1, 1, n)
        self.machine, self.map = make_machine(J, h, spec, self.v0, r_net)
        self.J = self.map.J.astype(float)                           # exact for integer codes, and numpy uses BLAS for floats
        self.sim = self.machine                                     # sim.now and sim.n_events
        self.n_runs = c.n_runs
        self.sizes = np.round(np.linspace(max(1, c.kick_start * n), max(1, c.kick_end * n), c.n_runs)).astype(int)
        self.best_s, self.best_E = None, np.inf
        self.start, self.end, self.E, self.t_run, self.why = [], [], [], [], []

    def energy(self, s):
        """
        Machine.energy: E = -1/2 s J s - h s
        """
        return -0.5 * s @ self.J @ s - self.map.h @ s

    def solve(self):
        """
        All runs. Return best state and energy.
        """
        n = len(self.v0)
        for i in range(self.n_runs):
            self.start.append(self.machine.s)
            quiet, t = self.machine.ctl_run()
            self.t_run.append(t)
            self.why.append('quiet' if quiet else 't_max')
            s = self.machine.s
            E = self.energy(s)
            self.end.append(s)
            self.E.append(E)
            if E < self.best_E:
                self.best_s, self.best_E = s, E
            if i < self.n_runs - 1:
                s = self.best_s.copy()
                s[self.rng.choice(n, self.sizes[i], replace=False)] *= -1
                self.machine.ctl_kick(s)
        return self.best_s, self.best_E

    def flips(self):
        """
        (time, spin) of every flip, per core: [core.flips for core in machine.cores].
        """
        m = self.machine
        t, p, k = m.flips_t[:m.n_flips], m.flips_p[:m.n_flips], m.flips_k[:m.n_flips]
        return [np.stack((t[p == q], k[p == q]), axis=1) for q in range(m.K)]
