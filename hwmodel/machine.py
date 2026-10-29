"""
machine.py

The whole multi-core machine. Multiple analog cores on a mesh.
"""

import numpy as np

from hwmodel.core import AnalogCore
from hwmodel.crossbar import Crossbar
from hwmodel.dac import CurrentDAC
from hwmodel.farunit import FarUnit
from hwmodel.flipenc import SendPort
from hwmodel.kernel import Simulator
from hwmodel.mapping import Mapping
from hwmodel.network import Mesh
from hwmodel.node import LinearNode
from hwmodel.timebase import Clock


class Machine:
    """
    Builds every block from J, h and the spec.
    """
    def __init__(self, J, h, spec, v0, rng, touch=lambda: None):
        ns = spec.node
        self.sim = Simulator()
        self.clock = Clock('core', spec.clock.f_core)
        cell = CurrentDAC(spec.core.cell_bits, ns.I_unit)
        hdac = CurrentDAC(spec.core.h_bits, ns.I_unit)
        fdac = CurrentDAC(spec.far.dac_bits, ns.I_unit)
        self.map = m = Mapping(J, h, spec, cell, hdac)
        s = np.where(v0 >= 0, 1, -1)
        self.mesh = Mesh(self.sim, self.clock, spec.net, m, rng)
        self.cores, self.ports, self.fars = [], [], []
        for p in range(m.K):
            ids = m.spins[p]
            port = SendPort(self.sim, self.clock, spec.clock.sync, spec.net, self.mesh, ids)
            node = LinearNode(ns, ns.V_th + v0[ids] * (ns.V_high - ns.V_th))
            xb = Crossbar(m.local[p], m.h[ids], node.s, cell, hdac, fdac)
            core = AnalogCore(self.sim, node, xb, ns.t_cmp, on_flip=port.flip)
            far = FarUnit(self.sim, core, m.rows[p], m.sram[p], s[m.rows[p]], fdac, spec.far, self.clock, touch)
            self.cores.append(core)
            self.ports.append(port)
            self.fars.append(far)
        self.mesh.far = self.fars

    @property
    def s(self):
        """
        Sigma of every spin.
        """
        s = np.zeros(len(self.map.core), dtype=int)
        for p, core in enumerate(self.cores):
            s[self.map.spins[p]] = core.xb.s
        return s

    def energy(self):
        """
        E = -1/2 s J s - h s
        """
        s = self.s
        return -0.5 * s @ self.map.J @ s - self.map.h @ s

