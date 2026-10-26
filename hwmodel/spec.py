"""
spec.py

Parameters for every block with units.
"""

from dataclasses import dataclass, field, fields

from hwmodel.timebase import ps


def p(default, unit='', doc=''):
    """
    A parameter with unit and description.
    """
    return field(default=default, metadata={'unit': unit, 'doc': doc})


class Spec:
    """
    Prints parameters as a table.
    """
    def __str__(self):
        lines = [type(self).__name__]
        for f in fields(self):
            lines.append(f"  {f.name:12s} {str(getattr(self, f.name)):>12s}  {f.metadata['unit']:9s} {f.metadata['doc']}")
        return '\n'.join(lines)


@dataclass
class NodeSpec(Spec):
    """
    The analog solver.
    """
    C       : float = p(50e-15, 'F', 'node capacitance (wire, gate, etc.)')
    I_unit  : float = p(100e-9, 'A', 'LSB current')
    V_low   : float = p(0.0, 'V', 'lower_rail')
    V_high  : float = p(0.8, 'V', 'upper rail')
    V_th    : float = p(0.4, 'V', 'comparator threshold')
    t_cmp   : int   = p(0, 'ps', 'comparator delay')

    def ramp_time(self, I):
        """
        Time from the upper rail to the threshold at a constant current I (A).
        """
        return ps(self.C * (self.V_high - self.V_th) / I)


@dataclass
class CoreSpec(Spec):
    """
    The analog solver core.
    """
    c           : int = p(64, 'spins', 'spins per core')
    cell_bits   : int = p(3, 'bits', 'magnitude bits of a coupling cell')
    h_bits      : int = p(6, 'bits', 'magnitude bits of an h-row cell')


@dataclass
class ClockSpec(Spec):
    """
    Digital logic.
    """
    f_core  : float = p(1e9, 'Hz', 'logic clock')
    sync    : int   = p(2, 'edges', 'cycles the comparator must be stable (2 = 2FF)')


@dataclass
class NetSpec(Spec):
    """
    Networking/routing for each analog core.

    NESW routers.
    """
    ports       : str  = p('per_core', '', "'per_core': one port per core, 'shared': one bus for all")
    multicast   : bool = p(True, '', 'send only to cores that hold a row for the spin')
    send_merge  : bool = p(True, '', 'send queue merges waiting flips of the same spin')
    t_inj       : int  = p(2, 'cycles', 'core -> router')
    t_hop       : int  = p(2, 'cycles', 'per router jump')
    t_link      : int  = p(1, 'cycles', 'per packet on a port')
    dither      : int  = p(0, 'cycles', 'random extra delay per packet (for stopping oscillation)')


@dataclass
class FarSpec(Spec):
    """
    Digital unit that receives spin flip events and reprograms the corresponding bias.
    """
    dac_bits    : int | None = p(3, 'bits', 'magnitude bits of an F_row DAC. None = exact')
    sram_format : str  = p('dense', '', "'dense': c codes per row, 'packed': (column, code) per nonzero")
    merge       : bool = p(True, '', 'dirty bit per row: waiting events of the same spin merge')
    lanes       : int  = p(1, 'rows/cyc', 'rows the far unit takes per cycle')
    t_pipe      : int  = p(4, 'cycles', 'row read -> adds -> DAC code')


@dataclass
class ControlSpec(Spec):
    """
    Controller. Responsible for kicking and the outer iterations.
    """
    n_runs      : int   = p(300, 'runs', 'restarts per solve')
    kick_start  : float = p(0.2, 'of N', 'cluster size of the first kick')
    kick_end    : float = p(0.0, 'of N', 'cluster size of the last kick (0 = one spin)')
    kick_mode   : str   = p('hold', '', "'hold': nodes are frozen while the kick is processed, 'queued': kicks propagate while cores are running")
    t_quiet     : int   = p(60_000, 'ps', 'read out after this long without activity')
    t_max       : int   = p(1_500_000, 'ps', 'read out at the latest after this long')
    t_restart   : int   = p(20, 'cycles', 'readout, energy, winner and kick between runs')
    seed        : int   = p(0, '', 'one seed for every random choice')


@dataclass
class MachineSpec(Spec):
    node        : NodeSpec       = field(default_factory=NodeSpec)
    core        : CoreSpec       = field(default_factory=CoreSpec)
    clock       : ClockSpec      = field(default_factory=ClockSpec)
    net         : NetSpec        = field(default_factory=NetSpec)
    far         : FarSpec        = field(default_factory=FarSpec)
    control     : ControlSpec    = field(default_factory=ControlSpec)

    def derived(self):
        """
        Core clock cycles per full-scale voltage ramp.
        """
        I_fs = self.node.I_unit * (2 ** self.core.cell_bits - 1)
        t_ramp = self.node.ramp_time(I_fs)
        cycles = t_ramp * self.clock.f_core / 1e12
        return (f'Derived\n  full-scale cell current {I_fs * 1e9:.0f} nA, ramp rail -> threshold {t_ramp} ps'
                f' = {cycles:.1f} core cycles')

    def __str__(self):
        return '\n'.join([str(getattr(self, f.name)) for f in fields(self)] + [self.derived()])
