import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from hwmodel.timebase import ps, Clock
    from hwmodel.kernel import Simulator, ANALOG, EDGE, CONTROL

    return ANALOG, CONTROL, Clock, EDGE, Simulator, ps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Timebase and Kernel
    """)
    return


@app.cell
def _(Clock, ps):
    core = Clock('core', 1e9)
    net = Clock('net', 0.8e9)
    print(core, net)
    print([core.next_edge(t) for t in (0, 1, 999, 1000, 1001)])     # next edge
    print(core.edge_after(1500, 2))                                 # 2nd edge after 1500
    print(net.period, [net.next_edge(t) for t in (1, 1250, 1251)])
    print(ps(1e-9), ps(30e-9), ps(2.5e-12))
    return (core,)


@app.cell
def _(ANALOG, CONTROL, EDGE, Simulator):
    sim = Simulator()
    log = []
    sim.schedule(1000, log.append, 'controller', priority=CONTROL)
    sim.schedule(1000, log.append, 'edge', priority=EDGE)
    sim.schedule(1000, log.append, 'analog', priority=ANALOG)
    h = sim.schedule(1500, log.append, 'cancelled')
    sim.schedule(2000, log.append, 'late')
    sim.cancel(h)
    print(sim.run(until=1800), sim.now, log)
    print(sim.run(), sim.now, log)
    return


@app.cell
def _(Simulator, core):
    sim2 = Simulator()
    ticks = []

    def tick():
        ticks.append(sim2.now)
        if len(ticks) == 3:
            sim2.stop()
        sim2.schedule(core.next_edge(sim2.now, after=True), tick)

    sim2.schedule(0, tick)
    print(sim2.run(), list(ticks))
    print(sim2.run(until=5500), ticks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Specs
    """)
    return


@app.cell
def _():
    from dataclasses import replace
    from hwmodel.spec import MachineSpec, NodeSpec

    spec = MachineSpec()
    print(spec)

    fast = replace(spec, clock=replace(spec.clock, f_core=3e9))     # one field changed, the rest kept
    print(fast.derived())
    return NodeSpec, replace, spec


@app.cell
def _(mo):
    mo.md(r"""
    ## Current DAC
    """)
    return


@app.cell
def _():
    from hwmodel.dac import CurrentDAC
    import numpy as np

    cell = CurrentDAC(3, 100e-9)    # magnitude bits, unit current

    # the nearst codes for 
    cell.code([-1.0, -0.3, 0.0, 0.3, 1.0, 2.0], lsb=1/7), cell.current(7)
    return CurrentDAC, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Node
    """)
    return


@app.cell
def _(NodeSpec, np):
    from hwmodel.node import LinearNode
    nspec = NodeSpec()
    node = LinearNode(nspec, [nspec.V_high])
    I = np.array([-700e-9])
    dt, k = node.next_crossing(I)
    dt, k, nspec.ramp_time(700e-9)
    return I, LinearNode, dt, node, nspec


@app.cell
def _(I, dt, node):
    node.advance(I, dt)
    node.flip(0)
    node.V, node.s
    return


@app.cell
def _(I, node):
    node.advance(I, 100_000)
    node.V, node.next_crossing(I)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Crossbar
    """)
    return


@app.cell
def _(CurrentDAC, np, nspec):
    from hwmodel.crossbar import Crossbar

    rng = np.random.default_rng(0)

    # Generate random quantized 4-bit J
    Jt = rng.integers(-7, 8, (64, 64))
    np.fill_diagonal(Jt, 0)

    # Generate random qunatized 6-bit h
    ht = rng.integers(-63, 64, 64)

    # Define the crossbar
    xb = Crossbar(Jt, ht, rng.choice([-1, 1], 64),
                  CurrentDAC(3, nspec.I_unit), CurrentDAC(6, nspec.I_unit), CurrentDAC(3, nspec.I_unit))

    # Flip 1000 random spins
    for _k in rng.integers(0, 64, 1000):
        xb.flip(_k)

    # Get the current
    I0 = xb.current()

    xb.F[5] = -7

    # Does the model current align with J @ x + h
    np.allclose(I0, nspec.I_unit * (Jt @ xb.s + ht)), xb.current()[5] - I0[5]
    return (Crossbar,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Core
    """)
    return


@app.cell
def _(Crossbar, CurrentDAC, LinearNode, Simulator, np, nspec):
    from scripts.lagrange import analog_event
    from hwmodel.core import AnalogCore

    ## Compare hwmodel against known good event solver

    # Generate random problem
    _rng = np.random.default_rng(0)
    _Jt = np.triu(_rng.integers(-7, 8, (64, 64)), 1)
    _Jt = _Jt + _Jt.T
    _ht = _rng.integers(-7, 8, 64)
    v0 = _rng.uniform(-1, 1, 64)

    # Run on good solver
    s_ref, t_ref, n_ref = analog_event(_Jt.astype(float), _ht.astype(float), v0.copy(), 10_000)

    # Set up hwmodel
    _sim = Simulator()
    _node = LinearNode(nspec, nspec.V_th + v0 * (nspec.V_high - nspec.V_th))
    _xb = Crossbar(_Jt, _ht, _node.s, CurrentDAC(3, nspec.I_unit), CurrentDAC(6, nspec.I_unit), CurrentDAC(3, nspec.I_unit))
    _core = AnalogCore(_sim, _node, _xb, nspec.t_cmp)
    _sim.run()
    unit = nspec.C * (nspec.V_high - nspec.V_th) / nspec.I_unit * 1e12     # ps per analog_event time unit

    np.array_equal(_xb.s, s_ref), len(_core.flips), n_ref, _sim.now, round(t_ref * unit)
    return (analog_event,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Mapping
    """)
    return


@app.cell
def _(CurrentDAC, np, replace, spec):
    from hwmodel.mapping import Mapping
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    # 8x8 lattice with open edges and random +-1 couplings -> 4 cores of 16 spins
    _L = 8
    _rng = np.random.default_rng(0)
    Jl = np.zeros((_L * _L, _L * _L))
    for _r in range(_L):
        for _q in range(_L):
            for _r2, _q2 in ((_r, _q + 1), (_r + 1, _q)):          # right and down neighbour
                if _r2 < _L and _q2 < _L:
                    Jl[_r * _L + _q, _r2 * _L + _q2] = Jl[_r2 * _L + _q2, _r * _L + _q] = _rng.choice([-1, 1])

    ml = Mapping(Jl, np.zeros(_L * _L), replace(spec, core=replace(spec.core, c=16)),
                    CurrentDAC(3, spec.node.I_unit), CurrentDAC(6, spec.node.I_unit))

    _colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']      # one colour per core
    _far, _grid, _bg = '#0b0b0b', '#c3c2b7', '#fcfcfb'
    fig_map, (_a, _o, _b) = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=_bg)

    def _pixels(order):
        """
        J as an RGB picture with the spins in this order: core colour inside a core, black between cores.
        """
        _img = np.ones((len(order), len(order), 3)) * to_rgb(_bg)
        _c = ml.core[order]
        for _i, _j in zip(*np.nonzero(ml.J[np.ix_(order, order)])):
            _img[_i, _j] = to_rgb(_colors[_c[_i]]) if _c[_i] == _c[_j] else to_rgb(_far)
        return _img

    # left: the lattice. Spins coloured by core, far couplings dashed black
    _y, _x = np.divmod(np.arange(_L * _L), _L)                     # position of spin i on the lattice
    for _i, _j in zip(*np.nonzero(np.triu(Jl))):                   # every coupling once
        _same = ml.core[_i] == ml.core[_j]
        _a.plot([_x[_i], _x[_j]], [_y[_i], _y[_j]], color=_colors[ml.core[_i]] if _same else _far,
                lw=1.5 if _same else 2, ls='-' if _same else '--', zorder=1)
    _a.scatter(_x, _y, s=260, c=[_colors[_p] for _p in ml.core], edgecolors=_bg, linewidths=2, zorder=2)
    for _i in range(_L * _L):
        _a.text(_x[_i], _y[_i], ml.core[_i], ha='center', va='center', fontsize=8, color=_far, zorder=3)
    _a.set_title('8x8 lattice: spins coloured by core\ndashed = far coupling (goes through the network)', fontsize=10)
    _a.invert_yaxis()
    _a.set_aspect('equal')
    _a.axis('off')

    # middle: J in the original spin order
    _o.imshow(_pixels(np.arange(_L * _L)), interpolation='nearest')
    _o.set_xticks(np.arange(0, _L * _L, _L))                         # a tick at the start of every lattice row
    _o.set_yticks(np.arange(0, _L * _L, _L))
    _o.set_title('J in the original spin order\nsame entries, the cores are interleaved', fontsize=10)

    # right: J with the spins reordered core by core
    _b.imshow(_pixels(np.concatenate(ml.spins)), interpolation='nearest')    # core 0's spins, then core 1's,
    _n = np.array([len(_s) for _s in ml.spins])
    for _e in np.cumsum(_n)[:-1] - 0.5:                            # lines between the cores
        _b.axhline(_e, color=_grid, lw=0.8)
        _b.axvline(_e, color=_grid, lw=0.8)
    _mid = np.cumsum(_n) - _n / 2 - 0.5
    _b.set_xticks(_mid, [f'core {_p}' for _p in range(ml.K)])
    _b.set_yticks(_mid, [f'core {_p}' for _p in range(ml.K)])
    _b.set_title('J reordered core by core\ncoloured blocks = crossbar cells, black = far SRAM rows', fontsize=10)

    for _ax in (_o, _b):
        _ax.tick_params(length=0)
        for _s in _ax.spines.values():
            _s.set_visible(False)
    fig_map.tight_layout()

    for _p in range(ml.K):
        print(f'core {_p}: spins {ml.spins[_p]}, local couplings {np.count_nonzero(np.triu(ml.local[_p]))}, '
                f'SRAM rows {len(ml.rows[_p])}: {ml.rows[_p]}')
    print('spins whose flips leave their core:', sum(len(_f) > 0 for _f in ml.fanout), 'of', _L * _L)
    fig_map
    return Mapping, plt


@app.cell
def _(CurrentDAC, Mapping, np, plt, replace, spec):
    from matplotlib.colors import ListedColormap

    # 64 spins in 4 hidden groups of 16. Strong couplings (3..7) inside a group, weak ones (1..4) between groups.
    # The spin numbers are shuffled, so the groups can't be seen in the original order.
    _n = 64
    _rng = np.random.default_rng(0)
    _group = _rng.permutation(np.repeat(np.arange(4), 16))         # hidden group of every spin
    _same = _group[:, None] == _group
    _w = np.where(_same, _rng.integers(3, 8, (_n, _n)), _rng.integers(1, 5, (_n, _n)))
    _on = _rng.random((_n, _n)) < np.where(_same, 0.4, 0.15)        # dense inside a group, sparse between
    Jc = np.triu(_w * _on * _rng.choice([-1, 1], (_n, _n)), 1)
    Jc = Jc + Jc.T

    mc = Mapping(Jc, np.zeros(_n), replace(spec, core=replace(spec.core, c=16)),
                    CurrentDAC(3, spec.node.I_unit), CurrentDAC(6, spec.node.I_unit))

    # |J| code 0..7 -> background + one blue shade per code (light = weak, dark = strong)
    _bg = '#fcfcfb'
    _cmap = ListedColormap([_bg, '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
    fig_strong, (_a, _b) = plt.subplots(1, 2, figsize=(10, 5), facecolor=_bg)

    # left: |J| in the original spin order
    _a.imshow(np.abs(mc.J), cmap=_cmap, vmin=-0.5, vmax=7.5, interpolation='nearest')
    _a.set_title('|J| in the original spin order', fontsize=10)

    # middle: |J| reordered core by core
    _order = np.concatenate(mc.spins)
    _im = _b.imshow(np.abs(mc.J[np.ix_(_order, _order)]), cmap=_cmap, vmin=-0.5, vmax=7.5, interpolation='nearest')
    for _e in np.arange(16, _n, 16) - 0.5:
        _b.axhline(_e, color='#52514e', lw=0.8)
        _b.axvline(_e, color='#52514e', lw=0.8)
    _b.set_xticks(np.arange(8, _n, 16) - 0.5, [f'core {_p}' for _p in range(mc.K)])
    _b.set_yticks(np.arange(8, _n, 16) - 0.5, [f'core {_p}' for _p in range(mc.K)])
    _b.set_title('|J| reordered core by core', fontsize=10)
    fig_strong.colorbar(_im, ax=_b, ticks=range(8), fraction=0.046, label='|J| code')
    for _ax in (_a, _b):
        _ax.tick_params(length=0)
        for _s in _ax.spines.values():
            _s.set_visible(False)

    fig_strong.tight_layout()
    fig_strong
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Far unit
    """)
    return


@app.cell
def _(Clock, CurrentDAC, Simulator, np, spec):
    from hwmodel.farunit import FarUnit

    class _Probe:
        """
        Stands in for the core: records every F write (time in ns, codes).
        """
        def __init__(self, sim):
            self.sim, self.log = sim, []

        def set_far(self, F):
            self.log.append((self.sim.now // 1000, F.tolist()))

    # A core with 4 lines and 3 far spins (IDs 10, 11, 12). SRAM row = a far spin's couplings to the 4 lines.
    _sram = np.array([[1,  0, 2, 0],      # spin 10
                      [0, -1, 0, 0],      # spin 11
                      [0,  0, 0, 3]])     # spin 12
    _sim = Simulator()
    _probe = _Probe(_sim)
    _far = FarUnit(_sim, _probe, [10, 11, 12], _sram, [1, 1, 1], CurrentDAC(None, spec.node.I_unit),
                   spec.far, Clock('core', spec.clock.f_core), lambda: None)

    _sim.schedule(0, _far.receive, 10)          # spin 10 flips to -1
    _sim.schedule(0, _far.receive, 11)          # spin 11 flips to -1 in the same cycle: waits one cycle (1 lane)
    _sim.schedule(10_000, _far.receive, 12)     # spin 12 flips ...
    _sim.schedule(10_000, _far.receive, 12)     # ... and back before its row is read: they cancel
    _sim.run()
    print(_probe.log)
    print(_far.stats)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mesh
    """)
    return


@app.cell
def _(Clock, CurrentDAC, Mapping, Simulator, np, replace, spec):
    from hwmodel.flipenc import SendPort
    from hwmodel.network import Mesh

    # 20 spins in 5 groups of 4 (couplings 1 inside a group) -> 5 cores of 4. Mesh 4 x 4 tiles, chips of 2 x 2 tiles.
    _J = np.kron(np.eye(5), np.ones((4, 4))) - np.eye(20)
    for _a, _b in [(0, 4), (0, 12), (0, 16), (1, 7), (2, 5), (3, 6)]:   # weak far couplings
        _J[_a, _b] = _J[_b, _a] = 0.5
    _spec = replace(spec, core=replace(spec.core, c=4), net=replace(spec.net, chip=2))
    _m = Mapping(_J, np.zeros(20), _spec, CurrentDAC(3, 1), CurrentDAC(6, 1))
    print('spins per core:', [list(_s) for _s in _m.spins])
    print('tile (row, col) per core:', [divmod(int(_t), 4) for _t in _m.tile])
    print('from core 0: hops', _m.hops[0], 'chip edges', _m.chips[0])
    print('fan-out of spins 0, 1, 2, 3, 5, 12:', [list(_m.fanout[_k]) for _k in (0, 1, 2, 3, 5, 12)])

    class _Rx:
        """
        Stands in for a far unit: logs (ns, core, spin) of every packet it receives.
        """
        def __init__(self, sim, q, log):
            self.sim, self.q, self.log = sim, q, log

        def receive(self, k):
            self.log.append((self.sim.now / 1000, self.q, int(k)))

    _sim = Simulator()
    _clk = Clock('core', spec.clock.f_core)
    _log = []
    _mesh = Mesh(_sim, _clk, _spec.net, _m, np.random.default_rng(0))
    _mesh.far = [_Rx(_sim, _q, _log) for _q in range(_m.K)]
    _port0, _port1, _port3 = (SendPort(_sim, _clk, spec.clock.sync, _spec.net, _mesh, _m.spins[_p]) for _p in (0, 1, 3))

    _sim.schedule(500, _port0.flip, 0)          # spin 0 flips
    _sim.schedule(2_500, _port0.flip, 1)        # spin 1 flips ...
    _sim.schedule(2_700, _port0.flip, 1)        # ... and back in the same cycle: no packet
    _sim.schedule(3_100, _port0.flip, 2)        # spins 2 and 3 in the same cycle: they share the link
    _sim.schedule(3_200, _port0.flip, 3)
    _sim.schedule(20_500, _port3.flip, 0)       # spin 12 (local 0 of core 3, 2 hops from core 0) ...
    _sim.schedule(22_500, _port1.flip, 1)       # ... and spin 5 (local 1 of core 1, 1 hop) both reach core 0 at 28 ns
    _sim.run()
    print(_log)
    print(_port0.stats)
    print(_mesh.stats)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Machine
    """)
    return


@app.cell
def _(analog_event, np, replace, spec):
    from hwmodel.machine import Machine

    # Test A: with every delay 0 the machine must do what analog_event does. 160 random spins -> 10 cores of 16.
    _zero = replace(spec, core=replace(spec.core, c=16), clock=replace(spec.clock, sync=0),
                    net=replace(spec.net, t_inj=0, t_hop=0, t_chip=0, t_link=0, t_rx=0), far=replace(spec.far, t_pipe=0))
    _rng = np.random.default_rng(42)
    _n = 160
    _J = np.triu(_rng.integers(-7, 8, (_n, _n)) * (_rng.random((_n, _n)) < 0.05), 1)
    _J = _J + _J.T
    _same, _flips, _ended = 0, 0, 0
    for _seed in range(40):
        _v0 = np.random.default_rng(_seed).uniform(-1, 1, _n)
        _mach = Machine(_J, np.zeros(_n), _zero, _v0, np.random.default_rng(_seed))
        _ended += _mach.sim.run(max_events=1_000_000) == 'empty'                 # 'max_events' = stuck at one ps
        _s_ref, _, _n_ref = analog_event(_mach.map.J.astype(float), _mach.map.h.astype(float), _v0.copy(), 1_000_000)
        _same += np.array_equal(_mach.s, _s_ref)
        _flips += sum(len(_c.flips) for _c in _mach.cores) == _n_ref
    _mach.map.K, _ended, _same, _flips
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
