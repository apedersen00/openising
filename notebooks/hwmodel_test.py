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
    return (NodeSpec,)


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
    return (np,)


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
    return I, dt, node


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
def _():
    return


if __name__ == "__main__":
    app.run()
