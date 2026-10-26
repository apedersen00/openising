import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from hwmodel.timebase import ps, Clock
    from hwmodel.kernel import Simulator, ANALOG, EDGE, CONTROL

    return ANALOG, CONTROL, Clock, EDGE, Simulator, ps


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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
