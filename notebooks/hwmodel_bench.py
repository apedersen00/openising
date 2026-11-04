import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import sys
    from dataclasses import replace
    from pathlib import Path
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from hwmodel.controller import Controller
    from hwmodel.spec import MachineSpec
    from hwmodel.trace import wrap


    def _bench(name='G14', best_known=3064, n_runs=300, n_seeds=5):
        lines = (Path(__file__).parent.parent / f'ising/benchmarks/G/{name}.txt').read_text().split('\n')
        n = int(lines[0].split()[0])
        J = np.zeros((n, n))                                                # Max-Cut: J = -W
        for line in lines[1:]:
            if line.strip():
                i, j, w = line.split()
                J[int(i) - 1, int(j) - 1] = J[int(j) - 1, int(i) - 1] = -float(w)

        s0 = MachineSpec()
        for label, c, mesh, dither in (('crossbar', n, 1, 0), ('4 tiles', n // 4, 2, 4)):
            best, t98, t99, total = [], [], [], []
            for seed in range(n_seeds):
                spec = replace(s0, core=replace(s0.core, c=c), net=replace(s0.net, mesh=mesh, chip=mesh, dither=dither),
                               control=replace(s0.control, n_runs=n_runs, seed=seed))
                ctl = Controller(J, np.zeros(n), spec)
                ends = []                                                   # hardware time at the end of every run
                wrap(ctl, 'run', after=lambda: ends.append(ctl.sim.now))
                ctl.solve()
                t = np.array(ends) / 1e6                                    # us
                cuts = np.array([0.25 * (s @ J @ s - J.sum()) for s in ctl.end])
                best.append(cuts.max())
                total.append(ctl.sim.now / 1e6)
                t98.append(t[np.argmax(cuts >= 0.98 * best_known)] if (cuts >= 0.98 * best_known).any() else np.nan)
                t99.append(t[np.argmax(cuts >= 0.99 * best_known)] if (cuts >= 0.99 * best_known).any() else np.nan)
            print(f'{label:9s} best cuts {[int(b) for b in best]}, mean gap {100 * (1 - np.mean(best) / best_known):.2f} %, '
                  f'time to 98 %: {np.nanmedian(t98):.1f} us ({np.sum(~np.isnan(t98))}/{n_seeds}), '
                  f'time to 99 %: {np.nanmedian(t99):.1f} us ({np.sum(~np.isnan(t99))}/{n_seeds}), '
                  f'whole solve {np.mean(total):.1f} us')


    _bench()
    return MachineSpec, Path, np, replace


@app.cell
def _(MachineSpec, Path, np, replace):
    import matplotlib.pyplot as _plt
    from hwmodel.dac import CurrentDAC as _CurrentDAC
    from hwmodel.mapping import Mapping as _Mapping


    def _show(name='G14'):
        lines = (Path(__file__).parent.parent / f'ising/benchmarks/G/{name}.txt').read_text().split('\n')
        n = int(lines[0].split()[0])
        J = np.zeros((n, n))                                                # Max-Cut: J = -W
        for line in lines[1:]:
            if line.strip():
                i, j, w = line.split()
                J[int(i) - 1, int(j) - 1] = J[int(j) - 1, int(i) - 1] = -float(w)

        s0 = MachineSpec()
        spec = replace(s0, core=replace(s0.core, c=n // 4), net=replace(s0.net, mesh=2, chip=2))
        m = _Mapping(J, np.zeros(n), spec, _CurrentDAC(3, 1), _CurrentDAC(6, 1))

        # the numbers
        deg = np.count_nonzero(J, axis=1)
        print(f'{name}: {n} spins, {np.count_nonzero(J) // 2} couplings, {100 * np.count_nonzero(J) / (n * (n - 1)):.2f} % of all pairs, '
              f'degree mean {deg.mean():.1f}, min {deg.min()}, max {deg.max()}')
        print(f'J values {np.unique(J[J != 0])}, machine codes {np.unique(m.J[m.J != 0])}, lsb {m.lsb:.3f}')
        inside = sum(np.count_nonzero(l) for l in m.local) / np.count_nonzero(m.J)
        print(f'{m.K} tiles of {spec.core.c} spins: {100 * inside:.0f} % of couplings inside a tile')
        for p in range(m.K):
            print(f'  tile {p}: {np.count_nonzero(m.local[p]) // 2} couplings inside, listens to {len(m.rows[p])} far spins, '
                  f'SRAM {m.sram[p].size * 4 / 1000:.0f} kbit')

        # the picture: the same couplings, as given and grouped by tile
        order = np.concatenate(m.spins)
        A = m.J[np.ix_(order, order)] != 0
        same = np.equal.outer(m.core[order], m.core[order])
        edges = np.arange(1, m.K) * spec.core.c
        centres = (np.arange(m.K) + 0.5) * spec.core.c

        fig, (a, b) = _plt.subplots(1, 2, figsize=(11, 5.8), facecolor='#fcfcfb')
        a.spy(J, markersize=0.7, color='#0b0b0b')
        a.set_title('J as given: one crossbar of 800', loc='left', fontsize=11, pad=22)
        a.set_xticks([0, 200, 400, 600, 799])
        a.set_yticks([0, 200, 400, 600, 799])
        a.set_xlabel('spin')
        a.set_ylabel('spin')
        b.spy(A & same, markersize=0.7, color='#2a78d6', label='inside a tile: crossbar cell')
        b.spy(A & ~same, markersize=0.7, color='#eb6834', label='between tiles: far unit SRAM')
        for e in edges:
            b.axhline(e - 0.5, color='#52514e', lw=1)
            b.axvline(e - 0.5, color='#52514e', lw=1)
        b.set_title('The same J, grouped into 4 tiles of 200', loc='left', fontsize=11, pad=22)
        b.set_xticks(centres, [f'tile {p}' for p in range(m.K)])
        b.set_yticks(centres, [f'tile {p}' for p in range(m.K)])
        b.tick_params(length=0)
        b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False, markerscale=6, fontsize=9)
        for ax in (a, b):
            ax.set_facecolor('#fcfcfb')
            ax.tick_params(labelsize=9)
        fig.tight_layout()
        return fig


    _show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
