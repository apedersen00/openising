"""
animate.py

Animation of a traced machine: spins ramping and flipping in every core, packets crossing the mesh,
packets waiting at the far units, F writes, kicks. Frames rebuilds the machine at every clock cycle;
render() turns that into a video (busy cycles in slow motion, quiet stretches fast forward) and
player() into one self-contained HTML page (hwmodel/player_template.html filled with the data).

    python -m hwmodel.animate      writes hwmodel/figures/mesh_lattice.{gif,mp4}, mesh_3regular.{gif,mp4}
                                   and mesh_player.html

*ai-generated*
"""

import io
import json
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from PIL import Image

UP, DOWN, INK, BG, GRID, WRITE, HOLD = '#2a78d6', '#eb6834', '#0b0b0b', '#fcfcfb', '#c3c2b7', '#1baf7a', '#e6e5e0'
PITCH, CHIP_GAP = 1.6, 0.7                      # tile pitch and the extra gap between chips (tile = 1 x 1)
PHASES = {'run': 'run {}: free running', 'end': 'run over (quiet)', 'hold': 'kick: cores frozen',
          'preset': 'kick: new state on the rails, far units catching up', 'release': 'released'}


def layout(mesh, chip):
    """
    Lower-left corner of every tile and the position of its router, on a mesh x mesh grid.
    Row 0 is at the top.
    """
    r, q = np.divmod(np.arange(mesh * mesh), mesh)
    x = q * PITCH + q // chip * CHIP_GAP
    y = -(r * PITCH + r // chip * CHIP_GAP)
    return np.c_[x, y], np.c_[x + 1.3, y - 0.3]


def route(a, b, mesh):
    """
    Tiles on the XY route from tile a to tile b: first along the row, then along the column.
    """
    (ra, qa), (rb, qb) = divmod(a, mesh), divmod(b, mesh)
    path = [a]
    while qa != qb:
        qa += np.sign(qb - qa)
        path.append(ra * mesh + qa)
    while ra != rb:
        ra += np.sign(rb - ra)
        path.append(ra * mesh + qa)
    return path


def chip_edge(a, b, net):
    """
    True if the link between tiles a and b crosses a chip edge.
    """
    (ra, qa), (rb, qb) = divmod(a, net.mesh), divmod(b, net.mesh)
    return ra // net.chip != rb // net.chip or qa // net.chip != qb // net.chip


def copies(trace, net, period):
    """
    One entry per packet copy: (times, tiles, t_deliver, q). The copy is at the router of tiles[i] at times[i]
    and moves in a straight line between them; it waits at the far unit of core q until t_deliver.
    """
    m = trace.map
    tile = m.tile
    sent = defaultdict(deque)                                   # (spin, q) -> send times, oldest first
    for t, k, p in trace.packets:
        for q in m.fanout[k]:
            sent[k, q].append(t)
    delivered = defaultdict(deque)                              # q -> delivery times, in arrival order
    for t, q, k in trace.deliveries:
        delivered[q].append(t)
    out = []
    for t_arrive, q, k in trace.arrivals:
        t_send, p = sent[k, q].popleft(), m.core[k]
        path = route(tile[p], tile[q], net.mesh)
        steps = [net.t_hop + net.t_chip * chip_edge(a, b, net) for a, b in zip(path, path[1:])]   # cycles per hop
        t0 = t_arrive - sum(steps) * period                     # leaves the source router (after t_inj and dither)
        times = [t_send, t0] + list(t0 + np.cumsum(steps) * period)
        out.append((np.array(times), [path[0]] + path, delivered[q].popleft(), q))
    return out


def energy(trace, J, h):
    """
    (times, E) after every comparator change.
    """
    s = trace.s0.copy()
    T, E = [0], [-0.5 * s @ J @ s - h @ s]
    for t, p, k in trace.flips:
        s[trace.map.spins[p][k]] *= -1
        T.append(t)
        E.append(-0.5 * s @ J @ s - h @ s)
    return np.array(T), np.array(E)


class Frames:
    """
    The machine at every clock cycle, rebuilt from a trace. The geometry is in tile units (a tile is 1 x 1).
    """
    def __init__(self, trace, net, period):
        m = trace.map
        self.trace, self.net, self.period, self.map = trace, net, period, m
        self.c = len(m.spins[0])
        side = int(np.ceil(np.sqrt(self.c)))
        self.corner, self.router = layout(net.mesh, net.chip)
        self.pos = self.corner[m.tile]                          # lower-left corner of each core's tile
        gx, gy = np.divmod(np.arange(self.c), side)
        self.spin_xy = np.concatenate([self.pos[p] + np.c_[0.12 + gy * 0.76 / (side - 1), 0.88 - gx * 0.76 / (side - 1)]
                                       for p in range(m.K)])    # spin k of core p is entry p * c + k
        self.chips = []                                         # (x0, y0, x1, y1, number)
        n = net.mesh // net.chip
        for a in range(n):
            for b in range(n):
                tiles = [(a * net.chip + i) * net.mesh + b * net.chip + j for i in range(net.chip) for j in range(net.chip)]
                self.chips.append((self.corner[tiles, 0].min() - 0.2, self.router[tiles, 1].min() - 0.2,
                                   self.router[tiles, 0].max() + 0.2, self.corner[tiles, 1].max() + 1.2, a * n + b))
        self.links = [(t, t2, chip_edge(t, t2, net)) for t in range(net.mesh ** 2)
                      for t2 in ([t + 1] if (t + 1) % net.mesh else []) + ([t + net.mesh] if t + net.mesh < net.mesh ** 2 else [])]
        self.copies = copies(trace, net, period)
        self.T_E, self.E = energy(trace, m.J, m.h)
        self.flip_t = np.array([t for t, _, _ in trace.flips])
        self.flip_i = np.array([p * self.c + k for _, p, k in trace.flips])
        self.packet_t = np.array([t for t, _, _ in trace.packets])
        self.write_t = np.array([t for t, _ in trace.writes])
        self.write_q = np.array([q for _, q in trace.writes])
        self.half = trace.node.V_high - trace.node.V_th

    def at(self, i):
        """
        The machine at cycle i: sigma and distance from the threshold (0 at V_th, 1 at a rail) of every spin,
        the spins that flipped in the last 3 cycles, packets on their way and waiting, cores whose F was written
        in the last 2 cycles, whether the cores are frozen, the phase and the counters.
        """
        t, tr, m = i * self.period, self.trace, self.map
        V = np.concatenate([tr.voltages(p, t) for p in range(m.K)])
        moving, waiting, queue = [], [], defaultdict(int)
        for times, path, t_deliver, q in self.copies:
            if t < times[0] or t >= t_deliver:
                continue
            if t >= times[-1]:                                  # at the far unit's input, waiting
                waiting.append(self.router[m.tile[q]] + [-0.12 - 0.11 * queue[q], 0.18])
                queue[q] += 1
                continue
            j = max(np.searchsorted(times, t, side='right') - 1, 0)
            f = (t - times[j]) / (times[j + 1] - times[j]) if times[j + 1] > times[j] else 0
            a, b = self.router[path[j]], self.router[path[min(j + 1, len(path) - 1)]]
            moving.append(a + f * (b - a))
        now = [name for tp, name in tr.phases if tp <= t]
        return SimpleNamespace(
            t=t,
            s=np.where(V >= tr.node.V_th, 1, -1),
            d=np.clip(np.abs(V - tr.node.V_th) / self.half, 0, 1),
            recent=self.flip_i[(self.flip_t > t - 3 * self.period) & (self.flip_t <= t)],
            moving=np.array(moving).reshape(-1, 2),
            waiting=np.array(waiting).reshape(-1, 2),
            written=set(self.write_q[(self.write_t > t - 2 * self.period) & (self.write_t <= t)].tolist()),
            frozen=bool(now) and now[-1] in ('hold', 'preset'),
            phase=PHASES[now[-1]].format(sum(name == 'run' for name in now)) if now else '',
            counts=(int(np.searchsorted(self.flip_t, t, 'right')), int(np.searchsorted(self.packet_t, t, 'right')),
                    len(moving), len(waiting), int(np.searchsorted(self.write_t, t, 'right'))))

    def busy(self, n):
        """
        Busy cycles 0..n: a packet is on its way or waiting, a comparator flipped or F was written
        in the last 2 cycles, or the cores are frozen.
        """
        period = self.period
        busy = np.zeros(n + 1, dtype=bool)
        for times, _, t_deliver, _ in self.copies:
            busy[int(times[0] // period):int(t_deliver // period) + 1] = True
        for t in np.r_[self.flip_t, self.write_t]:
            busy[int(t // period):int(t // period) + 3] = True
        events = self.trace.phases + [(n * period + 1, 'end')]
        for (t, name), (t_next, _) in zip(events, events[1:]):
            if name in ('hold', 'preset'):
                busy[int(t // period):int(t_next // period) + 1] = True
        return busy


def render(frames, stem, t_end, title, fps=24, slow=6, skip=4):
    """
    Writes stem.mp4 and stem.gif: the machine from 0 to t_end (ps). Busy cycles play at `slow` cycles per second;
    in quiet stretches every `skip`-th cycle is shown for one video frame.
    """
    fr, m, period = frames, frames.map, frames.period
    corner, router, pos = fr.corner, fr.router, fr.pos

    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    ax = fig.add_axes([0.01, 0.02, 0.62, 0.84])
    ax.set_facecolor(BG)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(corner[:, 0].min() - 0.4, router[:, 0].max() + 0.4)
    ax.set_ylim(router[:, 1].min() - 0.4, corner[:, 1].max() + 1.6)

    # chips, links, tiles, routers
    for x0, y0, x1, y1, number in fr.chips:
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0,rounding_size=0.15',
                                    fc='none', ec=GRID, lw=2))
        ax.text(x0 + 0.1, y1 + 0.05, f'chip {number}', fontsize=10, color='#52514e', va='bottom', ha='left',
                fontweight='bold')
    for a, b, cross in fr.links:
        ax.plot(*zip(router[a], router[b]), color=GRID, lw=2.5 if cross else 1.2, ls='--' if cross else '-', zorder=1)
    used = set(m.tile)
    tile_box = {}
    for t in range(len(corner)):
        box = FancyBboxPatch(corner[t], 1, 1, boxstyle='round,pad=0,rounding_size=0.06',
                             fc='white' if t in used else BG, ec=GRID, lw=1.2, zorder=2)
        ax.add_patch(box)
        tile_box[t] = box
    for p in range(m.K):
        ax.text(*(pos[p] + [1.0, 1.03]), f'core {p}', fontsize=7, color='#52514e', ha='right', va='bottom')
    ax.scatter(router[:, 0], router[:, 1], s=30, c='white', edgecolors='#52514e', linewidths=1, zorder=3)
    spins = ax.scatter(fr.spin_xy[:, 0], fr.spin_xy[:, 1], s=46, marker='s', linewidths=1.4, zorder=4)
    dots = ax.scatter([], [], s=16, c=INK, zorder=6)
    waits = ax.scatter([], [], s=14, marker='s', c=DOWN, edgecolors=INK, linewidths=0.4, zorder=6)
    clock = fig.text(0.01, 0.915, '', fontsize=15, fontweight='bold', color=INK, va='top', family='monospace')
    phase = fig.text(0.30, 0.912, '', fontsize=12.5, color=INK, va='top')
    speed = fig.text(0.97, 0.965, '', fontsize=10.5, color='#52514e', va='top', ha='right')

    # right: energy and counters
    ex = fig.add_axes([0.68, 0.52, 0.29, 0.33], facecolor=BG)
    ex.plot(fr.T_E / 1000, fr.E, color=INK, lw=1.2, drawstyle='steps-post')
    ex.set_xlim(0, t_end / 1000)
    ex.set_xlabel('time (ns)', fontsize=9)
    ex.set_title('energy  -1/2 s J s', fontsize=10, loc='left')
    ex.tick_params(labelsize=8)
    for sp in ('top', 'right'):
        ex.spines[sp].set_visible(False)
    cursor = ex.axvline(0, color=DOWN, lw=1)
    counters = fig.text(0.68, 0.43, '', fontsize=10.5, va='top', family='monospace', color=INK)
    fig.legend(handles=[Line2D([], [], marker='s', ls='', mfc=UP, mec=UP, ms=8, label='spin +1'),
                        Line2D([], [], marker='s', ls='', mfc=DOWN, mec=DOWN, ms=8, label='spin -1 (pale: near the threshold)'),
                        Line2D([], [], marker='s', ls='', mfc='white', mec=INK, ms=8, mew=1.4, label='comparator just flipped'),
                        Line2D([], [], marker='o', ls='', mfc=INK, mec=INK, ms=5, label='packet (a spin ID) on its way'),
                        Line2D([], [], marker='s', ls='', mfc=DOWN, mec=INK, ms=5, label='packet waiting at a far unit (1 enters per cycle)'),
                        Patch(fc='white', ec=WRITE, lw=2.4, label='new F codes written into the core'),
                        Patch(fc=HOLD, ec=GRID, label='core frozen (kick)'),
                        Line2D([], [], color=GRID, lw=2.5, ls='--', label='chip-to-chip link (+t_chip cycles)')],
               loc='lower left', bbox_to_anchor=(0.665, 0.03), frameon=False, fontsize=9, labelcolor='#52514e')
    fig.text(0.01, 0.97, title, fontsize=13, fontweight='bold', color=INK, va='top')

    up, down, white = np.array(to_rgb(UP)), np.array(to_rgb(DOWN)), np.ones(3)

    def frame(i):
        st = fr.at(i)
        d = st.d[:, None] * 0.85 + 0.15
        rgb = np.where(st.s[:, None] > 0, up, down) * d + white * (1 - d)
        edge = np.c_[rgb, np.ones(len(rgb))]
        edge[st.recent, :3] = to_rgb(INK)
        spins.set_facecolors(np.c_[rgb, np.ones(len(rgb))])
        spins.set_edgecolors(edge)
        dots.set_offsets(st.moving)
        waits.set_offsets(st.waiting)
        for p in range(m.K):
            box = tile_box[m.tile[p]]
            box.set_edgecolor(WRITE if p in st.written else GRID)
            box.set_linewidth(2.6 if p in st.written else 1.2)
            box.set_facecolor(HOLD if st.frozen else 'white')
        phase.set_text(st.phase)
        clock.set_text(f'cycle {i:4d}   ({st.t / 1000:.0f} ns)')
        cursor.set_xdata([st.t / 1000])
        counters.set_text('comparator flips  {:5d}\npackets sent      {:5d}\ncopies in flight  {:5d}\n'
                          'waiting at far    {:5d}\nF writes          {:5d}'.format(*st.counts))

    n = int(t_end // period)
    busy = fr.busy(n)
    repeat = max(1, round(fps / slow))
    stem = Path(stem)
    writer = FFMpegWriter(fps=fps, bitrate=2500)
    images, durations = [], []
    with writer.saving(fig, stem.with_suffix('.mp4'), dpi=100):
        for i in range(n + 1):
            if not busy[i] and i % skip:
                continue
            frame(i)
            speed.set_text(f'slow motion: {slow} cycles/s' if busy[i] else f'fast forward: {fps * skip} cycles/s')
            for _ in range(repeat if busy[i] else 1):
                writer.grab_frame()
            buf = io.BytesIO()
            fig.savefig(buf, dpi=60, format='png', facecolor=BG)
            images.append(Image.open(buf).convert('P', palette=Image.ADAPTIVE, colors=128))
            durations.append(round(1000 * (repeat if busy[i] else 1) / fps))
    images[0].save(stem.with_suffix('.gif'), save_all=True, append_images=images[1:], duration=durations, loop=0)
    plt.close(fig)


def player(scenes, path):
    """
    Writes the interactive player: hwmodel/player_template.html with the data of every scene filled in.
    The page rebuilds the machine at any moment from timelines, so it can play smoothly between clock cycles:
    the node voltages and slopes of every core after each change, every packet copy, flips, F writes and phases.
    scenes: [(name, title, frames, t_end)]. Units: ps, hundredths of a tile, 10 uV, uV/ns.
    """
    xy = lambda a: np.round(np.asarray(a) * 100).astype(int).ravel().tolist()
    uv10 = lambda v: np.round(np.asarray(v) * 1e5).astype(int).ravel().tolist()
    data = []
    for name, title, fr, t_end in scenes:
        tr, m, node = fr.trace, fr.map, fr.trace.node
        n = int(t_end // fr.period)
        data.append(dict(
            name=name, title=title, n=n + 1, period=fr.period, c=fr.c,
            sub=f'{m.K * fr.c} spins in {m.K} cores of {fr.c} on {len(fr.chips)} chips · '
                f'one clock cycle = {fr.period / 1000:g} ns',
            volts=uv10([node.V_low, node.V_th, node.V_high]),
            tiles=[xy(c) for c in fr.corner],
            routers=[xy(r) for r in fr.router],
            links=[[a, b, int(cross)] for a, b, cross in fr.links],
            chips=[xy(c[:4]) + [c[4]] for c in fr.chips],
            place=[int(t) for t in m.tile],                     # tile of every core
            spins=[xy(s) for s in fr.spin_xy],
            nodes=[dict(t=[int(e[0]) for e in E], held=''.join('1' if e[3] else '0' for e in E),
                        v=uv10([e[1] for e in E]), s=np.round(np.array([e[2] for e in E]) / node.C * 1e-3).astype(int).ravel().tolist())
                   for E in tr.cores],
            copies=[[[int(x) for x in times], [int(x) for x in path], int(t_deliver), int(q)]
                    for times, path, t_deliver, q in fr.copies],
            flips=[[int(t), p, int(k)] for t, p, k in tr.flips],
            packets=[int(t) for t, _, _ in tr.packets],
            writes=[[int(t), q] for t, q in tr.writes],
            phases=[[int(t), name] for t, name in tr.phases],
            energy=dict(t=fr.T_E.astype(int).tolist(), e=fr.E.tolist()),
            busy=''.join('1' if b else '0' for b in fr.busy(n))))
    html = (Path(__file__).parent / 'player_template.html').read_text()
    Path(path).write_text(html.replace('/*DATA*/null', json.dumps(data, separators=(',', ':'))))


if __name__ == '__main__':
    import networkx as nx
    from dataclasses import replace
    from hwmodel.spec import MachineSpec
    from hwmodel.controller import Controller
    from hwmodel.trace import Trace

    matplotlib.use('Agg')
    out = Path(__file__).parent / 'figures'
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    L = 16
    J_lat = np.zeros((L * L, L * L))
    for r in range(L):
        for q in range(L):
            for r2, q2 in ((r, q + 1), (r + 1, q)):
                if r2 < L and q2 < L:
                    J_lat[r * L + q, r2 * L + q2] = J_lat[r2 * L + q2, r * L + q] = rng.choice([-1, 1])
    J_reg = -nx.to_numpy_array(nx.random_regular_graph(3, 256, seed=0))

    spec = MachineSpec()
    spec = replace(spec, core=replace(spec.core, c=16), net=replace(spec.net, mesh=4, chip=2),
                   control=replace(spec.control, n_runs=2))
    scenes = []
    for name, J, label in (('lattice', J_lat, '16 × 16 spin-glass lattice'), ('3regular', J_reg, 'Random 3-regular max-cut')):
        ctl = Controller(J, np.zeros(len(J)), spec)
        trace = Trace(ctl)
        ctl.solve()
        t_end = [t for t, n in trace.phases if n == 'release'][0] + 100_000
        frames = Frames(trace, spec.net, ctl.machine.clock.period)
        render(frames, out / f'mesh_{name}', t_end, f'{label}: 256 spins on 4 chips × 4 cores of 16, 1 GHz')
        scenes.append((name, label, frames, t_end))
        print(name, 'done', t_end / 1000, 'ns')
    player(scenes, out / 'mesh_player.html')
