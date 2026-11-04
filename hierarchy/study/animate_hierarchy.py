"""
Animated HTML page of one run of the hierarchical solver (hierarchy_solver.HierarchySolver), step by step, against
one core solving the whole problem.

    python animate_hierarchy.py                                    G32, 4 KL cores, 4 nodes per metanode, 4 sweeps
    python animate_hierarchy.py dummy --cores 2 --nodes-per-metanode 3 --iterations 20
    python animate_hierarchy.py G/regenerated/G40_2000_11766_planar_2400 --sweeps 8 --iterations 200

Steps: sweep 0 "lower" (every core alone), then for every sweep "vote", "upper" and "lower". For every step:
- energy of the problem at the lower state, and at the view of the upper level (every node at the majority of its
  metanodes) after the vote and after the upper solve
- iterations spent so far: summed over all the solves, or with the cores of a stage counted once (run at the same time)
- after a lower solve, per core: the share of the core energy that comes from the field of the rest,
  |field| / (|internal| + |field|), with internal = -1/2 s J_core s - h_core s and field = -f s
- after the upper solve, per core: the field error, the field of the expanded spins (what the core will see)
  against the field of the real spins of the rest, on the nodes with a neighbor outside the core
The single core solves the whole problem with as many iterations as the hierarchy spent in total (one seed).
"""
# ruff: noqa: E501  (the lines of the HTML template at the end of the file)
import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "ising_hierarchy"))
from hierarchy_solver import HierarchySolver  # noqa: E402
from ising_solver import solve as solve_one_core  # noqa: E402
from metanode import expand  # noqa: E402
from partition import partition_greedy, partition_kl, partition_random  # noqa: E402
from problems import hand_dummy, load_maxcut  # noqa: E402

PARTITIONS = {"kl": partition_kl, "greedy": partition_greedy, "random": partition_random}


def load_problem(name):
    """J, h and a short name: the hand dummy, or a Max-Cut benchmark."""
    if name == "dummy":
        J, h = hand_dummy()
        return J, h, "dummy"
    J, h = load_maxcut(name)
    return J, h, Path(name).name.split("_")[0]


def boundary(core):
    """Nodes of the core with at least one neighbor outside it."""
    return np.array([any(core not in neighbor.in_cores for neighbor, _ in node.neighbors) for node in core.nodes])


def far_field(core):
    """Share of the energy of the core that comes from the field of the rest, after its solve."""
    s = core.best_spins
    internal = float(-0.5 * s @ core.J @ s - core.h @ s)
    field = float(-core.f @ s)
    total = abs(internal) + abs(field)
    return {"core": int(core.label), "share": 100 * abs(field) / total if total else 0.0,
            "internal": internal, "field": field}


def field_error(core, mask, seen, state):
    """Field of the spins `seen` against the field of the real spins `state`, on the nodes of `mask`."""
    true = core.field_from_outside(state).copy()
    view = core.field_from_outside(seen).copy()
    if not mask.any():
        return {"core": int(core.label), "boundary": 0, "wrong": 0.0, "mae": 0.0}
    return {"core": int(core.label), "boundary": int(mask.sum()),
            "wrong": float(100 * np.mean(np.sign(view[mask]) != np.sign(true[mask]))),
            "mae": float(np.abs(view - true)[mask].mean())}


def record_run(solver, nb_sweeps, seed, solve_args):
    """Run the solver and return one dict per step."""
    per_solve = solve_args["nb_iterations"] * solve_args["nb_seeds"]
    masks = [boundary(core) for core in solver.cores]
    spent = {"sum": 0, "parallel": 0}
    steps = []

    def on_step(s, sweep, step):
        if step == "lower":
            spent["sum"] += per_solve * len(s.cores)
            spent["parallel"] += per_solve
        elif step == "upper":
            spent["sum"] += per_solve
            spent["parallel"] += per_solve
        row = {"sweep": sweep, "step": step, "iter_sum": spent["sum"], "iter_parallel": spent["parallel"],
               "energy": float(s.energy(s.lower_state)), "view": None, "upper_energy": None,
               "far_field": None, "field_error": None}
        if step in ("vote", "upper"):
            row["view"] = float(s.energy(expand(s.nodes, s.upper_state, s.lower_state)))
        if step == "upper":
            row["upper_energy"] = float(s.upper_core.best_energy)
            row["field_error"] = [field_error(c, m, s.expanded, s.lower_state) for c, m in zip(s.cores, masks)]
        if step == "lower":
            row["far_field"] = [far_field(c) for c in s.cores]
        steps.append(row)

    solver.solve(nb_sweeps, seed=seed, on_step=on_step, **solve_args)
    return steps


def single_core_curve(J, h, nb_iterations, seed, max_time):
    """Best energy of one core on the whole problem after every iteration, kept where it changes."""
    J = np.ascontiguousarray(J, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    _, _, history = solve_one_core(J, h, nb_iterations, 0.9, 0.1, 42 + 100_000 * seed, None, 0.0, max_time)
    best = history[:, 1]
    keep = np.r_[True, best[1:] != best[:-1]]
    keep[-1] = True
    return {"x": (np.flatnonzero(keep) + 1).tolist(), "y": best[keep].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("benchmark", nargs="?", default="G/regenerated/G32_2000_4000_toroidal_1410",
                        help='"dummy" or a Max-Cut benchmark of ising/benchmarks')
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--partition", choices=PARTITIONS, default="kl")
    parser.add_argument("--nodes-per-metanode", type=int, default=4)
    parser.add_argument("--method", choices=["linkage", "original"], default="linkage")
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100, help="iterations of every solve")
    parser.add_argument("--seeds", type=int, default=1, help="seeds of every solve")
    parser.add_argument("--max-time", type=float, default=np.inf, help="time budget of every run")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--out", type=Path, help="html file (default: study/animations/<problem>_....html)")
    args = parser.parse_args()

    J, h, short = load_problem(args.benchmark)
    labels = PARTITIONS[args.partition](J, args.cores, seed=args.trial)
    solver = HierarchySolver(J, h, labels, args.nodes_per_metanode, args.method)
    solve_args = {"nb_iterations": args.iterations, "nb_seeds": args.seeds, "max_time": args.max_time}
    steps = record_run(solver, args.sweeps, args.trial, solve_args)
    single = single_core_curve(J, h, steps[-1]["iter_sum"], args.trial, args.max_time)

    sizes = ", ".join(str(len(core.nodes)) for core in solver.cores)
    data = {
        "title": f"Hierarchy on {short}",
        "subtitle": (f"{len(h)} spins · {len(solver.cores)} {args.partition} cores of {sizes} spins · "
                     f"{len(solver.metanodes)} metanodes of {args.nodes_per_metanode} ({args.method}) in one upper "
                     f"core · {args.sweeps} sweeps · every solve {args.iterations} iterations × {args.seeds} seeds · "
                     f"trial {args.trial}"),
        "cut_offset": float(J.sum() / 2),  # Max-Cut: cut = -(E + sum(J) / 2)
        "steps": steps,
        "single": single,
    }
    out = args.out or Path(__file__).parent / "animations" / (
        f"{short}_{args.cores}cores_{args.nodes_per_metanode}pm_{args.sweeps}sweeps_trial{args.trial}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    page = TEMPLATE.replace("__TITLE__", html.escape(data["title"]))
    page = page.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}")
    print(f"final energy {steps[-1]['energy']:.1f}, single core {single['y'][-1]:.1f} "
          f"after {steps[-1]['iter_sum']} iterations")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --div-better: #2a78d6; --div-worse: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    --div-better: #3987e5; --div-worse: #e66767;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  --div-better: #3987e5; --div-worse: #e66767;
}
html, body { margin: 0; }
body.viz-root { background: var(--page); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1200px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 16px; }
.controls { display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; margin-bottom: 16px; }
.controls button { font: inherit; padding: 6px 14px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--surface-1); color: var(--text-primary); cursor: pointer; min-width: 72px; }
.controls input[type=range] { width: min(380px, 100%); }
.controls label, .controls .step { color: var(--text-secondary); }
.controls .step { min-width: 190px; color: var(--text-primary); font-weight: 600; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile, .card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; }
.tile { padding: 10px 14px; }
.tile .label { color: var(--text-secondary); font-size: 12px; }
.tile .value { font-size: 22px; font-weight: 600; }
.tile .note { color: var(--text-muted); font-size: 12px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 520px), 1fr)); gap: 16px; }
.card { padding: 12px 14px 8px; min-width: 0; }
.card h2 { font-size: 15px; font-weight: 600; margin: 0; }
.card .desc { color: var(--text-secondary); font-size: 12.5px; margin: 2px 0 8px; }
.legend { display: flex; gap: 6px 14px; flex-wrap: wrap; font-size: 12.5px; color: var(--text-secondary); margin-bottom: 4px; }
.key-line { display: inline-block; width: 16px; height: 2px; border-radius: 1px; vertical-align: middle; margin-right: 6px; }
.key-rect { display: inline-block; width: 10px; height: 10px; border-radius: 2px; vertical-align: -1px; margin-right: 6px; }
.plot { position: relative; width: 100%; }
.plot svg { display: block; }
.plot svg text { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.plot svg text.direct { fill: var(--text-secondary); font-size: 12px; }
.plot .hit:focus { outline: none; }
.plot .hit:focus-visible { outline: 2px solid var(--text-secondary); }
.tip { position: absolute; pointer-events: none; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 8px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.12);
  white-space: nowrap; z-index: 2; display: none; }
.tip .title { color: var(--text-secondary); margin-bottom: 2px; }
.tip .row { display: flex; align-items: center; gap: 6px; }
.tip strong { font-weight: 600; color: var(--text-primary); }
.tip .name { color: var(--text-secondary); }
.empty { color: var(--text-muted); font-size: 13px; padding: 70px 0; text-align: center; }
details { margin-top: 16px; }
summary { cursor: pointer; color: var(--text-secondary); }
.table-wrap { overflow-x: auto; background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; margin-top: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 5px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--text-secondary); font-weight: 600; }
th:nth-child(-n+2), td:nth-child(-n+2) { text-align: left; }
tr.current td { background: color-mix(in srgb, var(--series-1) 12%, transparent); }
</style>
</head>
<body class="viz-root">
<main>
  <h1 id="title"></h1>
  <p class="sub" id="subtitle"></p>
  <div class="controls">
    <button id="play" type="button">Play</button>
    <input id="slider" type="range" min="0" value="0" step="1" aria-label="Step">
    <span class="step" id="step-name"></span>
    <span>Iterations:
      <label><input type="radio" name="mode" value="sum" checked> summed over all the solves</label>
      <label><input type="radio" name="mode" value="parallel"> cores of a stage at the same time</label>
    </span>
  </div>
  <section class="tiles" id="tiles"></section>
  <section class="grid">
    <div class="card"><h2>Energy of the problem at every step</h2>
      <p class="desc">Lower state: the spins the cores hold. Upper view: every node at the majority of its metanodes, after the vote and after the upper solve.</p>
      <div class="legend" id="legend-energy"></div><div class="plot" id="plot-energy"></div></div>
    <div class="card"><h2>Energy against iterations spent</h2>
      <p class="desc">The hierarchy's lower state, and one core solving the whole problem (an iteration of a core is a run on its own spins only).</p>
      <div class="legend" id="legend-single"></div><div class="plot" id="plot-single"></div></div>
    <div class="card"><h2>Loss against the single core at every step</h2>
      <p class="desc">Energy of the lower state minus the single core's energy after the same number of iterations.</p>
      <div class="legend" id="legend-gap"></div><div class="plot" id="plot-gap"></div></div>
    <div class="card"><h2>Share of each core's energy from the far field</h2>
      <p class="desc" id="desc-far">|field part| / (|internal part| + |field part|) after the last lower solve.</p>
      <div class="plot" id="plot-far"></div></div>
    <div class="card"><h2>Field error on each core's boundary</h2>
      <p class="desc" id="desc-error">Nodes with a neighbor outside the core whose field from the expanded spins has another sign than the field from the real spins.</p>
      <div class="plot" id="plot-error"></div></div>
  </section>
  <details><summary>Table of every step</summary><div class="table-wrap"><table id="table"></table></div></details>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const S = D.steps;
const NS = "http://www.w3.org/2000/svg";
let cur = 0, timer = null, mode = "sum";

function svg(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function token(name) { return getComputedStyle(document.body).getPropertyValue(name).trim(); }
function fmt(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toLocaleString("en-US", {minimumFractionDigits: digits, maximumFractionDigits: digits});
}
function compact(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toLocaleString("en-US", {maximumFractionDigits: 1}) + "M";
  if (a >= 1e3) return (v / 1e3).toLocaleString("en-US", {maximumFractionDigits: 1}) + "K";
  return v.toLocaleString("en-US", {maximumFractionDigits: 2});
}
const cut = (E) => -(E + D.cut_offset);
const stepName = (s) => s.sweep === 0 ? "sweep 0 · cores alone" :
  `sweep ${s.sweep} · ${{vote: "vote", upper: "upper solve", lower: "lower solve"}[s.step]}`;
const iterOf = (s) => mode === "sum" ? s.iter_sum : s.iter_parallel;
function singleAt(x) {
  const xs = D.single.x, ys = D.single.y;
  if (x < xs[0]) return null;
  let lo = 0, hi = xs.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid - 1; }
  return ys[lo];
}
const gapOf = (s) => { const y = singleAt(iterOf(s)); return y === null ? null : s.energy - y; };
function latest(key, upto) { for (let i = upto; i >= 0; i--) if (S[i][key]) return S[i]; return null; }
function niceTicks(lo, hi, n) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
  const step = (r >= 7.5 ? 10 : r >= 3.5 ? 5 : r >= 1.5 ? 2 : 1) * mag;
  const ticks = [];
  for (let t = Math.floor(lo / step) * step; t <= Math.ceil(hi / step) * step + step / 2; t += step) ticks.push(+t.toFixed(10));
  return ticks;
}

function showTip(box, x, y, title, rows) {
  let tip = box.querySelector(".tip");
  if (!tip) { tip = document.createElement("div"); tip.className = "tip"; box.appendChild(tip); }
  tip.textContent = "";
  const t = document.createElement("div"); t.className = "title"; t.textContent = title; tip.appendChild(t);
  for (const r of rows) {
    const row = document.createElement("div"); row.className = "row";
    if (r.color) { const k = document.createElement("span"); k.className = "key-line"; k.style.background = r.color; row.appendChild(k); }
    const v = document.createElement("strong"); v.textContent = r.value; row.appendChild(v);
    const n = document.createElement("span"); n.className = "name"; n.textContent = r.name; row.appendChild(n);
    tip.appendChild(row);
  }
  tip.style.display = "block";
  tip.style.left = Math.max(0, Math.min(x + 14, box.clientWidth - tip.offsetWidth)) + "px";
  tip.style.top = Math.max(0, y - tip.offsetHeight - 8) + "px";
}
function hideTip(box) { const t = box.querySelector(".tip"); if (t) t.style.display = "none"; }

function legend(id, items) {
  const box = document.getElementById(id); box.textContent = "";
  for (const it of items) {
    const span = document.createElement("span");
    const key = document.createElement("span"); key.className = it.rect ? "key-rect" : "key-line"; key.style.background = it.color;
    span.appendChild(key); span.appendChild(document.createTextNode(it.name)); box.appendChild(span);
  }
}

// Line chart: series {name, color, xs, ys (null = gap), dots, opacity, label}, a marker x, and a hover readout.
function lineChart(box, c) {
  box.textContent = "";
  const W = Math.max(280, box.clientWidth), H = 250, m = {l: 62, r: 16, t: 12, b: 30};
  const root = svg("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": c.aria}, box);
  const vals = c.series.flatMap(s => s.ys.filter(v => v !== null));
  if (!vals.length) return;
  const yt = niceTicks(Math.min(...vals), Math.max(...vals), 4), ylo = yt[0], yhi = yt[yt.length - 1];
  const X = x => m.l + (x - c.xMin) / ((c.xMax - c.xMin) || 1) * (W - m.l - m.r);
  const Y = y => m.t + (yhi - y) / ((yhi - ylo) || 1) * (H - m.t - m.b);
  for (const t of yt) {
    svg("line", {x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: token("--grid"), "stroke-width": 1}, root);
    svg("text", {x: m.l - 8, y: Y(t) + 4, "text-anchor": "end"}, root).textContent = compact(t);
  }
  svg("line", {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: token("--axis"), "stroke-width": 1}, root);
  for (const t of c.xTicks) svg("text", {x: X(t.x), y: H - m.b + 18, "text-anchor": "middle"}, root).textContent = t.label;
  svg("line", {x1: X(c.marker), x2: X(c.marker), y1: m.t, y2: H - m.b, stroke: token("--axis"), "stroke-width": 1}, root);
  const ends = [];
  for (const s of c.series) {
    let d = "", pen = false;
    s.xs.forEach((x, i) => {
      if (s.ys[i] === null) { pen = false; return; }
      d += (pen ? "L" : "M") + X(x).toFixed(1) + "," + Y(s.ys[i]).toFixed(1); pen = true;
    });
    svg("path", {d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: s.opacity ?? 1}, root);
    const pts = s.xs.map((x, i) => [x, s.ys[i]]).filter(p => p[1] !== null);
    const dots = s.dots ? pts : (s.label && pts.length ? [pts[pts.length - 1]] : []);
    for (const [x, y] of dots) svg("circle", {cx: X(x), cy: Y(y), r: 4, fill: s.color, stroke: token("--surface-1"), "stroke-width": 2}, root);
    if (s.label && pts.length) ends.push({x: X(pts[pts.length - 1][0]), y: Y(pts[pts.length - 1][1]), text: s.label});
  }
  // direct end labels, only when they do not collide (the legend and the tooltip carry the rest)
  const clear = ends.every((a, i) => ends.every((b, j) => i === j || Math.abs(a.y - b.y) > 16));
  if (clear) for (const e of ends) {
    const right = e.x + 8 + 7 * e.text.length < W;
    svg("text", {x: right ? e.x + 8 : e.x - 8, y: right ? e.y + 4 : e.y - 10, "text-anchor": right ? "start" : "end", class: "direct"}, root).textContent = e.text;
  }
  const cross = svg("line", {y1: m.t, y2: H - m.b, stroke: token("--text-muted"), "stroke-width": 1, visibility: "hidden"}, root);
  const hit = svg("rect", {x: m.l, y: 0, width: W - m.l - m.r, height: H - m.b, fill: "transparent"}, root);
  hit.addEventListener("pointermove", ev => {
    const r = root.getBoundingClientRect();
    const x = c.snap(c.xMin + (ev.clientX - r.left - m.l) / (W - m.l - m.r) * (c.xMax - c.xMin));
    if (x === null) return;
    cross.setAttribute("x1", X(x)); cross.setAttribute("x2", X(x)); cross.setAttribute("visibility", "visible");
    const tip = c.tip(x);
    showTip(box, X(x), ev.clientY - r.top, tip.title, tip.rows);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(box); });
}

function barPath(x, w, y0, y1) {
  const r = Math.min(4, w / 2, Math.abs(y1 - y0));
  if (y1 <= y0) return `M${x},${y0}V${y1 + r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 + r}V${y0}Z`;
  return `M${x},${y0}V${y1 - r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 - r}V${y0}Z`;
}

// Column chart: bars {label, value, color, title, rows}, from a zero baseline, a tooltip per bar.
function columnChart(box, c) {
  box.textContent = "";
  if (!c.bars.length) { const e = document.createElement("div"); e.className = "empty"; e.textContent = c.empty; box.appendChild(e); return; }
  const W = Math.max(280, box.clientWidth), H = 220, m = {l: 62, r: 12, t: 12, b: 30};
  const root = svg("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": c.aria}, box);
  const vals = c.bars.map(b => b.value);
  const yt = niceTicks(Math.min(0, ...vals), Math.max(0, c.yMin ?? 0, ...vals), 4), ylo = yt[0], yhi = yt[yt.length - 1];
  const Y = y => m.t + (yhi - y) / ((yhi - ylo) || 1) * (H - m.t - m.b);
  for (const t of yt) {
    svg("line", {x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: token(t === 0 ? "--axis" : "--grid"), "stroke-width": 1}, root);
    svg("text", {x: m.l - 8, y: Y(t) + 4, "text-anchor": "end"}, root).textContent = c.yFormat(t);
  }
  const n = c.slots ?? c.bars.length, band = (W - m.l - m.r) / n, bw = Math.max(2, Math.min(24, band - 2));
  const every = Math.max(1, Math.ceil(n * 44 / (W - m.l - m.r)));
  c.bars.forEach((b, i) => {
    const x0 = m.l + i * band + (band - bw) / 2;
    const bar = svg("path", {d: barPath(x0, bw, Y(0), Y(b.value)), fill: b.color}, root);
    if (i % every === 0) svg("text", {x: x0 + bw / 2, y: H - m.b + 18, "text-anchor": "middle"}, root).textContent = b.label;
    const hit = svg("rect", {x: m.l + i * band, y: m.t, width: band, height: H - m.t - m.b, fill: "transparent", tabindex: 0, class: "hit", "aria-label": `${b.title}: ${b.rows[0].value}`}, root);
    const on = () => { bar.setAttribute("opacity", 0.75); showTip(box, x0 + bw, Y(Math.max(0, b.value)), b.title, b.rows); };
    const off = () => { bar.setAttribute("opacity", 1); hideTip(box); };
    hit.addEventListener("pointermove", on); hit.addEventListener("focus", on);
    hit.addEventListener("pointerleave", off); hit.addEventListener("blur", off);
  });
}

function tiles() {
  const s = S[cur], single = singleAt(iterOf(s)), gap = gapOf(s);
  const items = [
    ["Step", stepName(s), `${fmt(iterOf(s), 0)} iterations spent`],
    ["Energy of the lower state", fmt(s.energy), `cut ${fmt(cut(s.energy), 0)}`],
    ["Energy of the upper view", fmt(s.view), s.view === null ? "only after a vote or an upper solve" : `cut ${fmt(cut(s.view), 0)}`],
    ["Single core, same iterations", fmt(single), single === null ? "" : `cut ${fmt(cut(single), 0)}`],
    ["Loss against the single core", gap === null ? "—" : (gap > 0 ? "+" : "") + fmt(gap), gap === null ? "" : gap > 0 ? "hierarchy higher (worse)" : "hierarchy lower or equal"],
  ];
  const box = document.getElementById("tiles"); box.textContent = "";
  for (const [label, value, note] of items) {
    const t = document.createElement("div"); t.className = "tile";
    for (const [cls, text] of [["label", label], ["value", value], ["note", note]]) {
      const e = document.createElement("div"); e.className = cls; e.textContent = text; t.appendChild(e);
    }
    box.appendChild(t);
  }
}

function table() {
  const tbl = document.getElementById("table"); tbl.textContent = "";
  const head = ["Sweep", "Step", "Iterations (summed)", "Iterations (parallel)", "Energy", "Cut", "Upper view",
                "Upper model energy", "Single core, same iterations", "Loss", "Far field (mean %)", "Field error (mean %)"];
  const tr = document.createElement("tr");
  for (const h of head) { const th = document.createElement("th"); th.textContent = h; tr.appendChild(th); }
  tbl.appendChild(tr);
  const mean = (a, k) => a ? a.reduce((x, y) => x + y[k], 0) / a.length : null;
  S.forEach((s, i) => {
    const row = document.createElement("tr"); if (i === cur) row.className = "current";
    const cells = [s.sweep, s.sweep === 0 ? "cores alone" : s.step, fmt(s.iter_sum, 0), fmt(s.iter_parallel, 0), fmt(s.energy),
                   fmt(cut(s.energy), 0), fmt(s.view), fmt(s.upper_energy), fmt(singleAt(iterOf(s))), fmt(gapOf(s)),
                   fmt(mean(s.far_field, "share")), fmt(mean(s.field_error, "wrong"))];
    for (const v of cells) { const td = document.createElement("td"); td.textContent = v; row.appendChild(td); }
    tbl.appendChild(row);
  });
}

function render() {
  const s = S[cur];
  document.getElementById("step-name").textContent = `${cur + 1} / ${S.length} · ${stepName(s)}`;
  document.getElementById("slider").value = cur;
  tiles();
  const C1 = token("--series-1"), C2 = token("--series-2"), C3 = token("--series-3");

  // 1. energy at every step
  const idx = S.map((_, i) => i), upto = (i) => i <= cur;
  const sweepStarts = S.map((t, i) => i).filter(i => i === 0 || S[i].sweep !== S[i - 1].sweep);
  const plotW = document.getElementById("plot-energy").clientWidth || 500;
  const every = Math.max(1, Math.ceil(sweepStarts.length * 40 / plotW));
  legend("legend-energy", [{name: "Lower state", color: C1}, {name: "Upper view", color: C2}]);
  lineChart(document.getElementById("plot-energy"), {
    aria: "Energy of the problem at every step",
    xMin: 0, xMax: Math.max(1, S.length - 1), marker: cur,
    xTicks: sweepStarts.filter((_, k) => k % every === 0).map(i => ({x: i, label: `S${S[i].sweep}`})),
    series: [
      {name: "Lower state", color: C1, xs: idx, ys: S.map((t, i) => upto(i) ? t.energy : null), label: "lower"},
      {name: "Upper view", color: C2, xs: idx, ys: S.map((t, i) => upto(i) ? t.view : null), dots: true},
    ],
    snap: (x) => Math.max(0, Math.min(cur, Math.round(x))),
    tip: (i) => ({title: stepName(S[i]), rows: [
      {value: fmt(S[i].energy), name: `lower state · cut ${fmt(cut(S[i].energy), 0)}`, color: C1},
      ...(S[i].view === null ? [] : [{value: fmt(S[i].view), name: `upper view · cut ${fmt(cut(S[i].view), 0)}`, color: C2}]),
    ]}),
  });

  // 2. energy against iterations spent
  const xMax = Math.max(D.single.x[D.single.x.length - 1], ...S.map(iterOf)), xNow = iterOf(s);
  const singleXs = [...D.single.x, xMax], singleYs = [...D.single.y, D.single.y[D.single.y.length - 1]];
  legend("legend-single", [{name: "Hierarchy, lower state", color: C1}, {name: "Single core, whole problem", color: C3}]);
  lineChart(document.getElementById("plot-single"), {
    aria: "Energy against iterations spent",
    xMin: 0, xMax, marker: xNow,
    xTicks: niceTicks(0, xMax, 5).filter(t => t <= xMax).map(t => ({x: t, label: compact(t)})),
    series: [
      {name: "single (ahead)", color: C3, xs: singleXs, ys: singleYs, opacity: 0.25},
      {name: "Single core", color: C3, xs: singleXs, ys: singleXs.map((x, i) => x <= xNow ? singleYs[i] : null), label: "single core"},
      {name: "Hierarchy", color: C1, xs: S.map(iterOf), ys: S.map((t, i) => upto(i) ? t.energy : null), label: "hierarchy"},
    ],
    snap: (x) => Math.max(0, Math.min(xNow, x)),
    tip: (x) => {
      let k = -1; S.forEach((t, i) => { if (i <= cur && iterOf(t) <= x) k = i; });
      return {title: `${fmt(x, 0)} iterations`, rows: [
        ...(k < 0 ? [] : [{value: fmt(S[k].energy), name: `hierarchy (${stepName(S[k])})`, color: C1}]),
        ...(singleAt(x) === null ? [] : [{value: fmt(singleAt(x)), name: "single core", color: C3}]),
      ]};
    },
  });

  // 3. loss against the single core
  const better = token("--div-better"), worse = token("--div-worse");
  legend("legend-gap", [{name: "Hierarchy higher than the single core (worse)", color: worse, rect: true},
                        {name: "Hierarchy lower (better)", color: better, rect: true}]);
  columnChart(document.getElementById("plot-gap"), {
    aria: "Loss against the single core at every step", slots: S.length,
    bars: S.slice(0, cur + 1).map((t) => ({gap: gapOf(t), t})).filter(b => b.gap !== null).map(({gap, t}) => ({
      label: `S${t.sweep}${t.sweep ? t.step[0] : ""}`, value: gap, color: gap > 0 ? worse : better, title: stepName(t),
      rows: [{value: (gap > 0 ? "+" : "") + fmt(gap), name: "loss"}, {value: fmt(t.energy), name: "hierarchy"},
             {value: fmt(singleAt(iterOf(t))), name: `single core after ${fmt(iterOf(t), 0)} iterations`}]})),
    yFormat: compact, empty: "No single-core value yet.",
  });

  // 4. far field share per core
  const lo = latest("far_field", cur);
  document.getElementById("desc-far").textContent = `|field part| / (|internal part| + |field part|), after the lower solve of sweep ${lo.sweep}${lo.sweep === 0 ? " (the cores alone: no field)" : ""}.`;
  columnChart(document.getElementById("plot-far"), {
    aria: "Share of each core's energy from the far field", yMin: 10,
    bars: lo.far_field.map(f => ({label: `core ${f.core}`, value: f.share, color: C1, title: `core ${f.core}, sweep ${lo.sweep}`,
      rows: [{value: fmt(f.share) + " %", name: "from the far field"}, {value: fmt(f.internal), name: "internal energy"},
             {value: fmt(f.field), name: "field energy"}]})),
    yFormat: (v) => compact(v) + " %", empty: "",
  });

  // 5. field error per core
  const up = latest("field_error", cur);
  document.getElementById("desc-error").textContent = up ? `Boundary nodes whose field from the expanded spins has another sign than the field from the real spins, at sweep ${up.sweep}.` : "The cores get a field from sweep 1 on.";
  columnChart(document.getElementById("plot-error"), {
    aria: "Field error on each core's boundary", yMin: 10,
    bars: up ? up.field_error.map(f => ({label: `core ${f.core}`, value: f.wrong, color: C1, title: `core ${f.core}, sweep ${up.sweep}`,
      rows: [{value: fmt(f.wrong) + " %", name: "wrong sign"}, {value: fmt(f.mae, 2), name: "mean |field error|"},
             {value: fmt(f.boundary, 0), name: "boundary nodes"}]})) : [],
    yFormat: (v) => compact(v) + " %", empty: "No field yet: the cores are solved alone at sweep 0.",
  });
  table();
}

function setStep(i) { cur = Math.max(0, Math.min(S.length - 1, i)); render(); }
function stop() { clearInterval(timer); timer = null; document.getElementById("play").textContent = "Play"; }
document.getElementById("title").textContent = D.title;
document.getElementById("subtitle").textContent = D.subtitle;
const slider = document.getElementById("slider");
slider.max = S.length - 1;
slider.addEventListener("input", () => { stop(); setStep(+slider.value); });
document.getElementById("play").addEventListener("click", () => {
  if (timer) return stop();
  if (cur === S.length - 1) setStep(0);
  document.getElementById("play").textContent = "Pause";
  timer = setInterval(() => { if (cur >= S.length - 1) stop(); else setStep(cur + 1); }, 900);
});
for (const r of document.querySelectorAll("input[name=mode]")) r.addEventListener("change", () => { mode = r.value; render(); });
let resizeTimer = null;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 120); });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
setStep(0);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
