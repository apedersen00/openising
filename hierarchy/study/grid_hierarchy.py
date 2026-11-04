"""
Energy of the hierarchical solver on a grid of (number of partitions, nodes per metanode), on several graphs, against
the best known cut (ground truth) and one core solving the whole problem: the study of 01_meeting_1_results redone
with hierarchy_solver.HierarchySolver.

    python grid_hierarchy.py                                  G32, G29, G40, 4 sweeps, 20 trials, all the CPU cores
    python grid_hierarchy.py --benchmarks G/regenerated/G32_2000_4000_toroidal_1410 --partitions 2 4 --trials 4

- partitions: KL (partition_kl), powers of 2 from 2 while a core keeps at least --min-core-size nodes
- nodes per metanode: powers of 2 from 2, below the number of spins
- every solve (a core or the upper core) has the same budget, --iterations x --seeds; the single core is one solve
  of that budget on the whole problem, as in the previous study
- energies as in openising and in the previous study: H = -cut (our E + sum(J) / 2). best_energy = -best known
  cut, read at the end of the benchmark file name
- one point (a graph, a number of partitions, a number of nodes per metanode: all its trials) per CPU core

Writes in study/results/ (or --out):
- <time>_grid.csv: one row per point, with the columns of the previous study, so that its plots read it
  (multi_core_utils.plot_energy_lines, plot_energy_heatmap; the single core is the solver "Multiplicative").
  en_min ... en_max: the energy after the last sweep; en_50_best: the median of the best energy over the sweeps;
  en_50_sweep<k>: the median after sweep k (0: the cores alone)
- <time>_trials.csv: one row per trial, <time>_settings.json, and one figure per graph, <time>_<graph>.png
"""
import os

for _variable in ("MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")  # one BLAS thread per process: the points run in parallel

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import datetime  # noqa: E402
from functools import cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1] / "ising_hierarchy"))
from hierarchy_solver import HierarchySolver  # noqa: E402
from ising_solver import sweep  # noqa: E402
from partition import partition_kl  # noqa: E402
from problems import load_maxcut  # noqa: E402

BENCHMARKS = [
    "G/regenerated/G32_2000_4000_toroidal_1410",
    "G/regenerated/G29_2000_19990_random_3405",
    "G/regenerated/G40_2000_11766_planar_2400",
]
BASELINE_SOLVER = "Multiplicative"  # the name the plots of the previous study give the single core
HIERARCHICAL_SOLVER = "Hierarchical_solver"
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]  # ordinal steps 250 ... 700 of the blue ramp


@cache
def problem(benchmark):
    """J, h of a benchmark, loaded once per process."""
    return load_maxcut(benchmark)


def best_known_cut(benchmark):
    """The best known cut, at the end of the file name of the regenerated benchmarks (None if it is not there)."""
    last = Path(benchmark).stem.split("_")[-1]
    return int(last) if last.isdigit() else None


def powers_of_two(low, high):
    """2, 4, 8, ... from low up to high (both included)."""
    values, value = [], 2
    while value <= high:
        if value >= low:
            values.append(value)
        value *= 2
    return values


def run_point(benchmark, labels, nb_partitions, nodes_per_metanode, settings):
    """All the trials of one point of the hierarchy. Energies as H = -cut."""
    J, h = problem(benchmark)
    offset = float(J.sum() / 2)
    start = time.time()
    solver = HierarchySolver(J, h, labels, nodes_per_metanode, settings["method"])
    build_time = time.time() - start
    trials = []
    for trial in range(settings["trials"]):
        start = time.time()
        _, best, history = solver.solve(settings["sweeps"], seed=trial, nb_iterations=settings["iterations"],
                                        nb_seeds=settings["seeds"], max_time=settings["max_time"])
        row = {"trial": trial, "time": time.time() - start, "final": history[-1]["energy"] + offset,
               "best": best + offset}
        row.update({f"sweep{r['sweep']}": r["energy"] + offset for r in history})
        trials.append(row)
    return {"benchmark": benchmark, "solver": HIERARCHICAL_SOLVER, "nb_partitions": nb_partitions,
            "nb_node_per_meta_nodes": nodes_per_metanode, "total_meta_nodes": len(solver.metanodes),
            "core_sizes": [len(core.nodes) for core in solver.cores], "build_time": build_time, "trials": trials}


def run_single(benchmark, settings):
    """All the trials of the single core: one solve of the budget of a solve on the whole problem."""
    J, h = problem(benchmark)
    offset = float(J.sum() / 2)
    trials = []
    for trial in range(settings["trials"]):
        seeds = [42 + 100_000 * trial + i for i in range(settings["seeds"])]
        start = time.time()
        _, energies, _ = sweep(J, h, seeds, settings["iterations"], max_time=settings["max_time"])
        energy = float(energies.min()) + offset
        trials.append({"trial": trial, "time": time.time() - start, "final": energy, "best": energy})
    return {"benchmark": benchmark, "solver": BASELINE_SOLVER, "nb_partitions": 1, "nb_node_per_meta_nodes": 0,
            "total_meta_nodes": 0, "core_sizes": [len(h)], "build_time": 0.0, "trials": trials}


def summarize(result, settings):
    """One row of the grid table, with the columns of the previous study."""
    J, h = problem(result["benchmark"])
    nb_spins = len(h)
    nb_couplings = np.count_nonzero(np.triu(J, 1))
    cut = best_known_cut(result["benchmark"])
    row = {
        "problem_type": "Maxcut", "benchmark": result["benchmark"] + ".txt", "num_spins": nb_spins,
        "connectivity": nb_couplings / (nb_spins * (nb_spins - 1) / 2),
        "connections_per_spin": 2 * nb_couplings / nb_spins,
        "best_energy": -cut if cut is not None else np.nan,
        "nb_partitions": result["nb_partitions"], "nb_node_per_meta_nodes": result["nb_node_per_meta_nodes"],
        "solver": result["solver"], "spins_per_partition": nb_spins / result["nb_partitions"],
        "total_meta_nodes": result["total_meta_nodes"],
        "meta_nodes_per_partition": result["total_meta_nodes"] / result["nb_partitions"],
        "spins_per_meta_node": nb_spins / result["total_meta_nodes"] if result["total_meta_nodes"] else 0,
        "nb_sweeps": settings["sweeps"], "nb_trials": settings["trials"],
        "core_sizes": " ".join(map(str, result["core_sizes"])), "build_time": result["build_time"],
        "error": result.get("error", np.nan), "error_where": result.get("error_where", np.nan),
    }
    trials = pd.DataFrame(result["trials"])
    if trials.empty:
        return row
    final = trials["final"]
    row.update({"en_min": final.min(), "en_25": final.quantile(0.25), "en_50": final.median(),
                "en_75": final.quantile(0.75), "en_max": final.max(), "en_mean": final.mean(),
                "time_mean": trials["time"].mean(), "en_50_best": trials["best"].median()})
    for column in trials.columns:
        if column.startswith("sweep"):
            row[f"en_50_{column}"] = trials[column].median()
    return row


def plot_benchmark(grid, benchmark, path):
    """Median energy against the number of partitions (one line per nodes per metanode), and the median heatmap."""
    import matplotlib  # noqa: PLC0415  (only when the figures are made)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import ListedColormap  # noqa: PLC0415

    points = grid[(grid["benchmark"] == benchmark + ".txt") & (grid["solver"] == HIERARCHICAL_SOLVER)].dropna(
        subset=["en_50"])
    single = grid[(grid["benchmark"] == benchmark + ".txt") & (grid["solver"] == BASELINE_SOLVER)]
    if points.empty:
        return
    ink, muted, grid_color = "#0b0b0b", "#898781", "#e1e0d9"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": "#52514e", "xtick.color": muted, "ytick.color": muted})
    fig, (ax_lines, ax_heat) = plt.subplots(1, 2, figsize=(14, 5.2), layout="constrained",
                                            gridspec_kw={"width_ratios": [1.1, 1]})
    sizes = sorted(points["nb_node_per_meta_nodes"].unique())
    colors = [BLUE_RAMP[round(i * (len(BLUE_RAMP) - 1) / max(1, len(sizes) - 1))] for i in range(len(sizes))]
    for size, color in zip(sizes, colors):
        line = points[points["nb_node_per_meta_nodes"] == size].sort_values("nb_partitions")
        ax_lines.plot(line["nb_partitions"], line["en_50"], color=color, linewidth=1.6, marker="o", markersize=5,
                      markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=f"{size}")
    references = []
    if not single.empty:
        references.append((single["en_50"].iloc[0], "single core", "#52514e", (0, (5, 3))))
    if points["best_energy"].notna().any():
        references.append((points["best_energy"].iloc[0], "best known", muted, (0, (5, 2, 1, 2))))
    for value, label, color, style in references:
        ax_lines.axhline(value, color=color, linewidth=1.3, linestyle=style)
        ax_lines.annotate(f"{label} {value:.0f}", (1, value), xycoords=("axes fraction", "data"), xytext=(4, 0),
                          textcoords="offset points", va="center", color="#52514e", fontsize=9)
    ax_lines.set_xscale("log", base=2)
    partitions = sorted(points["nb_partitions"].unique())
    ax_lines.set_xticks(partitions, [str(p) for p in partitions])
    ax_lines.minorticks_off()
    ax_lines.grid(axis="y", color=grid_color, linewidth=0.8)
    ax_lines.set_axisbelow(True)
    for side in ("top", "right"):
        ax_lines.spines[side].set_visible(False)
    ax_lines.set_xlabel("number of partitions (cores)")
    ax_lines.set_ylabel("median energy after the last sweep (−cut, lower is better)")
    ax_lines.legend(title="nodes per metanode", ncols=2, fontsize=8.5, title_fontsize=9, frameon=False,
                    loc="upper left", bbox_to_anchor=(1.18, 1))

    table = points.pivot_table(index="nb_partitions", columns="nb_node_per_meta_nodes", values="en_50")
    ramp = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
            "#1c5cab", "#184f95", "#104281", "#0d366b"]
    image = ax_heat.imshow(table.to_numpy(), cmap=ListedColormap(ramp[::-1]), aspect="auto")  # lower (better): darker
    low, high = np.nanmin(table.to_numpy()), np.nanmax(table.to_numpy())
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            value = table.iat[r, c]
            if not np.isnan(value):
                dark = high > low and (value - low) / (high - low) < 0.45
                ax_heat.text(c, r, f"{value:.0f}", ha="center", va="center", fontsize=8.5,
                             color="#ffffff" if dark else ink)
    ax_heat.set_xticks(range(table.shape[1]), [str(v) for v in table.columns])
    ax_heat.set_yticks(range(table.shape[0]), [str(v) for v in table.index])
    ax_heat.set_xlabel("nodes per metanode")
    ax_heat.set_ylabel("number of partitions (cores)")
    for side in ax_heat.spines.values():
        side.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax_heat, shrink=0.85)
    colorbar.outline.set_visible(False)
    colorbar.set_label("median energy (−cut)")
    name = Path(benchmark).name
    reference = []
    if not single.empty:
        reference.append(f"single core {single['en_50'].iloc[0]:.0f}")
    if points["best_energy"].notna().any():
        reference.append(f"best known {points['best_energy'].iloc[0]:.0f}")
    fig.suptitle(f"{name}: median of {int(points['nb_trials'].iloc[0])} trials, {int(points['nb_sweeps'].iloc[0])} "
                 f"sweeps ({', '.join(reference)})", color=ink, fontsize=12, x=0.01, ha="left")
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    parser.add_argument("--partitions", nargs="+", type=int, help="default: powers of 2 down to --min-core-size")
    parser.add_argument("--min-core-size", type=int, default=64, help="smallest core of the default partitions")
    parser.add_argument("--nodes-per-metanode", nargs="+", type=int, help="default: powers of 2 below the spins")
    parser.add_argument("--method", choices=["linkage", "original"], default="linkage")
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100, help="iterations of every solve")
    parser.add_argument("--seeds", type=int, default=1, help="seeds of every solve")
    parser.add_argument("--max-time", type=float, default=np.inf, help="time budget of every run")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="processes: one point per process")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    settings = {"method": args.method, "sweeps": args.sweeps, "trials": args.trials, "iterations": args.iterations,
                "seeds": args.seeds, "max_time": args.max_time}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for benchmark in args.benchmarks:
        J, h = problem(benchmark)
        nb_spins = len(h)
        partitions = args.partitions or powers_of_two(2, nb_spins // args.min_core_size)
        sizes = args.nodes_per_metanode or powers_of_two(2, nb_spins - 1)
        tasks.append((run_single, (benchmark, settings), f"{Path(benchmark).name.split('_')[0]} single core"))
        for nb_partitions in partitions:
            labels = partition_kl(J, nb_partitions)  # the same cores for every nodes per metanode
            for size in sizes:
                tasks.append((run_point, (benchmark, labels, nb_partitions, size, settings),
                              f"{Path(benchmark).name.split('_')[0]} p={nb_partitions} m={size}"))
    (args.out / f"{stamp}_settings.json").write_text(json.dumps({
        **settings, "max_time": str(args.max_time), "benchmarks": args.benchmarks, "partitions": args.partitions,
        "min_core_size": args.min_core_size, "nodes_per_metanode": args.nodes_per_metanode,
        "partition": "kl", "workers": args.workers, "points": len(tasks)}, indent=2))
    print(f"{len(tasks)} points on {args.workers} processes, writing {args.out / stamp}_*")

    rows, trial_rows = [], []
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(function, *arguments): name for function, arguments, name in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                result = future.result()
            except Exception as error:  # a point that fails does not stop the others
                print(f"{done}/{len(tasks)} {name}: {type(error).__name__}: {error}")
                continue
            row = summarize(result, settings)
            rows.append(row)
            point = {key: row[key] for key in ("benchmark", "solver", "nb_partitions", "nb_node_per_meta_nodes")}
            trial_rows += [{**point, **trial} for trial in result["trials"]]
            print(f"{done}/{len(tasks)} {name}: median {row['en_50']:.0f} (best known {row['best_energy']:.0f}), "
                  f"{row['time_mean']:.1f} s per trial, {time.time() - start:.0f} s so far", flush=True)
            pd.DataFrame(rows).sort_values(["benchmark", "nb_partitions", "nb_node_per_meta_nodes"]).to_csv(
                args.out / f"{stamp}_grid.csv", index=False)  # after every point: a long run can be read live
    grid = pd.DataFrame(rows).sort_values(["benchmark", "nb_partitions", "nb_node_per_meta_nodes"])
    grid.to_csv(args.out / f"{stamp}_grid.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(args.out / f"{stamp}_trials.csv", index=False)
    for benchmark in args.benchmarks:
        plot_benchmark(grid, benchmark, args.out / f"{stamp}_{Path(benchmark).name.split('_')[0]}.png")
    print(f"done in {time.time() - start:.0f} s: {args.out / stamp}_grid.csv")


if __name__ == "__main__":
    main()
