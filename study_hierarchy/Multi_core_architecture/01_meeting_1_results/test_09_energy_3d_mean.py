"""Test 09: plot_energy_3d_mean, the 3D mesh through one statistic of each point.

What is tested: multi_core_utils.plot_energy_3d_mean. It places the points of a run on the
(nb_partitions, nodes per meta node) grid at the height of one statistic (the 50 % by default),
links each point to its neighbours along the two axes and fills the cells between them.

Why: the mesh must go through the right energy at the right place of the grid, and a point the run
does not have must leave a hole instead of an error or a link through nothing.

The run is made here: 4 numbers of partitions x 3 numbers of nodes per meta node, with an en_50 and
an en_mean that are different at every point, and a single-core row below all of them.

Checks (PASS/FAIL, tolerance 0 unless said otherwise):
  1. one line per number of partitions and one per number of nodes per meta node, each through the
     50 % of its points in order (error: largest difference with the rows; inf when a line is missing)
  2. one marker per point, 6 filled cells for the 4 x 3 grid, and the ticks are the values of the run
  3. the color scale goes from the lowest to the highest 50 % of the points, and the single-core
     plane is drawn at the single-core 50 % (the energy axis reaches it)
  4. stat="en_mean": the lines go through en_mean, and the legend and the color scale say "mean"
  5. a run without the point (2 partitions, 2 nodes per meta node): no error, the two lines through
     it have a gap there, 11 markers, and the 4 cells around it are not filled (2 left)
  6. a run with one number of partitions: no error, no filled cell, one line through its 3 points

Run from the repo top: python no_backup/Multi_core_architecture/01_applications/test_09_energy_3d_mean.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Poly3DCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import multi_core_utils as utils  # noqa: E402

PARTITIONS = [1, 2, 4, 8]
META_NODES = [1, 2, 4]
NUM_SPINS = 64
SINGLE_CORE = {"en_50": -1200.0, "en_mean": -1150.0}

results = []


def check(name: str, error: float, tolerance: float = 0.0) -> None:
    passed = error <= tolerance
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: error = {error:.3e} (tolerance {tolerance:.1e})")


def energy(stat: str, nb_partitions: int, meta_nodes: int) -> float:
    """A different value at every point of the grid, and another one for the mean than for the 50 %."""
    return -1000.0 + 10 * nb_partitions + 3 * meta_nodes + nb_partitions * meta_nodes + (1.5 if stat == "en_mean" else 0.0)


def make_run(partitions: list[int] = PARTITIONS, missing: tuple = ()) -> pd.DataFrame:
    common = {"problem_type": "Maxcut", "benchmark": np.nan, "best_energy": np.nan, "num_spins": NUM_SPINS, "connectivity": 1.0}
    rows = [
        dict(
            common,
            nb_partitions=nb_partitions,
            nb_node_per_meta_nodes=meta_nodes,
            solver=utils.HIERARCHICAL_SOLVER,
            spins_per_partition=NUM_SPINS / nb_partitions,
            total_meta_nodes=NUM_SPINS // meta_nodes,
            nb_sweeps=2.0,
            en_50=energy("en_50", nb_partitions, meta_nodes),
            en_mean=energy("en_mean", nb_partitions, meta_nodes),
        )
        for nb_partitions in partitions
        for meta_nodes in META_NODES
        if (nb_partitions, meta_nodes) not in missing
    ]
    rows.append(dict(common, nb_partitions=1, nb_node_per_meta_nodes=0, solver=utils.BASELINE_SOLVER, nb_sweeps=np.nan, **SINGLE_CORE))
    return pd.DataFrame(rows)


def lines_error(ax, stat: str, partitions: list[int], missing: tuple = ()) -> float:
    """Largest difference between the lines of the mesh and the energies of the grid; inf when a line is missing."""
    drawn = [tuple(np.asarray(values, dtype=float) for values in line.get_data_3d()) for line in ax.get_lines()]
    expected = []
    for x, nb_partitions in enumerate(partitions):
        heights = [np.nan if (nb_partitions, value) in missing else energy(stat, nb_partitions, value) for value in META_NODES]
        expected.append((np.full(len(META_NODES), float(x)), np.arange(len(META_NODES), dtype=float), np.array(heights)))
    for y, meta_nodes in enumerate(META_NODES):
        heights = [np.nan if (value, meta_nodes) in missing else energy(stat, value, meta_nodes) for value in partitions]
        expected.append((np.arange(len(partitions), dtype=float), np.full(len(partitions), float(y)), np.array(heights)))
    if len(drawn) != len(expected):
        return float("inf")
    error = 0.0
    for xs, ys, zs in expected:
        match = [line for line in drawn if np.array_equal(line[0], xs) and np.array_equal(line[1], ys)]
        if len(match) != 1 or not np.array_equal(np.isnan(match[0][2]), np.isnan(zs)):
            return float("inf")
        error = max(error, float(np.nanmax(np.abs(match[0][2] - zs))))
    return error


def nb_markers(ax) -> int:
    return sum(isinstance(collection, Path3DCollection) for collection in ax.collections)


def nb_cells(fig) -> int:
    """The filled cells of the mesh: the faces of its collection, the first one of the axes; 0 without a mesh."""
    fig.canvas.draw()
    ax = fig.axes[0]
    first = ax.collections[0]
    return len(first.get_paths()) if isinstance(first, Poly3DCollection) and nb_markers(ax) > 1 and len(first.get_paths()) != 1 else 0


if __name__ == "__main__":
    run = make_run()
    fig = utils.plot_energy_3d_mean(run)
    ax, colorbar = fig.axes
    check("lines of the mesh: through the 50 % of every point, along both axes", lines_error(ax, "en_50", PARTITIONS), tolerance=1e-12)

    violations = abs(nb_markers(ax) - len(PARTITIONS) * len(META_NODES)) + abs(nb_cells(fig) - 6)
    violations += int([tick.get_text() for tick in ax.get_xticklabels()] != [str(value) for value in PARTITIONS])
    violations += int([tick.get_text() for tick in ax.get_yticklabels()] != [str(value) for value in META_NODES])
    check("12 markers, 6 filled cells, ticks at the values of the run", violations)

    energies = [energy("en_50", nb_partitions, value) for nb_partitions in PARTITIONS for value in META_NODES]
    low, high = colorbar.get_ylim()
    error = abs(low - min(energies)) + abs(high - max(energies))
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    error += float("single core 50 %" not in labels) + float(ax.get_zlim()[0] > SINGLE_CORE["en_50"])
    check("color scale from the lowest to the highest 50 %, single-core plane at its 50 %", error, tolerance=1e-12)
    plt.close("all")

    fig = utils.plot_energy_3d_mean(run, stat="en_mean")
    ax, colorbar = fig.axes
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    error = lines_error(ax, "en_mean", PARTITIONS)
    error += float(labels != ["mean of each point", "single core mean"]) + float(colorbar.get_ylabel() != "mean energy")
    error += float(ax.get_zlim()[0] > SINGLE_CORE["en_mean"])
    check('stat="en_mean": lines through en_mean, named "mean"', error, tolerance=1e-12)
    plt.close("all")

    missing = ((2, 2),)
    try:
        fig = utils.plot_energy_3d_mean(make_run(missing=missing))
        ax = fig.axes[0]
        error = lines_error(ax, "en_50", PARTITIONS, missing) + abs(nb_markers(ax) - 11) + abs(nb_cells(fig) - 2)
    except Exception as exception:
        print(f"       failed: {type(exception).__name__}: {exception}")
        error = float("inf")
    check("a point missing: a gap in its two lines, 11 markers, 2 filled cells left", error, tolerance=1e-12)
    plt.close("all")

    try:
        fig = utils.plot_energy_3d_mean(make_run(partitions=[4]), nb_partitions=4)
        ax = fig.axes[0]
        along = [line for line in ax.get_lines() if len(line.get_data_3d()[0]) == len(META_NODES)]
        heights = np.asarray(along[0].get_data_3d()[2], dtype=float) if len(along) == 1 else np.full(len(META_NODES), np.inf)
        error = float(np.max(np.abs(heights - [energy("en_50", 4, value) for value in META_NODES])))
        error += abs(nb_markers(ax) - len(META_NODES)) + nb_cells(fig)
    except Exception as exception:
        print(f"       failed: {type(exception).__name__}: {exception}")
        error = float("inf")
    check("one number of partitions: one line through its 3 points, no filled cell", error, tolerance=1e-12)

    print(f"{sum(results)}/{len(results)} passed")
