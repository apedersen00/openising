"""Test 10: a point that cannot be solved does not stop anything, and a study can be saved while it runs.

What is tested, in multi_core_runs.py and multi_core_utils.py:
- run_single, run_grid and run_study on a point the partitioner or the solver refuses: they used to
  raise and lose the whole study. They now print the problem, the error and where it was raised,
  go on with the next point, and give the row of the point without energies (NaN), with the reason
  in "error" and the place in "error_where". build_hierarchy and trace_sweeps return None, and the
  plots draw nothing for None and leave the points without energy out.
- run_study(live_folder=...): the rows so far are saved as live_run.yaml / live_run.csv each time a
  point is done, so that load_run("live_run") reads a study that is still going.

Why: one refused point at the end of a study of several hours must not cost the study, the message
must say which point of which problem and where, and the points already solved must be on disk.

Problem: the 64-spin Maxcut problem of the dummy creator, modularity partitioning, 2 trials.
Points that cannot be solved: (3, 2), 3 partitions is not a power of two; (32, 4), 16 meta nodes
for 32 partitions; (2, 0), no node per meta node. Points that can: (2, 2), (4, 2) and the single
core (1, 0).

Checks (PASS/FAIL, the error is the number of violations unless said otherwise, tolerance 0):
  1. run_single on the three points: no exception, no trial, NaN for the 6 energies and the time,
     the expected error in "error", a function with its file and line in "error_where"
  2. the message of run_single names the point, the problem, the error and the function that raised
     it with its file; quiet=True prints nothing; a solved point has no "error" and prints nothing
  3. run_grid, in this process: one row per point in the order of the grid, the three points without
     energy, one message per point that failed, and the rows of the other points are the ones of
     the grid without the failing points (error: largest difference of the energies)
  4. run_grid with 2 workers: the same table as in this process (error: largest difference of the
     energies; inf when the rows or their errors differ), and again one message per failed point
  5. a list of numbers of sweeps: a point that failed has one row per number of sweeps, all with its error
  6. on_point of run_grid: called once per point, with the rows so far, in the order of the grid;
     an on_point that raises, or a message that cannot be written, does not stop the grid: the
     table is complete and each point says that it could not be reported or saved
  7. run_study with live_folder, 2 problems: the live run is saved once per point; read with load_run
     after each save it has the rows so far; at the end it is the table returned; no last_run link
     and no temporary file are left; without live_folder nothing is saved
  8. save_run and load_run keep the rows without energy and their error; the energy plots (lines, 3D,
     3D mean, heatmap) draw the solved points only; a selection with failed points only is refused
  9. build_hierarchy: None with a message for a point without hierarchy, a hierarchy for (2, 2);
     plot_hierarchy(None) draws nothing
 10. trace_sweeps: None with a message for a point that cannot be solved and for one node per meta
     node, a trace for (2, 2); the three plot_sweep_* functions draw nothing for None
 11. make_partitioning with 3 partitions: None and a message with the error and where;
     measure_partitions then has no row; plot_partitions draws the other splits only, and nothing
     when there is none; partition_summary of no row is an empty table

Run from the repo top: python no_backup/Multi_core_architecture/01_applications/test_10_failed_points_and_live_run.py
"""

import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import multi_core_runs as runs  # noqa: E402
import multi_core_utils as utils  # noqa: E402
import utile_louvain_compare as louvain_compare  # noqa: E402

FAILING = {
    (3, 2): "ValueError: Eigenvector partitioning requires a power-of-two partition count",
    (32, 4): "ValueError: Each subproblem needs at least one upper node",
    (2, 0): "ZeroDivisionError: ",
}
SOLVED = [(2, 2), (4, 2), (1, 0)]
GRID = [(2, 2), (3, 2), (4, 2), (32, 4), (2, 0), (1, 0)]
ENERGIES = [*utils.ENERGY_STATS, "en_mean"]
NEXT_POINT = "not solved, going on with the next point"

results = []


def check(name: str, error: float, tolerance: float = 0.0) -> None:
    passed = error <= tolerance
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: error = {error:.3e} (tolerance {tolerance:.1e})")


def printed(function, *arguments, **keywords):
    """What a call returns and what it prints."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = function(*arguments, **keywords)
    return result, output.getvalue()


def point_of(row) -> tuple[int, int]:
    return int(row["nb_partitions"]), int(row["nb_node_per_meta_nodes"])


def failed_violations(frame) -> int:
    """Rows of the failing points that have an energy or not their error, and rows of the others that have an error."""
    violations = 0
    for _, row in frame.iterrows():
        expected = FAILING.get(point_of(row))
        if expected is None:
            violations += int(row["nb_trials"] == 0 or np.isnan(row[ENERGIES].to_numpy(dtype=float)).any() or isinstance(row["error"], str))
        else:
            violations += int(row["nb_trials"] != 0 or not np.isnan(row[[*ENERGIES, "time_mean"]].to_numpy(dtype=float)).all())
            violations += int(not str(row["error"]).startswith(expected))
            violations += int(re.search(r"\w+ \(.+\.py:\d+\)", str(row["error_where"])) is None)
    return violations


def energies_gap(frame, other) -> float:
    """Largest difference of the energies of two tables with the same rows; inf when the rows differ."""
    same = len(frame) == len(other) and [point_of(row) for _, row in frame.iterrows()] == [point_of(row) for _, row in other.iterrows()]
    if not same:
        return float("inf")
    left, right = frame[ENERGIES].to_numpy(dtype=float), other[ENERGIES].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(left), np.isnan(right)):
        return float("inf")
    return float(np.nanmax(np.abs(left - right)))


if __name__ == "__main__":
    config = utils.make_config(
        utils.load_config("maxcut_c1"), "Maxcut", dummy_size=64, dummy_connectivity=1.0, nb_runs=2, nb_sweeps_Hierarchical_solver=2
    )
    problem = runs.generate_problem(config)

    violations = 0
    for point, expected in FAILING.items():
        try:
            row = runs.run_single(problem, *point, quiet=True)
        except Exception as exception:
            print(f"       {point} raised {exception!r}")
            violations += 1
            continue
        violations += int(row["nb_trials"] != 0) + int(not all(np.isnan(row[name]) for name in [*ENERGIES, "time_mean"]))
        violations += int(not row["error"].startswith(expected)) + int(re.search(r"\w+ \(.+\.py:\d+\)", row["error_where"]) is None)
        violations += int(point_of(row) != point or row["num_spins"] != 64 or row["connectivity"] != 1.0)
    check("run_single on the 3 points that cannot be solved: a row without energy, with error and error_where", violations)

    _, message = printed(runs.run_single, problem, 3, 2)
    print("       " + message.strip().replace("\n", "\n       "))
    wanted = [
        "nb_partitions=3, nb_node_per_meta_nodes=2",
        "of connectivity 1",
        FAILING[(3, 2)],
        "raised in partition_by_eigenvector (ising/utils/partitioning_schemes.py:",
    ]
    violations = sum(text not in message for text in wanted)
    violations += int(printed(runs.run_single, problem, 3, 2, quiet=True)[1] != "")
    solved_row, message = printed(runs.run_single, problem, 2, 2)
    violations += int(message != "") + int("error" in solved_row or "error_where" in solved_row) + int(solved_row["nb_trials"] != 2)
    check("message of run_single: the point, the problem, the error and where; nothing when quiet or solved", violations)

    frame, message = printed(runs.run_grid, problem, GRID, nb_workers=1)
    reference = runs.run_grid(problem, SOLVED, nb_workers=1)
    error = float(failed_violations(frame) + abs(message.count(NEXT_POINT) - len(FAILING)))
    error += float([point_of(row) for _, row in frame.iterrows()] != GRID)
    error += energies_gap(frame[frame["error"].isna()], reference)
    check("run_grid in this process: every point has its row, the others are solved as without the failing ones", error)

    parallel, message = printed(runs.run_grid, problem, GRID, nb_workers=2)
    same_errors = parallel["error"].fillna("").tolist() == frame["error"].fillna("").tolist()
    error = energies_gap(parallel, frame) if same_errors else float("inf")
    error += abs(message.count(NEXT_POINT) - len(FAILING))
    check("run_grid with 2 workers: the same table, one message per point that failed", error)

    swept = runs.run_grid(problem, [(2, 2), (3, 2), (1, 0)], nb_sweeps=[1, 2], nb_workers=1)
    of_failed = swept[swept["nb_partitions"] == 3]
    violations = int(of_failed["nb_sweeps"].tolist() != [1, 2]) + int(not of_failed["error"].str.startswith(FAILING[(3, 2)]).all())
    violations += int(len(swept) != 5) + int(swept[swept["nb_partitions"] == 2]["en_50"].isna().any())
    check("list of numbers of sweeps: one row per number of sweeps for the point that failed", violations)

    calls = []
    runs.run_grid(problem, GRID, nb_workers=1, on_point=lambda rows: calls.append([point_of(row) for row in rows]))
    violations = int(calls != [GRID[: index + 1] for index in range(len(GRID))])

    def refuse(rows):
        raise OSError("no room left to save")

    kept, message = printed(runs.run_grid, problem, GRID, nb_workers=1, on_point=refuse)
    violations += int(energies_gap(kept, frame) != 0.0) + abs(message.count("could not be reported or saved: OSError: no room left to save") - len(GRID))
    describe_failure = runs.describe_failure
    runs.describe_failure = lambda *arguments: (_ for _ in ()).throw(NameError("name '_problem_name' is not defined"))
    try:
        kept, message = printed(runs.run_grid, problem, GRID, nb_workers=2)
    finally:
        runs.describe_failure = describe_failure
    violations += int(energies_gap(kept, frame) != 0.0) + abs(message.count("could not be reported or saved: NameError") - len(FAILING))
    check("on_point: once per point with the rows so far; a failure to report or to save does not stop the grid", violations)

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        saves = []
        save_run = runs.save_run

        def recording_save(config, frame, *arguments, **keywords):
            paths = save_run(config, frame, *arguments, **keywords)
            saves.append((len(frame), len(utils.load_run(utils.LIVE_RUN, folder)[1]), keywords.get("name")))
            return paths

        runs.save_run = recording_save
        try:
            study, _ = printed(runs.run_study, config, GRID, connectivity=[1.0, 0.5], nb_workers=2, live_folder=folder)
            live_saves = len(saves)
            printed(runs.run_study, config, SOLVED, nb_workers=1)
        finally:
            runs.save_run = save_run
        _, live = utils.load_run(utils.LIVE_RUN, folder)
        violations = abs(live_saves - 2 * len(GRID)) + int(len(saves) != live_saves)
        violations += int([count for count, _, _ in saves] != list(range(1, 2 * len(GRID) + 1)))
        violations += sum(written != read or name != utils.LIVE_RUN for written, read, name in saves)
        violations += int(sorted(path.name for path in folder.iterdir()) != ["live_run.csv", "live_run.yaml"])
        violations += int(len(study) != 2 * len(GRID)) + int(sorted(study["connectivity"].unique()) != [0.5, 1.0])
        error = float(violations) + energies_gap(live, study)
        check("run_study with live_folder: the live run saved after each point, equal to the study at the end", error)

        config_path, csv_path = utils.save_run(config, frame, folder)
        _, loaded = utils.load_run(csv_path.stem, folder)
        violations = failed_violations(loaded) + int(len(loaded) != len(GRID))
        plots = {
            "lines": lambda: utils.plot_energy_lines(loaded, x="nb_partitions", by="nb_node_per_meta_nodes"),
            "3D": lambda: utils.plot_energy_3d(loaded),
            "3D mean": lambda: utils.plot_energy_3d_mean(loaded),
            "heatmap": lambda: utils.plot_energy_heatmap(loaded),
        }
        for name, plot in plots.items():
            try:
                fig = plot()
                if name == "heatmap":
                    violations += int([tick.get_text() for tick in fig.axes[0].get_yticklabels()] != ["2", "4"])
                if name == "lines":
                    violations += int([list(line.get_xdata()) for line in fig.axes[0].get_lines() if line.get_label().startswith("nb_node")] != [[2, 4]])
            except Exception as exception:
                print(f"       {name} failed: {exception!r}")
                violations += 1
            plt.close("all")
        try:
            utils.plot_energy_heatmap(loaded, nb_node_per_meta_nodes=4)
            violations += 1
            print("       a selection with failed points only was not refused")
        except ValueError as refusal:
            violations += int("could be solved" not in str(refusal))
        check("saved run: rows without energy kept with their error, plots of the solved points only", violations)

    hierarchy, message = printed(runs.build_hierarchy, problem, 32, 4)
    violations = int(hierarchy is not None) + sum(text not in message for text in ("nb_partitions=32", "no hierarchy", FAILING[(32, 4)], "raised in make_hierarchy"))
    hierarchy, message = printed(runs.build_hierarchy, problem, 2, 2)
    violations += int(message != "") + int(hierarchy is None or hierarchy.upper_model.num_variables != 32)
    violations += int(utils.plot_hierarchy(None) is not None) + int(utils.plot_hierarchy(hierarchy) is None)
    plt.close("all")
    check("build_hierarchy: None and a message for a point without hierarchy; plot_hierarchy(None) draws nothing", violations)

    trace, message = printed(runs.trace_sweeps, problem, 3, 2, nb_sweeps=2)
    violations = int(trace is not None) + sum(text not in message for text in ("nb_partitions=3", "no trace", FAILING[(3, 2)]))
    trace, message = printed(runs.trace_sweeps, problem, 2, 1, nb_sweeps=2)
    violations += int(trace is not None) + int("one spin per meta node" not in message)
    trace, message = printed(runs.trace_sweeps, problem, 2, 2, nb_sweeps=2)
    violations += int(message != "") + int(trace is None or len(trace["sweeps"]) != 2)
    for plot in (utils.plot_sweep_couplings, utils.plot_sweep_diff, utils.plot_sweep_fields):
        violations += int(plot(None) is not None) + int(plot(trace) is None)
        plt.close("all")
    check("trace_sweeps: None and a message for a point without trace; the plot_sweep_* draw nothing for None", violations)

    labels, message = printed(runs.make_partitioning, problem, 3)
    wanted = ["3 partitions of connectivity 1 with modularity", FAILING[(3, 2)], "raised in partition_by_eigenvector (ising/utils/partitioning_schemes.py:"]
    violations = int(labels is not None) + sum(text not in message for text in wanted)
    rows, message = printed(runs.measure_partitions, problem, nb_partitions=3, nb_draws=4)
    violations += int(rows != []) + abs(message.count("no partitioning") - 1)
    violations += int(len(runs.measure_partitions(problem, nb_partitions=4)) != 1) + int(len(louvain_compare.partition_summary([])) != 0)
    splits = {"3 asked": labels, "4 asked": runs.make_partitioning(problem, 4)}
    fig = louvain_compare.plot_partitions(problem["ising_model"], splits)
    # One panel for the split that exists, and the color scale
    violations += int(fig is None or len(fig.axes) != 2) + int(louvain_compare.plot_partitions(problem["ising_model"], {"3 asked": None}) is not None)
    plt.close("all")
    check("make_partitioning: None and a message for a refused number of partitions; no row, no panel for it", violations)

    print(f"{sum(results)}/{len(results)} passed")
