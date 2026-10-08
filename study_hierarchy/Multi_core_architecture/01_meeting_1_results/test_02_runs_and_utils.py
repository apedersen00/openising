"""Test 02: the refactored modules multi_core_runs.py and multi_core_utils.py.

What is tested: generate_problem / run_single / run_grid, the saved runs (timestamped yaml
and csv, last_run links) and the plots made from a csv that was read back.

Why: the refactor replaces the config file + api.get_hamiltonian_energy call per point by
an in-memory config and a problem generated once. It must give the same trials as the
repo pipeline, whether a point is solved in this process or in a worker, and a saved run
must be enough to plot and to rebuild the same problem.

Checks (consistency, PASS/FAIL, tolerance 0 unless stated):
  1-3. MIN / 25 / 50 / 75 / MAX of run_single == those of the api.get_hamiltonian_energy energies, for three points
  4.   run_grid in 2 worker processes == run_grid in this process
  5.   the generated problem is not modified by the runs
  6.   total_meta_nodes == the meta-node count the pipeline gives to the solver (parse_hyperparameters)
  7.   connections_per_spin == num_spins - 1 on the fully connected problem
  8.   save_run writes <name>.yaml / <name>.csv and last_run.yaml / last_run.csv are symlinks to them
  9.   a second save_run moves the links to the new files and keeps the old ones
  10.  load_run gives back the table of the last save (1e-9)
  11.  the yaml of the saved run regenerates the same J
  12.  make_config refuses a key that is not in the config
  13.  every plot function, fed with the csv read back, returns a figure that can be saved
  14.  build_hierarchy: nb_partitions partitions covering every spin once, total_meta_nodes meta nodes
  15.  build_hierarchy: the J of each partition is the J of the problem on the spins of that partition
  16.  build_hierarchy: the upper J between two meta nodes is the MEAN of the J between their spins (1e-12)
  17.  trace_sweeps records one entry per sweep, with one J and one h per partition
  18.  trace_sweeps: the upper J and the J of every partition are the same at every sweep
  19.  trace_sweeps does not change the run: its energy == the first trial of run_single with the same sweeps
  20.  the three sweep plots (J and h per sweep, fields, difference of two sweeps) can be saved
  21.  the Maxcut generator used here (connectivity below 1 allowed) draws the same J as the repo one at connectivity 1
  22.  run_grid with nb_sweeps=[1, 2]: one row per point and number of sweeps, one single-core row;
       its rows with 2 sweeps == the run without nb_sweeps (2 is the value of the config)
  23.  run_grid with the single value nb_sweeps=2 == run_grid with the list [2]
  24.  run_study with connectivity=[1.0, 0.5]: every point for each connectivity, one single-core row per connectivity;
       its rows at connectivity 1.0 == run_grid on the fully connected problem
  25.  save_run on that study: lists of sweeps and connectivities in the name and in the yaml
  26.  the energy plots on the study: against nb_partitions / nb_node_per_meta_nodes for different sweeps and
       for different connectivities, 3D and heatmap of one configuration; a missing choice is refused
  27.  run_single_sweeps (one run with 4 sweeps, energy kept after each) == separate runs with 1, 2 and 4 sweeps,
       for a point with partitions solved and for a point with one node per meta node

Reported only (no PASS/FAIL): the range of the upper J against the range of J, and against the SUM
of the J between the spins of two meta nodes, which is what the energy of two blocks of aligned spins uses;
the largest change of a partition field h between two sweeps;
the connections per spin measured at connectivity 0.5 against 0.5 * (num_spins - 1).

Run from the repo top: python no_backup/Multi_core_architecture/01_applications/test_02_runs_and_utils.py
"""

import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import multi_core_runs as runs  # noqa: E402
import multi_core_utils as utils  # noqa: E402
from ising import api  # noqa: E402
from ising.utils.flow import parse_hyperparameters  # noqa: E402

PROBLEM_TYPE = "Maxcut"
SIZE = 24
GRID = [(1, 0), (2, 1), (2, 3), (4, 1), (4, 3)]
API_POINTS = [(1, 0), (2, 1), (4, 3)]
STATS = list(utils.ENERGY_STATS)
TOLERANCE = 0.0

results = []


def check(name: str, error: float, tolerance: float = TOLERANCE) -> None:
    passed = error <= tolerance
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: error = {error:.3e} (tolerance {tolerance:.1e})")


def api_energies(config: dict, point: tuple[int, int]) -> np.ndarray:
    """The repo path: one config file and one full pipeline run for the point."""
    solver = runs.solver_of(*point)
    config = dict(config, solvers=[solver], nb_cores=point[0], nb_partitions=point[0], nb_meta_nodes=point[1])
    del config["problem_type"]
    with tempfile.TemporaryDirectory() as folder:
        config_file = Path(folder) / "point.yaml"
        with config_file.open("w") as file:
            yaml.safe_dump(config, file)
        ans, _ = api.get_hamiltonian_energy(PROBLEM_TYPE, str(config_file), runs.LOGGING_LEVEL)
    return np.asarray(ans.energies[solver], dtype=float)


if __name__ == "__main__":
    config = utils.make_config(utils.load_config("maxcut_c1"), PROBLEM_TYPE, dummy_size=SIZE, nb_runs=10)
    problem = runs.generate_problem(config)
    model = problem["ising_model"]
    reference_J = model.J.copy()

    sequential = runs.run_grid(problem, GRID, nb_workers=1)
    for point in API_POINTS:
        row = sequential[(sequential["nb_partitions"] == point[0]) & (sequential["nb_node_per_meta_nodes"] == point[1])].iloc[0]
        expected = np.percentile(api_energies(config, point), [0, 25, 50, 75, 100])
        error = float(np.max(np.abs(row[STATS].to_numpy(dtype=float) - expected)))
        check(f"MIN/25/50/75/MAX of point {point} ({row['nb_trials']} trials, run_single vs api)", error)

    parallel = runs.run_grid(problem, GRID, nb_workers=2)
    error = float(np.nanmax(np.abs(sequential[STATS].to_numpy() - parallel[STATS].to_numpy())))
    check("run_grid in 2 workers vs in this process", error)

    unchanged = isinstance(problem["config"].nb_partitions, list) and problem["config"].solvers == config["solvers"]
    error = float(np.max(np.abs(model.J - reference_J))) if unchanged else float("inf")
    check("generated problem unchanged after the runs", error)

    error = 0.0
    for _, row in sequential[sequential["solver"] == utils.HIERARCHICAL_SOLVER].iterrows():
        arguments = Namespace(
            **dict(config, solvers=[row["solver"]], nb_partitions=row["nb_partitions"], nb_meta_nodes=row["nb_node_per_meta_nodes"])
        )
        error = max(error, abs(parse_hyperparameters(arguments, model)["nb_meta_nodes"] - row["total_meta_nodes"]))
    check("total_meta_nodes vs the count the pipeline gives to the solver (4 points)", float(error))

    check("connections_per_spin on the fully connected problem", abs(utils.connections_per_spin(model) - (SIZE - 1)))

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        config_path, csv_path = utils.save_run(config, sequential, folder)
        links = [folder / "last_run.yaml", folder / "last_run.csv"]
        linked = all(link.is_symlink() for link in links) and [link.resolve() for link in links] == [
            config_path.resolve(),
            csv_path.resolve(),
        ]
        print(f"       name: {csv_path.stem}")
        check("save_run writes the yaml and the csv, last_run.* are symlinks to them", 0.0 if linked else float("inf"))

        time.sleep(1.1)  # the timestamp of the name has a resolution of one second
        second_config_path, second_csv_path = utils.save_run(config, parallel, folder)
        moved = (
            second_csv_path != csv_path
            and csv_path.exists()
            and config_path.exists()
            and links[0].resolve() == second_config_path.resolve()
            and links[1].resolve() == second_csv_path.resolve()
        )
        check("a second save_run moves the links and keeps the first files", 0.0 if moved else float("inf"))

        loaded_config, loaded = utils.load_run(folder=folder)
        same_columns = list(loaded.columns) == list(parallel.columns)
        numeric = parallel.select_dtypes("number").columns
        error = float(np.nanmax(np.abs(loaded[numeric].to_numpy() - parallel[numeric].to_numpy()))) if same_columns else float("inf")
        check("load_run gives back the table", error, 1e-9)
        regenerated = runs.generate_problem(loaded_config)["ising_model"]
        check("the saved yaml regenerates the same J", float(np.max(np.abs(regenerated.J - reference_J))))

        try:
            utils.make_config(config, PROBLEM_TYPE, dummy_sise=10)
            refused = False
        except KeyError:
            refused = True
        check("make_config refuses a key that is not in the config", 0.0 if refused else float("inf"))

        plots = {
            "model": lambda: utils.plot_model(regenerated),
            "energy_3d": lambda: utils.plot_energy_3d(loaded),
            "energy_3d_spins_per_meta": lambda: utils.plot_energy_3d(loaded, y="spins_per_meta_node"),
            "lines_vs_partitions": lambda: utils.plot_energy_lines(loaded, x="nb_partitions", by="nb_node_per_meta_nodes"),
            "lines_vs_meta_nodes": lambda: utils.plot_energy_lines(loaded, x="nb_node_per_meta_nodes", by="nb_partitions"),
            "heatmap": lambda: utils.plot_energy_heatmap(loaded),
            "hierarchy": lambda: utils.plot_hierarchy(runs.build_hierarchy(problem, 4, 3)),
        }
        failed = 0
        for name, plot in plots.items():
            try:
                path = utils.save_figure(plot(), name, folder)
                failed += not path.stat().st_size
            except Exception as exception:
                print(f"       {name} failed: {exception!r}")
                failed += 1
        check(f"plots from the csv read back can be saved ({len(plots)} plots)", float(failed))

    hierarchy = runs.build_hierarchy(problem, 4, 3)
    spins = np.sort(np.concatenate([sub.original_nodes for sub in hierarchy.subproblems.values()]))
    covered = len(hierarchy.subproblems) == 4 and np.array_equal(spins, np.arange(SIZE))
    expected = sequential[(sequential["nb_partitions"] == 4) & (sequential["nb_node_per_meta_nodes"] == 3)].iloc[0]
    error = abs(hierarchy.upper_model.num_variables - expected["total_meta_nodes"]) if covered else float("inf")
    check("hierarchy of point (4, 3): 4 partitions covering every spin once, total_meta_nodes meta nodes", float(error))

    symmetric = reference_J + reference_J.T
    error = max(
        float(np.max(np.abs(sub.model.J + sub.model.J.T - symmetric[np.ix_(sub.original_nodes, sub.original_nodes)])))
        for sub in hierarchy.subproblems.values()
    )
    check("J of each partition vs J of the problem on its spins", error)

    upper = hierarchy.upper_model.J
    members = [np.flatnonzero(hierarchy.original_to_upper == node) for node in range(upper.shape[0])]
    blocks = {
        (a, b): symmetric[np.ix_(members[a], members[b])] for a in range(len(members)) for b in range(a + 1, len(members))
    }
    error = max(abs(upper[a, b] - block.mean()) for (a, b), block in blocks.items())
    check("upper J vs the mean of J between the spins of the two meta nodes", float(error), 1e-12)

    sizes = sorted({len(spins) for spins in members})
    print(f"       [reported] spins per meta node: {sizes}")
    print(f"       [reported] max |J| of the problem: {np.max(np.abs(symmetric)):.3g}")
    print(f"       [reported] max |upper J| (mean, as built): {np.max(np.abs(upper)):.3g}")
    print(f"       [reported] max |sum of J between two meta nodes|: {max(abs(block.sum()) for block in blocks.values()):.3g}")

    NB_SWEEPS = 4
    trace = runs.trace_sweeps(problem, 4, 3, nb_sweeps=NB_SWEEPS)
    sweeps = trace["sweeps"]
    complete = len(sweeps) == NB_SWEEPS and all(
        len(sweep["partition_J"]) == 4 and len(sweep["partition_h"]) == 4 for sweep in sweeps
    )
    check(f"trace_sweeps records {NB_SWEEPS} sweeps with 4 partitions each", 0.0 if complete else float("inf"))

    error = max(
        max(float(np.max(np.abs(a - b))) for a, b in zip([sweep["upper_J"], *sweep["partition_J"]], [sweeps[0]["upper_J"], *sweeps[0]["partition_J"]]))
        for sweep in sweeps
    )
    check("upper J and partition J at every sweep vs sweep 0", error)

    reference = utils.make_config(config, PROBLEM_TYPE, nb_runs=1, nb_sweeps_Hierarchical_solver=NB_SWEEPS)
    _, reference_ans = runs.run_single(runs.generate_problem(reference), 4, 3, return_ans=True)
    check("energy of the traced trial vs run_single", abs(trace["energy"] - float(reference_ans.energies[utils.HIERARCHICAL_SOLVER][0])))

    fields = np.array([np.concatenate(sweep["partition_h"]) for sweep in sweeps])
    print(f"       [reported] largest change of a partition field h between two sweeps: {np.max(np.abs(np.diff(fields, axis=0))):.3g}")

    with tempfile.TemporaryDirectory() as folder:
        failed = 0
        for name, plot in {"sweep_couplings": utils.plot_sweep_couplings, "sweep_fields": utils.plot_sweep_fields, "sweep_diff": utils.plot_sweep_diff}.items():
            try:
                failed += not utils.save_figure(plot(trace), name, Path(folder)).stat().st_size
            except Exception as exception:
                print(f"       {name} failed: {exception!r}")
                failed += 1
    check("the three sweep plots can be saved", float(failed))

    repo_J = runs.DummyCreatorStage.generate_dummy_maxcut(
        SIZE, round(config["dummy_precision"]), config["dummy_seed"], connectivity=1.0
    )["ising_model"].J
    check("J of the generator used here vs the repo generator, connectivity 1", float(np.max(np.abs(repo_J - reference_J))))

    def rows_of(frame, **values):
        for column, value in values.items():
            frame = frame[frame[column] == value]
        return frame.sort_values(["nb_partitions", "nb_node_per_meta_nodes"])[STATS].to_numpy()

    hierarchical_points = [point for point in GRID if point != (1, 0)]
    swept = runs.run_grid(problem, GRID, nb_sweeps=[1, 2], nb_workers=2)
    complete = len(swept) == 2 * len(hierarchical_points) + 1 and (swept["solver"] == utils.BASELINE_SOLVER).sum() == 1
    error = float(np.max(np.abs(rows_of(swept, nb_sweeps=2) - rows_of(sequential, nb_sweeps=2)))) if complete else float("inf")
    check("run_grid with nb_sweeps=[1, 2]: rows complete, 2 sweeps == the run without nb_sweeps", error)

    single = runs.run_grid(problem, GRID, nb_sweeps=2, nb_workers=2)
    error = float(np.nanmax(np.abs(single[STATS].to_numpy() - sequential[STATS].to_numpy()))) if len(single) == len(GRID) else float("inf")
    check("run_grid with the single value nb_sweeps=2 vs the run without nb_sweeps", error)

    study = runs.run_study(config, GRID, nb_sweeps=[1, 2], connectivity=[1.0, 0.5], nb_workers=2)
    complete = len(study) == 2 * len(swept) and (study["solver"] == utils.BASELINE_SOLVER).sum() == 2
    error = float(np.nanmax(np.abs(study[study["connectivity"] == 1.0][STATS].to_numpy() - swept[STATS].to_numpy()))) if complete else float("inf")
    check("run_study with connectivity=[1.0, 0.5]: rows complete, connectivity 1.0 == run_grid", error)
    measured = study[study["connectivity"] == 0.5]["connections_per_spin"].iloc[0]
    print(f"       [reported] connections per spin at connectivity 0.5: {measured:.2f} (0.5 * (num_spins - 1) = {0.5 * (SIZE - 1):.2f})")

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        config_path, csv_path = utils.save_run(config, study, folder)
        print(f"       name: {csv_path.stem}")
        saved_config, saved = utils.load_run(folder=folder)
        listed = (
            "_c1-0.5_" in csv_path.stem
            and "_sweeps1-2_" in csv_path.stem
            and saved_config["nb_sweeps_Hierarchical_solver"] == [1, 2]
            and saved_config["dummy_connectivity"] == [1.0, 0.5]
        )
        check("save_run on the study: sweeps and connectivities listed in the name and the yaml", 0.0 if listed else float("inf"))

        plots = {
            "partitions_by_sweeps": lambda: utils.plot_energy_lines(saved, x="nb_partitions", by="nb_sweeps", nb_node_per_meta_nodes=3, connectivity=1.0),
            "nodes_by_sweeps": lambda: utils.plot_energy_lines(saved, x="nb_node_per_meta_nodes", by="nb_sweeps", nb_partitions=2, connectivity=1.0),
            "partitions_by_connectivity": lambda: utils.plot_energy_lines(saved, x="nb_partitions", by="connectivity", relative=True, nb_node_per_meta_nodes=3, nb_sweeps=2),
            "nodes_by_connectivity": lambda: utils.plot_energy_lines(saved, x="nb_node_per_meta_nodes", by="connectivity", nb_partitions=2, nb_sweeps=2),
            "energy_3d": lambda: utils.plot_energy_3d(saved, nb_sweeps=2, connectivity=0.5),
            "heatmap": lambda: utils.plot_energy_heatmap(saved, nb_sweeps=1, connectivity=0.5),
        }
        failed = 0
        for name, plot in plots.items():
            try:
                failed += not utils.save_figure(plot(), name, folder).stat().st_size
            except Exception as exception:
                print(f"       {name} failed: {exception!r}")
                failed += 1
        try:
            utils.plot_energy_lines(saved, x="nb_partitions", by="nb_sweeps", connectivity=1.0)
            failed += 1
            print("       a plot without the choice of nb_node_per_meta_nodes was not refused")
        except ValueError:
            pass
    check(f"energy plots of the study ({len(plots)} plots) and refusal of a missing choice", float(failed))

    error = 0.0
    for point in [(4, 3), (2, 1)]:
        kept = runs.run_single_sweeps(problem, *point, nb_sweeps=[1, 2, 4])
        for row in kept:
            separate = runs.run_single(problem, *point, nb_sweeps=row["nb_sweeps"])
            error = max(error, max(abs(row[stat] - separate[stat]) for stat in STATS))
    check("energies kept after 1, 2, 4 sweeps of one run vs separate runs (2 points)", float(error))

    print(f"{sum(results)}/{len(results)} passed")
