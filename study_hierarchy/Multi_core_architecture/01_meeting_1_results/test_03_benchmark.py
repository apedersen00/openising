"""Test 03: a benchmark file of ising/benchmarks/ in place of the generated (dummy) problem.

What is tested: set_benchmark / benchmark_path, generate_problem and run_single on a benchmark file,
run_study with a list of benchmark files, and the saved run and plots of such a study.

Why: the runs were only made on problems of the dummy creator. Selecting a benchmark must give the
model of the file, the same trials as the repo pipeline on that file, and several benchmarks of the
same size and connectivity (G32, G33) must stay apart in the table, the saved run and the plots.

Checks (consistency, PASS/FAIL, tolerance 0 unless stated):
  1.  set_benchmark: dummy_creator off and benchmark is the path of the file; an unknown file is refused;
      None leaves the config as it is
  2.  generate_problem on G/G32.txt: 2000 spins, and J + J.T == -weight / 2 on every edge of the file, 0 elsewhere
  3.  the row of run_single: benchmark == "G/G32.txt", connectivity == connections_per_spin / (num_spins - 1)
  4.  MIN / 25 / 50 / 75 / MAX of run_single == those of the api.get_hamiltonian_energy energies on the file
  5.  run_study with benchmark=["G/G32.txt", "G/G33.txt"]: every point for each file, one single-core row per file;
      its rows of G32 == run_grid on the problem of G32
  6.  run_study without benchmark on a config set to G/G32.txt == those rows of G32
  7.  save_run on that study: G32-G33 in the name, both paths in the yaml, dummy_connectivity left as it was
  8.  load_run gives back the benchmark column; a csv without that column is read with an empty one
  9.  plots of the study with benchmark="G/G32.txt": saved, and their single-core level is the one of G32;
      a plot without the choice of the benchmark is refused
  10. a generated (dummy) problem still has an empty benchmark and its connectivity in the name of the run
  11. find_benchmarks("G/regenerated/G32_*.txt"): one file; its problem has the J of G/G32.txt; no match is refused
  12. the yaml maxcut_benchmark (dummy_creator off, dummy parameters commented out) gives the same model as
      maxcut_c1 set to that file, and the same row for a point
  13. best_energy of a row: minus the cut of the name for the regenerated G32 (-1410), the value of
      optimal_energy.txt for G/G32.txt (-1410), empty for a generated problem
  14. the energy plots of a run on the regenerated G32 have the best reported energy: a line at -1410 in each
      panel of the lines plot, at (-1410 - single core) / |single core| in the relative one, in the legend of
      the 3D plot and in the title of the heatmap; a run without best energy has none

Run from the repo top: python no_backup/Multi_core_architecture/01_applications/test_03_benchmark.py
"""

import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import multi_core_runs as runs  # noqa: E402
import multi_core_utils as utils  # noqa: E402
from ising import api  # noqa: E402

PROBLEM_TYPE = "Maxcut"
BENCHMARK = "G/G32.txt"
OTHER = "G/G33.txt"
POINT = (2, 2)
GRID = [(1, 0), POINT]
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
    base = utils.make_config(utils.load_config("maxcut_c1"), PROBLEM_TYPE, nb_runs=2)
    config = utils.set_benchmark(base, BENCHMARK)
    path = utils.REPO_TOP / config["benchmark"]
    selected = config["dummy_creator"] is False and path.is_file() and utils.benchmark_name(config["benchmark"]) == BENCHMARK
    try:
        utils.set_benchmark(base, "G/G0.txt")
        refused = False
    except FileNotFoundError:
        refused = True
    check("set_benchmark: dummy creator off, path of the file, unknown file refused, None changes nothing",
          0.0 if selected and refused and utils.set_benchmark(base, None) == base else float("inf"))

    problem = runs.generate_problem(config)
    model = problem["ising_model"]
    lines = path.read_text().splitlines()
    num_spins = int(lines[0].split()[0])
    expected = np.zeros((num_spins, num_spins))
    for line in lines[1:]:
        u, v, weight = line.split()
        expected[int(u) - 1, int(v) - 1] = expected[int(v) - 1, int(u) - 1] = -float(weight) / 2
    error = float(np.max(np.abs(model.J + model.J.T - expected))) if model.num_variables == num_spins == 2000 else float("inf")
    check(f"J of the generated problem vs the {len(lines) - 1} edges of {BENCHMARK}", error)
    print(f"       [reported] {num_spins} spins, {utils.connections_per_spin(model):.1f} connections per spin")

    row = runs.run_single(problem, *POINT)
    error = abs(row["connectivity"] - row["connections_per_spin"] / (num_spins - 1)) if row["benchmark"] == BENCHMARK else float("inf")
    check("row of run_single: name of the benchmark, connectivity measured on its J", float(error))

    reference = np.percentile(api_energies(config, POINT), [0, 25, 50, 75, 100])
    error = float(np.max(np.abs(np.array([row[stat] for stat in STATS]) - reference)))
    check(f"MIN/25/50/75/MAX of point {POINT} on {BENCHMARK} ({row['nb_trials']} trials, run_single vs api)", error)

    grid_rows = runs.run_grid(problem, GRID, nb_workers=2)
    study = runs.run_study(base, GRID, benchmark=[BENCHMARK, OTHER], nb_workers=2)
    of_benchmark = study[study["benchmark"] == BENCHMARK]
    complete = (
        len(study) == 2 * len(GRID)
        and sorted(study["benchmark"].unique()) == [BENCHMARK, OTHER]
        and (study.groupby("benchmark")["solver"].apply(lambda solvers: (solvers == utils.BASELINE_SOLVER).sum()) == 1).all()
    )
    error = float(np.max(np.abs(of_benchmark[STATS].to_numpy() - grid_rows[STATS].to_numpy()))) if complete else float("inf")
    check("run_study with two benchmarks: rows complete, rows of G32 == run_grid on G32", error)
    differ = float(np.max(np.abs(study[study["benchmark"] == OTHER][STATS].to_numpy() - of_benchmark[STATS].to_numpy())))
    print(f"       [reported] largest difference between the energies of G33 and of G32: {differ:.3g}")

    from_config = runs.run_study(config, GRID, nb_workers=2)
    same = len(from_config) == len(GRID) and (from_config["benchmark"] == BENCHMARK).all()
    error = float(np.max(np.abs(from_config[STATS].to_numpy() - of_benchmark[STATS].to_numpy()))) if same else float("inf")
    check("run_study without benchmark on a config set to G32 vs the rows of G32", error)

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        config_path, csv_path = utils.save_run(base, study, folder)
        print(f"       name: {csv_path.stem}")
        saved_config, saved = utils.load_run(folder=folder)
        listed = (
            csv_path.stem.startswith("Maxcut_N2000_G32-G33_p")
            and saved_config["benchmark"] == [utils.benchmark_path(BENCHMARK), utils.benchmark_path(OTHER)]
            and saved_config["dummy_connectivity"] == base["dummy_connectivity"]
        )
        check("save_run on the study: both benchmarks in the name and the yaml", 0.0 if listed else float("inf"))

        study.drop(columns="benchmark").to_csv(folder / "old.csv", index=False)
        (folder / "old.yaml").write_text(config_path.read_text())
        _, old = utils.load_run("old", folder)
        kept = list(saved["benchmark"]) == list(study["benchmark"]) and old["benchmark"].isna().all()
        check("load_run: benchmark column read back, empty for a csv without it", 0.0 if kept else float("inf"))

        level = of_benchmark[of_benchmark["solver"] == utils.BASELINE_SOLVER]["en_50"].iloc[0]
        failed = 0
        plots = {
            "lines": lambda: utils.plot_energy_lines(saved, x="nb_partitions", by="nb_node_per_meta_nodes", benchmark=BENCHMARK),
            "relative": lambda: utils.plot_energy_lines(saved, x="nb_partitions", by="nb_sweeps", relative=True, benchmark=BENCHMARK),
            "energy_3d": lambda: utils.plot_energy_3d(saved, benchmark=BENCHMARK),
            "heatmap": lambda: utils.plot_energy_heatmap(saved, benchmark=BENCHMARK),
        }
        for name, plot in plots.items():
            try:
                failed += not utils.save_figure(plot(), name, folder).stat().st_size
            except Exception as exception:
                print(f"       {name} failed: {exception!r}")
                failed += 1
        _, baseline, description = utils._select(saved, {"benchmark": BENCHMARK}, ["nb_partitions", "nb_node_per_meta_nodes"])
        failed += not (len(baseline) == 1 and baseline["en_50"].iloc[0] == level and f"benchmark = {BENCHMARK}" in description)
        try:
            utils.plot_energy_heatmap(saved)
            failed += 1
            print("       a plot without the choice of the benchmark was not refused")
        except ValueError:
            pass
    check(f"plots of one benchmark of the study ({len(plots)} plots), its single-core level, refusal of a missing choice", float(failed))

    dummy = utils.make_config(base, PROBLEM_TYPE, dummy_size=24)
    frame = runs.run_grid(runs.generate_problem(dummy), GRID, nb_workers=1)
    unchanged = frame["benchmark"].isna().all() and (frame["connectivity"] == dummy["dummy_connectivity"]).all()
    unchanged = unchanged and utils.run_name(dummy, frame).startswith("Maxcut_N24_c1_p2_")
    check("generated problem: empty benchmark, connectivity of the config, connectivity in the name", 0.0 if unchanged else float("inf"))

    found = utils.find_benchmarks("G/regenerated/G32_*.txt")
    print(f"       [reported] found: {found}")
    try:
        utils.find_benchmarks("G/regenerated/G0_*.txt")
        refused = False
    except FileNotFoundError:
        refused = True
    regenerated = runs.generate_problem(utils.set_benchmark(base, found[0]))["ising_model"]
    error = float(np.max(np.abs(regenerated.J - model.J))) if len(found) == 1 and refused else float("inf")
    check("find_benchmarks on the regenerated G32: one file, J of G/G32.txt, no match refused", error)

    from_yaml = utils.make_config(utils.load_config("maxcut_benchmark"), PROBLEM_TYPE, nb_runs=2)
    yaml_problem = runs.generate_problem(from_yaml)
    yaml_frame = runs.run_grid(yaml_problem, GRID, nb_workers=2)
    yaml_row, same_file_row = yaml_frame.iloc[1], runs.run_single(runs.generate_problem(utils.set_benchmark(base, found[0])), *POINT)
    commented = from_yaml["dummy_creator"] is False and not [key for key in from_yaml if key.startswith("dummy_") and key != "dummy_creator"]
    error = max(
        float(np.max(np.abs(yaml_problem["ising_model"].J - regenerated.J))),
        max(abs(yaml_row[stat] - same_file_row[stat]) for stat in STATS),
    ) if commented and yaml_row["benchmark"] == found[0] else float("inf")
    check("yaml maxcut_benchmark vs maxcut_c1 set to the same file: J and energies of a point", float(error))

    error = max(abs(yaml_row["best_energy"] + 1410), abs(row["best_energy"] + 1410))
    check("best_energy: regenerated G32 (name) and G/G32.txt (optimal_energy.txt) vs -1410",
          float(error) if np.isnan(frame["best_energy"]).all() else float("inf"))

    def best_lines(fig):
        return [[line.get_ydata()[0] for line in ax.get_lines() if str(line.get_label()).startswith("best reported")] for ax in fig.axes]

    single = yaml_frame[yaml_frame["solver"] == utils.BASELINE_SOLVER].iloc[0]
    absolute = best_lines(utils.plot_energy_lines(yaml_frame, x="nb_partitions", by="nb_node_per_meta_nodes"))
    relative = best_lines(utils.plot_energy_lines(yaml_frame, x="nb_partitions", by="nb_node_per_meta_nodes", relative=True))
    expected = [[(-1410 - single[stat]) / abs(single[stat])] for stat in STATS]
    error = float(np.max(np.abs(np.array(relative) - np.array(expected)))) if absolute == [[-1410.0]] * len(STATS) else float("inf")
    legend = [text.get_text() for text in utils.plot_energy_3d(yaml_frame).axes[0].get_legend().get_texts()]
    title = utils.plot_energy_heatmap(yaml_frame).axes[0].get_title()
    shown = "best reported energy (-1410)" in legend and "best reported energy: -1410" in title
    none = not any(best_lines(utils.plot_energy_lines(frame, x="nb_partitions", by="nb_node_per_meta_nodes")))
    check("best reported energy in the lines, relative lines, 3D and heatmap plots; none without it",
          error if shown and none else float("inf"), 1e-12)
    print(f"       [reported] single core 50 %: {single['en_50']:.5g}, best reported: {yaml_row['best_energy']:.5g}")

    print(f"{sum(results)}/{len(results)} passed")
