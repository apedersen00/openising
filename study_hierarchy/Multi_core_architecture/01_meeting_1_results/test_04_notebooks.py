"""Test 04: the notebooks still run from the first to the last cell, in dummy mode and on a benchmark.

What is tested: the code cells of 00_exploration_template.ipynb and of 01_max_cut_fixed_systems.ipynb,
executed in order as the notebook would, with a smaller configuration and the runs saved in a
temporary folder.

Why: the benchmark selection changed the configuration cell, the rows (benchmark, best_energy) and
the plots. The template was written before it and must still work as it is; 01 must work with
BENCHMARK = None (dummy mode) as it did before, and with a benchmark file.

Only the lines `NAME = ...` of the configuration cell are replaced (see SMALL below), also when they
are commented out (SIZE and CONNECTIVITY are, when the notebook is set to a benchmark) or when the
value is a list on several lines; every other line of the notebooks is run unchanged.

Checks (PASS/FAIL, the error is the number of cells that raised):
  1. 00_exploration_template.ipynb, as it is (dummy creator), 64 spins
  2. 01_max_cut_fixed_systems.ipynb with BENCHMARK = None (dummy mode), 64 spins:
     the rows have no benchmark and no best energy, the run is saved under its connectivities
  3. 01_max_cut_fixed_systems.ipynb with the BENCHMARK of the notebook, on a smaller grid:
     the rows have that benchmark and its best reported energy
  4. the BENCHMARK written in 01_max_cut_fixed_systems.ipynb is a file of ising/benchmarks/
  5. as 3, with a copy of the yaml where partitioning_technique is "random": tells a failure of the
     notebook apart from a failure of the partitioning of the yaml on that benchmark

Run from the repo top: python no_backup/Multi_core_architecture/01_applications/test_04_notebooks.py
"""

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FOLDER = Path(__file__).parent
sys.path.insert(0, str(FOLDER))
import multi_core_utils as utils  # noqa: E402

# Smaller than the notebooks: a few seconds in dummy mode, a few minutes on a 2000-spin benchmark
SMALL_DUMMY = {
    "SIZE": "64",
    "NB_RUNS": "2",
    "CONNECTIVITY": "[1.0, 0.5]",
    # The grids of the notebooks are made for their size: these are the ones of 64 spins
    "NB_PARTITIONS": "[1, 2, 4, 8]",
    "NB_NODE_PER_META_NODES": "[1, 2, 4]",
    "NB_SWEEPS": "[2, 3]",
    # The louvain extension of 00: against the run the notebook has just saved, with fewer splits
    "COMPARE_RUN": '"last_run"',
    "LOUVAIN_DRAWS": "2",
}
SMALL_BENCHMARK = {"NB_RUNS": "1", "NB_PARTITIONS": "[2, 4]", "NB_NODE_PER_META_NODES": "[2, 4]", "NB_SWEEPS": "[2, 3]"}

results = []


def check(name: str, error: float, tolerance: float = 0.0) -> None:
    passed = error <= tolerance
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: error = {error:.3e} (tolerance {tolerance:.1e})")


def run_notebook(name: str, replaced: dict, folder: Path) -> tuple[int, dict]:
    """Execute the code cells of a notebook in order, with some `NAME = ...` lines replaced.

    @return: the number of cells that raised, and the variables of the notebook at the end
    """
    replaced = dict(replaced, RESULTS_DIR=f"Path({str(folder)!r})")
    text = (FOLDER / name).read_text()
    cells = json.loads(text)["cells"]
    # A saved run the notebook reads by its name is read from the temporary folder: copy it there
    for run in set(re.findall(r'load_run\(\s*(?:name=)?\\"([^"\\]+)\\"', text)):
        for suffix in (".yaml", ".csv"):
            if (utils.RESULTS_DIR / f"{run}{suffix}").is_file():
                shutil.copy(utils.RESULTS_DIR / f"{run}{suffix}", folder / f"{run}{suffix}")
    variables, failed = {"Path": Path}, 0
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # The magics of the first cell are for the notebook only
        source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
        for variable, value in replaced.items():
            source = re.sub(rf"^#?{variable} = (\[[^\]]*\]|.*)$", f"{variable} = {value}", source, flags=re.MULTILINE)
        try:
            exec(compile(source, f"{name}[{index}]", "exec"), variables)
        except Exception as exception:
            print(f"       cell {index} of {name} failed: {exception!r}")
            failed += 1
        plt.close("all")
    return failed, variables


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        failed, variables = run_notebook("00_exploration_template.ipynb", SMALL_DUMMY, Path(folder))
        check("00_exploration_template.ipynb runs to the end (dummy creator)", float(failed))

    with tempfile.TemporaryDirectory() as folder:
        failed, variables = run_notebook("01_max_cut_fixed_systems.ipynb", dict(SMALL_DUMMY, BENCHMARK="None"), Path(folder))
        frame = variables.get("frame")
        dummy = (
            frame is not None
            and frame["benchmark"].isna().all()
            and frame["best_energy"].isna().all()
            and sorted(frame["connectivity"].unique()) == [0.5, 1.0]
            and "_N64_c1-0.5_" in variables["csv_path"].stem
            and variables["saved_config"]["dummy_creator"] is True
        )
        check("01_max_cut_fixed_systems.ipynb runs to the end with BENCHMARK = None (dummy mode)", float(failed) if dummy else float("inf"))

    with tempfile.TemporaryDirectory() as folder:
        failed, variables = run_notebook("01_max_cut_fixed_systems.ipynb", SMALL_BENCHMARK, Path(folder))
        frame, benchmark = variables.get("frame"), variables.get("BENCHMARK")
        selected = (
            frame is not None
            and list(frame["benchmark"].unique()) == utils.as_list(benchmark)
            and (frame["best_energy"] == frame["benchmark"].map(utils.best_reported_energy)).all()
            and variables["saved_config"]["dummy_creator"] is False
        )
        if frame is not None:
            print(f"       name: {variables['csv_path'].stem}")
            for name, rows in frame.groupby("benchmark", sort=False):
                print(f"       [reported] {Path(name).stem}: best reported energy {rows['best_energy'].iloc[0]:.5g}, lowest energy of the run {rows['en_min'].min():.5g}")
        check(f"01_max_cut_fixed_systems.ipynb runs to the end with BENCHMARK = {benchmark!r}", float(failed) if selected else float("inf"))

        exists = all((utils.REPO_TOP / utils.benchmark_path(name)).is_file() for name in utils.as_list(benchmark))
        check("the BENCHMARK of the notebook is a file of ising/benchmarks/", 0.0 if exists else float("inf"))

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        config = utils.load_config("maxcut_benchmark")
        with (folder / "random.yaml").open("w") as file:
            yaml.safe_dump(dict(config, partitioning_technique="random"), file)
        replaced = dict(SMALL_BENCHMARK, BASE_CONFIG=f"Path({str(folder / 'random.yaml')!r})")
        failed, variables = run_notebook("01_max_cut_fixed_systems.ipynb", replaced, folder)
        frame = variables.get("frame")
        selected = (
            frame is not None
            and list(frame["benchmark"].unique()) == utils.as_list(benchmark)
            and (frame["best_energy"] == frame["benchmark"].map(utils.best_reported_energy)).all()
        )
        if frame is not None:
            print(f"       name: {variables['csv_path'].stem}")
            for name, rows in frame.groupby("benchmark", sort=False):
                print(f"       [reported] {Path(name).stem}: best reported energy {rows['best_energy'].iloc[0]:.5g}, lowest energy of the run {rows['en_min'].min():.5g}")
        check("01_max_cut_fixed_systems.ipynb runs to the end on its BENCHMARK with random partitioning", float(failed) if selected else float("inf"))

    print(f"{sum(results)}/{len(results)} passed")
