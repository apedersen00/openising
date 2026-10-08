"""Test 01: split of run_point into generate_problem + run_point.

What is tested: fully_connected_test_multi_proc.py now builds the problem once
(generate_problem) and each grid point solves a deep copy of it (run_point), instead of
writing a config file per point and calling api.get_hamiltonian_energy.

Why: the split must only save generation time. For the same grid point, the energies of
all trials have to be identical to the ones of the old one-call-per-point path.

Checks (consistency, PASS/FAIL, tolerance 0 on the per-trial energies):
  1-3. energies of the new path == energies of api.get_hamiltonian_energy, for three points
  4.   the generated problem is not modified by solving a point (it is reused by the next)
  5.   the generated problem can be pickled (it is sent to every worker)

Run from the repo top: python no_backup/Multi_core_architecture/01_applications/test_01_split_run_point.py
"""

import pickle
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import fully_connected_test_multi_proc as script  # noqa: E402
from ising import api  # noqa: E402

POINTS = [(1, 0), (2, 1), (4, 3)]
TOLERANCE = 0.0

results = []


def check(name: str, error: float, tolerance: float = TOLERANCE) -> None:
    passed = error <= tolerance
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: error = {error:.3e} (tolerance {tolerance:.1e})")


def old_run_point(point: tuple[int, int]) -> np.ndarray:
    """The path before the split: one config file and one full pipeline run per point."""
    nb_cores, nb_meta_nodes = point
    config_new = deepcopy(script.config)
    config_new["nb_cores"] = nb_cores
    if nb_cores == 1 and nb_meta_nodes == 0:
        config_new["solvers"] = ["Multiplicative"]
    else:
        config_new["solvers"] = ["Hierarchical_solver"]
    config_new["nb_meta_nodes"] = nb_meta_nodes
    config_new["nb_partitions"] = nb_cores
    with tempfile.TemporaryDirectory() as folder:
        config_file = Path(folder) / "point.yaml"
        with config_file.open("w") as file:
            yaml.safe_dump(config_new, file)
        ans, _ = api.get_hamiltonian_energy(script.PROBLEM_TYPE, str(config_file), script.LOGGING_LEVEL)
    return np.asarray(ans.energies[config_new["solvers"][0]], dtype=float)


if __name__ == "__main__":
    script.RESULTS_CSV = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    problem = script.generate_problem()
    reference_J = problem["ising_model"].J.copy()
    script.init_worker(0, problem)

    for point in POINTS:
        solver = "Multiplicative" if point == (1, 0) else "Hierarchical_solver"
        new = np.asarray(script.run_point(point).energies[solver], dtype=float)
        old = old_run_point(point)
        error = float(np.max(np.abs(new - old))) if new.shape == old.shape else float("inf")
        check(f"energies of point {point} ({new.size} trials, new vs old path)", error)

    unchanged = isinstance(problem["config"].nb_partitions, list) and problem["config"].solvers == script.config["solvers"]
    error = float(np.max(np.abs(problem["ising_model"].J - reference_J))) if unchanged else float("inf")
    check("generated problem unchanged after solving the points", error)

    try:
        pickle.loads(pickle.dumps(problem))
        error = 0.0
    except Exception as exception:
        print(f"       pickling failed: {exception}")
        error = float("inf")
    check("generated problem can be pickled", error)

    print(f"{sum(results)}/{len(results)} passed")
