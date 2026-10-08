"""Test 06: the number of partitions is the one the partitioner returns.

What is tested: HierarchicalSolver.solve (ising/solvers/Hierarchical_solver.py). nb_partitions is
what is asked of the partitioner; a community algorithm such as louvain may return another number.
After the partitioning, nb_partitions is now set to the number of partitions actually returned, and
this number is the one of the log (the per-partition columns and the metadata).

Why: the per-partition columns of the log were sized with the requested number, so a run that is
logged failed as soon as the partitioner returned another number of partitions.

The partitioner of checks 1-4 is a stand-in put in place of louvain: it returns the three cliques of
the problem whatever number is asked. The louvain of the repo is not run, as it needs a networkx
with louvain_communities(max_level=...) and the one installed here (2.8.8) does not have it.

Checks (PASS/FAIL, the error is the number of violations, tolerance 0):
  1. premise: on three cliques of 6 spins, the stand-in asked for 2 partitions returns 3
  2. that problem, solved with the stand-in as "louvain" and a log file: the solve ends, with 3
     subproblems
  3. the log of that run has 3 values per sweep in each per-partition column
  4. the metadata of that log gives nb_partitions = 3
  5. a partitioner that returns the requested number (random, 2 partitions): 2 subproblems, 2 values
     per sweep and nb_partitions = 2 in the log, as before

Run from the repo top: python no_backup/Multi_core_architecture/01_applications/test_06_partitioner_sets_nb_partitions.py
"""

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import multi_core_runs  # noqa: E402, F401  (sets TOP and the path to the repo)
from ising.solvers import Hierarchical_solver  # noqa: E402
from ising.solvers.Hierarchical_solver import HierarchicalSolver  # noqa: E402
from ising.stages.model.ising import IsingModel  # noqa: E402

TOLERANCE = 0.0
PARTITION_COLUMNS = ["energy_partitions", "time_partitions", "operations_partitions", "operations_per_iteration"]
NB_SWEEPS = 2

results = []


def check(name: str, error: float, tolerance: float = TOLERANCE) -> None:
    passed = error <= tolerance
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: error = {error:.3e} (tolerance {tolerance:.1e})")


def three_cliques() -> IsingModel:
    coupling = np.zeros((18, 18))
    for start in (0, 6, 12):
        coupling[start : start + 6, start : start + 6] = 1.0
    return IsingModel(np.triu(coupling, k=1), np.zeros(18))


def cliques_partitioning(model: IsingModel, nb_cores: int) -> np.ndarray:
    """A partitioner that returns the three cliques, whatever number of partitions is asked."""
    return np.repeat(np.arange(3), 6)


def keep_state(model: IsingModel, initial_state: np.ndarray, **_) -> tuple:
    """A core solver that returns its initial state: state, energy, time, operations, iterations."""
    return initial_state.copy(), model.evaluate(initial_state), 0.0, 1, 1


def logged_run(technique: str, nb_partitions: int, logfile: Path) -> HierarchicalSolver:
    solver = HierarchicalSolver()
    solver.solve(
        model=three_cliques(),
        initial_state=np.random.default_rng(0).choice([-1, 1], 18),
        nb_sweeps=NB_SWEEPS,
        core_solver=keep_state,
        partitioning_technique=technique,
        nb_partitions=nb_partitions,
        nb_meta_nodes=6,
        core_solver_args={},
        file_multi_core=logfile,
    )
    return solver


def log_violations(logfile: Path, nb_partitions: int) -> tuple[int, int]:
    """Columns that do not have nb_partitions values per sweep, and a wrong nb_partitions in the metadata."""
    with h5py.File(logfile, "r") as log:
        shapes = {column: log[column].shape for column in PARTITION_COLUMNS}
        logged = log.attrs.get("nb_partitions")
    print(f"       shapes {shapes}, nb_partitions in the metadata: {logged}")
    wrong_columns = sum(shape != (NB_SWEEPS, nb_partitions) for shape in shapes.values())
    return wrong_columns, int(logged != nb_partitions)


if __name__ == "__main__":
    Hierarchical_solver.make_louvain_partitioning = cliques_partitioning
    returned = len(np.unique(cliques_partitioning(three_cliques(), 2)))
    check(f"premise: the stand-in asked for 2 partitions returns 3 (returned {returned})", abs(returned - 3))

    with tempfile.TemporaryDirectory() as folder:
        logfile = Path(folder) / "louvain.h5"
        try:
            solver = logged_run("louvain", 2, logfile)
            error = abs(len(solver.subproblems) - 3)
        except Exception as exception:
            print(f"       solve failed: {type(exception).__name__}: {exception}")
            error = float("inf")
        check("stand-in as louvain, 2 partitions asked, logged run: solved with 3 subproblems", error)
        wrong_columns, wrong_metadata = log_violations(logfile, 3) if error == 0 else (float("inf"), float("inf"))
        check("stand-in: 3 values per sweep in each per-partition column of the log", wrong_columns)
        check("stand-in: nb_partitions = 3 in the metadata of the log", wrong_metadata)

        logfile = Path(folder) / "random.h5"
        try:
            solver = logged_run("random", 2, logfile)
            error = abs(len(solver.subproblems) - 2) + sum(log_violations(logfile, 2))
        except Exception as exception:
            print(f"       solve failed: {type(exception).__name__}: {exception}")
            error = float("inf")
        check("random, 2 partitions: 2 subproblems, 2 values per sweep and nb_partitions = 2", error)

    print(f"{sum(results)}/{len(results)} passed")
