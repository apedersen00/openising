"""Runs for the multi-core exploration: generate a problem once, solve it for one or many points.

    config = make_config(load_config("maxcut_c1"), "Maxcut", dummy_size=1024)
    problem = generate_problem(config)                    # once per problem
    row = run_single(problem, 4, 2)                       # one point, in this process
    hierarchy = build_hierarchy(problem, 4, 2)            # the partitions and meta nodes of that point
    trace = trace_sweeps(problem, 4, 2, nb_sweeps=4)      # what each sweep of the first trial solves
    frame = run_grid(problem, grid, nb_sweeps=[1, 2])     # many points, in parallel
    frame = run_study(config, grid, nb_sweeps=[1, 2], connectivity=[1.0, 0.5])   # one problem per connectivity
    frame = run_study(config, grid, benchmark=["G/G32.txt", "G/G33.txt"])        # one problem per benchmark file

nb_sweeps, connectivity and benchmark take one value or a list of values. A benchmark is a file of
ising/benchmarks/; set_benchmark(config, "G/G32.txt") makes a config solve it instead of a generated problem.

A point is (nb_partitions, nb_node_per_meta_nodes). The point (1, 0) is the single-core baseline: it is
solved with the plain Multiplicative solver instead of the hierarchical one.

nb_node_per_meta_nodes is what the pipeline config calls nb_meta_nodes. The pipeline turns it into
num_spins // nb_node_per_meta_nodes meta
nodes in total (ising/utils/flow.py), shared between the partitions: the rows of a run give
these derived numbers next to the config value.

A point that cannot be solved (more partitions than meta nodes, a number of partitions the
partitioner refuses, ...) does not stop anything. Its problem, the error and where it was raised
are printed, and the next point is solved: run_single, run_grid and run_study give its row with no
trial, no energy (NaN), the reason in the column "error" and the place in "error_where";
build_hierarchy, trace_sweeps and make_partitioning return None. The plots leave these points out.

run_study(..., live_folder=RESULTS_DIR) saves the rows so far as live_run.yaml / live_run.csv each
time a point is done: load_run("live_run") plots a study that is still going, from another kernel.
"""

import os
import sys
from pathlib import Path

# One BLAS thread per process: the grid points themselves run in parallel
for _variable in ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_variable, "1")
# Silence the per-trial progress bars: parallel workers overwrite each other's bar.
# Set TQDM_DISABLE=0 before importing this module to get them back for single runs.
os.environ.setdefault("TQDM_DISABLE", "1")
# The ising package needs the repo top, which a notebook kernel does not always have
_REPO_TOP = Path(__file__).resolve().parents[3]
os.environ.setdefault("TOP", str(_REPO_TOP))
if str(_REPO_TOP) not in sys.path:
    sys.path.insert(0, str(_REPO_TOP))

import logging  # noqa: E402
import traceback  # noqa: E402
from argparse import Namespace  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from copy import deepcopy  # noqa: E402
from typing import Any  # noqa: E402

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tqdm  # noqa: E402

from ising import api  # noqa: E402
from ising.stages.stage import Stage  # noqa: E402
from ising.stages.model.ising import IsingModel  # noqa: E402
from ising.stages.dummy_creator_stage import DummyCreatorStage  # noqa: E402
from ising.stages.quantization_stage import QuantizationStage  # noqa: E402
from ising.stages.combine_nodes_stage import CombineNodesStage  # noqa: E402
from ising.stages.mismatch_stage import MismatchStage  # noqa: E402
from ising.stages.simulation_stage import SimulationStage  # noqa: E402
from ising.stages.initialization_stage import InitializationStage  # noqa: E402
from ising.solvers.Hierarchical_solver import HierarchicalSolver  # noqa: E402
from ising.utils.partitioning_schemes import (  # noqa: E402
    make_louvain_partitioning,
    make_modularity_partitioning,
    make_random_partitioning,
    make_greedy_partitioning,
    make_spectral_partitioning,
)

from multi_core_utils import (  # noqa: E402
    BASELINE_SOLVER,
    ENERGY_STATS,
    HIERARCHICAL_SOLVER,
    LIVE_RUN,
    as_list,
    benchmark_name,
    best_reported_energy,
    connections_per_spin,
    save_run,
    set_benchmark,
)

LOGGING_LEVEL = logging.WARNING
LOGGING_FORMAT = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT, stream=sys.stdout)

# Same stage choice as api.get_hamiltonian_energy
PARSER_STAGES = {
    "Maxcut": api.MaxcutParserStage,
    "TSP": api.TSPParserStage,
    "ATSP": api.ATSPParserStage,
    "MIMO": api.MIMOParserStage,
    "QKP": api.QKPParserStage,
    "Biqmac": api.BiqMacParserStage,
    "MPPI": api.MPPIParserStage,
}
ENERGY_CALC_STAGES = {
    "TSP": api.TSPEnergyCalcStage,
    "ATSP": api.TSPEnergyCalcStage,
    "MIMO": api.MIMOBerCalcStage,
}

# Same partitioners as HierarchicalSolver.solve
PARTITIONERS = {
    "random": make_random_partitioning,
    "modularity": make_modularity_partitioning,
    "spectral": make_spectral_partitioning,
    "greedy": make_greedy_partitioning,
    "louvain": make_louvain_partitioning,
}

# Partitioners that choose the number of partitions themselves: the number asked is not used
FREE_PARTITIONERS = {"louvain"}

# The problem of the current grid, set once per worker by _init_worker
_PROBLEM = None


class ProblemCaptureStage(Stage):
    """Ends the generation half of the pipeline: hands back what the parser stages built."""

    def run(self):
        yield self.kwargs


class _DummyCreator(DummyCreatorStage):
    """DummyCreatorStage with a Maxcut generator that also works for a connectivity below 1.

    The repo one stops on `p[ind] = 1.0 - connectivity` (ind is an array, p a list). This is a copy
    with that one line changed to p[ind[0]]; it draws the same J as the repo one at connectivity 1.
    To be removed once the line is fixed in ising/stages/dummy_creator_stage.py.
    """

    @staticmethod
    def generate_dummy_maxcut(N: int, dummy_bits: int = 2, seed: int = 0, connectivity: float = 1.0) -> dict:
        np.random.seed(seed)
        name = f"DummyMaxCut_N{N}_seed{seed}"
        values = np.arange(int(-(2 ** (dummy_bits - 1) - 1)), int(2 ** (dummy_bits - 1)))
        ind = np.flatnonzero(values == 0)
        if connectivity == 1.0:
            values = np.delete(values, ind)
            p = [1 / len(values) for _ in values]
        else:
            p = [connectivity / (len(values) - 1) for _ in values]
            p[ind[0]] = 1.0 - connectivity
        J = np.random.choice(values, (N, N), p=p)

        graph = nx.Graph(name=name)
        graph.add_nodes_from(range(1, N + 1))  # Nodes are 1-indexed in the graph
        for i in range(N):
            for j in range(i + 1, N):
                if J[i, j] != 0:
                    graph.add_edge(i + 1, j + 1, weight=-J[i, j] * 2)

        J = np.triu(J, k=1)
        ising_model = IsingModel(J, np.zeros((N,)), np.sum(J), name=name)
        return {"ising_model": ising_model, "graph": graph, "N": N, "seed": seed}


def generate_problem(config: dict) -> dict:
    """Build the Ising model of a config once, to be solved for as many points as wanted.

    @param config: the config as a dict, with its problem_type (see multi_core_utils.make_config)
    @return: what the pipeline built so far: config, ising_model, nx_graph, best_found, ...
    """
    problem_type = config["problem_type"]
    if problem_type not in PARSER_STAGES:
        raise NotImplementedError(f"Parser for {problem_type} is not implemented.")
    swept = [
        key
        for key in ("benchmark", "dummy_connectivity", "nb_sweeps_Hierarchical_solver")
        if isinstance(config.get(key), list)
    ]
    if swept:
        raise ValueError(f"{swept} hold several values: one problem needs one of each, pick it with make_config.")
    # What ConfigParserStage does, without going through a yaml file
    namespace = Namespace(**deepcopy(config))
    stages = [PARSER_STAGES[problem_type], ProblemCaptureStage]
    return next(_DummyCreator(stages, config=namespace).run())


def solver_of(nb_partitions: int, nb_node_per_meta_nodes: int) -> str:
    """No partition and no meta node: plain Multiplicative solver, not the hierarchical one."""
    return BASELINE_SOLVER if nb_partitions <= 1 and nb_node_per_meta_nodes == 0 else HIERARCHICAL_SOLVER


def _point_row(problem: dict, nb_partitions: int, nb_node_per_meta_nodes: int) -> dict:
    """The row of a point before it is solved: the problem, the point and what it means in spins."""
    config = problem["config"]
    model = problem["ising_model"]
    num_spins = model.num_variables
    solver = solver_of(nb_partitions, nb_node_per_meta_nodes)
    # A benchmark file has no dummy_connectivity: its connectivity is the one measured on its J
    if config.dummy_creator:
        benchmark, connectivity, best_energy = np.nan, float(getattr(config, "dummy_connectivity", np.nan)), np.nan
    else:
        benchmark, connectivity = benchmark_name(config.benchmark), connections_per_spin(model) / (num_spins - 1)
        # From the name of a regenerated file, else what the pipeline read in optimal_energy.txt
        best_energy = best_reported_energy(benchmark)
        if np.isnan(best_energy) and problem.get("best_found") is not None:
            best_energy = float(problem["best_found"])
    row = {
        "problem_type": config.problem_type,
        "benchmark": benchmark,
        "num_spins": num_spins,
        "connectivity": connectivity,
        "connections_per_spin": connections_per_spin(model),
        "best_energy": best_energy,
        "nb_partitions": nb_partitions,
        "nb_node_per_meta_nodes": nb_node_per_meta_nodes,
        "solver": solver,
    }
    if solver == HIERARCHICAL_SOLVER:
        # As the pipeline does: the config value divides the number of spins.
        # A point with no partition or no node per meta node cannot be solved: its sizes are empty.
        total_meta_nodes = int(num_spins / nb_node_per_meta_nodes) if nb_node_per_meta_nodes else 0
        row["spins_per_partition"] = num_spins / nb_partitions if nb_partitions else np.nan
        row["total_meta_nodes"] = total_meta_nodes
        row["meta_nodes_per_partition"] = total_meta_nodes / nb_partitions if nb_partitions else np.nan
        row["spins_per_meta_node"] = num_spins / total_meta_nodes if total_meta_nodes else np.nan
        row["nb_sweeps"] = config.nb_sweeps_Hierarchical_solver
        if config.partitioning_technique in FREE_PARTITIONERS:
            # The partitioner chooses the partitions: their sizes are not the ones of the number asked
            row["spins_per_partition"] = row["meta_nodes_per_partition"] = np.nan
    return row


def _failure(error: Exception) -> dict:
    """What the row of a point that could not be solved has in place of its energies: why and where.

    "error" is the error and its message; "error_where" the function that raised it and the two
    calls that led to it, with their file (from the repo top) and line.
    """

    def place(frame: traceback.FrameSummary) -> str:
        path = Path(frame.filename)
        path = path.relative_to(_REPO_TOP) if path.is_relative_to(_REPO_TOP) else Path(*path.parts[-2:])
        return f"{frame.name} ({path.as_posix()}:{frame.lineno})"

    calls = traceback.extract_tb(error.__traceback__)[-3:]
    return {
        "nb_trials": 0,
        **dict.fromkeys([*ENERGY_STATS, "en_mean", "time_mean"], np.nan),
        "error": f"{type(error).__name__}: {' '.join(str(error).split())}",
        "error_where": ", called from ".join(place(frame) for frame in reversed(calls)),
    }


def _problem_name(row: dict) -> str:
    """The problem of a row, as the messages name it: its benchmark file, or its connectivity."""
    return row["benchmark"] if isinstance(row["benchmark"], str) else f"connectivity {row['connectivity']:.3g}"


def describe_failure(row: dict, what: str = "not solved, its row has no energy") -> str:
    """The message of a point that could not be solved: its problem, the error and where it was raised."""
    return (
        f"Point (nb_partitions={row['nb_partitions']}, nb_node_per_meta_nodes={row['nb_node_per_meta_nodes']})"
        f" of {_problem_name(row)}: {what}\n    {row['error']}\n    raised in {row['error_where']}"
    )


def run_single(
    problem: dict,
    nb_partitions: int,
    nb_node_per_meta_nodes: int,
    nb_sweeps: int | None = None,
    return_ans: bool = False,
    quiet: bool = False,
):
    """Solve a copy of a generated problem for one (nb_partitions, nb_node_per_meta_nodes) point.

    A point that cannot be solved does not raise: its row has no trial, NaN energies, the reason
    in "error" and the place in "error_where" (a row of a solved point has neither).
    @param problem: the output of generate_problem, left untouched
    @param nb_sweeps: sweeps of the hierarchical solver; None keeps the value of the config
    @param return_ans: also return the full pipeline answer (states, energies of each trial, ...);
        None for a point that could not be solved
    @param quiet: do not print why and where a point could not be solved (run_grid prints it itself)
    @return: the row of the point (a dict): the point, what it means in spins and the energy statistics
    """
    problem = deepcopy(problem)
    config = problem["config"]
    solver = solver_of(nb_partitions, nb_node_per_meta_nodes)
    config.solvers = [solver]
    config.nb_cores = nb_partitions
    # The name of the pipeline config for the number of nodes per meta node
    config.nb_meta_nodes = nb_node_per_meta_nodes
    config.nb_partitions = nb_partitions
    if nb_sweeps is not None:
        config.nb_sweeps_Hierarchical_solver = int(nb_sweeps)

    stages = [
        ENERGY_CALC_STAGES.get(config.problem_type),
        QuantizationStage,
        CombineNodesStage,
        MismatchStage,
        SimulationStage,
        InitializationStage,
    ]
    stages = [s for s in stages if s is not None]
    row = _point_row(problem, nb_partitions, nb_node_per_meta_nodes)
    try:
        ans, _ = next(stages[0](stages[1:], **problem).run())
        row.update(_energy_stats(np.asarray(ans.energies[solver], dtype=float)))
        if row["nb_trials"]:
            row["time_mean"] = np.mean(ans.computation_time[solver])
    except Exception as error:
        # The partitioner or the solver refused the point: the row says so instead of stopping a study
        ans = None
        row.update(_failure(error))
        if not quiet:
            print(describe_failure(row))
    return (row, ans) if return_ans else row


def _energy_stats(energies: np.ndarray) -> dict:
    """The number of trials and the MIN / 25 % / 50 % / 75 % / MAX / mean of their energies."""
    stats = {"nb_trials": energies.size}
    if energies.size:
        stats["en_min"], stats["en_25"], stats["en_50"], stats["en_75"], stats["en_max"] = np.percentile(
            energies, [0, 25, 50, 75, 100]
        )
        stats["en_mean"] = np.mean(energies)
    return stats


class _SweepEnergySolver(HierarchicalSolver):
    """A HierarchicalSolver that keeps the energy reached after each sweep of each trial."""

    trials: list[list[float]] = []

    def solve(self, model, initial_state, nb_sweeps, core_solver, **arguments):
        energies = []
        upper_solves = 0

        def recording_solver(model, **solver_arguments):
            nonlocal upper_solves
            # A sweep starts with the solve of the upper model: the previous sweep is then complete
            if model is self.upper_model:
                if upper_solves:
                    energies.append(float(self.original_model.evaluate(self.assemble_state())))
                upper_solves += 1
            return core_solver(model=model, **solver_arguments)

        result = super().solve(model, initial_state, nb_sweeps, recording_solver, **arguments)
        energies.append(float(result[1]))
        self.trials.append(energies)
        return result


def run_single_sweeps(
    problem: dict, nb_partitions: int, nb_node_per_meta_nodes: int, nb_sweeps: list[int], quiet: bool = False
) -> list[dict]:
    """Solve one point once, with the largest number of sweeps, and keep the energy after each sweep.

    The energy after k sweeps of that run is the energy of a run with nb_sweeps = k: a sweep does not
    depend on how many follow. This gives the rows of all the numbers of sweeps for the cost of the
    largest one. Only the row of the largest has a time_mean: the solver reports one time per trial.
    @return: one row per number of sweeps, as run_single would give; for a point that cannot be
        solved, its row without energies for each of them
    """
    import ising.stages.simulation_stage as simulation_stage

    _SweepEnergySolver.trials = []
    simulation_stage.HierarchicalSolver = _SweepEnergySolver
    try:
        last_row = run_single(problem, nb_partitions, nb_node_per_meta_nodes, nb_sweeps=max(nb_sweeps), quiet=quiet)
    finally:
        simulation_stage.HierarchicalSolver = HierarchicalSolver
    if "error" in last_row:
        return [dict(last_row, nb_sweeps=int(value)) for value in nb_sweeps]
    # One line per trial, one column per sweep
    energies = np.array(_SweepEnergySolver.trials, dtype=float)
    rows = []
    for value in nb_sweeps:
        row = dict(last_row, nb_sweeps=int(value), **_energy_stats(energies[:, int(value) - 1]))
        if value != max(nb_sweeps):
            row["time_mean"] = np.nan
        rows.append(row)
    return rows


def build_hierarchy(problem: dict, nb_partitions: int, nb_node_per_meta_nodes: int) -> HierarchicalSolver | None:
    """Build the hierarchy the solver starts from for one point, without solving anything.

    The solver builds it once per trial, before the first sweep, and the couplings do not change
    during the sweeps (only the fields of the partitions do). With the "random" partitioning
    it differs from one trial to the next, so this one is then an example, not the one of a run.

    @return: a solver holding the hierarchy: upper_model is the model on the meta nodes,
        subproblems[i].model the model of partition i, subproblems[i].original_nodes its spins.
        None, with the reason printed, for a point that has no hierarchy (plot_hierarchy then draws nothing).
    """
    config = problem["config"]
    model = problem["ising_model"]
    hierarchy = HierarchicalSolver()
    try:
        partitioning = PARTITIONERS[config.partitioning_technique](model, nb_partitions)
        # As the pipeline does: the config value divides the number of spins
        hierarchy.make_hierarchy(model, partitioning, int(model.num_variables / nb_node_per_meta_nodes))
    except Exception as error:
        row = dict(_point_row(problem, nb_partitions, nb_node_per_meta_nodes), **_failure(error))
        print(describe_failure(row, "no hierarchy"))
        return None
    return hierarchy


def make_partitioning(problem: dict, nb_partitions: int = 1, technique: str | None = None) -> np.ndarray | None:
    """The partition of every spin of a generated problem, as the solver gets it from the partitioner.

    @param nb_partitions: the number asked; a partitioner of FREE_PARTITIONERS does not use it
    @param technique: a partitioner of PARTITIONERS; None: the partitioning_technique of the config
    @return: None, with the reason printed, when the partitioner refuses that number of partitions
    """
    technique = technique or problem["config"].partitioning_technique
    try:
        return PARTITIONERS[technique](problem["ising_model"], nb_partitions)
    except Exception as error:
        failure = _failure(error)
        print(
            f"{nb_partitions} partitions of {_problem_name(_point_row(problem, nb_partitions, 1))} with {technique}:"
            f" no partitioning\n    {failure['error']}\n    raised in {failure['error_where']}"
        )
        return None


def measure_partitions(
    problem: dict, technique: str | None = None, nb_partitions: int = 1, nb_draws: int = 1
) -> list[dict]:
    """What a partitioner makes of a generated problem, without solving anything.

    @param nb_draws: number of splits measured; more than one for a partitioner that draws its
        partitions ("louvain", "random"), as each split is then different
    @return: one row per split: the number of partitions asked and returned, the smallest and the
        largest one, cut_share (the share of the coupling weight |J| between two different partitions)
        and cut_share_random (what partitions of the same sizes drawn at random cut on average).
        No row, with the reason printed, when the partitioner refuses that number of partitions.
    """
    config = problem["config"]
    model = problem["ising_model"]
    technique = technique or config.partitioning_technique
    weight = np.abs(model.J + model.J.T).astype(float)
    num_spins = model.num_variables
    connectivity = float(config.dummy_connectivity) if config.dummy_creator else connections_per_spin(model) / (num_spins - 1)
    rows = []
    for _ in range(nb_draws):
        labels = make_partitioning(problem, nb_partitions, technique)
        if labels is None:
            # Refused: it would be at every draw
            break
        sizes = np.unique(labels, return_counts=True)[1]
        inside = weight[labels[:, None] == labels[None, :]].sum()
        rows.append(
            {
                "connectivity": connectivity,
                "partitioning": technique,
                "nb_partitions_asked": nb_partitions,
                "nb_partitions": len(sizes),
                "smallest": int(sizes.min()),
                "largest": int(sizes.max()),
                "cut_share": float(1 - inside / weight.sum()),
                "cut_share_random": float(1 - np.sum(sizes * (sizes - 1)) / (num_spins * (num_spins - 1))),
            }
        )
    return rows


class _RecordingSolver(HierarchicalSolver):
    """A HierarchicalSolver that keeps, at every sweep, the models it hands to the core solver."""

    sweeps: list[dict] = []

    def get_influence(self, subproblem):
        local_model = super().get_influence(subproblem)
        # The first partition of a sweep: the upper model has just been solved
        if subproblem is next(iter(self.subproblems.values())):
            self.sweeps.append(
                {
                    "upper_J": self.upper_model.J.copy(),
                    "upper_h": self.upper_model.h.copy(),
                    "upper_state": self.upper_state.copy(),
                    "partition_J": [],
                    "partition_h": [],
                    "partition_state": [],
                }
            )
        self.sweeps[-1]["partition_J"].append(local_model.J.copy())
        self.sweeps[-1]["partition_h"].append(local_model.h.copy())
        self.sweeps[-1]["partition_state"].append(subproblem.state.copy())
        return local_model


def trace_sweeps(problem: dict, nb_partitions: int, nb_node_per_meta_nodes: int, nb_sweeps: int = 4) -> dict | None:
    """Run the first trial of a point through the pipeline and record what each sweep solves.

    @return: "sweeps": one dict per sweep (0, 1, ...) with the upper_J / upper_h and the upper_state
        after the upper solve, and for each partition the partition_J / partition_h given to the core
        solver and the partition_state it starts the sweep from; "energy": the energy of the trial.
        None, with the reason printed, for a point that cannot be solved or that has nothing to
        trace (the plot_sweep_* functions then draw nothing).
    """
    import ising.stages.simulation_stage as simulation_stage

    problem = deepcopy(problem)
    problem["config"].nb_runs = 1
    problem["config"].nb_sweeps_Hierarchical_solver = nb_sweeps
    _RecordingSolver.sweeps = []
    simulation_stage.HierarchicalSolver = _RecordingSolver
    try:
        row, ans = run_single(problem, nb_partitions, nb_node_per_meta_nodes, return_ans=True, quiet=True)
    finally:
        simulation_stage.HierarchicalSolver = HierarchicalSolver
    if "error" in row:
        print(describe_failure(row, "no trace"))
        return None
    if not _RecordingSolver.sweeps:
        print(
            f"Point (nb_partitions={nb_partitions}, nb_node_per_meta_nodes={nb_node_per_meta_nodes}): no trace\n"
            "    with one spin per meta node the solver never solves the partitions"
        )
        return None
    return {"sweeps": _RecordingSolver.sweeps, "energy": float(ans.energies[HIERARCHICAL_SOLVER][0])}


def _init_worker(nice: int, problem: dict) -> None:
    """Give each worker its priority and its copy of the generated problem."""
    global _PROBLEM
    os.nice(nice)
    _PROBLEM = problem


def _run_point(point: tuple, problem: dict | None = None) -> list[dict]:
    """The rows of one point: several when its number of sweeps is a list."""
    problem = _PROBLEM if problem is None else problem
    if isinstance(point[2], list):
        return run_single_sweeps(problem, *point, quiet=True)
    return [run_single(problem, *point, quiet=True)]


def run_grid(
    problem: dict,
    grid: list[tuple[int, int]],
    nb_sweeps: int | list[int] | None = None,
    nb_workers: int | None = None,
    nice: int = 0,
    on_point: Any = None,
) -> pd.DataFrame:
    """Solve one generated problem for every (nb_partitions, nb_node_per_meta_nodes) point of a grid.

    A point that cannot be solved does not stop the grid: why and where it failed is printed when
    it happens, the other points are solved, and it keeps its rows, without energies and with the
    reason in "error" and "error_where" (see run_single).
    @param nb_sweeps: one number of sweeps, or a list of them: every point of the grid is then solved
        once with the largest and the energy after each listed number of sweeps is kept (see
        run_single_sweeps). None keeps the value of the config. The single-core baseline has no
        sweeps.
    @param nb_workers: points solved at the same time (default: all CPUs); 1 runs in this process
    @param nice: positive on a shared server, to leave the priority to other users
    @param on_point: called in this process each time a point is done, with the rows of all the
        points done so far, in the order of the grid (run_study saves a study that is going with it)
    @return: one row per point and number of sweeps, in the order of the grid (see run_single)
    """
    several = isinstance(nb_sweeps, (list, tuple)) and len(nb_sweeps) > 1
    sweeps = sorted(int(value) for value in nb_sweeps) if several else as_list(nb_sweeps)[0]
    points = [(*point, None if solver_of(*point) == BASELINE_SOLVER else sweeps) for point in grid]
    nb_workers = min(nb_workers or os.cpu_count(), len(points))
    # The rows of the points that are done, by the place of the point in the grid
    done: dict[int, list[dict]] = {}

    def rows_so_far() -> list[dict]:
        return [row for index in sorted(done) for row in done[index]]

    # disable=False: the only bar shown, as the per-trial ones are silenced
    with tqdm.tqdm(total=len(points), ascii="░▒█", desc="Grid points", disable=False) as bar:

        def keep(index: int, rows: list[dict]) -> None:
            done[index] = rows
            try:
                if "error" in rows[0]:
                    bar.write(describe_failure(rows[0], "not solved, going on with the next point"))
                if on_point is not None:
                    on_point(rows_so_far())
            except Exception as error:
                # Telling about a point or saving the run so far must never cost the points that are solved
                bar.write(f"Point {points[index][:2]} is done, but it could not be reported or saved: {type(error).__name__}: {error}")
            bar.update()

        if nb_workers <= 1:
            for index, point in enumerate(points):
                keep(index, _run_point(point, problem))
        else:
            with ProcessPoolExecutor(max_workers=nb_workers, initializer=_init_worker, initargs=(nice, problem)) as pool:
                futures = {pool.submit(_run_point, point): index for index, point in enumerate(points)}
                # As they finish, not in the order of the grid: a long point does not hold the others back
                for future in as_completed(futures):
                    keep(futures[future], future.result())
    return pd.DataFrame(rows_so_far())


def run_study(
    config: dict,
    grid: list[tuple[int, int]],
    nb_sweeps: int | list[int] | None = None,
    connectivity: float | list[float] | None = None,
    nb_workers: int | None = None,
    nice: int = 0,
    benchmark: str | list[str] | None = None,
    live_folder: Path | None = None,
) -> pd.DataFrame:
    """Solve the grid for one or several numbers of sweeps and connectivities or benchmarks, in one table.

    A connectivity changes the problem: one problem is generated per value (same seed) and the grid,
    with its single-core baseline, is solved on each. The sweeps are a setting of the solver: they
    are run on the same problem.
    @param nb_sweeps: one number of sweeps or a list; None keeps the value of the config
    @param connectivity: one connectivity (dummy_connectivity) or a list; None keeps the value of the config
    @param benchmark: one file of ising/benchmarks/ (e.g. "G/G32.txt") or a list: one problem per file,
        in place of the generated ones (connectivity is then not used). None keeps the problem of the config.
    @param live_folder: a folder (RESULTS_DIR) where the rows so far are saved each time a point is
        done, as live_run.yaml / live_run.csv: load_run("live_run", folder) reads the study while it is
        going, e.g. to plot it from another notebook. None: nothing is saved before the end.
    @return: the rows of all the runs; the columns benchmark, connectivity and nb_sweeps tell them apart.
        A point that cannot be solved is reported when it happens and keeps its rows (see run_grid).
    """
    if benchmark is not None:
        configs = {name: set_benchmark(config, name) for name in as_list(benchmark)}
    elif not config["dummy_creator"]:
        configs = {benchmark_name(config["benchmark"]): config}
    else:
        values = as_list(config["dummy_connectivity"] if connectivity is None else connectivity)
        configs = {f"connectivity {value}": dict(config, dummy_connectivity=value) for value in values}
    frames = []

    def save_live(rows: list[dict]) -> None:
        """The problems that are done and the points of the current one, under the name of the live run."""
        so_far = pd.concat([*frames, pd.DataFrame(rows)], ignore_index=True)
        save_run(config, so_far, live_folder, link_last=False, name=LIVE_RUN)

    if live_folder is not None:
        print(f"Live run: {live_folder / LIVE_RUN}.csv, updated after each point (load_run(\"{LIVE_RUN}\"))")
    for label, problem_config in configs.items():
        problem = generate_problem(problem_config)
        model = problem["ising_model"]
        print(f"{label}: {model.num_variables} spins, {connections_per_spin(model):.1f} connections per spin")
        on_point = save_live if live_folder is not None else None
        frames.append(run_grid(problem, grid, nb_sweeps=nb_sweeps, nb_workers=nb_workers, nice=nice, on_point=on_point))
    return pd.concat(frames, ignore_index=True)
