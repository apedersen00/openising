import logging
import sys
import os

# One BLAS thread per process: the grid points themselves run in parallel
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["NUMEXPR_NUM_THREADS"] = str(1)
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
# Silence the per-trial progress bars: parallel workers overwrite each other's bar
os.environ["TQDM_DISABLE"] = "1"

from ising import api
import matplotlib.pyplot as plt
import yaml
from ising.stages import TOP
from copy import deepcopy
import numpy as np
import tqdm
from concurrent.futures import ProcessPoolExecutor
from ising.stages.simulation_stage import Ans, SimulationStage
from ising.stages.stage import Stage
from ising.stages.config_parser_stage import ConfigParserStage
from ising.stages.dummy_creator_stage import DummyCreatorStage
from ising.stages.quantization_stage import QuantizationStage
from ising.stages.combine_nodes_stage import CombineNodesStage
from ising.stages.mismatch_stage import MismatchStage
from ising.stages.initialization_stage import InitializationStage

# Initialize the logger
LOGGING_LEVEL = logging.WARNING #logging.INFO
LOGGING_FORMAT = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT, stream=sys.stdout)

# Input file directory
PROBLEM_TYPE = "Maxcut"  # Specify the problem type [Maxcut, TSP, ATSP, MIMO, MPPI]
FOLDER = "no_backup/Multi_core_architecture/01_applications/"
CONFIG_TOP = FOLDER + "config_files"
FIGURE_TOP = TOP / (FOLDER + "figures")
CONFIG_TEST =  "maxcut_c1"
CONFIG_PATH = CONFIG_TEST + ".yaml"
RESULTS_CSV = FOLDER + "/results/" + CONFIG_TEST + "_results.csv"

# Number of GRID points simulated at the same time
NB_WORKER = os.cpu_count()
NICE = 0 # if on server set with a positive value to let priority to other users.

with (TOP / CONFIG_TOP / CONFIG_PATH).open("r") as file:
    config = yaml.safe_load(file)

# save the config file next to the csv too
with open(FOLDER + "/results/" + CONFIG_TEST + "_config.yaml", "w") as file:
    yaml.safe_dump(config, file)

NB_CORE_LIST = config["nb_partitions"]
NB_META_NODES_LIST  = config["nb_meta_nodes"]

GRID = [(nb_cores, nb_meta_nodes) for nb_cores in NB_CORE_LIST for nb_meta_nodes in NB_META_NODES_LIST ]
GRID.append((1, 0))


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

# The problem shared by all grid points, set once per worker by init_worker
PROBLEM = None


class ProblemCaptureStage(Stage):
    """Ends the generation half of the pipeline: hands back what the parser stages built."""

    def run(self):
        yield self.kwargs


def generate_problem() -> dict:
    """Parse the config and build the Ising model once, for all grid points."""
    stages = [DummyCreatorStage, PARSER_STAGES[PROBLEM_TYPE], ProblemCaptureStage]
    parser = ConfigParserStage(stages, config_path=CONFIG_TOP + "/" + CONFIG_PATH, problem_type=PROBLEM_TYPE)
    return next(parser.run())


def init_worker(nice: int, problem: dict) -> None:
    """Give each worker its priority and its copy of the generated problem."""
    global PROBLEM
    os.nice(nice)
    PROBLEM = problem


def run_point(point: tuple[int, int]) -> Ans:
    """Solve a copy of the generated problem for one (nb_cores, nb_meta_nodes) grid point."""
    nb_cores, nb_meta_nodes = point
    problem = deepcopy(PROBLEM)
    # No partition and no meta node: plain Multiplicative solver, not the hierarchical one
    solver = "Multiplicative" if nb_cores <= 1 and nb_meta_nodes == 0 else "Hierarchical_solver"
    config = problem["config"]
    config.solvers = [solver]
    config.nb_cores = nb_cores
    config.nb_meta_nodes = nb_meta_nodes
    config.nb_partitions = nb_cores
    stages = [
        ENERGY_CALC_STAGES.get(PROBLEM_TYPE),
        QuantizationStage,
        CombineNodesStage,
        MismatchStage,
        SimulationStage,
        InitializationStage,
    ]
    stages = [s for s in stages if s is not None]
    ans, _ = next(stages[0](stages[1:], **problem).run())
    with open(RESULTS_CSV, "a") as file:
        energies = np.asarray(ans.energies[solver], dtype=float)
        if energies.size == 0:
            print(f"Warning: no energies found for {solver} with {nb_cores} cores and {nb_meta_nodes} meta nodes")
            return ans
        q1, median, q3 = np.percentile(energies, [25, 50, 75])
        file.write(f"{nb_cores},{nb_meta_nodes},{solver},{np.min(energies)},{np.max(energies)},{q1},{median},{q3}\n")
        
    return ans


if __name__ == "__main__":
    ans_files = {core: {meta: Ans() for meta in NB_META_NODES_LIST } for core in NB_CORE_LIST}
    ans_files[1] = {0: Ans()}
    with open(RESULTS_CSV, "w") as f:
        f.write("nb_cores,nb_meta_nodes,solver,en_min, en_max, en_25, en_50, en_75\n")
    problem = generate_problem()
    with ProcessPoolExecutor(max_workers=NB_WORKER, initializer=init_worker, initargs=(NICE, problem)) as pool:
        results = zip(GRID, pool.map(run_point, GRID))
        # disable=False: the only bar shown, one step per finished grid point
        for (nb_cores, nb_meta_nodes), ans in tqdm.tqdm(
            results, total=len(GRID), ascii="░▒█", desc="Grid points", disable=False
        ):
            ans_files[nb_cores][nb_meta_nodes] = ans
                # solver = "Multiplicative" if nb_cores == 1 and nb_meta_nodes == 0 else "Hierarchical_solver"
                # energies = np.asarray(ans_files[nb_cores][nb_meta_nodes].energies.get(solver, []), dtype=float)
                # if energies.size == 0:
                #     print(f"Warning: no energies found for {solver} with {nb_cores} cores and {nb_meta_nodes} meta nodes")
                #     continue
                # q1, median, q3 = np.percentile(energies, [25, 50, 75])
                # iqr = q3 - q1
                # lower_bound = q1 - 1.5 * iqr
                # upper_bound = q3 + 1.5 * iqr
                # f.write(f"{nb_cores},{nb_meta_nodes},{solver},{np.min(energies)},{np.max(energies)},{q1},{median},{q3}\n")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    core_values = sorted({nb_cores for nb_cores, _ in GRID})
    meta_values = sorted({nb_meta_nodes for _, nb_meta_nodes in GRID})

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(core_values), vmax=max(core_values))

    for nb_cores, nb_meta_nodes in GRID:
        solver = "Multiplicative" if nb_cores == 1 and nb_meta_nodes == 0 else "Hierarchical_solver"
        energies = np.asarray(ans_files[nb_cores][nb_meta_nodes].energies.get(solver, []), dtype=float)
        if energies.size == 0:
            continue

        q1, median, q3 = np.percentile(energies, [25, 50, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        whisker_low = np.min(energies[energies >= lower_bound]) if np.any(energies >= lower_bound) else q1
        whisker_high = np.max(energies[energies <= upper_bound]) if np.any(energies <= upper_bound) else q3
        x0 = float(nb_cores) - 0.2
        y0 = float(nb_meta_nodes) - 0.2
        dx = 0.4
        dy = 0.4
        box_color = cmap(norm(nb_cores))

        ax.bar3d(
            x0,
            y0,
            q1,
            dx,
            dy,
            max(q3 - q1, 1e-6),
            color=box_color,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
        )

        ax.plot(
            [float(nb_cores), float(nb_cores)],
            [float(nb_meta_nodes), float(nb_meta_nodes)],
            [whisker_low, q1],
            color="black",
            linewidth=1.0,
        )
        ax.plot(
            [float(nb_cores), float(nb_cores)],
            [float(nb_meta_nodes), float(nb_meta_nodes)],
            [q3, whisker_high],
            color="black",
            linewidth=1.0,
        )
        ax.plot(
            [float(nb_cores), float(nb_cores)],
            [float(nb_meta_nodes), float(nb_meta_nodes)],
            [median, median],
            color="white",
            linewidth=2.0,
        )

    ax.set_title("Hamiltonian energy distribution across core/meta-node combinations")
    ax.set_xlabel("Number of cores")
    ax.set_ylabel("Number of meta nodes")
    ax.set_zlabel("Hamiltonian energy")
    ax.set_xticks(core_values)
    ax.set_yticks(meta_values)
    ax.view_init(elev=24, azim=45)
    ax.grid(True)
    plt.tight_layout()
    FIGURE_TOP.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_TOP / "fully_connected_test.pdf", bbox_inches="tight", dpi=600)
