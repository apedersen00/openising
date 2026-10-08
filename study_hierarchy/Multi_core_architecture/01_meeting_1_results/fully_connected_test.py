import logging
import sys
import os

os.environ["MKL_NUM_THREADS"] = str(4)
os.environ["NUMEXPR_NUM_THREADS"] = str(4)
os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["OPENBLAS_NUM_THREADS"] = str(4)

from ising import api
import matplotlib.pyplot as plt
import yaml
from ising.stages import TOP
from copy import deepcopy
import numpy as np
from ising.stages.simulation_stage import Ans

# Initialize the logger
logging_level = logging.INFO
logging_format = "%(asctime)s - %(filename)s - %(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging_level, format=logging_format, stream=sys.stdout)

# Input file directory
problem_type = "Maxcut"  # Specify the problem type [Maxcut, TSP, ATSP, MIMO, MPPI]
folder = "no_backup/Multi_core_architecture/01_applications/"
config_top = folder + "config_files"
figure_top = TOP / (folder + "figures")
config_path = "fully_connected_test.yaml"


with (TOP / config_top / config_path).open("r") as file:
    config = yaml.safe_load(file)

nb_cores_list = config["nb_partitions"]
nb_meta_nodes_list = config["nb_meta_nodes"]
config_new = deepcopy(config)
ans_files = {core: {meta: Ans() for meta in nb_meta_nodes_list} for core in nb_cores_list}
ans_files[1] = {0: Ans()}

grid = [(nb_cores, nb_meta_nodes) for nb_cores in nb_cores_list for nb_meta_nodes in nb_meta_nodes_list]
grid.append((1, 0))

for nb_cores, nb_meta_nodes in grid:
    config_new["nb_cores"] = nb_cores
    if nb_cores == 1 and nb_meta_nodes == 0:
        config_new["solvers"] = ["Multiplicative"]
    else:
        config_new["solvers"] = ["Hierarchical_solver"]
    config_new["nb_meta_nodes"] = nb_meta_nodes
    config_new["nb_partitions"] = nb_cores
    with (TOP / config_top / "fully_connected_test_new.yaml").open("w") as file:
        yaml.safe_dump(config_new, file)
    ans_files[nb_cores][nb_meta_nodes], _ = api.get_hamiltonian_energy(
        problem_type, config_top + "/fully_connected_test_new.yaml"
    )

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

core_values = sorted({nb_cores for nb_cores, _ in grid})
meta_values = sorted({nb_meta_nodes for _, nb_meta_nodes in grid})

cmap = plt.cm.viridis
norm = plt.Normalize(vmin=min(core_values), vmax=max(core_values))

for nb_cores, nb_meta_nodes in grid:
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
plt.savefig(figure_top / "fully_connected_test.pdf", bbox_inches="tight", dpi=600)
