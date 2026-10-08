"""Utilities for the multi-core exploration: configs, saved runs and plots.

Nothing here imports the ising package: the plots work on the csv of a saved run alone.
The runs themselves are in multi_core_runs.py.

A run is a table with one row per (nb_partitions, nb_node_per_meta_nodes) point, see multi_core_runs.run_grid.
"""

import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from matplotlib import ticker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd
import yaml

FOLDER = Path(__file__).resolve().parent
CONFIG_DIR = FOLDER / "config_files"
RESULTS_DIR = FOLDER / "results"
# The name of the run a study saves while it is going (multi_core_runs.run_study, live_folder)
LIVE_RUN = "live_run"
REPO_TOP = FOLDER.parents[2]
# Where the benchmark files are, from the repo top
BENCHMARK_DIR = "ising/benchmarks"
FIGURE_DIR = FOLDER / "figures"

BASELINE_SOLVER = "Multiplicative"
HIERARCHICAL_SOLVER = "Hierarchical_solver"

# The energy statistics of a point, in plotting order
ENERGY_STATS = {"en_min": "MIN", "en_25": "25 %", "en_50": "50 %", "en_75": "75 %", "en_max": "MAX"}

# One fixed color per series, in order of the values; the single-core baseline is gray
SERIES_COLORS = ["#FF1F5B", "#00CD6C", "#009ADE", "#AF58BA", "#FFC61E", "#F28522", "#A0B1BA", "#A6761D"]
BASELINE_COLOR = "#000000"
BEST_COLOR = "#B8860B"


#################################
## Configs
#################################


def load_config(name: str | Path = "maxcut_c1") -> dict:
    """Load a yaml config, by name in config_files/ or by path."""
    path = Path(name)
    if not path.suffix:
        path = CONFIG_DIR / f"{name}.yaml"
    with path.open("r") as file:
        return yaml.safe_load(file)


def make_config(base: dict, problem_type: str, **overrides: Any) -> dict:
    """Copy a config, set its problem type and change some of its keys.

    A key that is not in the base config is refused: it is most likely a typo, and the
    pipeline would silently ignore it.
    """
    unknown = [key for key in overrides if key not in base]
    if unknown:
        raise KeyError(f"Not in the base config: {unknown}")
    config = deepcopy(base)
    config.update(overrides)
    config["problem_type"] = problem_type
    return config


def benchmark_path(name: str) -> str:
    """The benchmark of the pipeline config for a file of ising/benchmarks/, e.g. "G/G32.txt"."""
    if not (REPO_TOP / BENCHMARK_DIR / name).is_file():
        raise FileNotFoundError(f"No benchmark {name} in {REPO_TOP / BENCHMARK_DIR}")
    return f"./{BENCHMARK_DIR}/{name}"


def find_benchmarks(pattern: str) -> list[str]:
    """The files of ising/benchmarks/ that match a pattern, e.g. "G/regenerated/G*_2000_*_toroidal_*.txt".

    The regenerated G files are named G<k>_<nodes>_<edges>_<type>_<best known cut>.txt.
    """
    folder = REPO_TOP / BENCHMARK_DIR
    names = sorted(path.relative_to(folder).as_posix() for path in folder.glob(pattern) if path.is_file())
    if not names:
        raise FileNotFoundError(f"No benchmark matches {pattern} in {folder}")
    return names


def best_reported_energy(name: str) -> float:
    """The best reported energy of a regenerated G file: minus the best known cut at the end of its name.

    "G/regenerated/G32_2000_4000_toroidal_1410.txt" gives -1410; NaN for a name that is not of that form.
    """
    fields = Path(name).stem.split("_")
    return -float(fields[-1]) if len(fields) == 5 and fields[-1].isdigit() else np.nan


def benchmark_name(path: str) -> str:
    """The reverse of benchmark_path: "./ising/benchmarks/G/G32.txt" is "G/G32.txt"."""
    return Path(path).relative_to(BENCHMARK_DIR).as_posix()


def set_benchmark(config: dict, name: str | None) -> dict:
    """Copy a config and make it solve a benchmark file instead of a generated (dummy) problem.

    @param name: a file of ising/benchmarks/, e.g. "G/G32.txt"; None leaves the config as it is
    """
    config = deepcopy(config)
    if name is not None:
        config["benchmark"] = benchmark_path(name)
        config["dummy_creator"] = False
    return config


def as_list(value: Any) -> list:
    """A parameter given as one value or as a list of values, as a list."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def make_grid(nb_partitions: list[int], nb_node_per_meta_nodes: list[int], baseline: bool = True) -> list[tuple[int, int]]:
    """All (nb_partitions, nb_node_per_meta_nodes) points, plus the (1, 0) single-core baseline."""
    grid = [(partitions, meta_nodes) for partitions in nb_partitions for meta_nodes in nb_node_per_meta_nodes]
    if baseline and (1, 0) not in grid:
        grid.append((1, 0))
    return grid


def connections_per_spin(model: Any) -> float:
    """Mean number of spins a spin is coupled to (non-zero J), measured on the model."""
    coupling = model.J + model.J.T
    return float(np.count_nonzero(coupling) / model.num_variables)


#################################
## Saved runs
#################################


def _join(values: list) -> str:
    return "-".join(str(value) for value in values)


def run_name(config: dict, frame: pd.DataFrame) -> str:
    """A verbose name for a run: problem, size, connectivities, grid, sweeps, trials, partitioning, timestamp.

    A run on benchmark files has their names (G32, ...) in place of the connectivities.
    """
    hierarchical = frame[frame["solver"] != BASELINE_SOLVER]
    # G32 for both G/G32.txt and G/regenerated/G32_2000_4000_toroidal_1410.txt
    benchmarks = [Path(name).stem.split("_")[0] for name in frame["benchmark"].dropna().unique()]
    connectivities = (f"{value:.3g}" for value in sorted(frame["connectivity"].unique(), reverse=True))
    return "_".join(
        [
            str(config["problem_type"]),
            f"N{int(frame['num_spins'].iloc[0])}",
            _join(benchmarks) if benchmarks else f"c{_join(connectivities)}",
            f"p{_join(sorted(hierarchical['nb_partitions'].unique()))}",
            f"m{_join(sorted(hierarchical['nb_node_per_meta_nodes'].unique()))}",
            f"sweeps{_join(sorted(int(value) for value in hierarchical['nb_sweeps'].unique()))}",
            f"runs{config['nb_runs']}",
            str(config["partitioning_technique"]),
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        ]
    )


def _link_last(path: Path) -> None:
    """Point last_run<suffix> to a file of the same folder (ln -s)."""
    link = path.parent / f"last_run{path.suffix}"
    if link.is_symlink():
        link.unlink()
    os.symlink(path.name, link)


def _write_whole(path: Path, write: Any) -> None:
    """Write a file under another name, then give it its name: a reader never sees half of it."""
    temporary = path.with_name(f"{path.name}.tmp")
    write(temporary)
    os.replace(temporary, path)


def save_run(
    config: dict, frame: pd.DataFrame, folder: Path = RESULTS_DIR, link_last: bool = True, name: str | None = None
) -> tuple[Path, Path]:
    """Save a run as <name>.yaml and <name>.csv, and link last_run.yaml / last_run.csv to them.

    The yaml is the pipeline config of the run, with the grid in nb_partitions / nb_meta_nodes
    (the name of the pipeline config for nb_node_per_meta_nodes). A run with several numbers of
    sweeps or connectivities has them as lists in nb_sweeps_Hierarchical_solver / dummy_connectivity,
    a run with several benchmark files has them as a list in benchmark.
    @param link_last: False leaves last_run.yaml / last_run.csv on the run they point to
    @param name: the name of the files; None: the verbose name of run_name, with its timestamp.
        A run saved again under the same name replaces the files (the live run of a study).
    @return: the paths of the yaml and of the csv
    """
    folder.mkdir(parents=True, exist_ok=True)
    name = name or run_name(config, frame)
    hierarchical = frame[frame["solver"] != BASELINE_SOLVER]
    config = deepcopy(config)
    config["nb_partitions"] = sorted(int(value) for value in hierarchical["nb_partitions"].unique())
    config["nb_meta_nodes"] = sorted(int(value) for value in hierarchical["nb_node_per_meta_nodes"].unique())
    sweeps = sorted(int(value) for value in hierarchical["nb_sweeps"].unique())
    connectivities = sorted((float(value) for value in frame["connectivity"].unique()), reverse=True)
    # No hierarchical point yet in a run that is still going: the value of the config stays
    if sweeps:
        config["nb_sweeps_Hierarchical_solver"] = sweeps if len(sweeps) > 1 else sweeps[0]
    benchmarks = [benchmark_path(name) for name in frame["benchmark"].dropna().unique()]
    if benchmarks:
        config["benchmark"] = benchmarks if len(benchmarks) > 1 else benchmarks[0]
    else:
        config["dummy_connectivity"] = connectivities if len(connectivities) > 1 else connectivities[0]

    config_path, csv_path = folder / f"{name}.yaml", folder / f"{name}.csv"
    _write_whole(config_path, lambda path: path.write_text(yaml.safe_dump(config)))
    _write_whole(csv_path, lambda path: frame.to_csv(path, index=False))
    if link_last:
        _link_last(config_path)
        _link_last(csv_path)
    return config_path, csv_path


def load_run(name: str = "last_run", folder: Path = RESULTS_DIR) -> tuple[dict, pd.DataFrame]:
    """Read back the config and the table of a saved run (the last one by default)."""
    with (folder / f"{name}.yaml").open("r") as file:
        config = yaml.safe_load(file)
    frame = pd.read_csv(folder / f"{name}.csv", float_precision="round_trip")
    # Older runs: the column had the name of the pipeline config, and the connectivity and the benchmark were not stored
    frame = frame.rename(columns={"nb_meta_nodes": "nb_node_per_meta_nodes"})
    if "connectivity" not in frame:
        frame["connectivity"] = frame["connections_per_spin"] / (frame["num_spins"] - 1)
    for column in ("benchmark", "best_energy"):
        if column not in frame:
            frame[column] = np.nan
    return config, frame


#################################
## Plots
#################################


# What a run can vary, and the columns derived from them
RUN_PARAMETERS = ["benchmark", "connectivity", "nb_partitions", "nb_node_per_meta_nodes", "nb_sweeps"]
DERIVED_FROM = {
    "connections_per_spin": "connectivity",
    "spins_per_partition": "nb_partitions",
    "total_meta_nodes": "nb_node_per_meta_nodes",
    "spins_per_meta_node": "nb_node_per_meta_nodes",
}


def _select(frame: pd.DataFrame, fixed: dict, axes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """The rows of one configuration: the parameters that are not on an axis hold a single value.

    @param fixed: parameter -> value, for the parameters of the run that have several values
    @param axes: the columns the plot spreads out
    @return: the hierarchical points, the single-core rows that go with them, and a description
    """
    unknown = [column for column in fixed if column not in RUN_PARAMETERS]
    if unknown:
        raise KeyError(f"{unknown} cannot be fixed: choose among {RUN_PARAMETERS}")
    points = frame[frame["solver"] != BASELINE_SOLVER]
    baseline = frame[frame["solver"] == BASELINE_SOLVER]
    for column, value in fixed.items():
        points = points[points[column] == value if isinstance(value, str) else np.isclose(points[column], value)]
    if points.empty:
        raise ValueError(f"No point of the run has {fixed}")
    # A point that could not be solved has no energy (see multi_core_runs.run_single): the plots leave it out
    points, baseline = points[points["en_50"].notna()], baseline[baseline["en_50"].notna()]
    if points.empty:
        raise ValueError(f"None of the points of the run with {fixed} could be solved: see their column error")
    spread = {DERIVED_FROM.get(column, column) for column in axes}
    held = [column for column in RUN_PARAMETERS if column not in spread]
    several = {column: sorted(points[column].unique()) for column in held if points[column].nunique() > 1}
    if several:
        raise ValueError(f"The run has several values of {several}: fix one of each, e.g. {next(iter(several))}=...")
    # The single-core run of the same problem(s)
    baseline = baseline[baseline["connectivity"].isin(points["connectivity"].unique())]
    if points["benchmark"].notna().any():
        baseline = baseline[baseline["benchmark"].isin(points["benchmark"].unique())]
    description = f"{points['problem_type'].iloc[0]}, {int(points['num_spins'].iloc[0])} spins"
    for column in held:
        value = points[column].iloc[0]
        # No benchmark on a generated problem
        if pd.notna(value):
            description += f", {column} = {value}" if isinstance(value, str) else f", {column} = {value:.3g}"
    return points, baseline, description


def _best_energy(points: pd.DataFrame) -> float | None:
    """The best reported energy of the problem of the points; None when unknown or when they are several problems."""
    values = points["best_energy"].dropna().unique()
    return float(values[0]) if len(values) == 1 and points["best_energy"].notna().all() else None


def plot_model(model: Any, title: str | None = None):
    """The couplings J (symmetric) and the fields h of a model, as heatmaps."""
    coupling = model.J + model.J.T
    bound = max(float(np.max(np.abs(coupling))), float(np.max(np.abs(model.h))), 1e-12)
    fig, (ax_j, ax_h) = plt.subplots(
        1, 2, figsize=(8.5, 6.5), gridspec_kw={"width_ratios": [20, 1]}, sharey=True, layout="constrained"
    )
    # Signed values: two colors around a neutral zero
    image = ax_j.imshow(coupling, cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="nearest", aspect="auto")
    ax_h.imshow(model.h[:, None], cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="nearest", aspect="auto")
    ax_j.set_title("J")
    ax_j.set_xlabel("spin")
    ax_j.set_ylabel("spin")
    ax_h.set_title("h")
    ax_h.set_xticks([])
    fig.colorbar(image, ax=[ax_j, ax_h], label="coupling / field")
    fig.suptitle(title or f"{model.num_variables} spins, {connections_per_spin(model):.1f} connections per spin")
    return fig


def _min_max(values: np.ndarray) -> str:
    return f"min {values.min():.3g}, max {values.max():.3g}" if values.size else "none"


def _bound(arrays: list[np.ndarray]) -> float:
    """A symmetric color limit covering all the arrays; 1 when they are all zero (color map from -1 to 1)."""
    return max((float(np.max(np.abs(array))) for array in arrays if array.size), default=0.0) or 1.0


def _model_axes(nb_rows: int, nb_models: int, nb_strips: int = 1, height: float = 3.0):
    """A grid of models: per model, one square axis for J and nb_strips narrow ones (h, ...) next to it."""
    fig, axes = plt.subplots(
        nb_rows,
        (1 + nb_strips) * nb_models,
        figsize=((2.7 + 0.35 * nb_strips) * nb_models + 2.5, height * nb_rows),
        width_ratios=[12, *[1] * nb_strips] * nb_models,
        squeeze=False,
        layout="constrained",
    )
    return fig, axes.reshape(nb_rows, nb_models, 1 + nb_strips)


def _draw_strip(ax: plt.Axes, values: np.ndarray, limit: float, name: str):
    image = ax.imshow(values[:, None], cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
    ax.set_title(name, fontsize="small")
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def _draw_model(axes: np.ndarray, coupling: np.ndarray, field: np.ndarray, j_limit: float, h_limit: float):
    """J (upper triangular, drawn symmetric) and h next to it, with their min and max written under J."""
    image_j = axes[0].imshow(
        coupling + coupling.T, cmap="RdBu_r", vmin=-j_limit, vmax=j_limit, interpolation="nearest"
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    # The couplings between two different nodes (the diagonal is not a coupling)
    couplings = coupling[np.triu_indices_from(coupling, k=1)]
    axes[0].set_xlabel(f"J: {_min_max(couplings)}\nh: {_min_max(field)}", fontsize="small")
    return image_j, _draw_strip(axes[1], field, h_limit, "h")


def plot_hierarchy(hierarchy: Any, title: str | None = None):
    """The J and h of a hierarchy (multi_core_runs.build_hierarchy): the upper model, then each partition.

    The upper model couples the meta nodes (mean coupling and mean field of their spins) and has its
    own color scales; the partitions share theirs, so that they can be compared. The h of a partition
    is the one of the problem: the influence of the other partitions is added during the sweeps.
    Under each model: the smallest and largest coupling and field.
    Nothing is drawn for None, what build_hierarchy returns for a point that has no hierarchy.
    """
    if hierarchy is None:
        return None
    partitions = [hierarchy.subproblems[key].model for key in sorted(hierarchy.subproblems)]
    upper = hierarchy.upper_model
    nb_columns = min(4, len(partitions) + 1)
    nb_rows = -(-(len(partitions) + 1) // nb_columns)
    fig, axes = _model_axes(nb_rows, nb_columns, height=3.4)
    axes = axes.reshape(nb_rows * nb_columns, 2)
    for ax in axes[len(partitions) + 1 :].ravel():
        ax.set_visible(False)

    image_j, _ = _draw_model(axes[0], upper.J, upper.h, _bound([upper.J]), _bound([upper.h]))
    axes[0, 0].set_title(f"Upper model: {upper.num_variables} meta nodes")
    fig.colorbar(image_j, ax=axes[0], label="mean coupling", location="left")
    j_limit, h_limit = _bound([model.J for model in partitions]), _bound([model.h for model in partitions])
    for index, model in enumerate(partitions):
        image_j, image_h = _draw_model(axes[index + 1], model.J, model.h, j_limit, h_limit)
        axes[index + 1, 0].set_title(f"Partition {index}: {model.num_variables} spins")
    fig.colorbar(image_j, ax=axes[1 : len(partitions) + 1], label="coupling J of the partitions")
    fig.colorbar(image_h, ax=axes[1 : len(partitions) + 1], label="field h of the partitions")
    nb_spins = sum(model.num_variables for model in partitions)
    per_partition = nb_spins / len(partitions)
    fig.suptitle(
        title
        or f"{nb_spins} spins: {len(partitions)} partitions, {upper.num_variables} meta nodes"
        f"  ({upper.num_variables}/{per_partition:.3g}) {SIZES_LEGEND}"
    )
    return fig


def plot_sweep_couplings(trace: dict, title: str | None = None):
    """The J and h solved at each sweep (multi_core_runs.trace_sweeps): one row per sweep.

    Columns: the upper model on its own color scales; the upper model again on the scales of the
    partitions; then each partition, with the h it is given at that sweep (its own h plus the
    influence of the other partitions). Under each model: the smallest and largest coupling and field.
    The title gives the largest change of any coupling and of any field since sweep 0.
    Nothing is drawn for None, what trace_sweeps returns for a point that has no trace.
    """
    if trace is None:
        return None
    sweeps = trace["sweeps"]
    nb_partitions = len(sweeps[0]["partition_J"])
    fig, axes = _model_axes(len(sweeps), nb_partitions + 2)
    upper_j_limit = _bound([sweep["upper_J"] for sweep in sweeps])
    upper_h_limit = _bound([sweep["upper_h"] for sweep in sweeps])
    j_limit = _bound([coupling for sweep in sweeps for coupling in sweep["partition_J"]])
    h_limit = _bound([field for sweep in sweeps for field in sweep["partition_h"]])
    change_j = change_h = 0.0
    for row, sweep in enumerate(sweeps):
        upper = (sweep["upper_J"], sweep["upper_h"])
        models = [upper, *zip(sweep["partition_J"], sweep["partition_h"])]
        first = [(sweeps[0]["upper_J"], sweeps[0]["upper_h"]), *zip(sweeps[0]["partition_J"], sweeps[0]["partition_h"])]
        change_j = max(change_j, max(float(np.max(np.abs(a[0] - b[0]))) for a, b in zip(models, first)))
        change_h = max(change_h, max(float(np.max(np.abs(a[1] - b[1]))) for a, b in zip(models, first)))

        scaled_j, _ = _draw_model(axes[row, 0], *upper, upper_j_limit, upper_h_limit)
        _draw_model(axes[row, 1], *upper, j_limit, h_limit)
        for index, model in enumerate(models[1:]):
            image_j, image_h = _draw_model(axes[row, index + 2], *model, j_limit, h_limit)
        axes[row, 0, 0].set_ylabel(f"sweep {row}")
    size = sweeps[0]["upper_J"].shape[0]
    axes[0, 0, 0].set_title(f"Upper, own scales\n{size} meta nodes")
    axes[0, 1, 0].set_title("Upper, scales of\nthe partitions")
    for index, coupling in enumerate(sweeps[0]["partition_J"]):
        axes[0, index + 2, 0].set_title(f"Partition {index}\n{coupling.shape[0]} spins")
    fig.colorbar(scaled_j, ax=axes[:, 0], label="mean coupling (own scale)", location="left")
    fig.colorbar(image_j, ax=axes[:, 1:], label="coupling J")
    fig.colorbar(image_h, ax=axes[:, 1:], label="field h")
    fig.suptitle(title or f"J and h at each sweep (largest change since sweep 0: J {change_j:.3g}, h {change_h:.3g})")
    return fig


def plot_sweep_diff(trace: dict, first: int = 0, second: int = 1, title: str | None = None):
    """What changed between two sweeps of a trace: sweep `second` minus sweep `first`.

    One model per column, the upper model then each partition: the difference of J, of h and, in
    the strip s, of the spins. For the upper model the spins are its state after the upper solve;
    for a partition, the state it starts the sweep from. A spin that flipped shows as -2 or +2.
    Under each model: the smallest and largest difference of J and h, and the number of flipped spins.
    All the models share the color scales, white meaning no change.
    Nothing is drawn for None, what trace_sweeps returns for a point that has no trace.
    """
    if trace is None:
        return None
    a, b = trace["sweeps"][first], trace["sweeps"][second]
    # The sign: the state the first sweep starts from is the continuous initial state of the trial
    spins = np.sign
    models = [
        (b["upper_J"] - a["upper_J"], b["upper_h"] - a["upper_h"], spins(b["upper_state"]) - spins(a["upper_state"])),
        *[
            (
                b["partition_J"][i] - a["partition_J"][i],
                b["partition_h"][i] - a["partition_h"][i],
                spins(b["partition_state"][i]) - spins(a["partition_state"][i]),
            )
            for i in range(len(a["partition_J"]))
        ],
    ]
    fig, axes = _model_axes(1, len(models), nb_strips=2, height=3.6)
    j_limit, h_limit = _bound([model[0] for model in models]), _bound([model[1] for model in models])
    flipped = 0
    for index, (coupling, field, state) in enumerate(models):
        image_j, image_h = _draw_model(axes[0, index], coupling, field, j_limit, h_limit)
        _draw_strip(axes[0, index, 2], state, 2, "s")
        flips = int(np.count_nonzero(state))
        flipped += flips
        label = axes[0, index, 0].get_xlabel()
        axes[0, index, 0].set_xlabel(f"{label}\ns: {flips} of {state.size} flipped", fontsize="small")
        size = coupling.shape[0]
        axes[0, index, 0].set_title(f"Upper: {size} meta nodes" if index == 0 else f"Partition {index - 1}: {size} spins")
    fig.colorbar(image_j, ax=axes[0], label="difference of J")
    fig.colorbar(image_h, ax=axes[0], label="difference of h")
    fig.suptitle(title or f"Sweep {second} minus sweep {first}: {flipped} spins flipped in total")
    return fig


def plot_sweep_fields(trace: dict, title: str | None = None):
    """What a sweep can change: the upper state and, through it, the fields h of the partitions.

    One row per sweep. Top: the spin of each meta node after the upper solve. Bottom: the field
    h of each spin as given to the core solver, the partitions side by side (gray lines between them).
    Nothing is drawn for None, what trace_sweeps returns for a point that has no trace.
    """
    if trace is None:
        return None
    sweeps = trace["sweeps"]
    states = np.array([sweep["upper_state"] for sweep in sweeps])
    fields = np.array([np.concatenate(sweep["partition_h"]) for sweep in sweeps])
    fig, (ax_state, ax_field) = plt.subplots(2, 1, figsize=(11, 1.2 * len(sweeps) + 3), layout="constrained")
    image = ax_state.imshow(states, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    ax_state.set_title("Upper state")
    ax_state.set_xlabel("meta node")
    fig.colorbar(image, ax=ax_state, label="spin", ticks=[-1, 1])

    bound = max(float(np.max(np.abs(fields))), 1e-12)
    image = ax_field.imshow(fields, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto", interpolation="nearest")
    for edge in np.cumsum([len(field) for field in sweeps[0]["partition_h"]])[:-1]:
        ax_field.axvline(edge - 0.5, color="#52514e", linewidth=1)
    ax_field.set_title("Field h of the partitions")
    ax_field.set_xlabel("spin, partition after partition")
    fig.colorbar(image, ax=ax_field, label="field")
    for ax in (ax_state, ax_field):
        ax.set_yticks(range(len(sweeps)))
        ax.set_ylabel("sweep")
    if title:
        fig.suptitle(title)
    return fig


SIZES_LEGEND = "(nodes of the upper model / nodes per partition)"


def _sizes(row: Any) -> str:
    """The sizes a point solves: (nodes of the upper model / nodes per partition)."""
    return f"({row['total_meta_nodes']:.0f}/{row['spins_per_partition']:.3g})"


def plot_energy_3d(frame: pd.DataFrame, y: str = "nb_node_per_meta_nodes", title: str | None = None, **fixed: Any):
    """Energy against nb_partitions and a meta-node column: MIN, 25 %, 50 %, 75 % and MAX of each point.

    The box goes from 25 % to 75 %, the vertical line from MIN to MAX, the black mark is the 50 %.
    @param y: nb_node_per_meta_nodes or a derived column such as total_meta_nodes
    @param fixed: the value of the other parameters of the run, e.g. nb_sweeps=2, connectivity=1.0
    """
    points, baseline, description = _select(frame, fixed, ["nb_partitions", y])
    x_values = sorted(points["nb_partitions"].unique())
    y_values = sorted(points[y].unique())
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Evenly spaced positions, labelled with the values: 2, 4, 8 partitions take the same room
    for _, row in points.iterrows():
        x, y_position = x_values.index(row["nb_partitions"]), y_values.index(row[y])
        color = SERIES_COLORS[x % len(SERIES_COLORS)]
        ax.bar3d(
            x - 0.2,
            y_position - 0.2,
            row["en_25"],
            0.4,
            0.4,
            max(row["en_75"] - row["en_25"], 1e-6),
            color=color,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.plot([x, x], [y_position, y_position], [row["en_min"], row["en_max"]], color=color, linewidth=1.5)
        ax.scatter([x, x], [y_position, y_position], [row["en_min"], row["en_max"]], color=color, marker="_", s=80)
        ax.scatter([x], [y_position], [row["en_50"]], color="black", marker="D", s=22, zorder=10)
        ax.text(x, y_position, row["en_max"], f" {_sizes(row)}", fontsize="x-small", ha="center", va="bottom", zorder=11)

    handles = [plt.Rectangle((0, 0), 1, 1, color="#52514e", alpha=0.75, label="25 % to 75 %")]
    handles.append(plt.Line2D([], [], linestyle="", label=SIZES_LEGEND))
    handles.append(plt.Line2D([], [], color="#52514e", marker="_", markersize=9, label="MIN to MAX"))
    handles.append(plt.Line2D([], [], color="black", marker="D", markersize=5, linestyle="", label="50 %"))
    if len(baseline):
        # The single-core 50 % as a reference plane under / over the boxes
        level = baseline["en_50"].iloc[0]
        plane_x, plane_y = np.meshgrid([-0.5, len(x_values) - 0.5], [-0.5, len(y_values) - 0.5])
        ax.plot_surface(plane_x, plane_y, np.full(plane_x.shape, level), color=BASELINE_COLOR, alpha=0.2)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=BASELINE_COLOR, alpha=0.3, label="single core 50 %"))
    best = _best_energy(points)
    if best is not None:
        plane_x, plane_y = np.meshgrid([-0.5, len(x_values) - 0.5], [-0.5, len(y_values) - 0.5])
        ax.plot_surface(plane_x, plane_y, np.full(plane_x.shape, best), color=BEST_COLOR, alpha=0.35)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=BEST_COLOR, alpha=0.5, label=f"best reported energy ({best:.5g})"))

    ax.set_title(title or description)
    ax.set_xlabel("nb_partitions")
    ax.set_ylabel(y)
    ax.set_zlabel("Hamiltonian energy")
    ax.set_xticks(range(len(x_values)), x_values)
    ax.set_yticks(range(len(y_values)), [f"{value:.3g}" for value in y_values])
    ax.view_init(elev=24, azim=45)
    ax.legend(handles=handles, frameon=False, loc="upper left")
    return fig


def plot_energy_3d_mean(
    frame: pd.DataFrame, stat: str = "en_50", y: str = "nb_node_per_meta_nodes", title: str | None = None, **fixed: Any
):
    """Energy against nb_partitions and a meta-node column: one statistic of each point (the 50 %), as a mesh.

    Each point is linked to its neighbours along nb_partitions and along the meta-node column. The
    faces of the mesh have the color of their mean energy; darker is lower (better), as in
    plot_energy_heatmap. A point the run does not have leaves a hole. The sizes of the points are
    not written here: they are in plot_energy_3d and plot_energy_heatmap.
    @param stat: en_50, or another statistic of the rows: en_min, en_25, en_75, en_max, en_mean
    @param y: nb_node_per_meta_nodes or a derived column such as total_meta_nodes
    @param fixed: the value of the other parameters of the run, e.g. nb_sweeps=2, connectivity=1.0
    """
    points, baseline, description = _select(frame, fixed, ["nb_partitions", y])
    table = points.pivot_table(index="nb_partitions", columns=y, values=stat)
    x_values, y_values = list(table.index), list(table.columns)
    energies = table.to_numpy(dtype=float)
    name = ENERGY_STATS.get(stat, stat.removeprefix("en_"))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Evenly spaced positions, labelled with the values: 2, 4, 8 partitions take the same room
    colors = plt.cm.ScalarMappable(norm=plt.Normalize(np.nanmin(energies), np.nanmax(energies)), cmap="Blues_r")
    faces = []
    for x in range(len(x_values) - 1):
        for y_position in range(len(y_values) - 1):
            corners = [(x, y_position), (x + 1, y_position), (x + 1, y_position + 1), (x, y_position + 1)]
            if all(np.isfinite(energies[corner]) for corner in corners):
                faces.append([(*corner, energies[corner]) for corner in corners])
    if faces:
        means = [np.mean([energy for _, _, energy in face]) for face in faces]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=colors.to_rgba(means), alpha=0.7))
    mesh = {"color": "#52514e", "linewidth": 1.5}
    for x in range(len(x_values)):
        ax.plot(np.full(len(y_values), x), range(len(y_values)), energies[x], **mesh)
    for y_position in range(len(y_values)):
        ax.plot(range(len(x_values)), np.full(len(x_values), y_position), energies[:, y_position], **mesh)
    for _, row in points.iterrows():
        x, y_position = x_values.index(row["nb_partitions"]), y_values.index(row[y])
        ax.scatter([x], [y_position], [row[stat]], color="black", marker="D", s=22, zorder=10)

    handles = [plt.Line2D([], [], marker="D", markersize=5, markerfacecolor="black", markeredgecolor="black", label=f"{name} of each point", **mesh)]
    if len(baseline):
        # The single-core level of the same statistic as a reference plane under / over the mesh
        level = baseline[stat].iloc[0]
        plane_x, plane_y = np.meshgrid([-0.5, len(x_values) - 0.5], [-0.5, len(y_values) - 0.5])
        ax.plot_surface(plane_x, plane_y, np.full(plane_x.shape, level), color=BASELINE_COLOR, alpha=0.2)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=BASELINE_COLOR, alpha=0.3, label=f"single core {name}"))
    best = _best_energy(points)
    if best is not None:
        plane_x, plane_y = np.meshgrid([-0.5, len(x_values) - 0.5], [-0.5, len(y_values) - 0.5])
        ax.plot_surface(plane_x, plane_y, np.full(plane_x.shape, best), color=BEST_COLOR, alpha=0.35)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=BEST_COLOR, alpha=0.5, label=f"best reported energy ({best:.5g})"))

    # A long description (the name of a benchmark) on several lines, away from the tick labels
    ax.set_title(title or description, wrap=True)
    ax.set_xlabel("nb_partitions")
    ax.set_ylabel(y)
    ax.set_zlabel("Hamiltonian energy", labelpad=12)
    ax.set_xticks(range(len(x_values)), x_values)
    ax.set_yticks(range(len(y_values)), [f"{value:.3g}" for value in y_values])
    ax.view_init(elev=24, azim=45)
    ax.legend(handles=handles, frameon=False, loc="upper left")
    fig.colorbar(colors, ax=ax, label=f"{name} energy", shrink=0.6, pad=0.1)
    return fig


def plot_energy_lines(
    frame: pd.DataFrame,
    x: str = "nb_partitions",
    by: str = "nb_node_per_meta_nodes",
    relative: bool = False,
    title: str | None = None,
    logscale: bool = None,
    **fixed: Any,
):
    """Energy against one parameter of the run, one line per value of another.

    One panel per statistic (MIN, 25 %, 50 %, 75 %, MAX), all on the same energy axis.
    The dashed lines are the same statistic of the single-core run: one gray line, or one per
    line color when the lines are connectivities (each connectivity is its own problem).
    The dash-dotted line is the best reported energy of the benchmark, when the rows have one.
    @param x, by: two of connectivity, nb_partitions, nb_node_per_meta_nodes, nb_sweeps (or a derived column)
    @param relative: plot (energy - single core) / |single core| of the same connectivity instead of
        the energy: 0 is the single-core level, negative is better. To compare connectivities,
        whose energies are on different scales.
    @param fixed: the value of the other parameters of the run, e.g. nb_partitions=4, connectivity=1.0
    """
    points, baseline, description = _select(frame, fixed, [x, by])
    by_parameter, x_parameter = DERIVED_FROM.get(by, by), DERIVED_FROM.get(x, x)
    best = _best_energy(points)
    if relative:
        if baseline.empty:
            raise ValueError("No single-core result to compare with: add the (1, 0) point.")
        reference = baseline.set_index("connectivity")
        points = points.copy()
        for stat in ENERGY_STATS:
            level = points["connectivity"].map(reference[stat])
            points[stat] = (points[stat] - level) / level.abs()
    by_values = sorted(points[by].unique())
    fig, axes = plt.subplots(
        1, len(ENERGY_STATS), figsize=(3.2 * len(ENERGY_STATS) + 2, 4), sharey=True, layout="constrained"
    )
    single_core = {"color": BASELINE_COLOR, "linestyle": "--", "linewidth": 1.5}

    def sizes_of(series: pd.DataFrame) -> str:
        return ", ".join(_sizes(row) for _, row in series.sort_values(x).iterrows())

    # The sizes of the points of a line, from left to right: once if all the lines have the same
    all_sizes = {sizes_of(points[points[by] == value]) for value in by_values}
    shared_sizes = len(all_sizes) == 1
    for ax, (stat, stat_label) in zip(axes, ENERGY_STATS.items()):
        for index, value in enumerate(by_values):
            series = points[points[by] == value].sort_values(x)
            color = SERIES_COLORS[index % len(SERIES_COLORS)]
            # Thinner lines and other markers on top: lines that coincide stay visible
            ax.plot(
                series[x],
                series[stat],
                color=color,
                linewidth=max(3.4 - 0.8 * index, 1.0),
                marker="osD^vP*X"[index % 8],
                markersize=max(9 - 1.5 * index, 4),
                label=f"{by} = {value:.3g}" + ("" if shared_sizes else f"  {sizes_of(series)}"),
            )
            level = baseline[baseline["connectivity"] == series["connectivity"].iloc[0]]
            if by_parameter == "connectivity" and not relative and len(level):
                ax.axhline(level[stat].iloc[0], color=color, linestyle="--", linewidth=1.2)
        if relative:
            ax.axhline(0, label="single core", **single_core)
        elif by_parameter == "connectivity":
            ax.plot([], [], label="single core (same color)", **single_core)
        elif x_parameter == "connectivity":
            ordered = baseline.sort_values(x)
            ax.plot(ordered[x], ordered[stat], label="single core", **single_core)
        elif len(baseline):
            ax.axhline(baseline[stat].iloc[0], label="single core", **single_core)
        if best is not None:
            # Relative: on the same scale as the points, against the single-core level of this statistic
            level = (best - baseline[stat].iloc[0]) / abs(baseline[stat].iloc[0]) if relative else best
            ax.axhline(level, color=BEST_COLOR, linestyle="-.", linewidth=2, label=f"best reported energy ({best:.5g})")
        ax.set_title(stat_label)
        ax.set_xlabel(x)
        ax.set_xticks(sorted(points[x].unique()))
        if logscale:
            ax.set_xscale("log", base=logscale)
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.grid(color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("(energy - single core) / |single core|" if relative else "Hamiltonian energy")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_title = f"{SIZES_LEGEND},\nfrom left to right"
    if shared_sizes:
        legend_title += f":\n{next(iter(all_sizes))}"
    fig.legend(
        handles, labels, frameon=False, loc="outside right center", title=legend_title, title_fontsize="small", alignment="left"
    )
    fig.suptitle(title or description)
    return fig


def plot_energy_heatmap(
    frame: pd.DataFrame, stat: str = "en_50", y: str = "nb_node_per_meta_nodes", title: str | None = None, **fixed: Any
):
    """One energy statistic over the (nb_partitions, meta-node column) grid; darker is lower (better).

    @param fixed: the value of the other parameters of the run, e.g. nb_sweeps=2, connectivity=1.0
    """
    points, baseline, description = _select(frame, fixed, ["nb_partitions", y])
    table = points.pivot_table(index="nb_partitions", columns=y, values=stat)
    sizes = points.assign(sizes=points.apply(_sizes, axis=1)).pivot_table(
        index="nb_partitions", columns=y, values="sizes", aggfunc="first"
    )
    fig, ax = plt.subplots(figsize=(1.3 * table.shape[1] + 3, 0.9 * table.shape[0] + 2))
    image = ax.imshow(table.to_numpy(), cmap="Blues_r", aspect="auto")
    low, high = image.get_clim()
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            value = table.iat[row, column]
            if not np.isnan(value):
                dark = high > low and (value - low) / (high - low) < 0.4
                text = f"{value:.5g}\n{sizes.iat[row, column]}"
                ax.text(column, row, text, ha="center", va="center", color="white" if dark else "#0b0b0b")
    ax.set_xticks(range(table.shape[1]), [f"{value:.3g}" for value in table.columns])
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel(y)
    ax.set_ylabel("nb_partitions")
    label = f"{ENERGY_STATS.get(stat, stat)} energy"
    reference = f" (single core: {baseline[stat].iloc[0]:.5g})" if len(baseline) else ""
    best = _best_energy(points)
    if best is not None:
        reference += f", best reported energy: {best:.5g}"
    ax.set_title(title or f"{label}{reference}\n{description}\nin each cell: energy and {SIZES_LEGEND}")
    fig.colorbar(image, ax=ax, label=label)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, name: str, folder: Path = FIGURE_DIR) -> Path:
    """Save a figure as <folder>/<name>.pdf and return its path."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=600)
    return path
