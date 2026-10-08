"""The plots of multi_core_utils again, with a run made with the louvain partitioning added to them.

A louvain run has one point per problem (connectivity) and number of nodes per meta node, at one
number of sweeps: louvain chooses its partitions itself, so the run has no number of partitions.
It is added to a plot only where it applies:

- plot_energy_lines against nb_partitions: a dotted horizontal line per line of the plot;
  against another parameter: a dotted line per line of the plot, or one line named "louvain" when
  the lines are the numbers of partitions
- plot_energy_3d and plot_energy_heatmap: one more position on the nb_partitions axis, named "louvain"

A configuration the louvain run does not have (another number of sweeps, ...) gives the plot of
multi_core_utils unchanged, with "no louvain point" in its title.

Each plot takes `partitions`, the table of partition_summary: the number of partitions louvain made
of the problem is then written next to its name (smallest to largest over the splits measured).

Also here: what the partitioners make of a problem before any solve (partition_summary, plot_partitions).
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import multi_core_utils as utils

LOUVAIN = "louvain"
# The louvain line when it has no line of the plot to take its color from
LOUVAIN_COLOR = "#52514e"
LOUVAIN_STYLE = {"linestyle": ":", "linewidth": 2.4}


#################################
## Partitions, before any solve
#################################


def partition_summary(rows: list[dict]) -> pd.DataFrame:
    """The splits measured by multi_core_runs.measure_partitions, one line per problem and partitioner.

    Over the splits of a line: how many there are, the smallest, mean and largest number of
    partitions, the smallest and the largest partition, and the mean cut_share and cut_share_random.
    """
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .groupby(["connectivity", "partitioning", "nb_partitions_asked"], sort=False)
        .agg(
            splits=("nb_partitions", "size"),
            partitions_min=("nb_partitions", "min"),
            partitions_mean=("nb_partitions", "mean"),
            partitions_max=("nb_partitions", "max"),
            smallest=("smallest", "min"),
            largest=("largest", "max"),
            cut_share=("cut_share", "mean"),
            cut_share_random=("cut_share_random", "mean"),
        )
        .reset_index()
    )


def plot_partitions(model: Any, partitionings: dict[str, np.ndarray], title: str | None = None):
    """The couplings J of a model with its spins ordered partition after partition, one panel per split.

    The lines are the limits of the partitions, from the largest to the smallest: the squares on the
    diagonal are the couplings a partition solves, the rest is what links two partitions.
    @param partitionings: name -> partition of every spin (multi_core_runs.make_partitioning); a
        split that is None (the partitioner refused that number of partitions) has no panel
    """
    partitionings = {name: labels for name, labels in partitionings.items() if labels is not None}
    if not partitionings:
        return None
    coupling = model.J + model.J.T
    bound = max(float(np.max(np.abs(coupling))), 1e-12)
    fig, axes = plt.subplots(
        1, len(partitionings), figsize=(4.6 * len(partitionings) + 1, 5), squeeze=False, layout="constrained"
    )
    for ax, (name, labels) in zip(axes[0], partitionings.items()):
        parts, sizes = np.unique(labels, return_counts=True)
        largest_first = np.argsort(-sizes, kind="stable")
        order = np.concatenate([np.flatnonzero(labels == parts[index]) for index in largest_first])
        image = ax.imshow(
            coupling[np.ix_(order, order)], cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="nearest"
        )
        for edge in np.cumsum(sizes[largest_first])[:-1]:
            ax.axhline(edge - 0.5, color="#000000", linewidth=0.8)
            ax.axvline(edge - 0.5, color="#000000", linewidth=0.8)
        ax.set_title(f"{name}\n{len(sizes)} partitions of {sizes.min()} to {sizes.max()} spins", fontsize="medium")
        ax.set_xlabel("spin, partition after partition")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes, label="coupling J")
    fig.suptitle(title or f"{model.num_variables} spins, {utils.connections_per_spin(model):.1f} connections per spin")
    return fig


#################################
## The plots of a run, with louvain
#################################


def louvain_rows(louvain: pd.DataFrame, fixed: dict) -> pd.DataFrame:
    """The points of a louvain run at a configuration; the number of partitions is not one of its parameters."""
    # Without the points that could not be solved: they have no energy
    rows = louvain[(louvain["solver"] != utils.BASELINE_SOLVER) & louvain["en_50"].notna()]
    for column, value in fixed.items():
        if column != "nb_partitions":
            rows = rows[rows[column] == value if isinstance(value, str) else np.isclose(rows[column], value)]
    return rows


def partitions_text(partitions: pd.DataFrame | None, connectivity: float) -> str:
    """The number of louvain partitions of a problem, from the table of partition_summary: "7 to 9 partitions".

    Empty without a table or when it has no louvain split of that connectivity.
    """
    if partitions is None:
        return ""
    rows = partitions[(partitions["partitioning"] == LOUVAIN) & np.isclose(partitions["connectivity"], connectivity)]
    if rows.empty:
        return ""
    low, high = int(rows["partitions_min"].min()), int(rows["partitions_max"].max())
    return f"{low} partitions" if low == high else f"{low} to {high} partitions"


def _no_louvain(fig: plt.Figure):
    title = fig._suptitle.get_text() if fig._suptitle is not None else fig.axes[0].get_title()
    (fig.suptitle if fig._suptitle is not None else fig.axes[0].set_title)(f"{title}\nno louvain point for this configuration")
    return fig


def plot_energy_lines(
    frame: pd.DataFrame,
    louvain: pd.DataFrame,
    x: str = "nb_partitions",
    by: str = "nb_node_per_meta_nodes",
    relative: bool = False,
    title: str | None = None,
    partitions: pd.DataFrame | None = None,
    **fixed: Any,
):
    """multi_core_utils.plot_energy_lines of a run, with the louvain run of the same problems in dotted lines.

    Against nb_partitions, louvain is a horizontal line (it has no number of partitions), in the
    color of the line of the plot with the same value of `by`. Against another parameter it is a
    line of its own: in the color of the line with the same value of `by`, or named "louvain" when
    the lines are the numbers of partitions.
    @param frame: the table of a saved run (load_run), with its single-core baseline
    @param louvain: the table of the louvain run on the same problems
    @param relative: louvain is then relative to the single-core run of `frame` at the same connectivity
    @param partitions: the table of partition_summary: the number of louvain partitions of the problem
        is written in the legend, or at each point when the plot is against the connectivity
    """
    fig = utils.plot_energy_lines(frame, x=x, by=by, relative=relative, title=title, **fixed)
    points, baseline, _ = utils._select(frame, fixed, [x, by])
    rows = louvain_rows(louvain, fixed)
    by_values = sorted(points[by].unique())
    if by != "nb_partitions":
        rows = rows[np.isclose(rows[by].to_numpy()[:, None], np.array(by_values, dtype=float)[None, :]).any(axis=1)]
    if x != "nb_partitions":
        rows = rows[rows[x].isin(points[x].unique())]
    if rows.empty:
        return _no_louvain(fig)
    if relative:
        reference = baseline.set_index("connectivity")
        rows = rows.copy()
        for stat in utils.ENERGY_STATS:
            level = rows["connectivity"].map(reference[stat])
            rows[stat] = (rows[stat] - level) / level.abs()

    def named(label: str, line: pd.DataFrame) -> str:
        """The label of a louvain line, with the number of partitions of its problem when it has one problem."""
        connectivities = line["connectivity"].unique()
        text = partitions_text(partitions, connectivities[0]) if len(connectivities) == 1 else ""
        return f"{label} ({text})" if text else label

    # Per louvain line: its label, its color and its rows
    if by == "nb_partitions":
        series = [(named(LOUVAIN, rows), LOUVAIN_COLOR, rows)]
    else:
        series = [
            (named(f"{LOUVAIN}, {by} = {value:.3g}", line), utils.SERIES_COLORS[index % len(utils.SERIES_COLORS)], line)
            for index, value in enumerate(by_values)
            for line in [rows[np.isclose(rows[by], value)]]
        ]
    for ax, stat in zip(fig.axes, utils.ENERGY_STATS):
        for label, color, line in series:
            if line.empty:
                continue
            if x == "nb_partitions":
                if len(line) > 1:
                    raise ValueError(f"The louvain run has several points for {label}: fix its other parameters.")
                ax.axhline(line[stat].iloc[0], color=color, label=label, **LOUVAIN_STYLE)
            else:
                line = line.sort_values(x)
                ax.plot(line[x], line[stat], color=color, marker="X", markersize=7, label=label, **LOUVAIN_STYLE)
                if x == "connectivity":
                    # One problem per point: its number of partitions next to it
                    for _, row in line.iterrows():
                        text = partitions_text(partitions, row["connectivity"]).removesuffix(" partitions")
                        ax.annotate(text, (row[x], row[stat]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize="x-small", color=color)
    legend = fig.legends[0]
    legend_title = legend.get_title().get_text()
    legend.remove()
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, loc="outside right center", title=legend_title, title_fontsize="small", alignment="left"
    )
    return fig


def _with_louvain(frame: pd.DataFrame, louvain: pd.DataFrame, fixed: dict, y: str) -> tuple[pd.DataFrame, list, bool]:
    """A run with the louvain points as one more number of partitions, after its largest one.

    @return: the table, the numbers of partitions of the plot in order, and whether louvain is the last
    """
    points, _, _ = utils._select(frame, fixed, ["nb_partitions", y])
    rows = louvain_rows(louvain, fixed)
    rows = rows[rows[y].isin(points[y].unique())]
    positions = sorted(points["nb_partitions"].unique())
    if rows.empty:
        return frame, positions, False
    slot = positions[-1] + 1
    return pd.concat([frame, rows.assign(nb_partitions=slot)], ignore_index=True), [*positions, slot], True


def _louvain_name(partitions: pd.DataFrame | None, fixed: dict) -> str:
    """The name of the louvain position of a plot on one problem, with its number of partitions."""
    text = partitions_text(partitions, fixed["connectivity"]) if "connectivity" in fixed else ""
    return f"{LOUVAIN}\n({text})" if text else LOUVAIN


def _name_sizes(ax: plt.Axes) -> None:
    """The sizes written at a louvain point: its partitions have no single size."""
    for text in ax.texts:
        text.set_text(text.get_text().replace("/nan)", "/-)"))


def plot_energy_3d(
    frame: pd.DataFrame,
    louvain: pd.DataFrame,
    y: str = "nb_node_per_meta_nodes",
    title: str | None = None,
    partitions: pd.DataFrame | None = None,
    **fixed: Any,
):
    """multi_core_utils.plot_energy_3d of a run, with the louvain run after its largest number of partitions.

    @param partitions: the table of partition_summary: the number of louvain partitions is written under its name
    """
    merged, positions, added = _with_louvain(frame, louvain, fixed, y)
    fig = utils.plot_energy_3d(merged, y=y, title=title, **fixed)
    if not added:
        return _no_louvain(fig)
    ax = fig.axes[0]
    ax.set_xticks(range(len(positions)), [*positions[:-1], _louvain_name(partitions, fixed)])
    _name_sizes(ax)
    return fig


def plot_energy_3d_mean(
    frame: pd.DataFrame,
    louvain: pd.DataFrame,
    y: str = "nb_node_per_meta_nodes",
    title: str | None = None,
    partitions: pd.DataFrame | None = None,
    **fixed: Any,
):
    """multi_core_utils.plot_energy_3d of a run, with the louvain run after its largest number of partitions.

    @param partitions: the table of partition_summary: the number of louvain partitions is written under its name
    """
    merged, positions, added = _with_louvain(frame, louvain, fixed, y)
    fig = utils.plot_energy_3d_mean(merged, y=y, title=title, **fixed)
    if not added:
        return _no_louvain(fig)
    ax = fig.axes[0]
    ax.set_xticks(range(len(positions)), [*positions[:-1], _louvain_name(partitions, fixed)])
    _name_sizes(ax)
    return fig


def plot_energy_heatmap(
    frame: pd.DataFrame,
    louvain: pd.DataFrame,
    stat: str = "en_50",
    y: str = "nb_node_per_meta_nodes",
    title: str | None = None,
    partitions: pd.DataFrame | None = None,
    **fixed: Any,
):
    """multi_core_utils.plot_energy_heatmap of a run, with the louvain run as one more line of the grid.

    @param partitions: the table of partition_summary: the number of louvain partitions is written under its name
    """
    merged, positions, added = _with_louvain(frame, louvain, fixed, y)
    fig = utils.plot_energy_heatmap(merged, stat=stat, y=y, title=title, **fixed)
    if not added:
        return _no_louvain(fig)
    ax = fig.axes[0]
    ax.set_yticks(range(len(positions)), [*positions[:-1], _louvain_name(partitions, fixed)])
    _name_sizes(ax)
    return fig
