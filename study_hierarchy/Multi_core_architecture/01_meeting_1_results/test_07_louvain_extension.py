"""Test 07: the louvain extension at the end of 00_exploration_template.ipynb.

What is tested: the cells added after the plots of the notebook. They measure the louvain split of
the problems of a saved run (multi_core_runs.measure_partitions, and partition_summary and
plot_partitions of utile_louvain_compare.py), solve these problems with the louvain partitioning
(run_study, save_run with link_last=False) and play the plots of the saved run again with the
louvain run added (plot_energy_lines, plot_energy_3d, plot_energy_heatmap of utile_louvain_compare.py).

Why: louvain returns the number of partitions it wants, of any sizes. The run must go through the
solver with them, be saved without becoming the last run (the plots above it read last_run), and
the plots must show the louvain points at the right place, and only where the louvain run has one.

The notebook is run as in test 04, on 64 spins: the saved run it compares with is the one it has
just made (COMPARE_RUN = "last_run") instead of the 1024-spin one written in the cell.

Checks (PASS/FAIL, the error is the number of violations, tolerance 0):
  1. the notebook runs to the end, extension included
  2. the partition table has one line per problem for louvain and one per problem and number of
     partitions for the partitioner of the saved run, and every cut_share is between 0 and 1
  3. for the partitioner of the saved run (balanced partitions), the number of partitions is the
     one asked and cut_share_random is 1 - (s - 1) / (N - 1) for partitions of s spins
  4. the louvain run has one row per problem and nodes per meta node of the saved run, all at
     LOUVAIN_SWEEPS sweeps, with the trials of the saved run and an energy
  5. plot_energy_lines with louvain (largest difference with the rows of the louvain run,
     tolerance 1e-12; inf when a line is missing or one too many), absolute and relative:
     - against the nodes per meta node, lines = numbers of partitions: one more line, "louvain"
     - against the number of partitions, lines = nodes per meta node: one horizontal line per
       number of nodes per meta node, at the louvain energy
     - lines = numbers of sweeps: a louvain line for LOUVAIN_SWEEPS only
     - at a number of sweeps the louvain run does not have: no louvain line
     In all of them the lines of the saved run are the ones multi_core_utils.plot_energy_lines draws.
  6. plot_energy_heatmap and plot_energy_3d with louvain: one more position named "louvain" after
     the numbers of partitions; the heatmap holds the louvain 50 % energies on that line
  7. with partitions=partition_table, the number of louvain partitions of the table is written:
     in the label of the louvain line, in the name of the louvain position of the heatmap and of
     the 3D plot, and next to each point of a plot against the connectivity
  8. the louvain run is saved under a name with "louvain", and last_run still is the saved run
  9. multi_core_runs.measure_partitions on a problem of two cliques split in its two cliques:
     cut_share is the weight of the one link between them over the total weight

Run from the repo top: python no_backup/Multi_core_architecture/01_applications/test_07_louvain_extension.py
"""

import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).parent
sys.path.insert(0, str(FOLDER))
import multi_core_runs as runs  # noqa: E402
import multi_core_utils as utils  # noqa: E402
import utile_louvain_compare as louvain_compare  # noqa: E402
from test_04_notebooks import SMALL_DUMMY, run_notebook  # noqa: E402
from ising.stages.model.ising import IsingModel  # noqa: E402

NUM_SPINS = int(SMALL_DUMMY["SIZE"])

results = []


def check(name: str, error: float, tolerance: float = 0.0) -> None:
    passed = error <= tolerance
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: error = {error:.3e} (tolerance {tolerance:.1e})")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        failed, variables = run_notebook("00_exploration_template.ipynb", SMALL_DUMMY, folder)
        check("00_exploration_template.ipynb runs to the end with the louvain extension", float(failed))

        table, louvain, compare = (variables.get(name) for name in ("partition_table", "louvain", "compare"))
        if failed or table is None or louvain is None:
            for name in ("partition table", "balanced partitions", "louvain rows", "lines with louvain", "heatmap and 3D", "number of partitions", "saved run"):
                check(f"{name}: not reached", float("inf"))
        else:
            connectivities, partitions = variables["connectivities"], variables["partitions"]
            technique = variables["compare_technique"]
            print(table.to_string(index=False))
            of_louvain = table[table["partitioning"] == "louvain"]
            of_compare = table[table["partitioning"] == technique]
            violations = abs(len(of_louvain) - len(connectivities))
            violations += abs(len(of_compare) - len(connectivities) * len(partitions))
            violations += int((of_louvain["splits"] != int(SMALL_DUMMY["LOUVAIN_DRAWS"])).sum())
            violations += int((~table["cut_share"].between(0, 1)).sum())
            check("partition table: one line per problem and partitioner, cut_share in [0, 1]", violations)

            sizes = NUM_SPINS / of_compare["nb_partitions_asked"]
            violations = int((of_compare["partitions_mean"] != of_compare["nb_partitions_asked"]).sum())
            violations += int((~np.isclose(of_compare["cut_share_random"], 1 - (sizes - 1) / (NUM_SPINS - 1))).sum())
            check(f"{technique}: number of partitions asked, cut_share_random of balanced partitions", violations)

            nb_sweeps, meta_nodes = variables["LOUVAIN_SWEEPS"], variables["meta_nodes"]
            violations = abs(len(louvain) - len(connectivities) * len(meta_nodes))
            violations += int((louvain["nb_sweeps"] != nb_sweeps).sum())
            violations += int((louvain["nb_trials"] != int(SMALL_DUMMY["NB_RUNS"])).sum())
            violations += int(louvain["en_50"].isna().sum())
            violations += int((louvain["solver"] != utils.HIERARCHICAL_SOLVER).sum())
            check(f"louvain run: one row per problem and nodes per meta node, at {nb_sweeps} sweeps", violations)

            stats = list(utils.ENERGY_STATS)
            connectivity = connectivities[-1]
            single_core = compare[(compare["solver"] == utils.BASELINE_SOLVER) & (compare["connectivity"] == connectivity)]
            at_connectivity = louvain[louvain["connectivity"] == connectivity].sort_values("nb_node_per_meta_nodes")
            other_sweeps = next(int(value) for value in compare["nb_sweeps"].dropna().unique() if value != nb_sweeps)

            def lines_error(expected: dict, relative: bool, **arguments) -> float:
                """Largest difference between the louvain lines drawn and expected (label -> x, rows)."""
                fig = louvain_compare.plot_energy_lines(compare, louvain, relative=relative, **arguments)
                plain = utils.plot_energy_lines(compare, relative=relative, **arguments)
                error = 0.0
                for ax, plain_ax, stat in zip(fig.axes, plain.axes, stats):
                    drawn = {line.get_label(): line for line in ax.get_lines() if line.get_label().startswith("louvain")}
                    kept = [line.get_label() for line in ax.get_lines() if not line.get_label().startswith("louvain")]
                    if sorted(drawn) != sorted(expected) or kept != [line.get_label() for line in plain_ax.get_lines()]:
                        return float("inf")
                    for label, (x_values, rows) in expected.items():
                        energies = rows[stat].to_numpy(dtype=float)
                        if relative:
                            energies = (energies - single_core[stat].iloc[0]) / abs(single_core[stat].iloc[0])
                        if x_values is None:
                            # A horizontal line: the same energy at both ends
                            energies = np.repeat(energies, 2)
                        elif list(drawn[label].get_xdata()) != list(x_values):
                            return float("inf")
                        error = max(error, float(np.max(np.abs(np.asarray(drawn[label].get_ydata(), dtype=float) - energies))))
                return error

            error = 0.0
            for relative in (False, True):
                error = max(error, lines_error(
                    {"louvain": (at_connectivity["nb_node_per_meta_nodes"], at_connectivity)}, relative,
                    x="nb_node_per_meta_nodes", by="nb_partitions", connectivity=connectivity, nb_sweeps=nb_sweeps,
                ))
                error = max(error, lines_error(
                    {f"louvain, nb_node_per_meta_nodes = {value:.3g}": (None, at_connectivity[at_connectivity["nb_node_per_meta_nodes"] == value]) for value in meta_nodes},
                    relative, x="nb_partitions", by="nb_node_per_meta_nodes", connectivity=connectivity, nb_sweeps=nb_sweeps,
                ))
                error = max(error, lines_error(
                    {f"louvain, nb_sweeps = {nb_sweeps:.3g}": (at_connectivity["nb_node_per_meta_nodes"], at_connectivity)}, relative,
                    x="nb_node_per_meta_nodes", by="nb_sweeps", connectivity=connectivity, nb_partitions=partitions[0],
                ))
                error = max(error, lines_error(
                    {}, relative, x="nb_node_per_meta_nodes", by="nb_partitions", connectivity=connectivity, nb_sweeps=other_sweeps,
                ))
            check("plot_energy_lines with louvain: its lines where it applies, none elsewhere", error, tolerance=1e-12)

            fixed = {"connectivity": connectivity, "nb_sweeps": nb_sweeps}
            compare_partitions = sorted(compare[compare["solver"] != utils.BASELINE_SOLVER]["nb_partitions"].unique())
            names = [*[str(value) for value in compare_partitions], "louvain"]
            heatmap = louvain_compare.plot_energy_heatmap(compare, louvain, stat="en_50", **fixed).axes[0]
            violations = int([tick.get_text() for tick in heatmap.get_yticklabels()] != names)
            written = {text.get_position(): text.get_text() for text in heatmap.texts}
            for column, (_, row) in enumerate(at_connectivity.iterrows()):
                violations += int(not written.get((column, len(names) - 1), "").startswith(f"{row['en_50']:.5g}\n"))
            violations += int(any("nan" in text for text in written.values()))
            three_d = louvain_compare.plot_energy_3d(compare, louvain, **fixed).axes[0]
            violations += int([tick.get_text() for tick in three_d.get_xticklabels()] != names)
            unchanged = louvain_compare.plot_energy_heatmap(compare, louvain, stat="en_50", connectivity=connectivity, nb_sweeps=other_sweeps).axes[0]
            violations += int([tick.get_text() for tick in unchanged.get_yticklabels()] != names[:-1])
            check("heatmap and 3D plot with louvain: one more position, with the louvain energies", violations)

            of_problem = of_louvain[of_louvain["connectivity"] == connectivity].iloc[0]
            low, high = int(of_problem["partitions_min"]), int(of_problem["partitions_max"])
            count = f"{low} partitions" if low == high else f"{low} to {high} partitions"
            print(f"       connectivity {connectivity}: {count}")
            fig = louvain_compare.plot_energy_lines(compare, louvain, x="nb_node_per_meta_nodes", by="nb_partitions", partitions=table, **fixed)
            violations = int(f"louvain ({count})" not in [line.get_label() for line in fig.axes[0].get_lines()])
            violations += int(f"louvain ({count})" not in [text.get_text() for text in fig.legends[0].get_texts()])
            heatmap = louvain_compare.plot_energy_heatmap(compare, louvain, partitions=table, **fixed).axes[0]
            violations += int(heatmap.get_yticklabels()[-1].get_text() != f"louvain\n({count})")
            three_d = louvain_compare.plot_energy_3d(compare, louvain, partitions=table, **fixed).axes[0]
            violations += int(three_d.get_xticklabels()[-1].get_text() != f"louvain\n({count})")
            fig = louvain_compare.plot_energy_lines(compare, louvain, x="connectivity", by="nb_node_per_meta_nodes", relative=True, partitions=table, nb_sweeps=nb_sweeps, nb_partitions=partitions[0])
            noted = sorted(text.get_text() for text in fig.axes[0].texts)
            expected = sorted(
                (str(int(row["partitions_min"])) if row["partitions_min"] == row["partitions_max"] else f"{int(row['partitions_min'])} to {int(row['partitions_max'])}")
                for _, row in of_louvain.iterrows()
            ) * len(meta_nodes)
            violations += int(noted != sorted(expected))
            check("number of louvain partitions written on the lines, the heatmap and the 3D plot", violations)

            name = variables["LOUVAIN_RUN"]
            last = (folder / "last_run.csv").resolve().stem
            violations = int("_louvain_" not in name) + int(not (folder / f"{name}.csv").is_file())
            violations += int(not (folder / f"{name}.yaml").is_file()) + int(last == name)
            violations += int(last != variables["csv_path"].stem)
            print(f"       louvain run: {name}\n       last_run:    {last}")
            check("louvain run saved under its name, last_run left on the saved run", violations)

    # Two cliques of 4 and 6 spins (weight 2 inside) joined by one link of weight 1
    coupling = np.zeros((10, 10))
    coupling[:4, :4] = 2.0
    coupling[4:, 4:] = 2.0
    coupling[3, 4] = 1.0
    model = IsingModel(np.triu(coupling, k=1), np.zeros(10))
    problem = {"ising_model": model, "config": Namespace(partitioning_technique="cliques", dummy_creator=True, dummy_connectivity=1.0)}
    runs.PARTITIONERS["cliques"] = lambda model, nb_cores: np.repeat([0, 1], [4, 6])
    row = runs.measure_partitions(problem, nb_partitions=2)[0]
    del runs.PARTITIONERS["cliques"]
    inside = 2.0 * (4 * 3 / 2 + 6 * 5 / 2)
    error = abs(row["cut_share"] - 1.0 / (inside + 1.0)) + abs(row["cut_share_random"] - (1 - (4 * 3 + 6 * 5) / 90))
    error += abs(row["nb_partitions"] - 2) + abs(row["smallest"] - 4) + abs(row["largest"] - 6)
    check("two cliques split in their cliques: cut_share, cut_share_random and sizes", error, tolerance=1e-12)

    print(f"{sum(results)}/{len(results)} passed")
