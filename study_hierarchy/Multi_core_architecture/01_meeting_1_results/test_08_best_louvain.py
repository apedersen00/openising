"""Test 08: the louvain partitioning is the best of LOUVAIN_RUNS louvain runs.

What is tested: make_louvain_partitioning (ising/utils/partitioning_schemes.py). It used to return
one louvain run with no seed: another split, with another number of partitions, at each call. It now
runs louvain LOUVAIN_RUNS times (seeds 0, 1, ...) and returns the split with the highest modularity.

Why: a problem must get one louvain split, the same at every trial and in every plot, and the
best one louvain finds rather than whichever comes first.

Problems: three cliques of 6, 8 and 10 spins joined by weak links (a clear structure), and the
64-spin Maxcut problem of the dummy creator at connectivity 0.5 (no structure, signed couplings).

Checks (PASS/FAIL, the error is the number of violations unless said otherwise, tolerance 0):
  1. LOUVAIN_RUNS is 20
  2. two calls on the same problem return the same split (both problems)
  3. the modularity of the split returned is the largest of the modularities of the LOUVAIN_RUNS
     runs done one by one (error: the difference, both problems)
  4. the split returned is the first of the runs that reaches that modularity (both problems)
  5. on the three cliques the split returned is the three cliques
  6. every spin has a partition and the partitions are numbered 0, 1, ... (both problems)

Run from the repo top: python no_backup/Multi_core_architecture/01_applications/test_08_best_louvain.py
"""

import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import multi_core_runs as runs  # noqa: E402
import multi_core_utils as utils  # noqa: E402
from ising.stages.model.ising import IsingModel  # noqa: E402
from ising.utils import partitioning_schemes  # noqa: E402
from ising.utils.partitioning_schemes import make_louvain_partitioning  # noqa: E402

warnings.simplefilter("ignore", FutureWarning)
community = nx.algorithms.community

results = []


def check(name: str, error: float, tolerance: float = 0.0) -> None:
    passed = error <= tolerance
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: error = {error:.3e} (tolerance {tolerance:.1e})")


def three_cliques() -> IsingModel:
    coupling = np.zeros((24, 24))
    for start, end in ((0, 6), (6, 14), (14, 24)):
        coupling[start:end, start:end] = 1.0
    coupling[5, 6] = coupling[13, 14] = 0.1
    return IsingModel(np.triu(coupling, k=1), np.zeros(24))


def as_sets(labels: np.ndarray) -> set:
    return {frozenset(np.flatnonzero(labels == label).tolist()) for label in np.unique(labels)}


if __name__ == "__main__":
    check("LOUVAIN_RUNS is 20", abs(partitioning_schemes.LOUVAIN_RUNS - 20))

    config = utils.make_config(utils.load_config("maxcut_c1"), "Maxcut", dummy_size=64, dummy_connectivity=0.5)
    models = {"three cliques": three_cliques(), "Maxcut, 64 spins": runs.generate_problem(config)["ising_model"]}
    different, gap, not_first, badly_numbered = 0, 0.0, 0, 0
    for name, model in models.items():
        labels = make_louvain_partitioning(model, 4)
        different += int(not np.array_equal(labels, make_louvain_partitioning(model, 4)))

        graph = nx.Graph(model.J)
        splits = [community.louvain_communities(graph, seed=seed) for seed in range(partitioning_schemes.LOUVAIN_RUNS)]
        modularities = [community.modularity(graph, split) for split in splits]
        returned = community.modularity(graph, [set(part) for part in as_sets(labels)])
        print(
            f"       {name}: {len(np.unique(labels))} partitions returned, modularity {returned:.6g}; "
            f"runs one by one: {min(len(split) for split in splits)} to {max(len(split) for split in splits)} partitions, "
            f"modularity {min(modularities):.6g} to {max(modularities):.6g}"
        )
        gap = max(gap, abs(returned - max(modularities)))
        first_best = splits[int(np.argmax(modularities))]
        not_first += int(as_sets(labels) != {frozenset(part) for part in first_best})
        badly_numbered += int(labels.shape != (model.num_variables,))
        badly_numbered += int(not np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))
    check("two calls on the same problem return the same split", different)
    check("modularity of the split returned: the largest of the runs one by one", gap)
    check("the split returned is the first run with that modularity", not_first)

    cliques = {frozenset(range(0, 6)), frozenset(range(6, 14)), frozenset(range(14, 24))}
    check("three cliques: the split returned is the three cliques", int(as_sets(make_louvain_partitioning(models["three cliques"], 4)) != cliques))
    check("every spin has a partition, numbered 0, 1, ...", badly_numbered)

    print(f"{sum(results)}/{len(results)} passed")
