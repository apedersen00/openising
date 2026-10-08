"""Test 05: the hierarchy is built when the partitions have different sizes.

What is tested: HierarchicalSolver.make_hierarchy (ising/solvers/Hierarchical_solver.py). It used to
give every partition the same number of meta nodes and raised "More upper nodes requested than local
nodes" as soon as a partition was smaller than that share (greedy or louvain partitioning). It now
recovers the nodes per meta node k = num_spins // nb_meta_nodes and gives a partition of s spins
ceil(s / k) meta nodes: when s is not a multiple of k its meta nodes are smaller.

Why: any community algorithm must be usable, whatever the sizes of its partitions, and the runs with
balanced partitions must not change.

Checks (PASS/FAIL, the error is the number of violations, tolerance 0):
  1. balanced partitions whose size is a multiple of k: the same meta nodes per partition as the
     even split used before
  2. unequal partitions (the shape of the greedy splits of G29, G40 and G32, scaled to 400 spins),
     k = 1, 2, 4: the hierarchy is built and partition i has ceil(size_i / k) meta nodes
  3. same hierarchies: no meta node is empty or larger than k, a meta node stays inside one
     partition, and the upper model has one variable per meta node
  4. same hierarchies: the meta nodes of a partition differ by at most one spin
  5. a case that raised before (partitions of 3 and 17 spins, 10 meta nodes), solved from the first
     to the last sweep with the greedy partitioning: the returned energy is the one of the state
  6. the two remaining errors are still raised: fewer meta nodes than partitions, more than spins

Run from the repo top: python no_backup/Multi_core_architecture/01_applications/test_05_unequal_partitions.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import multi_core_runs  # noqa: E402, F401  (sets TOP and the path to the repo)
from ising.solvers.Hierarchical_solver import HierarchicalSolver  # noqa: E402
from ising.stages.model.ising import IsingModel  # noqa: E402

TOLERANCE = 0.0
UNEQUAL_SIZES = {
    "G29 in 8": [1, 1, 1, 5, 49, 96, 119, 128],
    "G40 in 8": [5, 7, 9, 15, 53, 98, 101, 112],
    "G32 in 2": [192, 208],
}
NODES_PER_META = [1, 2, 4]

results = []


def check(name: str, error: float, tolerance: float = TOLERANCE) -> None:
    passed = error <= tolerance
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: error = {error:.3e} (tolerance {tolerance:.1e})")


def random_model(num_spins: int, seed: int = 0) -> IsingModel:
    rng = np.random.default_rng(seed)
    coupling = np.triu(rng.choice([-1.0, 0.0, 1.0], (num_spins, num_spins), p=[0.1, 0.8, 0.1]), k=1)
    return IsingModel(coupling, np.zeros(num_spins))


def partitioning_of(sizes: list[int]) -> np.ndarray:
    return np.repeat(np.arange(len(sizes)), sizes)


def build(sizes: list[int], nb_meta_nodes: int) -> HierarchicalSolver:
    hierarchy = HierarchicalSolver()
    hierarchy.make_hierarchy(random_model(sum(sizes)), partitioning_of(sizes), nb_meta_nodes)
    return hierarchy


def meta_nodes_per_partition(hierarchy: HierarchicalSolver) -> list[int]:
    return [len(np.unique(subproblem.upper_nodes)) for subproblem in hierarchy.subproblems.values()]


def keep_state(model: IsingModel, initial_state: np.ndarray, **_) -> tuple:
    """A core solver that returns its initial state: state, energy, time, operations, iterations."""
    return initial_state.copy(), model.evaluate(initial_state), 0.0, 1, 1


if __name__ == "__main__":
    violations = 0
    for nb_partitions, size, k in [(2, 8, 2), (4, 12, 4), (4, 12, 1), (8, 16, 4), (3, 9, 3)]:
        nb_meta_nodes = nb_partitions * size // k
        per_part, remainder = divmod(nb_meta_nodes, nb_partitions)
        before = [per_part + int(index < remainder) for index in range(nb_partitions)]
        violations += meta_nodes_per_partition(build([size] * nb_partitions, nb_meta_nodes)) != before
    check("balanced partitions: meta nodes per partition as before (5 cases)", violations)

    count_violations = 0
    membership_violations = 0
    evenness_violations = 0
    for label, sizes in UNEQUAL_SIZES.items():
        for k in NODES_PER_META:
            try:
                hierarchy = build(sizes, sum(sizes) // k)
            except ValueError as error:
                print(f"       {label}, k = {k}: {error}")
                count_violations += 1
                continue
            count_violations += meta_nodes_per_partition(hierarchy) != [-(-size // k) for size in sizes]
            meta_sizes = np.bincount(hierarchy.original_to_upper)
            membership_violations += int(meta_sizes.min() < 1) + int(meta_sizes.max() > k)
            membership_violations += hierarchy.upper_model.num_variables != len(meta_sizes)
            upper_sets = [set(subproblem.upper_nodes) for subproblem in hierarchy.subproblems.values()]
            membership_violations += sum(len(upper_set) for upper_set in upper_sets) != len(set().union(*upper_sets))
            for subproblem in hierarchy.subproblems.values():
                sizes_in_partition = np.unique(subproblem.upper_nodes, return_counts=True)[1]
                evenness_violations += int(sizes_in_partition.max() - sizes_in_partition.min() > 1)
    check("unequal partitions: built, ceil(size / k) meta nodes per partition (9 cases)", count_violations)
    check("unequal partitions: meta nodes not empty, at most k spins, inside one partition", membership_violations)
    check("unequal partitions: meta nodes of a partition differ by at most one spin", evenness_violations)

    # Two cliques of 3 and 17 spins: the greedy partitioning returns them as the two partitions
    coupling = np.zeros((20, 20))
    coupling[:3, :3] = 1.0
    coupling[3:, 3:] = 1.0
    model = IsingModel(np.triu(coupling, k=1), np.zeros(20))
    initial_state = np.random.default_rng(0).choice([-1, 1], 20)
    try:
        solver = HierarchicalSolver()
        state, energy, *_ = solver.solve(
            model=model,
            initial_state=initial_state,
            nb_sweeps=2,
            core_solver=keep_state,
            partitioning_technique="greedy",
            nb_partitions=2,
            nb_meta_nodes=10,
            core_solver_args={},
        )
        sizes = sorted(subproblem.model.num_variables for subproblem in solver.subproblems.values())
        print(f"       partitions of {sizes} spins, {solver.upper_model.num_variables} meta nodes")
        error = float(abs(energy - model.evaluate(state))) if sizes == [3, 17] and state.shape == (20,) else float("inf")
    except Exception as exception:
        print(f"       solve failed: {type(exception).__name__}: {exception}")
        error = float("inf")
    check("partitions of 3 and 17 spins, 10 meta nodes: solved with the greedy partitioning", error)

    missing = 0
    for nb_meta_nodes in (1, 21):
        try:
            build([3, 17], nb_meta_nodes)
            missing += 1
        except ValueError:
            pass
    check("errors kept: fewer meta nodes than partitions, more meta nodes than spins", missing)

    print(f"{sum(results)}/{len(results)} passed")
