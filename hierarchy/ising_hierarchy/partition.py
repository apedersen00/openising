"""
Partitions: the core of every node, labels[i] in 0 ... nb_cores - 1.

The partitioners cut a graph built from J, chosen by `weights`:
- "abs": the weight of a coupling is |J_ij|
- "adj": every coupling weighs 1 (the adjacency, what the previous study used)
"""
import networkx as nx
import numpy as np


def graph_weights(J, weights="abs"):
    """Weights of the graph the partitioners cut: |J| ("abs") or the adjacency ("adj")."""
    if weights == "abs":
        return np.abs(J)
    if weights == "adj":
        return (J != 0).astype(float)
    raise ValueError(f"weights must be 'abs' or 'adj', not {weights!r}")


def partition_random(J, nb_cores, seed=0):
    """Random cores of equal sizes: what any partition should beat."""
    return np.random.default_rng(seed).permutation(len(J)) % nb_cores


def partition_kl(J, nb_cores, weights="abs", seed=0):
    """
    Recursive Kernighan-Lin bisection (networkx): split in two, then each part again, until nb_cores cores.
    Balanced cores with few couplings between them.
    """
    rng = np.random.default_rng(seed)
    W = graph_weights(J, weights)
    labels = np.zeros(len(J), dtype=int)
    stack = [(np.arange(len(J)), nb_cores, 0)]  # (nodes, number of cores to make with them, first label)
    while stack:
        nodes, k, first = stack.pop()
        if k == 1:
            labels[nodes] = first
            continue
        k1 = k // 2
        n1 = round(len(nodes) * k1 / k)  # uneven splits, so any nb_cores works
        perm = rng.permutation(len(nodes))
        a, b = nx.algorithms.community.kernighan_lin_bisection(
            nx.from_numpy_array(W[np.ix_(nodes, nodes)]),
            partition=(set(perm[:n1].tolist()), set(perm[n1:].tolist())),
            weight="weight",
            seed=int(rng.integers(1 << 30)),
        )
        if len(a) != n1:  # networkx may return the two sides swapped
            a, b = b, a
        stack.append((nodes[sorted(a)], k1, first))
        stack.append((nodes[sorted(b)], k - k1, first + k1))
    return labels


def partition_greedy(J, nb_cores, weights="abs", seed=0):
    """
    Greedy modularity communities (networkx). With weights="adj": the partitioner of the previous study.
    The cores can have very different sizes. Deterministic: seed is not used.
    """
    communities = nx.algorithms.community.greedy_modularity_communities(
        nx.from_numpy_array(graph_weights(J, weights)), weight="weight", cutoff=nb_cores, best_n=nb_cores
    )
    labels = np.empty(len(J), dtype=int)
    for label, nodes in enumerate(communities):
        labels[list(nodes)] = label
    return labels
