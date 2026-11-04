"""
Metanodes: groups of nodes. A metanode is one node of the level above: a Node with is_metanode=True and the nodes
of its group inside (metanode.nodes).

- group_nodes gives the group of every node, with one of the two methods of the previous study:
  "original": a group takes the free nodes with the strongest coupling to its first node
  "linkage": a group grows by the free node with the strongest total coupling to the nodes it already has
- make_metanodes creates the metanodes of these groups
- connect_metanodes gives the metanodes their couplings and biases: the model of the level above
- vote gives the metanodes their spins from the spins of their nodes (lower -> upper)
- expand gives the nodes the spins of their metanodes (upper -> lower)
"""
import numpy as np

from node import Node


def group_sizes(nb_nodes, nodes_per_metanode):
    """Groups of nodes_per_metanode nodes, the last one smaller when nb_nodes is not a multiple of it."""
    nb_full, rest = divmod(nb_nodes, nodes_per_metanode)
    return np.array([nodes_per_metanode] * nb_full + ([rest] if rest else []))


def group_nodes(J, nodes_per_metanode, method="linkage"):
    """
    Group of every node coupled by J, groups[i] in 0, 1, ..., nodes_per_metanode nodes per group (the last smaller).

    The groups start from the least connected free node. A strong coupling is a large positive J_ij, two spins that
    want to be equal: the sign of J counts.
    """
    if not 1 <= nodes_per_metanode <= len(J):
        raise ValueError(f"{nodes_per_metanode} nodes per metanode for {len(J)} nodes")
    sizes = group_sizes(len(J), nodes_per_metanode)
    groups = np.full(len(J), -1)
    group = 0
    for first in np.argsort(np.count_nonzero(J, axis=1), kind="stable"):  # least connected first
        if groups[first] >= 0:
            continue
        groups[first] = group
        if method == "original":
            strongest = np.argsort(J[first])[::-1]
            free = strongest[groups[strongest] < 0]
            groups[free[: sizes[group] - 1]] = group
        elif method == "linkage":
            attraction = J[first].astype(float)  # total coupling of every node to the nodes of the group
            for _ in range(sizes[group] - 1):
                attraction[groups >= 0] = -np.inf
                best = np.argmax(attraction)
                groups[best] = group
                attraction += J[best]
        else:
            raise ValueError(f"method must be 'original' or 'linkage', not {method!r}")
        group += 1
    return groups


def make_metanodes(nodes, groups):
    """
    One metanode (a Node with is_metanode=True) per group, with the nodes of the group inside.

    - nodes: the nodes to group, nodes[i] in group groups[i]
    - groups: the group of every node, e.g. from group_nodes
    Returns the metanodes, metanodes[g] holding the nodes of group g.
    """
    metanodes = [Node(row=g, is_metanode=True) for g in range(groups.max() + 1)]  # row g in the level above
    for node, g in zip(nodes, groups):
        metanodes[g].add_node(node)
    return metanodes


def connect_metanodes(metanodes):
    """
    Model of the level above, in the metanodes: the coupling between two metanodes is the sum of the couplings
    between their nodes, the bias of a metanode the sum of the biases of its nodes. Couplings inside a metanode
    are left out (a constant when all its nodes take its spin).

    Every node must be in exactly one metanode. Fills the neighbors and the bias of every metanode, so that
    Core(label, metanodes).init_J_h() gives the J and h of the level above.
    """
    for metanode in metanodes:
        couplings = {}  # other metanode -> summed coupling
        for node in metanode.nodes:
            for neighbor, weight in node.neighbors:
                other = neighbor.in_metanode[0]
                if other is not metanode:
                    couplings[other] = couplings.get(other, 0.0) + weight
        for other, weight in couplings.items():
            metanode.add_neighbor(other, weight)
        metanode.bias = sum(node.bias for node in metanode.nodes)


def vote(metanodes, state, upper_state):
    """
    Majority vote, lower -> upper: each metanode takes the sign of the sum of the spins of its nodes, and keeps its
    spin in upper_state on a tie.

    - state: spins of the level below, by node.row
    - upper_state: current spins of the metanodes, by metanode.row (np.ones for a first vote: a tie gives +1)
    Returns the new spins of the metanodes, by metanode.row.
    """
    voted = upper_state.copy()
    for metanode in metanodes:
        total = sum(state[node.row] for node in metanode.nodes)
        if total != 0:
            voted[metanode.row] = np.sign(total)
    return voted


def expand(nodes, upper_state, state):
    """
    Majority vote, upper -> lower: each node takes the sign of the sum of the spins of its metanodes, and keeps its
    spin in state on a tie (or with no metanode). With one metanode per node: the spin of its metanode.

    - nodes: the nodes of the level below
    - upper_state: spins of the metanodes, by metanode.row
    - state: current spins of the level below, by node.row
    Returns the spins of the level below as the level above sees them, by node.row.
    """
    expanded = state.copy()
    for node in nodes:
        total = sum(upper_state[metanode.row] for metanode in node.in_metanode)
        if total != 0:
            expanded[node.row] = np.sign(total)
    return expanded
