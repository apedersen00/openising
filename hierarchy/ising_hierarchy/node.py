import itertools
import logging


class Node:
    """
    A spin of the problem, with its neighbors and the cores and metanodes it belongs to.

    - idx: unique number of the node, from one counter shared by all the nodes (metanodes included)
    - row: index of the node in its level: its row in the J of the level, its spin is state[row]
    - bias: h of the node
    - is_metanode: True if the node is a metanode, False otherwise
    - neighbors: list of [neighbor Node, coupling J_ij]
    - in_cores: list of the cores (Core objects) the node belongs to
    - in_metanode: list of the metanodes (Node objects with is_metanode=True) the node belongs to
    - nodes: if the node is a metanode, the nodes inside it
    """

    _counter = itertools.count()  # shared by every Node: never the same idx twice

    def __init__(self, row, bias=None, is_metanode=False):
        self.in_cores = []
        self.in_metanode = []
        self.nodes = []
        self.is_metanode = is_metanode
        self.idx = next(Node._counter)
        self.row = row
        self.neighbors = []  # [neighbor, weight]
        self.bias = bias
        logging.debug(
            "Node %d created at row %d. (h: %s) Is metanode: %s", self.idx, self.row, self.bias, self.is_metanode
        )

    def add_neighbor(self, neighbor, weight=1.0):
        """Add a neighbor Node, with the coupling J_ij between the two."""
        self.neighbors.append([neighbor, weight])
        logging.debug("Node %d added neighbor %d (weight: %f)", self.idx, neighbor.idx, weight)

    def add_in_core(self, core):
        """Record that the node belongs to this Core."""
        self.in_cores.append(core)
        logging.debug("Node %d added to core %d", self.idx, core.idx)

    def add_node(self, node):
        """Put node inside this metanode, and record this metanode in node."""
        if not self.is_metanode:
            raise ValueError(f"Node {self.idx} is not a metanode")
        self.nodes.append(node)
        node.add_in_metanode(self)
        logging.debug("Metanode %d: node %d added inside", self.idx, node.idx)

    def add_in_metanode(self, metanode):
        """Record that the node belongs to this metanode. Called by metanode.add_node."""
        self.in_metanode.append(metanode)
        logging.debug("Node %d added to metanode %d", self.idx, metanode.idx)

    def __repr__(self):
        string_repr = f"Node {self.idx} (row {self.row}, h: {self.bias})\n"
        string_repr += f"  Is metanode: {self.is_metanode}\n"
        string_repr += f"  In cores: {[core.idx for core in self.in_cores]}\n"
        string_repr += f"  In metanodes: {[metanode.idx for metanode in self.in_metanode]}\n"
        if self.is_metanode:
            string_repr += f"  Nodes inside: {[node.idx for node in self.nodes]}\n"
        string_repr += f"  Neighbors: {[neighbor.idx for neighbor, _ in self.neighbors]}\n"
        return string_repr
