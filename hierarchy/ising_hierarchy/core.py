import itertools
import logging

import numpy as np

from ising_solver import sweep

NB_ITERATIONS = 200
NB_SEEDS = 80


class Core:
    """
    A part of the problem, solved on its own by the event-based multiplicative solver.

    - idx: unique number of the core, from one counter shared by all the cores
    - label: number of the core in its level (its label in the partition)
    - nodes: list of the nodes (Node objects) in the core
    - J: the submatrix of J corresponding to the nodes in the core
    - h: the subvector of h corresponding to the nodes in the core
    - set by solve: sweep_spins, sweep_energy and history of every seed,
      best_spins, best_energy and best_history of the best seed
    """

    _counter = itertools.count()  # shared by every Core: never the same idx twice

    def __init__(self, label, nodes):
        self.nodes = nodes
        self.idx = next(Core._counter)
        self.label = label
        self.J = None # submatrix of J corresponding to the nodes in the core
        self.h = None # subvector of h corresponding to the nodes in the core
        self.f = None # field of the rest of the problem on each node
        logging.debug("Core %d (label %d) created with %d nodes.", self.idx, self.label, len(self.nodes))
        for node in self.nodes:
            node.add_in_core(self)

        # set by solve
        self.sweep_spins = None
        self.sweep_energy = None
        self.history = None
        self.best_spins = None
        self.best_energy = None
        self.best_history = None

    def init_J_h(self):
        """
        Build J and h of the core from its nodes and their neighbors.
        Suboptimal, but it allows to put nodes in different partitions and still have the correct J.
        """
        N = len(self.nodes)
        self.J = np.zeros((N, N))
        self.h = np.zeros(N)
        self.f = np.zeros(N)

        for i, node_i in enumerate(self.nodes):
            self.h[i] = node_i.bias if node_i.bias is not None else 0
            for j, node_j in enumerate(self.nodes):
                if i != j:
                    # Find the weight of the edge between node_i and node_j
                    for neighbor, weight in node_i.neighbors:
                        if neighbor.idx == node_j.idx:
                            self.J[i, j] = weight
                            break
        logging.debug("Core %d initialized J and h matrices.", self.idx)

    def solve(self, nb_iterations=NB_ITERATIONS, nb_seeds=NB_SEEDS, max_time = None, v0=None, seed=42):
        """
        Solve the core with NB_SEEDS seeds of NB_ITERATIONS runs each, and keep the best seed.

        - v0: start spins (or voltages) of every seed, random if None
        - seed: the seeds are seed, seed + 1, ..., seed + nb_seeds - 1 (start voltages and kicks)
        """
        logging.debug("Solving Core %d with J:\n%s\nand h:\n%s", self.idx, self.J, self.h)
        if self.J is None or self.h is None:
            logging.info("Core %d: J and h not initialized. Initializing now.", self.idx)
            self.init_J_h()
        seeds = [seed + i for i in range(nb_seeds)]  # fixed seeds, for reproducibility
        if max_time is None:
            max_time = 80 / max(abs(self.J).sum(axis=1)) # 80 / max abs (jrow sum)
        
        bias = self.h + self.f
        self.sweep_spins, self.sweep_energy, self.history = sweep(self.J, bias, n_iter=nb_iterations, seeds=seeds, v0=v0, max_time=max_time)

        # the best seed is the one with the lowest best energy
        best_energy_idx = np.argmin(self.sweep_energy)
        self.best_spins = self.sweep_spins[best_energy_idx]
        self.best_energy = self.sweep_energy[best_energy_idx]
        self.best_history = self.history[best_energy_idx]
        
    def field_from_outside(self, spins):
        """
        Field of the rest of the problem on each node of the core: sum of J_ij * spins[j] over the neighbors j
        that are not in this core.

        - spins: value of every node of this level, by node.row. Built by the caller; metanodes do not appear here.
        A neighbor in several cores counts as inside if this core is one of them.

        Note that in originial openising (from Sofie) it was the toplevel computing a new update value to the lower level.
        """
        self.f = np.zeros(len(self.nodes))
        for i, node in enumerate(self.nodes):
            for neighbor, weight in node.neighbors:
                if self not in neighbor.in_cores:
                    self.f[i] += weight * spins[neighbor.row]
        return self.f

    def __repr__(self):
        string_repr = f"Core {self.idx} (label {self.label})\n"
        string_repr += f"  Nodes: {[node.idx for node in self.nodes]}\n"
        string_repr += f"  J: \n{self.J}\n"
        string_repr += f"  h: \n{self.h}\n"
        string_repr += f"  f: \n{self.f}\n"
        return string_repr
