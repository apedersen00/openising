"""
Hierarchical solver with two levels, as built step by step in study/01_hierarchy_solver_build_step_by_step.py.

- level 0: the spins of the problem, split into cores by `labels` (e.g. from partition.py)
- level 1: metanodes of nodes_per_metanode nodes (group_nodes on the whole problem), all in one upper core

solve(nb_sweeps):
    start: every core alone, without the field of the rest
    then nb_sweeps sweeps of
        up:    vote, every metanode at the majority of its nodes
        upper: the upper core solved from the vote
        down:  expand, every node at the majority of its metanodes
        lower: every core solved from its spins, with the field of the expanded spins outside it
"""
import logging

import numpy as np

from core import Core
from ising_solver import energy
from metanode import connect_metanodes, expand, group_nodes, make_metanodes, vote
from problems import generate_nodes


class HierarchySolver:
    """
    - J, h: the problem
    - nodes, cores: the nodes of the problem and the cores of level 0
    - metanodes, upper_core: the metanodes and the core of level 1
    - lower_state, upper_state: the spins of level 0 (by node.row) and of level 1 (by metanode.row)
    - history: one dict per sweep (sweep 0: the cores alone) with the energy of the problem after the lower solves,
      and from sweep 1 the energy of the upper solve and the energy of the problem at the expanded spins
    - best_state, best_energy: the best lower state after a sweep
    """

    def __init__(self, J, h, labels, nodes_per_metanode, method="linkage"):
        self.J = np.ascontiguousarray(J, dtype=np.float64)
        self.h = np.ascontiguousarray(h, dtype=np.float64)
        self.nodes = generate_nodes(self.J, self.h)
        self.cores = [Core(c, [self.nodes[i] for i in np.flatnonzero(labels == c)]) for c in np.unique(labels)]
        self.metanodes = make_metanodes(self.nodes, group_nodes(self.J, nodes_per_metanode, method))
        connect_metanodes(self.metanodes)
        self.upper_core = Core(0, self.metanodes)
        for core in [*self.cores, self.upper_core]:
            core.init_J_h()
        self.lower_state = np.zeros(len(self.h))
        self.upper_state = np.ones(len(self.metanodes))
        self.expanded = None  # the spins of the nodes as the upper level sees them, set at every sweep
        self.history = []
        self.best_state, self.best_energy = None, np.inf

    def energy(self, state):
        """Energy of the problem at state (spins by node.row)."""
        return energy(state, self.J, self.h)

    def solve_lower(self, spins, seed, **solve_args):
        """
        Every core solved, its best spins written in lower_state.

        - spins: the spins seen outside the cores, the same for every core. None: no field, every core alone from
          random voltages. Otherwise every core starts from its current spins.
        """
        for core in self.cores:
            ids = [node.row for node in core.nodes]
            if spins is None:
                core.f = np.zeros(len(ids))
                v0 = None
            else:
                core.field_from_outside(spins)
                v0 = self.lower_state[ids]
            core.solve(v0=v0, seed=seed, **solve_args)
            self.lower_state[ids] = core.best_spins

    def solve_upper(self, seed, **solve_args):
        """The upper core solved from upper_state, its best spins written in upper_state."""
        self.upper_core.solve(v0=self.upper_state, seed=seed, **solve_args)
        self.upper_state = self.upper_core.best_spins.copy()

    def solve(self, nb_sweeps, seed=0, on_step=None, **solve_args):
        """
        The cores alone, then nb_sweeps sweeps of up, upper, down and lower. Starts again from scratch at every call.

        - seed: the trial; every solve gets its own seeds from it
        - on_step: if given, called as on_step(self, sweep, step) after every step: "lower" (the cores alone at
          sweep 0, then the lower solves of every sweep), "vote", and "upper" (after the upper solve and the
          expansion: self.expanded is set)
        - solve_args: passed to Core.solve (nb_iterations, nb_seeds, max_time)
        Returns the best lower state, its energy and the history.
        """
        self.upper_state = np.ones(len(self.metanodes))  # a tie of the first vote gives +1
        self.expanded = None
        self.history = []
        self.best_state, self.best_energy = None, np.inf
        base = 42 + 100_000 * seed
        self.solve_lower(None, base, **solve_args)
        self._record(0)
        if on_step is not None:
            on_step(self, 0, "lower")
        for k in range(1, nb_sweeps + 1):
            self.upper_state = vote(self.metanodes, self.lower_state, self.upper_state)
            if on_step is not None:
                on_step(self, k, "vote")
            self.solve_upper(base + 1000 * k, **solve_args)
            self.expanded = expand(self.nodes, self.upper_state, self.lower_state)
            if on_step is not None:
                on_step(self, k, "upper")
            self.solve_lower(self.expanded, base + 1000 * k + 500, **solve_args)
            self._record(k, self.expanded)
            if on_step is not None:
                on_step(self, k, "lower")
        return self.best_state.copy(), self.best_energy, self.history

    def _record(self, k, expanded=None):
        """History row of sweep k, and the best state."""
        row = {"sweep": k, "energy": self.energy(self.lower_state)}
        if expanded is not None:
            row["upper_energy"] = self.upper_core.best_energy
            row["expanded_energy"] = self.energy(expanded)
        self.history.append(row)
        if row["energy"] < self.best_energy:
            self.best_state, self.best_energy = self.lower_state.copy(), row["energy"]
        logging.debug("Sweep %d: energy %f", k, row["energy"])
