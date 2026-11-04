import sys
from pathlib import Path
from node import Node
import numpy as np

# openising (the ising package) lives at the root of the repo
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))
from ising.stages.maxcut_parser_stage import MaxcutParserStage  # noqa: E402

GSET = REPO / "ising" / "benchmarks" 


def from_edges(edges):
    """
    Max-Cut as an Ising model from a list of (u, v, weight): J = -A/2 (symmetric), h = 0
    """
    n = max(max(u, v) for u, v, weight in edges) + 1
    J = np.zeros((n, n))
    for u, v, weight in edges:
        J[u, v] = -weight / 2
        J[v, u] = -weight / 2
    h = np.zeros(n)
    return J, h


def hand_dummy():
    """
    A dummy model for testing:
    The dummy: two squares and a bridge
        0 ─ 1           /4 ─ 5
        │   │          │ │   │
        3 ─ 2          │ 7 ─ 6
        |______________/
            (bridge 3–4)

    Brute-force optimum: E = -10.5 (4 states).
    """

    edges = [(0, 1, 3), (1, 2, 2), (2, 3, -2), (3, 0, 3),
             (4, 5, 2), (5, 6, 1), (6, 7, 7), (7, 4, -4),
             (3, 4, 3)]
    return from_edges(edges)


def benchmark_path(benchmark):
    """
    Path of a Max-Cut benchmark: a Gset name ("G1", ..., "K2000") or the path of a benchmark file.
    """
    path = Path(benchmark)
    return path if path.is_file() else GSET / f"{benchmark}.txt"


def maxcut_model(benchmark):
    """
    Max-Cut benchmark read by openising, as openising's IsingModel: J upper triangular, H = -s^T J s - h^T s + c,
    and H = -cut.
    """
    graph, _ = MaxcutParserStage.G_parser(benchmark_path(benchmark))
    return MaxcutParserStage.generate_maxcut(graph)


def from_ising_model(model):
    """
    J, h in our convention from an openising IsingModel: J symmetric (J + J^T), E = -1/2 s^T J s - h^T s = H - c.
    """
    J = np.asarray(model.J, dtype=np.float64)
    return J + J.T, np.asarray(model.h, dtype=np.float64)


def load_maxcut(benchmark):
    """
    Max-Cut benchmark as J, h in our convention: J = -A/2 (symmetric), h = 0, cut(s) = -(E(s) + J.sum() / 2).
    """
    return from_ising_model(maxcut_model(benchmark))


def maxcut_best_cut(benchmark):
    """
    Best known cut of a Max-Cut benchmark, from optimal_energy.txt next to it (None if unknown).
    """
    best_found = MaxcutParserStage.get_optim_value(benchmark_path(benchmark))  # openising gives it as -cut
    return None if best_found is None else -best_found


def generate_nodes(J, h):
    """
    Generate a list of Node objects from J, h.
    """

    N = len(h)
    nodes = [Node(row=i, bias=h[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if J[i, j] != 0:
                nodes[i].add_neighbor(nodes[j], J[i, j])
                nodes[j].add_neighbor(nodes[i], J[i, j])
    return nodes