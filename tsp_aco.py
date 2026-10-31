"""Traveling Salesman Problem solved with Ant System (Ant Colony Optimization).

Each iteration, every ant builds a complete tour by choosing the next city with
probability proportional to ``pheromone^alpha * (1 / distance)^beta``. Pheromone
then evaporates by a factor ``rho`` and each ant deposits ``Q / tour_length`` on
the edges it used, so short tours attract more ants over time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_MATRIX = Path(__file__).resolve().parent / "distances.txt"


@dataclass(frozen=True)
class Parameters:
    alpha: float = 1.0  # pheromone influence
    beta: float = 5.0  # visibility (1 / distance) influence
    rho: float = 0.5  # evaporation rate, 0 < rho <= 1
    q: float = 100.0  # pheromone deposited per unit of inverse tour length
    ants: int = 15
    iterations: int = 30
    initial_pheromone: float = 1.0


@dataclass
class Result:
    route: list[int]
    distance: float
    history: list[float] = field(default_factory=list)  # best distance after each iteration
    pheromones: np.ndarray | None = None


def load_matrix(path: str | Path) -> np.ndarray:
    """Load a square distance matrix (whitespace-separated, ``#`` comments allowed)."""
    matrix = np.loadtxt(path, comments="#", dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance matrix must be square")
    return matrix


def tour_length(route: list[int], distances: np.ndarray) -> float:
    """Length of the closed tour that visits ``route`` in order and returns to the start."""
    return float(sum(distances[route[i], route[(i + 1) % len(route)]] for i in range(len(route))))


def build_tour(start: int, pheromones: np.ndarray, visibility: np.ndarray, params: Parameters,
               rng: np.random.Generator) -> list[int]:
    n = len(pheromones)
    route = [start]
    unvisited = np.ones(n, dtype=bool)
    unvisited[start] = False

    for _ in range(n - 1):
        current = route[-1]
        candidates = np.flatnonzero(unvisited)
        weights = pheromones[current, candidates] ** params.alpha * visibility[current, candidates] ** params.beta
        total = weights.sum()
        probabilities = weights / total if total > 0 else np.full(len(candidates), 1 / len(candidates))
        nxt = int(rng.choice(candidates, p=probabilities))
        route.append(nxt)
        unvisited[nxt] = False
    return route


def ant_system(distances: np.ndarray, params: Parameters = Parameters(), seed: int | None = None) -> Result:
    """Run Ant System and return the best tour found."""
    if not 0 < params.rho <= 1:
        raise ValueError("rho must be in (0, 1]")

    rng = np.random.default_rng(seed)
    n = len(distances)
    with np.errstate(divide="ignore"):
        visibility = np.where(distances > 0, 1.0 / distances, 0.0)
    pheromones = np.full((n, n), params.initial_pheromone)

    best = Result(route=[], distance=float("inf"))
    for _ in range(params.iterations):
        tours = [build_tour(int(rng.integers(n)), pheromones, visibility, params, rng) for _ in range(params.ants)]
        lengths = [tour_length(tour, distances) for tour in tours]

        # Evaporate once, then let every ant of this iteration deposit pheromone.
        pheromones *= 1 - params.rho
        for tour, length in zip(tours, lengths):
            deposit = params.q / length
            for i in range(n):
                a, b = tour[i], tour[(i + 1) % n]
                pheromones[a, b] += deposit
                pheromones[b, a] += deposit

        iteration_best = int(np.argmin(lengths))
        if lengths[iteration_best] < best.distance:
            best.route, best.distance = tours[iteration_best], lengths[iteration_best]
        best.history.append(best.distance)

    best.pheromones = pheromones
    return best


def rotate_to_start(route: list[int], start: int = 0) -> list[int]:
    i = route.index(start)
    return route[i:] + route[:i]


def plot_result(result: Result, distances: np.ndarray, output: str | None = None) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    n = len(distances)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        a, b = result.route[i], result.route[(i + 1) % n]
        graph.add_edge(a, b, weight=int(distances[a, b]))

    fig, (ax_tour, ax_conv) = plt.subplots(1, 2, figsize=(13, 6))
    pos = nx.circular_layout(range(n))
    nx.draw(graph, pos, ax=ax_tour, with_labels=True, node_color="lightblue", node_size=600, edge_color="tab:red")
    nx.draw_networkx_edge_labels(graph, pos, ax=ax_tour, edge_labels=nx.get_edge_attributes(graph, "weight"), font_size=8)
    ax_tour.set_title(f"Best tour — length {result.distance:g}")

    ax_conv.plot(range(1, len(result.history) + 1), result.history, marker="o")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Best tour length")
    ax_conv.set_title("Convergence")
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=110)
    else:
        plt.show()


PRESETS = {
    "balanced": Parameters(alpha=1, beta=5, rho=0.5),
    "slow-evaporation": Parameters(alpha=2, beta=6, rho=0.01),
    "pheromone-heavy": Parameters(alpha=5, beta=5, rho=0.1),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="TSP with Ant System (ACO).")
    parser.add_argument("-m", "--matrix", default=DEFAULT_MATRIX, help="distance matrix file")
    parser.add_argument("--alpha", type=float, default=Parameters.alpha)
    parser.add_argument("--beta", type=float, default=Parameters.beta)
    parser.add_argument("--rho", type=float, default=Parameters.rho, help="evaporation rate")
    parser.add_argument("--ants", type=int, default=Parameters.ants)
    parser.add_argument("--iterations", type=int, default=Parameters.iterations)
    parser.add_argument("--seed", type=int, help="random seed for reproducible runs")
    parser.add_argument("--compare", action="store_true", help="run the three preset parameter sets")
    parser.add_argument("--no-plot", action="store_true", help="do not open the plot window")
    parser.add_argument("--save", metavar="FILE", help="save the plot to an image file instead of showing it")
    args = parser.parse_args()

    distances = load_matrix(args.matrix)

    if args.compare:
        print(f"{'preset':<18}{'alpha':>6}{'beta':>6}{'rho':>7}{'best length':>13}")
        for name, params in PRESETS.items():
            result = ant_system(distances, params, seed=args.seed)
            print(f"{name:<18}{params.alpha:>6g}{params.beta:>6g}{params.rho:>7g}{result.distance:>13g}")
        return

    params = Parameters(alpha=args.alpha, beta=args.beta, rho=args.rho, ants=args.ants, iterations=args.iterations)
    result = ant_system(distances, params, seed=args.seed)
    print("Best route:", " -> ".join(map(str, rotate_to_start(result.route) + [0])))
    print(f"Best length: {result.distance:g}")

    if args.save:
        plot_result(result, distances, output=args.save)
    elif not args.no_plot:
        plot_result(result, distances)


if __name__ == "__main__":
    main()
