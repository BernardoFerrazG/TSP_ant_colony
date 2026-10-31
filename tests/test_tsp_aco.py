import itertools

import numpy as np
import pytest

from tsp_aco import DEFAULT_MATRIX, Parameters, ant_system, load_matrix, rotate_to_start, tour_length

OPTIMUM_15_CITIES = 291  # exact optimum of distances.txt, computed with Held-Karp


def held_karp(distances):
    """Exact TSP by dynamic programming over subsets (reference for small instances)."""
    n = len(distances)
    best = {(1, 0): 0.0}
    for size in range(2, n + 1):
        for subset in itertools.combinations(range(1, n), size - 1):
            mask = 1
            for city in subset:
                mask |= 1 << city
            for last in subset:
                previous = mask & ~(1 << last)
                options = [best[(previous, k)] + distances[k][last] for k in range(n) if (previous, k) in best]
                best[(mask, last)] = min(options)
    full = (1 << n) - 1
    return min(best[(full, last)] + distances[last][0] for last in range(1, n))


def test_reference_optimum_of_sample_instance():
    assert held_karp(load_matrix(DEFAULT_MATRIX)) == OPTIMUM_15_CITIES


def test_tour_is_a_permutation_and_length_is_consistent():
    distances = load_matrix(DEFAULT_MATRIX)
    result = ant_system(distances, Parameters(iterations=5), seed=1)
    assert sorted(result.route) == list(range(len(distances)))
    assert result.distance == tour_length(result.route, distances)


def test_history_is_non_increasing():
    result = ant_system(load_matrix(DEFAULT_MATRIX), Parameters(iterations=20), seed=3)
    assert all(a >= b for a, b in zip(result.history, result.history[1:]))


def test_same_seed_gives_same_result():
    distances = load_matrix(DEFAULT_MATRIX)
    assert ant_system(distances, seed=7).route == ant_system(distances, seed=7).route


@pytest.mark.parametrize("seed", range(5))
def test_finds_optimum_on_small_random_instances(seed):
    rng = np.random.default_rng(seed)
    points = rng.random((8, 2))
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    result = ant_system(distances, Parameters(ants=10, iterations=40), seed=seed)
    assert result.distance == pytest.approx(held_karp(distances))


def test_default_parameters_get_close_to_optimum():
    result = ant_system(load_matrix(DEFAULT_MATRIX), seed=0)
    assert result.distance <= OPTIMUM_15_CITIES * 1.05


def test_rotate_to_start():
    assert rotate_to_start([3, 1, 0, 2]) == [0, 2, 3, 1]
