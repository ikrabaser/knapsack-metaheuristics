"""Algoritma testleri: doğruluk, uygunluk, bütçe sayımı, tekrarlanabilirlik, operatörler.

Küçük örneklerde (``n = 14``) brute-force optimumu referans alınır.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from algorithms import (
    ALGORITHM_REGISTRY,
    AntColonyOptimization,
    BinaryPSO,
    GeneticAlgorithm,
    HybridGASA,
    SimulatedAnnealing,
)
from algorithms.base import RunTracker, population_diversity
from algorithms.binary_pso import sigmoid
from algorithms.simulated_annealing import acceptance_probability, generate_neighbor, temperature_at
from core.constraints import repair_population
from core.exact import brute_force
from core.fitness import FitnessEvaluator
from core.knapsack import FEASIBILITY_TOL, KnapsackProblem

ITERATIONS = 60
SEEDS = (11, 12, 13, 14, 15)


def small_problem(seed: int = 7, n_items: int = 14) -> KnapsackProblem:
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 30, n_items).astype(float)
    values = rng.integers(5, 100, n_items).astype(float)
    return KnapsackProblem.from_arrays(weights, values, 0.5)


def make(name: str, problem: KnapsackProblem, **overrides: object):
    """Test için küçük, hızlı parametrelerle algoritma örneği."""
    base: dict[str, dict[str, object]] = {
        "GA": dict(population_size=30, mutation_probability=1 / problem.n_items),
        "ACO": dict(n_ants=20),
        "BPSO": dict(swarm_size=30, inertia=0.9),
        "SA": dict(neighbors_per_iteration=20),
        "GA-SA": dict(mutation_probability=1 / problem.n_items),
    }
    params = {**base[name], **overrides}
    return ALGORITHM_REGISTRY[name](problem, max_iterations=ITERATIONS, **params)


def expected_evaluations(name: str, optimizer) -> int:
    """Her algoritmanın tasarımından türetilen beklenen değerlendirme sayısı."""
    if name == "GA":
        return optimizer.population_size + ITERATIONS * (optimizer.population_size - optimizer.elitism_size)
    if name == "ACO":
        return optimizer.n_ants * ITERATIONS
    if name == "BPSO":
        return optimizer.swarm_size * (ITERATIONS + 1)
    if name == "SA":
        return 1 + optimizer.neighbors_per_iteration * ITERATIONS
    if name == "GA-SA":
        return optimizer.population_size + ITERATIONS * optimizer.evaluations_per_generation
    raise AssertionError(name)


ALL = ["GA", "ACO", "BPSO", "SA", "GA-SA"]


# ----------------------------------------------------------------------------- #
# Genel sözleşme: her algoritma uygun çözüm döndürmeli
# ----------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL)
def test_result_is_feasible_consistent_and_monotone(name: str) -> None:
    problem = small_problem()
    optimum = brute_force(problem).value
    for seed in SEEDS:
        result = make(name, problem).run(seed)
        assert problem.is_feasible(result.best_solution)
        assert result.total_weight <= problem.capacity + FEASIBILITY_TOL
        assert result.best_value == pytest.approx(problem.solution_value(result.best_solution))
        assert result.n_selected == int(result.best_solution.sum())
        assert result.best_value <= optimum + 1e-9  # optimumu aşamaz
        assert len(result.history) == ITERATIONS == len(result.evaluations_history)
        assert (np.diff(result.history) >= 0).all()  # best-so-far azalmaz
        assert result.history[-1] == pytest.approx(result.best_value)
        assert (np.diff(result.evaluations_history) > 0).all()
        assert result.elapsed_time > 0 and result.seed == seed


@pytest.mark.parametrize("name", ALL)
def test_evaluation_count_matches_design(name: str) -> None:
    problem = small_problem()
    optimizer = make(name, problem)
    result = optimizer.run(3)
    assert result.n_evaluations == expected_evaluations(name, optimizer)
    assert result.evaluations_history[-1] == result.n_evaluations


@pytest.mark.parametrize("name", ALL)
def test_same_seed_is_reproducible(name: str) -> None:
    problem = small_problem()
    first, second = make(name, problem).run(42), make(name, problem).run(42)
    assert (first.best_solution == second.best_solution).all()
    assert (first.history == second.history).all()
    assert first.n_evaluations == second.n_evaluations


@pytest.mark.parametrize("name", ["GA", "BPSO", "SA", "GA-SA"])
def test_different_seeds_explore_differently(name: str) -> None:
    problem = KnapsackProblem.from_arrays(
        np.random.default_rng(0).integers(1, 50, 60).astype(float),
        np.random.default_rng(1).integers(5, 100, 60).astype(float), 0.5,
    )
    histories = {tuple(make(name, problem).run(seed).history) for seed in (1, 2, 3)}
    assert len(histories) > 1


@pytest.mark.parametrize("name", ["GA", "ACO", "SA", "GA-SA"])
def test_finds_optimum_on_small_instance(name: str) -> None:
    problem = small_problem()
    optimum = brute_force(problem).value
    values = [make(name, problem).run(seed).best_value for seed in SEEDS]
    assert max(values) == pytest.approx(optimum)
    assert min(values) >= 0.9 * optimum


def test_binary_pso_reaches_good_quality_on_small_instance() -> None:
    problem = small_problem()
    optimum = brute_force(problem).value
    values = [make("BPSO", problem, inertia=1.0).run(seed).best_value for seed in SEEDS]
    assert max(values) >= 0.95 * optimum


@pytest.mark.parametrize("name", ALL)
def test_hard_evaluation_cap_is_respected_exactly(name: str) -> None:
    problem = small_problem()
    cap = 8040
    optimizer = ALGORITHM_REGISTRY[name](
        problem, max_iterations=1000, max_evaluations=cap, **_cap_params(name, problem)
    )
    assert optimizer.run(5).n_evaluations == cap


def _cap_params(name: str, problem: KnapsackProblem) -> dict[str, object]:
    return {
        "GA": dict(population_size=40),
        "ACO": dict(n_ants=40),
        "BPSO": dict(swarm_size=40),
        "SA": dict(neighbors_per_iteration=40),
        "GA-SA": dict(),
    }[name]


@pytest.mark.parametrize("strategy", ["ratio", "random", "ratio_fill"])
@pytest.mark.parametrize("name", ["GA", "BPSO", "SA", "GA-SA"])
def test_all_repair_strategies_give_feasible_results(name: str, strategy: str) -> None:
    problem = small_problem()
    result = make(name, problem, repair_strategy=strategy).run(9)
    assert problem.is_feasible(result.best_solution)


# ----------------------------------------------------------------------------- #
# Parametre doğrulama
# ----------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "factory",
    [
        lambda p: GeneticAlgorithm(p, population_size=1),
        lambda p: GeneticAlgorithm(p, mutation_probability=1.5),
        lambda p: GeneticAlgorithm(p, crossover_probability=-0.1),
        lambda p: GeneticAlgorithm(p, elitism_size=50, population_size=40),
        lambda p: GeneticAlgorithm(p, crossover_type="bogus"),
        lambda p: AntColonyOptimization(p, evaporation_rate=0.0),
        lambda p: AntColonyOptimization(p, n_ants=0),
        lambda p: AntColonyOptimization(p, reinforcement="bogus"),
        lambda p: BinaryPSO(p, v_max=0.0),
        lambda p: BinaryPSO(p, swarm_size=1),
        lambda p: SimulatedAnnealing(p, cooling_rate=1.0),
        lambda p: SimulatedAnnealing(p, initial_temperature=-5.0),
        lambda p: SimulatedAnnealing(p, cooling_schedule="bogus"),
        lambda p: HybridGASA(p, sa_evaluations_per_generation=5, refine_top_k=2),
        lambda p: GeneticAlgorithm(p, max_iterations=0),
        lambda p: GeneticAlgorithm(p, repair_strategy="bogus"),
    ],
)
def test_invalid_parameters_raise(factory) -> None:
    with pytest.raises(ValueError):
        factory(small_problem())


def test_default_mutation_probability_is_one_over_n() -> None:
    problem = small_problem()
    assert GeneticAlgorithm(problem).mutation_probability == pytest.approx(1 / problem.n_items)


# ----------------------------------------------------------------------------- #
# GA operatörleri
# ----------------------------------------------------------------------------- #
@pytest.mark.parametrize("crossover_type", ["uniform", "one_point", "two_point"])
def test_crossover_children_inherit_genes_from_parents(crossover_type: str) -> None:
    problem = small_problem(n_items=30)
    ga = GeneticAlgorithm(problem, crossover_type=crossover_type, crossover_probability=1.0)
    rng = np.random.default_rng(0)
    parents_a = rng.integers(0, 2, (50, 30)).astype(np.int8)
    parents_b = rng.integers(0, 2, (50, 30)).astype(np.int8)
    child_a, child_b = ga._crossover(parents_a, parents_b, rng)
    # her gen iki ebeveynden birine ait olmalı; iki çocuk birbirinin tamamlayıcısı olmalı
    assert ((child_a == parents_a) | (child_a == parents_b)).all()
    assert (child_a + child_b == parents_a + parents_b).all()


def test_crossover_probability_zero_copies_parents() -> None:
    problem = small_problem(n_items=30)
    ga = GeneticAlgorithm(problem, crossover_probability=0.0)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, (20, 30)).astype(np.int8)
    b = rng.integers(0, 2, (20, 30)).astype(np.int8)
    child_a, child_b = ga._crossover(a, b, rng)
    assert (child_a == a).all() and (child_b == b).all()


def test_mutation_extremes_and_expected_rate() -> None:
    problem = small_problem(n_items=50)
    rng = np.random.default_rng(0)
    children = rng.integers(0, 2, (400, 50)).astype(np.int8)
    assert (GeneticAlgorithm(problem, mutation_probability=0.0)._mutate(children, rng) == children).all()
    assert (GeneticAlgorithm(problem, mutation_probability=1.0)._mutate(children, rng) == 1 - children).all()
    flipped = GeneticAlgorithm(problem, mutation_probability=0.1)._mutate(children, rng) != children
    assert flipped.mean() == pytest.approx(0.1, abs=0.02)


def test_tournament_of_full_population_selects_best() -> None:
    problem = small_problem()
    ga = GeneticAlgorithm(problem, population_size=10, tournament_size=500)
    fitness = np.arange(10, dtype=float)
    winners = ga._tournament_select(fitness, 20, np.random.default_rng(0))
    assert (winners == 9).all()


class _PopulationRecorder(GeneticAlgorithm):
    """Her nesil sonunda popülasyonun kendi en iyi uygunluğunu kaydeden GA."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.population_best: list[float] = []

    def _after_generation(self, population, fitness, generation, rng, evaluator, tracker) -> None:
        self.population_best.append(float(fitness.max()))


def test_elitism_preserves_population_best_but_its_absence_does_not() -> None:
    problem = small_problem(n_items=30)
    heavy_mutation = dict(population_size=20, mutation_probability=0.3, max_iterations=40)
    with_elitism = _PopulationRecorder(problem, elitism_size=2, **heavy_mutation)
    with_elitism.run(4)
    assert (np.diff(with_elitism.population_best) >= 0).all()  # elitist: popülasyon en iyisi azalmaz
    without_elitism = _PopulationRecorder(problem, elitism_size=0, **heavy_mutation)
    without_elitism.run(4)
    assert (np.diff(without_elitism.population_best) < 0).any()  # testin ayırt edici gücü


# ----------------------------------------------------------------------------- #
# SA bileşenleri
# ----------------------------------------------------------------------------- #
def test_metropolis_acceptance_probability() -> None:
    assert acceptance_probability(5.0, 10.0) == 1.0
    assert acceptance_probability(0.0, 10.0) == 1.0
    assert acceptance_probability(-10.0, 10.0) == pytest.approx(math.exp(-1))
    assert acceptance_probability(-10.0, 1e-9) == pytest.approx(0.0)
    assert acceptance_probability(-10.0, 1e6) > 0.999  # yüksek sıcaklıkta neredeyse hepsi kabul
    with pytest.raises(ValueError):
        acceptance_probability(-1.0, 0.0)


def test_cooling_schedules() -> None:
    geometric = [temperature_at("geometric", 100.0, 0.9, k, 200) for k in range(5)]
    assert geometric == pytest.approx([100.0 * 0.9**k for k in range(5)])
    for schedule in ("geometric", "linear", "logarithmic"):
        temps = [temperature_at(schedule, 100.0, 0.95, k, 200) for k in range(200)]
        assert all(t > 0 for t in temps) and all(a >= b for a, b in zip(temps, temps[1:]))
    assert temperature_at("linear", 100.0, 0.95, 100, 200) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        temperature_at("bogus", 1.0, 0.9, 0, 10)


def test_neighbors_are_feasible_binary_and_close() -> None:
    problem = small_problem(n_items=30)
    rng = np.random.default_rng(2)
    current = np.zeros(30, dtype=np.int8)
    current[:5] = 1
    for swap_probability in (0.0, 1.0):
        for _ in range(100):
            neighbor = generate_neighbor(problem, current, rng, swap_probability)
            assert problem.is_feasible(neighbor)
            assert set(np.unique(neighbor)) <= {0, 1}
            assert (neighbor != current).sum() <= 3 + int(current.sum())  # tek hamle + onarım


def test_sa_accepts_some_worse_moves_when_hot_and_none_when_cold() -> None:
    problem = small_problem(n_items=40)
    hot = SimulatedAnnealing(problem, initial_temperature=1e6, cooling_rate=0.999, max_iterations=20).run(1)
    cold = SimulatedAnnealing(problem, initial_temperature=1e-4, cooling_rate=0.5, max_iterations=20).run(1)
    assert np.nanmean(hot.extra_history["worse_accept_rate"]) > 0.9
    assert np.nanmax(np.nan_to_num(cold.extra_history["worse_accept_rate"])) == 0.0


# ----------------------------------------------------------------------------- #
# Binary PSO ve ACO bileşenleri
# ----------------------------------------------------------------------------- #
def test_sigmoid_matches_definition() -> None:
    for v in (-4.0, -1.0, 0.5, 1.0, 2.5, 4.0):
        assert sigmoid(np.array([v]))[0] == pytest.approx(1.0 / (1.0 + math.exp(-v)), rel=1e-12)


def test_aco_pheromone_update_evaporation_reinforcement_and_bounds() -> None:
    problem = small_problem(n_items=6)
    best = np.array([1, 0, 0, 0, 0, 0], dtype=np.int8)
    other = np.array([0, 1, 0, 0, 0, 0], dtype=np.int8)
    ub = 100.0
    aco = AntColonyOptimization(problem, evaporation_rate=0.5, deposit_strength=1.0, reinforcement="iteration_best")
    pheromone = np.full(6, 1.0)
    aco.update_pheromone(pheromone, (best, 50.0), (other, 99.0), ub, (0.02, 2.0))
    # buharlaşma: 1.0 * (1 - 0.5) = 0.5; iterasyon-en-iyi ürüne +Q*f/UB = +0.5 -> 1.0;
    # "iteration_best" modunda global en iyi (ürün 1) pekiştirilmez.
    assert pheromone == pytest.approx([1.0, 0.5, 0.5, 0.5, 0.5, 0.5])

    both = AntColonyOptimization(problem, evaporation_rate=0.5, reinforcement="both")
    pheromone = np.full(6, 1.0)
    both.update_pheromone(pheromone, (best, 50.0), (other, 80.0), ub, (0.02, 2.0))
    # ürün 0: 0.5 + 50/100 = 1.0 ; ürün 1 (global en iyi): 0.5 + 80/100 = 1.3
    assert pheromone == pytest.approx([1.0, 1.3, 0.5, 0.5, 0.5, 0.5])

    pheromone = np.full(6, 0.03)  # buharlaşma tau_min'in altına düşürür -> kırpılmalı
    aco.update_pheromone(pheromone, (best, 0.0 + 1e-12), (best, 1e-12), ub, (0.02, 2.0))
    assert pheromone.min() == pytest.approx(0.02)
    pheromone = np.full(6, 2.0)  # büyük pekiştirme tau_max'i aşmamalı
    aco.update_pheromone(pheromone, (best, 1000.0), (best, 1000.0), ub, (0.02, 2.0))
    assert pheromone.max() == pytest.approx(2.0)


def test_sigmoid_transfer_function() -> None:
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    v = np.linspace(-20, 20, 41)
    s = sigmoid(v)
    assert (np.diff(s) > 0).all() and ((s > 0) & (s < 1)).all()
    assert s + sigmoid(-v) == pytest.approx(np.ones_like(s))
    assert sigmoid(np.array([-1e4, 1e4])).tolist() == [0.0, 1.0]  # taşma olmadan


def test_aco_construction_is_feasible_and_maximal() -> None:
    problem = small_problem(n_items=40)
    aco = AntColonyOptimization(problem)
    rng = np.random.default_rng(0)
    tau = np.ones(problem.n_items)
    ants = aco._construct_solutions(tau, aco.heuristic_information(), 100, rng)
    for ant in ants:
        assert problem.is_feasible(ant)
        slack = problem.capacity - problem.solution_weight(ant)
        assert (problem.weights[ant == 0] > slack + FEASIBILITY_TOL).all()  # hiçbir ürün sığmıyor


def test_aco_prefers_high_ratio_items_under_heuristic() -> None:
    weights = np.full(20, 10.0)
    values = np.arange(1.0, 21.0)  # ürün 19 en iyi
    problem = KnapsackProblem.from_arrays(weights, values, 0.1)  # yalnızca 2 ürün sığar
    aco = AntColonyOptimization(problem, beta=4.0)
    ants = aco._construct_solutions(np.ones(20), aco.heuristic_information(), 2000, np.random.default_rng(1))
    selection_frequency = ants.mean(axis=0)
    assert selection_frequency[-1] > selection_frequency[0] * 5


def test_aco_pheromone_bounds_and_entropy_trace() -> None:
    problem = small_problem()
    result = make("ACO", problem).run(1)
    entropy = result.extra_history["mean_pheromone_entropy"]
    assert ((entropy > 0) & (entropy <= 1 + 1e-12)).all()


def test_population_diversity_measure() -> None:
    same = np.tile(np.array([[1, 0, 1, 1]], dtype=np.int8), (5, 1))
    assert population_diversity(same) == 0.0
    opposite = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int8)
    assert population_diversity(opposite) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- #
# Hibrit
# ----------------------------------------------------------------------------- #
def test_hybrid_budget_split_and_degenerate_case() -> None:
    problem = small_problem()
    hybrid = HybridGASA(problem, evaluations_per_generation=40, sa_evaluations_per_generation=12, elitism_size=2)
    assert hybrid.population_size == 30  # 28 çocuk + 2 elit
    assert hybrid._offspring_per_generation() + hybrid.sa_evaluations_per_generation == 40
    pure_ga = HybridGASA(problem, sa_evaluations_per_generation=0, max_iterations=30)
    assert pure_ga.run(1).n_evaluations == pure_ga.population_size + 30 * 40


def test_hybrid_sa_refinement_never_worsens_individuals() -> None:
    problem = small_problem(n_items=40)
    hybrid = HybridGASA(problem, max_iterations=3)
    rng = np.random.default_rng(0)
    population = np.random.default_rng(1).integers(0, 2, (30, 40)).astype(np.int8)
    population = repair_population(problem, population, "ratio")
    evaluator = FitnessEvaluator(problem)
    fitness = evaluator.evaluate_batch(population)
    before = fitness.copy()
    hybrid._after_generation(population, fitness, 0, rng, evaluator, RunTracker())
    assert (fitness >= before).all()
    assert evaluator.n_evaluations == 30 + hybrid.sa_evaluations_per_generation
    assert all(problem.is_feasible(row) for row in population)
