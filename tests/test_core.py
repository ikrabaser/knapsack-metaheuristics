"""Çekirdek altyapı testleri: problem, veri, kısıt yönetimi, uygunluk, kesin çözücüler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from core.constraints import greedy_fill, repair_population, repair_solution
from core.data_loader import build_problem, clean_dataset, dataset_summary
from core.exact import brute_force, fractional_upper_bound, greedy_solution, solve_dp, solve_milp
from core.fitness import FitnessEvaluator, InfeasibleSolutionError
from core.knapsack import KnapsackProblem


def random_problem(seed: int, n_items: int = 12, capacity_ratio: float = 0.5) -> KnapsackProblem:
    """Küçük, rastgele (tamsayı ağırlıklı) test problemi."""
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 30, n_items).astype(float)
    values = rng.integers(5, 100, n_items).astype(float)
    return KnapsackProblem.from_arrays(weights, values, capacity_ratio)


@pytest.fixture(scope="module")
def real_problem() -> KnapsackProblem:
    problem, _, _ = build_problem(config.DATA_PATH, config.CAPACITY_RATIO)
    return problem


# ----------------------------------------------------------------------------- #
# Problem ve veri seti
# ----------------------------------------------------------------------------- #
def test_real_dataset_statistics(real_problem: KnapsackProblem) -> None:
    assert real_problem.n_items == 100
    assert real_problem.total_weight == 2724.0
    assert real_problem.total_value == 5121.0


def test_capacity_is_exactly_70_percent_and_not_rounded(real_problem: KnapsackProblem) -> None:
    assert real_problem.capacity == pytest.approx(0.70 * 2724.0, abs=1e-9)
    assert real_problem.capacity == pytest.approx(1906.8, abs=1e-9)
    assert real_problem.capacity != round(real_problem.capacity)  # yuvarlama yapılmamış


@pytest.mark.parametrize(
    "weights, values",
    [([1.0, -2.0], [1.0, 1.0]), ([1.0, 2.0], [1.0, 0.0]), ([1.0], [1.0, 2.0]), ([], []),
     ([1.0, np.nan], [1.0, 1.0])],
)
def test_invalid_problem_rejected(weights: list[float], values: list[float]) -> None:
    with pytest.raises(ValueError):
        KnapsackProblem.from_arrays(np.array(weights), np.array(values), 0.7)


def test_invalid_capacity_ratio_rejected() -> None:
    with pytest.raises(ValueError):
        KnapsackProblem.from_arrays(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1.5)


def test_problem_arrays_are_read_only(real_problem: KnapsackProblem) -> None:
    with pytest.raises(ValueError):
        real_problem.weights[0] = 999.0


def test_cleaning_logs_every_action() -> None:
    frame = pd.DataFrame(
        {
            "item": [1, 2, 2, 3, 4, 5],
            "weight": [5, 6, 6, -1, 7, "x"],
            "value": [10, 20, 20, 30, 0, 50],
        }
    )
    clean, log = clean_dataset(frame)
    assert list(clean["item"]) == [1, 2]
    removed = [step["removed"] for step in log["steps"]]
    assert removed == [1, 1, 2]  # NaN/sayısal-olmayan, tekrar ID, geçersiz (<=0)
    assert log["rows_final"] == 2 and log["cleaning_needed"] is True


def test_real_dataset_needs_no_cleaning() -> None:
    _, _, log = build_problem(config.DATA_PATH)
    assert log["rows_removed_total"] == 0 and log["cleaning_needed"] is False


def test_dataset_summary_shape(real_problem: KnapsackProblem) -> None:
    summary = dataset_summary(real_problem)
    assert len(summary) == 3 and summary.loc[0, "Toplam"] == 2724.0


# ----------------------------------------------------------------------------- #
# Kısıt yönetimi
# ----------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["ratio", "random", "ratio_fill"])
def test_repair_always_yields_feasible(strategy: str, real_problem: KnapsackProblem) -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = (rng.random(real_problem.n_items) < rng.uniform(0.3, 1.0)).astype(np.int8)
        repaired = repair_solution(real_problem, x, strategy, rng)
        assert real_problem.is_feasible(repaired)


def test_repair_does_not_change_feasible_solution() -> None:
    problem = random_problem(0)
    x = np.zeros(problem.n_items, dtype=np.int8)
    x[0] = 1
    assert (repair_solution(problem, x, "ratio") == x).all()


def test_ratio_repair_only_removes_and_drops_lowest_ratio_first() -> None:
    weights = np.array([10.0, 10.0, 10.0, 10.0])
    values = np.array([100.0, 10.0, 50.0, 20.0])  # oranlar: 10, 1, 5, 2
    problem = KnapsackProblem.from_arrays(weights, values, 0.5)  # C = 20
    repaired = repair_solution(problem, np.ones(4, dtype=np.int8), "ratio")
    assert repaired.tolist() == [1, 0, 1, 0]  # 1. ve 3. en kötü oranlar çıkarıldı
    original = np.array([1, 1, 1, 0], dtype=np.int8)
    assert (repair_solution(problem, original, "ratio") <= original).all()  # ürün eklenmez


def test_ratio_fill_is_maximal(real_problem: KnapsackProblem) -> None:
    rng = np.random.default_rng(3)
    x = (rng.random(real_problem.n_items) < 0.9).astype(np.int8)
    repaired = repair_solution(real_problem, x, "ratio_fill")
    slack = real_problem.capacity - real_problem.solution_weight(repaired)
    unselected = np.flatnonzero(repaired == 0)
    assert (real_problem.weights[unselected] > slack).all()


def test_random_repair_requires_rng(real_problem: KnapsackProblem) -> None:
    with pytest.raises(ValueError):
        repair_solution(real_problem, np.ones(real_problem.n_items, dtype=np.int8), "random")


def test_unknown_strategy_rejected(real_problem: KnapsackProblem) -> None:
    with pytest.raises(ValueError):
        repair_solution(real_problem, np.ones(real_problem.n_items, dtype=np.int8), "bogus")


def test_population_repair_matches_single_repair(real_problem: KnapsackProblem) -> None:
    rng = np.random.default_rng(5)
    population = (rng.random((20, real_problem.n_items)) < 0.8).astype(np.int8)
    snapshot = population.copy()
    batch = repair_population(real_problem, population, "ratio")
    for repaired_row, original_row in zip(batch, snapshot):
        assert (repaired_row == repair_solution(real_problem, original_row, "ratio")).all()
    assert (population == snapshot).all()  # girdi matrisi değiştirilmemeli


def test_greedy_fill_in_place(real_problem: KnapsackProblem) -> None:
    x = np.zeros(real_problem.n_items, dtype=np.int8)
    greedy_fill(real_problem, x)
    assert real_problem.is_feasible(x) and x.sum() > 0


# ----------------------------------------------------------------------------- #
# Uygunluk değerlendirici
# ----------------------------------------------------------------------------- #
def test_evaluator_counts_every_evaluation() -> None:
    problem = random_problem(1)
    evaluator = FitnessEvaluator(problem)
    zero = np.zeros(problem.n_items, dtype=np.int8)
    assert evaluator.evaluate(zero) == 0.0
    evaluator.evaluate_batch(np.zeros((7, problem.n_items), dtype=np.int8))
    assert evaluator.n_evaluations == 8


def test_evaluator_rejects_infeasible_when_strict() -> None:
    problem = random_problem(2)
    with pytest.raises(InfeasibleSolutionError):
        FitnessEvaluator(problem).evaluate(np.ones(problem.n_items, dtype=np.int8))
    with pytest.raises(InfeasibleSolutionError):
        FitnessEvaluator(problem).evaluate_batch(np.ones((3, problem.n_items), dtype=np.int8))


def test_evaluator_budget_is_enforced() -> None:
    problem = random_problem(3)
    evaluator = FitnessEvaluator(problem, max_evaluations=5)
    zeros = np.zeros((5, problem.n_items), dtype=np.int8)
    evaluator.evaluate_batch(zeros)
    assert evaluator.remaining == 0
    with pytest.raises(RuntimeError):
        evaluator.evaluate(zeros[0])


# ----------------------------------------------------------------------------- #
# Kesin çözücüler ve doğrulama
# ----------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(15))
def test_dp_equals_bruteforce_and_milp_on_small_instances(seed: int) -> None:
    problem = random_problem(seed, n_items=12, capacity_ratio=[0.3, 0.5, 0.7][seed % 3])
    exact_bf = brute_force(problem)
    exact_dp = solve_dp(problem)
    exact_milp = solve_milp(problem)
    assert exact_dp.value == pytest.approx(exact_bf.value)
    assert exact_milp.value == pytest.approx(exact_bf.value)
    assert problem.is_feasible(exact_dp.solution)
    assert greedy_solution(problem).value <= exact_bf.value + 1e-9
    assert fractional_upper_bound(problem) >= exact_bf.value - 1e-9


def test_dp_handles_fractional_capacity_exactly() -> None:
    weights = np.array([3.0, 3.0, 4.0])
    values = np.array([10.0, 10.0, 15.0])
    problem = KnapsackProblem(weights, values, capacity=6.99)  # kesirli C: 3+4=7 sığmaz
    assert solve_dp(problem).value == brute_force(problem).value == 20.0


def test_real_dataset_optimum_regression(real_problem: KnapsackProblem) -> None:
    dp, milp_solution = solve_dp(real_problem), solve_milp(real_problem)
    assert dp.value == milp_solution.value == 4560.0
    assert real_problem.is_feasible(dp.solution) and dp.weight <= 1906.8
    assert greedy_solution(real_problem).value == 4552.0


def test_brute_force_size_guard() -> None:
    with pytest.raises(ValueError):
        brute_force(random_problem(0, n_items=30))
