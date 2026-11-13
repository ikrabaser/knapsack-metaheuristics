"""Meta-sezgisel yöntemlerin dışında kalan referans (kesin) çözücüler.

* :func:`solve_dp` — dinamik programlama (ağırlıklar tamsayı ise kesin optimum)
* :func:`solve_milp` — ``scipy.optimize.milp`` ile tamsayılı doğrusal programlama
  (bağımsız çapraz doğrulama)
* :func:`brute_force` — küçük örnekler için tam sayım (birim testlerde)
* :func:`fractional_upper_bound` — doğrusal gevşetme (Dantzig) üst sınırı

Bu çözücüler yalnızca referans/doğrulama içindir; algoritma karşılaştırmasına
girmez ve nesne fonksiyonu sayaçlarını etkilemez.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from core.knapsack import FEASIBILITY_TOL, KnapsackProblem


@dataclass(frozen=True)
class ExactSolution:
    """Kesin çözücü çıktısı."""

    method: str
    value: float
    solution: np.ndarray
    weight: float

    @property
    def n_selected(self) -> int:
        """Seçilen ürün sayısı."""
        return int(self.solution.sum())


def _has_integer_weights(problem: KnapsackProblem) -> bool:
    return bool(np.allclose(problem.weights, np.round(problem.weights)))


def solve_dp(problem: KnapsackProblem) -> ExactSolution:
    """Dinamik programlama ile gerçek optimumu hesaplar.

    Ağırlıklar tamsayı olduğunda, ``C`` kesirli olsa bile uygun küme
    ``floor(C)`` kapasitesi ile aynıdır; bu nedenle DP kesin sonuç verir
    (kapasite yalnızca DP tablosunun boyutu için tabana yuvarlanır, problem
    tanımında yuvarlama yoktur).

    Zaman karmaşıklığı ``O(n * floor(C))``, bellek ``O(n * floor(C))`` bit.

    Raises:
        ValueError: Ağırlıklar tamsayı değilse (bu durumda :func:`solve_milp` kullanın).
    """
    if not _has_integer_weights(problem):
        raise ValueError("DP yalnızca tamsayı ağırlıklar için kesindir.")
    n = problem.n_items
    capacity = int(np.floor(problem.capacity + FEASIBILITY_TOL))
    weights = np.round(problem.weights).astype(np.int64)
    values = problem.values

    best = np.zeros(capacity + 1, dtype=np.float64)
    keep = np.zeros((n, capacity + 1), dtype=bool)
    for item in range(n):
        weight = int(weights[item])
        if weight > capacity:
            continue
        candidate = best[: capacity + 1 - weight] + values[item]
        improved = candidate > best[weight:]
        keep[item, weight:] = improved
        best[weight:] = np.where(improved, candidate, best[weight:])

    solution = np.zeros(n, dtype=np.int8)
    remaining = capacity
    for item in range(n - 1, -1, -1):
        if keep[item, remaining]:
            solution[item] = 1
            remaining -= int(weights[item])
    value = float(problem.values @ solution)
    if abs(value - best[capacity]) > 1e-6:
        raise AssertionError("DP geri izleme tutarsızlığı.")
    return ExactSolution("DP", value, solution, problem.solution_weight(solution))


def solve_milp(problem: KnapsackProblem, time_limit: float = 60.0) -> ExactSolution:
    """``scipy.optimize.milp`` (HiGHS) ile tamsayılı doğrusal program çözer."""
    n = problem.n_items
    result = milp(
        c=-problem.values,
        constraints=LinearConstraint(problem.weights[None, :], -np.inf, problem.capacity),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit},
    )
    if result.x is None or result.status not in (0,):
        raise RuntimeError(f"MILP optimum bulamadı: {result.message}")
    solution = np.round(result.x).astype(np.int8)
    return ExactSolution(
        "MILP", float(problem.values @ solution), solution, problem.solution_weight(solution)
    )


def brute_force(problem: KnapsackProblem, max_items: int = 22) -> ExactSolution:
    """Tüm ``2^n`` çözümü sayarak kesin optimumu bulur (yalnızca küçük ``n``)."""
    n = problem.n_items
    if n > max_items:
        raise ValueError(f"brute_force en fazla {max_items} ürün için tasarlanmıştır.")
    codes = np.arange(2**n, dtype=np.int64)
    bits = ((codes[:, None] >> np.arange(n)) & 1).astype(np.float64)
    weights = bits @ problem.weights
    values = bits @ problem.values
    values[weights > problem.capacity + FEASIBILITY_TOL] = -np.inf
    best_code = int(np.argmax(values))
    solution = bits[best_code].astype(np.int8)
    return ExactSolution(
        "brute-force", float(values[best_code]), solution, problem.solution_weight(solution)
    )


def fractional_upper_bound(problem: KnapsackProblem) -> float:
    """Doğrusal gevşetme (kesirli sırt çantası) üst sınırı.

    Ürünler değer/ağırlık oranıyla büyükten küçüğe doldurulur; son ürün kesirli
    alınır. Optimum değer bu sınırı aşamaz.
    """
    order = problem.ratio_order_ascending[::-1]
    remaining = problem.capacity
    bound = 0.0
    for item in order:
        if problem.weights[item] <= remaining:
            bound += problem.values[item]
            remaining -= problem.weights[item]
        else:
            bound += problem.values[item] * remaining / problem.weights[item]
            break
    return float(bound)


def greedy_solution(problem: KnapsackProblem) -> ExactSolution:
    """Oran-açgözlü sezgisel (referans için): oran sırasıyla sığanları ekler."""
    solution = np.zeros(problem.n_items, dtype=np.int8)
    slack = problem.capacity
    for item in problem.ratio_order_ascending[::-1]:
        if problem.weights[item] <= slack + FEASIBILITY_TOL:
            solution[item] = 1
            slack -= problem.weights[item]
    return ExactSolution(
        "greedy-ratio",
        float(problem.values @ solution),
        solution,
        problem.solution_weight(solution),
    )
