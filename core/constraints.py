"""Kısıt yönetimi: onarım (repair) mekanizmaları.

Tüm algoritmalar için **tek ve tutarlı** bir kısıt stratejisi kullanılır:
kapasiteyi aşan bir çözüm, nesne fonksiyonu hesaplanmadan *önce* onarılır.
Onarım çözüme geri yazılır (Lamarckian onarım).

Desteklenen stratejiler:

``"ratio"`` (varsayılan)
    Kapasite aşılıyorsa, seçili ürünler arasından **değer/ağırlık oranı en düşük**
    olanlar kapasite sağlanana dek çıkarılır. Çıkarma dışında ürün eklenmez.
``"random"``
    Çıkarılacak ürünler rastgele sırayla seçilir (sezgisel bilgi kullanmaz).
``"ratio_fill"``
    ``"ratio"`` onarımının ardından oran sırasıyla sığan ürünler eklenir
    (çözüm "maksimal" hâle gelir). Algoritmalara en çok sezgisel bilgi
    sağlayan, ablasyon çalışmasında kullanılan güçlü stratejidir.

ACO ve SA'nın komşu/çözüm üretimi kısıtı zaten gözeterek yapılır (ACO) ya da
aynı onarımı kullanır (SA); bu nedenle strateji karşılaştırması GA ve BPSO
üzerinde bir ablasyon olarak raporlanır.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from core.knapsack import FEASIBILITY_TOL, KnapsackProblem

RepairStrategy = Literal["ratio", "random", "ratio_fill"]
REPAIR_STRATEGIES: tuple[str, ...] = ("ratio", "random", "ratio_fill")


def is_feasible(problem: KnapsackProblem, solution: np.ndarray) -> bool:
    """``KnapsackProblem.is_feasible`` için işlevsel kısayol."""
    return problem.is_feasible(solution)


def _drop_until_feasible(
    problem: KnapsackProblem, solution: np.ndarray, drop_order: np.ndarray
) -> None:
    """``solution`` üzerinde (yerinde) ``drop_order`` sırasıyla ürün çıkarır."""
    excess = float(problem.weights @ solution) - problem.capacity
    if excess <= FEASIBILITY_TOL:
        return
    selected_in_order = drop_order[solution[drop_order] == 1]
    cumulative_removed = np.cumsum(problem.weights[selected_in_order])
    # çıkarılan toplam ağırlığın fazlalığı karşıladığı ilk konum
    cut = int(np.searchsorted(cumulative_removed, excess - FEASIBILITY_TOL, side="left"))
    cut = min(cut + 1, selected_in_order.size)
    solution[selected_in_order[:cut]] = 0


def greedy_fill(problem: KnapsackProblem, solution: np.ndarray) -> None:
    """Oran sırasıyla (büyükten küçüğe) kapasiteye sığan ürünleri ekler (yerinde)."""
    slack = problem.capacity - float(problem.weights @ solution)
    for item in problem.ratio_order_ascending[::-1]:
        if solution[item] == 0 and problem.weights[item] <= slack + FEASIBILITY_TOL:
            solution[item] = 1
            slack -= problem.weights[item]


def repair_solution(
    problem: KnapsackProblem,
    solution: np.ndarray,
    strategy: RepairStrategy = "ratio",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Çözümün kapasiteyi aşmayan bir kopyasını döndürür.

    Args:
        problem: Problem örneği.
        solution: İkili çözüm vektörü (0/1).
        strategy: ``"ratio"``, ``"random"`` veya ``"ratio_fill"``.
        rng: ``"random"`` stratejisi için rastgele sayı üreteci.

    Returns:
        ``int8`` tipinde, uygun (feasible) çözüm. Girdi değiştirilmez.
    """
    if strategy not in REPAIR_STRATEGIES:
        raise ValueError(f"Bilinmeyen onarım stratejisi: {strategy!r}")
    repaired = np.array(solution, dtype=np.int8, copy=True)
    if strategy == "random":
        if rng is None:
            raise ValueError("'random' onarım stratejisi için rng gereklidir.")
        _drop_until_feasible(problem, repaired, rng.permutation(problem.n_items))
    else:
        _drop_until_feasible(problem, repaired, problem.ratio_order_ascending)
        if strategy == "ratio_fill":
            greedy_fill(problem, repaired)
    return repaired


def repair_population(
    problem: KnapsackProblem,
    population: np.ndarray,
    strategy: RepairStrategy = "ratio",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Popülasyon matrisinin (``N x n``) her satırını gerekirse onarır.

    Yalnızca kapasiteyi aşan satırlar (``"ratio_fill"`` stratejisinde tüm
    satırlar) işlenir. Girdi değiştirilmez; yeni ``int8`` matris döndürülür.
    """
    repaired = np.array(population, dtype=np.int8, copy=True)
    if strategy not in REPAIR_STRATEGIES:
        raise ValueError(f"Bilinmeyen onarım stratejisi: {strategy!r}")
    if strategy == "random" and rng is None:
        raise ValueError("'random' onarım stratejisi için rng gereklidir.")
    total_weights = repaired.astype(np.float64) @ problem.weights
    overweight_rows = np.flatnonzero(total_weights > problem.capacity + FEASIBILITY_TOL)
    rows = np.arange(repaired.shape[0]) if strategy == "ratio_fill" else overweight_rows
    for row in rows:
        if strategy == "random":
            order = rng.permutation(problem.n_items)  # type: ignore[union-attr]
        else:
            order = problem.ratio_order_ascending
        _drop_until_feasible(problem, repaired[row], order)
        if strategy == "ratio_fill":
            greedy_fill(problem, repaired[row])
    return repaired
