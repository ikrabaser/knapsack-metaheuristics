"""Genetik Algoritma (GA) — ikili kromozomlu, turnuva seçimli, elitist.

Bileşenler
----------
* **Temsil:** ``n`` uzunlukta ikili kromozom; ``x_i = 1`` ürün ``i`` seçili.
* **Başlangıç:** her gen 0.5 olasılıkla 1; ardından ortak onarım.
* **Uygunluk:** seçilen ürünlerin toplam değeri (onarımdan sonra, daima uygun).
* **Seçim:** boyutu ``tournament_size`` olan turnuva seçimi.
* **Çaprazlama:** ``crossover_probability`` ile tek-nokta / iki-nokta / tekdüze.
* **Mutasyon:** her gen bağımsız olarak ``mutation_probability`` ile çevrilir (bit-flip).
* **Elitizm:** en iyi ``elitism_size`` birey değerlendirilmeden aynen aktarılır.
* **Kısıt:** çocuklar onarım mekanizmasından geçer.

Mutasyon, keşif–sömürü dengesinin ana ayarıdır: ``p_m`` küçükken popülasyon
çaprazlama ile hızla homojenleşir (sömürü baskın, erken yakınsama riski);
``p_m`` büyüdükçe her çocukta ortalama ``n * p_m`` gen değişir ve iyi
çözümlerin yapısı bozulmaya başlar (keşif artar, sömürü zayıflar).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algorithms.base import BaseOptimizer, RunTracker, random_feasible_solutions
from core.constraints import repair_population
from core.fitness import FitnessEvaluator
from core.knapsack import KnapsackProblem

CROSSOVER_TYPES: tuple[str, ...] = ("uniform", "one_point", "two_point")


class GeneticAlgorithm(BaseOptimizer):
    """Elitist, turnuva seçimli ikili Genetik Algoritma.

    Args:
        problem: Problem örneği.
        population_size: Popülasyon büyüklüğü ``N``.
        crossover_probability: Bir çiftin çaprazlanma olasılığı ``p_c``.
        mutation_probability: Gen başına bit-flip olasılığı ``p_m``
            (``None`` ise ``1/n``).
        tournament_size: Turnuva boyutu ``k``.
        elitism_size: Doğrudan aktarılan elit birey sayısı.
        crossover_type: ``"uniform"``, ``"one_point"`` veya ``"two_point"``.
        init_probability: Başlangıçta bir genin 1 olma olasılığı.
        **base_kwargs: ``max_iterations``, ``max_evaluations``, ``repair_strategy``.
    """

    name = "GA"

    def __init__(
        self,
        problem: KnapsackProblem,
        population_size: int = 40,
        crossover_probability: float = 0.9,
        mutation_probability: float | None = None,
        tournament_size: int = 3,
        elitism_size: int = 2,
        crossover_type: str = "uniform",
        init_probability: float = 0.5,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(problem, **base_kwargs)
        if mutation_probability is None:
            mutation_probability = 1.0 / problem.n_items
        if population_size < 2:
            raise ValueError("population_size en az 2 olmalıdır.")
        if not 0.0 <= crossover_probability <= 1.0:
            raise ValueError("crossover_probability [0, 1] aralığında olmalıdır.")
        if not 0.0 <= mutation_probability <= 1.0:
            raise ValueError("mutation_probability [0, 1] aralığında olmalıdır.")
        if tournament_size < 1:
            raise ValueError("tournament_size en az 1 olmalıdır.")
        if not 0 <= elitism_size < population_size:
            raise ValueError("elitism_size [0, population_size) aralığında olmalıdır.")
        if crossover_type not in CROSSOVER_TYPES:
            raise ValueError(f"crossover_type {CROSSOVER_TYPES} içinde olmalıdır.")
        if not 0.0 < init_probability < 1.0:
            raise ValueError("init_probability (0, 1) aralığında olmalıdır.")
        self.population_size = int(population_size)
        self.crossover_probability = float(crossover_probability)
        self.mutation_probability = float(mutation_probability)
        self.tournament_size = int(tournament_size)
        self.elitism_size = int(elitism_size)
        self.crossover_type = crossover_type
        self.init_probability = float(init_probability)

    def get_params(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "crossover_probability": self.crossover_probability,
            "mutation_probability": self.mutation_probability,
            "tournament_size": self.tournament_size,
            "elitism_size": self.elitism_size,
            "crossover_type": self.crossover_type,
            "init_probability": self.init_probability,
        }

    # ------------------------------------------------------------------ #
    # Genetik operatörler
    # ------------------------------------------------------------------ #
    def _tournament_select(
        self, fitness: np.ndarray, n_select: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Yerine koymalı turnuva seçimi; kazanan indekslerini döndürür."""
        contenders = rng.integers(0, fitness.size, size=(n_select, self.tournament_size))
        winners = np.argmax(fitness[contenders], axis=1)
        return contenders[np.arange(n_select), winners]

    def _crossover(
        self, parents_a: np.ndarray, parents_b: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ebeveyn çiftlerinden iki çocuk matrisi üretir."""
        n_pairs, n_items = parents_a.shape
        positions = np.arange(n_items)[None, :]
        if self.crossover_type == "uniform":
            take_first = rng.random((n_pairs, n_items)) < 0.5
        elif self.crossover_type == "one_point":
            cut = rng.integers(1, n_items, size=n_pairs)
            take_first = positions < cut[:, None]
        else:  # two_point
            first = rng.integers(0, n_items, size=n_pairs)
            second = rng.integers(0, n_items, size=n_pairs)
            low, high = np.minimum(first, second), np.maximum(first, second)
            take_first = ~((positions >= low[:, None]) & (positions < high[:, None]))
        crossed = rng.random(n_pairs) < self.crossover_probability
        take_first = np.where(crossed[:, None], take_first, True)  # çaprazlanmayanlar kopyalanır
        child_a = np.where(take_first, parents_a, parents_b)
        child_b = np.where(take_first, parents_b, parents_a)
        return child_a, child_b

    def _mutate(self, children: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Bit-flip mutasyonu: her gen bağımsız olarak ``p_m`` ile çevrilir."""
        flips = (rng.random(children.shape) < self.mutation_probability).astype(np.int8)
        return np.bitwise_xor(children.astype(np.int8), flips)

    def _make_offspring(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        n_offspring: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Seçim + çaprazlama + mutasyon + onarım ile ``n_offspring`` çocuk üretir."""
        n_pairs = (n_offspring + 1) // 2
        parent_index = self._tournament_select(fitness, 2 * n_pairs, rng)
        parents_a = population[parent_index[:n_pairs]]
        parents_b = population[parent_index[n_pairs:]]
        child_a, child_b = self._crossover(parents_a, parents_b, rng)
        children = np.vstack([child_a, child_b])[:n_offspring]
        children = self._mutate(children, rng)
        return repair_population(self.problem, children, self.repair_strategy, rng)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Kancalar (hibrit alt sınıf tarafından genişletilir)
    # ------------------------------------------------------------------ #
    def _offspring_per_generation(self) -> int:
        """Bir nesilde üretilen/değerlendirilen çocuk sayısı."""
        return self.population_size - self.elitism_size

    def _after_generation(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        generation: int,
        rng: np.random.Generator,
        evaluator: FitnessEvaluator,
        tracker: RunTracker,
    ) -> None:
        """Nesil sonrası kanca (varsayılan: hiçbir şey yapmaz). Yerinde değiştirir."""

    # ------------------------------------------------------------------ #
    def _optimize(
        self, rng: np.random.Generator, evaluator: FitnessEvaluator, tracker: RunTracker
    ) -> None:
        if evaluator.remaining < self.population_size:
            raise ValueError("Değerlendirme bütçesi başlangıç popülasyonuna yetmiyor.")
        population = random_feasible_solutions(
            self.problem, self.population_size, rng, self.repair_strategy, self.init_probability
        )
        fitness = evaluator.evaluate_batch(population)
        tracker.update_batch(population, fitness)

        for generation in range(self.max_iterations):
            n_offspring = min(self._offspring_per_generation(), evaluator.remaining)
            if n_offspring <= 0:
                break
            children = self._make_offspring(population, fitness, n_offspring, rng)
            child_fitness = evaluator.evaluate_batch(children)
            tracker.update_batch(children, child_fitness)

            elite_index = np.argsort(-fitness, kind="stable")[: self.elitism_size]
            population = np.vstack([population[elite_index], children])
            fitness = np.concatenate([fitness[elite_index], child_fitness])

            self._after_generation(population, fitness, generation, rng, evaluator, tracker)
            tracker.end_iteration(
                evaluator.n_evaluations, diversity=tracker.measure_diversity(population)
            )
