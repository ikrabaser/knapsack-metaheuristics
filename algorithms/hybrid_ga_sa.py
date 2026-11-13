"""GA–SA Hibriti (memetik algoritma) — bonus yöntem.

Fikir: **GA geniş çözüm uzayını tarar, SA ise GA'nın bulduğu iyi çözümleri
yerel olarak iyileştirir.** Her nesil iki aşamadan oluşur:

1. **GA aşaması (küresel keşif):** turnuva seçimi, çaprazlama, bit-flip
   mutasyonu ve onarımla ``N - elitizm`` çocuk üretilir.
2. **SA aşaması (yerel sömürü):** nesildeki en iyi ``refine_top_k`` birey için
   kısa bir SA zinciri (``steps`` komşu değerlendirmesi) çalıştırılır.
   Zincir boyunca bulunan **en iyi** çözüm bireyin yerine geçer
   (Lamarckian yerleştirme; birey asla kötüleşmez).

SA'nın sıcaklığı **nesil sayacıyla** soğur: ``T_g = T_0 * alpha^g``. Erken
nesillerde ılımlı sıcaklık küçük çukurlardan kaçmaya izin verir, geç
nesillerde zincir saf yerel aramaya (hill-climbing) dönüşür.

Adil bütçe: nesil başına toplam değerlendirme
``(N - elitizm) + refine_top_k * steps = evaluations_per_generation``
(varsayılan 40; GA/PSO/ACO/SA'nın nesil başına 40 değerlendirmesiyle aynı).
``sa_evaluations_per_generation = 0`` bu hibriti saf GA'ya indirger.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.simulated_annealing import acceptance_probability, generate_neighbor
from algorithms.base import RunTracker
from core.fitness import FitnessEvaluator
from core.knapsack import KnapsackProblem


class HybridGASA(GeneticAlgorithm):
    """GA + SA yerel iyileştirmeli hibrit (memetik) algoritma.

    Args:
        problem: Problem örneği.
        evaluations_per_generation: Nesil başına toplam değerlendirme bütçesi
            (GA çocukları + SA komşuları).
        sa_evaluations_per_generation: Bunun SA aşamasına ayrılan kısmı.
            ``refine_top_k`` ile tam bölünmelidir; 0 ise saf GA.
        refine_top_k: SA ile iyileştirilen en iyi birey sayısı.
        sa_initial_temperature: SA aşamasının ``T_0`` değeri.
        sa_cooling_rate: Nesil başına geometrik soğutma katsayısı.
        swap_probability: SA komşu üretiminde swap olasılığı.
        elitism_size: Elit birey sayısı.
        **ga_kwargs: ``crossover_probability``, ``mutation_probability``,
            ``tournament_size``, ``crossover_type`` ve temel parametreler.
    """

    name = "GA-SA"

    def __init__(
        self,
        problem: KnapsackProblem,
        evaluations_per_generation: int = 40,
        sa_evaluations_per_generation: int = 12,
        refine_top_k: int = 2,
        sa_initial_temperature: float = 10.0,
        sa_cooling_rate: float = 0.95,
        swap_probability: float = 0.5,
        elitism_size: int = 2,
        **ga_kwargs: Any,
    ) -> None:
        if not 0 <= sa_evaluations_per_generation < evaluations_per_generation:
            raise ValueError("0 <= sa_evaluations_per_generation < evaluations_per_generation olmalı.")
        if refine_top_k < 1:
            raise ValueError("refine_top_k en az 1 olmalıdır.")
        if sa_evaluations_per_generation % refine_top_k != 0:
            raise ValueError("sa_evaluations_per_generation, refine_top_k ile tam bölünmelidir.")
        if sa_initial_temperature <= 0:
            raise ValueError("sa_initial_temperature pozitif olmalıdır.")
        if not 0.0 < sa_cooling_rate < 1.0:
            raise ValueError("sa_cooling_rate (0, 1) aralığında olmalıdır.")
        offspring = evaluations_per_generation - sa_evaluations_per_generation
        super().__init__(
            problem,
            population_size=offspring + elitism_size,
            elitism_size=elitism_size,
            **ga_kwargs,
        )
        if refine_top_k > self.population_size:
            raise ValueError("refine_top_k popülasyondan büyük olamaz.")
        self.evaluations_per_generation = int(evaluations_per_generation)
        self.sa_evaluations_per_generation = int(sa_evaluations_per_generation)
        self.refine_top_k = int(refine_top_k)
        self.sa_steps_per_individual = self.sa_evaluations_per_generation // self.refine_top_k
        self.sa_initial_temperature = float(sa_initial_temperature)
        self.sa_cooling_rate = float(sa_cooling_rate)
        self.swap_probability = float(swap_probability)

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params.update(
            {
                "evaluations_per_generation": self.evaluations_per_generation,
                "sa_evaluations_per_generation": self.sa_evaluations_per_generation,
                "refine_top_k": self.refine_top_k,
                "sa_steps_per_individual": self.sa_steps_per_individual,
                "sa_initial_temperature": self.sa_initial_temperature,
                "sa_cooling_rate": self.sa_cooling_rate,
                "swap_probability": self.swap_probability,
            }
        )
        return params

    def _after_generation(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        generation: int,
        rng: np.random.Generator,
        evaluator: FitnessEvaluator,
        tracker: RunTracker,
    ) -> None:
        """En iyi bireyleri kısa SA zincirleriyle iyileştirir (yerinde)."""
        if self.sa_steps_per_individual == 0:
            return
        temperature = max(self.sa_initial_temperature * self.sa_cooling_rate**generation, 1e-6)
        top_index = np.argsort(-fitness, kind="stable")[: self.refine_top_k]
        for index in top_index:
            current = population[index].copy()
            current_value = float(fitness[index])
            best_solution, best_value = current.copy(), current_value
            for _ in range(self.sa_steps_per_individual):
                if evaluator.remaining <= 0:
                    break
                candidate = generate_neighbor(
                    self.problem, current, rng, self.swap_probability, self.repair_strategy
                )
                candidate_value = evaluator.evaluate(candidate)
                delta = candidate_value - current_value
                if rng.random() < acceptance_probability(delta, temperature):
                    current, current_value = candidate, candidate_value
                if candidate_value > best_value:
                    best_solution, best_value = candidate.copy(), candidate_value
            population[index] = best_solution
            fitness[index] = best_value
            tracker.update_single(best_solution, best_value)
