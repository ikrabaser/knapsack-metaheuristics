"""Simulated Annealing (SA) — Metropolis kriterli, bit-flip/swap komşuluklu.

Bileşenler
----------
* **Başlangıç:** rastgele ikili vektör + onarım (GA/BPSO ile aynı yöntem).
* **Komşuluk:** olasılık ``swap_probability`` ile *swap* (seçili bir ürünü
  çıkar, seçilmemiş birini ekle); aksi hâlde tek *bit-flip*. Kapasiteyi
  aşan komşu ortak onarımdan geçer.
* **Kabul olasılığı (Metropolis):** ``P = min(1, exp(Δf / T))``,
  ``Δf = f(x_yeni) − f(x_mevcut)``. Maksimizasyonda iyileşme (``Δf ≥ 0``)
  her zaman kabul edilir; kötüleşme ``exp(Δf/T)`` ile kabul edilir.
* **Soğutma:** her iterasyonda sıcaklık ``cooling_schedule`` ile güncellenir
  (geometrik: ``T_k = T_0 · α^k``).
* **İterasyon:** her sıcaklıkta ``neighbors_per_iteration`` komşu değerlendirilir
  (popülasyon tabanlı yöntemlerle benzer değerlendirme bütçesi için).

Kötü çözümlerin olasılıksal kabulü, yüksek sıcaklıkta arama uzayında geniş
dolaşımı (keşif), düşük sıcaklıkta ise açgözlü yerel aramayı (sömürü)
sağlayarak yerel optimumdan kaçışı mümkün kılar.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from algorithms.base import BaseOptimizer, RunTracker, random_feasible_solutions
from core.constraints import repair_solution
from core.fitness import FitnessEvaluator
from core.knapsack import KnapsackProblem

COOLING_SCHEDULES: tuple[str, ...] = ("geometric", "linear", "logarithmic")
MIN_TEMPERATURE: float = 1e-6


def acceptance_probability(delta: float, temperature: float) -> float:
    """Metropolis kabul olasılığı ``min(1, exp(Δf / T))`` (maksimizasyon).

    Args:
        delta: ``f(x_yeni) − f(x_mevcut)``.
        temperature: ``T > 0``.
    """
    if temperature <= 0:
        raise ValueError("temperature pozitif olmalıdır.")
    if delta >= 0:
        return 1.0
    return math.exp(delta / temperature)


def temperature_at(
    schedule: str,
    initial_temperature: float,
    cooling_rate: float,
    iteration: int,
    total_iterations: int,
) -> float:
    """``iteration``. adımdaki (0 tabanlı) sıcaklık.

    * ``geometric``: ``T_0 · α^k`` (``α = cooling_rate``)
    * ``linear``: ``T_0 · (1 − k / K)``
    * ``logarithmic``: ``T_0 / ln(k + e)``
    """
    if schedule == "geometric":
        temperature = initial_temperature * cooling_rate**iteration
    elif schedule == "linear":
        temperature = initial_temperature * (1.0 - iteration / total_iterations)
    elif schedule == "logarithmic":
        temperature = initial_temperature / math.log(iteration + math.e)
    else:
        raise ValueError(f"Bilinmeyen soğutma planı: {schedule!r}")
    return max(temperature, MIN_TEMPERATURE)


def generate_neighbor(
    problem: KnapsackProblem,
    solution: np.ndarray,
    rng: np.random.Generator,
    swap_probability: float = 0.5,
    repair_strategy: str = "ratio",
) -> np.ndarray:
    """Bit-flip veya swap ile komşu çözüm üretir ve uygun hâle getirir."""
    neighbor = np.array(solution, dtype=np.int8, copy=True)
    selected = np.flatnonzero(solution == 1)
    unselected = np.flatnonzero(solution == 0)
    if selected.size and unselected.size and rng.random() < swap_probability:
        neighbor[selected[rng.integers(selected.size)]] = 0
        neighbor[unselected[rng.integers(unselected.size)]] = 1
    else:
        position = rng.integers(problem.n_items)
        neighbor[position] ^= 1
    return repair_solution(problem, neighbor, repair_strategy, rng)  # type: ignore[arg-type]


class SimulatedAnnealing(BaseOptimizer):
    """Metropolis kriterli Simulated Annealing.

    Args:
        problem: Problem örneği.
        initial_temperature: Başlangıç sıcaklığı ``T_0``.
        cooling_rate: Geometrik soğutma katsayısı ``α`` (0 < α < 1).
        cooling_schedule: ``"geometric"``, ``"linear"`` veya ``"logarithmic"``.
        neighbors_per_iteration: Her sıcaklıkta değerlendirilen komşu sayısı.
        swap_probability: Komşu üretiminde swap operatörünün seçilme olasılığı.
        **base_kwargs: ``max_iterations``, ``max_evaluations``, ``repair_strategy``.
    """

    name = "SA"

    def __init__(
        self,
        problem: KnapsackProblem,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        cooling_schedule: str = "geometric",
        neighbors_per_iteration: int = 40,
        swap_probability: float = 0.5,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(problem, **base_kwargs)
        if initial_temperature <= 0:
            raise ValueError("initial_temperature pozitif olmalıdır.")
        if not 0.0 < cooling_rate < 1.0:
            raise ValueError("cooling_rate (0, 1) aralığında olmalıdır.")
        if cooling_schedule not in COOLING_SCHEDULES:
            raise ValueError(f"cooling_schedule {COOLING_SCHEDULES} içinde olmalıdır.")
        if neighbors_per_iteration < 1:
            raise ValueError("neighbors_per_iteration en az 1 olmalıdır.")
        if not 0.0 <= swap_probability <= 1.0:
            raise ValueError("swap_probability [0, 1] aralığında olmalıdır.")
        self.initial_temperature = float(initial_temperature)
        self.cooling_rate = float(cooling_rate)
        self.cooling_schedule = cooling_schedule
        self.neighbors_per_iteration = int(neighbors_per_iteration)
        self.swap_probability = float(swap_probability)

    def get_params(self) -> dict[str, Any]:
        return {
            "initial_temperature": self.initial_temperature,
            "cooling_rate": self.cooling_rate,
            "cooling_schedule": self.cooling_schedule,
            "neighbors_per_iteration": self.neighbors_per_iteration,
            "swap_probability": self.swap_probability,
        }

    def _optimize(
        self, rng: np.random.Generator, evaluator: FitnessEvaluator, tracker: RunTracker
    ) -> None:
        current = random_feasible_solutions(self.problem, 1, rng, self.repair_strategy)[0]
        current_value = evaluator.evaluate(current)
        tracker.update_single(current, current_value)

        for iteration in range(self.max_iterations):
            n_neighbors = min(self.neighbors_per_iteration, evaluator.remaining)
            if n_neighbors <= 0:
                break
            temperature = temperature_at(
                self.cooling_schedule,
                self.initial_temperature,
                self.cooling_rate,
                iteration,
                self.max_iterations,
            )
            worse_proposed = 0
            worse_accepted = 0
            for _ in range(n_neighbors):
                candidate = generate_neighbor(
                    self.problem, current, rng, self.swap_probability, self.repair_strategy
                )
                candidate_value = evaluator.evaluate(candidate)
                delta = candidate_value - current_value
                if delta < 0:
                    worse_proposed += 1
                if rng.random() < acceptance_probability(delta, temperature):
                    if delta < 0:
                        worse_accepted += 1
                    current, current_value = candidate, candidate_value
                    tracker.update_single(current, current_value)
            tracker.end_iteration(
                evaluator.n_evaluations,
                temperature=temperature,
                worse_accept_rate=(
                    worse_accepted / worse_proposed if worse_proposed else float("nan")
                ),
            )
