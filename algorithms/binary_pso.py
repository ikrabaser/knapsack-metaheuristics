"""İkili (Binary) Parçacık Sürü Optimizasyonu — Kennedy & Eberhart (1997).

Hız güncellemesi (sürekli PSO ile aynı)::

    v_id <- w * v_id + c1 * r1 * (pBest_id - x_id) + c2 * r2 * (gBest_d - x_id)
    v_id <- clip(v_id, -v_max, +v_max)

Konum güncellemesi (ikili PSO'ya özgü) — hız bir *olasılığa* dönüştürülür::

    S(v) = 1 / (1 + exp(-v))            (sigmoid transfer fonksiyonu)
    x_id <- 1   eğer  rand() < S(v_id),   aksi hâlde 0

Sürekli PSO'dan farkı: ``x <- x + v`` ile konum doğrudan kaydırılmaz; hız,
bitin 1 olma *olasılığını* belirler. Dolayısıyla ``w`` (atalet) konum
momentumunu değil, **bit-olasılığının değişim eğilimini** korur. ``v ≈ 0``
iken ``S(v) = 0.5`` olduğundan, sürü yakınsarken hız sönümlendiğinde bitler
rastgele çevrilmeye başlar; küçük ``w`` bu gürültüyü artırır.

Kapasite kısıtı konum güncellemesinden sonra ortak onarımla sağlanır.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algorithms.base import BaseOptimizer, RunTracker, random_feasible_solutions
from core.constraints import repair_population
from core.fitness import FitnessEvaluator
from core.knapsack import KnapsackProblem


def sigmoid(velocity: np.ndarray) -> np.ndarray:
    """Sayısal olarak kararlı sigmoid transfer fonksiyonu ``S(v) = 1/(1+e^{-v})``."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(velocity, dtype=np.float64)))


class BinaryPSO(BaseOptimizer):
    """Sigmoid transfer fonksiyonlu ikili PSO.

    Args:
        problem: Problem örneği.
        swarm_size: Parçacık sayısı.
        inertia: Atalet katsayısı ``w``.
        c1: Bilişsel (kişisel en iyi) katsayı.
        c2: Sosyal (global en iyi) katsayı.
        v_max: Hız sınırı ``|v| <= v_max`` (sigmoidin doymasını önler).
        init_probability: Başlangıç konumunda bir bitin 1 olma olasılığı.
        **base_kwargs: ``max_iterations``, ``max_evaluations``, ``repair_strategy``.
    """

    name = "BPSO"

    def __init__(
        self,
        problem: KnapsackProblem,
        swarm_size: int = 40,
        inertia: float = 0.7,
        c1: float = 2.0,
        c2: float = 2.0,
        v_max: float = 4.0,
        init_probability: float = 0.5,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(problem, **base_kwargs)
        if swarm_size < 2:
            raise ValueError("swarm_size en az 2 olmalıdır.")
        if inertia < 0:
            raise ValueError("inertia negatif olamaz.")
        if c1 < 0 or c2 < 0:
            raise ValueError("c1 ve c2 negatif olamaz.")
        if v_max <= 0:
            raise ValueError("v_max pozitif olmalıdır.")
        if not 0.0 < init_probability < 1.0:
            raise ValueError("init_probability (0, 1) aralığında olmalıdır.")
        self.swarm_size = int(swarm_size)
        self.inertia = float(inertia)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.v_max = float(v_max)
        self.init_probability = float(init_probability)

    def get_params(self) -> dict[str, Any]:
        return {
            "swarm_size": self.swarm_size,
            "inertia": self.inertia,
            "c1": self.c1,
            "c2": self.c2,
            "v_max": self.v_max,
            "init_probability": self.init_probability,
        }

    def _optimize(
        self, rng: np.random.Generator, evaluator: FitnessEvaluator, tracker: RunTracker
    ) -> None:
        size, n_items = self.swarm_size, self.problem.n_items
        if evaluator.remaining < size:
            raise ValueError("Değerlendirme bütçesi başlangıç sürüsüne yetmiyor.")

        positions = random_feasible_solutions(
            self.problem, size, rng, self.repair_strategy, self.init_probability
        )
        velocities = rng.uniform(-1.0, 1.0, size=(size, n_items))
        fitness = evaluator.evaluate_batch(positions)
        tracker.update_batch(positions, fitness)

        personal_best = positions.copy()
        personal_best_fitness = fitness.copy()
        leader = int(np.argmax(personal_best_fitness))
        global_best = personal_best[leader].copy()
        global_best_fitness = float(personal_best_fitness[leader])

        for _ in range(self.max_iterations):
            if evaluator.remaining < size:
                break
            r1 = rng.random((size, n_items))
            r2 = rng.random((size, n_items))
            velocities = (
                self.inertia * velocities
                + self.c1 * r1 * (personal_best - positions)
                + self.c2 * r2 * (global_best[None, :] - positions)
            )
            np.clip(velocities, -self.v_max, self.v_max, out=velocities)

            probabilities = sigmoid(velocities)
            positions = (rng.random((size, n_items)) < probabilities).astype(np.int8)
            positions = repair_population(
                self.problem, positions, self.repair_strategy, rng  # type: ignore[arg-type]
            )
            fitness = evaluator.evaluate_batch(positions)
            tracker.update_batch(positions, fitness)

            improved = fitness > personal_best_fitness
            personal_best[improved] = positions[improved]
            personal_best_fitness[improved] = fitness[improved]
            leader = int(np.argmax(personal_best_fitness))
            if personal_best_fitness[leader] > global_best_fitness:
                global_best = personal_best[leader].copy()
                global_best_fitness = float(personal_best_fitness[leader])

            tracker.end_iteration(
                evaluator.n_evaluations, diversity=tracker.measure_diversity(positions)
            )
