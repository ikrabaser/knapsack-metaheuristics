"""Karınca Kolonisi Optimizasyonu (ACO) — 0/1 Knapsack uyarlaması.

Klasik rota problemlerinden farklar
-----------------------------------
TSP gibi problemlerde feromon **kenarlara** (şehir çiftlerine) konur ve bir
karınca şehirleri *sıralı bir tur* olarak gezer. 0/1 Knapsack'te ise:

* çözüm bir **küme**dir (sıra önemsizdir); bu yüzden feromon **ürünlere**
  (``tau_i``: "ürün ``i`` iyi çözümlerde yer alır") konur,
* karınca bir tur çizmez; **boş çantadan** başlayarak her adımda henüz seçilmemiş
  ve **kalan kapasiteye sığan** adaylar arasından bir ürün ekler,
* aday kalmadığında inşa biter (çözüm yapısı gereği uygundur ve "maksimaldir";
  onarıma gerek yoktur),
* sezgisel bilgi mesafe tersi yerine **değer/ağırlık oranıdır**.

Olasılıksal seçim (aday kümesi ``N_k``)::

    p_i = (tau_i ** alpha) * (eta_i ** beta) / sum_{j in N_k} (tau_j ** alpha) * (eta_j ** beta)
    eta_i = (v_i / w_i) / max_j (v_j / w_j)

Feromon güncellemesi (MAX–MIN Ant System tarzı)::

    tau_i <- (1 - rho) * tau_i                      (buharlaşma)
    tau_i <- tau_i + Q * f(x_best) / UB   i in x_best  (pekiştirme)
    tau_i <- clip(tau_i, tau_min, tau_max)          (erken yakınsamayı önleyen sınırlar)

``UB`` doğrusal gevşetme üst sınırıdır (yalnızca ölçekleme için; optimum
bilgisi kullanılmaz). Pekiştirme, iterasyonun en iyi karıncası ve/veya
şimdiye dek bulunan global en iyi çözümle yapılır.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algorithms.base import BaseOptimizer, RunTracker
from core.exact import fractional_upper_bound
from core.fitness import FitnessEvaluator
from core.knapsack import FEASIBILITY_TOL, KnapsackProblem

REINFORCEMENT_MODES: tuple[str, ...] = ("iteration_best", "global_best", "both")


class AntColonyOptimization(BaseOptimizer):
    """Ürün-feromonlu, MAX–MIN tarzı ACO (0/1 Knapsack).

    Args:
        problem: Problem örneği.
        n_ants: Her iterasyonda çözüm inşa eden karınca sayısı.
        alpha: Feromon etkisinin üssü ``α``.
        beta: Sezgisel bilgi (değer/ağırlık) etkisinin üssü ``β``.
        evaporation_rate: Buharlaşma oranı ``ρ`` (0 < ρ <= 1).
        deposit_strength: Feromon pekiştirme şiddeti ``Q``.
        tau_min_ratio: ``tau_min / tau_max`` oranı (0 < oran < 1).
        reinforcement: ``"iteration_best"``, ``"global_best"`` veya ``"both"``.
        **base_kwargs: ``max_iterations``, ``max_evaluations`` (``repair_strategy``
            ACO'da kullanılmaz; inşa zaten uygun çözüm üretir).
    """

    name = "ACO"

    def __init__(
        self,
        problem: KnapsackProblem,
        n_ants: int = 40,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.3,
        deposit_strength: float = 1.0,
        tau_min_ratio: float = 0.01,
        reinforcement: str = "both",
        **base_kwargs: Any,
    ) -> None:
        super().__init__(problem, **base_kwargs)
        if n_ants < 1:
            raise ValueError("n_ants en az 1 olmalıdır.")
        if alpha < 0 or beta < 0:
            raise ValueError("alpha ve beta negatif olamaz.")
        if not 0.0 < evaporation_rate <= 1.0:
            raise ValueError("evaporation_rate (0, 1] aralığında olmalıdır.")
        if deposit_strength <= 0:
            raise ValueError("deposit_strength pozitif olmalıdır.")
        if not 0.0 < tau_min_ratio < 1.0:
            raise ValueError("tau_min_ratio (0, 1) aralığında olmalıdır.")
        if reinforcement not in REINFORCEMENT_MODES:
            raise ValueError(f"reinforcement {REINFORCEMENT_MODES} içinde olmalıdır.")
        self.n_ants = int(n_ants)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.evaporation_rate = float(evaporation_rate)
        self.deposit_strength = float(deposit_strength)
        self.tau_min_ratio = float(tau_min_ratio)
        self.reinforcement = reinforcement

    def get_params(self) -> dict[str, Any]:
        return {
            "n_ants": self.n_ants,
            "alpha": self.alpha,
            "beta": self.beta,
            "evaporation_rate": self.evaporation_rate,
            "deposit_strength": self.deposit_strength,
            "tau_min_ratio": self.tau_min_ratio,
            "reinforcement": self.reinforcement,
        }

    # ------------------------------------------------------------------ #
    def heuristic_information(self) -> np.ndarray:
        """Normalize değer/ağırlık oranı ``eta_i`` (en iyi ürün için 1)."""
        return self.problem.ratios / self.problem.ratios.max()

    def _construct_solutions(
        self, pheromone: np.ndarray, eta: np.ndarray, n_ants: int, rng: np.random.Generator
    ) -> np.ndarray:
        """``n_ants`` karıncanın çözümünü **eş zamanlı** (vektörize) inşa eder.

        Karıncalar birbirinden bağımsızdır; vektörizasyon yalnızca hızlandırma
        içindir ve her karıncanın ayrı ayrı inşa edilmesiyle aynı dağılımı verir.
        """
        problem = self.problem
        attractiveness = (pheromone**self.alpha) * (eta**self.beta)
        selected = np.zeros((n_ants, problem.n_items), dtype=np.int8)
        slack = np.full(n_ants, problem.capacity, dtype=np.float64)
        tiny = np.finfo(np.float64).tiny
        for _ in range(problem.n_items):
            candidates = (selected == 0) & (problem.weights[None, :] <= slack[:, None] + FEASIBILITY_TOL)
            scores = candidates * attractiveness[None, :]
            totals = scores.sum(axis=1)
            active = np.flatnonzero(totals > 0.0)
            if active.size == 0:
                break
            cumulative = np.cumsum(scores[active], axis=1)
            thresholds = np.maximum(rng.random(active.size) * totals[active], tiny)
            choice = np.minimum((cumulative < thresholds[:, None]).sum(axis=1), problem.n_items - 1)
            selected[active, choice] = 1
            slack[active] -= problem.weights[choice]
        return selected

    def update_pheromone(
        self,
        pheromone: np.ndarray,
        iteration_best: tuple[np.ndarray, float],
        global_best: tuple[np.ndarray, float],
        upper_bound: float,
        bounds: tuple[float, float],
    ) -> None:
        """Feromon güncellemesi (yerinde): buharlaşma, pekiştirme, sınırlama.

        ``tau <- (1-rho) tau``; ardından seçilen en iyi çözüm(ler)in ürünlerine
        ``Q * f / UB`` eklenir ve ``tau`` ``[tau_min, tau_max]`` aralığına kırpılır.

        Args:
            pheromone: Ürün başına feromon vektörü (değiştirilir).
            iteration_best: ``(çözüm, değer)`` — bu iterasyonun en iyi karıncası.
            global_best: ``(çözüm, değer)`` — şimdiye dek bulunan en iyi çözüm.
            upper_bound: Ölçekleme için doğrusal gevşetme üst sınırı ``UB``.
            bounds: ``(tau_min, tau_max)``.
        """
        pheromone *= 1.0 - self.evaporation_rate
        if self.reinforcement in ("iteration_best", "both"):
            solution, value = iteration_best
            pheromone[solution == 1] += self.deposit_strength * value / upper_bound
        if self.reinforcement in ("global_best", "both"):
            solution, value = global_best
            pheromone[solution == 1] += self.deposit_strength * value / upper_bound
        np.clip(pheromone, bounds[0], bounds[1], out=pheromone)

    def _optimize(
        self, rng: np.random.Generator, evaluator: FitnessEvaluator, tracker: RunTracker
    ) -> None:
        problem = self.problem
        eta = self.heuristic_information()
        upper_bound = fractional_upper_bound(problem)
        n_depositors = 2 if self.reinforcement == "both" else 1
        tau_max = self.deposit_strength * n_depositors / self.evaporation_rate
        tau_min = self.tau_min_ratio * tau_max
        pheromone = np.full(problem.n_items, tau_max)  # başlangıç: keşfi serbest bırakan üst sınır

        for _ in range(self.max_iterations):
            n_ants = min(self.n_ants, evaluator.remaining)
            if n_ants <= 0:
                break
            ants = self._construct_solutions(pheromone, eta, n_ants, rng)
            values = evaluator.evaluate_batch(ants)
            tracker.update_batch(ants, values)
            iteration_best = int(np.argmax(values))

            assert tracker.best_solution is not None
            self.update_pheromone(
                pheromone,
                (ants[iteration_best], float(values[iteration_best])),
                (tracker.best_solution, tracker.best_value),
                upper_bound,
                (tau_min, tau_max),
            )

            tracker.end_iteration(
                evaluator.n_evaluations,
                diversity=tracker.measure_diversity(ants),
                mean_pheromone_entropy=_normalized_entropy(pheromone),
            )


def _normalized_entropy(pheromone: np.ndarray) -> float:
    """Feromon dağılımının normalize Shannon entropisi (1: düzgün, 0: tek ürüne yığılma)."""
    probabilities = pheromone / pheromone.sum()
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-300))
    return float(entropy / np.log(pheromone.size))
