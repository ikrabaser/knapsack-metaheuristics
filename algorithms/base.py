"""Tüm algoritmalar için ortak arayüz, sonuç yapısı ve izleme araçları."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.constraints import REPAIR_STRATEGIES, repair_population
from core.fitness import FitnessEvaluator
from core.knapsack import KnapsackProblem


@dataclass
class OptimizationResult:
    """Tek bir bağımsız çalıştırmanın sonucu.

    Attributes:
        algorithm: Algoritma adı.
        seed: Rastgelelik tohumu.
        best_solution: En iyi çözüm vektörü (``int8``, 0/1).
        best_value: En iyi nesne fonksiyonu değeri.
        total_weight: Seçilen ürünlerin toplam ağırlığı.
        n_selected: Seçilen ürün sayısı.
        history: Her iterasyon sonundaki *best-so-far* değeri (uzunluk = iterasyon sayısı).
        evaluations_history: Her iterasyon sonundaki kümülatif değerlendirme sayısı.
        elapsed_time: Çalışma süresi (saniye, tanılama hesapları hariç).
        n_evaluations: Toplam nesne fonksiyonu değerlendirme sayısı.
        params: Kullanılan parametreler.
        diversity_history: Popülasyon çeşitliliği (ortalama ikili Hamming / n); SA için ``None``.
        extra_history: Algoritmaya özgü ek izler (ör. SA sıcaklığı, kabul oranı).
    """

    algorithm: str
    seed: int
    best_solution: np.ndarray
    best_value: float
    total_weight: float
    n_selected: int
    history: np.ndarray
    evaluations_history: np.ndarray
    elapsed_time: float
    n_evaluations: int
    params: dict[str, Any]
    diversity_history: np.ndarray | None = None
    extra_history: dict[str, np.ndarray] = field(default_factory=dict)

    def solution_string(self) -> str:
        """Çözüm vektörünü ``'0101...'`` biçiminde döndürür."""
        return "".join(str(int(bit)) for bit in self.best_solution)


def population_diversity(population: np.ndarray) -> float:
    """Popülasyonun ortalama ikili Hamming uzaklığı, ``n`` ile normalize edilmiş (0..1).

    ``H(i, j) = s_i + s_j - 2 <x_i, x_j>`` formülüyle ``O(N^2 n)`` yerine
    matris çarpımıyla hesaplanır.
    """
    matrix = np.asarray(population, dtype=np.float64)
    size, n_items = matrix.shape
    if size < 2:
        return 0.0
    sums = matrix.sum(axis=1)
    hamming = sums[:, None] + sums[None, :] - 2.0 * (matrix @ matrix.T)
    return float(hamming.sum() / (size * (size - 1)) / n_items)


class RunTracker:
    """Global en iyiyi ve iterasyon geçmişini tutar.

    ``diagnostic_time`` süre ölçümünden düşülür; böylece çeşitlilik gibi
    yalnızca analiz amaçlı hesaplar algoritma süresini etkilemez.
    """

    def __init__(self) -> None:
        self.best_value: float = -np.inf
        self.best_solution: np.ndarray | None = None
        self.history: list[float] = []
        self.evaluations_history: list[int] = []
        self.diversity: list[float] = []
        self.extras: dict[str, list[float]] = {}
        self.diagnostic_time: float = 0.0

    def update_single(self, solution: np.ndarray, value: float) -> bool:
        """Tek bir çözümle global en iyiyi günceller (kesin iyileşmede)."""
        if value > self.best_value:
            self.best_value = float(value)
            self.best_solution = np.array(solution, dtype=np.int8, copy=True)
            return True
        return False

    def update_batch(self, population: np.ndarray, values: np.ndarray) -> bool:
        """Bir popülasyonun en iyisiyle global en iyiyi günceller."""
        index = int(np.argmax(values))
        return self.update_single(population[index], float(values[index]))

    def end_iteration(
        self,
        n_evaluations: int,
        diversity: float | None = None,
        **extras: float,
    ) -> None:
        """Bir iterasyonun sonunu kaydeder."""
        self.history.append(self.best_value)
        self.evaluations_history.append(int(n_evaluations))
        if diversity is not None:
            self.diversity.append(float(diversity))
        for key, value in extras.items():
            self.extras.setdefault(key, []).append(float(value))

    def measure_diversity(self, population: np.ndarray) -> float:
        """Çeşitliliği hesaplar; harcanan süreyi tanılama süresine ekler."""
        start = time.perf_counter()
        value = population_diversity(population)
        self.diagnostic_time += time.perf_counter() - start
        return value


class BaseOptimizer(ABC):
    """Dört (artı hibrit) algoritmanın ortak soyut sınıfı.

    Alt sınıflar yalnızca :meth:`_optimize` metodunu uygular; tohumlama,
    sayaç, süre ölçümü, sonuç derleme ve uygunluk doğrulaması burada yapılır.

    Args:
        problem: Çözülecek problem.
        max_iterations: Ana döngü (iterasyon/nesil) sayısı.
        max_evaluations: İsteğe bağlı sert değerlendirme üst sınırı.
        repair_strategy: Kısıt onarım stratejisi (bkz. :mod:`core.constraints`).
    """

    name: str = "base"

    def __init__(
        self,
        problem: KnapsackProblem,
        max_iterations: int = 200,
        max_evaluations: int | None = None,
        repair_strategy: str = "ratio",
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations pozitif olmalıdır.")
        if repair_strategy not in REPAIR_STRATEGIES:
            raise ValueError(f"Bilinmeyen onarım stratejisi: {repair_strategy!r}")
        self.problem = problem
        self.max_iterations = int(max_iterations)
        self.max_evaluations = max_evaluations
        self.repair_strategy = repair_strategy

    # ------------------------------------------------------------------ #
    @abstractmethod
    def _optimize(
        self, rng: np.random.Generator, evaluator: FitnessEvaluator, tracker: RunTracker
    ) -> None:
        """Algoritmanın ana döngüsü; ``tracker`` üzerinden en iyiyi/geçmişi kaydeder."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Algoritmaya özgü parametreler (JSON'a yazılabilir)."""

    # ------------------------------------------------------------------ #
    def run(self, seed: int) -> OptimizationResult:
        """Tek bir bağımsız çalıştırma yapar.

        Args:
            seed: ``numpy.random.default_rng`` için tohum.

        Returns:
            Uygun (kapasiteyi aşmayan) en iyi çözümü ve tüm ölçümleri içeren sonuç.

        Raises:
            RuntimeError: Algoritma hiç çözüm üretemediyse.
            AssertionError: Raporlanan değer/ağırlık yeniden hesaplamayla tutmazsa.
        """
        rng = np.random.default_rng(seed)
        evaluator = FitnessEvaluator(self.problem, max_evaluations=self.max_evaluations)
        tracker = RunTracker()

        start = time.perf_counter()
        self._optimize(rng, evaluator, tracker)
        elapsed = time.perf_counter() - start - tracker.diagnostic_time

        if tracker.best_solution is None:
            raise RuntimeError(f"{self.name}: hiç çözüm üretilemedi.")
        best = tracker.best_solution
        # Bağımsız doğrulama: nihai çözüm gerçekten uygun ve değer tutarlı mı?
        if not self.problem.is_feasible(best):
            raise AssertionError(f"{self.name}: nihai çözüm kapasiteyi aşıyor!")
        recomputed = self.problem.solution_value(best)
        if abs(recomputed - tracker.best_value) > 1e-6:
            raise AssertionError(f"{self.name}: raporlanan değer yeniden hesapla tutmuyor.")

        params = {"max_iterations": self.max_iterations, "repair_strategy": self.repair_strategy}
        params.update(self.get_params())
        return OptimizationResult(
            algorithm=self.name,
            seed=int(seed),
            best_solution=best,
            best_value=float(tracker.best_value),
            total_weight=self.problem.solution_weight(best),
            n_selected=int(best.sum()),
            history=np.asarray(tracker.history, dtype=np.float64),
            evaluations_history=np.asarray(tracker.evaluations_history, dtype=np.int64),
            elapsed_time=float(elapsed),
            n_evaluations=int(evaluator.n_evaluations),
            params=params,
            diversity_history=(np.asarray(tracker.diversity) if tracker.diversity else None),
            extra_history={k: np.asarray(v) for k, v in tracker.extras.items()},
        )


def random_feasible_solutions(
    problem: KnapsackProblem,
    n_solutions: int,
    rng: np.random.Generator,
    strategy: str = "ratio",
    inclusion_probability: float = 0.5,
) -> np.ndarray:
    """Rastgele ikili çözümler üretir ve onarır (GA/BPSO/SA ortak başlangıcı)."""
    raw =(rng.random((n_solutions, problem.n_items)) < inclusion_probability).astype(np.int8)
    return repair_population(problem, raw, strategy=strategy, rng=rng)  # type: ignore[arg-type]
