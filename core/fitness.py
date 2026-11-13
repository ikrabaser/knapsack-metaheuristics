"""Ortak uygunluk (fitness) değerlendirme altyapısı.

Dört (artı hibrit) algoritmanın tamamı nesne fonksiyonunu **bu sınıf üzerinden**
hesaplar. Böylece:

* amaç fonksiyonu tüm algoritmalar için özdeştir,
* her değerlendirme sayılır (başlangıç popülasyonu dâhil),
* kapasiteyi aşan bir çözüm değerlendirilmeye çalışılırsa hata verilir
  (nihai sonuçların gerçekten uygun olduğunu garanti eder).
"""

from __future__ import annotations

import numpy as np

from core.knapsack import FEASIBILITY_TOL, KnapsackProblem


class InfeasibleSolutionError(ValueError):
    """Kapasiteyi aşan bir çözüm değerlendirilmek istendiğinde fırlatılır."""


class FitnessEvaluator:
    """Sayaçlı ve opsiyonel bütçeli nesne fonksiyonu değerlendiricisi.

    Args:
        problem: Problem örneği.
        max_evaluations: İsteğe bağlı sert değerlendirme üst sınırı.
        strict: ``True`` ise uygun olmayan çözümde hata fırlatılır.
    """

    def __init__(
        self,
        problem: KnapsackProblem,
        max_evaluations: int | None = None,
        strict: bool = True,
    ) -> None:
        if max_evaluations is not None and max_evaluations <= 0:
            raise ValueError("max_evaluations pozitif olmalıdır.")
        self.problem = problem
        self.max_evaluations = max_evaluations
        self.strict = strict
        self.n_evaluations = 0

    @property
    def remaining(self) -> int:
        """Kalan değerlendirme hakkı (sınırsızsa çok büyük bir tamsayı)."""
        if self.max_evaluations is None:
            return 2**62
        return max(self.max_evaluations - self.n_evaluations, 0)

    def evaluate(self, solution: np.ndarray) -> float:
        """Tek bir çözümün değerini döndürür ve sayacı 1 artırır."""
        if self.remaining <= 0:
            raise RuntimeError("Değerlendirme bütçesi aşıldı.")
        x = np.asarray(solution, dtype=np.float64)
        if self.strict and float(self.problem.weights @ x) > self.problem.capacity + FEASIBILITY_TOL:
            raise InfeasibleSolutionError("Çözüm kapasiteyi aşıyor.")
        self.n_evaluations += 1
        return float(self.problem.values @ x)

    def evaluate_batch(self, population: np.ndarray) -> np.ndarray:
        """``N x n`` matrisin her satırını değerlendirir; sayaç ``N`` artar."""
        matrix = np.asarray(population, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("population iki boyutlu olmalıdır.")
        if matrix.shape[0] > self.remaining:
            raise RuntimeError("Değerlendirme bütçesi aşılacak.")
        if self.strict:
            weights = matrix @ self.problem.weights
            if (weights > self.problem.capacity + FEASIBILITY_TOL).any():
                raise InfeasibleSolutionError("Popülasyonda kapasiteyi aşan çözüm var.")
        self.n_evaluations += matrix.shape[0]
        return matrix @ self.problem.values
