"""0/1 Knapsack problem tanımı.

Matematiksel model::

    maximize   f(x) = sum_i v_i * x_i
    subject to sum_i w_i * x_i <= C,   x_i in {0, 1}

Kapasite ``C = oran * sum(w)`` olarak hesaplanır ve **yuvarlanmaz**.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Kapasite karşılaştırmalarında kayan noktalı sayı hatalarını yutmak için
#: kullanılan mutlak tolerans. Kapasiteyi yuvarlamak anlamına gelmez.
FEASIBILITY_TOL: float = 1e-9


@dataclass(frozen=True, eq=False)
class KnapsackProblem:
    """Değişmez (immutable) 0/1 Knapsack problem örneği.

    Attributes:
        weights: Ürün ağırlıkları, şekil ``(n,)``, hepsi pozitif.
        values: Ürün değerleri, şekil ``(n,)``, hepsi pozitif.
        capacity: Çanta kapasitesi ``C`` (yuvarlanmamış).
        capacity_ratio: ``C / sum(w)`` oranı (bilgi amaçlı).
        name: Problem etiketi (raporlama için).
    """

    weights: np.ndarray
    values: np.ndarray
    capacity: float
    capacity_ratio: float = float("nan")
    name: str = "knapsack"
    ratios: np.ndarray = field(init=False, repr=False)
    ratio_order_ascending: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64).copy()
        values = np.asarray(self.values, dtype=np.float64).copy()
        if weights.ndim != 1 or values.ndim != 1:
            raise ValueError("weights ve values tek boyutlu olmalıdır.")
        if weights.shape != values.shape:
            raise ValueError("weights ve values aynı uzunlukta olmalıdır.")
        if weights.size == 0:
            raise ValueError("Problem en az bir ürün içermelidir.")
        if not (np.isfinite(weights).all() and np.isfinite(values).all()):
            raise ValueError("weights/values sonlu sayılar olmalıdır.")
        if (weights <= 0).any() or (values <= 0).any():
            raise ValueError("weights ve values kesinlikle pozitif olmalıdır.")
        if not np.isfinite(self.capacity) or self.capacity <= 0:
            raise ValueError("capacity pozitif ve sonlu olmalıdır.")
        ratios = values / weights
        weights.setflags(write=False)
        values.setflags(write=False)
        ratios.setflags(write=False)
        # kararlı sıralama: eşit oranlarda ürün sırası korunur (tekrarlanabilirlik)
        order = np.argsort(ratios, kind="stable")
        order.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "ratios", ratios)
        object.__setattr__(self, "ratio_order_ascending", order)

    # ------------------------------------------------------------------ #
    # Yapıcılar
    # ------------------------------------------------------------------ #
    @classmethod
    def from_arrays(
        cls,
        weights: np.ndarray,
        values: np.ndarray,
        capacity_ratio: float = 0.70,
        name: str = "knapsack",
    ) -> "KnapsackProblem":
        """Kapasiteyi ``capacity_ratio * sum(weights)`` olarak kurar (yuvarlamadan)."""
        if not 0.0 < capacity_ratio <= 1.0:
            raise ValueError("capacity_ratio (0, 1] aralığında olmalıdır.")
        weights = np.asarray(weights, dtype=np.float64)
        capacity = float(capacity_ratio * weights.sum())
        return cls(weights, values, capacity, capacity_ratio=capacity_ratio, name=name)

    def with_capacity_ratio(self, capacity_ratio: float) -> "KnapsackProblem":
        """Aynı ürünlerle farklı kapasite oranına sahip yeni problem döndürür."""
        return KnapsackProblem.from_arrays(
            self.weights, self.values, capacity_ratio, name=self.name
        )

    # ------------------------------------------------------------------ #
    # Özellikler ve yardımcılar
    # ------------------------------------------------------------------ #
    @property
    def n_items(self) -> int:
        """Ürün sayısı ``n``."""
        return int(self.weights.size)

    @property
    def total_weight(self) -> float:
        """Tüm ürünlerin toplam ağırlığı."""
        return float(self.weights.sum())

    @property
    def total_value(self) -> float:
        """Tüm ürünlerin toplam değeri."""
        return float(self.values.sum())

    def solution_weight(self, solution: np.ndarray) -> float:
        """Seçilen ürünlerin toplam ağırlığı."""
        return float(self.weights @ np.asarray(solution, dtype=np.float64))

    def solution_value(self, solution: np.ndarray) -> float:
        """Seçilen ürünlerin toplam değeri (kısıt kontrolü yapmaz)."""
        return float(self.values @ np.asarray(solution, dtype=np.float64))

    def is_feasible(self, solution: np.ndarray) -> bool:
        """Çözüm ikili vektör mü ve kapasiteyi aşmıyor mu?"""
        x = np.asarray(solution)
        if x.shape != (self.n_items,):
            return False
        if not np.isin(x, (0, 1)).all():
            return False
        return self.solution_weight(x) <= self.capacity + FEASIBILITY_TOL
