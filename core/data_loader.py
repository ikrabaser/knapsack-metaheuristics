"""Veri seti yükleme, temizleme ve betimsel istatistikler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.knapsack import KnapsackProblem

REQUIRED_COLUMNS: tuple[str, ...] = ("item", "weight", "value")


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """CSV dosyasını olduğu gibi (temizlemeden) okur.

    Raises:
        FileNotFoundError: Dosya yoksa.
        ValueError: Gerekli sütunlar (item, weight, value) eksikse.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Veri seti bulunamadı: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Veri setinde eksik sütun(lar): {missing}")
    return frame[list(REQUIRED_COLUMNS)]


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Eksik, tekrarlı ve geçersiz kayıtları denetler; yapılan işlemleri kaydeder.

    Uygulanan adımlar (sırasıyla):

    1. sayısal olmayan değerler ``NaN`` yapılır,
    2. eksik (``NaN``) değer içeren satırlar silinir,
    3. aynı ``item`` numarasını taşıyan tekrar kayıtlar silinir (ilki korunur),
    4. ağırlığı ya da değeri ``<= 0`` olan satırlar silinir.

    Aynı (ağırlık, değer) çiftine sahip *farklı* ürünler meşrudur; silinmez,
    yalnızca bilgi olarak raporlanır.

    Returns:
        ``(temiz_veri, log)`` — ``log`` JSON'a yazılabilir bir sözlüktür.
    """
    log: dict[str, Any] = {"rows_initial": int(len(frame)), "steps": []}
    work = frame.copy()
    for column in REQUIRED_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    n_missing = int(work.isna().any(axis=1).sum())
    work = work.dropna()
    log["steps"].append({"step": "eksik/sayısal olmayan değer içeren satırların silinmesi", "removed": n_missing})

    n_duplicate_ids = int(work.duplicated(subset="item").sum())
    work = work.drop_duplicates(subset="item", keep="first")
    log["steps"].append({"step": "tekrarlayan item numarasının silinmesi", "removed": n_duplicate_ids})

    invalid = (work["weight"] <= 0) | (work["value"] <= 0)
    n_invalid = int(invalid.sum())
    work = work.loc[~invalid]
    log["steps"].append({"step": "ağırlığı veya değeri <= 0 olan satırların silinmesi", "removed": n_invalid})

    n_same_pair = int(work.duplicated(subset=["weight", "value"], keep=False).sum())
    log["info_identical_weight_value_pairs_kept"] = n_same_pair

    work = work.reset_index(drop=True)
    work["item"] = work["item"].astype(int)
    log["rows_final"] = int(len(work))
    log["rows_removed_total"] = log["rows_initial"] - log["rows_final"]
    log["cleaning_needed"] = log["rows_removed_total"] > 0
    return work, log


def build_problem(
    path: str | Path, capacity_ratio: float = 0.70, log_path: str | Path | None = None
) -> tuple[KnapsackProblem, pd.DataFrame, dict[str, Any]]:
    """Veri setini yükler, temizler ve :class:`KnapsackProblem` üretir.

    Args:
        path: CSV dosya yolu.
        capacity_ratio: ``C = oran * toplam ağırlık`` (varsayılan 0.70).
        log_path: Verilirse temizleme günlüğü JSON olarak yazılır.

    Returns:
        ``(problem, temiz_veri, temizleme_günlüğü)``
    """
    clean, log = clean_dataset(load_raw_dataset(path))
    problem = KnapsackProblem.from_arrays(
        clean["weight"].to_numpy(), clean["value"].to_numpy(), capacity_ratio, name=Path(path).stem
    )
    log["capacity_ratio"] = capacity_ratio
    log["capacity"] = problem.capacity
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return problem, clean, log


def dataset_summary(problem: KnapsackProblem) -> pd.DataFrame:
    """Veri seti için betimsel istatistik tablosu (ağırlık, değer, oran)."""
    rows = []
    for label, array in (
        ("Ağırlık (w)", problem.weights),
        ("Değer (v)", problem.values),
        ("Değer/ağırlık (v/w)", problem.ratios),
    ):
        rows.append(
            {
                "Değişken": label,
                "Min": float(array.min()),
                "Maks": float(array.max()),
                "Ortalama": float(array.mean()),
                "Medyan": float(np.median(array)),
                "Std. sapma": float(array.std(ddof=1)),
                "Toplam": float(array.sum()) if "v/w" not in label else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def dataset_overview(problem: KnapsackProblem) -> dict[str, float]:
    """Tek satırlık genel bilgiler (ürün sayısı, toplamlar, kapasite, korelasyon)."""
    return {
        "n_items": problem.n_items,
        "total_weight": problem.total_weight,
        "total_value": problem.total_value,
        "capacity_ratio": problem.capacity_ratio,
        "capacity": problem.capacity,
        "corr_weight_value_pearson": float(np.corrcoef(problem.weights, problem.values)[0, 1]),
    }
