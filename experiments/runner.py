"""Deney çalıştırma altyapısı (ana deney, duyarlılık ve ek deneyler ortak kullanır).

Adil karşılaştırma ilkeleri:

* tüm konfigürasyonlar **aynı problem örneği** ve **aynı tohum listesiyle** çalışır,
* çalıştırmalar **tohumlar arasında serpiştirilir** (her tohum için tüm
  konfigürasyonlar art arda): zamana bağlı sistem yükü (arka plan işlemleri,
  CPU saat değişimi) bir algoritmayı sistematik olarak kayırmaz,
* süre ölçümünden önce her konfigürasyon için kayıt dışı bir **ısınma** turu yapılır,
* her çalıştırmanın nihai çözümü bağımsız olarak doğrulanır
  (:meth:`algorithms.base.BaseOptimizer.run`).
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

import config
from algorithms import ALGORITHM_REGISTRY
from core.exact import ExactSolution, solve_dp, solve_milp
from core.knapsack import KnapsackProblem


@dataclass(frozen=True)
class RunSpec:
    """Bir deney konfigürasyonu: algoritma + parametreler + ek sütun bilgileri."""

    label: str
    algorithm: str
    params: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def build(self, problem: KnapsackProblem) -> Any:
        """Algoritma örneğini üretir."""
        return ALGORITHM_REGISTRY[self.algorithm](problem, **self.params)


@dataclass
class ExperimentData:
    """Bir deney grubunun tüm ham çıktıları."""

    name: str
    runs: pd.DataFrame
    curves: dict[str, np.ndarray] = field(default_factory=dict)
    evaluation_curves: dict[str, np.ndarray] = field(default_factory=dict)
    diversity: dict[str, np.ndarray] = field(default_factory=dict)
    extras: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    def save(self, directory: Path = config.RAW_DIR) -> None:
        """``<ad>_runs.csv`` ve ``<ad>_curves.npz`` dosyalarını yazar."""
        directory.mkdir(parents=True, exist_ok=True)
        self.runs.to_csv(directory / f"{self.name}_runs.csv", index=False, encoding="utf-8")
        arrays: dict[str, np.ndarray] = {}
        for label, array in self.curves.items():
            arrays[f"{slugify(label)}__history"] = array
        for label, array in self.evaluation_curves.items():
            arrays[f"{slugify(label)}__evals"] = array
        for label, array in self.diversity.items():
            arrays[f"{slugify(label)}__diversity"] = array
        for label, mapping in self.extras.items():
            for key, array in mapping.items():
                arrays[f"{slugify(label)}__x_{key}"] = array
        if arrays:
            np.savez_compressed(directory / f"{self.name}_curves.npz", **arrays)


def slugify(label: str) -> str:
    """Etiketi dosya/anahtar adı olarak güvenli hâle getirir."""
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", label).strip("_")


def load_experiment(name: str, directory: Path = config.RAW_DIR) -> ExperimentData:
    """Kaydedilmiş bir deney grubunu geri yükler."""
    runs = pd.read_csv(directory / f"{name}_runs.csv", encoding="utf-8")
    data = ExperimentData(name=name, runs=runs)
    npz_path = directory / f"{name}_curves.npz"
    if not npz_path.is_file():
        return data
    store = np.load(npz_path)
    for label in runs["label"].unique():
        key = slugify(label)
        if f"{key}__history" in store:
            data.curves[label] = store[f"{key}__history"]
        if f"{key}__evals" in store:
            data.evaluation_curves[label] = store[f"{key}__evals"]
        if f"{key}__diversity" in store:
            data.diversity[label] = store[f"{key}__diversity"]
        prefix = f"{key}__x_"
        extra = {k[len(prefix):]: store[k] for k in store.files if k.startswith(prefix)}
        if extra:
            data.extras[label] = extra
    return data


# --------------------------------------------------------------------------- #
# Referans optimum ve metrikler
# --------------------------------------------------------------------------- #
def reference_optimum(problem: KnapsackProblem, verify_with_milp: bool = True) -> ExactSolution:
    """Gerçek optimumu hesaplar (DP; ağırlıklar tamsayı değilse MILP).

    ``verify_with_milp`` açıksa DP sonucu bağımsız bir MILP çözümüyle doğrulanır.
    """
    try:
        exact = solve_dp(problem)
    except ValueError:
        return solve_milp(problem)
    if verify_with_milp:
        check = solve_milp(problem)
        if abs(check.value - exact.value) > 1e-6:
            raise AssertionError("DP ve MILP optimum değerleri tutarsız!")
    return exact


def gap_percent(optimum: float, value: float) -> float:
    """Göreli optimalite farkı: ``(Optimum − Değer) / Optimum × 100``."""
    return (optimum - value) / optimum * 100.0


def iterations_to_threshold(history: np.ndarray, optimum: float, fraction: float) -> float:
    """En iyi değerin ``fraction * optimum`` düzeyine ilk ulaştığı (1 tabanlı) iterasyon.

    Hiç ulaşılamazsa ``NaN`` (sağdan sansürlü) döndürür.
    """
    reached = np.flatnonzero(history >= fraction * optimum - 1e-9)
    return float(reached[0] + 1) if reached.size else float("nan")


def _pad(arrays: list[np.ndarray]) -> np.ndarray:
    """Farklı uzunluktaki izleri son değerle doldurup matris yapar."""
    width = max(len(a) for a in arrays)
    out = np.empty((len(arrays), width), dtype=np.float64)
    for i, array in enumerate(arrays):
        out[i, : len(array)] = array
        out[i, len(array):] = array[-1]
    return out


# --------------------------------------------------------------------------- #
# Çalıştırma
# --------------------------------------------------------------------------- #
def run_specs(
    problem: KnapsackProblem,
    specs: list[RunSpec],
    seeds: tuple[int, ...] | list[int],
    optimum: float,
    name: str,
    keep_curves: bool = True,
    warmup: bool = True,
    verbose: bool = True,
    timing_repeats: int = 1,
) -> ExperimentData:
    """Tüm konfigürasyonları aynı tohumlarla, tohumlar arası serpiştirerek çalıştırır.

    Args:
        problem: Ortak problem örneği.
        specs: Çalıştırılacak konfigürasyonlar (etiketler benzersiz olmalı).
        seeds: Bağımsız çalıştırma tohumları.
        optimum: Referans optimum değeri (gap ve yakınsama metrikleri için).
        name: Deney grubu adı (dosya adı öneki).
        keep_curves: Yakınsama eğrilerini sakla.
        warmup: Süre ölçümünden önce kayıt dışı ısınma turu yap.
        verbose: İlerleme çıktısı yaz.
        timing_repeats: Her (konfigürasyon, tohum) çiftinin art arda koşturulma sayısı; süre olarak
            **medyan** alınır ve tekrarların aynı çözümü verdiği doğrulanır (sistem yüküne karşı sağlamlık).

    Returns:
        Çalıştırma kayıtları ve eğriler.
    """
    labels = [spec.label for spec in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("Konfigürasyon etiketleri benzersiz olmalıdır.")
    built = {spec.label: spec.build(problem) for spec in specs}

    if warmup:
        for optimizer in built.values():
            optimizer.run(config.PILOT_SEEDS[0])

    records: list[dict[str, Any]] = []
    histories: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    eval_hist: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    diversity: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    extras: dict[str, dict[str, list[np.ndarray]]] = {label: {} for label in labels}

    started = time.perf_counter()
    for run_index, seed in enumerate(seeds):
        for spec in specs:
            repeats = [built[spec.label].run(seed) for _ in range(max(1, timing_repeats))]
            result = repeats[0]
            if any(r.solution_string() != result.solution_string() or r.n_evaluations != result.n_evaluations
                   for r in repeats[1:]):
                raise AssertionError(f"{spec.label} (tohum {seed}): tekrarlar aynı sonucu vermedi (determinizm ihlali).")
            times = np.array([r.elapsed_time for r in repeats])
            result.elapsed_time = float(np.median(times))
            record: dict[str, Any] = {
                "experiment": name,
                "label": spec.label,
                "algorithm": spec.algorithm,
                **spec.meta,
                "seed": seed,
                "run_index": run_index,
                "best_value": result.best_value,
                "total_weight": result.total_weight,
                "n_selected": result.n_selected,
                "elapsed_s": result.elapsed_time,
                "elapsed_min_s": float(times.min()),
                "elapsed_max_s": float(times.max()),
                "timing_repeats": len(repeats),
                "n_evaluations": result.n_evaluations,
                "optimum": optimum,
                "gap_pct": gap_percent(optimum, result.best_value),
                "hit_optimum": bool(result.best_value >= optimum - 1e-9),
                "feasible": bool(problem.is_feasible(result.best_solution)),
                "capacity": problem.capacity,
                "params_json": json.dumps(result.params, sort_keys=True, default=float),
                "solution": result.solution_string(),
            }
            for fraction in config.CONVERGENCE_THRESHOLDS:
                reached = iterations_to_threshold(result.history, optimum, fraction)
                record[f"iters_to_{int(round(fraction * 1000))}"] = reached
                if np.isnan(reached):
                    record[f"evals_to_{int(round(fraction * 1000))}"] = float("nan")
                else:
                    record[f"evals_to_{int(round(fraction * 1000))}"] = float(
                        result.evaluations_history[int(reached) - 1]
                    )
            records.append(record)
            if keep_curves:
                histories[spec.label].append(result.history)
                eval_hist[spec.label].append(result.evaluations_history)
                if result.diversity_history is not None:
                    diversity[spec.label].append(result.diversity_history)
                for key, series in result.extra_history.items():
                    extras[spec.label].setdefault(key, []).append(series)
        if verbose:
            elapsed = time.perf_counter() - started
            print(f"  [{name}] tohum {run_index + 1}/{len(seeds)} tamam ({elapsed:6.1f} sn)", flush=True)

    data = ExperimentData(name=name, runs=pd.DataFrame.from_records(records))
    if not data.runs["feasible"].all():
        raise AssertionError("Kapasiteyi aşan nihai çözüm tespit edildi!")
    if keep_curves:
        for label in labels:
            data.curves[label] = _pad(histories[label])
            data.evaluation_curves[label] = _pad(eval_hist[label]).astype(np.int64)
            if diversity[label]:
                data.diversity[label] = _pad(diversity[label])
            if extras[label]:
                data.extras[label] = {k: _pad(v) for k, v in extras[label].items()}
    return data


def prepare_problem(
    capacity_ratio: float = config.CAPACITY_RATIO, write_logs: bool = True
) -> tuple[KnapsackProblem, ExactSolution]:
    """Veri setini yükler/temizler, gerçek optimumu (DP + MILP doğrulaması) hesaplar.

    ``write_logs`` açıksa temizleme günlüğü ve referans çözüm JSON olarak yazılır.
    """
    from core.data_loader import build_problem
    from core.exact import fractional_upper_bound, greedy_solution

    log_path = config.RAW_DIR / "data_cleaning_log.json" if write_logs else None
    problem, _, _ = build_problem(config.DATA_PATH, capacity_ratio, log_path)
    optimum = reference_optimum(problem)
    if write_logs:
        greedy = greedy_solution(problem)
        reference = {
            "capacity_ratio": capacity_ratio,
            "capacity": problem.capacity,
            "optimum_value_dp": optimum.value,
            "optimum_weight": optimum.weight,
            "optimum_n_selected": optimum.n_selected,
            "optimum_solution": "".join(str(int(b)) for b in optimum.solution),
            "milp_cross_check": "DP ve scipy.optimize.milp (HiGHS) aynı optimum değeri verdi",
            "greedy_ratio_value": greedy.value,
            "greedy_gap_pct": gap_percent(optimum.value, greedy.value),
            "lp_relaxation_upper_bound": fractional_upper_bound(problem),
        }
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        (config.RAW_DIR / "reference_solution.json").write_text(
            json.dumps(reference, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return problem, optimum


def _windows_query(command: str) -> str | None:
    """Windows'ta tek satırlık PowerShell sorgusu çalıştırır (başarısızsa ``None``)."""
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _os_label() -> str:
    """Okunur işletim sistemi adı (Python 3.11 Windows 11'i 'Windows-10' olarak bildirir; derleme >= 22000 ise düzeltilir)."""
    text = platform.platform()
    match = re.search(r"10\.0\.(\d+)", text)
    if sys.platform == "win32" and match and int(match.group(1)) >= 22000:
        return f"Windows 11 (derleme {match.group(1)})"
    return text


def environment_info() -> dict[str, Any]:
    """Tekrarlanabilirlik için donanım/yazılım ortamı bilgisi."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "os": _os_label(),
        "machine": platform.machine(),
        "cpu_count_logical": os.cpu_count(),
    }
    cpu_name = _windows_query("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
    ram_bytes = _windows_query("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
    info["cpu"] = cpu_name or platform.processor()
    if ram_bytes and ram_bytes.isdigit():
        info["ram_gb"] = round(int(ram_bytes) / 1024**3, 1)
    try:
        import matplotlib

        info["matplotlib"] = matplotlib.__version__
    except ImportError:  # pragma: no cover
        pass
    return info
