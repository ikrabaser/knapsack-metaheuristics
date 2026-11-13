"""Veri seti analizi: temizleme günlüğü, betimsel istatistikler, referans optimum.

Çıktılar:

* ``results/tables/dataset_summary.csv`` — ağırlık/değer/oran istatistikleri
* ``results/tables/dataset_overview.csv`` — ürün sayısı, toplamlar, kapasite, korelasyon, açgözlü kesim oranı
* ``results/raw/data_cleaning_log.json`` — yapılan/denetlenen temizleme adımları
* ``results/raw/reference_solution.json`` — DP optimumu (MILP ile doğrulanmış), açgözlü ve LP üst sınırı
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config
from core.data_loader import build_problem, dataset_overview, dataset_summary
from core.exact import greedy_solution, solve_dp, solve_milp
from experiments.runner import prepare_problem


def _median_time_ms(function, repeats: int) -> float:
    """``function()``'ın medyan çalışma süresi (milisaniye)."""
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(samples))


def run_dataset_analysis(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Veri setini inceler ve tabloları/günlükleri yazar."""
    problem, _, log = build_problem(
        config.DATA_PATH, config.CAPACITY_RATIO, config.RAW_DIR / "data_cleaning_log.json"
    )
    _, optimum = prepare_problem(write_logs=True)
    greedy = greedy_solution(problem)
    dp_time_ms = _median_time_ms(lambda: solve_dp(problem), repeats=20)
    milp_time_ms = _median_time_ms(lambda: solve_milp(problem), repeats=5)
    greedy_time_ms = _median_time_ms(lambda: greedy_solution(problem), repeats=50)

    summary = dataset_summary(problem)
    overview = dataset_overview(problem)
    selected_ratios = problem.ratios[greedy.solution == 1]
    overview["greedy_critical_ratio"] = float(selected_ratios.min())
    overview["optimum_value_dp"] = optimum.value
    overview["optimum_weight"] = optimum.weight
    overview["optimum_n_selected"] = optimum.n_selected
    overview["dp_median_time_ms"] = dp_time_ms
    overview["milp_median_time_ms"] = milp_time_ms
    overview["greedy_median_time_ms"] = greedy_time_ms
    overview["n_duplicate_weight_value_pairs"] = log["info_identical_weight_value_pairs_kept"]
    overview["rows_removed_by_cleaning"] = log["rows_removed_total"]
    overview_frame = pd.DataFrame([overview])

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.TABLES_DIR / "dataset_summary.csv", index=False, encoding="utf-8")
    overview_frame.to_csv(config.TABLES_DIR / "dataset_overview.csv", index=False, encoding="utf-8")
    if verbose:
        print(summary.to_string(index=False))
        print(overview_frame.T.to_string(header=False))
    return {"summary": summary, "overview": overview_frame}


if __name__ == "__main__":
    run_dataset_analysis()
