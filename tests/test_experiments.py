"""Deney altyapısı ve istatistik fonksiyonları testleri."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.exact import brute_force
from core.knapsack import KnapsackProblem
from experiments.pilot_selection import _select_best
from experiments.runner import (
    RunSpec,
    gap_percent,
    iterations_to_threshold,
    load_experiment,
    run_specs,
    slugify,
)
from experiments.statistical_analysis import (
    a12_magnitude,
    bootstrap_confidence_interval,
    cohens_d,
    holm_adjust,
    pairwise_table,
    summary_table,
    t_confidence_interval,
    vargha_delaney_a12,
)


def small_problem() -> KnapsackProblem:
    rng = np.random.default_rng(5)
    return KnapsackProblem.from_arrays(
        rng.integers(1, 30, 16).astype(float), rng.integers(5, 100, 16).astype(float), 0.5
    )


def synthetic_runs(values: dict[str, list[float]], optimum: float = 100.0) -> pd.DataFrame:
    rows = []
    for label, series in values.items():
        for seed, value in enumerate(series):
            rows.append(
                {
                    "label": label, "seed": seed, "best_value": value, "optimum": optimum,
                    "elapsed_s": 0.1 + 0.01 * seed, "n_evaluations": 8000, "n_selected": 10,
                    "total_weight": 50.0, "hit_optimum": value >= optimum,
                    "iters_to_990": 5.0 if value >= 0.99 * optimum else np.nan,
                    "iters_to_995": 8.0 if value >= 0.995 * optimum else np.nan,
                    "evals_to_990": 200.0 if value >= 0.99 * optimum else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- #
# Çalıştırıcı
# ----------------------------------------------------------------------------- #
def test_gap_and_threshold_helpers() -> None:
    assert gap_percent(4560.0, 4560.0) == 0.0
    assert gap_percent(4560.0, 4104.0) == pytest.approx(10.0)
    history = np.array([50.0, 90.0, 99.0, 100.0])
    assert iterations_to_threshold(history, 100.0, 0.99) == 3.0
    assert iterations_to_threshold(history, 100.0, 1.0) == 4.0
    assert np.isnan(iterations_to_threshold(np.array([10.0, 20.0]), 100.0, 0.99))


def test_run_specs_end_to_end_and_roundtrip(tmp_path) -> None:
    problem = small_problem()
    optimum = brute_force(problem).value
    specs = [
        RunSpec("GA", "GA", {"population_size": 20, "max_iterations": 15}, {"group": "x"}),
        RunSpec("SA", "SA", {"neighbors_per_iteration": 10, "max_iterations": 15}),
    ]
    data = run_specs(problem, specs, [1, 2, 3], optimum, "unit", warmup=False, verbose=False)
    assert len(data.runs) == 6 and data.runs["feasible"].all()
    assert (data.runs["gap_pct"] >= -1e-9).all() and (data.runs["best_value"] <= optimum + 1e-9).all()
    assert set(data.runs["seed"]) == {1, 2, 3} and data.curves["GA"].shape == (3, 15)
    assert (np.diff(data.curves["GA"], axis=1) >= 0).all()
    assert all(len(s) == 16 and set(s) <= {"0", "1"} for s in data.runs["solution"])
    data.save(tmp_path)
    loaded = load_experiment("unit", tmp_path)
    assert loaded.runs["best_value"].tolist() == data.runs["best_value"].tolist()
    assert np.array_equal(loaded.curves["SA"], data.curves["SA"])
    assert "temperature" in loaded.extras["SA"]


def test_run_specs_is_deterministic_for_same_seeds() -> None:
    problem = small_problem()
    optimum = brute_force(problem).value
    specs = [RunSpec("GA", "GA", {"population_size": 20, "max_iterations": 10})]
    first = run_specs(problem, specs, [7, 8], optimum, "a", warmup=False, verbose=False)
    second = run_specs(problem, specs, [7, 8], optimum, "b", warmup=False, verbose=False)
    assert first.runs["solution"].tolist() == second.runs["solution"].tolist()


def test_duplicate_labels_rejected() -> None:
    problem = small_problem()
    spec = RunSpec("GA", "GA", {"max_iterations": 5})
    with pytest.raises(ValueError):
        run_specs(problem, [spec, spec], [1], 1.0, "dup", warmup=False, verbose=False)


def test_slugify_is_filesystem_safe() -> None:
    assert slugify("GA|1/n") == "GA_1_n"
    assert "/" not in slugify("GA|2/n") and "|" not in slugify("GA|2/n")


def test_pilot_tie_break_prefers_lower_std_then_middle_level() -> None:
    group = pd.DataFrame({"mean": [10.0, 10.0, 10.0], "std": [0.0, 0.0, 0.0]}, index=[4, 5, 6])
    assert _select_best(group) == 5  # eşitlikte ortadaki düzey
    group = pd.DataFrame({"mean": [10.0, 10.0, 9.0], "std": [2.0, 1.0, 0.0]}, index=[0, 1, 2])
    assert _select_best(group) == 1  # aynı ortalamada düşük std.
    group = pd.DataFrame({"mean": [9.0, 11.0, 10.0], "std": [0.0, 5.0, 0.0]}, index=[0, 1, 2])
    assert _select_best(group) == 1  # önce ortalama


# ----------------------------------------------------------------------------- #
# İstatistik
# ----------------------------------------------------------------------------- #
def test_holm_adjustment() -> None:
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert (holm_adjust(np.array([0.5, 0.9])) <= 1.0).all()
    assert (holm_adjust(np.array([0.2, 0.01, 0.5])) >= np.array([0.2, 0.01, 0.5])).all()


def test_vargha_delaney_a12() -> None:
    a, b = np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0])
    assert vargha_delaney_a12(a, b) == pytest.approx(2.0 / 9.0)
    assert vargha_delaney_a12(b, a) == pytest.approx(7.0 / 9.0)
    assert vargha_delaney_a12(np.array([5.0, 6.0]), np.array([1.0, 2.0])) == 1.0
    assert a12_magnitude(0.5) == "ihmal edilebilir" and a12_magnitude(0.9) == "büyük"
    assert a12_magnitude(0.1) == "büyük"  # simetrik


def test_cohens_d() -> None:
    a, b = np.array([1.0, 2.0, 3.0, 4.0]), np.array([3.0, 4.0, 5.0, 6.0])
    assert cohens_d(a, b) == pytest.approx(-2.0 / a.std(ddof=1))
    assert np.isnan(cohens_d(np.ones(5), np.ones(5)))


def test_confidence_intervals() -> None:
    values = np.random.default_rng(0).normal(100, 5, 30)
    low, high = t_confidence_interval(values)
    reference = stats.t.interval(0.95, 29, loc=values.mean(), scale=stats.sem(values))
    assert (low, high) == pytest.approx(reference)
    assert low < values.mean() < high
    assert bootstrap_confidence_interval(values) == bootstrap_confidence_interval(values)  # sabit tohum
    assert t_confidence_interval(np.full(10, 3.0)) == (3.0, 3.0)  # sabit veri: dejenere aralık


def test_pairwise_table_separates_real_difference_and_survives_ties() -> None:
    rng = np.random.default_rng(1)
    runs = synthetic_runs(
        {
            "A": [100.0] * 30,
            "B": [100.0] * 30,  # A ile tamamen özdeş
            "C": list(95.0 + rng.normal(0, 1.0, 30)),
        }
    )
    table = pairwise_table(runs, ["A", "B", "C"]).set_index(["a", "b"])
    assert table.loc[("A", "B"), "wilcoxon_p"] == 1.0 and table.loc[("A", "B"), "mannwhitney_p"] == 1.0
    assert table.loc[("A", "B"), "a12"] == 0.5
    assert table.loc[("A", "C"), "wilcoxon_p_holm"] < 1e-4 and bool(table.loc[("A", "C"), "significant_holm_wilcoxon"])
    assert table.loc[("A", "C"), "a12"] == 1.0 and table.loc[("A", "C"), "mean_diff"] > 4
    assert (table["wilcoxon_p_holm"] >= table["wilcoxon_p"] - 1e-12).all()


def test_summary_table_metrics() -> None:
    runs = synthetic_runs({"A": [100.0, 99.0, 98.0, 100.0], "B": [90.0, 90.0, 90.0, 90.0]})
    summary = summary_table(runs, ["A", "B"]).set_index("label")
    assert summary.loc["A", "best"] == 100.0 and summary.loc["A", "worst"] == 98.0
    assert summary.loc["A", "mean"] == pytest.approx(99.25)
    assert summary.loc["A", "std"] == pytest.approx(np.std([100, 99, 98, 100], ddof=1))
    assert summary.loc["A", "mean_gap_pct"] == pytest.approx(0.75)
    assert summary.loc["A", "hit_optimum_pct"] == 50.0 and summary.loc["B", "hit_optimum_pct"] == 0.0
    assert summary.loc["B", "std"] == 0.0 and summary.loc["B", "cv_pct"] == 0.0
    assert summary.loc["A", "reached_990_pct"] == 75.0 and summary.loc["B", "reached_990_pct"] == 0.0
    assert summary.loc["A", "ci95_t_low"] < summary.loc["A", "mean"] < summary.loc["A", "ci95_t_high"]
