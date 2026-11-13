"""İstatistiksel analiz: özet tablolar, güven aralıkları, hipotez testleri, etki büyüklükleri.

Test seçimi ve varsayımlar
--------------------------
* Nihai uygunluk değerleri **ayrık**, **tavanlı** (optimum ile sınırlı) ve çoğu zaman
  **çok eşitlikli**dir; normal dağılım varsayımı (Shapiro–Wilk ile denetlenir) tutmaz.
  Bu nedenle **parametrik olmayan** testler kullanılır.
* **Friedman** (omnibus) ve **Wilcoxon işaretli sıra** (ikili) testleri, deneyin
  *tohum eşleştirmeli* tasarımına göre uygulanır (aynı tohum = aynı blok).
  Uyarı: algoritmalar farklı rastgele sayı akışları tükettiği için eşleştirme
  *nominaldir*; eşleşmenin gücü sınırlıdır. Bu yüzden eşleştirilmemiş
  **Kruskal–Wallis** / **Mann–Whitney U** sonuçları da yanında verilir ve ikisinin
  uyumu bir sağlamlık denetimi olarak değerlendirilir.
* ikili karşılaştırmalarda **Holm** düzeltmesi (aile-bazlı hata kontrolü).
* Etki büyüklüğü: **Vargha–Delaney A12** (``P(a>b) + 0.5 P(a=b)``), **Cliff delta**
  (``2·A12 − 1``) ve Cohen d (ortak std. ile; std. = 0 ise tanımsız).
* ``%95`` güven aralıkları: t-dağılımlı ve yüzdelik bootstrap (10 000 yeniden örnekleme).
* n = 30: büyük örneklem yaklaşımı için yeterli, ancak çok eşitlikli veride
  asimptotik p-değerleri yaklaşıktır; küçük n'de (ör. n = 10) güç düşer.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import config
from experiments.runner import ExperimentData, load_experiment

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20240101


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm basamaklı düzeltme (aile-bazlı hata oranını kontrol eder)."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.empty_like(p)
    running_max = 0.0
    for rank, index in enumerate(np.argsort(p)):
        running_max = max(running_max, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def vargha_delaney_a12(a: np.ndarray, b: np.ndarray) -> float:
    """``P(a > b) + 0.5 · P(a = b)``; 0.5 etkisizdir, >0.5 ``a`` lehinedir."""
    u_statistic = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic").statistic
    return float(u_statistic / (len(a) * len(b)))


def a12_magnitude(a12: float) -> str:
    """Vargha–Delaney eşikleriyle etki büyüklüğü etiketi (0.56 / 0.64 / 0.71)."""
    distance = max(a12, 1.0 - a12)
    if distance < 0.56:
        return "ihmal edilebilir"
    if distance < 0.64:
        return "küçük"
    if distance < 0.71:
        return "orta"
    return "büyük"


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Ortak standart sapmayla Cohen d (her iki grup sabitse tanımsız -> NaN)."""
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled == 0.0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def t_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Ortalama için t-dağılımlı güven aralığı."""
    n = len(values)
    mean, sem = float(values.mean()), float(stats.sem(values))
    if sem == 0.0 or n < 2:
        return mean, mean
    low, high = stats.t.interval(confidence, df=n - 1, loc=mean, scale=sem)
    return float(low), float(high)


def bootstrap_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Ortalama için yüzdelik bootstrap güven aralığı (sabit tohumlu)."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _censored_median(values: pd.Series, censor_at: float) -> float:
    """Sansürlü (ulaşılamayan = NaN) iterasyon sayılarının ortancası."""
    return float(np.median(values.fillna(censor_at).to_numpy()))


# --------------------------------------------------------------------------- #
# Özet tablo
# --------------------------------------------------------------------------- #
def summary_table(
    runs: pd.DataFrame, labels: list[str], curves: dict[str, np.ndarray] | None = None
) -> pd.DataFrame:
    """Her etiket için ödevin istediği metrikleri ve ek göstergeleri hesaplar.

    Sütunlar: en iyi/ortalama/std/ortanca/en kötü, CV, %95 GA (t ve bootstrap),
    optimalite farkı, optimumu bulma oranı, süre, değerlendirme sayısı,
    yakınsama hızı (ilk %99 / %99.5'e ulaşılan iterasyon) ve eğri altı alan.
    """
    rows = []
    optimum = float(runs["optimum"].iloc[0])
    for label in labels:
        group = runs[runs["label"] == label].sort_values("seed")
        values = group["best_value"].to_numpy()
        t_low, t_high = t_confidence_interval(values)
        b_low, b_high = bootstrap_confidence_interval(values)
        n_iterations = int(curves[label].shape[1]) if curves and label in curves else config.ITERATIONS
        row = {
            "label": label,
            "n_runs": len(values),
            "best": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "median": float(np.median(values)),
            "worst": float(values.min()),
            "cv_pct": float(values.std(ddof=1) / values.mean() * 100.0),
            "ci95_t_low": t_low,
            "ci95_t_high": t_high,
            "ci95_boot_low": b_low,
            "ci95_boot_high": b_high,
            "best_gap_pct": float((optimum - values.max()) / optimum * 100.0),
            "mean_gap_pct": float((optimum - values.mean()) / optimum * 100.0),
            "hit_optimum_pct": float(group["hit_optimum"].mean() * 100.0),
            "time_mean_s": float(group["elapsed_s"].mean()),
            "time_std_s": float(group["elapsed_s"].std(ddof=1)),
            "evals_mean": float(group["n_evaluations"].mean()),
            "n_selected_mean": float(group["n_selected"].mean()),
            "weight_mean": float(group["total_weight"].mean()),
            "iters_to_990_median": _censored_median(group["iters_to_990"], n_iterations + 1),
            "iters_to_995_median": _censored_median(group["iters_to_995"], n_iterations + 1),
            "reached_990_pct": float(group["iters_to_990"].notna().mean() * 100.0),
            "reached_995_pct": float(group["iters_to_995"].notna().mean() * 100.0),
            "evals_to_990_median": _censored_median(
                group["evals_to_990"], float(group["n_evaluations"].max()) + 1
            ),
        }
        if curves and label in curves:
            gap_curve = (optimum - curves[label]) / optimum * 100.0
            row["auc_gap_pct"] = float(gap_curve.mean())  # ortalama gap, tüm iterasyon ve çalıştırmalarda
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Hipotez testleri
# --------------------------------------------------------------------------- #
def _aligned(runs: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Tohum x etiket matrisi (aynı tohum = aynı blok)."""
    wide = runs[runs["label"].isin(labels)].pivot(index="seed", columns="label", values="best_value")
    return wide[labels].sort_index()


def normality_table(runs: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Shapiro–Wilk testi; sabit veride test tanımsızdır."""
    rows = []
    for label in labels:
        values = runs[runs["label"] == label]["best_value"].to_numpy()
        if np.ptp(values) == 0.0:
            statistic, p_value, note = float("nan"), float("nan"), "tüm değerler özdeş (varyans 0)"
        else:
            statistic, p_value = stats.shapiro(values)
            note = "normallik reddedildi (p<0.05)" if p_value < 0.05 else "normallik reddedilemedi"
        rows.append({"label": label, "shapiro_w": statistic, "shapiro_p": p_value, "note": note,
                     "unique_values": int(len(np.unique(values)))})
    return pd.DataFrame(rows)


def omnibus_table(runs: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Friedman (eşleştirilmiş) ve Kruskal–Wallis (eşleştirilmemiş) omnibus testleri."""
    wide = _aligned(runs, labels)
    columns = [wide[label].to_numpy() for label in labels]
    n_blocks, k = wide.shape
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if k >= 3:
            try:
                statistic, p_value = stats.friedmanchisquare(*columns)
                rows.append({"test": "Friedman", "statistic": float(statistic), "df": k - 1,
                             "p_value": float(p_value),
                             "effect_size_name": "Kendall W",
                             "effect_size": float(statistic / (n_blocks * (k - 1))),
                             "n_per_group": n_blocks})
            except ValueError:
                pass
        try:
            statistic, p_value = stats.kruskal(*columns)
            n_total = n_blocks * k
            rows.append({"test": "Kruskal-Wallis", "statistic": float(statistic), "df": k - 1,
                         "p_value": float(p_value),
                         "effect_size_name": "epsilon^2",
                         "effect_size": float(statistic * (n_total + 1) / (n_total**2 - 1)),
                         "n_per_group": n_blocks})
        except ValueError:
            pass
    return pd.DataFrame(rows)


def pairwise_table(runs: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Tüm ikililer için Wilcoxon (eşleştirilmiş) + Mann–Whitney, Holm düzeltmeli, etki büyüklüklü.

    ``a12 > 0.5`` ve ``mean_diff > 0`` ilk etiketin (``a``) daha yüksek (daha iyi)
    uygunluk ürettiği anlamına gelir.
    """
    wide = _aligned(runs, labels)
    rows = []
    for first, second in itertools.combinations(labels, 2):
        a, b = wide[first].to_numpy(), wide[second].to_numpy()
        differences = a - b
        if np.all(differences == 0):
            wilcoxon_p, wilcoxon_stat = 1.0, 0.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            wilcoxon_p, wilcoxon_stat = float(result.pvalue), float(result.statistic)
        if np.ptp(np.concatenate([a, b])) == 0.0:
            mwu_p, mwu_stat = 1.0, float(len(a) * len(b) / 2)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = stats.mannwhitneyu(a, b, alternative="two-sided")
            mwu_p, mwu_stat = float(result.pvalue), float(result.statistic)
        a12 = vargha_delaney_a12(a, b) if np.ptp(np.concatenate([a, b])) > 0 else 0.5
        rows.append(
            {
                "a": first, "b": second,
                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "mean_diff": float(a.mean() - b.mean()),
                "median_diff": float(np.median(a) - np.median(b)),
                "n_nonzero_pairs": int((differences != 0).sum()),
                "wilcoxon_stat": wilcoxon_stat, "wilcoxon_p": wilcoxon_p,
                "mannwhitney_u": mwu_stat, "mannwhitney_p": mwu_p,
                "a12": a12, "cliffs_delta": 2.0 * a12 - 1.0, "cohens_d": cohens_d(a, b),
                "effect_magnitude": a12_magnitude(a12),
            }
        )
    table = pd.DataFrame(rows)
    table["wilcoxon_p_holm"] = holm_adjust(table["wilcoxon_p"].to_numpy())
    table["mannwhitney_p_holm"] = holm_adjust(table["mannwhitney_p"].to_numpy())
    table["significant_holm_wilcoxon"] = table["wilcoxon_p_holm"] < 0.05
    table["significant_holm_mannwhitney"] = table["mannwhitney_p_holm"] < 0.05
    return table


# --------------------------------------------------------------------------- #
# Toplu analiz
# --------------------------------------------------------------------------- #
def analyze(
    runs: pd.DataFrame,
    labels: list[str],
    tag: str,
    curves: dict[str, np.ndarray] | None = None,
    directory=config.TABLES_DIR,
) -> dict[str, pd.DataFrame]:
    """Bir deney grubunun özet, normallik, omnibus ve ikili test tablolarını üretir ve yazar."""
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "summary": summary_table(runs, labels, curves),
        "normality": normality_table(runs, labels),
        "omnibus": omnibus_table(runs, labels),
        "pairwise": pairwise_table(runs, labels),
    }
    for name, table in tables.items():
        table.to_csv(directory / f"{name}_{tag}.csv", index=False, encoding="utf-8")
    return tables


def _combine(*datasets: ExperimentData) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    runs = pd.concat([d.runs for d in datasets], ignore_index=True)
    curves: dict[str, np.ndarray] = {}
    for dataset in datasets:
        curves.update(dataset.curves)
    return runs, curves


def run_statistical_analysis(verbose: bool = True) -> dict[str, dict[str, pd.DataFrame]]:
    """Tüm kayıtlı deneyler için istatistik tablolarını üretir."""
    results: dict[str, dict[str, pd.DataFrame]] = {}
    main = load_experiment("main")
    results["main"] = analyze(main.runs, list(config.ALGORITHM_ORDER), "main", main.curves)

    try:
        bonus = load_experiment("bonus_hybrid")
        runs, curves = _combine(main, bonus)
        labels = ["GA", "SA", config.BONUS_ALGORITHM]
        results["bonus"] = analyze(runs, labels, "bonus_hybrid", curves)
    except FileNotFoundError:
        pass

    try:
        sensitivity = load_experiment("sensitivity")
        frames = []
        for algorithm in config.ALGORITHM_ORDER:
            labels = list(
                sensitivity.runs[sensitivity.runs["algorithm"] == algorithm]["label"].unique()
            )
            tables = analyze(sensitivity.runs, labels, f"sensitivity_{algorithm}", sensitivity.curves)
            results[f"sensitivity_{algorithm}"] = tables
            summary = tables["summary"].copy()
            meta = (
                sensitivity.runs[["label", "parameter", "level", "level_value", "is_extension", "is_main_config"]]
                .drop_duplicates("label")
            )
            summary = summary.merge(meta, on="label")
            summary.insert(0, "algorithm", algorithm)
            omnibus = tables["omnibus"]
            kw = omnibus[omnibus["test"] == "Kruskal-Wallis"]
            summary["kruskal_p_all_levels"] = float(kw["p_value"].iloc[0]) if not kw.empty else np.nan
            frames.append(summary)
        pd.concat(frames, ignore_index=True).to_csv(
            config.TABLES_DIR / "summary_sensitivity_all.csv", index=False, encoding="utf-8"
        )
    except FileNotFoundError:
        pass

    for name, tag in (("extra_equal_fe", "equal_fe"), ("extra_aco_beta", "aco_beta"),
                      ("extra_hybrid_share", "hybrid_share")):
        try:
            data = load_experiment(name)
        except FileNotFoundError:
            continue
        results[tag] = analyze(data.runs, list(data.runs["label"].unique()), tag, data.curves)

    try:
        repair = load_experiment("extra_repair")
        for algorithm in ("GA", "BPSO", "SA"):
            labels = list(repair.runs[repair.runs["algorithm"] == algorithm]["label"].unique())
            results[f"repair_{algorithm}"] = analyze(repair.runs, labels, f"repair_{algorithm}", None)
    except FileNotFoundError:
        pass

    capacity_path = config.RAW_DIR / "extra_capacity_runs.csv"
    if capacity_path.is_file():
        capacity = pd.read_csv(capacity_path)
        frames = []
        for ratio, group in capacity.groupby("capacity_ratio"):
            labels = list(config.ALGORITHM_ORDER) + [config.BONUS_ALGORITHM]
            summary = summary_table(group, labels)
            summary.insert(0, "capacity_ratio", ratio)
            frames.append(summary)
            pairwise = pairwise_table(group, labels)
            pairwise.insert(0, "capacity_ratio", ratio)
            pairwise.to_csv(
                config.TABLES_DIR / f"pairwise_capacity_{int(round(ratio * 100))}.csv",
                index=False, encoding="utf-8",
            )
        pd.concat(frames, ignore_index=True).to_csv(
            config.TABLES_DIR / "summary_capacity.csv", index=False, encoding="utf-8"
        )

    if verbose:
        summary = results["main"]["summary"]
        print(summary[["label", "best", "mean", "std", "median", "worst", "mean_gap_pct",
                       "hit_optimum_pct", "time_mean_s", "evals_mean"]].to_string(index=False))
        print(results["main"]["omnibus"].to_string(index=False))
    return results


if __name__ == "__main__":
    run_statistical_analysis()
