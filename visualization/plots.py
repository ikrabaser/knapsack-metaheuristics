"""Tüm grafiklerin üretimi (ham deney sonuçlarından).

Her grafik bir araştırma sorusuna yanıt verir:

=====  ===========================================  ==================================================
No     Dosya                                         Soru
=====  ===========================================  ==================================================
01     fig01_dataset_overview                        Veri seti nasıl bir yapıya sahip?
02     fig02_convergence_all                         Hangi algoritma ne hızla yakınsıyor?
03     fig03_convergence_separate                    Her algoritmanın çalıştırmalar arası değişkenliği?
04     fig04_boxplot                                 Nihai çözüm dağılımları nasıl?
05     fig05_mean_fitness                            Ortalama kalite ve %95 güven aralığı?
06     fig06_execution_time                          Hangisi daha hızlı?
07     fig07_std_comparison                          Hangisi daha kararlı?
08     fig08_quality_vs_runtime                      Kalite–süre dengesi?
09     fig09_sensitivity_boxplots                    Parametre nihai kaliteyi nasıl etkiliyor?
10     fig10_sensitivity_convergence                 Parametre yakınsamayı nasıl etkiliyor?
11     fig11_diversity_diagnostics                   Keşif–sömürü davranışı (çeşitlilik, sıcaklık)?
12     fig12_bonus_hybrid                            Hibrit, bileşenlerinden iyi mi?
13     fig13_hybrid_share                            SA payı hibritte önemli mi?
14     fig14_capacity_ratios                         Sonuçlar kapasite oranına bağlı mı?
15     fig15_knowledge_ablation                      Onarım/sezgisel bilgi bir algoritmayı kayırıyor mu?
16     fig16_equal_budget                            Sonuçlar eşit değerlendirme bütçesinde değişiyor mu?
17     fig05_07_summary_panel                        (Rapor için 05–07 birleşik panel)
=====  ===========================================  ==================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import config
from core.knapsack import KnapsackProblem
from experiments.runner import ExperimentData, load_experiment, prepare_problem
from experiments.statistical_analysis import summary_table, t_confidence_interval
from visualization.style import (
    ALGORITHM_COLORS,
    ALGORITHM_LINESTYLES,
    ALGORITHM_MARKERS,
    AXIS,
    GRID,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ORDINAL_RAMP_3,
    ORDINAL_RAMP_4,
    ORDINAL_RAMP_5,
    OPTIMUM_COLOR,
    apply_style,
    gap_axis,
    label,
    save_figure,
)

OPTIMUM_LABEL = "Gerçek optimum (DP)"
SHORT_PARAMETER = {"GA": "Mutasyon $p_m$", "ACO": "Buharlaşma $\\rho$", "BPSO": "Atalet $w$", "SA": "Soğutma $\\alpha$"}
ITER_LABEL = "İterasyon"
FINAL_LABEL = "Nihai en iyi değer $f(x^*)$"
GAP_LABEL = "Optimalite farkı (%)  [simetrik-log ölçek]"


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def _stats(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Çalıştırmalar (satır) boyunca ortalama ve örneklem std."""
    return curves.mean(axis=0), curves.std(axis=0, ddof=1)


def _gap(curves: np.ndarray, optimum: float) -> np.ndarray:
    """Değer eğrilerini optimalite farkına (%) çevirir."""
    return (optimum - curves) / optimum * 100.0


def _strip(ax: plt.Axes, position: float, values: np.ndarray, color: str, seed: int = 0,
           spread: float = 0.16) -> None:
    """Tek tek çalıştırmaları (eşit değerler dahil) yatay saçılımla gösterir."""
    rng = np.random.default_rng(seed)
    ax.scatter(position + rng.uniform(-spread, spread, len(values)), values, s=6, color=color,
               alpha=0.65, linewidths=0, zorder=3)


def _boxes(ax: plt.Axes, data: list[np.ndarray], positions: list[float], colors: list[str],
           hatches: list[str | None] | None = None, width: float = 0.55) -> None:
    """Kutu grafiği + tek tek noktalar + ortalama işaretçisi (elmas)."""
    artists = ax.boxplot(
        data, positions=positions, widths=width, patch_artist=True, showfliers=False,
        medianprops={"color": INK_PRIMARY, "linewidth": 1.4},
        whiskerprops={"color": INK_SECONDARY, "linewidth": 0.9},
        capprops={"color": INK_SECONDARY, "linewidth": 0.9},
        boxprops={"linewidth": 1.0},
    )
    for index, patch in enumerate(artists["boxes"]):
        patch.set_facecolor(to_rgba(colors[index], 0.28))
        patch.set_edgecolor(colors[index])
        if hatches and hatches[index]:
            patch.set_hatch(hatches[index])
    for index, (values, position) in enumerate(zip(data, positions)):
        _strip(ax, position, np.asarray(values), colors[index], seed=index)
        ax.scatter([position], [np.mean(values)], marker="D", s=18, facecolor="white",
                   edgecolor=INK_PRIMARY, linewidth=1.0, zorder=5)


def _optimum_line(ax: plt.Axes, optimum: float, horizontal: bool = True) -> None:
    """Gerçek optimum referans çizgisi (durum paleti kırmızısı, kesikli)."""
    (ax.axhline if horizontal else ax.axvline)(optimum, color=OPTIMUM_COLOR, linestyle=(0, (4, 3)),
                                              linewidth=1.1, zorder=2)


def _panel_tag(ax: plt.Axes, tag: str) -> None:
    ax.text(-0.02, 1.04, tag, transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=INK_PRIMARY, ha="right", va="bottom")


def _algorithm_legend(ax: plt.Axes, algorithms: list[str], **kwargs) -> None:
    handles = [
        Line2D([0], [0], color=ALGORITHM_COLORS[a], linestyle=ALGORITHM_LINESTYLES[a],
               marker=ALGORITHM_MARKERS[a], markersize=5, linewidth=1.8, label=label(a))
        for a in algorithms
    ]
    ax.legend(handles=handles, **kwargs)


def _short(algorithm: str) -> str:
    """Dar eksenlerde çakışmayı önleyen kısa etiket (Binary PSO -> BPSO)."""
    return {"BPSO": "BPSO", "GA-SA": "GA-SA"}.get(algorithm, label(algorithm))


def _run_values(runs: pd.DataFrame, run_label: str) -> np.ndarray:
    return runs[runs["label"] == run_label].sort_values("seed")["best_value"].to_numpy()


def _optimum_of(runs: pd.DataFrame) -> float:
    return float(runs["optimum"].iloc[0])


# --------------------------------------------------------------------------- #
# 01 — Veri seti
# --------------------------------------------------------------------------- #
def fig_dataset_overview(problem: KnapsackProblem, optimum_solution: np.ndarray) -> list[Path]:
    """Ağırlık/değer/oran dağılımları ve optimum çözümün seçtiği ürünler."""
    fig, axes = plt.subplots(1, 4, figsize=(6.85, 1.95))
    blue = ALGORITHM_COLORS["GA"]
    for ax, data, title, xlabel in (
        (axes[0], problem.weights, "(a) Ağırlık", "$w_i$"),
        (axes[1], problem.values, "(b) Değer", "$v_i$"),
        (axes[2], problem.ratios, "(c) Değer/ağırlık oranı", "$v_i / w_i$"),
    ):
        is_ratio = data is problem.ratios
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 21) if is_ratio else 20
        ax.hist(data, bins=bins, color=to_rgba(blue, 0.55), edgecolor="white", linewidth=0.6)
        if is_ratio:
            ax.set_xscale("log")  # oranlar çok çarpık (maks/min ≈ 280): log eksen
        ax.axvline(float(np.mean(data)), color=INK_PRIMARY, linewidth=0.9, linestyle="--")
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"{xlabel}   (kesikli: ort. = {np.mean(data):.1f})" + ("  [log]" if is_ratio else ""))
    axes[0].set_ylabel("Ürün sayısı")
    ax = axes[3]
    chosen = optimum_solution == 1
    ax.scatter(problem.weights[~chosen], problem.values[~chosen], s=12, facecolor="white",
               edgecolor=INK_MUTED, linewidth=0.8, label="Seçilmeyen", zorder=3)
    ax.scatter(problem.weights[chosen], problem.values[chosen], s=12, color=blue, alpha=0.85,
               linewidths=0, label="Seçilen (optimum)", zorder=4)
    ax.set_title("(d) Ağırlık–değer", loc="left")
    ax.set_xlabel("$w_i$")
    ax.set_ylabel("$v_i$")
    ax.set_ylim(top=118)
    ax.legend(loc="upper center", ncol=2, fontsize=5.6, handletextpad=0.1, columnspacing=0.6, borderpad=0.2)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    fig.tight_layout(w_pad=0.8)
    return save_figure(fig, "fig01_dataset_overview")


# --------------------------------------------------------------------------- #
# 02–03 — Yakınsama
# --------------------------------------------------------------------------- #
def fig_convergence_all(main: ExperimentData) -> list[Path]:
    """Dört algoritmanın ortalama ± std yakınsama eğrileri (değer ve optimalite farkı)."""
    optimum = _optimum_of(main.runs)
    algorithms = list(config.ALGORITHM_ORDER)
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.85, 2.65))
    for algorithm in algorithms:
        curves = main.curves[algorithm]
        iterations = np.arange(1, curves.shape[1] + 1)
        mean, std = _stats(curves)
        left.fill_between(iterations, mean - std, np.minimum(mean + std, optimum + 8),
                          color=ALGORITHM_COLORS[algorithm], alpha=0.13, linewidth=0)
        for ax, transform in ((left, lambda c: c), (right, lambda c: _gap(c, optimum))):
            ax.plot(iterations, transform(mean), color=ALGORITHM_COLORS[algorithm],
                    linestyle=ALGORITHM_LINESTYLES[algorithm], marker=ALGORITHM_MARKERS[algorithm],
                    markevery=25, markersize=3.6, markerfacecolor="white", markeredgewidth=1.0, zorder=3)
    _optimum_line(left, optimum)
    left.set_ylim(top=optimum + 8)
    left.set_title("(a) En iyi değer (ortalama ± 1 std)", loc="left")
    left.set_ylabel(FINAL_LABEL)
    gap_axis(right, top=100)
    right.set_title("(b) Optimalite farkı (ortalama)", loc="left")
    right.set_ylabel(GAP_LABEL)
    for ax in (left, right):
        ax.set_xlabel(ITER_LABEL)
        ax.set_xlim(1, curves.shape[1])
    handles = [
        Line2D([0], [0], color=ALGORITHM_COLORS[a], linestyle=ALGORITHM_LINESTYLES[a], marker=ALGORITHM_MARKERS[a],
               markersize=5, linewidth=1.8, label=label(a)) for a in algorithms
    ]
    handles.append(Line2D([0], [0], color=OPTIMUM_COLOR, linestyle=(0, (4, 3)), linewidth=1.1,
                          label=f"{OPTIMUM_LABEL} = {optimum:.0f}"))
    left.legend(handles=handles, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    return save_figure(fig, "fig02_convergence_all")


def fig_convergence_separate(main: ExperimentData) -> list[Path]:
    """Her algoritma için ayrı yakınsama grafiği: tek tek çalıştırmalar, ortalama ve ± std bandı."""
    optimum = _optimum_of(main.runs)
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 3.75), sharex=True)
    for ax, algorithm, tag in zip(axes.ravel(), config.ALGORITHM_ORDER, "abcd"):
        curves = main.curves[algorithm]
        iterations = np.arange(1, curves.shape[1] + 1)
        color = ALGORITHM_COLORS[algorithm]
        for run in curves:
            ax.plot(iterations, run, color=color, alpha=0.14, linewidth=0.7, zorder=2)
        mean, std = _stats(curves)
        ax.fill_between(iterations, mean - std, np.minimum(mean + std, optimum), color=color, alpha=0.25,
                        linewidth=0, zorder=3)
        ax.plot(iterations, mean, color=color, linewidth=1.7, linestyle=ALGORITHM_LINESTYLES[algorithm],
                zorder=4)
        _optimum_line(ax, optimum)
        lowest = float(curves.min())
        span = optimum - lowest
        ax.set_ylim(lowest - 0.03 * span, optimum + 0.07 * span + 1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.set_title(f"({tag}) {label(algorithm)}", loc="left")
        ax.set_xlim(1, curves.shape[1])
        if tag in "cd":
            ax.set_xlabel(ITER_LABEL)
        if tag in "ac":
            ax.set_ylabel("En iyi değer")
    handles = [
        Line2D([0], [0], color=INK_SECONDARY, linewidth=2.0, label="Ortalama"),
        Patch(facecolor=to_rgba(INK_SECONDARY, 0.25), label="± 1 std"),
        Line2D([0], [0], color=INK_SECONDARY, linewidth=0.8, alpha=0.5, label="Tek tek çalıştırmalar"),
        Line2D([0], [0], color=OPTIMUM_COLOR, linestyle=(0, (4, 3)), linewidth=1.1, label=OPTIMUM_LABEL),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save_figure(fig, "fig03_convergence_separate")


# --------------------------------------------------------------------------- #
# 04 — Kutu grafiği
# --------------------------------------------------------------------------- #
def fig_boxplot(main: ExperimentData) -> list[Path]:
    """Nihai uygunluk dağılımları: tüm aralık ve optimuma yakın bölgenin yakınlaştırması."""
    optimum = _optimum_of(main.runs)
    algorithms = list(config.ALGORITHM_ORDER)
    data = [_run_values(main.runs, a) for a in algorithms]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.85, 2.75), gridspec_kw={"width_ratios": [1.15, 1]})
    colors = [ALGORITHM_COLORS[a] for a in algorithms]
    _boxes(left, data, list(range(len(algorithms))), colors)
    _optimum_line(left, optimum)
    left.set_xticks(range(len(algorithms)), [label(a) for a in algorithms])
    left.set_ylabel(FINAL_LABEL)
    left.set_title("(a) Tüm algoritmalar", loc="left")
    near = [a for a in algorithms if np.min(_run_values(main.runs, a)) >= optimum - 0.01 * optimum]
    _boxes(right, [_run_values(main.runs, a) for a in near], list(range(len(near))),
           [ALGORITHM_COLORS[a] for a in near], width=0.5)
    _optimum_line(right, optimum)
    right.set_xticks(range(len(near)), [label(a) for a in near])
    right.set_ylim(optimum - 14, optimum + 2.5)
    right.set_title("(b) Optimuma yakın bölge (yakınlaştırma)", loc="left")
    handles = [
        Patch(facecolor=to_rgba(INK_SECONDARY, 0.28), edgecolor=INK_SECONDARY, label="Kutu: Q1–Q3, çizgi: medyan"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=INK_PRIMARY,
               label="Ortalama"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK_MUTED, alpha=0.7,
               label=f"Tek tek çalıştırmalar (n={len(data[0])})"),
        Line2D([0], [0], color=OPTIMUM_COLOR, linestyle=(0, (4, 3)), label=OPTIMUM_LABEL),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save_figure(fig, "fig04_boxplot")


# --------------------------------------------------------------------------- #
# 05–07 — Özet çubuk/nokta grafikleri
# --------------------------------------------------------------------------- #
def draw_mean_fitness(ax: plt.Axes, summary: pd.DataFrame, optimum: float) -> None:
    """Ortalama ± %95 güven aralığı (t) nokta grafiği."""
    for position, row in enumerate(summary.itertuples()):
        color = ALGORITHM_COLORS[row.label]
        ax.errorbar(position, row.mean, yerr=[[row.mean - row.ci95_t_low], [row.ci95_t_high - row.mean]],
                    fmt=ALGORITHM_MARKERS[row.label], color=color, markersize=6, markerfacecolor=color,
                    markeredgecolor="white", markeredgewidth=0.7, capsize=4, elinewidth=1.3, capthick=1.3, zorder=4)
        ax.annotate(f"{row.mean:.1f}", (position, row.ci95_t_high), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=6.5, color=INK_SECONDARY)
    _optimum_line(ax, optimum)
    ax.set_xticks(range(len(summary)), [_short(a) for a in summary["label"]])
    ax.set_xlim(-0.6, len(summary) - 0.4)
    ax.set_ylabel("Ortalama $f(x^*)$ ± %95 güven aralığı")
    ax.set_ylim(top=optimum + 6)


def draw_time(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Ortalama çalışma süresi ± std (çubuklar sıfırdan başlar)."""
    positions = range(len(summary))
    ax.bar(positions, summary["time_mean_s"], width=0.6,
           color=[to_rgba(ALGORITHM_COLORS[a], 0.85) for a in summary["label"]], zorder=3)
    ax.errorbar(positions, summary["time_mean_s"], yerr=summary["time_std_s"], fmt="none", ecolor=INK_PRIMARY,
                elinewidth=1.0, capsize=3, zorder=4)
    for position, row in zip(positions, summary.itertuples()):
        ax.annotate(f"{row.time_mean_s:.3f} s", (position, row.time_mean_s + row.time_std_s), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK_SECONDARY)
    ax.set_xticks(list(positions), [_short(a) for a in summary["label"]])
    ax.set_ylabel("Çalışma süresi (s), ortalama ± std")
    ax.set_ylim(0, float((summary["time_mean_s"] + summary["time_std_s"]).max()) * 1.18)


def draw_std(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Nihai uygunluk standart sapması (düşük = daha kararlı)."""
    positions = range(len(summary))
    ax.bar(positions, summary["std"], width=0.6,
           color=[to_rgba(ALGORITHM_COLORS[a], 0.85) for a in summary["label"]], zorder=3)
    for position, row in zip(positions, summary.itertuples()):
        ax.annotate(f"{row.std:.2f}", (position, row.std), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK_SECONDARY)
    ax.set_xticks(list(positions), [_short(a) for a in summary["label"]])
    ax.set_ylabel("Std. sapma ($n-1$ ile)")
    ax.set_ylim(0, float(summary["std"].max()) * 1.18 + 1e-9)


def fig_mean_fitness(summary: pd.DataFrame, optimum: float) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    draw_mean_fitness(ax, summary, optimum)
    ax.set_title("Ortalama nihai kalite (%95 güven aralığı)", loc="left")
    ax.text(0.02, 0.03, "Kırmızı kesikli: gerçek optimum", transform=ax.transAxes, fontsize=7.5, color=OPTIMUM_COLOR)
    fig.tight_layout()
    return save_figure(fig, "fig05_mean_fitness")


def fig_execution_time(summary: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    draw_time(ax, summary)
    ax.set_title("Ortalama çalışma süresi", loc="left")
    fig.tight_layout()
    return save_figure(fig, "fig06_execution_time")


def fig_std_comparison(summary: pd.DataFrame) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    draw_std(ax, summary)
    ax.set_title("Kararlılık: nihai değerin std. sapması", loc="left")
    fig.tight_layout()
    return save_figure(fig, "fig07_std_comparison")


def fig_summary_panel(summary: pd.DataFrame, optimum: float) -> list[Path]:
    """05–07 grafiklerinin rapor için tek satırlık birleşik paneli."""
    fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.45))
    draw_mean_fitness(axes[0], summary, optimum)
    axes[0].set_title("(a) Ortalama kalite\n(%95 güven aralığı)", loc="left")
    draw_time(axes[1], summary)
    axes[1].set_title("(b) Çalışma süresi", loc="left")
    draw_std(axes[2], summary)
    axes[2].set_title("(c) Standart sapma", loc="left")
    fig.tight_layout()
    return save_figure(fig, "fig05_07_summary_panel")


def fig_quality_vs_runtime(summary: pd.DataFrame) -> list[Path]:
    """Çözüm kalitesi (optimalite farkı) ile çalışma süresi ilişkisi; sol-alt köşe en iyidir."""
    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    for row in summary.itertuples():
        color = ALGORITHM_COLORS[row.label]
        ax.errorbar(row.time_mean_s, row.mean_gap_pct, xerr=row.time_std_s,
                    yerr=[[0.0], [row.std / row.optimum_ref * 100.0]],
                    fmt=ALGORITHM_MARKERS[row.label], color=color, markersize=7, markeredgecolor="white",
                    markeredgewidth=0.7, elinewidth=1.0, capsize=2.5, zorder=4)
        ax.annotate(label(row.label), (row.time_mean_s, row.mean_gap_pct), xytext=(8, 6),
                    textcoords="offset points", fontsize=7, color=INK_PRIMARY)
    ax.set_xscale("log")
    gap_axis(ax, top=10)
    ax.set_xlabel("Ortalama çalışma süresi (s)  [log ölçek]")
    ax.set_ylabel(GAP_LABEL)
    ax.set_title("Kalite–süre dengesi", loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    fig.tight_layout()
    return save_figure(fig, "fig08_quality_vs_runtime")


# --------------------------------------------------------------------------- #
# 09–10 — Parametre duyarlılığı
# --------------------------------------------------------------------------- #
def _level_table(sensitivity: ExperimentData, algorithm: str) -> pd.DataFrame:
    """Bir algoritmanın düzeylerini (sıralı) ve işaretlerini döndürür."""
    runs = sensitivity.runs[sensitivity.runs["algorithm"] == algorithm]
    table = runs.drop_duplicates("label")[["label", "level", "level_value", "is_extension", "is_main_config"]]
    return table.sort_values("level_value").reset_index(drop=True)


def _ramp_for(n_levels: int) -> tuple[str, ...]:
    return {3: ORDINAL_RAMP_3, 4: ORDINAL_RAMP_4, 5: ORDINAL_RAMP_5}[n_levels]


def fig_sensitivity_boxplots(sensitivity: ExperimentData) -> list[Path]:
    """Dört algoritma için parametre düzeylerine göre nihai uygunluk dağılımı (2x2)."""
    grids = config.sensitivity_grids(100)
    optimum = _optimum_of(sensitivity.runs)
    fig, axes = plt.subplots(1, 4, figsize=(6.85, 2.55))
    for ax, algorithm, tag in zip(axes.ravel(), config.ALGORITHM_ORDER, "abcd"):
        table = _level_table(sensitivity, algorithm)
        ramp = _ramp_for(len(table))
        data = [_run_values(sensitivity.runs, lab) for lab in table["label"]]
        hatches = ["///" if ext else None for ext in table["is_extension"]]
        _boxes(ax, data, list(range(len(table))), list(ramp), hatches=hatches)
        _optimum_line(ax, optimum)
        ticks = []
        for row in table.itertuples():
            note = "\n(ana)" if row.is_main_config else ("\n(ek)" if row.is_extension else "")
            ticks.append(f"{row.level}{note}")
        ax.set_xticks(range(len(table)), ticks)
        ax.set_xlabel(SHORT_PARAMETER[algorithm])
        ax.set_ylabel("Nihai $f(x^*)$" if tag == "a" else "")
        ax.set_title(f"({tag}) {label(algorithm)}", loc="left")
        ax.set_xlim(-0.6, len(table) - 0.4)
        lower = min(float(np.min(d)) for d in data)
        ax.set_ylim(lower - 0.06 * (optimum - lower) - 2, optimum + 0.12 * (optimum - lower) + 3)
    handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=INK_PRIMARY,
               label="Ortalama"),
        Patch(facecolor=to_rgba(INK_SECONDARY, 0.28), edgecolor=INK_SECONDARY, label="Kutu: Q1–Q3, çizgi: medyan"),
        Patch(facecolor="white", edgecolor=INK_SECONDARY, hatch="///", label="(ek): ödev dışı post-hoc düzey"),
        Line2D([0], [0], color=OPTIMUM_COLOR, linestyle=(0, (4, 3)), label=OPTIMUM_LABEL),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
    return save_figure(fig, "fig09_sensitivity_boxplots")


def fig_sensitivity_convergence(sensitivity: ExperimentData) -> list[Path]:
    """Parametre düzeylerine göre ortalama yakınsama eğrileri (optimalite farkı, 2x2)."""
    grids = config.sensitivity_grids(100)
    optimum = _optimum_of(sensitivity.runs)
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 3.6), sharex=True)
    for ax, algorithm, tag in zip(axes.ravel(), config.ALGORITHM_ORDER, "abcd"):
        table = _level_table(sensitivity, algorithm)
        ramp = _ramp_for(len(table))
        for index, row in enumerate(table.itertuples()):
            curves = sensitivity.curves[row.label]
            iterations = np.arange(1, curves.shape[1] + 1)
            mean = curves.mean(axis=0)
            ax.plot(iterations, _gap(mean, optimum), color=ramp[index], linestyle="--" if row.is_extension else "-",
                    linewidth=1.9, label=f"{row.level}" + (" (ek)" if row.is_extension else ""))
        gap_axis(ax, top=1000)  # üstte lejant için boş bir ondalık bırakılır
        ax.set_title(f"({tag}) {label(algorithm)} – {SHORT_PARAMETER[algorithm]}", loc="left")
        ax.legend(loc="upper center", ncol=len(table), fontsize=6, handlelength=1.4, columnspacing=0.9,
                  borderpad=0.2)
        if tag in "cd":
            ax.set_xlabel(ITER_LABEL)
        if tag in "ac":
            ax.set_ylabel("Optimalite farkı (%)")
        ax.set_xlim(1, curves.shape[1])
    fig.tight_layout()
    return save_figure(fig, "fig10_sensitivity_convergence")


# --------------------------------------------------------------------------- #
# 11 — Keşif–sömürü tanılama
# --------------------------------------------------------------------------- #
def fig_diversity(main: ExperimentData) -> list[Path]:
    """Popülasyon çeşitliliği, ACO feromon entropisi, SA sıcaklığı ve kötü-hamle kabul oranı."""
    fig, axes = plt.subplots(2, 2, figsize=(3.9, 3.55))
    ax = axes[0, 0]
    for algorithm in ("GA", "ACO", "BPSO"):
        curves = main.diversity[algorithm]
        mean, std = _stats(curves)
        iterations = np.arange(1, len(mean) + 1)
        ax.fill_between(iterations, mean - std, mean + std, color=ALGORITHM_COLORS[algorithm], alpha=0.13, linewidth=0)
        ax.plot(iterations, mean, color=ALGORITHM_COLORS[algorithm], linestyle=ALGORITHM_LINESTYLES[algorithm],
                label=label(algorithm))
    ax.set_title("(a) Çeşitlilik", loc="left")
    ax.set_xlabel(ITER_LABEL)
    ax.set_ylabel("Ort. Hamming / n")
    ax.legend(loc="upper right")
    ax = axes[0, 1]
    entropy = main.extras["ACO"]["mean_pheromone_entropy"]
    mean, std = _stats(entropy)
    iterations = np.arange(1, len(mean) + 1)
    ax.fill_between(iterations, mean - std, mean + std, color=ALGORITHM_COLORS["ACO"], alpha=0.15, linewidth=0)
    ax.plot(iterations, mean, color=ALGORITHM_COLORS["ACO"], linestyle=ALGORITHM_LINESTYLES["ACO"])
    ax.set_title("(b) ACO entropisi", loc="left")
    ax.set_xlabel(ITER_LABEL)
    ax.set_ylabel("Norm. entropi")
    ax = axes[1, 0]
    temperature = main.extras["SA"]["temperature"].mean(axis=0)
    ax.plot(np.arange(1, len(temperature) + 1), temperature, color=ALGORITHM_COLORS["SA"],
            linestyle=ALGORITHM_LINESTYLES["SA"], linewidth=2.2)
    ax.set_yscale("log")
    ax.set_title("(c) SA sıcaklığı", loc="left")
    ax.set_xlabel(ITER_LABEL)
    ax.set_ylabel("Sıcaklık $T$  [log ölçek]")
    ax = axes[1, 1]
    rate = main.extras["SA"]["worse_accept_rate"]
    mean = np.nanmean(rate, axis=0)
    ax.plot(np.arange(1, len(mean) + 1), mean, color=ALGORITHM_COLORS["SA"], linestyle=ALGORITHM_LINESTYLES["SA"],
            linewidth=2.2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(d) SA kötü-hamle kabulü", loc="left")
    ax.set_xlabel(ITER_LABEL)
    ax.set_ylabel("Kabul oranı")
    fig.tight_layout()
    return save_figure(fig, "fig11_diversity_diagnostics")


# --------------------------------------------------------------------------- #
# 12–13 — Bonus: GA-SA hibriti
# --------------------------------------------------------------------------- #
def fig_bonus_hybrid(main: ExperimentData, bonus: ExperimentData) -> list[Path]:
    """GA, SA ve GA-SA: yakınsama (optimalite farkı) ve nihai dağılım."""
    optimum = _optimum_of(main.runs)
    algorithms = ["GA", "SA", config.BONUS_ALGORITHM]
    curves = {"GA": main.curves["GA"], "SA": main.curves["SA"], config.BONUS_ALGORITHM: bonus.curves[config.BONUS_ALGORITHM]}
    runs = pd.concat([main.runs, bonus.runs], ignore_index=True)
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.85, 2.5))
    for algorithm in algorithms:
        mean = curves[algorithm].mean(axis=0)
        iterations = np.arange(1, len(mean) + 1)
        gap = _gap(mean, optimum)
        left.plot(iterations, gap, color=ALGORITHM_COLORS[algorithm], linestyle=ALGORITHM_LINESTYLES[algorithm],
                  marker=ALGORITHM_MARKERS[algorithm], markevery=25, markersize=4.5, markerfacecolor="white",
                  markeredgewidth=1.2)
    gap_axis(left, top=10)
    left.set_xlabel(ITER_LABEL)
    left.set_ylabel(GAP_LABEL)
    left.set_title("(a) Yakınsama (ortalama optimalite farkı)", loc="left")
    left.set_xlim(1, 200)
    _algorithm_legend(left, algorithms, loc="upper right")
    data = [_run_values(runs, a) for a in algorithms]
    _boxes(right, data, list(range(3)), [ALGORITHM_COLORS[a] for a in algorithms], width=0.5)
    _optimum_line(right, optimum)
    right.set_xticks(range(3), [label(a) for a in algorithms])
    right.set_ylim(optimum - 14, optimum + 2.5)
    right.set_ylabel(FINAL_LABEL)
    right.set_title("(b) Nihai değer dağılımı", loc="left")
    fig.tight_layout()
    return save_figure(fig, "fig12_bonus_hybrid")


def fig_hybrid_share(share: ExperimentData) -> list[Path]:
    """Hibritte nesil başına SA değerlendirme payının etkisi (0 = saf GA)."""
    optimum = _optimum_of(share.runs)
    shares = list(config.HYBRID_SA_SHARES)
    labels = [f"GA-SA|sa={s}" for s in shares]
    data = [_run_values(share.runs, lab) for lab in labels]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    _boxes(ax, data, list(range(len(shares))), list(ORDINAL_RAMP_5))
    _optimum_line(ax, optimum)
    ax.set_xticks(range(len(shares)), [f"{s}\n(saf GA)" if s == 0 else str(s) for s in shares])
    ax.set_xlabel("Nesil başına SA değerlendirmesi (toplam 40)")
    ax.set_ylabel(FINAL_LABEL)
    ax.set_ylim(optimum - 14, optimum + 2.5)
    ax.set_title("Hibritte SA payı ablasyonu", loc="left")
    fig.tight_layout()
    return save_figure(fig, "fig13_hybrid_share")


# --------------------------------------------------------------------------- #
# 14 — Kapasite oranları
# --------------------------------------------------------------------------- #
def fig_capacity_ratios(capacity_summary: pd.DataFrame) -> list[Path]:
    """Kapasite oranına göre ortalama optimalite farkı ve optimumu bulma oranı."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.85, 2.55))
    algorithms = [*config.ALGORITHM_ORDER, config.BONUS_ALGORITHM]
    ratios = sorted(capacity_summary["capacity_ratio"].unique())
    for algorithm in algorithms:
        subset = capacity_summary[capacity_summary["label"] == algorithm].sort_values("capacity_ratio")
        style = dict(color=ALGORITHM_COLORS[algorithm], linestyle=ALGORITHM_LINESTYLES[algorithm],
                     marker=ALGORITHM_MARKERS[algorithm], markersize=6, markerfacecolor="white",
                     markeredgewidth=1.3)
        left.plot(subset["capacity_ratio"] * 100, subset["mean_gap_pct"], **style)
        right.plot(subset["capacity_ratio"] * 100, subset["hit_optimum_pct"], **style)
    gap_axis(left, top=10)
    left.set_title("(a) Ortalama optimalite farkı", loc="left")
    left.set_ylabel(GAP_LABEL)
    right.set_title("(b) Gerçek optimumu bulan çalıştırma oranı", loc="left")
    right.set_ylabel("Optimumu bulma oranı (%)")
    right.set_ylim(-3, 103)
    for ax in (left, right):
        ax.set_xticks([r * 100 for r in ratios], [f"%{r * 100:.0f}" for r in ratios])
        ax.set_xlabel("Kapasite oranı  $C / \\sum w_i$")
    handles = [
        Line2D([0], [0], color=ALGORITHM_COLORS[a], linestyle=ALGORITHM_LINESTYLES[a], marker=ALGORITHM_MARKERS[a],
               markersize=5, linewidth=1.6, label=label(a)) for a in algorithms
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(algorithms), bbox_to_anchor=(0.5, -0.02), fontsize=6.8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save_figure(fig, "fig14_capacity_ratios")


# --------------------------------------------------------------------------- #
# 15 — Bilgi enjeksiyonu ablasyonu
# --------------------------------------------------------------------------- #
def fig_knowledge_ablation(repair: ExperimentData, aco_beta: ExperimentData) -> list[Path]:
    """Onarım stratejisi (GA, BPSO, SA) ve ACO sezgisel bilgi (beta) ablasyonu."""
    optimum = _optimum_of(repair.runs)
    strategies = [("ratio", "oran-çıkar"), ("random", "rastgele-çıkar"), ("ratio_fill", "oran-çıkar+doldur")]
    strategy_colors = [ALGORITHM_COLORS["GA"], ALGORITHM_COLORS["ACO"], ALGORITHM_COLORS["BPSO"]]
    fig, axes = plt.subplots(1, 4, figsize=(6.85, 2.6))
    for ax, algorithm, tag in zip(axes[:3], ("GA", "BPSO", "SA"), "abc"):
        data = [_run_values(repair.runs, f"{algorithm}|{key}") for key, _ in strategies]
        _boxes(ax, data, [0, 1, 2], strategy_colors, width=0.6)
        _optimum_line(ax, optimum)
        ax.set_xticks([0, 1, 2], ["oran", "rastgele", "oran+\ndoldur"], fontsize=6.5)
        ax.set_title(f"({tag}) {label(algorithm)}", loc="left")
        lower = min(float(np.min(d)) for d in data)
        ax.set_ylim(lower - 0.08 * (optimum - lower) - 3, optimum + 0.1 * (optimum - lower) + 3)
        if tag == "a":
            ax.set_ylabel(FINAL_LABEL)
    ax = axes[3]
    labels = [f"ACO|beta={b:g}" for b in config.ACO_BETA_LEVELS]
    data = [_run_values(aco_beta.runs, lab) for lab in labels]
    _boxes(ax, data, list(range(len(labels))), list(ORDINAL_RAMP_3), width=0.6)
    _optimum_line(ax, optimum)
    ax.set_xticks(range(len(labels)), [f"β={b:g}" for b in config.ACO_BETA_LEVELS])
    ax.set_title("(d) ACO sezgisel bilgi", loc="left")
    lower = min(float(np.min(d)) for d in data)
    ax.set_ylim(lower - 0.08 * (optimum - lower) - 3, optimum + 0.1 * (optimum - lower) + 3)
    fig.tight_layout()
    return save_figure(fig, "fig15_knowledge_ablation")


# --------------------------------------------------------------------------- #
# 16 — Eşit değerlendirme bütçesi
# --------------------------------------------------------------------------- #
def fig_equal_budget(main: ExperimentData, equal: ExperimentData) -> list[Path]:
    """Ana deney (200 iterasyon) ile eşit sert bütçe (8040 değerlendirme) karşılaştırması."""
    optimum = _optimum_of(main.runs)
    algorithms = list(config.ALGORITHM_ORDER)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for position, algorithm in enumerate(algorithms):
        color = ALGORITHM_COLORS[algorithm]
        for offset, runs, filled in ((-0.12, main.runs, False), (0.12, equal.runs, True)):
            values = _run_values(runs, algorithm)
            low, high = t_confidence_interval(values)
            ax.errorbar(position + offset, values.mean(), yerr=[[values.mean() - low], [high - values.mean()]],
                        fmt=ALGORITHM_MARKERS[algorithm], color=color, markersize=8,
                        markerfacecolor=color if filled else "white", markeredgecolor=color,
                        markeredgewidth=1.5, capsize=4, elinewidth=1.4, zorder=4)
    _optimum_line(ax, optimum)
    ax.set_xticks(range(len(algorithms)), [label(a) for a in algorithms])
    ax.set_xlim(-0.6, len(algorithms) - 0.4)
    ax.set_ylabel("Ortalama $f(x^*)$ ± %95 güven aralığı")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=INK_SECONDARY,
               markeredgewidth=1.5, label="Ana deney: 200 iterasyon (FE farklı)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK_SECONDARY, markeredgecolor=INK_SECONDARY,
               label=f"Eşit bütçe: {config.EQUAL_BUDGET_EVALUATIONS} FE"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_title("Eşit değerlendirme bütçesi denetimi", loc="left")
    fig.tight_layout()
    return save_figure(fig, "fig16_equal_budget")


# --------------------------------------------------------------------------- #
# Toplu üretim
# --------------------------------------------------------------------------- #
def generate_all_figures(verbose: bool = True) -> list[Path]:
    """Kayıtlı ham sonuçlardan tüm grafikleri üretir (var olmayan deneyler atlanır)."""
    apply_style()
    problem, optimum = prepare_problem(write_logs=False)
    main = load_experiment("main")
    summary = summary_table(main.runs, list(config.ALGORITHM_ORDER), main.curves)
    summary["optimum_ref"] = optimum.value
    created: list[Path] = []

    def add(paths: list[Path]) -> None:
        created.extend(paths)
        if verbose:
            print("  yazıldı:", paths[0].name)

    add(fig_dataset_overview(problem, optimum.solution))
    add(fig_convergence_all(main))
    add(fig_convergence_separate(main))
    add(fig_boxplot(main))
    add(fig_mean_fitness(summary, optimum.value))
    add(fig_execution_time(summary))
    add(fig_std_comparison(summary))
    add(fig_summary_panel(summary, optimum.value))
    add(fig_quality_vs_runtime(summary))
    add(fig_diversity(main))

    for name, function in (
        ("sensitivity", lambda d: [fig_sensitivity_boxplots(d), fig_sensitivity_convergence(d)]),
    ):
        try:
            data = load_experiment(name)
        except FileNotFoundError:
            continue
        for paths in function(data):
            add(paths)

    try:
        bonus = load_experiment("bonus_hybrid")
        add(fig_bonus_hybrid(main, bonus))
    except FileNotFoundError:
        pass
    try:
        add(fig_hybrid_share(load_experiment("extra_hybrid_share")))
    except FileNotFoundError:
        pass
    capacity_path = config.TABLES_DIR / "summary_capacity.csv"
    if capacity_path.is_file():
        add(fig_capacity_ratios(pd.read_csv(capacity_path)))
    try:
        add(fig_knowledge_ablation(load_experiment("extra_repair"), load_experiment("extra_aco_beta")))
    except FileNotFoundError:
        pass
    try:
        add(fig_equal_budget(main, load_experiment("extra_equal_fe")))
    except FileNotFoundError:
        pass
    return created


if __name__ == "__main__":
    generate_all_figures()
