"""Raporun deney sonucu bölümleri (4: ana sonuçlar, 5: parametre duyarlılığı).

Sayıların tamamı CSV tablolarından okunur. Nitel cümleler, ``_check`` denetimleriyle sayısal
sonuçlara bağlanmıştır; sonuçlar değişirse üretim hata verir ve metin gözden geçirilmelidir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from reportlab.platypus import Flowable, Spacer

import config
from experiments.runner import load_experiment
from reports.pdf_helpers import make_table
from reports.report_data import cite, figure_path, num, pval
from reports.sections_static import Ctx

NAMES = {"GA": "GA", "ACO": "ACO", "BPSO": "Binary PSO", "SA": "SA", "GA-SA": "GA-SA"}


def _sci(value: float) -> str:
    """Bilimsel gösterim (ReportLab üst simgesiyle): 7e-08 -> 7×10<super>-8</super>."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}×10<super>{int(exponent)}</super>"


def _check(condition: bool, message: str) -> None:
    """Metindeki nitel iddianın sayısal sonuçla tutarlılığını denetler."""
    if not condition:
        raise AssertionError(f"Rapor metni sonuçlarla tutarsız: {message}")


def main_claims(ctx: Ctx) -> dict[str, str]:
    """Ana deneyden türetilen sıralamalar (metindeki isimlendirmeyi doğrular)."""
    s = ctx.data.summary_main
    claims = {
        "best_mean": s["mean"].idxmax(),
        "fastest": s["time_mean_s"].idxmin(),
        "slowest": s["time_mean_s"].idxmax(),
        "most_stable": s["std"].idxmin(),
        "least_stable": s["std"].idxmax(),
        "worst_mean": s["mean"].idxmin(),
    }
    _check(claims["best_mean"] == "ACO", "en iyi ortalama ACO olmalı")
    _check(set(s["time_mean_s"].nsmallest(2).index) == {"GA", "BPSO"}, "en hızlı iki yöntem GA ve Binary PSO olmalı")
    _check(claims["slowest"] == "ACO", "en yavaş ACO olmalı")
    _check(claims["most_stable"] == "ACO", "en kararlı ACO olmalı")
    _check(claims["least_stable"] == "BPSO" and claims["worst_mean"] == "BPSO", "en kötü/kararsız BPSO olmalı")
    return claims


# --------------------------------------------------------------------------- #
# 4. Deney sonuçları
# --------------------------------------------------------------------------- #
def section_results(ctx: Ctx) -> list[Flowable]:
    d = ctx.data
    s = d.summary_main
    main_claims(ctx)
    runs = pd.read_csv(config.RAW_DIR / "main_runs.csv")
    capacity = float(runs["capacity"].iloc[0])
    max_weight = float(runs["total_weight"].max())
    _check(bool(runs["feasible"].all()) and max_weight <= capacity + 1e-9, "tüm nihai çözümler uygun olmalı")
    n_runs = int(s["n_runs"].iloc[0])
    ratio_aco_ga = s.loc["ACO", "time_mean_s"] / s.loc["GA", "time_mean_s"]

    out: list[Flowable] = [ctx.h1("4. Deney Sonuçları")]
    out.append(ctx.h2("4.1 Çözüm kalitesi, süre ve istatistiksel anlamlılık"))

    header = ["Algoritma", "En iyi", "Ortalama", "Std", "Medyan", "En kötü", "CV (%)", "Ort. fark (%)",
              "Optimum bulma", "Süre (s)", "FE"]
    rows = [header]
    for a in config.ALGORITHM_ORDER:
        r = s.loc[a]
        rows.append([NAMES[a], num(r["best"], 0), num(r["mean"], 2), num(r["std"], 2), num(r["median"], 1),
                     num(r["worst"], 0), num(r["cv_pct"], 3), num(r["mean_gap_pct"], 3),
                     f"{r['hit_optimum_pct']:.1f}%", f"{r['time_mean_s']:.3f} ± {r['time_std_s']:.3f}",
                     f"{r['evals_mean']:.0f}"])
    out.append(ctx.fig.table_title(
        f"Ana karşılaştırma ({n_runs} bağımsız çalıştırma; gerçek optimum = {num(d.reference['optimum_value_dp'], 0)}). "
        "Std: örneklem std. sapması; Optimum bulma: optimuma ulaşan çalıştırma oranı."))
    out.append(make_table(rows, [ctx.width * f for f in (0.115, 0.06, 0.1, 0.055, 0.085, 0.075, 0.065, 0.09, 0.11, 0.175, 0.07)],
                          ctx.styles))
    out.append(Spacer(1, 3))

    aco, ga, sa, pso = (s.loc[k] for k in ("ACO", "GA", "SA", "BPSO"))
    out.append(ctx.p(
        f"Tüm nihai çözümler uygundur (en büyük toplam ağırlık {num(max_weight, 0)} ≤ <i>C</i> = {num(capacity, 1)}). "
        f"<b>ACO</b> {n_runs} çalıştırmanın tamamında gerçek optimumu ({num(aco['best'], 0)}) bulmuş (std = 0); "
        f"<b>GA</b> ortalama {num(ga['mean'], 2)} ile ikinci (optimum bulma %{ga['hit_optimum_pct']:.0f}); "
        f"<b>SA</b> {num(sa['mean'], 2)} (%{sa['hit_optimum_pct']:.1f}); <b>Binary PSO</b> (<i>w</i> = 0.9) "
        f"{num(pso['mean'], 1)} (fark %{num(pso['mean_gap_pct'], 2)}) ile geride kalmış, optimuma hiç ulaşamamıştır. "
        f"En yüksek <i>tekil</i> sonuç GA, ACO ve SA için aynıdır ({num(ga['best'], 0)}); algoritmalar tekil en iyi "
        "değerle değil ortalama ve dağılımla ayrışır. GA ve Binary PSO en hızlı "
        f"({num(ga['time_mean_s'], 3)} s ve {num(pso['time_mean_s'], 3)} s), ACO en yavaştır "
        f"({num(aco['time_mean_s'], 2)} s; GA'nın ≈{ratio_aco_ga:.0f} katı)."))

    om = d.omnibus_main
    fr, kw = om.loc["Friedman"], om.loc["Kruskal-Wallis"]
    pw = d.pairwise_main
    n_pairs = len(pw)
    n_sig = int(pw["significant_holm_wilcoxon"].sum())
    n_sig_mw = int(pw["significant_holm_mannwhitney"].sum())
    shapiro_reject = [NAMES[a] for a in config.ALGORITHM_ORDER if d.normality_main.loc[a, "shapiro_p"] < 0.05]
    out.append(ctx.p(
        f"Friedman χ²(3) = {num(fr['statistic'], 1)}, <i>p</i> {pval(fr['p_value'])} (Kendall <i>W</i> = "
        f"{num(fr['effect_size'], 2)}); Kruskal–Wallis <i>H</i> = {num(kw['statistic'], 1)}, <i>p</i> {pval(kw['p_value'])} "
        f"(ε² = {num(kw['effect_size'], 2)}). Normallik {', '.join(shapiro_reject)} için reddedilmiş, ACO'da değerler "
        f"özdeş olduğundan test tanımsızdır. İkili testlerde (Tablo 4) Holm sonrası anlamlı çift sayısı: Wilcoxon "
        f"{n_sig}/{n_pairs}, Mann–Whitney {n_sig_mw}/{n_pairs} (iki yaklaşım uyumlu)."))

    rows = [["Karşılaştırma (a – b)", "Δ ortalama", "Wilcoxon <i>p</i><sub>Holm</sub>", "Mann–Whitney <i>p</i><sub>Holm</sub>",
             "<i>A</i><sub>12</sub>", "Cohen <i>d</i>", "Etki (VD)"]]
    for r in pw.itertuples():
        rows.append([f"{NAMES[r.a]} – {NAMES[r.b]}", f"{r.mean_diff:+.2f}", pval(r.wilcoxon_p_holm), pval(r.mannwhitney_p_holm),
                     num(r.a12, 3), num(r.cohens_d, 2), r.effect_magnitude])
    out.append(ctx.fig.table_title(
        "İkili testler (n = 30 çift; Holm düzeltmeli). <i>A</i><sub>12</sub> &gt; 0.5 ve Δ &gt; 0: ilk algoritma "
        "daha yüksek (daha iyi) değer üretir."))
    out.append(make_table(rows, [ctx.width * f for f in (0.22, 0.12, 0.17, 0.19, 0.1, 0.1, 0.1)], ctx.styles))
    out.append(Spacer(1, 3))
    gap = abs(pw[(pw["a"] == "GA") & (pw["b"] == "ACO")]["mean_diff"].iloc[0])
    pso_gap = abs(pw[(pw["a"] == "GA") & (pw["b"] == "BPSO")]["mean_diff"].iloc[0])
    out.append(ctx.p(
        f"<b>İstatistiksel ve pratik anlamlılık.</b> ACO–GA farkı anlamlıdır, ancak <i>pratik</i> fark yalnızca "
        f"{num(gap, 1)} birimdir (optimumun %{gap / d.reference['optimum_value_dp'] * 100:.3f} kadarı); asıl ayrım "
        f"ortalamada değil <i>güvenilirliktedir</i> (optimum bulma %100 – %{ga['hit_optimum_pct']:.0f}). Binary "
        f"PSO'nun ≈{pso_gap:.0f} birimlik açığı ise hem anlamlı hem pratik olarak büyüktür."))

    # ---- 4.2 Yakınsama ----
    out.append(ctx.h2("4.2 Yakınsama davranışı"))
    t99 = s["iters_to_990_median"]
    out.append(ctx.p(
        f"Optimumun %99'una ulaşmak için medyan iterasyon sayısı ACO {num(t99['ACO'], 0)}, GA {num(t99['GA'], 0)}, "
        f"SA {num(t99['SA'], 1)} iterasyondur; Binary PSO bu eşiğe hiç ulaşamamıştır. ACO'nun hızlı yakınsaması, değer/ağırlık "
        "sezgisinin ilk iterasyonlardan itibaren iyi çözümler üretmesindendir; GA bu bilgiye yalnızca onarımla "
        "dolaylı erişir; SA ise <i>T</i> yeterince düşene (≈50. iterasyon) dek dolaşır."))
    out.append(ctx.fig.figure(
        figure_path("fig02_convergence_all"), ctx.width,
        "Ortalama best-so-far yakınsama eğrileri (30 çalıştırma): (a) değer (± 1 std bandı), (b) optimalite farkı "
        "(simetrik-log ölçek). İşaretçi ve çizgi stili renge ek (ikincil) kodlamadır."))
    out.append(ctx.fig.figure(
        figure_path("fig03_convergence_separate"), ctx.width * 0.8,
        "Her algoritma için ayrı yakınsama: soluk çizgiler tek tek çalıştırmalar, kalın çizgi ortalama, bant ± 1 std."))

    # ---- 4.3 Dağılım ----
    out.append(ctx.h2("4.3 Dağılım, kararlılık ve kalite–süre dengesi"))
    cv = s["cv_pct"]
    out.append(ctx.p(
        f"<b>Kararlılık.</b> Std. sapma sırası ACO ({num(aco['std'], 2)}) &lt; GA ({num(ga['std'], 2)}) &lt; SA "
        f"({num(sa['std'], 2)}) &lt; Binary PSO ({num(pso['std'], 2)}). ACO ve GA'nın dağılımları optimuma yapışıktır "
        f"(en kötü {num(aco['worst'], 0)} ve {num(ga['worst'], 0)}); SA'nın alt kuyruğu {num(sa['worst'], 0)} "
        f"değerine uzanır. CV en fazla %{cv.max():.2f} ({NAMES[cv.idxmax()]}); değerler dar bir aralıkta olduğundan "
        "std. yanında dağılım biçimi ve optimum bulma oranı birlikte okunmalıdır."))
    out.append(ctx.fig.figure(
        figure_path("fig04_boxplot"), ctx.width,
        "Nihai değerlerin dağılımı (30 çalıştırma): (a) tüm algoritmalar, (b) optimuma yakın bölge. Noktalar tek "
        "tek çalıştırmalardır."))
    out.append(ctx.fig.figure(
        figure_path("fig05_07_summary_panel"), ctx.width,
        "(a) Ortalama nihai değer ± %95 güven aralığı (t), (b) ortalama süre ± std, (c) nihai değerin std. sapması. "
        "Ayrı dosyalar: fig05, fig06, fig07."))
    quality_text: list[Flowable] = [
        ctx.p(
            "<b>Kalite–süre.</b> Sol-alt köşe idealdir. GA, en hızlı yöntemlerden biriyle optimuma "
            f"%{ga['mean_gap_pct']:.2f} yakındır; ACO sıfır farka ulaşır ama ≈{ratio_aco_ga:.0f}× yavaştır. SA, GA'dan hem "
            "yavaş hem daha az isabetlidir; Binary PSO GA ile aynı hızdadır ama kalitesi çok düşüktür. GA ile ACO "
            "<i>Pareto-etkin</i> uçlardır (hız ↔ güvenilirlik).", "body_left"),
        ctx.p(
            "<i>Not:</i> süreler uygulamaya bağlıdır; ACO'nun inşası <i>n</i> adımlı olduğundan FE eşit olsa da "
            "iterasyon başı maliyeti yüksektir.", "note"),
    ]
    out.append(ctx.fig.beside_text(
        figure_path("fig08_quality_vs_runtime"),
        "Kalite–süre ilişkisi (ortalama optimalite farkı, simetrik-log; yatay çubuk: süre std.).",
        quality_text, ctx.width, figure_fraction=0.46))

    # ---- 4.4 Keşif–sömürü tanılama ----
    main = load_experiment("main")
    div = {a: main.diversity[a].mean(axis=0) for a in ("GA", "ACO", "BPSO")}
    out.append(ctx.h2("4.4 Keşif–sömürü ve çeşitlilik"))
    wr = main.extras["SA"]["worse_accept_rate"]
    early, late = float(np.nanmean(wr[:, :10])), float(np.nanmean(wr[:, -10:]))
    _check(div["ACO"][0] < 0.5 * div["GA"][0], "ACO başlangıç çeşitliliği GA'dan çok düşük olmalı")
    _check(div["BPSO"][-1] > div["GA"][-1], "BPSO son çeşitliliği GA'dan yüksek olmalı")
    _check(early > 10 * late, "SA kötü-hamle kabulü zamanla düşmeli")
    entropy = main.extras["ACO"]["mean_pheromone_entropy"].mean(axis=0)
    aco_lock = int(np.argmax(np.abs(entropy - entropy[-1]) < 0.002)) + 1
    temperature = main.extras["SA"]["temperature"].mean(axis=0)
    diversity_text = (
        f"<b>Çeşitlilik (a).</b> Başlangıç değeri GA ve Binary PSO için ≈{div['GA'][0]:.2f} (rastgele başlangıç), ACO için "
        f"yalnızca ≈{div['ACO'][0]:.2f}: sezgisel bilgi karıncaları baştan benzer çözümlere yönlendirir (erken sömürü). "
        f"Son değerler: GA {div['GA'][-1]:.2f}, ACO {div['ACO'][-1]:.2f}, Binary PSO {div['BPSO'][-1]:.2f}. PSO'nun yüksek kalan "
        "çeşitliliği, hız sönümlenince <i>S</i>(<i>v</i>) → 0.5 olması ve bitlerin gürültüyle çevrilmesiyle uyumludur "
        "(Bölüm 5.3). "
        f"<b>ACO (b).</b> Feromon entropisi ≈{aco_lock}. iterasyonda sabitlenir; iz kilitlendikten sonra arama "
        "sezgisel bilgiye ve olasılıksal seçime dayanır. "
        f"<b>SA (c, d).</b> Sıcaklık geometrik olarak {temperature[0]:.0f} → {temperature[-1]:.3f} iner; kötü hamle kabul oranı "
        f"ilk 10 iterasyonda ortalama {early:.2f}, son 10 iterasyonda {late:.3f}: arama keşiften sömürüye geçer.")
    out.append(ctx.fig.beside_text(
        figure_path("fig11_diversity_diagnostics"),
        "Tanılama: (a) çeşitlilik (ort. Hamming/<i>n</i>), (b) ACO feromon entropisi, (c) SA sıcaklığı, "
        "(d) SA kötü-hamle kabul oranı.",
        [ctx.p(diversity_text, "body_left")], ctx.width, figure_fraction=0.5))
    return out


# --------------------------------------------------------------------------- #
# 5. Parametre duyarlılığı
# --------------------------------------------------------------------------- #
def section_sensitivity(ctx: Ctx) -> list[Flowable]:
    d = ctx.data
    sens = d.sensitivity.set_index("label")
    out: list[Flowable] = [ctx.h1("5. Parametre Duyarlılığı")]
    out.append(ctx.p(
        "Her algoritmada tek bir parametre değiştirilmiş, diğerleri ana konfigürasyonda sabit tutulmuştur; her düzey "
        f"ana deneyle <i>aynı</i> {len(config.SEEDS)} tohumla çalıştırılmıştır. Bu bölüm ana karşılaştırmadan "
        "<b>ayrıdır</b>; düzeyler Kruskal–Wallis ile sınanmıştır."))

    header = ["Algoritma / parametre", "Düzey", "Ortalama ± std", "Optimum bulma", "Süre (s)", "t<sub>99</sub> (iter.)"]
    rows = [header]
    param_names = {"GA": "<i>p<sub>m</sub></i>", "ACO": "<i>ρ</i>", "BPSO": "<i>w</i>", "SA": "<i>α</i>"}
    for algorithm in config.ALGORITHM_ORDER:
        group = d.sensitivity[d.sensitivity["algorithm"] == algorithm].sort_values("level_value")
        for i, r in enumerate(group.itertuples()):
            tag = " (ana)" if r.is_main_config else (" (ek)" if r.is_extension else "")
            t99 = "—" if r.reached_990_pct == 0 else (
                num(r.iters_to_990_median, 0) + ("" if r.reached_990_pct >= 100 else f" (%{r.reached_990_pct:.0f} ulaştı)"))
            rows.append([f"{NAMES[algorithm]} – {param_names[algorithm]}" if i == 0 else "", f"{r.level}{tag}",
                         f"{r.mean:.2f} ± {r.std:.2f}", f"{r.hit_optimum_pct:.0f}%", f"{r.time_mean_s:.3f}", t99])
    out.append(ctx.fig.table_title(
        "Parametre duyarlılığı (30 çalıştırma/düzey). t<sub>99</sub>: optimumun %99'una ulaşmak için medyan iterasyon; "
        "(ek): ödev dışı post-hoc düzey."))
    out.append(make_table(rows, [ctx.width * f for f in (0.22, 0.17, 0.22, 0.13, 0.1, 0.16)], ctx.styles, numeric_from_col=2))
    out.append(Spacer(1, 3))
    kw = d.sensitivity_kw
    t0 = config.main_parameters(int(d.dataset_overview["n_items"]))["SA"]["initial_temperature"]
    ga = {k: sens.loc[f"GA|{k}"] for k in ("1/n", "2/n", "5/n")}
    aco = {k: sens.loc[f"ACO|{k}"] for k in ("0.1", "0.3", "0.5")}
    pso = {k: sens.loc[f"BPSO|{k}"] for k in ("0.4", "0.7", "0.9", "1.0")}
    sa = {k: sens.loc[f"SA|{k}"] for k in ("0.90", "0.95", "0.99")}
    _check(ga["5/n"]["mean"] < ga["1/n"]["mean"] - 10 and ga["5/n"]["mean"] < ga["2/n"]["mean"] - 10, "GA 5/n belirgin kötü olmalı")
    _check(pso["0.4"]["mean"] < pso["0.7"]["mean"] < pso["0.9"]["mean"] < pso["1.0"]["mean"], "BPSO w ile monoton artmalı")
    _check(sa["0.99"]["mean"] < sa["0.90"]["mean"] < sa["0.95"]["mean"] + 3, "SA 0.99 en kötü olmalı")
    _check(aco["0.1"]["mean"] >= aco["0.5"]["mean"], "ACO ρ=0.1 en az ρ=0.5 kadar iyi olmalı")

    out.append(ctx.h2("5.1 GA – mutasyon olasılığı"))
    out.append(ctx.p(
        f"<i>p<sub>m</sub></i> = 1/<i>n</i> ve 2/<i>n</i> benzerdir ({ga['1/n']['mean']:.2f}; {ga['2/n']['mean']:.2f}); "
        f"5/<i>n</i> belirgin kötüdür ({ga['5/n']['mean']:.1f} ± {ga['5/n']['std']:.1f}; optimum hiç bulunmaz; "
        f"Kruskal–Wallis <i>p</i> {pval(kw['GA'])}). Beklenen değişen gen sayısı <i>n·p<sub>m</sub></i> = 1, 2, 5'tir: "
        "5/<i>n</i>'de her çocuk elit çözümün ≈5 genini bozar, bu da son aşamadaki ince ayarı engeller, "
        f"değişkenliği artırır (std {ga['1/n']['std']:.1f} → {ga['5/n']['std']:.1f}) ve yakınsamayı yavaşlatır "
        f"(%99 eşiği medyan {num(ga['5/n']['iters_to_990_median'], 0)} iterasyon, yalnızca %{ga['5/n']['reached_990_pct']:.0f} "
        f"çalıştırmada; 1/<i>n</i>'de {num(ga['1/n']['iters_to_990_median'], 0)}). Düşük mutasyon sömürüyü, yüksek "
        "mutasyon keşfi artırır; bu problemde orta-düşük mutasyon yeterlidir."))
    out.append(ctx.h2("5.2 ACO – buharlaşma oranı"))
    out.append(ctx.p(
        f"ACO tüm ρ değerlerinde optimuma çok yakındır ({aco['0.1']['mean']:.2f}; {aco['0.3']['mean']:.2f}; "
        f"{aco['0.5']['mean']:.2f}; Kruskal–Wallis <i>p</i> {pval(kw['ACO'])}). ρ = 0.5'te çalıştırmaların "
        f"yaklaşık %{100 - aco['0.5']['hit_optimum_pct']:.0f} kadarı optimumun altında kalır (en kötü {aco['0.5']['worst']:.0f}): hızlı "
        "buharlaşma eski bilgiyi çabuk siler, feromon izi en iyi çözüme erken kilitlenir (azalan keşif); ρ = "
        "0.1–0.3'te yavaş unutma daha uzun keşif sağlar. Sezgisel bilgi (η) güçlü olduğundan feromonun payı "
        "sınırlıdır ve duyarlılık küçüktür."))
    out.append(ctx.h2("5.3 Binary PSO – atalet katsayısı"))
    out.append(ctx.p(
        f"En duyarlı yöntemdir: <i>w</i> = 0.4, 0.7, 0.9 için ortalama {pso['0.4']['mean']:.0f}, {pso['0.7']['mean']:.0f}, "
        f"{pso['0.9']['mean']:.0f} (Kruskal–Wallis <i>p</i> {pval(kw['BPSO'])}). Sürekli PSO sezgisinin tersine "
        "<b>büyük atalet daha iyidir</b>: küçük <i>w</i> hızı hızla sıfıra sönümler, <i>v</i> ≈ 0 iken "
        "<i>S</i>(<i>v</i>) ≈ 0.5 olduğundan bitler neredeyse yazı-tura ile belirlenir (yüksek gürültü, düşük "
        f"sömürü; bkz. Bölüm 4.4). Optimum ızgara sınırında görüldüğü için tek bir <b>post-hoc</b> düzey "
        f"eklenmiştir: <i>w</i> = 1.0 (özgün Kennedy–Eberhart ikili PSO'su {cite('kennedy97')}) "
        f"{pso['1.0']['mean']:.2f} ± {pso['1.0']['std']:.2f} verir ve GA ile aynı düzeye ({d.main('GA', 'mean'):.2f}) "
        "çıkar. Yani ana deneydeki zayıflık algoritmanın değil, ödevdeki <i>w</i> aralığının sonucudur."))
    out.append(ctx.h2("5.4 SA – soğutma katsayısı"))
    out.append(ctx.p(
        f"α = 0.95 en iyi ({sa['0.95']['mean']:.2f}), 0.90 yakın ({sa['0.90']['mean']:.2f}), 0.99 belirgin kötüdür "
        f"({sa['0.99']['mean']:.1f}; Kruskal–Wallis <i>p</i> {pval(kw['SA'])}). 200 iterasyonda son sıcaklık "
        f"<i>T</i><sub>0</sub>·α<super>200</super> = {_sci(t0 * 0.90 ** 200)}, {_sci(t0 * 0.95 ** 200)} ve {t0 * 0.99 ** 200:.1f} "
        "olur: α = 0.99'da arama bütçe boyunca sıcak kalır, kötü hamleler kabul edilmeye devam eder ve yerel "
        "iyileştirmeye zaman kalmaz (%99 eşiğine hiç ulaşılamaz). α = 0.90'da sıcaklık çok hızlı düşer: arama "
        f"daha erken yakınsar (%99 eşiği medyan {num(sa['0.90']['iters_to_990_median'], 1)} iterasyon; α = 0.95'te "
        f"{num(sa['0.95']['iters_to_990_median'], 1)}) ama keşif süresi kısaldığı için nihai kalite biraz düşer. Soğutma "
        "hızı bütçeye uygun seçilmelidir."))
    out.append(ctx.fig.figure(
        figure_path("fig09_sensitivity_boxplots"), ctx.width,
        "Parametre düzeylerine göre nihai değer dağılımı (koyuluk = parametre değeri; taralı = post-hoc ek düzey). "
        "Düzeylere göre yakınsama eğrileri: results/figures/fig10_sensitivity_convergence."))

    return out
