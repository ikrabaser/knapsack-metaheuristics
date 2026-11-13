"""Rapor Bölüm 6: bonus (GA-SA hibriti) ve ek deneyler (eşit bütçe, kısıt/bilgi ablasyonu, kapasite oranları).

Bu bölüm ana deney sonuçlarından **ayrıdır** (``bonus_*`` ve ``extra_*`` kayıtları).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from reportlab.platypus import Flowable, Spacer

import config
from reports.pdf_helpers import make_table
from reports.report_data import figure_path, num, pval
from reports.sections_results import NAMES, _check
from reports.sections_static import Ctx


def section_extras(ctx: Ctx) -> list[Flowable]:
    out: list[Flowable] = [ctx.h1("6. Bonus Çalışma ve Ek Deneyler (ana deneyden ayrı)")]
    out += _hybrid(ctx)
    out += _equal_budget(ctx)
    out += _knowledge_ablation(ctx)
    out += _capacity(ctx)
    return out


# --------------------------------------------------------------------------- #
def _hybrid(ctx: Ctx) -> list[Flowable]:
    d = ctx.data
    ga, sa = d.summary_main.loc["GA"], d.summary_main.loc["SA"]
    hy = d.summary_bonus.loc["GA-SA"]
    share = d.summary_hybrid_share
    pb = d.pairwise_bonus

    out: list[Flowable] = [ctx.h2("6.1 GA-SA hibriti")]
    def row(first: str, second: str):
        r = pb[(pb["a"] == first) & (pb["b"] == second)]
        return r.iloc[0]

    gh, sh, gs = row("GA", "GA-SA"), row("SA", "GA-SA"), row("GA", "SA")
    shares = list(config.HYBRID_SA_SHARES)
    share_means = [share.loc[f"GA-SA|sa={s}", "mean"] for s in shares]
    _check(max(share_means[:4]) - min(share_means[:4]) < 1.5, "SA payı 0–20 arasında düz olmalı")
    _check(share_means[-1] < min(share_means[:4]), "SA payı 30 en kötü olmalı")
    out.append(ctx.p(
        f"<b>Sonuç.</b> Hibrit ortalama {num(hy['mean'], 2)} ± {num(hy['std'], 2)} (en kötü {num(hy['worst'], 0)}; optimum "
        f"%{hy['hit_optimum_pct']:.1f}; {hy['time_mean_s']:.3f} s; {hy['evals_mean']:.0f} FE) bulmuştur (GA: {num(ga['mean'], 2)}, "
        f"%{ga['hit_optimum_pct']:.1f}; SA: {num(sa['mean'], 2)}, %{sa['hit_optimum_pct']:.1f}). SA'dan ortalamada {num(-sh['mean_diff'], 2)} "
        f"birim iyidir, ancak fark Holm sonrası anlamlı değildir (Wilcoxon <i>p</i><sub>Holm</sub> = "
        f"{pval(sh['wilcoxon_p_holm'])}, Mann–Whitney {pval(sh['mannwhitney_p_holm'])}); GA'dan ise <b>ayırt edilemez</b> "
        f"(GA {num(gh['mean_diff'], 2)} birim önde; <i>p</i><sub>Holm</sub> = {pval(gh['wilcoxon_p_holm'])}, "
        f"{pval(gh['mannwhitney_p_holm'])}). GA–SA farkı ise anlamlıdır ({pval(gs['wilcoxon_p_holm'])}). "
        "<b>Neden?</b> SA payı ablasyonunda (nesil başına 0, 6, 12, 20 SA değerlendirmesi) ortalamalar "
        f"{', '.join(num(m, 2) for m in share_means[:4])} ile düzdür (0 = saf GA); yalnızca 30 değerlendirme "
        f"belirgin kötüdür ({num(share_means[-1], 2)}), çünkü GA çocuk sayısı azalır (grafik: "
        "<font name='DV-Mono' size='7'>fig13_hybrid_share</font>). GA bu örnekte optimuma zaten ~2 birim yakındır; "
        "kalan fark, kısa SA zincirlerinin (6–20 komşu) aşamadığı çok-ürünlü değişimler gerektirir. Dürüst yorum: "
        "<i>hibritin değeri bu kolay örnekte kanıtlanamamıştır</i>."))
    out.append(ctx.fig.figure(
        figure_path("fig12_bonus_hybrid"), ctx.width,
        "Hibrit, GA ve SA: (a) ortalama yakınsama (optimalite farkı), (b) nihai değer dağılımı."))
    return out


# --------------------------------------------------------------------------- #
def _equal_budget(ctx: Ctx) -> list[Flowable]:
    d = ctx.data
    equal = d.summary_equal_fe
    main = d.summary_main
    out: list[Flowable] = [ctx.h2("6.2 Eşit değerlendirme bütçesi denetimi")]
    rows = [["Algoritma", "Ana deney FE", "Ana deney ortalama", f"Eşit bütçe ({config.EQUAL_BUDGET_EVALUATIONS} FE) ortalama",
             "Fark", "Optimum bulma (ana → eşit)"]]
    worst_shift = 0.0
    for name in (*config.ALGORITHM_ORDER, config.BONUS_ALGORITHM):
        base = d.summary_bonus.loc[name] if name == config.BONUS_ALGORITHM else main.loc[name]
        eq = equal.loc[name]
        shift = eq["mean"] - base["mean"]
        worst_shift = max(worst_shift, abs(shift))
        rows.append([NAMES[name], f"{base['evals_mean']:.0f}", num(base["mean"], 2), num(eq["mean"], 2), f"{shift:+.2f}",
                     f"%{base['hit_optimum_pct']:.0f} → %{eq['hit_optimum_pct']:.0f}"])
    _check(worst_shift < 1.0, "eşit bütçe sonuçları ana sonuçlardan <1 birim sapmalı")
    out.append(ctx.fig.table_title(
        "Tüm algoritmalar tam eşit bütçeyle (sert FE üst sınırı; aynı tohum ve parametreler) çalıştırıldığında."))
    out.append(make_table(rows, [ctx.width * f for f in (0.16, 0.12, 0.17, 0.25, 0.1, 0.2)], ctx.styles))
    out.append(Spacer(1, 3))
    out.append(ctx.p(
        f"Bütçeler tam eşitlendiğinde hiçbir ortalama {worst_shift:.2f} birimden fazla değişmemiş ve sıralama "
        "korunmuştur: algoritmalar ilk ~60–120 iterasyonda yakınsadığı için fazladan ≈400 değerlendirme etki "
        "etmez; ana deneydeki ≈%5'lik bütçe farkı sonuçları <i>çarpıtmamaktadır</i>."))
    return out


# --------------------------------------------------------------------------- #
def _knowledge_ablation(ctx: Ctx) -> list[Flowable]:
    d = ctx.data
    repair, beta, main = d.repair, d.summary_aco_beta, d.summary_main
    out: list[Flowable] = [ctx.h2("6.3 Kısıt yönetimi ve sezgisel bilgi ablasyonu")]
    strategies = [("ratio", "oran-çıkar (ana)"), ("random", "rastgele-çıkar"), ("ratio_fill", "oran-çıkar + doldur")]
    rows = [["Onarım stratejisi", "GA", "Binary PSO", "SA"]]
    for key, name in strategies:
        cells = [name]
        for algorithm in ("GA", "BPSO", "SA"):
            r = repair[algorithm].loc[f"{algorithm}|{key}"]
            cells.append(f"{r['mean']:.2f} ± {r['std']:.2f} (%{r['hit_optimum_pct']:.0f})")
        rows.append(cells)
    for b in config.ACO_BETA_LEVELS:
        r = beta.loc[f"ACO|beta={b:g}"]
        tag = " (ana; yapısal uygun + maksimal inşa)" if b == config.ACO_BETA_LEVELS[-1] else ""
        rows.append([f"ACO, <i>β</i> = {b:g}{tag}", f"{r['mean']:.2f} ± {r['std']:.2f} (%{r['hit_optimum_pct']:.0f})", "—", "—"])
    out.append(ctx.fig.table_title(
        "Onarım stratejisi ve ACO sezgisel bilgi ablasyonu: ortalama ± std (optimumu bulma oranı), 30 çalıştırma "
        "(ACO satırları ilk sütundadır)."))
    out.append(make_table(rows, [ctx.width * f for f in (0.34, 0.22, 0.22, 0.22)], ctx.styles))
    out.append(Spacer(1, 3))
    fill_cost = float(np.mean([repair[a].loc[f"{a}|ratio_fill", "time_mean_s"] / repair[a].loc[f"{a}|ratio", "time_mean_s"]
                               for a in ("GA", "BPSO", "SA")]))
    gains = {a: repair[a].loc[f"{a}|ratio_fill", "mean"] - repair[a].loc[f"{a}|ratio", "mean"] for a in ("GA", "BPSO", "SA")}
    losses = {a: repair[a].loc[f"{a}|ratio", "mean"] - repair[a].loc[f"{a}|random", "mean"] for a in ("GA", "BPSO", "SA")}
    _check(all(v > 0 for v in gains.values()) and all(v > 0 for v in losses.values()),
           "oran-çıkar, rastgele-çıkardan; doldurma ise oran-çıkardan iyi olmalı")
    out.append(ctx.p(
        "<b>Onarım algoritmalara avantaj sağlar mı? Evet, hem de güçlü biçimde.</b> Rastgele-çıkarandan "
        f"oran-çıkarana geçiş ortalamayı GA'da {losses['GA']:+.1f}, Binary PSO'da {losses['BPSO']:+.0f}, SA'da "
        f"{losses['SA']:+.1f} birim değiştirir; boş kapasiteyi oran sırasıyla dolduran onarım ek olarak "
        f"{gains['GA']:+.1f}, {gains['BPSO']:+.0f} ve {gains['SA']:+.1f} birim kazandırır ve SA ile Binary PSO'yu ACO "
        "düzeyine (≈4560) çıkarır. Yani ana deneydeki sıralama kısıt yönetiminin bir <i>fonksiyonudur</i>: onarım "
        "(ve ACO'nun maksimal inşası) değer/ağırlık bilgisini ve 'kapasiteyi doldur' kuralını arama sürecine "
        "zorlar, arama uzayı fiilen açgözlü çözümün çevresine daralır. ACO'da sezgisel bilgi kapatılınca "
        f"(<i>β</i> = 0) ortalama {num(beta.loc['ACO|beta=0', 'mean'], 2)}'e düşer; dolayısıyla ACO üstünlüğü "
        f"<i>kısmen bilgi avantajıdır</i>. Doldurmalı onarım pahalıdır (süre ≈{fill_cost:.0f}× artar; Python döngüsü)."))
    out.append(ctx.fig.figure(
        figure_path("fig15_knowledge_ablation"), ctx.width,
        "Bilgi enjeksiyonu ablasyonu: (a–c) onarım stratejisi (oran = oran-çıkar [ana], rastgele = rastgele-çıkar, oran+doldur = "
        "oran-çıkar + doldur); (d) ACO sezgisel bilgi üssü β."))
    return out


# --------------------------------------------------------------------------- #
def _capacity(ctx: Ctx) -> list[Flowable]:
    d = ctx.data
    cap = d.capacity
    out: list[Flowable] = [ctx.h2("6.4 Kapasite oranı analizi")]
    ratios = sorted(cap["capacity_ratio"].unique())
    algorithms = [*config.ALGORITHM_ORDER, config.BONUS_ALGORITHM]
    rows = [["Algoritma"] + [f"%{r * 100:.0f}" for r in ratios]]
    for a in algorithms:
        cells = [NAMES[a]]
        for r in ratios:
            row = cap[(cap["label"] == a) & (np.isclose(cap["capacity_ratio"], r))].iloc[0]
            cells.append(f"{row['mean_gap_pct']:.3f} (%{row['hit_optimum_pct']:.0f})")
        rows.append(cells)
    out.append(ctx.fig.table_title(
        "Kapasite oranına göre ortalama optimalite farkı % (optimumu bulma oranı); her oranda DP ile gerçek optimum, "
        "30 çalıştırma."))
    out.append(make_table(rows, [ctx.width * 0.2] + [ctx.width * 0.2] * len(ratios), ctx.styles))
    out.append(Spacer(1, 3))
    return out + capacity_narrative(ctx)


def capacity_narrative(ctx: Ctx) -> list[Flowable]:
    """Kapasite oranı bulgularının yorumu (rakamlar tablolardan; iddialar denetlenir)."""
    d = ctx.data
    cap = d.capacity
    ratios = sorted(cap["capacity_ratio"].unique())
    gap = cap.pivot(index="label", columns="capacity_ratio", values="mean_gap_pct")
    hit = cap.pivot(index="label", columns="capacity_ratio", values="hit_optimum_pct")
    low, mid, high = ratios[0], ratios[1], ratios[-1]

    _check(all(gap[r].idxmin() == "ACO" for r in ratios), "ACO her kapasite oranında en düşük ortalama farka sahip olmalı")
    _check(gap[low].idxmax() == "SA", "%30'da en kötü SA olmalı")
    _check(all(gap[r].idxmax() == "BPSO" for r in ratios[1:]), "BPSO %50 ve üstünde en kötü olmalı")
    _check(all(gap.loc["BPSO", a] < gap.loc["BPSO", b] for a, b in zip(ratios, ratios[1:])), "BPSO farkı kapasiteyle artmalı")
    _check(hit.loc["ACO", low] < 100 and hit.loc["ACO", mid] < 100, "ACO düşük kapasitede %100 olmamalı")
    _check(hit.loc["GA", low] > hit.loc["ACO", low], "GA %30'da ACO'dan yüksek isabet oranına sahip olmalı")
    _check(gap.loc["GA-SA", low] > gap.loc["GA", low] and gap.loc["GA-SA", mid] > gap.loc["GA", mid],
           "hibrit düşük kapasitelerde GA'dan kötü olmalı")

    def p_holm(ratio: float, first: str, second: str) -> float:
        table = pd.read_csv(config.TABLES_DIR / f"pairwise_capacity_{int(round(ratio * 100))}.csv")
        row = table[((table["a"] == first) & (table["b"] == second)) | ((table["a"] == second) & (table["b"] == first))]
        return float(row["wilcoxon_p_holm"].iloc[0])

    p_low, p_mid = p_holm(low, "GA", "GA-SA"), p_holm(mid, "GA", "GA-SA")
    aco_gaps = ", ".join("%{:.0f}: {:.3f}".format(r * 100, gap.loc["ACO", r]) for r in ratios)
    out: list[Flowable] = [ctx.p(
        f"ACO her oranda en düşük ortalama farka sahiptir ({aco_gaps}), ancak <i>her çalıştırmada</i> optimumu "
        f"bulma özelliği yalnızca %{ratios[2] * 100:.0f} ve %{high * 100:.0f}'a <i>özgüdür</i>: %{low * 100:.0f} ve "
        f"%{mid * 100:.0f}'de isabet oranı %{hit.loc['ACO', low]:.0f} ve %{hit.loc['ACO', mid]:.0f}'tür; %{low * 100:.0f}'da "
        f"GA'nın isabet oranı (%{hit.loc['GA', low]:.0f}) daha yüksektir. Binary PSO %{mid * 100:.0f} ve üstünde en "
        f"kötüdür ve farkı kapasiteyle artar ({gap.loc['BPSO', low]:.2f} → {gap.loc['BPSO', high]:.2f}); "
        f"%{low * 100:.0f}'de en kötü yöntem SA'dır ({gap.loc['SA', low]:.2f}). <b>Hibrit</b> düşük kapasitelerde "
        f"GA'dan daha kötüdür (%{low * 100:.0f}: {gap.loc['GA-SA', low]:.3f} vs {gap.loc['GA', low]:.3f}, <i>p</i><sub>Holm</sub> = "
        f"{pval(p_low)}; %{mid * 100:.0f}: {gap.loc['GA-SA', mid]:.3f} vs {gap.loc['GA', mid]:.3f}, <i>p</i><sub>Holm</sub> = "
        f"{pval(p_mid)}) ve yüksek kapasitelerde GA ile benzerdir; SA aşaması hiçbir kapasitede net katkı "
        "sağlamamıştır. Sıralamanın kapasiteye bağlı olması, tek bir orandan çıkan sonuçların genellenemeyeceğini "
        "gösterir.")]
    out.append(ctx.fig.figure(
        figure_path("fig14_capacity_ratios"), ctx.width,
        "Kapasite oranına göre (a) ortalama optimalite farkı (simetrik-log ölçek) ve (b) optimumu bulan çalıştırma oranı."))
    return out
