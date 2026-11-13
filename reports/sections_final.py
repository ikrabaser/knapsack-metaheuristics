"""Raporun özet, tartışma, sonuç ve kaynakça bölümleri."""

from __future__ import annotations

import numpy as np
import pandas as pd
from reportlab.platypus import Flowable, Paragraph

import config
from experiments.runner import load_experiment
from reports.report_data import num, references
from reports.sections_results import NAMES, _check
from reports.sections_static import Ctx


def n_runs(summary: pd.DataFrame) -> int:
    """Çalıştırma sayısı."""
    return int(summary["n_runs"].iloc[0])


def abstract_and_keywords(ctx: Ctx) -> tuple[str, str]:
    """Özet metni ve anahtar kelimeler (sayılar gerçek sonuç tablolarından)."""
    d = ctx.data
    s, ref = d.summary_main, d.reference
    n = int(d.dataset_overview["n_items"])
    bonus = d.summary_bonus.loc["GA-SA"]
    repair = d.repair
    text = (
        f"Bu çalışmada GA, ACO, ikili PSO ve SA; {n} ürünlü ve kapasitesi toplam ağırlığın %70'i "
        f"({num(d.dataset_overview['capacity'], 1)}) olan bir 0/1 Knapsack örneğinde, aynı veri, tohum ve "
        f"karşılaştırılabilir değerlendirme bütçesiyle ({config.ITERATIONS} iterasyon, {config.POPULATION_SIZE} birey/"
        f"karınca/komşu) {len(config.SEEDS)} bağımsız çalıştırmayla karşılaştırılmıştır. Gerçek optimum "
        f"({num(ref['optimum_value_dp'], 0)}) dinamik programlama ile hesaplanmış ve MILP ile doğrulanmıştır. "
        f"ACO tüm çalıştırmalarda optimumu bulmuş; GA ortalama {num(s.loc['GA', 'mean'], 2)}, SA {num(s.loc['SA', 'mean'], 2)}, "
        f"Binary PSO {num(s.loc['BPSO', 'mean'], 1)} değerine ulaşmıştır; GA ve Binary PSO en hızlı, ACO en yavaş yöntemdir. "
        "Parametre duyarlılığı en çok Binary PSO'da (atalet) görülmüştür. Önemli bir bulgu olarak, kısıt "
        "yönetiminin (onarım) algoritma seçiminden daha belirleyici olduğu saptanmıştır: oran-sıralı onarım + "
        f"doldurma ile SA ortalaması {num(repair['SA'].loc['SA|ratio_fill', 'mean'], 2)}, Binary PSO "
        f"{num(repair['BPSO'].loc['BPSO|ratio_fill', 'mean'], 2)} olmakta, ACO düzeyine gelmektedir. Bonus GA-SA "
        f"hibriti (ortalama {num(bonus['mean'], 2)}) bu kolay örnekte GA'dan istatistiksel olarak ayırt "
        "edilememiştir. Sonuçlar tek bir örneğe aittir ve genellenmemelidir."
    )
    keywords = ("meta-sezgisel optimizasyon, 0/1 Knapsack, genetik algoritma, karınca kolonisi, parçacık sürüsü, "
                "simulated annealing, hibrit algoritma, istatistiksel karşılaştırma, kısıt yönetimi")
    return text, keywords


def section_discussion(ctx: Ctx) -> list[Flowable]:
    d = ctx.data
    s = d.summary_main
    ov = d.dataset_overview
    pw = d.pairwise_main
    ga, aco, sa, pso = (s.loc[k] for k in ("GA", "ACO", "SA", "BPSO"))
    repair = d.repair
    beta = d.summary_aco_beta
    out: list[Flowable] = [ctx.h1("7. Tartışma ve Bilimsel Yorum")]

    # S1 --------------------------------------------------------------------
    _check(beta.loc["ACO|beta=0", "mean"] < beta.loc["ACO|beta=2", "mean"] - 3, "ACO beta=0 belirgin kötü olmalı")
    _check(repair["SA"].loc["SA|ratio_fill", "mean"] > repair["SA"].loc["SA|ratio", "mean"] + 3, "SA ratio_fill belirgin iyi olmalı")
    _check(repair["BPSO"].loc["BPSO|ratio_fill", "mean"] > repair["BPSO"].loc["BPSO|ratio", "mean"] + 100, "BPSO ratio_fill büyük iyileşmeli")
    gap_aco_ga = abs(pw[(pw["a"] == "GA") & (pw["b"] == "ACO")]["mean_diff"].iloc[0])
    out.append(ctx.h2("S1. Hangi algoritma en iyi çözümü üretti ve neden?"))
    out.append(ctx.p(
        f"<b>ACO:</b> gerçek optimumu ({num(aco['best'], 0)}) {n_runs(s)} çalıştırmanın tamamında bulmuş (ortalama "
        f"{num(aco['mean'], 2)}, std = 0). <i>En yüksek tekil sonuç</i> GA, ACO ve SA için aynıdır "
        f"({num(ga['best'], 0)}); bu yüzden sıralama ortalama, dağılım ve optimum bulma oranına dayanır (ACO %100 &gt; "
        f"GA %{ga['hit_optimum_pct']:.0f} &gt; SA %{sa['hit_optimum_pct']:.1f} &gt; Binary PSO %0) ve tüm ikili "
        f"karşılaştırmalarda anlamlıdır (Tablo 4); ACO–GA farkı pratikte yalnızca {num(gap_aco_ga, 1)} birimdir."))
    out.append(ctx.p(
        "<b>Neden?</b> (i) <i>Problem yapısı:</i> ürünler bağımsız seçilir ve iyi çözümler yüksek değer/ağırlık "
        "oranlı ürünlerden oluşur; bu sıralamayı doğrudan kullanan yöntem avantajlıdır. ACO oranı sezgisel bilgi "
        "olarak kullanır ve çözümü <i>maksimal</i> inşa eder; sezgisel bilgi kapatılınca (<i>β</i> = 0) ortalaması "
        f"{num(beta.loc['ACO|beta=0', 'mean'], 2)}'e (%{beta.loc['ACO|beta=0', 'hit_optimum_pct']:.0f} optimum) düşer. "
        "(ii) <i>Kısıt yönetimi:</i> GA, PSO ve SA'daki oran-çıkaran onarım maksimal çözüm üretmez; doldurmalı "
        f"onarımla SA {num(repair['SA'].loc['SA|ratio', 'mean'], 2)} → {num(repair['SA'].loc['SA|ratio_fill', 'mean'], 2)}, "
        f"Binary PSO {num(repair['BPSO'].loc['BPSO|ratio', 'mean'], 1)} → {num(repair['BPSO'].loc['BPSO|ratio_fill', 'mean'], 2)}, "
        f"GA {num(repair['GA'].loc['GA|ratio', 'mean'], 2)} → {num(repair['GA'].loc['GA|ratio_fill', 'mean'], 2)} olur (Bölüm 6.3). "
        "Yani ana deneydeki ACO üstünlüğü kısmen <i>algoritmik değil bilgi</i> avantajıdır. (iii) <i>Parametre:</i> "
        f"Binary PSO'nun düşük sonucu büyük ölçüde atalet seçimindendir (<i>w</i> = 1.0 ile {num(d.sens('BPSO|1.0', 'mean'), 2)})."))

    # S2 --------------------------------------------------------------------
    out.append(ctx.h2("S2. Hangi algoritma daha hızlıdır?"))
    dp_ms = float(ov["dp_median_time_ms"])
    t_ga, t_aco, t_sa, t_pso = (s.loc[k, "time_mean_s"] for k in ("GA", "ACO", "SA", "BPSO"))
    _check(set(s["time_mean_s"].nsmallest(2).index) == {"GA", "BPSO"}, "en hızlı iki yöntem GA ve Binary PSO olmalı")
    _check(abs(t_pso / t_ga - 1) < 0.25, "GA ve Binary PSO süreleri birbirine yakın olmalı")
    iters = s.loc[["GA", "ACO"], "iters_to_990_median"]
    approx_ga = t_ga * iters["GA"] / config.ITERATIONS
    approx_aco = t_aco * iters["ACO"] / config.ITERATIONS
    out.append(ctx.p(
        f"Süre sırası GA {num(t_ga, 3)} s ≈ Binary PSO {num(t_pso, 3)} s (fark %{abs(t_pso / t_ga - 1) * 100:.0f}) &lt; SA "
        f"{num(t_sa, 3)} s &lt; ACO {num(t_aco, 2)} s'dir; GA ile PSO pratikte aynı hızdadır. "
        "FE sayıları yakın olduğundan (7640–8040) fark <i>FE başına maliyetten</i> gelir: GA ve PSO popülasyonu "
        "toplu (vektörize) değerlendirir; SA'nın 8000 komşusu ardışıktır; ACO'nun inşası ürün ürün ilerler. "
        f"Yakınsama hızı sıralamayı değiştirir: ACO %99 kaliteye {num(iters['ACO'], 0)}, GA {num(iters['GA'], 0)} "
        f"iterasyonda ulaşır; iterasyon başı sürenin sabit olduğu varsayımıyla bu ≈{approx_aco:.3f} s (ACO) ve "
        f"≈{approx_ga:.3f} s (GA) eder: <i>%99 kaliteye ulaşma süresi</i> aynı mertebededir, ACO'nun ek süresi son %1'i "
        f"sıfırlamak içindir. Referans: DP bu örneği <b>{dp_ms:.1f} ms</b>'de kesin çözer (bkz. S5)."))

    # S3 --------------------------------------------------------------------
    out.append(ctx.h2("S3. Hangi algoritma daha kararlıdır?"))
    out.append(ctx.p(
        f"ACO (std = 0; 30/30 optimum) &gt; GA ({num(ga['std'], 2)}) &gt; SA ({num(sa['std'], 2)}) &gt; Binary PSO "
        f"({num(pso['std'], 2)}). Dağılım biçimleri farklıdır: GA optimuma yapışık ve sola çarpık (en kötü "
        f"{num(ga['worst'], 0)}), SA uzun alt kuyruklu ({num(sa['worst'], 0)}), Binary PSO geniş ve simetriktir. "
        f"Kararlılık parametreye de bağlıdır: GA'da 5/<i>n</i> ile std {num(d.sens('GA|5/n', 'std'), 1)}, SA'da α = 0.99 "
        f"ile {num(d.sens('SA|0.99', 'std'), 1)}, Binary PSO'da <i>w</i> = 0.4 ile {num(d.sens('BPSO|0.4', 'std'), 1)}."))

    # S4 --------------------------------------------------------------------
    out.append(ctx.h2("S4. Algoritmalar arasındaki fark neden oluştu?"))
    main_exp = load_experiment("main")
    div = {a: main_exp.diversity[a].mean(axis=0) for a in ("GA", "ACO", "BPSO")}
    entropy = main_exp.extras["ACO"]["mean_pheromone_entropy"].mean(axis=0)
    worse_rate = np.nanmean(main_exp.extras["SA"]["worse_accept_rate"], axis=0)
    ga_collapse = int(np.argmax(div["GA"] < 0.05)) + 1
    aco_lock = int(np.argmax(np.abs(entropy - entropy[-1]) < 0.002)) + 1
    sa_zero = int(np.argmax(worse_rate < 0.005)) + 1
    _check(div["GA"][-1] < 0.05 and div["BPSO"][-1] > 0.15, "GA çeşitliliği çökmeli, BPSO yüksek kalmalı")
    out.append(ctx.p(
        f"<b>Keşif–sömürü ve çeşitlilik.</b> GA'nın çeşitliliği ilk {ga_collapse} nesilde 0.05'in altına çöker "
        f"({div['GA'][0]:.2f} → {div['GA'][-1]:.2f}): elitizm ve turnuva baskısı popülasyonu optimuma yakın bir kümeye "
        "yığar, sonrasında yalnızca mutasyon (1/<i>n</i>) keşif yapar; bu, son 1–2 birimin nadiren kapanmasının "
        f"nedenidir (erken yakınsama). ACO'da sezgisel bilgi çeşitliliği baştan düşük tutar ({div['ACO'][0]:.2f}) ve "
        f"feromon entropisi ≈{aco_lock}. iterasyonda sabitlenir; yine de optimumu bulur, çünkü olasılıksal seçim sınır "
        f"ürünlerde yeterli çeşitlilik sağlar. SA'nın kötü hamle kabulü ≈{sa_zero}. iterasyonda sıfıra iner, kalan "
        f"≈{config.ITERATIONS - sa_zero} iterasyon saf yerel aramadır (çizelge bütçeye göre fazla hızlı; α = 0.99 aşırı "
        f"yavaş). Binary PSO'da hız sönümlendikçe bit gürültüsü sürer (kalan çeşitlilik {div['BPSO'][-1]:.2f}). "
        "<b>Ayrık yapı:</b> amaç fonksiyonu, tamsayı ağırlıkların kapasiteye 'yaslandığı' platolara sahiptir; "
        "iyi çözümler arası geçiş bir ürünü çıkarıp başkasını eklemeyi (swap) gerektirir. Tek bit çeviren "
        "operatörler kapasiteyi ya aşar (onarım geri alır) ya da boş bırakır; bu yüzden swap komşuluğu, maksimal "
        "inşa ve doldurmalı onarım avantaj sağlar. "
        "Sürekli PSO'daki 'küçük atalet = daha fazla keşif' sezgisinin ikili PSO'da tersine dönmesi de bu ayrık yapıdandır."))

    # S5 --------------------------------------------------------------------
    out.append(ctx.h2("S5. Bu problem için hangi algoritma tercih edilmelidir?"))
    out.extend(ctx.bullets([
        f"<b>Güvenilirlik öncelikliyse → ACO:</b> 30/30 optimum, std = 0; {num(t_aco, 1)} s mutlak olarak küçüktür. "
        "Bedel: iterasyon başı pahalı inşa.",
        f"<b>Hız/kalite dengesi öncelikliyse → GA:</b> {num(t_ga, 2)} s, optimuma %{ga['mean_gap_pct']:.2f} yakın, "
        f"en basit ayar; güvenilirlik daha düşük (%{ga['hit_optimum_pct']:.0f}).",
        "<b>Kaçınılacak:</b> ödevdeki aralıkta (<i>w</i> ≤ 0.9) Binary PSO ve α = 0.99 SA; PSO ancak <i>w</i> = 1.0 ile rekabetçidir.",
        f"<b>Pratik not:</b> ağırlıklar tamsayı ve <i>n</i> = {int(ov['n_items'])} olduğundan DP bu örneği "
        f"{float(ov['dp_median_time_ms']):.1f} ms'de <i>kesin</i> çözer; bu veri için tercih edilecek yöntem DP'dir. "
        "Meta-sezgisellerin değeri, DP'nin uygulanamadığı varyantlarda (çok kısıtlı, doğrusal olmayan, çok büyük "
        "kapasiteli) ortaya çıkar; bu bulgular o varyantlara <i>doğrudan</i> genellenemez.",
        "<b>Algoritmadan önce kısıt yönetimine yatırım yapın:</b> doldurmalı onarım SA ve Binary PSO'yu ACO "
        "düzeyine çıkarmıştır.",
    ]))

    # Sınırlılıklar ----------------------------------------------------------
    first_pass = pd.read_csv(config.RAW_DIR / "timing_first_pass.csv").groupby("label")["elapsed_s"].mean()
    final_times = d.summary_main["time_mean_s"].copy()
    final_times["GA-SA"] = d.summary_bonus.loc["GA-SA", "time_mean_s"]
    slowdown = (first_pass / final_times).dropna()
    aco_ga_first, aco_ga_final = first_pass["ACO"] / first_pass["GA"], final_times["ACO"] / final_times["GA"]
    corr = float(ov["corr_weight_value_pearson"])
    _check(abs(corr) < 0.3, "ağırlık–değer ilişkisi zayıf olmalı")
    pilot_row = d.pilot[(d.pilot["algorithm"] == "GA-SA") & d.pilot["selected"]].iloc[0]
    hybrid_mean = float(d.summary_bonus.loc["GA-SA", "mean"])
    out.append(ctx.h2("Sınırlılıklar"))
    out.extend(ctx.bullets([
        f"<b>Tek, kolay örnek:</b> {int(ov['n_items'])} ürün, zayıf ağırlık–değer ilişkisi "
        f"(<i>r</i> = {num(corr, 2)}); açgözlü sezgisel bile %{d.reference['greedy_gap_pct']:.2f} yakındır. "
        "Güçlü ilişkili, büyük veya çok kısıtlı örneklerde sıralama değişebilir (bkz. Bölüm 6.4).",
        f"<b>Sınırlı tekrar ve ayar:</b> {len(config.SEEDS)} çalıştırma; her algoritmada yalnızca bir parametre "
        "değiştirilmiş (<i>c</i><sub>1</sub>, <i>c</i><sub>2</sub>, <i>T</i><sub>0</sub>, <i>β</i>, turnuva boyutu sabit); "
        "ana değerler 10 pilot tohum ve 3 adaylı ızgarayla seçilmiş, Binary PSO için post-hoc bir düzey eklenmiştir; "
        "bütçe ≈8000 FE ile sınırlıdır (daha uzun bütçede sıralama değişebilir).",
        "<b>Süre ölçümü:</b> tek makinede, vektörize NumPy ile; her çalıştırmanın medyan-of-3 süresi. İlk (tek geçişli, "
        "makine yüklü) ölçümde süreler "
        f"{slowdown.min():.1f}–{slowdown.max():.1f}× daha yüksek çıkmış, algoritmalar arası oranlar ise korunmuştur "
        f"(ACO/GA: {aco_ga_first:.0f}× ve {aco_ga_final:.0f}×). Sıralama güvenilirdir, mutlak süreler değildir.",
        f"<b>Hibrit:</b> tek bir tasarım denenmiştir; 9 adaylı pilot seçimin pilot ortalaması ({pilot_row['mean']:.1f}) "
        f"raporlanan tohumlarda düşmüştür ({hybrid_mean:.1f}; seçim yanlılığı).",
    ]))
    return out


def section_conclusion(ctx: Ctx) -> list[Flowable]:
    d = ctx.data
    s = d.summary_main
    _check(int(d.pairwise_main["significant_holm_wilcoxon"].sum()) == len(d.pairwise_main), "6 ikili farkın tamamı anlamlı olmalı")
    pb = d.pairwise_bonus
    ga_vs_hyb = pb[(pb["a"] == "GA") & (pb["b"] == "GA-SA")].iloc[0]
    sa_vs_hyb = pb[(pb["a"] == "SA") & (pb["b"] == "GA-SA")].iloc[0]
    _check(ga_vs_hyb["wilcoxon_p_holm"] >= 0.05 and ga_vs_hyb["mannwhitney_p_holm"] >= 0.05, "GA-SA ≈ GA olmalı")
    _check(sa_vs_hyb["wilcoxon_p_holm"] >= 0.05 and sa_vs_hyb["mannwhitney_p_holm"] >= 0.05, "GA-SA > SA anlamlı olmamalı")
    _check(d.summary_bonus.loc["GA-SA", "mean"] > s.loc["SA", "mean"], "GA-SA ortalaması SA'dan yüksek olmalı")
    out: list[Flowable] = [ctx.h1("8. Sonuç")]
    out.extend(ctx.bullets([
        f"<b>Kalite:</b> ACO {n_runs(s)}/{n_runs(s)} çalıştırmada gerçek optimumu ({num(d.reference['optimum_value_dp'], 0)}) "
        f"buldu; GA {num(s.loc['GA', 'mean'], 2)}, SA {num(s.loc['SA', 'mean'], 2)}, Binary PSO (<i>w</i> = 0.9) "
        f"{num(s.loc['BPSO', 'mean'], 1)}. Altı ikili farkın tamamı Holm sonrası anlamlıdır; ACO–GA farkı pratikte "
        "küçük, güvenilirlikte büyüktür.",
        f"<b>Hız ve kararlılık:</b> GA ve Binary PSO en hızlı ({num(s.loc['GA', 'time_mean_s'], 3)} s ve {num(s.loc['BPSO', 'time_mean_s'], 3)} s), ACO en yavaş "
        f"({num(s.loc['ACO', 'time_mean_s'], 2)} s) ama en kararlı (std = 0).",
        "<b>Parametreler:</b> GA'da 5/<i>n</i> mutasyon, SA'da α = 0.99, Binary PSO'da <i>w</i> ≤ 0.9 belirgin kayıp "
        "yaratır; ACO ρ'ya duyarsızdır; Binary PSO'da atalet büyüdükçe sonuç iyileşir (sigmoid gürültüsü).",
        "<b>Kısıt yönetimi:</b> doldurmalı onarım ve sezgisel bilgi algoritma seçiminden daha belirleyicidir; ana "
        "deneydeki ACO üstünlüğü kısmen bilgi avantajıdır.",
        "<b>Bonus (GA-SA):</b> bu kolay örnekte GA'dan ayırt edilemedi, SA'ya göre ortalamada iyi ama anlamlı değil; "
        "hibritin değeri daha zor örneklerde sınanmalıdır.",
        "<b>Öneri:</b> güvenilirlik için ACO, hız için GA; bu veri setinde ise DP milisaniyelerde kesin sonuç "
        "verdiğinden pratikte önerilen yöntem DP'dir.",
    ]))
    out.append(ctx.p(
        "<b>Gelecek çalışmalar:</b> güçlü ilişkili, büyük ve çok boyutlu (MKP) örnekler; çoklu parametre ayarı "
        "(ör. irace); gerçek zaman bütçesi; GA→SA ardışık cilalama ve yerel aramalı Binary PSO gibi hibritler."))
    return out


def section_references(ctx: Ctx) -> list[Flowable]:
    out: list[Flowable] = [ctx.h1("Kaynakça")]
    for index, (_, text) in enumerate(references(), start=1):
        out.append(Paragraph(f"[{index}] {text}", ctx.styles["ref"]))
    return out
