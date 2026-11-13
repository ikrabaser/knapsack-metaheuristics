"""Raporun sonuçlardan bağımsız bölümleri: kapak/özet, giriş, algoritmalar, deneysel yöntem."""

from __future__ import annotations

from dataclasses import dataclass

from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Flowable, Paragraph, Spacer, Table, TableStyle

import config
from reports.pdf_helpers import EquationCounter, FigureNumbering, INK_SOFT, make_table
from reports.report_data import ReportData, cite, figure_path, num


@dataclass
class Ctx:
    """Bölüm üreticilerinin paylaştığı bağlam."""

    styles: dict[str, ParagraphStyle]
    eq: EquationCounter
    fig: FigureNumbering
    width: float
    data: ReportData

    def p(self, text: str, style: str = "body") -> Paragraph:
        return Paragraph(text, self.styles[style])

    def h1(self, text: str) -> Paragraph:
        return Paragraph(text, self.styles["h1"])

    def h2(self, text: str) -> Paragraph:
        return Paragraph(text, self.styles["h2"])

    def bullets(self, items: list[str]) -> list[Flowable]:
        return [Paragraph(item, self.styles["bullet"], bulletText="•") for item in items]


# --------------------------------------------------------------------------- #
# Kapak ve özet
# --------------------------------------------------------------------------- #
STUDENT_INFO: dict[str, str] = {
    "Öğrenci adı soyadı": "",
    "Öğrenci numarası": "",
    "Ders": "",
    "Öğretim üyesi": "",
    "Teslim tarihi": "",
}


def cover(ctx: Ctx, abstract_text: str, keywords: str) -> list[Flowable]:
    """Başlık, düzenlenebilir öğrenci bilgileri, özet ve anahtar kelimeler."""
    s = ctx.styles
    rows = [
        [Paragraph(f"<b>{key}</b>", s["cell"]), Paragraph(value or "&nbsp;", s["cell"])]
        for key, value in STUDENT_INFO.items()
    ]
    info = Table(rows, colWidths=[ctx.width * 0.22, ctx.width * 0.78])
    info.setStyle(TableStyle([
        ("LINEBELOW", (1, 0), (1, -1), 0.5, INK_SOFT), ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return [
        Paragraph("Meta-Sezgisel Optimizasyon Algoritmalarının 0/1 Knapsack Problemi Üzerinde "
                  "Karşılaştırmalı Analizi", s["title"]),
        Paragraph("Genetik Algoritma, Karınca Kolonisi, İkili Parçacık Sürüsü ve Simulated Annealing; "
                  "bonus: GA-SA hibriti", s["subtitle"]),
        info,
        Spacer(1, 6),
        ctx.h2("Özet"),
        Paragraph(abstract_text, s["abstract"]),
        Paragraph(f"<b>Anahtar kelimeler:</b> {keywords}", s["abstract"]),
    ]


# --------------------------------------------------------------------------- #
# 1. Giriş ve problem tanımı
# --------------------------------------------------------------------------- #
def section_intro(ctx: Ctx) -> list[Flowable]:
    d, ov, ref = ctx.data, ctx.data.dataset_overview, ctx.data.reference
    n, total_w, total_v = int(ov["n_items"]), ov["total_weight"], ov["total_value"]
    capacity = ov["capacity"]
    out: list[Flowable] = [ctx.h1("1. Giriş ve Problem Tanımı")]
    out.append(ctx.p(
        "Meta-sezgisel optimizasyon yöntemleri, arama uzayı büyük olduğu için kesin yöntemlerin uygulanamadığı "
        "kombinatoryal problemlerde, problemden büyük ölçüde bağımsız arama stratejileriyle iyi çözümler bulmayı "
        f"amaçlar; evrim (GA) {cite('holland', 'goldberg')}, sürü zekâsı (PSO) {cite('kennedy95', 'kennedy97')}, "
        f"karınca davranışı (ACO) {cite('dorigo')} ve fiziksel tavlama (SA) {cite('kirkpatrick')} gibi doğal "
        f"süreçlerden esinlenirler. Hiçbir algoritma tüm problemlerde üstün olamayacağından {cite('wolpert')}, "
        "yöntemler belirli bir problemde adil ve istatistiksel olarak desteklenen bir tasarımla karşılaştırılmalıdır."))
    out.append(ctx.p(
        "<b>0/1 Knapsack</b> problemi, her ürünün ya tamamen alındığı ya da hiç alınmadığı ve kapasitenin "
        f"aşılmadığı koşulda toplam değerin enbüyüklendiği NP-zor bir problemdir {cite('kellerer', 'martello')}. "
        "Ağırlıklar tamsayı olduğunda dinamik programlama (DP) <i>O</i>(<i>n</i>⌊<i>C</i>⌋) sürede kesin çözüm "
        "verir; bu çalışmada DP, meta-sezgisellerden bağımsız bir <i>referans</i> olarak kullanılır "
        "(<i>x<sub>i</sub></i> = 1: ürün <i>i</i> çantada):"))
    out.append(ctx.eq(r"\max\; f(\mathbf{x})=\sum_{i=1}^{n} v_i\,x_i"))
    out.append(ctx.eq(r"\sum_{i=1}^{n} w_i\,x_i \leq C,\qquad x_i\in\{0,1\},\qquad C=0.70\sum_{i=1}^{n} w_i"))
    dup_pairs = int(ov["n_duplicate_weight_value_pairs"])
    out.append(ctx.p(
        f"<b>Veri seti.</b> {n} ürünlük veri seti kullanılmıştır (Tablo 1). Eksik, tekrarlanan veya geçersiz "
        f"(≤ 0) kayıt yoktur, hiçbir satır silinmemiştir (günlük: <font name='DV-Mono' size='7'>"
        f"results/raw/data_cleaning_log.json</font>); {dup_pairs} ürün aynı (ağırlık, değer) çiftini paylaşır ancak "
        f"farklı ürün olduğu için korunmuştur. Kapasite <i>C</i> = 0.70 × {num(total_w, 0)} = "
        f"<b>{num(capacity, 1)}</b> olup <b>yuvarlanmamıştır</b> (ağırlıklar tamsayı olduğundan uygun küme "
        f"⌊<i>C</i>⌋ = {int(capacity)} ile aynıdır). DP ile <b>gerçek optimum {num(ref['optimum_value_dp'], 0)}</b> "
        f"(ağırlık {num(ref['optimum_weight'], 0)}, {int(ref['optimum_n_selected'])} ürün); bağımsız bir MILP "
        f"çözücüsüyle (HiGHS {cite('huangfu')}) doğrulanmıştır. Oran-açgözlü sezgisel {num(ref['greedy_ratio_value'], 0)} "
        f"(%{num(ref['greedy_gap_pct'], 2)} fark), doğrusal gevşetme üst sınırı {num(ref['lp_relaxation_upper_bound'], 2)}'dir: "
        "örnek meta-sezgiseller için <i>görece kolaydır</i> ve algoritmalar arasındaki farklar dar bir "
        "aralıktadır; bu, istatistiksel test ve etki büyüklüğü analizinin önemini artırır."))
    ds = d.dataset_summary
    rows = [["Değişken", "Min", "Maks", "Ortalama", "Medyan", "Std. sapma"]]
    for name in ("Ağırlık (w)", "Değer (v)", "Değer/ağırlık (v/w)"):
        r = ds.loc[name]
        digits = 2 if "v/w" in name else 0
        rows.append([name, num(r["Min"], digits), num(r["Maks"], digits), num(r["Ortalama"], 2), num(r["Medyan"], 2),
                     num(r["Std. sapma"], 2)])
    corr = ov["corr_weight_value_pearson"]
    out.append(ctx.fig.table_title(
        f"Veri seti özeti (n = {n}; toplam ağırlık {num(total_w, 0)}, toplam değer {num(total_v, 0)}, "
        f"Pearson r(w, v) = {num(corr, 3)})."))
    out.append(make_table(rows, [ctx.width * 0.28] + [ctx.width * 0.144] * 5, ctx.styles))
    out.append(Spacer(1, 3))
    out.append(ctx.fig.figure(
        figure_path("fig01_dataset_overview"), ctx.width,
        "Veri seti: (a) ağırlık, (b) değer, (c) değer/ağırlık dağılımı (log eksen) ve (d) ağırlık–değer saçılımı; "
        "DP optimumu yüksek oranlı ürünleri seçer."))
    out.append(ctx.p(
        "<b>Amaç.</b> GA, ACO, ikili PSO ve SA, aynı veri/kapasite/tohum ve benzer değerlendirme bütçesiyle "
        "karşılaştırılarak (S1) en iyi çözümü hangisinin ürettiği, (S2) hangisinin daha hızlı, (S3) daha kararlı "
        "olduğu, (S4) parametrelerin etkisi ve (S5) hangisinin tercih edilmesi gerektiği yanıtlanır. Bonus olarak "
        "<b>GA-SA hibriti</b> geliştirilmiş ve ana deneyden ayrı değerlendirilmiştir."))
    return out


# --------------------------------------------------------------------------- #
# 2. Algoritmalar
# --------------------------------------------------------------------------- #
def section_algorithms(ctx: Ctx) -> list[Flowable]:
    out: list[Flowable] = [ctx.h1("2. Kullanılan Algoritmalar")]
    out.append(ctx.p(
        "<b>Ortak altyapı.</b> Tüm algoritmalar nesne fonksiyonunu <b>aynı sayaçlı değerlendiriciden</b> hesaplar "
        "(başlangıç popülasyonu dâhil her değerlendirme sayılır; kapasiteyi aşan çözüm değerlendirilemez). "
        "Kapasiteyi aşan çözümler, <b>ortak onarım operatöründen</b> geçer: seçili ürünler arasından "
        "<i>değer/ağırlık oranı en düşük</i> olanlar kapasite sağlanana dek çıkarılır ve sonuç bireye geri "
        f"yazılır {cite('michalewicz')}. GA, ikili PSO ve SA onarımı kullanır; ACO çözümü baştan uygun inşa "
        "eder. Onarımın algoritmalara avantaj sağlayıp sağlamadığı Bölüm 6.3'te ablasyonla ölçülmüştür."))

    out.append(ctx.h2("2.1 Genetik Algoritma (GA)"))
    out.append(ctx.p(
        "Kromozom <i>n</i> uzunluklu ikili vektördür; başlangıç popülasyonu (<i>N</i> = 40) her geni 0.5 olasılıkla "
        "1 olan bireylerdir. <b>Seçim</b>: <i>k</i> = 3 turnuva; <b>çaprazlama</b>: <i>p<sub>c</sub></i> = 0.9, tekdüze; "
        "<b>mutasyon</b>: her gen bağımsız olarak <i>p<sub>m</sub></i> olasılıkla çevrilir (bit-flip); "
        "<b>elitizm</b>: en iyi 2 birey aktarılır; çocuklar onarılır. Mutasyon keşif–sömürü dengesinin ana "
        "ayarıdır: bir çocukta beklenen değişen gen sayısı <i>n·p<sub>m</sub></i>'dir. Küçük <i>p<sub>m</sub></i>'de "
        "popülasyon hızla homojenleşir (sömürü, erken yakınsama); büyük <i>p<sub>m</sub></i>'de iyi çözümlerin "
        "yapısı bozulur (keşif artar, kararlılık azalır)."))

    out.append(ctx.h2("2.2 Karınca Kolonisi Optimizasyonu (ACO)"))
    out.append(ctx.p(
        "Rota problemlerinde (TSP) feromon <i>kenarlara</i> konur ve karınca sıralı bir tur izler. 0/1 Knapsack'te "
        "çözüm sırasız bir <i>kümedir</i>; bu yüzden feromon <b>ürünlere</b> (<i>τ<sub>i</sub></i>) konur. Karınca boş "
        "çantadan başlar ve her adımda henüz seçilmemiş <b>ve kalan kapasiteye sığan</b> adaylar kümesi <i>N</i> "
        "içinden olasılıksal olarak bir ürün ekler; aday kalmayınca inşa biter (çözüm uygun ve maksimaldir). "
        "Sezgisel bilgi mesafe tersi değil, <b>değer/ağırlık oranıdır</b>:"))
    out.append(ctx.eq(r"p_i=\frac{\tau_i^{\alpha}\,\eta_i^{\beta}}{\sum_{j\in\mathcal{N}}\tau_j^{\alpha}\,\eta_j^{\beta}},"
                      r"\qquad \eta_i=\frac{v_i/w_i}{\max_j\,(v_j/w_j)}"))
    out.append(ctx.p(
        f"Feromon güncellemesi MAX–MIN Ant System tarzındadır {cite('stuetzle')}: buharlaşma (<i>ρ</i>), "
        "iterasyon-en-iyi ve global-en-iyi çözümün ürünlerine <i>Q·f/UB</i> pekiştirmesi (<i>UB</i>: doğrusal "
        "gevşetme üst sınırı; yalnızca ölçekleme) ve erken yakınsamayı önleyen sınırlar:"))
    out.append(ctx.eq(r"\tau_i\leftarrow(1-\rho)\,\tau_i+\sum_{s\in\{ib,\,gb\}}\frac{Q\,f(\mathbf{x}^{s})}{UB}\,[x_i^{s}=1],"
                      r"\qquad \tau_{\min}\leq\tau_i\leq\tau_{\max}"))

    out.append(ctx.h2("2.3 İkili Parçacık Sürü Optimizasyonu (Binary PSO)"))
    out.append(ctx.p(
        f"Ayrık problem için Kennedy–Eberhart ikili PSO'su {cite('kennedy97')} kullanılmıştır. Parçacığın ikili "
        "konumu <b>x</b>, hızı <b>v</b>, kişisel en iyisi <b>p</b> ve sürü en iyisi <b>g</b>'dir. Hız, sürekli PSO "
        "ile aynı güncellenir; hız ise bir <i>olasılığa</i> dönüştürülerek konum örneklenir:"))
    out.append(ctx.eq(r"v_{id}\leftarrow w\,v_{id}+c_1 r_1\,(p_{id}-x_{id})+c_2 r_2\,(g_{d}-x_{id}),\qquad |v_{id}|\leq v_{\max}"))
    out.append(ctx.eq(r"S(v_{id})=\frac{1}{1+e^{-v_{id}}},\qquad x_{id}=1\;\mathrm{if}\;U(0,1)<S(v_{id})\;\mathrm{else}\;0"))
    out.append(ctx.p(
        "<b>Sürekli PSO'dan farklar:</b> konum <i>x ← x + v</i> ile kaydırılmaz; hız yalnızca bitin 1 olma "
        "<i>olasılığını</i> belirler, bu yüzden <i>w</i> konum momentumunu değil olasılığın değişim eğilimini korur. "
        "<i>v</i> ≈ 0 iken <i>S</i>(<i>v</i>) = 0.5 olduğundan, hız sönümlendikçe bitler rastgele çevrilmeye başlar "
        "(küçük <i>w</i> bu gürültüyü artırır). Parametreler: <i>c</i><sub>1</sub> = <i>c</i><sub>2</sub> = 2, "
        "<i>v</i><sub>max</sub> = 4; <i>w</i> Bölüm 5'te incelenir."))

    out.append(ctx.h2("2.4 Simulated Annealing (SA)"))
    out.append(ctx.p(
        "SA tek bir çözümü geliştirir (başlangıç: rastgele vektörün onarımı). <b>Komşuluk</b>: olasılık 0.5 ile "
        "<i>swap</i> (seçili bir ürünü çıkar, seçilmemiş birini ekle), aksi hâlde tek <i>bit-flip</i>; kapasiteyi "
        f"aşan komşu onarılır. Kabul, <b>Metropolis</b> kriteriyledir {cite('metropolis')}; sıcaklık her iterasyonda "
        "geometrik soğur ve her sıcaklıkta 40 komşu değerlendirilir:"))
    out.append(ctx.eq(r"P=\min\left(1,\,e^{\Delta f/T}\right),\qquad \Delta f=f(\mathbf{x}')-f(\mathbf{x}),\qquad T_{k}=T_0\,\alpha^{k}"))
    out.append(ctx.p(
        "<b>Kötü çözümün kabulü:</b> Δ<i>f</i> &lt; 0 olan komşu <i>e</i><super>Δ<i>f</i>/<i>T</i></super> olasılıkla kabul "
        "edilir. Yüksek <i>T</i>'de neredeyse her hamle kabul edilir (keşif; yerel optimum çukurlarından kaçış), "
        "<i>T</i> düştükçe kabul olasılığı sıfıra yaklaşır ve algoritma açgözlü yerel aramaya dönüşür (sömürü)."))

    out.append(ctx.h2("2.5 GA-SA hibriti (bonus)"))
    out.append(ctx.p(
        f"<b>Fikir:</b> GA geniş uzayı tarar, SA GA'nın bulduğu iyi çözümleri yerel iyileştirir (memetik algoritma "
        f"{cite('moscato')}). Her nesilde (1) 28 çocuk GA ile üretilir; (2) nesildeki en iyi iki birey için 6'şar "
        "komşulu kısa bir SA zinciri çalıştırılır ve zincirdeki <i>en iyi</i> çözüm bireyin yerine geçer (birey "
        "asla kötüleşmez). SA sıcaklığı nesil sayacıyla soğur (<i>T<sub>g</sub></i> = <i>T</i><sub>0</sub>·0.95<super>g</super>). "
        "Nesil başı değerlendirme 28 + 12 = 40'tır, yani bütçe diğer yöntemlerle uyumludur."))
    return out


# --------------------------------------------------------------------------- #
# 3. Deneysel yöntem
# --------------------------------------------------------------------------- #
def _os_label(platform_text: str) -> str:
    """Python 3.11'in Windows 11'i 'Windows-10' diye bildirmesini derleme numarasından düzeltir."""
    import re

    match = re.search(r"10\.0\.(\d+)", platform_text)
    if platform_text.startswith("Windows") and match and int(match.group(1)) >= 22000:
        return f"Windows 11 (derleme {match.group(1)})"
    return platform_text


def section_method(ctx: Ctx) -> list[Flowable]:
    d = ctx.data
    env = d.metadata["environment"]
    n = int(d.dataset_overview["n_items"])
    params = config.main_parameters(n)
    ev = d.metadata["evaluations_per_run"]
    pop, it = config.POPULATION_SIZE, config.ITERATIONS
    out: list[Flowable] = [ctx.h1("3. Deneysel Yöntem")]
    ram = f", {env['ram_gb']} GB RAM" if "ram_gb" in env else ""
    out.append(ctx.p(
        f"<b>Ortam.</b> Tek makine, tek iş parçacığı: {env['cpu']} ({env['cpu_count_logical']} mantıksal çekirdek{ram}), "
        f"{env.get('os') or _os_label(env['platform'])}; Python {env['python']}, NumPy {env['numpy']} {cite('harris')}, SciPy {env['scipy']} "
        f"{cite('virtanen')}, pandas {env['pandas']}, matplotlib {env['matplotlib']}. Algoritmalar saf Python/NumPy "
        "ile yazılmıştır (SciPy-HiGHS yalnızca referans doğrulaması içindir). Makinede deney sırasında başka "
        "uygulamalar da çalışıyordu; bu yüzden süre olarak üç ardışık tekrarın <b>medyanı</b> alınır."))
    out.append(ctx.p(
        f"<b>Tekrar ve rastgelelik.</b> Her konfigürasyon, önceden tanımlı <b>{len(config.SEEDS)} bağımsız tohumla</b> "
        f"({config.SEEDS[0]}–{config.SEEDS[-1]}; ödevin asgari 10 koşulundan fazla, çünkü koşular ucuzdur ve örnek "
        "sayısı istatistiksel gücü artırır) çalıştırılmıştır. <font name='DV-Mono' size='7'>numpy.random.default_rng(seed)</font> "
        "kullanılır; aynı algoritma+parametre+tohum bit düzeyinde aynı sonucu verir (duyarlılık deneyinin ana düzeyi "
        "ana deneyle birebir aynı çıktıyı üretmiştir). Koşular tohumlar arasında serpiştirilir ve süre ölçümünden önce "
        "kayıt dışı ısınma turu yapılır."))

    ga, aco, pso, sa, hyb = (params[k] for k in ("GA", "ACO", "BPSO", "SA", "GA-SA"))
    nominal = config.EQUAL_BUDGET_EVALUATIONS

    def dev(key: str) -> str:
        return f"{ev[key]} ({(ev[key] / nominal - 1) * 100:+.1f}%)"

    rows = [
        ["Algoritma", "Ana deney parametreleri", "FE (sapma)"],
        ["GA", f"<i>N</i>={ga['population_size']}, <i>p<sub>c</sub></i>={ga['crossover_probability']} (tekdüze), "
               f"<i>p<sub>m</sub></i>=1/<i>n</i>={ga['mutation_probability']:.2f}, <i>k</i>={ga['tournament_size']}, "
               f"elitizm={ga['elitism_size']}; {pop} + {it}×({pop}−2)", dev("GA")],
        ["ACO", f"{aco['n_ants']} karınca, <i>α</i>={aco['alpha']:g}, <i>β</i>={aco['beta']:g}, <i>ρ</i>={aco['evaporation_rate']:g}, "
                f"<i>Q</i>={aco['deposit_strength']:g}, <i>τ</i><sub>min</sub>/<i>τ</i><sub>max</sub>={aco['tau_min_ratio']:g}; "
                f"{it}×{pop}", dev("ACO")],
        ["Binary PSO", f"{pso['swarm_size']} parçacık, <i>w</i>={pso['inertia']:g}, <i>c</i><sub>1</sub>=<i>c</i><sub>2</sub>={pso['c1']:g}, "
                       f"<i>v</i><sub>max</sub>={pso['v_max']:g}; {pop} + {it}×{pop}", dev("BPSO")],
        ["SA", f"<i>T</i><sub>0</sub>={sa['initial_temperature']:g}, geometrik <i>α</i>={sa['cooling_rate']:g}, {sa['neighbors_per_iteration']} "
               f"komşu/iterasyon, swap 0.5; 1 + {it}×{pop}", dev("SA")],
        ["GA-SA (bonus)", f"GA: <i>p<sub>m</sub></i>=1/<i>n</i>, elitizm=2, 28 çocuk/nesil; SA: {hyb['sa_evaluations_per_generation']} "
                          f"değerlendirme/nesil (en iyi {hyb['refine_top_k']} birey), <i>T</i><sub>0</sub>={hyb['sa_initial_temperature']:g}, "
                          f"<i>α</i>={hyb['sa_cooling_rate']:g}; 30 + {it}×40", dev("GA-SA")],
    ]
    out.append(ctx.fig.table_title(
        f"Ana karşılaştırma parametreleri ({it} iterasyon, {pop} birey/karınca/komşu) ve nesne fonksiyonu "
        f"değerlendirme (FE) bütçeleri; sapma, nominal {pop}×({it}+1) = {nominal} bütçesine göredir."))
    out.append(make_table(rows, [ctx.width * 0.13, ctx.width * 0.7, ctx.width * 0.17], ctx.styles, numeric_from_col=99))
    out.append(Spacer(1, 2))
    out.append(ctx.p(
        f"<b>Ana parametre seçimi.</b> Her algoritmanın duyarlılık ızgarasındaki 3 değer, raporlanan tohumlardan "
        f"<i>ayrı</i> {len(config.PILOT_SEEDS)} pilot tohumla denenmiş, en yüksek ortalamalı değer seçilmiştir "
        "(eşitlikte düşük std., sonra ortadaki değer): her algoritma <i>eşit ayar çabası</i> alır ve seçim yanlılığı "
        "raporlanan sonuçlara sızmaz (hibrit 9 adaylı ızgaradan seçildi; bu hibrit lehine bir yanlılıktır). "
        "<b>Adil bütçe.</b> Aynı iterasyon sayısı aynı maliyet demek değildir; bu yüzden her değerlendirme sayılmış "
        "ve bütçe farkı (en fazla ≈%5; GA elitleri yeniden değerlendirmez) Tablo 2'de verilmiştir. Farkın "
        f"sonuçları değiştirmediği, tüm algoritmaların <b>tam {nominal} FE</b> ile çalıştırıldığı ek deneyle "
        "doğrulanmıştır (Bölüm 6.2)."))
    out.append(ctx.p(
        "<b>Metrikler.</b> En iyi/ortalama/en kötü/ortanca nihai değer, örneklem std. sapması (<i>n</i>−1), "
        "CV = std/ortalama, süre (ortalama, std.), FE sayısı, <b>optimalite farkı</b> ve optimumu bulma oranı, "
        "<b>yakınsama hızı</b> (best-so-far değerinin optimumun %99'una ilk ulaştığı iterasyon; ulaşamayan "
        "çalıştırmalar sansürlü):"))
    out.append(ctx.eq(r"\mathrm{Gap}(\%)=\frac{f^{*}-f_{\mathrm{best}}}{f^{*}}\times 100"))
    out.append(ctx.p(
        "<b>İstatistik.</b> Nihai değerler ayrık, optimumla tavanlı ve çok eşitlikli olduğundan normallik "
        f"varsayımı tutmaz (Shapiro–Wilk); parametrik olmayan testler kullanılır: Friedman {cite('friedman')} "
        f"(tohum = blok) ve Kruskal–Wallis {cite('kruskal')} (omnibus); ikili karşılaştırmada Wilcoxon işaretli sıra "
        f"{cite('wilcoxon')} ve Mann–Whitney U {cite('mann')}, <b>Holm</b> düzeltmesiyle {cite('holm')}; etki büyüklüğü "
        f"olarak Vargha–Delaney <i>A</i><sub>12</sub> {cite('vargha')} ve Cohen <i>d</i>; %95 güven aralığı (t ve "
        f"bootstrap) {cite('derrac')}. Algoritmalar farklı rastgele akışları tükettiğinden tohum eşleştirmesi "
        "<i>nominaldir</i>; bu yüzden eşleştirilmiş ve eşleştirilmemiş testlerin uyumu sağlamlık denetimi olarak "
        "birlikte raporlanır."))
    return out
