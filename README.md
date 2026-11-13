# Meta-Sezgisel Algoritmaların 0/1 Knapsack Problemi Üzerinde Karşılaştırılması

Genetik Algoritma (GA), Karınca Kolonisi Optimizasyonu (ACO), İkili Parçacık Sürü
Optimizasyonu (Binary PSO) ve Simulated Annealing (SA) algoritmaları, **aynı** 100 ürünlü
0/1 Knapsack örneği üzerinde, **aynı** hesaplama bütçesi ve **aynı** tohumlarla
karşılaştırılır. Bonus olarak **GA-SA hibrit (memetik)** yöntemi geliştirilmiştir.

Problem: `max Σ vᵢxᵢ  koşul  Σ wᵢxᵢ ≤ C,  xᵢ ∈ {0,1}`, kapasite `C = 0.70 · Σ wᵢ = 1906.8`
(yuvarlama yapılmaz). Veri seti `data/dataset.csv` (100 ürün: `item, weight, value`).
Gerçek optimum, metasezgisellerden bağımsız olarak **dinamik programlama** ile hesaplanır ve
**MILP (HiGHS)** ile çapraz doğrulanır.

> Tüm sayısal sonuçlar gerçek çalıştırmalardan gelir (`results/raw/*.csv`); raporlanan her
> değer `results/tables/*.csv` dosyalarından üretilir. Elle girilmiş sonuç yoktur.

## Proje yapısı

```
├── data/dataset.csv                 # 100 ürünlük veri seti
├── core/                            # ortak altyapı
│   ├── knapsack.py                  #   problem sınıfı (kapasite = oran × toplam ağırlık)
│   ├── constraints.py               #   onarım (repair) stratejileri
│   ├── fitness.py                   #   sayaçlı ortak nesne fonksiyonu (FE sayımı)
│   ├── exact.py                     #   DP, MILP, brute-force, LP üst sınırı, açgözlü
│   └── data_loader.py               #   yükleme, temizleme günlüğü, betimsel istatistik
├── algorithms/                      # aynı arayüz: Algoritma(problem, **param).run(seed)
│   ├── genetic_algorithm.py         #   GA
│   ├── ant_colony.py                #   ACO (ürün-feromonlu, MAX–MIN tarzı)
│   ├── binary_pso.py                #   Binary PSO (sigmoid transfer)
│   ├── simulated_annealing.py       #   SA (bit-flip + swap, Metropolis)
│   └── hybrid_ga_sa.py              #   GA-SA hibriti (bonus)
├── experiments/
│   ├── runner.py                    #   ortak çalıştırıcı (serpiştirilmiş tohumlar, ısınma, doğrulama)
│   ├── pilot_selection.py           #   ana konfigürasyonun pilot tohumlarla seçimi
│   ├── benchmark.py                 #   ana deney (+ hibrit koşusu)
│   ├── parameter_sensitivity.py     #   parametre duyarlılığı
│   ├── extra_experiments.py         #   eşit bütçe, onarım/ACO ablasyonu, hibrit payı, kapasite oranları
│   ├── statistical_analysis.py      #   özet tablolar, güven aralıkları, Friedman/Wilcoxon, Holm, etki büyüklüğü
│   └── dataset_analysis.py          #   veri seti istatistikleri
├── visualization/{style,plots}.py   # tüm grafikler (PNG 300 dpi + PDF)
├── reports/build_report.py          # Türkçe akademik PDF raporu
├── tests/                           # birim testleri (brute-force doğrulamalı)
├── docs/SAVUNMA_REHBERI.md          # sunum/savunma rehberi (jüri soruları, bulgular)
├── results/{raw,tables,figures}/    # ham koşular, tablolar, grafikler
├── config.py                        # tüm parametreler, tohumlar, yollar (tek merkez)
├── main.py                          # komut satırı girişi
└── requirements.txt
```

## Kurulum

Python ≥ 3.10 önerilir (geliştirme/sonuçlar: Python 3.11.0).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma

Tüm hat (testler → veri analizi → pilot → ana deney → duyarlılık → ek deneyler → istatistik →
grafikler → PDF rapor):

```bash
python main.py
```

Tek tek aşamalar (her biri `results/` altına yazar):

```bash
python main.py --stage tests          # birim testleri
python main.py --stage data           # veri seti analizi, referans optimum (DP + MILP)
python main.py --stage pilot          # ana konfigürasyonun pilot seçimi (raporlanmaz tohumlarla)
python main.py --stage benchmark      # ana deney (GA, ACO, BPSO, SA) + GA-SA hibriti
python main.py --stage sensitivity    # parametre duyarlılığı
python main.py --stage extras         # eşit bütçe, ablasyonlar, kapasite oranları
python main.py --stage stats          # istatistik tabloları
python main.py --stage figures        # grafikler
python main.py --stage report         # reports/final_report.pdf
```

Hızlı duman testi (3 çalıştırma, 30 iterasyon; gerçek sonuçları **ezmez**, `results_quick/` kullanır):

```bash
python main.py --quick --skip-tests
```

Ham sonuçlar korunarak yalnızca grafik ve raporu yeniden üretmek için:

```bash
python main.py --stage figures
python main.py --stage report
```

Toplam süre kabaca 25–35 dakikadır (en yavaş bileşen ACO'dur ve süre ölçümü 3 tekrar içerir); süreler makineye bağlıdır.

## Ana sonuçlar (30 bağımsız çalıştırma; gerçek optimum = 4560, DP + MILP)

| Algoritma | Ortalama ± std | En iyi / en kötü | Optimumu bulma |
|---|---|---|---|
| ACO | 4560.00 ± 0.00 | 4560 / 4560 | %100 |
| GA | 4558.10 ± 2.44 | 4560 / 4552 | %53 |
| SA | 4554.57 ± 7.01 | 4560 / 4523 | %6.7 |
| Binary PSO (w = 0.9) | 4418.17 ± 37.78 | 4499 / 4346 | %0 |
| *GA-SA hibriti (bonus, ayrı)* | 4557.33 ± 3.16 | 4560 / 4548 | %37 |

Süreler, istatistik testleri, parametre duyarlılığı ve ablasyonlar için `results/tables/` ve rapora bakın.
Bu bulgular **tek ve görece kolay bir örneğe** aittir (açgözlü sezgisel bile %0.18 yakındır).

## Parametreleri değiştirme

Her şey `config.py` içindedir:

| Ne | Nerede |
|---|---|
| İterasyon sayısı, popülasyon/sürü/karınca sayısı, çalıştırma sayısı | `ITERATIONS`, `POPULATION_SIZE`, `N_RUNS` |
| Önceden tanımlı tohumlar (raporlanan) / pilot tohumlar | `SEEDS`, `PILOT_SEEDS` |
| Kapasite oranı | `CAPACITY_RATIO` (0.70), `CAPACITY_RATIOS` (bonus) |
| Algoritma parametreleri | `main_parameters(n_items)` |
| Duyarlılık ızgaraları | `sensitivity_grids(n_items)` |

Bir algoritmayı doğrudan kullanmak:

```python
from core.data_loader import build_problem
from algorithms import GeneticAlgorithm

problem, _, _ = build_problem("data/dataset.csv", capacity_ratio=0.70)
result = GeneticAlgorithm(problem, population_size=40, mutation_probability=0.02,
                          max_iterations=200).run(seed=2000)
print(result.best_value, result.total_weight, result.n_selected, result.n_evaluations)
```

Her `run(seed)` çağrısı en iyi çözüm vektörünü, amaç değerini, toplam ağırlığı, seçilen ürün
sayısını, *best-so-far* yakınsama geçmişini, süreyi ve nesne fonksiyonu değerlendirme sayısını döndürür.

## Deneyleri yeniden üretme ve tekrarlanabilirlik

* Rastgelelik `numpy.random.default_rng(seed)` ile yönetilir; tohumlar `config.SEEDS` içindedir
  ve `results/raw/run_metadata.json` dosyasına kaydedilir. Aynı algoritma + parametre + tohum
  **birebir aynı sonucu** üretir (duyarlılık deneyi, ana düzeyde ana deneyle bit düzeyinde aynı
  çıktıyı verdiğini otomatik denetler).
* **Süre ölçümü protokolü:** her (algoritma, tohum) çifti `config.TIMING_REPEATS` (=3) kez art arda
  koşturulur, süre olarak **medyan** alınır ve üç tekrarın aynı çözümü verdiği otomatik denetlenir.
  Koşular tohumlar arasında serpiştirilir ve ısınma turu yapılır. Kalite sonuçları makineden bağımsızdır;
  süreler makineye ve yüke bağlıdır (ilk tek-geçişli ölçüm, makine yüklüyken 2.6–3.2× daha yüksek
  süreler vermiştir; `results/raw/timing_first_pass.csv`, oranlar korunmuştur).
* Ana karşılaştırma ve parametre duyarlılığı **ayrı** dosyalardadır (`main_*` / `sensitivity_*`);
  bonus ve ek deneyler de ayrıdır (`bonus_*`, `extra_*`).

## Testler

```bash
python -m pytest
```

Testler; DP/MILP'in brute-force ile uyumunu, onarımın uygunluğunu, her algoritmanın uygun
çözüm ürettiğini, değerlendirme sayısı formüllerini, tohum tekrarlanabilirliğini, GA
operatörlerini, Metropolis kabulünü, sigmoid dönüşümü ve ACO feromon güncellemesini denetler.

## PDF rapor

`python main.py --stage report` komutu `reports/final_report.pdf` dosyasını üretir
(ReportLab; fontlar matplotlib ile gelen DejaVu ailesidir, ek kurulum gerekmez). Kapaktaki
öğrenci bilgileri `reports/build_report.py` içindeki `STUDENT_INFO` sözlüğünden düzenlenir.
