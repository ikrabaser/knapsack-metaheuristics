# GitHub'a Aşama Aşama Commit Planı

Proje klasöründe (içinde `main.py` olan klasör) çalıştırın. Komutlar Windows PowerShell ve Git Bash'te
aynen çalışır (`;` ile ayrılmıştır). Her commit, ödevin bir aşamasına karşılık gelir.

## 0. Depoyu başlatın

```bash
git init -b main
```

## 1. Commitler (sırayla)

**Aşama 1 — Veri ve tasarım: iskelet, bağımlılıklar, veri seti, merkezi yapılandırma**

```bash
git add .gitignore requirements.txt pytest.ini config.py data/dataset.csv; git commit -m "chore: proje iskeleti, bağımlılıklar, veri seti ve merkezi yapılandırma"
```

**Aşama 2a — Ortak problem altyapısı**

```bash
git add core/; git commit -m "feat(core): 0/1 Knapsack modeli, ortak sayaçlı uygunluk, onarım stratejileri ve DP/MILP/brute-force çözücüler"
```

**Aşama 2b — Algoritmalar**

```bash
git add algorithms/; git commit -m "feat(algorithms): GA, ACO, ikili PSO, SA ve GA-SA hibriti (aynı arayüz, FE sayımı, global-en-iyi izleme)"
```

**Aşama 3 — Doğrulama testleri**

```bash
git add tests/test_core.py tests/test_algorithms.py; git commit -m "test: brute-force doğrulamalı birim testleri (DP/MILP, onarım, operatörler, FE sayımı, tekrarlanabilirlik)"
```

**Aşama 4a — Deney altyapısı, veri analizi, pilot seçim**

```bash
git add experiments/__init__.py experiments/runner.py experiments/dataset_analysis.py experiments/pilot_selection.py results/tables/dataset_summary.csv results/tables/dataset_overview.csv results/tables/pilot_selection.csv results/raw/data_cleaning_log.json results/raw/reference_solution.json; git commit -m "feat(experiments): ortak deney çalıştırıcı, veri seti analizi, referans optimum ve pilot tohumlarla ana parametre seçimi"
```

**Aşama 4b — Ana deney (4 algoritma × 30 çalıştırma + bonus hibrit)**

```bash
git add experiments/benchmark.py results/raw/main_runs.csv results/raw/main_curves.npz results/raw/bonus_hybrid_runs.csv results/raw/bonus_hybrid_curves.npz results/raw/run_metadata.json results/raw/timing_first_pass.csv; git commit -m "feat(experiments): ana karşılaştırma deneyi (30 tohum, medyan-of-3 süre ölçümü) ve ayrı bonus hibrit koşusu"
```

**Aşama 5 — Parametre duyarlılığı**

```bash
git add experiments/parameter_sensitivity.py results/raw/sensitivity_runs.csv results/raw/sensitivity_curves.npz; git commit -m "feat(experiments): parametre duyarlılığı (GA mutasyon, ACO buharlaşma, PSO atalet, SA soğutma)"
```

**Aşama 5b — Ek deneyler ve ablasyonlar**

```bash
git add experiments/extra_experiments.py results/raw/extra_*; git commit -m "feat(experiments): eşit bütçe denetimi, onarım ve ACO sezgisel-bilgi ablasyonu, hibrit SA payı ve kapasite oranı analizi"
```

**Aşama 6a — İstatistiksel analiz ve tablolar**

```bash
git add experiments/statistical_analysis.py tests/test_experiments.py results/tables/; git commit -m "feat(analysis): Friedman/Kruskal-Wallis, Wilcoxon/Mann-Whitney (Holm), etki büyüklükleri, güven aralıkları ve özet tablolar"
```

**Aşama 6b — Grafikler**

```bash
git add visualization/ results/figures/; git commit -m "feat(viz): akademik grafikler (yakınsama, kutu, ortalama/süre/std, kalite-süre, duyarlılık, ablasyon; PNG 300 dpi + PDF)"
```

**Aşama 7 — Türkçe akademik PDF rapor**

```bash
git add reports/; git commit -m "feat(report): Türkçe akademik PDF rapor (10 sayfa; sayılar doğrudan sonuç tablolarından, iddia denetimli)"
```

**Aşama 8 — CLI, belgeler, son kontrol**

```bash
git add main.py README.md docs/; git commit -m "docs: tek komutlu CLI (main.py), README ve savunma rehberi"
```

Kalan dosya kalmadığını doğrulayın (çıktı boş olmalı):

```bash
git status --short
```

## 2. GitHub'a gönderin

GitHub'da **boş** bir depo oluşturun (README/lisans eklemeden), sonra:

```bash
git remote add origin https://github.com/KULLANICI_ADINIZ/knapsack-metaheuristics.git
```

```bash
git push -u origin main
```

> Not: `results/` klasörü bilerek repoya eklenir (ödev gerçek deney çıktılarını ister). Boyutu birkaç MB'tır.
