"""Merkezi deney yapılandırması.

Tüm yollar, rastgele tohumlar, algoritma parametreleri ve parametre-duyarlılığı
ızgaraları burada tanımlanır. Deneyleri değiştirmek için yalnızca bu dosyayı
düzenlemek yeterlidir.

Ana konfigürasyon seçimi
------------------------
Her algoritmanın ana parametresi, **raporlanmayan** pilot tohumlarla
(``PILOT_SEEDS``), parametre-duyarlılığı ızgarasındaki 3 değer arasından
seçilmiştir (her algoritmaya eşit ayar çabası; bkz.
``experiments/pilot_selection.py``). Pilot tohumlar raporlanan 30 çalıştırmanın
tohumlarından ayrıdır; dolayısıyla ana karşılaştırma seçim yanlılığı taşımaz.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Hızlı duman testi modu (yalnızca boru hattını denemek içindir; sonuçlar raporlanmaz):
#   KNAPSACK_QUICK=1  -> 3 çalıştırma, 30 iterasyon
#   KNAPSACK_RESULTS_DIR=<yol> -> çıktıları gerçek results/ yerine bu klasöre yazar
QUICK_MODE: bool = os.environ.get("KNAPSACK_QUICK") == "1"

# --------------------------------------------------------------------------- #
# Yollar
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
_results_override = os.environ.get("KNAPSACK_RESULTS_DIR")
RESULTS_DIR = Path(_results_override) if _results_override else PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
RAW_DIR = RESULTS_DIR / "raw"
REPORTS_DIR = RESULTS_DIR / "report" if _results_override else PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS_DIR / "final_report.pdf"

# --------------------------------------------------------------------------- #
# Problem ve deney tasarımı
# --------------------------------------------------------------------------- #
CAPACITY_RATIO: float = 0.70  # C = 0.70 * toplam ağırlık (yuvarlama yok)
ITERATIONS: int = 30 if QUICK_MODE else 200  # ana döngü sayısı (tüm algoritmalar için aynı)
POPULATION_SIZE: int = 40  # popülasyon / sürü / karınca sayısı; SA'da iterasyon başına komşu sayısı
N_RUNS: int = 3 if QUICK_MODE else 30  # bağımsız çalıştırma sayısı (ödev alt sınırı: 10)
#: Süre ölçümü sağlamlığı: her (konfigürasyon, tohum) çifti art arda bu kadar koşturulur ve MEDYAN süre alınır
#: (sonuçlar deterministiktir; tekrarların aynı çözümü verdiği ayrıca doğrulanır).
TIMING_REPEATS: int = 1 if QUICK_MODE else 3

#: Raporlanan deneylerin önceden tanımlı, sabit tohumları (len == N_RUNS).
SEEDS: tuple[int, ...] = tuple(range(2000, 2000 + N_RUNS))
#: Parametre seçimi için ayrı pilot tohumlar (raporlanan sonuçlara girmez).
PILOT_SEEDS: tuple[int, ...] = tuple(range(9001, 9011))

#: Yakınsama hızı ölçütleri: en iyi değerin optimumun bu oranına ilk ulaştığı iterasyon.
CONVERGENCE_THRESHOLDS: tuple[float, ...] = (0.99, 0.995)

ALGORITHM_ORDER: tuple[str, ...] = ("GA", "ACO", "BPSO", "SA")
BONUS_ALGORITHM: str = "GA-SA"
ALGORITHM_DISPLAY_NAMES: dict[str, str] = {
    "GA": "GA",
    "ACO": "ACO",
    "BPSO": "Binary PSO",
    "SA": "SA",
    "GA-SA": "GA-SA (hibrit)",
}

#: Kapasite oranı deneyi (bonus): C = oran * toplam ağırlık.
CAPACITY_RATIOS: tuple[float, ...] = (0.30, 0.50, 0.70, 0.90)

#: Eşit değerlendirme bütçesi kontrolü: pop * (iterasyon + 1) = 8040.
EQUAL_BUDGET_EVALUATIONS: int = POPULATION_SIZE * (ITERATIONS + 1)


# --------------------------------------------------------------------------- #
# Ana (karşılaştırma) parametreleri
# --------------------------------------------------------------------------- #
def main_parameters(n_items: int) -> dict[str, dict[str, Any]]:
    """Ana karşılaştırmada kullanılan algoritma parametreleri.

    Args:
        n_items: Ürün sayısı ``n`` (GA mutasyon olasılığı ``k/n`` biçiminde).
    """
    return {
        "GA": {
            "population_size": POPULATION_SIZE,
            "crossover_probability": 0.9,
            "mutation_probability": 1.0 / n_items,  # pilot: {1/n, 2/n, 5/n} içinde en iyi
            "tournament_size": 3,
            "elitism_size": 2,
            "crossover_type": "uniform",
        },
        "ACO": {
            "n_ants": POPULATION_SIZE,
            "alpha": 1.0,
            "beta": 2.0,
            "evaporation_rate": 0.3,  # pilot: üç değer de eşit; ortadaki seçildi
            "deposit_strength": 1.0,
            "tau_min_ratio": 0.01,
            "reinforcement": "both",
        },
        "BPSO": {
            "swarm_size": POPULATION_SIZE,
            "inertia": 0.9,  # pilot: {0.4, 0.7, 0.9} içinde en iyi
            "c1": 2.0,
            "c2": 2.0,
            "v_max": 4.0,
        },
        "SA": {
            "initial_temperature": 100.0,
            "cooling_rate": 0.95,  # pilot: {0.90, 0.95, 0.99} içinde en iyi
            "cooling_schedule": "geometric",
            "neighbors_per_iteration": POPULATION_SIZE,
            "swap_probability": 0.5,
        },
        # Bonus: GA-SA hibriti (nesil başına 40 değerlendirme = 28 GA + 12 SA).
        # GA bileşeni ana GA ile aynıdır (p_m = 1/n); SA payı ve T0 pilot taramadan seçildi.
        "GA-SA": {
            "evaluations_per_generation": POPULATION_SIZE,
            "sa_evaluations_per_generation": 12,  # pilot: 9 aday içinde en iyi (12 / T0=100)
            "refine_top_k": 2,
            "elitism_size": 2,
            "crossover_probability": 0.9,
            "mutation_probability": 1.0 / n_items,
            "tournament_size": 3,
            "crossover_type": "uniform",
            "sa_initial_temperature": 100.0,
            "sa_cooling_rate": 0.95,
            "swap_probability": 0.5,
        },
    }


# --------------------------------------------------------------------------- #
# Parametre duyarlılığı ızgaraları
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SensitivityGrid:
    """Tek bir algoritma için parametre duyarlılığı tasarımı.

    Attributes:
        algorithm: Algoritma anahtarı.
        parameter: Değiştirilen yapıcı parametre adı.
        display_name: Grafik/rapor etiketi.
        levels: ``(etiket, değer)`` çiftleri — ödevde istenen değerler.
        extension_levels: Ödev dışı, **post-hoc** eklenen ek düzeyler (varsa).
    """

    algorithm: str
    parameter: str
    display_name: str
    levels: tuple[tuple[str, float], ...]
    extension_levels: tuple[tuple[str, float], ...] = ()


def sensitivity_grids(n_items: int) -> dict[str, SensitivityGrid]:
    """Ödevde istenen parametre ızgaraları (+ BPSO için tek ek düzey)."""
    return {
        "GA": SensitivityGrid(
            "GA",
            "mutation_probability",
            "Mutasyon olasılığı $p_m$",
            (("1/n", 1.0 / n_items), ("2/n", 2.0 / n_items), ("5/n", 5.0 / n_items)),
        ),
        "ACO": SensitivityGrid(
            "ACO",
            "evaporation_rate",
            "Buharlaşma oranı $\\rho$",
            (("0.1", 0.1), ("0.3", 0.3), ("0.5", 0.5)),
        ),
        "BPSO": SensitivityGrid(
            "BPSO",
            "inertia",
            "Atalet katsayısı $w$",
            (("0.4", 0.4), ("0.7", 0.7), ("0.9", 0.9)),
            # Optimum ızgara sınırında (w=0.9) olduğundan, ızgara tek bir düzey
            # genişletildi: w = 1.0 özgün Kennedy-Eberhart ikili PSO'dur (atalet yok).
            extension_levels=(("1.0", 1.0),),
        ),
        "SA": SensitivityGrid(
            "SA",
            "cooling_rate",
            "Soğutma katsayısı $\\alpha$",
            (("0.90", 0.90), ("0.95", 0.95), ("0.99", 0.99)),
        ),
    }


# Hibrit için SA payı ablasyonu: nesil başına SA'ya ayrılan değerlendirme sayısı.
HYBRID_SA_SHARES: tuple[int, ...] = (0, 6, 12, 20, 30)
# ACO sezgisel bilgi ablasyonu: beta = 0 sezgisel bilgiyi tamamen kapatır.
ACO_BETA_LEVELS: tuple[float, ...] = (0.0, 1.0, 2.0)
