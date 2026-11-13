"""Rapor için veri erişimi ve biçimlendirme.

Rapordaki **tüm sayılar** ``results/tables`` ve ``results/raw`` altındaki gerçek deney
çıktılarından okunur; metne elle sayı yazılmaz. Nitel iddialar ``assert_claims`` ile
sayısal sonuçlara karşı denetlenir: sonuçlar değişirse rapor üretimi hata verir ve
metnin gözden geçirilmesi gerekir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config


def read_csv(name: str) -> pd.DataFrame:
    """``results/tables/<name>.csv`` dosyasını okur."""
    return pd.read_csv(config.TABLES_DIR / f"{name}.csv", encoding="utf-8")


def read_json(name: str) -> dict:
    """``results/raw/<name>.json`` dosyasını okur."""
    return json.loads((config.RAW_DIR / f"{name}.json").read_text(encoding="utf-8"))


def num(value: float, digits: int = 2) -> str:
    """Ondalık nokta ile sabit basamaklı sayı; NaN için 'n/a'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def pval(value: float) -> str:
    """p-değeri biçimi (çok küçükler '<0.001')."""
    if value is None or np.isnan(value):
        return "n/a"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def pct(value: float, digits: int = 1) -> str:
    return f"%{value:.{digits}f}"


@dataclass
class ReportData:
    """Rapor boyunca kullanılan tüm tablolar ve meta veriler."""

    summary_main: pd.DataFrame
    pairwise_main: pd.DataFrame
    omnibus_main: pd.DataFrame
    normality_main: pd.DataFrame
    summary_bonus: pd.DataFrame
    pairwise_bonus: pd.DataFrame
    sensitivity: pd.DataFrame
    summary_equal_fe: pd.DataFrame
    summary_hybrid_share: pd.DataFrame
    summary_aco_beta: pd.DataFrame
    repair: dict[str, pd.DataFrame]
    capacity: pd.DataFrame
    dataset_summary: pd.DataFrame
    dataset_overview: pd.Series
    reference: dict
    metadata: dict
    pilot: pd.DataFrame
    sensitivity_kw: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls) -> "ReportData":
        sensitivity = read_csv("summary_sensitivity_all")
        kruskal = {
            algorithm: float(group["kruskal_p_all_levels"].iloc[0])
            for algorithm, group in sensitivity.groupby("algorithm")
        }
        return cls(
            summary_main=read_csv("summary_main").set_index("label"),
            pairwise_main=read_csv("pairwise_main"),
            omnibus_main=read_csv("omnibus_main").set_index("test"),
            normality_main=read_csv("normality_main").set_index("label"),
            summary_bonus=read_csv("summary_bonus_hybrid").set_index("label"),
            pairwise_bonus=read_csv("pairwise_bonus_hybrid"),
            sensitivity=sensitivity,
            summary_equal_fe=read_csv("summary_equal_fe").set_index("label"),
            summary_hybrid_share=read_csv("summary_hybrid_share").set_index("label"),
            summary_aco_beta=read_csv("summary_aco_beta").set_index("label"),
            repair={a: read_csv(f"summary_repair_{a}").set_index("label") for a in ("GA", "BPSO", "SA")},
            capacity=read_csv("summary_capacity"),
            dataset_summary=read_csv("dataset_summary").set_index("Değişken"),
            dataset_overview=read_csv("dataset_overview").iloc[0],
            reference=read_json("reference_solution"),
            metadata=read_json("run_metadata"),
            pilot=read_csv("pilot_selection"),
            sensitivity_kw=kruskal,
        )

    # ------------------------------------------------------------------ #
    def main(self, algorithm: str, column: str) -> float:
        """Ana deney özet tablosundan bir değer."""
        return float(self.summary_main.loc[algorithm, column])

    def pair(self, first: str, second: str, column: str, table: pd.DataFrame | None = None) -> float:
        """İkili karşılaştırma tablosundan (sıra bağımsız) bir değer; ters sırada işareti çevirir."""
        table = self.pairwise_main if table is None else table
        row = table[(table["a"] == first) & (table["b"] == second)]
        if not row.empty:
            return float(row[column].iloc[0])
        row = table[(table["a"] == second) & (table["b"] == first)]
        value = float(row[column].iloc[0])
        if column in ("mean_diff", "median_diff", "cohens_d", "cliffs_delta"):
            return -value
        if column == "a12":
            return 1.0 - value
        return value

    def sens(self, label: str, column: str) -> float:
        """Duyarlılık özet tablosundan bir değer (etiket: ``'GA|1/n'``)."""
        return float(self.sensitivity.set_index("label").loc[label, column])


def references() -> list[tuple[str, str]]:
    """Kaynakça: ``(anahtar, biçimlenmiş metin)``; alfabetik sıralı."""
    items = {
        "derrac": "Derrac, J., García, S., Molina, D., &amp; Herrera, F. (2011). A practical tutorial on the use of nonparametric statistical tests as a methodology for comparing evolutionary and swarm intelligence algorithms. <i>Swarm and Evolutionary Computation</i>, 1(1), 3–18.",
        "dorigo": "Dorigo, M., Maniezzo, V., &amp; Colorni, A. (1996). Ant system: Optimization by a colony of cooperating agents. <i>IEEE Transactions on Systems, Man, and Cybernetics – Part B</i>, 26(1), 29–41.",
        "friedman": "Friedman, M. (1937). The use of ranks to avoid the assumption of normality implicit in the analysis of variance. <i>Journal of the American Statistical Association</i>, 32(200), 675–701.",
        "goldberg": "Goldberg, D. E. (1989). <i>Genetic Algorithms in Search, Optimization, and Machine Learning</i>. Addison-Wesley.",
        "harris": "Harris, C. R., et al. (2020). Array programming with NumPy. <i>Nature</i>, 585, 357–362.",
        "holland": "Holland, J. H. (1975). <i>Adaptation in Natural and Artificial Systems</i>. University of Michigan Press.",
        "holm": "Holm, S. (1979). A simple sequentially rejective multiple test procedure. <i>Scandinavian Journal of Statistics</i>, 6(2), 65–70.",
        "huangfu": "Huangfu, Q., &amp; Hall, J. A. J. (2018). Parallelizing the dual revised simplex method. <i>Mathematical Programming Computation</i>, 10(1), 119–142.",
        "kellerer": "Kellerer, H., Pferschy, U., &amp; Pisinger, D. (2004). <i>Knapsack Problems</i>. Springer.",
        "kennedy95": "Kennedy, J., &amp; Eberhart, R. (1995). Particle swarm optimization. <i>Proc. IEEE International Conference on Neural Networks</i>, Vol. 4, 1942–1948.",
        "kennedy97": "Kennedy, J., &amp; Eberhart, R. C. (1997). A discrete binary version of the particle swarm algorithm. <i>Proc. IEEE International Conference on Systems, Man, and Cybernetics</i>, Vol. 5, 4104–4108.",
        "kirkpatrick": "Kirkpatrick, S., Gelatt, C. D., &amp; Vecchi, M. P. (1983). Optimization by simulated annealing. <i>Science</i>, 220(4598), 671–680.",
        "kruskal": "Kruskal, W. H., &amp; Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. <i>Journal of the American Statistical Association</i>, 47(260), 583–621.",
        "mann": "Mann, H. B., &amp; Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. <i>Annals of Mathematical Statistics</i>, 18(1), 50–60.",
        "martello": "Martello, S., &amp; Toth, P. (1990). <i>Knapsack Problems: Algorithms and Computer Implementations</i>. Wiley.",
        "metropolis": "Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., &amp; Teller, E. (1953). Equation of state calculations by fast computing machines. <i>Journal of Chemical Physics</i>, 21(6), 1087–1092.",
        "michalewicz": "Michalewicz, Z. (1996). <i>Genetic Algorithms + Data Structures = Evolution Programs</i> (3rd ed.). Springer.",
        "moscato": "Moscato, P. (1989). <i>On evolution, search, optimization, genetic algorithms and martial arts: Towards memetic algorithms</i> (Caltech Concurrent Computation Program, C3P Report 826). California Institute of Technology.",
        "stuetzle": "Stützle, T., &amp; Hoos, H. H. (2000). MAX–MIN Ant System. <i>Future Generation Computer Systems</i>, 16(8), 889–914.",
        "vargha": "Vargha, A., &amp; Delaney, H. D. (2000). A critique and improvement of the CL common language effect size statistics of McGraw and Wong. <i>Journal of Educational and Behavioral Statistics</i>, 25(2), 101–132.",
        "virtanen": "Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. <i>Nature Methods</i>, 17, 261–272.",
        "wilcoxon": "Wilcoxon, F. (1945). Individual comparisons by ranking methods. <i>Biometrics Bulletin</i>, 1(6), 80–83.",
        "wolpert": "Wolpert, D. H., &amp; Macready, W. G. (1997). No free lunch theorems for optimization. <i>IEEE Transactions on Evolutionary Computation</i>, 1(1), 67–82.",
    }
    return sorted(items.items(), key=lambda kv: kv[1].lower())


def citation_map() -> dict[str, int]:
    """Anahtar -> kaynakça numarası (alfabetik sıraya göre 1'den başlar)."""
    return {key: index + 1 for index, (key, _) in enumerate(references())}


def cite(*keys: str) -> str:
    """``cite('holland', 'goldberg')`` -> ``'[4, 6]'`` (sıralı numaralar)."""
    mapping = citation_map()
    numbers = sorted(mapping[key] for key in keys)
    return "[" + ", ".join(str(n) for n in numbers) + "]"


def figure_path(name: str) -> Path:
    """Şekil PNG yolu (var olmalıdır)."""
    path = config.FIGURES_DIR / f"{name}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Şekil bulunamadı: {path}; önce 'python main.py --stage figures' çalıştırın.")
    return path
