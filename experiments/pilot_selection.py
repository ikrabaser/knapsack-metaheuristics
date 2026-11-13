"""Pilot seçim deneyi: ana konfigürasyonun **raporlanmayan** pilot tohumlarla seçimi.

Her algoritmanın parametre-duyarlılığı ızgarasındaki (``config.sensitivity_grids``)
her düzey, ``config.PILOT_SEEDS`` ile çalıştırılır; en yüksek ortalama uygunluğa
sahip düzey ana konfigürasyon olur (eşitlikte: düşük std., sonra ortadaki düzey).
Böylece **her algoritma eşit ayar çabası** alır ve seçim, raporlanan
çalıştırmalardan bağımsız olduğu için karşılaştırmayı yanlı kılmaz.

Hibrit için ayrıca küçük bir (SA payı x başlangıç sıcaklığı) ızgarası taranır;
hibrite **daha fazla** ayar çabası verilmiştir (9 aday), bu da hibrit lehine bir
ön yargıdır ve raporda belirtilir.

Çıktı: ``results/tables/pilot_selection.csv``. Betik ayrıca ``config.py`` içindeki
ana değerlerin pilot sonuçlarıyla tutarlı olup olmadığını denetler.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import config
from experiments.runner import RunSpec, prepare_problem, run_specs

HYBRID_PILOT_SA_EVALUATIONS: tuple[int, ...] = (6, 12, 20)
HYBRID_PILOT_TEMPERATURES: tuple[float, ...] = (10.0, 30.0, 100.0)


def _summarise(runs: pd.DataFrame, extra_columns: list[str]) -> pd.DataFrame:
    grouped = runs.groupby(["label", *extra_columns], sort=False)
    return grouped.agg(
        mean=("best_value", "mean"),
        std=("best_value", lambda v: float(np.std(v, ddof=1))),
        hits=("hit_optimum", "sum"),
    ).reset_index()


def _select_best(group: pd.DataFrame) -> int:
    """Seçim kuralı: en yüksek ortalama; eşitlikte düşük std.; yine eşitse ortadaki düzey.

    Returns:
        Seçilen satırın dizini (``group.index`` etiketi).
    """
    top_mean = group["mean"].max()
    tied = group[np.isclose(group["mean"], top_mean, atol=1e-9)]
    tied = tied[np.isclose(tied["std"], tied["std"].min(), atol=1e-9)]
    positions = {index: position for position, index in enumerate(group.index)}
    middle = (len(group) - 1) / 2.0
    return int(min(tied.index, key=lambda index: abs(positions[index] - middle)))


def run_pilot(verbose: bool = True) -> pd.DataFrame:
    """Pilot deneyi çalıştırır, tabloyu kaydeder ve seçilen düzeyleri döndürür."""
    problem, optimum = prepare_problem(write_logs=False)
    n_items = problem.n_items
    main_params = config.main_parameters(n_items)
    specs: list[RunSpec] = []
    for name, grid in config.sensitivity_grids(n_items).items():
        for label, value in grid.levels:
            specs.append(
                RunSpec(
                    f"{name}|{label}",
                    name,
                    {**main_params[name], grid.parameter: value, "max_iterations": config.ITERATIONS},
                    {"parameter": grid.parameter, "level": label, "level_value": value},
                )
            )
    for sa_evals, temperature in itertools.product(HYBRID_PILOT_SA_EVALUATIONS, HYBRID_PILOT_TEMPERATURES):
        specs.append(
            RunSpec(
                f"GA-SA|sa={sa_evals},T0={temperature:g}",
                "GA-SA",
                {
                    **main_params["GA-SA"],
                    "sa_evaluations_per_generation": sa_evals,
                    "sa_initial_temperature": temperature,
                    "max_iterations": config.ITERATIONS,
                },
                {"parameter": "sa_evals/T0", "level": f"{sa_evals}/{temperature:g}", "level_value": float(sa_evals)},
            )
        )

    data = run_specs(
        problem, specs, config.PILOT_SEEDS, optimum.value, "pilot", keep_curves=False, verbose=verbose
    )
    table = _summarise(data.runs, ["algorithm", "parameter", "level"])
    table["selected"] = False
    for _, group in table.groupby("algorithm"):
        table.loc[_select_best(group), "selected"] = True
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TABLES_DIR / "pilot_selection.csv", index=False, encoding="utf-8")
    if verbose:
        print(table.to_string(index=False))
        _check_config(table, n_items)
    return table


def _check_config(table: pd.DataFrame, n_items: int) -> None:
    """``config.main_parameters`` ile pilot seçimin tutarlılığını yazdırır."""
    params = config.main_parameters(n_items)
    checks = {
        "GA": ("mutation_probability", params["GA"]["mutation_probability"]),
        "ACO": ("evaporation_rate", params["ACO"]["evaporation_rate"]),
        "BPSO": ("inertia", params["BPSO"]["inertia"]),
        "SA": ("cooling_rate", params["SA"]["cooling_rate"]),
    }
    grids = config.sensitivity_grids(n_items)
    print("\nAna konfigürasyon denetimi (config.py vs. pilot en iyisi):")
    for algorithm, (parameter, configured) in checks.items():
        picked = table[(table["algorithm"] == algorithm) & table["selected"]].iloc[0]
        picked_value = dict(grids[algorithm].levels)[picked["level"]]
        status = "UYUMLU" if np.isclose(picked_value, configured) else "UYUMSUZ -> config.py güncellenmeli"
        print(f"  {algorithm:5s} {parameter:22s} config={configured:.4g}  pilot-en-iyi={picked['level']}  -> {status}")
    hybrid = table[(table["algorithm"] == "GA-SA") & table["selected"]].iloc[0]
    hybrid_cfg = params["GA-SA"]
    configured = f"{hybrid_cfg['sa_evaluations_per_generation']}/{hybrid_cfg['sa_initial_temperature']:g}"
    status = "UYUMLU" if configured == hybrid["level"] else "UYUMSUZ -> config.py güncellenmeli"
    print(f"  GA-SA  (SA değerlendirmesi/T0) config={configured}  pilot-en-iyi={hybrid['level']}  -> {status}")


if __name__ == "__main__":
    run_pilot()
