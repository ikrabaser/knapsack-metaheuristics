"""Parametre duyarlılığı deneyleri (ana deneyden **ayrı** kayıt).

Her algoritma için tek bir parametre değiştirilir, diğerleri ana konfigürasyonda
sabit tutulur:

========  ==========================  ==========================
Algoritma Parametre                   Düzeyler
========  ==========================  ==========================
GA        mutasyon olasılığı p_m      1/n, 2/n, 5/n
ACO       buharlaşma oranı rho        0.1, 0.3, 0.5
BPSO      atalet katsayısı w          0.4, 0.7, 0.9 (+ ek: 1.0)
SA        soğutma katsayısı alpha     0.90, 0.95, 0.99
========  ==========================  ==========================

Her düzey, ana deneyle **aynı** ``N_RUNS`` tohumla çalıştırılır. Ana
konfigürasyona denk gelen düzey, ana deneyin sonuçlarıyla birebir aynı nihai
değerleri üretmelidir (tutarlılık denetimi :func:`check_consistency_with_main`).

``w = 1.0`` düzeyi ödevde istenmeyen, **post-hoc** eklenmiş bir ek noktadır;
ızgara sınırında (w = 0.9) bir optimum görüldüğü için genişletilmiştir ve
tabloda/grafikte açıkça "ek" olarak işaretlenir.
"""

from __future__ import annotations

import numpy as np

import config
from experiments.runner import (
    ExperimentData,
    RunSpec,
    load_experiment,
    prepare_problem,
    run_specs,
)


def build_sensitivity_specs(n_items: int) -> list[RunSpec]:
    """Tüm duyarlılık konfigürasyonlarını (ana düzey işaretli) üretir."""
    main_params = config.main_parameters(n_items)
    specs: list[RunSpec] = []
    for name, grid in config.sensitivity_grids(n_items).items():
        main_value = main_params[name][grid.parameter]
        for is_extension, levels in ((False, grid.levels), (True, grid.extension_levels)):
            for level_label, value in levels:
                specs.append(
                    RunSpec(
                        label=f"{name}|{level_label}",
                        algorithm=name,
                        params={
                            **main_params[name],
                            grid.parameter: value,
                            "max_iterations": config.ITERATIONS,
                        },
                        meta={
                            "parameter": grid.parameter,
                            "level": level_label,
                            "level_value": float(value),
                            "is_extension": is_extension,
                            "is_main_config": bool(np.isclose(value, main_value)),
                        },
                    )
                )
    return specs


def check_consistency_with_main(sensitivity: ExperimentData, main: ExperimentData) -> None:
    """Ana düzeyin nihai değerlerinin ana deneyle birebir aynı olduğunu doğrular.

    Aynı algoritma + aynı parametre + aynı tohum aynı sonucu vermelidir
    (tekrarlanabilirlik denetimi).
    """
    runs = sensitivity.runs
    for algorithm in config.ALGORITHM_ORDER:
        subset = runs[(runs["algorithm"] == algorithm) & runs["is_main_config"]]
        if subset.empty:
            continue
        sens_values = subset.sort_values("seed")["best_value"].to_numpy()
        main_values = (
            main.runs[main.runs["label"] == algorithm].sort_values("seed")["best_value"].to_numpy()
        )
        if not np.array_equal(sens_values, main_values):
            raise AssertionError(f"{algorithm}: ana düzey sonuçları ana deneyle tutarsız!")


def run_sensitivity(verbose: bool = True) -> ExperimentData:
    """Duyarlılık deneylerini çalıştırır, kaydeder ve ana deneyle tutarlılığı denetler."""
    problem, optimum = prepare_problem(write_logs=False)
    specs = build_sensitivity_specs(problem.n_items)
    if verbose:
        print(f"Parametre duyarlılığı: {len(specs)} konfigürasyon x {len(config.SEEDS)} çalıştırma")
    data = run_specs(
        problem, specs, config.SEEDS, optimum.value, "sensitivity", verbose=verbose,
        timing_repeats=config.TIMING_REPEATS,
    )
    data.save()
    try:
        check_consistency_with_main(data, load_experiment("main"))
        if verbose:
            print("Tutarlılık denetimi geçti: ana düzey sonuçları ana deneyle birebir aynı.")
    except FileNotFoundError:
        if verbose:
            print("Uyarı: ana deney bulunamadı, tutarlılık denetimi atlandı.")
    return data


if __name__ == "__main__":
    run_sensitivity()
