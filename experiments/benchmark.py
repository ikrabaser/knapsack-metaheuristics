"""Ana karşılaştırma deneyi (GA, ACO, Binary PSO, SA) ve bonus hibrit koşusu.

Çıktılar (``results/raw``):

* ``main_runs.csv`` / ``main_curves.npz`` — dört algoritma, ``N_RUNS`` bağımsız çalıştırma
* ``bonus_hybrid_runs.csv`` / ``bonus_hybrid_curves.npz`` — GA-SA hibriti (ana deneyden **ayrı**)
* ``run_metadata.json`` — ortam, tohumlar, parametreler, değerlendirme bütçeleri
* ``reference_solution.json``, ``data_cleaning_log.json``
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import config
from experiments.runner import (
    ExperimentData,
    RunSpec,
    environment_info,
    prepare_problem,
    run_specs,
)


def build_specs(n_items: int, algorithms: tuple[str, ...]) -> list[RunSpec]:
    """Ana parametrelerle ``RunSpec`` listesi üretir."""
    parameters = config.main_parameters(n_items)
    return [
        RunSpec(
            label=name,
            algorithm=name,
            params={**parameters[name], "max_iterations": config.ITERATIONS},
        )
        for name in algorithms
    ]


def run_benchmark(verbose: bool = True) -> tuple[ExperimentData, ExperimentData]:
    """Ana deney + bonus hibrit koşusunu çalıştırır ve ham sonuçları kaydeder."""
    problem, optimum = prepare_problem()
    if verbose:
        print(
            f"Problem: n={problem.n_items}, kapasite C={problem.capacity:.1f}, "
            f"gerçek optimum (DP)={optimum.value:.0f}"
        )
        print(f"Ana deney: {len(config.SEEDS)} çalıştırma x {len(config.ALGORITHM_ORDER)} algoritma")

    started = time.perf_counter()
    main_data = run_specs(
        problem,
        build_specs(problem.n_items, config.ALGORITHM_ORDER),
        config.SEEDS,
        optimum.value,
        name="main",
        verbose=verbose,
        timing_repeats=config.TIMING_REPEATS,
    )
    main_data.save()

    if verbose:
        print("Bonus: GA-SA hibriti (aynı tohumlar, ana deneyden ayrı kayıt)")
    bonus_data = run_specs(
        problem,
        build_specs(problem.n_items, (config.BONUS_ALGORITHM,)),
        config.SEEDS,
        optimum.value,
        name="bonus_hybrid",
        verbose=verbose,
        timing_repeats=config.TIMING_REPEATS,
    )
    bonus_data.save()

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick_mode": config.QUICK_MODE,
        "environment": environment_info(),
        "problem": {
            "n_items": problem.n_items,
            "total_weight": problem.total_weight,
            "total_value": problem.total_value,
            "capacity_ratio": problem.capacity_ratio,
            "capacity": problem.capacity,
            "optimum_value_dp": optimum.value,
        },
        "design": {
            "iterations": config.ITERATIONS,
            "population_size": config.POPULATION_SIZE,
            "n_runs": len(config.SEEDS),
            "seeds": list(config.SEEDS),
            "pilot_seeds": list(config.PILOT_SEEDS),
            "timing_repeats_median": config.TIMING_REPEATS,
            "nominal_budget_pop_x_(iter+1)": config.EQUAL_BUDGET_EVALUATIONS,
        },
        "main_parameters": config.main_parameters(problem.n_items),
        "evaluations_per_run": {
            str(label): int(count)
            for runs in (main_data.runs, bonus_data.runs)
            for label, count in runs.groupby("label")["n_evaluations"].first().items()
        },
        "total_wallclock_s": time.perf_counter() - started,
    }
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (config.RAW_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    return main_data, bonus_data


if __name__ == "__main__":
    run_benchmark()
