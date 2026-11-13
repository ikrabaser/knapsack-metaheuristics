"""Ek deneyler (ana deneyden ayrı): adillik denetimleri, ablasyonlar ve bonus analizler.

1. ``extra_equal_fe`` — **eşit değerlendirme bütçesi** denetimi: tüm algoritmalar
   tam 8040 nesne fonksiyonu değerlendirmesiyle (sert üst sınır) çalıştırılır.
2. ``extra_repair`` — **onarım stratejisi** ablasyonu (GA, BPSO, SA × 3 strateji):
   onarımın algoritmalara sezgisel bilgi sağlayıp sağlamadığını ölçer.
3. ``extra_aco_beta`` — ACO'da sezgisel bilgi ablasyonu (beta = 0, 1, 2).
4. ``extra_hybrid_share`` — GA-SA hibritinde SA payı ablasyonu
   (nesil başına 0/6/12/20/30 SA değerlendirmesi; 0 = saf GA).
5. ``extra_capacity`` — **kapasite oranı** analizi (%30, %50, %70, %90),
   her oran için DP ile gerçek optimum hesaplanır.
"""

from __future__ import annotations

import pandas as pd

import config
from experiments.runner import ExperimentData, RunSpec, prepare_problem, run_specs


def _spec(
    name: str, n_items: int, label: str | None = None, meta: dict | None = None, **overrides: object
) -> RunSpec:
    params = {**config.main_parameters(n_items)[name], "max_iterations": config.ITERATIONS, **overrides}
    return RunSpec(label or name, name, params, meta or {})


def run_equal_budget(verbose: bool = True) -> ExperimentData:
    """Tüm algoritmalar için eşit (sert üst sınırlı) değerlendirme bütçesi denetimi."""
    problem, optimum = prepare_problem(write_logs=False)
    n = problem.n_items
    specs = [
        _spec(
            name, n,
            max_iterations=10_000,  # bağlayıcı değil; durdurma ölçütü değerlendirme bütçesi
            max_evaluations=config.EQUAL_BUDGET_EVALUATIONS,
        )
        for name in (*config.ALGORITHM_ORDER, config.BONUS_ALGORITHM)
    ]
    data = run_specs(problem, specs, config.SEEDS, optimum.value, "extra_equal_fe",
                     keep_curves=False, verbose=verbose)
    data.save()
    return data


def run_repair_ablation(verbose: bool = True) -> ExperimentData:
    """GA, BPSO ve SA için üç onarım stratejisini karşılaştırır."""
    problem, optimum = prepare_problem(write_logs=False)
    n = problem.n_items
    specs = [
        _spec(name, n, label=f"{name}|{strategy}",
              meta={"strategy": strategy}, repair_strategy=strategy)
        for name in ("GA", "BPSO", "SA")
        for strategy in ("ratio", "random", "ratio_fill")
    ]
    data = run_specs(problem, specs, config.SEEDS, optimum.value, "extra_repair",
                     keep_curves=False, verbose=verbose)
    data.save()
    return data


def run_aco_beta_ablation(verbose: bool = True) -> ExperimentData:
    """ACO'da sezgisel bilgi üssünü (beta) değiştirir; beta = 0 sezgiselsiz ACO'dur."""
    problem, optimum = prepare_problem(write_logs=False)
    n = problem.n_items
    specs = [
        _spec("ACO", n, label=f"ACO|beta={beta:g}", meta={"beta": beta}, beta=beta)
        for beta in config.ACO_BETA_LEVELS
    ]
    data = run_specs(problem, specs, config.SEEDS, optimum.value, "extra_aco_beta", verbose=verbose)
    data.save()
    return data


def run_hybrid_share_ablation(verbose: bool = True) -> ExperimentData:
    """Hibritte nesil başına SA'ya ayrılan değerlendirme sayısını değiştirir (0 = saf GA)."""
    problem, optimum = prepare_problem(write_logs=False)
    n = problem.n_items
    specs = [
        _spec(
            "GA-SA", n, label=f"GA-SA|sa={share}", meta={"sa_evaluations": share},
            sa_evaluations_per_generation=share,
        )
        for share in config.HYBRID_SA_SHARES
    ]
    data = run_specs(problem, specs, config.SEEDS, optimum.value, "extra_hybrid_share", verbose=verbose)
    data.save()
    return data


def run_capacity_ratios(verbose: bool = True) -> pd.DataFrame:
    """Farklı kapasite oranlarında beş yöntemi karşılaştırır (her oranda DP optimumu)."""
    frames: list[pd.DataFrame] = []
    for ratio in config.CAPACITY_RATIOS:
        problem, optimum = prepare_problem(ratio, write_logs=False)
        specs = [
            _spec(name, problem.n_items)
            for name in (*config.ALGORITHM_ORDER, config.BONUS_ALGORITHM)
        ]
        if verbose:
            print(f"Kapasite oranı {ratio:.0%}: C={problem.capacity:.1f}, DP optimum={optimum.value:.0f}")
        data = run_specs(problem, specs, config.SEEDS, optimum.value, f"capacity_{int(ratio * 100)}",
                         keep_curves=False, verbose=False)
        data.runs["capacity_ratio"] = ratio
        frames.append(data.runs)
    combined = pd.concat(frames, ignore_index=True)
    combined["experiment"] = "extra_capacity"
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(config.RAW_DIR / "extra_capacity_runs.csv", index=False, encoding="utf-8")
    return combined


def run_all_extras(verbose: bool = True) -> None:
    """Tüm ek deneyleri sırayla çalıştırır."""
    for runner in (
        run_equal_budget,
        run_repair_ablation,
        run_aco_beta_ablation,
        run_hybrid_share_ablation,
        run_capacity_ratios,
    ):
        if verbose:
            print(f"\n== {runner.__name__} ==")
        runner(verbose=verbose)


if __name__ == "__main__":
    run_all_extras()
