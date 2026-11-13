"""Komut satırı giriş noktası: tüm deney hattını veya tek bir aşamayı çalıştırır.

Örnekler::

    python main.py                      # tüm aşamalar (testler dahil), gerçek ayarlarla
    python main.py --stage benchmark    # yalnızca ana deney
    python main.py --stage figures      # kayıtlı ham sonuçlardan grafikleri yeniden üret
    python main.py --quick              # hızlı duman testi (3 çalıştırma, 30 iterasyon;
                                        #   çıktılar results_quick/ klasörüne yazılır)

Aşamalar sırayla: tests, data, pilot, benchmark, sensitivity, extras, stats, figures, report.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

STAGES = ("tests", "data", "pilot", "benchmark", "sensitivity", "extras", "stats", "figures", "report")


def _run_stage(stage: str) -> None:
    """Tek bir aşamayı çalıştırır (modüller, ortam değişkenleri ayarlandıktan sonra içe aktarılır)."""
    if stage == "tests":
        code = subprocess.call([sys.executable, "-m", "pytest"], cwd=Path(__file__).parent)
        if code != 0:
            raise SystemExit(f"Birim testleri başarısız (çıkış kodu {code}); hat durduruldu.")
    elif stage == "data":
        from experiments.dataset_analysis import run_dataset_analysis

        run_dataset_analysis()
    elif stage == "pilot":
        from experiments.pilot_selection import run_pilot

        run_pilot()
    elif stage == "benchmark":
        from experiments.benchmark import run_benchmark

        run_benchmark()
    elif stage == "sensitivity":
        from experiments.parameter_sensitivity import run_sensitivity

        run_sensitivity()
    elif stage == "extras":
        from experiments.extra_experiments import run_all_extras

        run_all_extras()
    elif stage == "stats":
        from experiments.statistical_analysis import run_statistical_analysis

        run_statistical_analysis()
    elif stage == "figures":
        from visualization.plots import generate_all_figures

        generate_all_figures()
    elif stage == "report":
        from reports.build_report import build_report

        print("PDF:", build_report())
    else:  # pragma: no cover - argparse zaten kısıtlar
        raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description="0/1 Knapsack meta-sezgisel karşılaştırma hattı")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all", help="çalıştırılacak aşama")
    parser.add_argument("--quick", action="store_true", help="hızlı duman testi (sonuçlar raporlanmaz)")
    parser.add_argument("--skip-tests", action="store_true", help="'all' modunda birim testlerini atla")
    args = parser.parse_args()

    if args.quick:
        os.environ["KNAPSACK_QUICK"] = "1"
        os.environ.setdefault("KNAPSACK_RESULTS_DIR", str(Path(__file__).parent / "results_quick"))

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    if args.skip_tests and "tests" in stages:
        stages.remove("tests")
    started = time.perf_counter()
    for stage in stages:
        print(f"\n===== Aşama: {stage} =====", flush=True)
        if args.quick and stage == "report":
            print("Atlandı: hızlı modda (3 çalıştırma, 30 iterasyon) rapor metni denetimleri anlamlı değildir.")
            continue
        _run_stage(stage)
    print(f"\nTamamlandı ({time.perf_counter() - started:.1f} sn).")


if __name__ == "__main__":
    main()
