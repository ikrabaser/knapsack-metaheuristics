"""Türkçe akademik PDF raporunu üretir (``reports/final_report.pdf``).

Rapordaki tüm sayılar ``results/`` altındaki gerçek deney çıktılarından okunur.
Çalıştırma::

    python main.py --stage report        # veya: python -m reports.build_report

Öğrenci bilgileri ``reports/sections_static.py`` içindeki ``STUDENT_INFO`` sözlüğünden düzenlenir.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Spacer

import config
from reports.pdf_helpers import EquationCounter, FigureNumbering, NumberedCanvas, build_styles, register_fonts
from reports.report_data import ReportData
from reports.sections_extra import section_extras
from reports.sections_final import abstract_and_keywords, section_conclusion, section_discussion, section_references
from reports.sections_results import section_results, section_sensitivity
from reports.sections_static import Ctx, cover, section_algorithms, section_intro, section_method

MIN_PAGES, MAX_PAGES = 5, 10


def build_report(output: Path | None = None) -> Path:
    """Raporu üretir ve sayfa sayısını denetler (ödev: 5–10 sayfa)."""
    output = output or config.REPORT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    register_fonts()
    styles = build_styles()
    data = ReportData.load()
    text_width = A4[0] - 3.6 * cm - 12  # ReportLab çerçeve iç boşluğu (2 x 6 pt)
    ctx = Ctx(styles, EquationCounter(styles, text_width), FigureNumbering(styles), text_width, data)

    abstract, keywords = abstract_and_keywords(ctx)
    story = []
    story += cover(ctx, abstract, keywords)
    story += section_intro(ctx)
    story += section_algorithms(ctx)
    story += section_method(ctx)
    story += section_results(ctx)
    story += section_sensitivity(ctx)
    story += section_extras(ctx)
    story += section_discussion(ctx)
    story += section_conclusion(ctx)
    story += section_references(ctx)

    document = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.7 * cm,
        bottomMargin=1.5 * cm,
        title="Meta-Sezgisel Optimizasyon Algoritmalarının 0/1 Knapsack Problemi Üzerinde Karşılaştırmalı Analizi",
        subject="GA, ACO, Binary PSO, SA ve GA-SA hibriti", keywords="knapsack, GA, ACO, PSO, SA",
    )
    document.build(story, canvasmaker=NumberedCanvas)

    from pypdf import PdfReader

    n_pages = len(PdfReader(str(output)).pages)
    if not MIN_PAGES <= n_pages <= MAX_PAGES:
        raise AssertionError(f"Rapor {n_pages} sayfa; ödev aralığı {MIN_PAGES}–{MAX_PAGES} sayfadır.")
    print(f"Rapor yazıldı: {output} ({n_pages} sayfa)")
    return output


if __name__ == "__main__":
    build_report()
