"""PDF raporu için yardımcılar: Türkçe destekli fontlar, denklem görüntüleri, tablo/şekil bileşenleri.

* Fontlar: matplotlib ile birlikte gelen **DejaVu** ailesi (Türkçe karakterleri, Yunan
  harflerini ve matematik sembollerini içerir; ek kurulum gerektirmez).
* Denklemler: matplotlib *mathtext* ile 300 dpi saydam PNG olarak üretilip gömülür.
* Alt/üst simgeler: ReportLab ``<sub>``/``<super>`` etiketleri (Unicode alt simge yok).
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib.figure import Figure
from PIL import Image as PilImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import Flowable, Image, KeepTogether, Paragraph, Spacer, Table, TableStyle

# --------------------------------------------------------------------------- #
# Fontlar
# --------------------------------------------------------------------------- #
_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
_FONTS = {
    "Serif": "DejaVuSerif.ttf",
    "Serif-Bold": "DejaVuSerif-Bold.ttf",
    "Serif-Italic": "DejaVuSerif-Italic.ttf",
    "Serif-BoldItalic": "DejaVuSerif-BoldItalic.ttf",
    "Sans": "DejaVuSans.ttf",
    "Sans-Bold": "DejaVuSans-Bold.ttf",
    "Sans-Oblique": "DejaVuSans-Oblique.ttf",
    "Sans-BoldOblique": "DejaVuSans-BoldOblique.ttf",
    "Mono": "DejaVuSansMono.ttf",
}

INK = colors.HexColor("#1b1b1a")
INK_SOFT = colors.HexColor("#52514e")
ACCENT = colors.HexColor("#1c5cab")
RULE = colors.HexColor("#b9b8b0")
ZEBRA = colors.HexColor("#f4f3ef")


def register_fonts() -> None:
    """DejaVu fontlarını ReportLab'e kaydeder (tekrar çağrılması zararsızdır)."""
    for name, filename in _FONTS.items():
        if f"DV-{name}" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(f"DV-{name}", str(_FONT_DIR / filename)))
    pdfmetrics.registerFontFamily(
        "DV-Serif", normal="DV-Serif", bold="DV-Serif-Bold", italic="DV-Serif-Italic",
        boldItalic="DV-Serif-BoldItalic",
    )
    pdfmetrics.registerFontFamily(
        "DV-Sans", normal="DV-Sans", bold="DV-Sans-Bold", italic="DV-Sans-Oblique",
        boldItalic="DV-Sans-BoldOblique",
    )


def build_styles() -> dict[str, ParagraphStyle]:
    """Raporun paragraf stilleri."""
    register_fonts()
    body = ParagraphStyle("body", fontName="DV-Serif", fontSize=8.4, leading=11.1, alignment=TA_JUSTIFY,
                          textColor=INK, spaceAfter=2.4)
    return {
        "body": body,
        "body_left": ParagraphStyle("body_left", parent=body, alignment=TA_LEFT),
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=11, bulletIndent=2, spaceAfter=1.6),
        "h1": ParagraphStyle("h1", fontName="DV-Sans-Bold", fontSize=11.2, leading=13.5, textColor=ACCENT,
                             spaceBefore=5, spaceAfter=2.5, keepWithNext=1),
        "h2": ParagraphStyle("h2", fontName="DV-Sans-Bold", fontSize=9.0, leading=11.5, textColor=INK,
                             spaceBefore=3.5, spaceAfter=1.5, keepWithNext=1),
        "title": ParagraphStyle("title", fontName="DV-Sans-Bold", fontSize=19, leading=24, textColor=INK,
                                alignment=TA_LEFT, spaceAfter=5),
        "subtitle": ParagraphStyle("subtitle", fontName="DV-Sans", fontSize=11, leading=15, textColor=INK_SOFT,
                                   spaceAfter=10),
        "abstract": ParagraphStyle("abstract", parent=body, fontSize=8.8, leading=12.2, leftIndent=0),
        "caption": ParagraphStyle("caption", fontName="DV-Sans", fontSize=7.0, leading=8.8, textColor=INK_SOFT,
                                  alignment=TA_LEFT, spaceBefore=1, spaceAfter=3.5),
        "table_caption": ParagraphStyle("table_caption", fontName="DV-Sans-Bold", fontSize=7.8, leading=10,
                                        textColor=INK, spaceBefore=4, spaceAfter=2, keepWithNext=1),
        "cell": ParagraphStyle("cell", fontName="DV-Sans", fontSize=7.0, leading=8.6, textColor=INK),
        "cell_bold": ParagraphStyle("cell_bold", fontName="DV-Sans-Bold", fontSize=7.0, leading=8.6, textColor=INK),
        "cell_bold_right": ParagraphStyle("cell_bold_right", fontName="DV-Sans-Bold", fontSize=7.0, leading=8.6,
                                          textColor=INK, alignment=2),
        "cell_right": ParagraphStyle("cell_right", fontName="DV-Sans", fontSize=7.0, leading=8.6, textColor=INK,
                                     alignment=2),
        "note": ParagraphStyle("note", fontName="DV-Sans", fontSize=7.1, leading=9.0, textColor=INK_SOFT,
                               spaceBefore=1, spaceAfter=4),
        "ref": ParagraphStyle("ref", fontName="DV-Serif", fontSize=7.1, leading=8.9, leftIndent=11,
                              firstLineIndent=-11, textColor=INK, spaceAfter=1.0, alignment=TA_LEFT),
        "center": ParagraphStyle("center", parent=body, alignment=TA_CENTER),
    }


# --------------------------------------------------------------------------- #
# Denklemler
# --------------------------------------------------------------------------- #
_EQUATION_DPI = 300
_equation_cache: dict[tuple[str, float], tuple[bytes, float, float]] = {}


def _render_equation(latex: str, fontsize: float) -> tuple[bytes, float, float]:
    """LaTeX (mathtext) ifadesini saydam PNG'ye çevirir; (bayt, genişlik pt, yükseklik pt)."""
    key = (latex, fontsize)
    if key in _equation_cache:
        return _equation_cache[key]
    fig = Figure(figsize=(0.1, 0.1), dpi=_EQUATION_DPI)
    fig.patch.set_alpha(0.0)
    fig.text(0, 0, f"${latex}$", fontsize=fontsize, color="#1b1b1a",
             fontproperties=font_manager.FontProperties(family="DejaVu Serif"))
    buffer = io.BytesIO()
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    fig.savefig(buffer, format="png", dpi=_EQUATION_DPI, bbox_inches="tight", pad_inches=0.03, transparent=True)
    data = buffer.getvalue()
    width_px, height_px = PilImage.open(io.BytesIO(data)).size
    result = (data, width_px / _EQUATION_DPI * 72.0, height_px / _EQUATION_DPI * 72.0)
    _equation_cache[key] = result
    return result


def equation_image(latex: str, fontsize: float = 10.5) -> Image:
    """Denklemi ReportLab ``Image`` bileşeni olarak döndürür."""
    data, width_pt, height_pt = _render_equation(latex, fontsize)
    return Image(io.BytesIO(data), width=width_pt, height=height_pt)


class EquationCounter:
    """Denklem numaralandırıcı: ``(1)``, ``(2)``, ..."""

    def __init__(self, styles: dict[str, ParagraphStyle], text_width: float) -> None:
        self.styles = styles
        self.text_width = text_width
        self.number = 0

    def __call__(self, latex: str, fontsize: float = 10.5) -> Table:
        self.number += 1
        number_style = ParagraphStyle("eqno", parent=self.styles["body"], alignment=2, fontName="DV-Serif")
        table = Table(
            [["", equation_image(latex, fontsize), Paragraph(f"({self.number})", number_style)]],
            colWidths=[self.text_width * 0.06, self.text_width * 0.82, self.text_width * 0.12],
        )
        table.setStyle(TableStyle([
            ("ALIGN", (1, 0), (1, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        return table


# --------------------------------------------------------------------------- #
# Tablo ve şekil
# --------------------------------------------------------------------------- #
def make_table(
    rows: list[list[str]],
    col_widths: list[float],
    styles: dict[str, ParagraphStyle],
    header_rows: int = 1,
    bold_rows: tuple[int, ...] = (),
    numeric_from_col: int = 1,
    zebra: bool = True,
    highlight_cells: dict[tuple[int, int], colors.Color] | None = None,
) -> Table:
    """Booktabs tarzı (üst/alt çizgi, ince başlık çizgisi) tablo. Hücreler ``Paragraph``'tır."""
    cells: list[list[Paragraph]] = []
    for r, row in enumerate(rows):
        converted = []
        for c, value in enumerate(row):
            if r < header_rows or r in bold_rows:
                style = styles["cell_bold_right"] if c >= numeric_from_col else styles["cell_bold"]
            elif c >= numeric_from_col:
                style = styles["cell_right"]
            else:
                style = styles["cell"]
            converted.append(Paragraph(str(value), style))
        cells.append(converted)
    table = Table(cells, colWidths=col_widths, repeatRows=header_rows)
    commands = [
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, INK),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.5, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.2), ("RIGHTPADDING", (0, 0), (-1, -1), 3.2),
    ]
    if zebra:
        for r in range(header_rows, len(rows)):
            if (r - header_rows) % 2 == 1:
                commands.append(("BACKGROUND", (0, r), (-1, r), ZEBRA))
    for (r, c), color in (highlight_cells or {}).items():
        commands.append(("BACKGROUND", (c, r), (c, r), color))
    table.setStyle(TableStyle(commands))
    return table


class FigureNumbering:
    """Şekil/tablo numaralandırıcı ve başlık üretici."""

    def __init__(self, styles: dict[str, ParagraphStyle]) -> None:
        self.styles = styles
        self.figure_number = 0
        self.table_number = 0

    def figure(self, path: Path, width: float, caption: str) -> KeepTogether:
        """Görseli en-boy oranını koruyarak ``width`` genişliğinde ekler ve başlık yazar."""
        self.figure_number += 1
        with PilImage.open(path) as image:
            pixel_w, pixel_h = image.size
        picture = Image(str(path), width=width, height=width * pixel_h / pixel_w)
        caption_text = f"<b>Şekil {self.figure_number}.</b> {caption}"
        return KeepTogether([picture, Paragraph(caption_text, self.styles["caption"])])

    def _cell(self, path: Path, width: float, caption: str) -> list[Flowable]:
        """Tablo hücresi için [görsel, başlık] listesi (numara otomatik artar)."""
        self.figure_number += 1
        with PilImage.open(path) as image:
            pixel_w, pixel_h = image.size
        picture = Image(str(path), width=width, height=width * pixel_h / pixel_w)
        return [picture, Paragraph(f"<b>Şekil {self.figure_number}.</b> {caption}", self.styles["caption"])]

    def pair(self, left: tuple[Path, str], right: tuple[Path, str], total_width: float) -> Table:
        """İki şekli yan yana (her biri yarım genişlikte) yerleştirir."""
        cell_width = total_width / 2 - 4
        table = Table([[self._cell(left[0], cell_width, left[1]), self._cell(right[0], cell_width, right[1])]],
                      colWidths=[total_width / 2, total_width / 2])
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 0),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        return table

    def beside_text(self, path: Path, caption: str, text: list[Flowable], total_width: float,
                    figure_fraction: float = 0.52) -> Table:
        """Solda şekil, sağda metin bileşenleri."""
        figure_width = total_width * figure_fraction
        table = Table([[self._cell(path, figure_width - 6, caption), text]],
                      colWidths=[figure_width, total_width - figure_width])
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                   ("RIGHTPADDING", (0, 0), (0, 0), 6), ("RIGHTPADDING", (1, 0), (1, 0), 0),
                                   ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        return table

    def table_title(self, caption: str) -> Paragraph:
        self.table_number += 1
        return Paragraph(f"Tablo {self.table_number}. {caption}", self.styles["table_caption"])


class NumberedCanvas(pdf_canvas.Canvas):
    """Sayfa X / Y numarası ve ince üstbilgi çizen tuval (kapak sayfasında üstbilgi yok)."""

    header_text = "0/1 Knapsack – Meta-Sezgisel Karşılaştırma"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: list[dict] = []

    def showPage(self) -> None:  # noqa: N802 (ReportLab API)
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_chrome(total)
            super().showPage()
        super().save()

    def _draw_chrome(self, total: int) -> None:
        width, height = self._pagesize
        self.setFont("DV-Sans", 7.2)
        self.setFillColor(INK_SOFT)
        if self._pageNumber > 1:
            self.drawString(2.0 * cm, height - 1.25 * cm, self.header_text)
            self.setStrokeColor(RULE)
            self.setLineWidth(0.4)
            self.line(2.0 * cm, height - 1.38 * cm, width - 2.0 * cm, height - 1.38 * cm)
        self.drawRightString(width - 2.0 * cm, 1.05 * cm, f"Sayfa {self._pageNumber} / {total}")


__all__ = [
    "ACCENT", "EquationCounter", "FigureNumbering", "Flowable", "INK", "INK_SOFT", "KeepTogether",
    "NumberedCanvas", "Paragraph", "Spacer", "Table", "TableStyle", "build_styles", "cm",
    "equation_image", "make_table", "register_fonts",
]
