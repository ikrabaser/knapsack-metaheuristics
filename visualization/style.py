"""Ortak grafik stili: doğrulanmış renk paleti, ikincil kodlamalar, kaydetme yardımcıları.

Renk kuralları (dataviz yöntemi):

* kategorik renkler **sabit sırayla** ve **varlığa bağlı** atanır (bir algoritma tüm
  grafiklerde aynı rengi taşır); palet ``validate_palette.js`` ile doğrulanmıştır
  (bitişik ikililerde CVD ΔE >= 9.1, normal görme ΔE >= 19.6),
* ışık yüzeyinde kontrastı 3:1 altında kalan renkler (aqua, sarı, magenta) için
  **ikincil kodlama** kullanılır: işaretçi biçimi, çizgi stili ve lejant,
* sıralı (ordinal) düzeyler (ör. parametre değerleri) tek renkli, açıktan koyuya
  bir rampayla gösterilir,
* metin asla seri rengiyle yazılmaz; ızgara ve eksenler arka planda kalır.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ekransız ortamda da çalışır
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

import config  # noqa: E402

# --- Kategorik palet (sabit sıra: mavi, turuncu, aqua, sarı, magenta) ------- #
ALGORITHM_COLORS: dict[str, str] = {
    "GA": "#2a78d6",
    "ACO": "#eb6834",
    "BPSO": "#1baf7a",
    "SA": "#eda100",
    "GA-SA": "#e87ba4",
}
ALGORITHM_MARKERS: dict[str, str] = {"GA": "o", "ACO": "s", "BPSO": "^", "SA": "D", "GA-SA": "P"}
ALGORITHM_LINESTYLES: dict[str, str] = {"GA": "-", "ACO": "--", "BPSO": "-.", "SA": ":", "GA-SA": (0, (5, 1, 1, 1))}  # type: ignore[dict-item]
ALGORITHM_LABELS: dict[str, str] = config.ALGORITHM_DISPLAY_NAMES

# --- Sıralı mavi rampa (ordinal düzeyler; en açık adım 250 >= 2:1 kontrast) -- #
ORDINAL_RAMP_3: tuple[str, ...] = ("#86b6ef", "#2a78d6", "#104281")  # 250, 450, 650
ORDINAL_RAMP_4: tuple[str, ...] = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # 250, 400, 550, 700
ORDINAL_RAMP_5: tuple[str, ...] = ("#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b")

# --- Mürekkep ve yüzey ------------------------------------------------------ #
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
OPTIMUM_COLOR = "#d03b3b"  # durum paleti: referans optimum çizgisi (seri rengi değil)

#: Metin-etiketli çubuk/nokta için seri rengi üzerinde okunur beyaz/siyah seçimi.
STATUS_NOTE = "Kırmızı kesikli çizgi: DP ile hesaplanan gerçek optimum."


def apply_style() -> None:
    """Akademik, sade ve okunaklı matplotlib stili uygular."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK_PRIMARY,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": SURFACE,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "legend.frameon": False,
            "legend.fontsize": 7,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,  # PDF'te gömülü TrueType (Türkçe karakterler korunur)
        }
    )


def save_figure(fig: Figure, name: str, directory: Path | None = None) -> list[Path]:
    """Şekli yüksek çözünürlüklü PNG (300 dpi) ve vektörel PDF olarak kaydeder."""
    directory = directory or config.FIGURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "pdf"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    return paths


def gap_axis(ax: plt.Axes, top: float = 10.0) -> None:
    """Optimalite farkı (%) ekseni: 0'ı da gösteren simetrik-log ölçek."""
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.6)
    ax.set_ylim(0, top)
    ticks = [t for t in (0, 0.01, 0.1, 1, 10, 100) if t <= top]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_locator(plt.NullLocator())


def label(algorithm: str) -> str:
    """Algoritma anahtarından görüntü adı."""
    return ALGORITHM_LABELS.get(algorithm, algorithm)
