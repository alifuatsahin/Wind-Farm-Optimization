"""Shared plot styling: a single-hue sequential colormap for magnitude fields (replacing the..."""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

#: Serif stack for the paper figures; the fallbacks are the Linux metric-compatibles.
PAPER_SERIF = ["Times New Roman", "Nimbus Roman", "Liberation Serif",
               "Tinos", "DejaVu Serif"]

#: Base text size in points. Figures are drawn at their printed width, so this is
#: the size the reader sees; the paper body is 12 pt.
BASE_PT = 9.5

#: Page white. The figures sit on the white of a journal page, so anything off-white
#: reads as a grey panel once the PDF is printed; it is set in the rcParams below as
#: well as passed explicitly, so a figure that forgets `facecolor=SURFACE` still
#: comes out white rather than reverting to this module's old warm tint.
SURFACE = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_paper_style():
    """Set the serif family and the base text sizes for every figure in the paper."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "serif",
        "font.serif": PAPER_SERIF,
        "mathtext.fontset": "stix",
        "font.size": BASE_PT,
        "axes.titlesize": BASE_PT,
        "axes.labelsize": BASE_PT,
        "xtick.labelsize": BASE_PT - 1.0,
        "ytick.labelsize": BASE_PT - 1.0,
        "legend.fontsize": BASE_PT - 1.0,
        "figure.titlesize": BASE_PT + 1.0,
        "axes.formatter.use_mathtext": True,
    })


apply_paper_style()

SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
])

DIV_BLUE_RUST = LinearSegmentedColormap.from_list("div_blue_rust", [
    "#0d366b", "#184f95", "#256abf", "#5598e7", "#9ec5f4", "#cde2fb",
    SURFACE,
    "#f7ddc8", "#efc09b", "#e2a271", "#d0804b", "#b0562a", "#7d3a17",
])


def style_figure(fig):
    """Set the figure background to the shared chart surface color."""
    fig.set_facecolor(SURFACE)


def style_axes(ax, grid=True):
    """Apply shared chrome to a single axes: surface, recessive gridlines, dark legible text."""
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_PRIMARY, labelcolor=INK_PRIMARY)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    ax.title.set_color(INK_PRIMARY)
    if grid:
        ax.grid(color=GRID, linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)


def style_colorbar(cbar):
    """Apply shared chrome to a colorbar: no outline, dark legible label/ticks."""
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.label.set_color(INK_PRIMARY)
    cbar.ax.tick_params(colors=INK_PRIMARY, labelcolor=INK_PRIMARY)
