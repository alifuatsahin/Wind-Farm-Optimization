"""Shared plot styling: a single-hue sequential colormap for magnitude fields
(replacing the rainbow 'turbo'), and consistent chrome (ink/grid/spine colors)
so wake and optimization figures read as one family."""
from matplotlib.colors import LinearSegmentedColormap

SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
])

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


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
