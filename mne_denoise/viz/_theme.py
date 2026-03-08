"""Shared publication-quality theme for mne-denoise figures.

Provides a colorblind-safe color palette, font-size constants, and
helper functions that any plotting function can use to produce
consistent, journal-ready figures.

Usage inside a plotting function::

    from ._theme import COLORS, FONTS, style_axes, pub_figure, pub_legend

Usage from a notebook (context-managed, restores rcParams on exit)::

    from mne_denoise.viz import use_style

    with use_style("paper"):
        ...

Or to set the style globally (use only in notebooks, NOT in libraries)::

    from mne_denoise.viz import set_pub_style

    set_pub_style()

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# =====================================================================
# Colorblind-safe Wong palette (Nature Methods, 2011)
# =====================================================================
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "cyan": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#BBBBBB",
    "light_gray": "#DDDDDD",
    "dark": "#333333",
}

# Semantic aliases — use these in plotting code
COLORS["primary"] = COLORS["blue"]
COLORS["secondary"] = COLORS["orange"]
COLORS["accent"] = COLORS["red"]
COLORS["success"] = COLORS["green"]
COLORS["muted"] = COLORS["gray"]
COLORS["text"] = COLORS["dark"]
COLORS["before"] = COLORS["dark"]  # PSD / signal before cleaning
COLORS["after"] = COLORS["green"]  # PSD / signal after cleaning
COLORS["line_marker"] = COLORS["red"]
COLORS["no_artifact"] = COLORS["gray"]
COLORS["edge"] = COLORS["dark"]  # bar / scatter edge color
COLORS["placeholder"] = "#999999"  # "no data" text colour
COLORS["separator"] = "#e0e0e0"  # subtle separator lines
COLORS["label_secondary"] = "#555555"  # secondary stat labels
COLORS["highlight"] = COLORS["yellow"]  # best-method star etc.

# =====================================================================
# Method colors — consistent palette for denoising method comparisons
# =====================================================================
METHOD_COLORS = {
    "original": COLORS["dark"],
    "before": COLORS["dark"],
    "after": COLORS["green"],
    "dss": COLORS["blue"],
    "zapline": COLORS["orange"],
    "dss_smooth": COLORS["cyan"],
    "dss_segment": COLORS["purple"],
    "clean": COLORS["green"],
}


def _get_color(key, fallback=None):
    """Return a color from COLORS or METHOD_COLORS by key.

    Parameters
    ----------
    key : str
        A key into :data:`COLORS` or :data:`METHOD_COLORS`.
    fallback : str | None
        Returned if *key* is not found in either dict.
        Defaults to ``COLORS["dark"]``.

    Returns
    -------
    str
        Hex color string.
    """
    if key in COLORS:
        return COLORS[key]
    if key in METHOD_COLORS:
        return METHOD_COLORS[key]
    return fallback if fallback is not None else COLORS["dark"]


# =====================================================================
# Font sizes (pt) — tuned for single-column journal figures (~3.5 in)
# and two-column figures (~7 in).
# =====================================================================
FONTS = {
    "suptitle": 13,
    "title": 10,
    "label": 9,
    "tick": 8,
    "legend": 7.5,
    "annotation": 7.5,
}

# =====================================================================
# Default figure parameters
# =====================================================================
DEFAULT_DPI = 200
DEFAULT_FIGSIZE = (11, 8.5)  # landscape letter


# =====================================================================
# Axes styling
# =====================================================================
def style_axes(ax, grid=False):
    """Apply publication styling to a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to style.
    grid : bool
        If True, add a subtle background grid.
    """
    ax.tick_params(
        labelsize=FONTS["tick"],
        direction="out",
        length=3,
        width=0.5,
        color="#999999",
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_linewidth(0.5)
        ax.spines[sp].set_color("#666666")
    if grid:
        ax.grid(True, alpha=0.12, linewidth=0.4, color="#888888")
        ax.set_axisbelow(True)


# =====================================================================
# Figure factory
# =====================================================================
def pub_figure(nrows=1, ncols=1, figsize=None, dpi=None, gridspec_kw=None, **kwargs):
    """Create a figure + axes with publication defaults.

    Thin wrapper around ``plt.subplots`` that applies the theme's
    default DPI, white background, and calls :func:`style_axes` on
    every axes.

    Parameters
    ----------
    nrows, ncols : int
        Subplot grid dimensions.
    figsize : tuple | None
        (width, height) in inches.  Defaults to :data:`DEFAULT_FIGSIZE`.
    dpi : int | None
        Resolution. Defaults to :data:`DEFAULT_DPI`.
    gridspec_kw : dict | None
        Forwarded to ``plt.subplots``.
    **kwargs
        Additional keyword arguments forwarded to ``plt.subplots``.

    Returns
    -------
    fig : Figure
    axes : Axes or ndarray of Axes
    """
    if figsize is None:
        figsize = DEFAULT_FIGSIZE
    if dpi is None:
        dpi = DEFAULT_DPI

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw=gridspec_kw,
        **kwargs,
    )
    fig.set_facecolor("white")

    # Apply style_axes to every axes object
    import numpy as np

    for ax in np.atleast_1d(axes).flat:
        style_axes(ax)

    return fig, axes


# =====================================================================
# Legend helper
# =====================================================================
def pub_legend(ax, **kwargs):
    """Add a clean, minimal legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    **kwargs
        Overrides forwarded to ``ax.legend()``.

    Returns
    -------
    legend : Legend
    """
    defaults = {
        "fontsize": FONTS["legend"],
        "frameon": True,
        "fancybox": False,
        "edgecolor": "#cccccc",
    }
    defaults.update(kwargs)
    return ax.legend(**defaults)


# =====================================================================
# Internal helpers for plot functions
# =====================================================================
def _apply_style(ax, grid=False):
    """Apply publication styling to one or more axes.

    This is the single entry-point that every ``plot_*`` function should
    call to ensure consistent appearance.

    Parameters
    ----------
    ax : Axes | ndarray of Axes
        One or more matplotlib axes.
    grid : bool
        Whether to enable a subtle background grid.
    """
    import numpy as np

    for a in np.atleast_1d(ax).flat:
        style_axes(a, grid=grid)


def _finalize_fig(fig, show=True, fname=None, tight=True):
    """Apply tight layout, optionally save, and/or show a figure.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure.
    show : bool
        If True, call ``plt.show()``.
    fname : str | Path | None
        If given, save the figure to this path.
    tight : bool
        If True (default), call ``fig.tight_layout()``.

    Returns
    -------
    fig : Figure
    """
    if tight:
        with contextlib.suppress(Exception):
            fig.tight_layout()
    if fname is not None:
        fname = Path(fname)
        fname.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname, dpi=300, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    elif fname is not None:
        # Figure was saved to disk; close to free memory.
        plt.close(fig)
    # If show=False and fname=None the caller owns the figure.
    return fig


# =====================================================================
# rcParams dict (shared between set_pub_style and use_style)
# =====================================================================
_PUB_RC = {
    # Font sizes
    "font.size": FONTS["label"],
    "axes.titlesize": FONTS["title"],
    "axes.labelsize": FONTS["label"],
    "xtick.labelsize": FONTS["tick"],
    "ytick.labelsize": FONTS["tick"],
    "legend.fontsize": FONTS["legend"],
    "figure.titlesize": FONTS["suptitle"],
    # Spines
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#666666",
    # Ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.color": "#999999",
    "ytick.color": "#999999",
    # Grid
    "axes.grid": False,
    "grid.alpha": 0.12,
    "grid.linewidth": 0.4,
    "grid.color": "#888888",
    # Figure
    "figure.facecolor": "white",
    "figure.dpi": DEFAULT_DPI,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    # Legend
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "#cccccc",
    # Lines
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
}


# =====================================================================
# Context-manager style application (recommended for library use)
# =====================================================================
@contextlib.contextmanager
def use_style(style="paper"):
    """Context manager that temporarily applies the mne-denoise theme.

    On exit, matplotlib rcParams are restored to their previous values.
    This is safe for library code and tests because it never mutates
    global state permanently.

    Parameters
    ----------
    style : str
        Currently only ``"paper"`` is supported.

    Examples
    --------
    >>> from mne_denoise.viz import use_style
    >>> with use_style("paper"):
    ...     fig, ax = plt.subplots()
    ...     ax.plot([0, 1], [0, 1])
    """
    if style != "paper":
        raise ValueError(f"Unknown style {style!r}; only 'paper' is supported.")
    with mpl.rc_context(rc=_PUB_RC):
        yield


# =====================================================================
# Global rcParams setter (opt-in, for notebooks only)
# =====================================================================
def set_pub_style():
    """Apply the mne-denoise publication theme to matplotlib rcParams.

    Call this once at the top of a notebook to ensure *all* subsequent
    figures — including those from MNE-Python's own plotting functions —
    share the same clean look.

    .. warning::

        This mutates **global** matplotlib state.  Prefer
        :func:`use_style` in library code or tests.  ``set_pub_style``
        is intended for interactive notebook sessions only.
    """
    plt.rcParams.update(_PUB_RC)
