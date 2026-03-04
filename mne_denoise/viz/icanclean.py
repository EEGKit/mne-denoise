"""Visualization hooks for ICanClean QC.

This module provides publication-ready QC plots for inspecting ICanClean
outputs: correlation score distributions, rejected components, and
before/after comparisons.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure


def plot_correlation_scores(
    icanclean: Any,
    *,
    ax: "plt.Axes | None" = None,
    show: bool = True,
) -> "Figure":
    """Plot per-window canonical correlation scores.

    Displays the :math:`R^2` values for each canonical component across all
    sliding windows processed by :class:`~mne_denoise.icanclean.ICanClean`.

    Parameters
    ----------
    icanclean : ICanClean
        A fitted ``ICanClean`` instance (must have ``correlations_``).
    ax : matplotlib Axes | None
        Axes to plot on. If ``None`` a new figure is created.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : Figure
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    corr = icanclean.correlations_
    if corr.size == 0:
        raise ValueError("No correlations stored — was ICanClean fitted?")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    n_windows, n_comp = corr.shape
    for j in range(n_comp):
        vals = corr[:, j]
        valid = ~np.isnan(vals)
        ax.plot(np.where(valid)[0], vals[valid], lw=0.8, alpha=0.7, label=f"CC {j+1}")

    # Threshold line
    thr = icanclean.threshold
    if thr != "auto":
        ax.axhline(float(thr), color="red", ls="--", lw=1.2, label=f"threshold={thr}")

    ax.set_xlabel("Window index")
    ax.set_ylabel("$R^2$")
    ax.set_title("ICanClean — Canonical Correlation Scores per Window")
    ax.set_ylim(-0.02, 1.02)
    if n_comp <= 10:
        ax.legend(fontsize=7, ncols=min(n_comp + 1, 5))
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_removal_summary(
    icanclean: Any,
    *,
    ax: "plt.Axes | None" = None,
    show: bool = True,
) -> "Figure":
    """Plot a bar chart of components removed per window.

    Parameters
    ----------
    icanclean : ICanClean
        A fitted ``ICanClean`` instance.
    ax : matplotlib Axes | None
        Axes to plot on.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : Figure
    """
    import matplotlib.pyplot as plt

    n_removed = icanclean.n_removed_
    if n_removed is None or len(n_removed) == 0:
        raise ValueError("No removal data — was ICanClean fitted?")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure

    ax.bar(range(len(n_removed)), n_removed, color="#2E5BFF", alpha=0.8, width=1.0)
    ax.axhline(n_removed.mean(), color="red", ls="--", lw=1, label=f"mean={n_removed.mean():.1f}")
    ax.set_xlabel("Window index")
    ax.set_ylabel("Components removed")
    ax.set_title("ICanClean — Components Removed per Window")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_psd_comparison(
    data_before: np.ndarray,
    data_after: np.ndarray,
    sfreq: float,
    *,
    channels: list[int] | None = None,
    fmax: float = 100.0,
    ax: "plt.Axes | None" = None,
    show: bool = True,
) -> "Figure":
    """Plot before/after PSD comparison.

    Parameters
    ----------
    data_before : ndarray, shape (n_channels, n_times)
        Data before cleaning.
    data_after : ndarray, shape (n_channels, n_times)
        Data after cleaning.
    sfreq : float
        Sampling frequency.
    channels : list of int | None
        Channel indices to average over. If ``None``, all channels are used.
    fmax : float
        Maximum frequency to display.
    ax : matplotlib Axes | None
        Axes to plot on.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : Figure
    """
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    if channels is not None:
        data_before = data_before[channels]
        data_after = data_after[channels]

    nfft = min(2048, data_before.shape[1])
    f, psd_before = welch(data_before, fs=sfreq, nperseg=nfft, axis=-1)
    _, psd_after = welch(data_after, fs=sfreq, nperseg=nfft, axis=-1)

    # Average across channels
    psd_before = psd_before.mean(axis=0)
    psd_after = psd_after.mean(axis=0)

    freq_mask = f <= fmax

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    ax.semilogy(f[freq_mask], psd_before[freq_mask], label="Before", alpha=0.8)
    ax.semilogy(f[freq_mask], psd_after[freq_mask], label="After", alpha=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title("ICanClean — Before / After PSD")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_timeseries_comparison(
    data_before: np.ndarray,
    data_after: np.ndarray,
    sfreq: float,
    *,
    channel: int = 0,
    tmin: float = 0.0,
    tmax: float | None = None,
    ax: "plt.Axes | None" = None,
    show: bool = True,
) -> "Figure":
    """Plot before/after time-series overlay for a single channel.

    Parameters
    ----------
    data_before : ndarray, shape (n_channels, n_times)
        Data before cleaning.
    data_after : ndarray, shape (n_channels, n_times)
        Data after cleaning.
    sfreq : float
        Sampling frequency.
    channel : int
        Channel index to plot.
    tmin : float
        Start time in seconds.
    tmax : float | None
        End time in seconds. ``None`` plots to the end.
    ax : matplotlib Axes | None
        Axes to plot on.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    fig : Figure
    """
    import matplotlib.pyplot as plt

    n_times = data_before.shape[1]
    t = np.arange(n_times) / sfreq
    if tmax is None:
        tmax = t[-1]

    mask = (t >= tmin) & (t <= tmax)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.figure

    ax.plot(t[mask], data_before[channel, mask], label="Before", alpha=0.6, lw=0.8)
    ax.plot(t[mask], data_after[channel, mask], label="After", alpha=0.8, lw=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"ICanClean — Channel {channel} Time Series")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    return fig
