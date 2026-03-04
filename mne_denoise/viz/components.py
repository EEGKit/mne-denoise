"""Visualization for denoising components.

This module contains reusable component-level plots that work across DSS,
ZapLine, and related estimators exposing component scores, patterns, or
sources.

This module contains:
1. Component score and selection visualizations.
2. Spatial pattern visualizations with topomap or line-plot fallback.
3. Component source summaries in time, image, and spectrogram views.

Authors: Sina Esmaeili <sina.esmaeili@umontreal.ca>
         Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.gridspec import GridSpec
from mne.time_frequency import tfr_array_multitaper

from ._utils import _get_components, _get_info, _get_patterns, _get_scores
from .theme import (
    COLORS,
    FONTS,
    _finalize_fig,
    style_axes,
    themed_figure,
    themed_legend,
)


def _resolve_component_indices(
    n_components,
    n_available,
    *,
    default_max,
):
    """Normalize component selection to an explicit list of indices."""
    if n_components is None:
        return list(range(min(default_max, n_available)))

    if isinstance(n_components, int):
        if n_components < 0:
            raise ValueError("n_components must be non-negative.")
        return list(range(min(n_components, n_available)))

    indices = [int(idx) for idx in n_components]
    invalid = [idx for idx in indices if idx < 0 or idx >= n_available]
    if invalid:
        raise ValueError(f"Component indices out of range: {invalid}")
    return indices


def _get_time_axis(data, n_times, sfreq=None):
    """Return a display-ready time axis and x-axis label."""
    if data is not None and hasattr(data, "times"):
        times = np.asarray(data.times)
        if times.shape[0] == n_times:
            return times, "Time (s)"

    if sfreq is not None:
        return np.arange(n_times) / sfreq, "Time (s)"

    return np.arange(n_times), "Time (samples)"


def _get_topomap_picks(info):
    """Pick a single channel type suitable for topomap plotting."""
    ch_types_dict = mne.channel_indices_by_type(info)
    priority = (
        "grad",
        "mag",
        "eeg",
        "seeg",
        "ecog",
        "hbo",
        "hbr",
        "fnirs_cw_amplitude",
    )

    for ch_type in priority:
        idxs = ch_types_dict.get(ch_type, [])
        if len(idxs) > 0:
            return idxs

    for idxs in ch_types_dict.values():
        if len(idxs) > 0:
            return idxs

    return None


def plot_component_score_curve(
    estimator,
    mode="raw",
    ax=None,
    show=True,
    fname=None,
):
    """Plot component scores for a fitted estimator.

    Parameters
    ----------
    estimator : object
        Fitted estimator exposing ``eigenvalues_`` or ``scores_``.
    mode : {'raw', 'cumulative', 'ratio'}
        Score display mode.
    ax : matplotlib.axes.Axes | None
        Target axes. If None, a new figure is created.
    show : bool
        If True, show the figure.
    fname : path-like | None
        Optional output path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    """
    valid_modes = {"raw", "cumulative", "ratio"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}")

    scores = _get_scores(estimator)
    if scores is None:
        raise ValueError("Estimator does not expose component scores.")

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Component scores must be a non-empty 1D array.")

    if ax is None:
        fig, ax = themed_figure(figsize=(7, 4))
    else:
        fig = ax.figure

    x = np.arange(1, scores.size + 1)
    if mode == "cumulative":
        y = np.cumsum(scores)
        y = y / y[-1]
        ylabel = "Cumulative Score (Normalized)"
    elif mode == "ratio":
        y = scores
        ylabel = "Power Ratio"
    else:
        y = scores
        ylabel = "Score / Eigenvalue"

    ax.plot(
        x,
        y,
        ".-",
        color=COLORS["primary"],
        linewidth=1.6,
        markersize=5,
        label="Scores",
    )

    if mode != "cumulative":
        mean_score = np.mean(scores)
        ax.axhline(
            mean_score,
            color=COLORS["muted"],
            linestyle="--",
            linewidth=0.9,
            label=f"Mean ({mean_score:.3g})",
        )

        n_selected = getattr(estimator, "n_selected_", None)
        if n_selected is None:
            n_selected = getattr(estimator, "n_removed_", None)
        if n_selected is not None and 0 < n_selected < scores.size:
            ax.axvline(
                n_selected + 0.5,
                color=COLORS["accent"],
                linestyle="--",
                linewidth=1.0,
                label=f"Cutoff ({n_selected})",
            )

        themed_legend(ax, loc="best")

    ax.set_xlabel("Component")
    ax.set_ylabel(ylabel)
    ax.set_title("Component Scores")
    style_axes(ax, grid=True)

    return _finalize_fig(fig, show=show, fname=fname)


def plot_component_patterns(
    estimator,
    info=None,
    n_components=None,
    ax=None,
    show=True,
    fname=None,
):
    """Plot spatial component patterns.

    When compatible MNE channel information is available, the patterns are
    rendered as topomaps. Otherwise, the function falls back to plotting the
    selected component weights across channels on a standard axes.

    Parameters
    ----------
    estimator : object
        Fitted estimator exposing ``patterns_``.
    info : mne.Info | None
        Measurement info. If None, try to resolve it from the estimator.
    n_components : int | sequence of int | None
        Components to plot. If an int, plot the first ``n_components``.
    ax : matplotlib.axes.Axes | None
        Optional target axes. Supported only for the line-plot fallback or
        when rendering a single topomap.
    show : bool
        If True, show the figure.
    fname : path-like | None
        Optional output path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure handle.
    """
    patterns = np.asarray(_get_patterns(estimator))
    if patterns.ndim != 2:
        raise ValueError(
            "patterns_ must be a 2D array of shape (n_channels, n_components)."
        )

    indices = _resolve_component_indices(
        n_components,
        patterns.shape[1],
        default_max=6,
    )
    if not indices:
        raise ValueError("No components selected for plotting.")

    info = _get_info(estimator, info)
    picks = _get_topomap_picks(info) if info is not None else None
    palette = [
        COLORS["primary"],
        COLORS["accent"],
        COLORS["secondary"],
        COLORS["success"],
        COLORS["purple"],
        COLORS["cyan"],
    ]

    if info is not None and picks is not None:
        topo_info = mne.pick_info(info, picks)
        if ax is not None:
            if len(indices) != 1:
                raise ValueError("ax can only be used when plotting a single topomap.")
            fig = ax.figure
            mne.viz.plot_topomap(
                patterns[picks, indices[0]],
                topo_info,
                axes=ax,
                show=False,
                contours=4,
            )
            ax.set_title(f"Comp {indices[0]}")
            return _finalize_fig(fig, show=show, fname=fname)

        n_show = len(indices)
        n_cols = min(4, n_show)
        n_rows = int(np.ceil(n_show / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3 * n_cols, 3 * n_rows),
            dpi=200,
            squeeze=False,
        )
        fig.set_facecolor("white")
        flat_axes = axes.ravel()

        for plot_ax, comp_idx in zip(flat_axes, indices):
            mne.viz.plot_topomap(
                patterns[picks, comp_idx],
                topo_info,
                axes=plot_ax,
                show=False,
                contours=4,
            )
            plot_ax.set_title(
                f"Comp {comp_idx}",
                fontsize=FONTS["tick"],
                color=palette[indices.index(comp_idx) % len(palette)],
            )

        for plot_ax in flat_axes[len(indices) :]:
            plot_ax.axis("off")

        fig.suptitle(
            "Component Patterns", fontsize=FONTS["title"], fontweight="semibold"
        )
        return _finalize_fig(fig, show=show, fname=fname)

    if ax is None:
        fig, ax = themed_figure(figsize=(8, 4.5))
    else:
        fig = ax.figure

    for i, comp_idx in enumerate(indices):
        ax.plot(
            patterns[:, comp_idx],
            marker="o",
            markersize=4,
            linewidth=1.3,
            alpha=0.85,
            color=palette[i % len(palette)],
            label=f"Comp {comp_idx}",
        )

    ax.axhline(0, color=COLORS["muted"], linestyle="-", alpha=0.35)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Pattern Weight")
    ax.set_title("Component Patterns")
    style_axes(ax, grid=True)
    themed_legend(ax, loc="best")

    return _finalize_fig(fig, show=show, fname=fname)


def plot_component_summary(
    estimator,
    data=None,
    info=None,
    n_components=None,
    show=True,
    plot_ci=True,
    fname=None,
):
    """Plot a compact per-component summary dashboard.

    Each selected component gets one row with spatial pattern, time course,
    and power spectral density.
    """
    info = _get_info(estimator, info)
    patterns = np.asarray(_get_patterns(estimator))
    sources = _get_components(estimator, data)

    indices = _resolve_component_indices(
        n_components,
        patterns.shape[1],
        default_max=5,
    )
    if not indices:
        raise ValueError("No components selected for plotting.")

    fig = plt.figure(figsize=(12, 3 * len(indices)), constrained_layout=True)
    gs = GridSpec(len(indices), 3, figure=fig, width_ratios=[1, 2, 1])
    sfreq = info["sfreq"] if info is not None else 1000.0
    picks = _get_topomap_picks(info) if info is not None else None
    times_template, time_label = _get_time_axis(data, sources.shape[1], sfreq=sfreq)

    for row_idx, comp_idx in enumerate(indices):
        ax_topo = fig.add_subplot(gs[row_idx, 0])
        if info is not None and picks is not None:
            topo_info = mne.pick_info(info, picks)
            topo_data = patterns[picks, comp_idx]
            mne.viz.plot_topomap(topo_data, topo_info, axes=ax_topo, show=False)
            ax_topo.set_title(f"Comp {comp_idx} Pattern")
        else:
            ax_topo.text(0.5, 0.5, "No topomap info", ha="center", va="center")
            ax_topo.set_axis_off()

        ax_time = fig.add_subplot(gs[row_idx, 1])
        if sources.ndim == 3:
            comp_data = sources[comp_idx]
            mean_tc = comp_data.mean(axis=1)
            ax_time.plot(times_template, mean_tc, label="Mean", color=COLORS["before"])

            if plot_ci:
                std_tc = comp_data.std(axis=1) / np.sqrt(comp_data.shape[1])
                ax_time.fill_between(
                    times_template,
                    mean_tc - 2 * std_tc,
                    mean_tc + 2 * std_tc,
                    color=COLORS["muted"],
                    alpha=0.3,
                    label="95% CI (SEM)",
                )
                themed_legend(ax_time, loc="best")
        else:
            comp_data = sources[comp_idx]
            ax_time.plot(times_template, comp_data, color=COLORS["before"])

        ax_time.set_title(f"Comp {comp_idx} Time Course")
        ax_time.set_xlabel(time_label)
        style_axes(ax_time, grid=True)

        ax_psd = fig.add_subplot(gs[row_idx, 2])
        if sources.ndim == 3:
            d_flat = sources[comp_idx].T
        else:
            d_flat = sources[comp_idx][np.newaxis, :]

        psd_spec, freqs = mne.time_frequency.psd_array_welch(
            d_flat,
            sfreq=sfreq,
            fmin=0,
            fmax=np.inf,
            n_fft=min(2048, d_flat.shape[-1]),
            verbose=False,
        )
        mean_psd = np.mean(psd_spec, axis=0)

        ax_psd.semilogy(freqs, mean_psd, color=COLORS["primary"])
        ax_psd.set_title("PSD")
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_xlim(0, min(100, sfreq / 2))
        style_axes(ax_psd, grid=True)

    return _finalize_fig(fig, show=show, fname=fname)


def plot_component_epochs_image(
    estimator,
    data=None,
    n_components=None,
    show=True,
    fname=None,
):
    """Plot component activity as an epoch-by-time image."""
    sources = _get_components(estimator, data)
    if sources.ndim == 2:
        sources = sources[:, :, np.newaxis]
    if sources.ndim != 3:
        raise ValueError("Component sources must be 2D or 3D.")

    indices = _resolve_component_indices(
        n_components,
        sources.shape[0],
        default_max=5,
    )
    if not indices:
        raise ValueError("No components selected for plotting.")

    fig, axes = plt.subplots(
        len(indices),
        1,
        figsize=(8, 2 * len(indices)),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, comp_idx in zip(axes, indices):
        img = sources[comp_idx].T
        ax.imshow(img, aspect="auto", origin="lower", cmap="RdBu_r")
        ax.set_title(f"Comp {comp_idx}")
        ax.set_ylabel("Epochs")

    axes[-1].set_xlabel("Time (samples)")
    return _finalize_fig(fig, show=show, fname=fname)


def plot_component_time_series(
    estimator,
    data=None,
    n_components=None,
    show=True,
    ax=None,
    fname=None,
):
    """Plot stacked component time series with standardized vertical offsets."""
    sources = _get_components(estimator, data)
    scores = _get_scores(estimator)

    if sources.ndim == 3:
        sources = sources.mean(axis=2)

    indices = _resolve_component_indices(
        n_components,
        sources.shape[0],
        default_max=20,
    )
    if not indices:
        raise ValueError("No components selected for plotting.")

    if ax is None:
        fig, ax = themed_figure(figsize=(10, max(4.0, len(indices) * 0.5)))
    else:
        fig = ax.figure

    time_axis, time_label = _get_time_axis(data, sources.shape[1], sfreq=None)
    x_min = float(time_axis[0])
    x_max = float(time_axis[-1])
    x_pad = 0.03 * (x_max - x_min if x_max != x_min else 1.0)
    label_x = x_max + x_pad * 0.25
    palette = [
        COLORS["primary"],
        COLORS["accent"],
        COLORS["secondary"],
        COLORS["success"],
        COLORS["purple"],
        COLORS["cyan"],
    ]
    offset_step = 3.0

    for row_idx, comp_idx in enumerate(indices):
        comp = sources[comp_idx]
        std = np.std(comp)
        if std < 1e-15:
            std = 1.0
        comp_norm = comp / std
        offset = -row_idx * offset_step
        color = palette[row_idx % len(palette)]

        ax.plot(time_axis, comp_norm + offset, color=color, linewidth=1.5)

        label = f"Comp {comp_idx}"
        if scores is not None and comp_idx < len(scores):
            label += f" (λ={scores[comp_idx]:.2f})"
        ax.text(
            label_x, offset, label, va="center", fontsize=FONTS["tick"], color=color
        )

    ax.set_xlim(x_min, x_max + x_pad)
    ax.set_yticks([])
    ax.set_xlabel(time_label)
    ax.set_title("Component Time Series")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return _finalize_fig(fig, show=show, fname=fname)


def plot_component_spectrogram(
    component_data,
    sfreq,
    freqs=None,
    n_cycles=None,
    title="Component Spectrogram",
    ax=None,
    show=True,
    fname=None,
):
    """Plot a time-frequency power view for one component.

    Parameters
    ----------
    component_data : ndarray, shape (n_times,) or (n_epochs, n_times)
        Single-component time series or repeated epochs of one component.
    sfreq : float
        Sampling frequency.
    freqs : ndarray | None
        Frequencies to compute. Defaults to 1 Hz through Nyquist, capped at 50 Hz.
    n_cycles : float | ndarray | None
        Number of cycles for multitaper estimation.
    """
    component_data = np.asarray(component_data)
    if component_data.ndim == 1:
        data = component_data[np.newaxis, np.newaxis, :]
    elif component_data.ndim == 2:
        data = component_data[:, np.newaxis, :]
    else:
        raise ValueError("component_data must be 1D or 2D.")

    if freqs is None:
        fmax = int(max(2, min(50, np.floor(sfreq / 2))))
        freqs = np.arange(1, fmax + 1, 1)
    if n_cycles is None:
        n_cycles = freqs / 4.0

    tfr = tfr_array_multitaper(
        data,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        verbose=False,
    )
    power = tfr[:, 0].mean(axis=0)
    times = np.arange(power.shape[1]) / sfreq

    if ax is None:
        fig, ax = themed_figure(figsize=(10, 5))
    else:
        fig = ax.figure

    im = ax.pcolormesh(times, freqs, power, shading="gouraud", cmap="viridis")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Power")

    return _finalize_fig(fig, show=show, fname=fname)
