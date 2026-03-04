"""Visualization for time-domain and evoked signal plots.

This module contains reusable signal-level plots for comparing denoised
outputs in the time domain, across channels, and across grouped evoked
responses.

Authors: Sina Esmaeili <sina.esmaeili@umontreal.ca>
         Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np

from .theme import (
    COLORS,
    FONTS,
    _finalize_fig,
    style_axes,
    themed_figure,
    themed_legend,
)


def _add_time_windows(ax, windows=None, *, colors=None, alpha=0.06, scale=1000.0):
    """Shade named time windows on an axes."""
    if not windows:
        return

    for name, (t0, t1) in windows.items():
        color = (
            colors.get(name, COLORS["gray"]) if colors is not None else COLORS["gray"]
        )
        ax.axvspan(t0 * scale, t1 * scale, alpha=alpha, color=color)


def _ch_index(epochs_or_array, ch_name):
    """Resolve a channel name or numeric string to an integer index."""
    if hasattr(epochs_or_array, "ch_names"):
        return epochs_or_array.ch_names.index(ch_name)
    return int(ch_name)


def _get_times_ms(epochs):
    """Return an object's time axis in milliseconds."""
    if hasattr(epochs, "times"):
        return np.asarray(epochs.times) * 1000
    raise ValueError("Cannot determine times for plain array input.")


def _compute_gfp(inst):
    """Compute GFP or RMS-style amplitude summary across channels."""
    data = inst.get_data()
    if data.ndim == 3:
        evoked_data = data.mean(axis=0)
        return np.sqrt(np.mean(evoked_data**2, axis=0))
    return np.sqrt(np.mean(data**2, axis=0))


def _bootstrap_gfp(inst, ci, n_boot):
    """Bootstrap confidence interval for GFP of the evoked mean."""
    data = inst.get_data()
    n_epochs = data.shape[0]
    rng = np.random.default_rng(42)

    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n_epochs, n_epochs, replace=True)
        evoked_boot = data[idx].mean(axis=0)
        boots.append(np.sqrt(np.mean(evoked_boot**2, axis=0)))

    boots = np.asarray(boots)
    alpha = (1 - ci) / 2
    return np.percentile(boots, [100 * alpha, 100 * (1 - alpha)], axis=0)


def _get_var(inst):
    """Compute per-channel variance for Raw, Epochs, or Evoked."""
    if isinstance(inst, mne.io.BaseRaw | mne.Evoked):
        data = inst.get_data()
        return np.var(data, axis=1)
    if isinstance(inst, mne.BaseEpochs):
        data = inst.get_data()
        return np.mean(np.var(data, axis=2), axis=0)
    raise ValueError("Unknown data type")


def _get_gfp_for_summary(inst):
    """Compute GFP used in the denoising summary figure."""
    if isinstance(inst, mne.io.BaseRaw | mne.Evoked):
        data = inst.get_data()
        return np.std(data, axis=0)
    if isinstance(inst, mne.BaseEpochs):
        data = inst.get_data()
        gfp_trials = np.std(data, axis=1)
        return np.mean(gfp_trials, axis=0)
    return None


def _get_data_1d(inst):
    """Extract a 1D signal and time axis from common MNE or array inputs."""
    if hasattr(inst, "get_data"):
        data = inst.get_data().flatten()
        if hasattr(inst, "times") and len(inst.times) == len(data):
            times = np.asarray(inst.times)
        else:
            times = np.arange(len(data))
    else:
        data = np.asarray(inst).flatten()
        times = np.arange(len(data))
    return times, data


def plot_evoked_gfp_comparison(
    inst_before,
    inst_after,
    ci=0.95,
    n_boot=1000,
    colors=(COLORS["before"], COLORS["after"]),
    linestyles=("-", "-"),
    labels=("Before", "After"),
    show=True,
    ax=None,
    fname=None,
):
    """Plot a GFP comparison with optional bootstrap confidence bands."""
    if ax is None:
        fig, ax = themed_figure(figsize=(10, 6))
    else:
        fig = ax.figure

    times = inst_before.times
    gfp_before = _compute_gfp(inst_before)
    ax.plot(
        times,
        gfp_before,
        color=colors[0],
        linestyle=linestyles[0],
        label=labels[0],
        linewidth=1.5,
    )

    if ci is not None and isinstance(inst_before, mne.BaseEpochs):
        ci_low, ci_high = _bootstrap_gfp(inst_before, ci=ci, n_boot=n_boot)
        ax.fill_between(times, ci_low, ci_high, color=colors[0], alpha=0.2, linewidth=0)

    gfp_after = _compute_gfp(inst_after)
    ax.plot(
        times,
        gfp_after,
        color=colors[1],
        linestyle=linestyles[1],
        label=labels[1],
        linewidth=1.5,
    )

    if ci is not None and isinstance(inst_after, mne.BaseEpochs):
        ci_low, ci_high = _bootstrap_gfp(inst_after, ci=ci, n_boot=n_boot)
        ax.fill_between(times, ci_low, ci_high, color=colors[1], alpha=0.2, linewidth=0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Global Field Power")
    ax.set_title("Evoked GFP Comparison")
    style_axes(ax, grid=True)
    themed_legend(ax, loc="best")

    return _finalize_fig(fig, show=show, fname=fname)


def plot_channel_time_course_comparison(
    inst_before,
    inst_after,
    picks=None,
    start=0,
    stop=None,
    show=True,
    fname=None,
):
    """Plot before/after channel time courses for selected channels."""
    if picks is None:
        picks = list(range(min(5, len(inst_before.ch_names))))

    ch_names = inst_before.ch_names
    resolved = []
    for pick in picks:
        resolved.append(ch_names.index(pick) if isinstance(pick, str) else pick)
    pick_labels = [ch_names[idx] for idx in resolved]

    times = inst_before.times

    if isinstance(inst_before, mne.io.BaseRaw):
        data_before = inst_before.get_data(picks=resolved, start=start, stop=stop)
        data_after = inst_after.get_data(picks=resolved, start=start, stop=stop)
        if start is not None:
            times = times[start:stop] if stop else times[start:]
    elif isinstance(inst_before, mne.BaseEpochs):
        data_before = inst_before.get_data(picks=resolved)
        data_after = inst_after.get_data(picks=resolved)
    else:
        data_before = inst_before.get_data(picks=resolved)
        data_after = inst_after.get_data(picks=resolved)

    fig, axes = themed_figure(
        len(resolved),
        1,
        sharex=True,
        figsize=(10, 2 * len(resolved)),
    )
    axes = np.atleast_1d(axes)

    for row_idx, label in enumerate(pick_labels):
        ax = axes[row_idx]
        if data_before.ndim == 3:
            y_before = data_before[:, row_idx, :].mean(axis=0)
            y_after = data_after[:, row_idx, :].mean(axis=0)
        else:
            y_before = data_before[row_idx]
            y_after = data_after[row_idx]

        ax.plot(times, y_before, label="Before", color=COLORS["before"], alpha=0.7)
        ax.plot(times, y_after, label="After", color=COLORS["after"], alpha=0.7)
        ax.set_ylabel(label)
        style_axes(ax, grid=True)
        if row_idx == 0:
            themed_legend(ax, loc="best")

    axes[-1].set_xlabel("Time (s)")
    return _finalize_fig(fig, show=show, fname=fname)


def plot_power_ratio_map(
    inst_before,
    inst_after,
    info=None,
    show=True,
    ax=None,
    fname=None,
):
    """Plot a topomap of the preserved power ratio after denoising."""
    if info is None:
        if hasattr(inst_before, "info"):
            info = inst_before.info
        else:
            raise ValueError("info is required")

    var_before = _get_var(inst_before)
    var_after = _get_var(inst_after)
    ratio = np.divide(
        var_after,
        var_before,
        out=np.full_like(var_after, np.nan, dtype=float),
        where=var_before != 0,
    )

    if ax is None:
        fig, ax = themed_figure(figsize=(5, 5))
    else:
        fig = ax.figure

    im, _ = mne.viz.plot_topomap(
        ratio,
        info,
        axes=ax,
        show=False,
        names=getattr(inst_before, "ch_names", None),
        vlim=(0, 1),
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="Power Ratio (After / Before)")
    ax.set_title("Preserved Power Fraction")

    return _finalize_fig(fig, show=show, fname=fname)


def plot_signal_overlay(
    inst_before,
    inst_after,
    start=None,
    stop=None,
    scale_after=True,
    title=None,
    show=True,
    fname=None,
):
    """Overlay before/after signals to inspect reconstruction quality."""
    t_before, data_before = _get_data_1d(inst_before)
    t_after, data_after = _get_data_1d(inst_after)

    if start is not None:
        mask = t_before >= start
        t_before = t_before[mask]
        data_before = data_before[mask]
        t_after = t_after[mask[: len(t_after)]]
        data_after = data_after[mask[: len(data_after)]]

    if stop is not None:
        mask = t_before <= stop
        t_before = t_before[mask]
        data_before = data_before[mask]
        t_after = t_after[mask[: len(t_after)]]
        data_after = data_after[mask[: len(data_after)]]

    n_samples = min(len(data_before), len(data_after))
    times = t_before[:n_samples]
    data_before = data_before[:n_samples]
    data_after = data_after[:n_samples]

    if scale_after:
        scaler = np.std(data_before) / (np.std(data_after) + 1e-9)
        data_after = data_after * scaler

    fig, ax = themed_figure(figsize=(12, 4))
    ax.plot(
        times,
        data_before,
        color=COLORS["before"],
        label="Before",
        alpha=0.5,
        linewidth=1,
    )
    ax.plot(
        times,
        data_after,
        color=COLORS["after"],
        linestyle="--",
        label="After",
        alpha=0.85,
        linewidth=1.2,
    )
    ax.set_xlabel("Time (s)" if hasattr(inst_before, "times") else "Time (samples)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or "Signal Overlay Comparison")
    style_axes(ax, grid=True)
    themed_legend(ax, loc="best")

    return _finalize_fig(fig, show=show, fname=fname)


def plot_grand_average_evokeds(
    all_evokeds,
    *,
    channels=("Cz", "Pz"),
    time_windows=None,
    suptitle=None,
    group_order=None,
    group_colors=None,
    group_labels=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Plot group-mean evokeds with between-subject SEM bands."""
    if group_order is None:
        group_order = sorted(all_evokeds.keys())
    default_cycle = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()
        .get(
            "color",
            [
                COLORS["primary"],
                COLORS["secondary"],
                COLORS["accent"],
                COLORS["success"],
            ],
        )
    )
    n_channels = len(channels)
    if figsize is None:
        figsize = (8 * n_channels, 5)

    fig, axes = themed_figure(1, n_channels, figsize=figsize)
    if n_channels == 1:
        axes = np.array([axes])

    for col_idx, ch_name in enumerate(channels):
        ax = axes.flat[col_idx]
        for group_idx, group in enumerate(group_order):
            evoked_list = all_evokeds[group]
            n_sub = len(evoked_list)
            ch_idx = evoked_list[0].ch_names.index(ch_name)
            times_ms = evoked_list[0].times * 1000

            stacked = np.array([ev.data[ch_idx] * 1e6 for ev in evoked_list])
            grand_mean = stacked.mean(axis=0)
            grand_sem = (
                stacked.std(axis=0, ddof=1) / np.sqrt(n_sub)
                if n_sub > 1
                else np.zeros_like(grand_mean)
            )

            color = (
                group_colors[group]
                if group_colors is not None and group in group_colors
                else default_cycle[group_idx % len(default_cycle)]
            )
            label = (
                group_labels[group]
                if group_labels is not None and group in group_labels
                else group
            )

            ax.plot(
                times_ms,
                grand_mean,
                color=color,
                lw=1.8,
                alpha=0.85,
                label=label,
            )
            if n_sub > 1:
                ax.fill_between(
                    times_ms,
                    grand_mean - grand_sem,
                    grand_mean + grand_sem,
                    color=color,
                    alpha=0.15,
                    lw=0,
                )

        ax.axvline(0, color=COLORS["gray"], ls="--", alpha=0.5)
        ax.axhline(0, color=COLORS["gray"], alpha=0.3)
        _add_time_windows(ax, time_windows)
        ax.set_xlabel("Time (ms)", fontsize=FONTS["label"])
        ax.set_ylabel("Amplitude (µV)", fontsize=FONTS["label"])
        ax.set_title(f"Grand Average at {ch_name}", fontsize=FONTS["title"])
        themed_legend(ax)
        style_axes(ax)

    n_total = len(next(iter(all_evokeds.values())))
    title = suptitle or f"Grand-Average Evoked ± Between-Subject SEM (N = {n_total})"
    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)
