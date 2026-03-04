"""Visualization for spectral and time-frequency diagnostics.

This module contains reusable plots focused on frequency-domain and
time-frequency views that are not specific to a single denoising method.

This module contains:
1. PSD comparisons for before/after denoising outputs.
2. Component-spectrum comparisons for extracted sources.
3. Spectrogram and time-frequency mask visualizations.
4. Narrowband scan summaries for spectral sweeps.

Authors: Sina Esmaeili <sina.esmaeili@umontreal.ca>
         Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>
"""

from __future__ import annotations

import mne
import numpy as np
from scipy import signal

from .theme import (
    COLORS,
    DIVERGING_CMAP,
    FONTS,
    SEQUENTIAL_CMAP,
    _finalize_fig,
    get_series_color,
    style_axes,
    themed_figure,
    themed_legend,
)

PSD_ZOOM_HALF_WIDTH_HZ = 8.0
PSD_FOCUS_HALF_WIDTH_HZ = 10.0


def _compute_mean_psd(inst, fmin, fmax):
    """Compute the mean PSD across non-frequency axes."""
    spectrum = inst.compute_psd(fmin=fmin, fmax=fmax)
    psd = np.asarray(spectrum.get_data(return_freqs=False))
    mean_axes = tuple(range(psd.ndim - 1))
    if mean_axes:
        psd = psd.mean(axis=mean_axes)
    return spectrum.freqs, np.asarray(psd)


def _compute_array_psd(data, sfreq, fmin, fmax):
    """Compute PSDs for array-like inputs using Welch's method."""
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim > 2:
        data = data.reshape(-1, data.shape[-1])

    nperseg = min(data.shape[-1], int(sfreq * 2))
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    keep = (freqs >= fmin) & (freqs <= fmax)
    return freqs[keep], psd[..., keep]


def _prepare_component_data(components):
    """Normalize component data to shape ``(n_components, n_times)``."""
    if isinstance(components, (mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked)):
        data = components.get_data()
    else:
        data = np.asarray(components, dtype=float)

    if data.ndim == 1:
        return data[np.newaxis, :]
    if data.ndim == 2:
        return data
    if data.ndim != 3:
        raise ValueError(
            "components must be 1D, 2D, or 3D data, or an MNE Raw/Epochs/Evoked object."
        )

    time_axis = int(np.argmax(data.shape))
    non_time_axes = [axis for axis in range(data.ndim) if axis != time_axis]
    component_axis = min(non_time_axes, key=lambda axis: data.shape[axis])
    data = np.moveaxis(data, (component_axis, time_axis), (0, 2))
    return data.mean(axis=1)


def _resolve_tfr_picks(inst, picks):
    """Resolve sensible default picks for time-frequency plots."""
    if picks is not None:
        return picks

    data_picks = mne.pick_types(
        inst.info, meg=True, eeg=True, ref_meg=False, exclude="bads"
    )
    if len(data_picks) == 0:
        data_picks = mne.pick_types(inst.info, misc=True, exclude="bads")
    return data_picks if len(data_picks) > 0 else "all"


def _compute_mean_tfr(inst, freqs, picks):
    """Compute a mean time-frequency map across non time-frequency axes."""
    tfr = inst.compute_tfr(
        method="multitaper",
        freqs=freqs,
        n_cycles=freqs / 2.0,
        picks=_resolve_tfr_picks(inst, picks),
    )
    data = np.asarray(tfr.data)
    mean_axes = tuple(range(data.ndim - 2))
    if mean_axes:
        data = data.mean(axis=mean_axes)
    return np.asarray(data)


def _add_colorbar(fig, ax, image, label):
    """Add a lightly styled colorbar to an axis."""
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(label, fontsize=FONTS["label"])
    colorbar.ax.tick_params(labelsize=FONTS["tick"])
    colorbar.outline.set_edgecolor(COLORS["edge"])
    colorbar.outline.set_linewidth(0.5)
    return colorbar


def _add_line_markers(ax, line_freq, fmax):
    """Add line-frequency markers and visible harmonics to an axis."""
    if line_freq is None:
        return

    ax.axvline(
        line_freq,
        color=COLORS["line_marker"],
        linestyle="--",
        alpha=0.7,
        label=f"{line_freq:g} Hz",
    )

    harmonic = 2
    while line_freq * harmonic <= fmax:
        ax.axvline(
            line_freq * harmonic,
            color=COLORS["line_marker"],
            linestyle="--",
            alpha=0.3,
        )
        harmonic += 1


def _series_color(index, series_name=None, series_colors=None, default=None):
    """Resolve a color for a named PSD series."""
    if series_colors and series_name in series_colors:
        return series_colors[series_name]
    if default is not None:
        return default
    return get_series_color(index)


def _series_label(series_name, series_labels=None):
    """Resolve a display label for a named PSD series."""
    if series_labels and series_name in series_labels:
        return series_labels[series_name]
    return series_name


def _validate_zoom_freqs(zoom_freqs):
    """Normalize the frequencies used for zoomed PSD panels."""
    zoom_freqs = np.asarray(zoom_freqs, dtype=float)
    if zoom_freqs.ndim != 1 or len(zoom_freqs) == 0:
        raise ValueError("zoom_freqs must be a non-empty 1D array-like of frequencies.")
    return zoom_freqs


def plot_narrowband_score_scan(
    frequencies,
    eigenvalues,
    peak_freq=None,
    true_freqs=None,
    ax=None,
    show=True,
    fname=None,
):
    """Plot the score spectrum from a narrowband frequency scan."""
    frequencies = np.asarray(frequencies, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)

    if frequencies.ndim != 1:
        raise ValueError("frequencies must be a 1D array.")
    if eigenvalues.ndim not in (1, 2):
        raise ValueError("eigenvalues must be a 1D or 2D array.")
    if eigenvalues.shape[0] != frequencies.shape[0]:
        raise ValueError(
            "frequencies and eigenvalues must have matching first dimensions."
        )

    if ax is None:
        fig, ax = themed_figure(figsize=(10, 4))
    else:
        fig = ax.figure

    dominant = eigenvalues[:, 0] if eigenvalues.ndim == 2 else eigenvalues
    if eigenvalues.ndim == 2 and eigenvalues.shape[1] > 1:
        ax.plot(
            frequencies,
            eigenvalues[:, 1:],
            color=COLORS["muted"],
            linestyle="-",
            alpha=0.6,
            linewidth=1.2,
        )

    ax.plot(
        frequencies,
        dominant,
        color=COLORS["primary"],
        marker="o",
        linestyle="-",
        markersize=4,
        linewidth=1.8,
        label="Dominant component",
    )

    if peak_freq is not None:
        peak_idx = np.argmin(np.abs(frequencies - peak_freq))
        ax.plot(
            peak_freq,
            dominant[peak_idx],
            color=COLORS["accent"],
            marker="*",
            linestyle="none",
            markersize=14,
            label=f"Peak: {peak_freq:.1f} Hz",
        )
        ax.axvline(peak_freq, color=COLORS["accent"], linestyle="--", alpha=0.5)

    if true_freqs is not None:
        palette = [
            COLORS["accent"],
            COLORS["success"],
            COLORS["secondary"],
            COLORS["purple"],
        ]
        for idx, freq in enumerate(true_freqs):
            ax.axvline(
                freq,
                color=palette[idx % len(palette)],
                linestyle="--",
                alpha=0.5,
                label=f"True: {freq:g} Hz",
            )

    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_ylabel("Score / Eigenvalue", fontsize=FONTS["label"])
    ax.set_title("Narrowband Score Scan", fontsize=FONTS["title"])
    style_axes(ax, grid=True)

    if peak_freq is not None or true_freqs is not None:
        themed_legend(ax, loc="best")

    return _finalize_fig(fig, show=show, fname=fname, tight=False)


def plot_psd_comparison(
    inst_before,
    inst_after,
    fmin=0,
    fmax=np.inf,
    sfreq=None,
    line_freq=None,
    show=True,
    average=True,
    ax=None,
    fname=None,
):
    """Plot PSD comparison for original and denoised data.

    Parameters
    ----------
    inst_before, inst_after : MNE object | ndarray
        Inputs to compare. Supported MNE inputs are Raw, Epochs, and Evoked.
        Array inputs are interpreted with the last axis as time. When either
        input is an array, ``sfreq`` must be provided.
    fmin, fmax : float
        Frequency bounds to display.
    sfreq : float | None
        Sampling frequency for array inputs.
    line_freq : float | None
        Optional line frequency to mark, along with visible harmonics.
    show : bool
        Whether to display the figure.
    average : bool
        If True, average PSDs across non-frequency axes.
    ax : Axes | None
        Optional axis to draw into.
    fname : path-like | None
        Optional output path.
    """
    if ax is None:
        fig, ax = themed_figure(figsize=(8, 4))
    else:
        fig = ax.figure

    for inst, label, color in [
        (inst_before, "Before", COLORS["before"]),
        (inst_after, "After", COLORS["after"]),
    ]:
        if isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs, mne.Evoked)):
            spectrum = inst.compute_psd(fmin=fmin, fmax=fmax)
            psd = np.asarray(spectrum.get_data(return_freqs=False))
            freqs = spectrum.freqs
        else:
            if sfreq is None:
                raise ValueError(
                    "sfreq must be provided when plotting PSDs from arrays."
                )
            freqs, psd = _compute_array_psd(inst, sfreq=sfreq, fmin=fmin, fmax=fmax)

        if average:
            axis = tuple(range(psd.ndim - 1))
            psd_mean = np.mean(psd, axis=axis)
            ax.semilogy(freqs, psd_mean, label=label, color=color)
        else:
            psd = psd.reshape(-1, psd.shape[-1])
            ax.semilogy(freqs, psd.T, color=color, alpha=0.2)
            ax.plot([], [], color=color, label=label)

    display_fmax = float(np.max(freqs)) if np.isinf(fmax) else fmax
    _add_line_markers(ax, line_freq=line_freq, fmax=display_fmax)

    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
    ax.set_title("PSD Comparison", fontsize=FONTS["title"])
    ax.set_xlim(left=max(0.0, fmin), right=display_fmax)
    style_axes(ax, grid=True)
    themed_legend(ax, loc="best")

    return _finalize_fig(fig, show=show, fname=fname, tight=False)


def plot_psd_zoom_comparison(
    freqs_before,
    psd_before,
    freqs_after,
    psd_after,
    *,
    series_name="",
    title="",
    zoom_freqs,
    zoom_annotations=None,
    fmax=125.0,
    series_colors=None,
    series_labels=None,
    fname=None,
    show=True,
):
    """Plot a PSD comparison with zoomed panels around frequencies of interest."""
    freqs_before = np.asarray(freqs_before, dtype=float)
    psd_before = np.asarray(psd_before, dtype=float)
    freqs_after = np.asarray(freqs_after, dtype=float)
    psd_after = np.asarray(psd_after, dtype=float)
    zoom_freqs = _validate_zoom_freqs(zoom_freqs)

    n_zoom = len(zoom_freqs)
    fig, axes = themed_figure(1, 1 + n_zoom, figsize=(4 * (1 + n_zoom), 4))
    axes = np.atleast_1d(axes)

    series_color = _series_color(
        0,
        series_name=series_name,
        series_colors=series_colors,
        default=COLORS["after"],
    )
    series_label = _series_label(series_name, series_labels) if series_name else "After"

    ax = axes[0]
    ax.semilogy(
        freqs_before,
        psd_before,
        color=COLORS["before"],
        alpha=0.5,
        lw=1,
        label="Before",
    )
    ax.semilogy(
        freqs_after,
        psd_after,
        color=series_color,
        lw=1.5,
        label=series_label,
    )
    for freq in zoom_freqs:
        ax.axvline(freq, color=COLORS["line_marker"], ls="--", alpha=0.2)
    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
    ax.set_title("PSD Comparison", fontsize=FONTS["title"])
    ax.set_xlim(0, fmax)
    themed_legend(ax)
    style_axes(ax, grid=True)

    for idx, freq in enumerate(zoom_freqs):
        ax = axes[1 + idx]
        zoom = PSD_ZOOM_HALF_WIDTH_HZ
        before_mask = (freqs_before >= freq - zoom) & (freqs_before <= freq + zoom)
        after_mask = (freqs_after >= freq - zoom) & (freqs_after <= freq + zoom)
        ax.semilogy(
            freqs_before[before_mask],
            psd_before[before_mask],
            color=COLORS["before"],
            alpha=0.5,
            lw=1,
        )
        ax.semilogy(
            freqs_after[after_mask],
            psd_after[after_mask],
            color=series_color,
            lw=1.5,
        )
        ax.axvline(freq, color=COLORS["line_marker"], ls="--", alpha=0.4)

        panel_title = f"{freq:.0f} Hz"
        if zoom_annotations is not None and idx < len(zoom_annotations):
            panel_title += f"\n{zoom_annotations[idx]}"
        ax.set_title(panel_title, fontsize=FONTS["tick"])
        ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
        if idx == 0:
            ax.set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
        style_axes(ax, grid=True)

    if title:
        fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")

    return _finalize_fig(fig, show=show, fname=fname)


def plot_psd_gallery(
    freqs_reference,
    psd_reference,
    series_psds,
    *,
    zoom_freqs,
    fmax=125.0,
    title="",
    series_order=None,
    series_colors=None,
    series_labels=None,
    fname=None,
    show=True,
):
    """Plot a gallery of full-spectrum and zoomed PSD comparisons across series."""
    freqs_reference = np.asarray(freqs_reference, dtype=float)
    psd_reference = np.asarray(psd_reference, dtype=float)
    zoom_freqs = _validate_zoom_freqs(zoom_freqs)
    if series_order is None:
        series_order = sorted(series_psds.keys())

    n_rows = len(series_order)
    n_cols = 1 + len(zoom_freqs)
    fig, axes = themed_figure(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.asarray(axes, dtype=object)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, series_name in enumerate(series_order):
        if series_name not in series_psds:
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                ax.text(
                    0.5,
                    0.5,
                    f"{series_name}\nno data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=FONTS["label"],
                    color=COLORS["placeholder"],
                )
                ax.axis("off")
            continue

        freqs_series, psd_series = series_psds[series_name]
        freqs_series = np.asarray(freqs_series, dtype=float)
        psd_series = np.asarray(psd_series, dtype=float)
        color = _series_color(
            row_idx, series_name=series_name, series_colors=series_colors
        )
        label = _series_label(series_name, series_labels)

        ax = axes[row_idx, 0]
        ax.semilogy(
            freqs_reference,
            psd_reference,
            color=COLORS["before"],
            alpha=0.4,
            lw=0.8,
            label="Before",
        )
        ax.semilogy(freqs_series, psd_series, color=color, lw=1.2, label=label)
        for freq in zoom_freqs:
            ax.axvline(freq, color=COLORS["line_marker"], ls="--", alpha=0.15)
        ax.set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
        if row_idx == 0:
            ax.set_title("Full PSD", fontsize=FONTS["title"])
        ax.text(
            0.01,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONTS["annotation"],
            color=color,
        )
        ax.set_xlim(0, fmax)
        themed_legend(ax, fontsize=6)
        style_axes(ax, grid=True)

        for col_idx, freq in enumerate(zoom_freqs):
            ax = axes[row_idx, 1 + col_idx]
            zoom = PSD_ZOOM_HALF_WIDTH_HZ
            before_mask = (freqs_reference >= freq - zoom) & (
                freqs_reference <= freq + zoom
            )
            series_mask = (freqs_series >= freq - zoom) & (freqs_series <= freq + zoom)
            ax.semilogy(
                freqs_reference[before_mask],
                psd_reference[before_mask],
                color=COLORS["before"],
                alpha=0.4,
                lw=0.8,
            )
            ax.semilogy(
                freqs_series[series_mask],
                psd_series[series_mask],
                color=color,
                lw=1.2,
            )
            ax.axvline(freq, color=COLORS["line_marker"], ls="--", alpha=0.3)
            if row_idx == 0:
                ax.set_title(f"{freq:.0f} Hz", fontsize=FONTS["title"])
            if col_idx == 0:
                ax.set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
            ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
            style_axes(ax, grid=True)

    if title:
        fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold", y=1.01)

    return _finalize_fig(fig, show=show, fname=fname)


def plot_psd_overlay(
    freqs_reference,
    psd_reference,
    series_psds,
    *,
    focus_freq,
    fmax=125.0,
    n_harmonics=3,
    title="",
    series_order=None,
    series_colors=None,
    series_labels=None,
    fname=None,
    show=True,
):
    """Plot full-spectrum and focused-overlay PSD comparisons across series."""
    freqs_reference = np.asarray(freqs_reference, dtype=float)
    psd_reference = np.asarray(psd_reference, dtype=float)

    if series_order is None:
        series_order = sorted(series_psds.keys())

    fig, axes = themed_figure(1, 2, figsize=(16, 5))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.semilogy(
        freqs_reference,
        psd_reference,
        color=COLORS["before"],
        alpha=0.4,
        lw=1,
        label="Before",
    )
    for idx, series_name in enumerate(series_order):
        if series_name not in series_psds:
            continue
        freqs_series, psd_series = series_psds[series_name]
        ax.semilogy(
            freqs_series,
            psd_series,
            color=_series_color(
                idx, series_name=series_name, series_colors=series_colors
            ),
            lw=1.2,
            label=_series_label(series_name, series_labels),
        )
    for harmonic_idx in range(1, n_harmonics + 2):
        harmonic = focus_freq * harmonic_idx
        if harmonic < fmax:
            ax.axvline(harmonic, color=COLORS["line_marker"], ls="--", alpha=0.15)
    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
    ax.set_title(title or "Full Spectrum Comparison", fontsize=FONTS["title"])
    ax.set_xlim(0, fmax)
    themed_legend(ax)
    style_axes(ax, grid=True)

    ax = axes[1]
    zoom = PSD_FOCUS_HALF_WIDTH_HZ
    before_mask = (freqs_reference >= focus_freq - zoom) & (
        freqs_reference <= focus_freq + zoom
    )
    ax.semilogy(
        freqs_reference[before_mask],
        psd_reference[before_mask],
        color=COLORS["before"],
        alpha=0.4,
        lw=1,
        label="Before",
    )
    for idx, series_name in enumerate(series_order):
        if series_name not in series_psds:
            continue
        freqs_series, psd_series = series_psds[series_name]
        series_mask = (freqs_series >= focus_freq - zoom) & (
            freqs_series <= focus_freq + zoom
        )
        ax.semilogy(
            freqs_series[series_mask],
            psd_series[series_mask],
            color=_series_color(
                idx, series_name=series_name, series_colors=series_colors
            ),
            lw=1.5,
            label=_series_label(series_name, series_labels),
        )
    ax.axvline(focus_freq, color=COLORS["line_marker"], ls="--", alpha=0.4)
    ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_title(f"Zoom at {focus_freq:.0f} Hz", fontsize=FONTS["title"])
    themed_legend(ax)
    style_axes(ax, grid=True)

    return _finalize_fig(fig, show=show, fname=fname)


def plot_component_psd_comparison(
    inst_before,
    components,
    sfreq,
    peak_freq=None,
    fmin=1,
    fmax=40,
    show=True,
    fname=None,
):
    """Plot original PSD alongside the PSD of extracted components."""
    fig, axes = themed_figure(1, 2, figsize=(12, 4), constrained_layout=True)
    axes = np.atleast_1d(axes)

    freqs_before, psd_before = _compute_mean_psd(inst_before, fmin=fmin, fmax=fmax)
    axes[0].semilogy(freqs_before, psd_before, color=COLORS["before"], label="Before")
    axes[0].set_title("Original Data PSD", fontsize=FONTS["title"])
    axes[0].set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    axes[0].set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
    style_axes(axes[0], grid=True)

    component_data = _prepare_component_data(components)
    n_components = min(3, component_data.shape[0])
    component_raw = mne.io.RawArray(
        component_data[:n_components],
        mne.create_info(n_components, sfreq, "misc"),
        verbose=False,
    )
    component_spectrum = component_raw.compute_psd(fmin=fmin, fmax=fmax, picks="all")
    freqs_components = component_spectrum.freqs
    psd_components = np.asarray(
        component_spectrum.get_data(picks="all", return_freqs=False)
    )
    psd_components = np.atleast_2d(psd_components)
    component_palette = [
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["accent"],
        COLORS["purple"],
    ]
    for idx, component_psd in enumerate(psd_components):
        axes[1].semilogy(
            freqs_components,
            component_psd,
            color=component_palette[idx % len(component_palette)],
            label=f"Component {idx}",
        )

    axes[1].set_title("Component PSD", fontsize=FONTS["title"])
    axes[1].set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
    axes[1].set_ylabel("Power Spectral Density", fontsize=FONTS["label"])
    style_axes(axes[1], grid=True)

    if peak_freq is not None:
        for axis in axes:
            axis.axvline(
                peak_freq,
                color=COLORS["line_marker"],
                linestyle="--",
                alpha=0.7,
            )

    themed_legend(axes[0], loc="best")
    themed_legend(axes[1], loc="best")

    return _finalize_fig(fig, show=show, fname=fname, tight=False)


def plot_spectrogram_comparison(
    inst_before,
    inst_after,
    fmin=1,
    fmax=40,
    n_freqs=20,
    picks=None,
    show=True,
    fname=None,
):
    """Compare time-frequency spectrograms averaged over channels."""
    if n_freqs < 2:
        raise ValueError("n_freqs must be at least 2.")
    if fmax <= fmin:
        raise ValueError("fmax must be greater than fmin.")

    freqs = np.linspace(fmin, fmax, n_freqs)
    data_before = _compute_mean_tfr(inst_before, freqs, picks=picks)
    data_after = _compute_mean_tfr(inst_after, freqs, picks=picks)
    diff = data_before - data_after

    fig, axes = themed_figure(
        1, 3, figsize=(15, 4), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    vmax = float(max(np.max(data_before), np.max(data_after)))
    diff_limit = float(np.max(np.abs(diff)))
    times = inst_before.times

    def _plot_im(ax, data, title, cmap, vlims, colorbar_label):
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            vmin=vlims[0],
            vmax=vlims[1],
        )
        ax.set_title(title, fontsize=FONTS["title"])
        ax.set_xlabel("Time (s)", fontsize=FONTS["label"])
        ax.set_ylabel("Frequency (Hz)", fontsize=FONTS["label"])
        style_axes(ax, grid=False)
        _add_colorbar(fig, ax, im, colorbar_label)

    _plot_im(
        axes[0],
        data_before,
        "Before",
        cmap=SEQUENTIAL_CMAP,
        vlims=(0.0, vmax),
        colorbar_label="Power",
    )
    _plot_im(
        axes[1],
        data_after,
        "After",
        cmap=SEQUENTIAL_CMAP,
        vlims=(0.0, vmax),
        colorbar_label="Power",
    )
    _plot_im(
        axes[2],
        diff,
        "Before - After",
        cmap=DIVERGING_CMAP,
        vlims=(-diff_limit, diff_limit),
        colorbar_label="Power Difference",
    )

    return _finalize_fig(fig, show=show, fname=fname, tight=False)


def plot_time_frequency_mask(
    mask,
    times,
    freqs,
    title="Time-Frequency Mask",
    ax=None,
    show=True,
    fname=None,
):
    """Visualize a time-frequency mask."""
    mask = np.asarray(mask)
    times = np.asarray(times)
    freqs = np.asarray(freqs)

    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array.")
    if times.ndim != 1 or freqs.ndim != 1:
        raise ValueError("times and freqs must be 1D arrays.")
    if mask.shape != (len(freqs), len(times)):
        raise ValueError("mask shape must match (len(freqs), len(times)).")

    if ax is None:
        fig, ax = themed_figure(figsize=(10, 5))
    else:
        fig = ax.figure

    im = ax.pcolormesh(
        times,
        freqs,
        mask,
        shading="auto",
        cmap=SEQUENTIAL_CMAP,
        vmin=0,
        vmax=1,
    )
    ax.set_ylabel("Frequency (Hz)", fontsize=FONTS["label"])
    ax.set_xlabel("Time (s)", fontsize=FONTS["label"])
    ax.set_title(title, fontsize=FONTS["title"])
    style_axes(ax, grid=False)
    _add_colorbar(fig, ax, im, "Mask Weight")

    return _finalize_fig(fig, show=show, fname=fname)
