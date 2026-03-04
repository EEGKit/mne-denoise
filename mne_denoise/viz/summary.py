"""Composed multi-panel summary figures for mne-denoise visualizations.

This module groups the higher-level figures in :mod:`mne_denoise.viz` that
combine several primitive plots into method- or workflow-level summaries.

This module contains:
1. General denoising summary dashboards.
2. DSS summary dashboards and DSS mode-comparison figures.
3. ZapLine summary dashboards.
4. Combined benchmark trade-off summary figures.
5. ERP signal and endpoint summary figures.

Authors: Sina Esmaeili <sina.esmaeili@umontreal.ca>
         Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import welch as _welch

from ..qa import spectral_distortion, suppression_ratio, variance_removed
from ._seaborn import _suppress_seaborn_plot_warnings, _try_import_seaborn
from ._utils import _get_info, _get_scores
from .components import plot_component_patterns, plot_component_score_curve
from .signals import (
    _add_time_windows,
    _ch_index,
    _get_gfp_for_summary,
    _get_times_ms,
    plot_power_ratio_map,
)
from .spectra import plot_psd_comparison
from .stats import plot_single_metric_comparison, plot_tradeoff_scatter
from .theme import (
    COLORS,
    DEFAULT_DPI,
    DEFAULT_FIGSIZE,
    DEFAULT_PIPE_COLORS,
    FONTS,
    _finalize_fig,
    style_axes,
    themed_figure,
    themed_legend,
    use_theme,
)

DEFAULT_METHOD_LABELS = {
    "M0": "Baseline (no cleaning)",
    "M1": "ZapLine (auto)",
    "M2": "ZapLine+ (adaptive)",
    "M3": "Notch filter",
}

DEFAULT_METHOD_ORDER = ["M0", "M1", "M2", "M3"]

DEFAULT_PIPE_LABELS: dict[str, str] = {
    "C0": "Baseline",
    "C1": "Paper (ICA)",
    "C2": "DSS (AverageBias)",
}

DEFAULT_PIPE_ORDER: list[str] = ["C0", "C1", "C2"]

DEFAULT_ERP_WINDOWS: dict[str, tuple[float, float]] = {
    "N1": (0.080, 0.130),
    "MMN": (0.100, 0.250),
    "P300": (0.250, 0.500),
}

_ERP_WIN_COLORS: dict[str, str] = {
    "N1": COLORS["blue"],
    "MMN": COLORS["purple"],
    "P300": COLORS["green"],
}


def _pipe_color(pipe, pipe_colors=None):
    """Resolve pipeline color."""
    if pipe_colors and pipe in pipe_colors:
        return pipe_colors[pipe]
    return DEFAULT_PIPE_COLORS.get(pipe, COLORS["dark"])


def _pipe_label(pipe, pipe_labels=None):
    """Resolve pipeline label."""
    if pipe_labels and pipe in pipe_labels:
        return pipe_labels[pipe]
    return DEFAULT_PIPE_LABELS.get(pipe, pipe)


def _get_n_selected(estimator):
    """Return number of components auto-selected (``n_selected_``)."""
    if hasattr(estimator, "n_selected_") and estimator.n_selected_ is not None:
        return estimator.n_selected_
    if hasattr(estimator, "n_removed_") and estimator.n_removed_ is not None:
        return estimator.n_removed_
    return 0


def _get_eigenvalues(estimator):
    """Eigenvalue vector, or ``None``."""
    return _get_scores(estimator)


def _get_segment_results(estimator):
    """Return ``segment_results_`` list-of-dict or ``[]``."""
    if hasattr(estimator, "segment_results_") and estimator.segment_results_:
        return estimator.segment_results_
    if hasattr(estimator, "adaptive_results_") and estimator.adaptive_results_:
        return estimator.adaptive_results_.get("chunk_info", [])
    return []


def _is_segmented(estimator):
    """Check whether the result is from segmented / adaptive mode."""
    if hasattr(estimator, "segmented") and estimator.segmented:
        return True
    return bool(_get_segment_results(estimator))


def _get_dss_removed(estimator):
    """Return removed artifact array (n_channels, n_times) or ``None``."""
    if hasattr(estimator, "removed") and estimator.removed is not None:
        return estimator.removed
    if hasattr(estimator, "removed_") and estimator.removed_ is not None:
        return estimator.removed_
    return None


def _get_bias_name(estimator):
    """Human-readable bias function name."""
    bias = getattr(estimator, "bias", None)
    if bias is None:
        return "Unknown"
    name = type(bias).__name__
    rename = {
        "BandpassBias": "Bandpass",
        "TrialAverageBias": "Trial Average",
        "SmoothingBias": "Smoothing",
        "PeakFilterBias": "Peak Filter",
        "CombFilterBias": "Comb Filter",
        "NotchBias": "Notch",
    }
    return rename.get(name, name)


def _get_zapline_n_removed(result):
    """Extract ``n_removed`` from a ZapLine result, handling both modes."""
    n_removed = 0
    if hasattr(result, "n_removed_") and result.n_removed_ is not None:
        n_removed = result.n_removed_
    elif hasattr(result, "n_removed") and result.n_removed is not None:
        n_removed = result.n_removed
    elif isinstance(result, dict):
        n_removed = result.get("n_removed", 0) or 0
    return n_removed


def _get_zapline_removed(result):
    """Extract removed data from a ZapLine result, handling both modes."""
    if hasattr(result, "removed") and result.removed is not None:
        return result.removed
    if hasattr(result, "adaptive_results_") and result.adaptive_results_ is not None:
        return result.adaptive_results_.get("removed")
    if isinstance(result, dict):
        return result.get("removed")
    return None


def _get_cleaned(result):
    """Extract cleaned data from a ZapLine result, handling both modes."""
    if hasattr(result, "cleaned") and result.cleaned is not None:
        return result.cleaned
    if hasattr(result, "adaptive_results_") and result.adaptive_results_ is not None:
        return result.adaptive_results_.get("cleaned")
    if isinstance(result, dict):
        return result.get("cleaned")
    return None


def _is_adaptive(result):
    """Check whether the result is from adaptive mode."""
    if hasattr(result, "adaptive") and result.adaptive:
        return True
    if hasattr(result, "adaptive_results_") and result.adaptive_results_ is not None:
        return True
    return bool(isinstance(result, dict) and "chunk_info" in result)


def _get_chunk_info(result):
    """Extract per-chunk info from an adaptive ZapLine result."""
    if hasattr(result, "adaptive_results_") and result.adaptive_results_ is not None:
        return result.adaptive_results_.get("chunk_info", [])
    if isinstance(result, dict):
        return result.get("chunk_info", [])
    return []


def _fallback_chunk_info(result):
    """Synthesize a single adaptive chunk when detailed chunk info is unavailable."""
    line_freq = getattr(result, "line_freq", None)
    removed = _get_zapline_removed(result)
    cleaned = _get_cleaned(result)
    n_times = 1
    for data in (removed, cleaned):
        if isinstance(data, np.ndarray) and data.ndim >= 2:
            n_times = data.shape[-1]
            break
    return [
        {
            "n_removed": _get_zapline_n_removed(result),
            "fine_freq": 0.0 if line_freq is None else line_freq,
            "frequency": 0.0 if line_freq is None else line_freq,
            "artifact_present": _get_zapline_n_removed(result) > 0,
            "start": 0,
            "end": n_times,
        }
    ]


def _new_summary_figure(*, figsize=None, dpi=None, constrained_layout=False):
    """Create a themed figure for multi-panel summary dashboards."""
    if figsize is None:
        figsize = DEFAULT_FIGSIZE
    if dpi is None:
        dpi = DEFAULT_DPI
    with use_theme():
        fig = plt.figure(
            figsize=figsize,
            dpi=dpi,
            facecolor="white",
            constrained_layout=constrained_layout,
        )
    return fig


def _new_summary_grid(*, figsize=None, dpi=None, hspace=0.42, wspace=0.30):
    """Create the shared 3x2 dashboard grid used by summary plots."""
    from matplotlib.gridspec import GridSpec

    fig = _new_summary_figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        3,
        2,
        figure=fig,
        hspace=hspace,
        wspace=wspace,
        left=0.07,
        right=0.97,
        top=0.92,
        bottom=0.06,
    )
    return fig, gs


def _removed_rms_scale(removed_rms):
    """Return scaling factor and display unit for removed RMS values."""
    if np.max(removed_rms) < 1e-3:
        return 1e6, "uV"
    if np.max(removed_rms) < 1.0:
        return 1e3, "mV"
    return 1.0, "V"


def _draw_summary_table(ax, rows, *, title="(f)  Summary"):
    """Render the repeated right-column summary key/value table."""
    ax.axis("off")

    n_rows = len(rows)
    row_h = min(0.085, 0.90 / max(n_rows, 1))
    y_top = 0.94

    for i, (label, value) in enumerate(rows):
        y = y_top - i * row_h
        ax.text(
            0.03,
            y,
            label,
            transform=ax.transAxes,
            fontsize=FONTS["label"] - 0.5,
            color=COLORS["label_secondary"],
            va="top",
            fontfamily="sans-serif",
        )
        ax.text(
            0.97,
            y,
            value,
            transform=ax.transAxes,
            fontsize=FONTS["label"] - 0.5,
            color=COLORS["text"],
            va="top",
            ha="right",
            fontfamily="monospace",
            fontweight="bold",
        )
        y_line = y - row_h * 0.35
        ax.plot(
            [0.03, 0.97],
            [y_line, y_line],
            color=COLORS["separator"],
            lw=0.4,
            transform=ax.transAxes,
            clip_on=False,
        )

    ax.set_title(
        title,
        fontsize=FONTS["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )


def _plot_selection_eigenvalues_panel(
    ax,
    values,
    selected_count,
    *,
    selected_label,
    title="(a)  Component scores",
):
    """Plot the repeated selected-versus-retained eigenvalue panel."""
    style_axes(ax, grid=True)

    if values is not None and values.size > 0:
        n_values = len(values)
        x_vals = np.arange(n_values)
        bar_colors = [
            COLORS["accent"] if i < selected_count else COLORS["primary"]
            for i in range(n_values)
        ]
        ax.bar(x_vals, values, color=bar_colors, edgecolor="none", width=0.75)

        mean_value = np.mean(values)
        ax.axhline(mean_value, color=COLORS["muted"], ls="--", lw=0.8, zorder=3)

        if 0 < selected_count < n_values:
            ax.axvline(
                selected_count - 0.5,
                color=COLORS["accent"],
                ls="-",
                lw=1.0,
                zorder=4,
            )

        themed_legend(
            ax,
            loc="upper right",
            handles=[
                plt.Rectangle((0, 0), 1, 1, fc=COLORS["accent"], label=selected_label),
                plt.Rectangle((0, 0), 1, 1, fc=COLORS["primary"], label="Retained"),
                plt.Line2D(
                    [],
                    [],
                    color=COLORS["muted"],
                    ls="--",
                    lw=0.8,
                    label=f"Mean ({mean_value:.3f})",
                ),
            ],
        )
        ax.set_xlim(-0.6, n_values - 0.4)
    else:
        ax.text(
            0.5,
            0.5,
            "No eigenvalues available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTS["label"],
            color=COLORS["placeholder"],
        )

    ax.set_xlabel("Component", fontsize=FONTS["label"])
    ax.set_ylabel("Eigenvalue", fontsize=FONTS["label"])
    ax.set_title(
        title,
        fontsize=FONTS["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )


def _plot_patterns_panel(
    fig,
    subplot_spec,
    patterns,
    *,
    info=None,
    channel_names=None,
    max_components=4,
    title="(b)  Spatial mixing patterns",
):
    """Plot the repeated component-pattern panel with topomap fallback."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    has_patterns = patterns is not None and patterns.size > 0
    palette = [
        COLORS["accent"],
        COLORS["primary"],
        COLORS["secondary"],
        COLORS["success"],
    ]

    if has_patterns and info is not None:
        import mne

        n_ch, n_pat = patterns.shape
        n_show = min(n_pat, max_components)
        gs_inner = GridSpecFromSubplotSpec(
            1, n_show, subplot_spec=subplot_spec, wspace=0.25
        )
        for k in range(n_show):
            ax_k = fig.add_subplot(gs_inner[0, k])
            mne.viz.plot_topomap(
                patterns[:, k],
                info,
                axes=ax_k,
                show=False,
                contours=4,
            )
            ax_k.set_title(
                f"Comp {k + 1}",
                fontsize=FONTS["tick"],
                pad=3,
                color=palette[k % len(palette)],
                fontweight="semibold",
            )
        pos = fig.add_subplot(subplot_spec).get_position()
        fig.axes[-1].set_visible(False)
        fig.text(
            pos.x0,
            pos.y1 + 0.015,
            title,
            fontsize=FONTS["title"],
            fontweight="semibold",
            ha="left",
            va="bottom",
        )
        return

    ax = fig.add_subplot(subplot_spec)
    style_axes(ax, grid=True)

    if has_patterns:
        n_ch, n_pat = patterns.shape
        n_show = min(n_pat, max_components)
        x_ch = np.arange(n_ch)
        for k in range(n_show):
            markerline, stemlines, _baseline = ax.stem(
                x_ch,
                patterns[:, k],
                linefmt="-",
                markerfmt="o",
                basefmt="",
                label=f"Comp {k + 1}",
            )
            color = palette[k % len(palette)]
            markerline.set(color=color, markersize=3)
            stemlines.set(color=color, linewidth=0.7, alpha=0.6)

        if channel_names is not None and len(channel_names) == n_ch:
            if n_ch <= 20:
                tick_idx = x_ch
            else:
                step = max(1, n_ch // 12)
                tick_idx = x_ch[::step]
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(
                [channel_names[i] for i in tick_idx],
                rotation=45,
                ha="right",
                fontsize=FONTS["tick"] - 0.5,
            )
        else:
            step = max(1, n_ch // 12)
            ax.set_xticks(x_ch[::step])

        themed_legend(ax, loc="upper right")
    else:
        ax.text(
            0.5,
            0.5,
            "No patterns available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTS["label"],
            color=COLORS["placeholder"],
        )

    ax.set_xlabel("Channel", fontsize=FONTS["label"])
    ax.set_ylabel("Weight", fontsize=FONTS["label"])
    ax.set_title(
        title,
        fontsize=FONTS["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )


def _plot_removed_power_panel(
    fig,
    ax,
    removed,
    *,
    info=None,
    channel_names=None,
    title="(c)  Removed artifact power",
    bar_color=None,
    no_data_text="No removal data\n(run fit_transform first)",
):
    """Plot the repeated removed-artifact power panel."""
    bar_color = COLORS["accent"] if bar_color is None else bar_color
    has_removed = removed is not None and np.any(removed != 0)

    if has_removed and info is not None:
        import matplotlib.ticker as mticker
        import mne

        removed_rms = np.sqrt(np.mean(removed**2, axis=1))
        scale, unit = _removed_rms_scale(removed_rms)

        im, _ = mne.viz.plot_topomap(
            removed_rms,
            info,
            axes=ax,
            show=False,
            contours=4,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"RMS ({unit})", fontsize=FONTS["tick"])
        cbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _, s=scale: f"{x * s:.1f}")
        )
        cbar.ax.tick_params(labelsize=FONTS["tick"] - 1)
    else:
        style_axes(ax, grid=True)
        if has_removed:
            removed_rms = np.sqrt(np.mean(removed**2, axis=1))
            n_ch = len(removed_rms)
            x_pos = np.arange(n_ch)
            scale, unit = _removed_rms_scale(removed_rms)

            ax.bar(
                x_pos,
                removed_rms * scale,
                color=bar_color,
                edgecolor="none",
                width=0.75,
                alpha=0.85,
            )

            if channel_names is not None and len(channel_names) == n_ch:
                step = max(1, n_ch // 16) if n_ch > 20 else 1
                tick_idx = x_pos[::step]
                ax.set_xticks(tick_idx)
                ax.set_xticklabels(
                    [channel_names[i] for i in range(0, n_ch, step)],
                    rotation=45,
                    ha="right",
                    fontsize=FONTS["tick"] - 0.5,
                )
            else:
                step = max(1, n_ch // 12)
                ax.set_xticks(x_pos[::step])

            ax.set_xlim(-0.6, n_ch - 0.4)
            ax.set_ylabel(f"RMS ({unit})", fontsize=FONTS["label"])
        else:
            ax.text(
                0.5,
                0.5,
                no_data_text,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=FONTS["label"],
                color=COLORS["placeholder"],
            )

        ax.set_xlabel("Channel", fontsize=FONTS["label"])

    ax.set_title(
        title,
        fontsize=FONTS["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )


def _plot_source_trace_panel(
    ax,
    sources,
    *,
    sfreq=None,
    title,
    empty_text="No source data\n(run fit_transform first)",
):
    """Plot the repeated short-window component/source trace panel."""
    style_axes(ax)

    if sources is not None and sources.size > 0 and sfreq is not None:
        n_src = sources.shape[0]
        n_show = min(n_src, 4)
        n_samples = sources.shape[1]
        t_full = np.arange(n_samples) / sfreq
        show_dur = min(2.0, n_samples / sfreq)
        win_samples = int(show_dur * sfreq)

        if n_samples > win_samples:
            rng = np.random.RandomState(42)
            win_start = rng.randint(0, n_samples - win_samples)
        else:
            win_start = 0
        win_end = win_start + win_samples
        sl = slice(win_start, win_end)
        t_win = t_full[sl] - t_full[win_start]

        palette = [
            COLORS["accent"],
            COLORS["primary"],
            COLORS["secondary"],
            COLORS["success"],
        ]
        base_std = np.std(sources[0, sl]) if np.std(sources[0, sl]) > 0 else 1.0
        for k in range(n_show):
            sig = sources[k, sl]
            offset = -k * base_std * 3.5
            ax.plot(
                t_win,
                sig + offset,
                color=palette[k % len(palette)],
                linewidth=0.5,
                alpha=0.85,
                label=f"Comp {k + 1}",
            )

        ax.set_xlabel("Time (s)", fontsize=FONTS["label"])
        ax.set_ylabel("Amplitude (a.u.)", fontsize=FONTS["label"])
        ax.set_yticks([])
        themed_legend(ax, loc="upper right")
    elif sources is not None and sources.size > 0:
        ax.text(
            0.5,
            0.5,
            "sfreq required for time axis",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTS["label"],
            color=COLORS["placeholder"],
        )
    else:
        ax.text(
            0.5,
            0.5,
            empty_text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTS["label"],
            color=COLORS["placeholder"],
        )

    ax.set_title(
        title,
        fontsize=FONTS["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )


def _plot_before_after_psd_panel(
    ax,
    *,
    freqs=None,
    psd_before=None,
    psd_after=None,
    line_freq=None,
    fmax=100.0,
    title="(e)  Power spectral density",
    before_label="Before",
    after_label="After",
    harmonic_count=0,
    fallback_values=None,
    fallback_message="Provide data_before / data_after\nfor PSD comparison",
):
    """Plot the repeated PSD summary panel with optional fallback bars."""
    style_axes(ax, grid=True)

    has_psd = freqs is not None and psd_before is not None and psd_after is not None
    if has_psd:
        mean_before = np.mean(psd_before, axis=0)
        mean_after = np.mean(psd_after, axis=0)

        ax.semilogy(
            freqs,
            mean_before,
            color=COLORS["before"],
            linewidth=2.0,
            linestyle="-",
            alpha=1.0,
            label=before_label,
            zorder=2,
        )
        ax.semilogy(
            freqs,
            mean_after,
            color=COLORS["after"],
            linewidth=1.2,
            linestyle="-",
            alpha=0.9,
            label=after_label,
            zorder=3,
        )

        if line_freq is not None:
            ax.axvline(
                line_freq,
                color=COLORS["accent"],
                ls=":",
                lw=0.8,
                alpha=0.8,
                label=f"{line_freq:.0f} Hz",
                zorder=4,
            )
            for h in range(2, harmonic_count + 2):
                harmonic = line_freq * h
                if harmonic < fmax:
                    ax.axvline(
                        harmonic,
                        color=COLORS["accent"],
                        ls=":",
                        lw=0.5,
                        alpha=0.25,
                    )

        ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
        ax.set_ylabel(r"PSD (V$^2\!$/Hz)", fontsize=FONTS["label"])
        ax.set_xlim(0, fmax)
        themed_legend(ax, loc="upper right")
    elif fallback_values is not None and np.size(fallback_values) > 0:
        ax.bar(
            range(len(fallback_values)),
            fallback_values,
            color=COLORS["primary"],
            edgecolor="none",
        )
        ax.axhline(np.mean(fallback_values), color=COLORS["accent"], ls="--", lw=0.9)
        ax.set_xlabel("Component", fontsize=FONTS["label"])
        ax.set_ylabel("Eigenvalue", fontsize=FONTS["label"])
    else:
        ax.text(
            0.5,
            0.5,
            fallback_message,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=FONTS["label"],
            color=COLORS["placeholder"],
        )

    ax.set_title(
        title,
        fontsize=FONTS["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )


def plot_denoising_summary(inst_orig, inst_denoised, info=None, show=True, fname=None):
    """Plot a compact denoising dashboard."""
    from matplotlib.gridspec import GridSpec

    with use_theme():
        fig = _new_summary_figure(figsize=(12, 10), constrained_layout=True)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

        ax_map = fig.add_subplot(gs[0, 0])
        plot_power_ratio_map(inst_orig, inst_denoised, info=info, ax=ax_map, show=False)

        ax_psd = fig.add_subplot(gs[0, 1])
        plot_psd_comparison(inst_orig, inst_denoised, ax=ax_psd, show=False)

        ax_gfp = fig.add_subplot(gs[1, :])

    gfp1 = _get_gfp_for_summary(inst_orig)
    gfp2 = _get_gfp_for_summary(inst_denoised)
    times = inst_orig.times

    if gfp1 is not None:
        ax_gfp.plot(
            times, gfp1, label="Original GFP", color=COLORS["before"], alpha=0.7
        )
        ax_gfp.plot(times, gfp2, label="Denoised GFP", color=COLORS["after"], alpha=0.7)
        ax_gfp.fill_between(
            times, gfp1, gfp2, color=COLORS["muted"], alpha=0.2, label="Difference"
        )
        themed_legend(ax_gfp, loc="best")
        ax_gfp.set_xlabel("Time (s)")
        ax_gfp.set_ylabel("Global Field Power")
        ax_gfp.set_title("Temporal Signal Comparison (GFP)")
        style_axes(ax_gfp, grid=True)

    fig.suptitle("Denoising Summary", fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


def plot_metric_tradeoff_summary(
    df,
    *,
    method_order=None,
    method_colors=None,
    method_labels=None,
    fname=None,
    show=True,
):
    """Two-panel benchmark figure: trade-off scatter plus R(f0) comparison."""
    if method_order is None:
        method_order = DEFAULT_METHOD_ORDER

    fig, axes = themed_figure(1, 2, figsize=(16, 6))

    plot_tradeoff_scatter(
        df,
        group_col="method",
        group_order=method_order,
        group_colors=method_colors,
        group_labels=method_labels,
        x_col="below_noise_pct",
        y_col="peak_attenuation_db",
        x_label="Sub-Peak DeltaPower (%) - closer to 0 is better",
        y_label="Peak Attenuation (dB) - higher is better",
        title="Attenuation vs Sub-Peak Distortion Trade-off",
        reference_x=0.0,
        reference_y=10.0,
        ax=axes[0],
        show=False,
    )

    plot_single_metric_comparison(
        df,
        metric_col="R_f0",
        metric_label="R(f0) - Noise-Surround Ratio",
        group_col="method",
        subject_col="subject",
        group_order=method_order,
        group_colors=method_colors,
        group_labels=method_labels,
        title="Residual Line Noise - R(f0)",
        reference_value=1.0,
        reference_label="Ideal (R=1)",
        ax=axes[1],
        show=False,
    )

    fig.suptitle("Trade-off Analysis", fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


def plot_dss_summary(
    estimator,
    data_before=None,
    data_after=None,
    sfreq=None,
    channel_names=None,
    info=None,
    max_components=4,
    fmax=None,
    title=None,
    figsize=None,
    dpi=200,
    show=True,
    fname=None,
):
    """Plot the standard DSS diagnostics dashboard."""
    if _is_segmented(estimator):
        return plot_dss_segmented_summary(
            estimator,
            data_before=data_before,
            data_after=data_after,
            sfreq=sfreq,
            channel_names=channel_names,
            info=info,
            title=title,
            figsize=figsize,
            dpi=dpi,
            show=show,
            fname=fname,
        )

    if sfreq is None:
        sfreq = getattr(estimator, "sfreq", None)
        if sfreq is None:
            bias = getattr(estimator, "bias", None)
            sfreq = getattr(bias, "sfreq", None)

    if info is None:
        info = _get_info(estimator)

    eigenvalues = _get_eigenvalues(estimator)
    patterns = getattr(estimator, "patterns_", None)
    n_selected = _get_n_selected(estimator)
    removed = _get_dss_removed(estimator)
    sources = getattr(estimator, "sources_", None)
    has_psd = data_before is not None and data_after is not None and sfreq is not None

    if removed is None and data_before is not None and data_after is not None:
        removed = data_before - data_after

    if sources is None and data_before is not None:
        try:
            old_rt = getattr(estimator, "return_type", None)
            if old_rt is not None:
                estimator.return_type = "sources"
            sources = estimator.transform(data_before)
            if old_rt is not None:
                estimator.return_type = old_rt
        except Exception:
            sources = None

    if fmax is None:
        fmax = min(100.0, sfreq / 2) if sfreq else 100.0

    freqs = psd_before = psd_after = None
    if has_psd:
        nperseg = min(data_before.shape[1], int(sfreq * 2))
        freqs, psd_before = signal.welch(data_before, sfreq, nperseg=nperseg)
        _, psd_after = signal.welch(data_after, sfreq, nperseg=nperseg)

    fig, gs = _new_summary_grid(figsize=figsize, dpi=dpi, hspace=0.42)

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_selection_eigenvalues_panel(
        ax_a,
        eigenvalues,
        n_selected,
        selected_label=f"Selected ({n_selected})",
        title="(a)  DSS eigenvalues",
    )

    _plot_patterns_panel(
        fig,
        gs[0, 1],
        patterns,
        info=info,
        channel_names=channel_names,
        max_components=max_components,
        title="(b)  Spatial mixing patterns",
    )

    ax_c = fig.add_subplot(gs[1, 0])
    _plot_removed_power_panel(
        fig,
        ax_c,
        removed,
        info=info,
        channel_names=channel_names,
        title="(c)  Removed artifact power",
        bar_color=COLORS["accent"],
    )

    ax_d = fig.add_subplot(gs[1, 1])
    _plot_source_trace_panel(ax_d, sources, sfreq=sfreq, title="(d)  DSS sources (2 s)")

    bias = getattr(estimator, "bias", None)
    bias_freq = getattr(bias, "freq", None) or getattr(bias, "line_freq", None)
    ax_e = fig.add_subplot(gs[2, 0])
    _plot_before_after_psd_panel(
        ax_e,
        freqs=freqs,
        psd_before=psd_before,
        psd_after=psd_after,
        line_freq=bias_freq,
        fmax=fmax,
        title="(e)  Power spectral density",
        fallback_values=eigenvalues,
    )

    ax_f = fig.add_subplot(gs[2, 1])

    bias_name = _get_bias_name(estimator)
    rows = [
        ("Bias function", bias_name),
        ("Mode", "Standard"),
        ("n_components", str(getattr(estimator, "n_components", "N/A"))),
        ("Components selected", str(n_selected) if n_selected else "N/A"),
        ("Selection method", str(getattr(estimator, "selection_method", "N/A"))),
    ]

    smooth = getattr(estimator, "smooth", None)
    if smooth is not None:
        if isinstance(smooth, int):
            rows.append(("Smoothing", f"window={smooth} samples"))
        else:
            rows.append(("Smoothing", type(smooth).__name__))

    if eigenvalues is not None and eigenvalues.size > 0:
        rows.append(("Max eigenvalue", f"{eigenvalues[0]:.4f}"))
        if n_selected > 0 and n_selected < len(eigenvalues):
            rows.append(
                (
                    "Eigen gap (sel/next)",
                    f"{eigenvalues[n_selected - 1]:.4f} / {eigenvalues[n_selected]:.4f}",
                )
            )

    if has_psd and freqs is not None:
        bias = getattr(estimator, "bias", None)
        bias_freq = getattr(bias, "freq", None) or getattr(bias, "line_freq", None)
        if bias_freq is not None:
            idx = np.argmin(np.abs(freqs - bias_freq))
            pb = np.mean(psd_before[:, idx])
            pa = np.mean(psd_after[:, idx])
            if pa > 0 and pb > 0:
                reduction_db = 10 * np.log10(pb / pa)
                rows.append(
                    (f"Power reduction @ {bias_freq:.0f} Hz", f"{reduction_db:.1f} dB")
                )

    if removed is not None and np.any(removed != 0) and data_before is not None:
        total_var = np.var(data_before)
        removed_var = np.var(removed)
        if total_var > 0:
            pct = 100 * removed_var / total_var
            rows.append(("Variance removed", f"{pct:.4f} %"))

    _draw_summary_table(ax_f, rows)

    if title is None:
        title = f"DSS Diagnostics ({bias_name})"
    fig.suptitle(
        title,
        fontsize=FONTS["suptitle"],
        fontweight="bold",
        color=COLORS["text"],
    )
    return _finalize_fig(fig, show=show, fname=fname)


def plot_dss_segmented_summary(
    estimator,
    data_before=None,
    data_after=None,
    sfreq=None,
    channel_names=None,
    info=None,
    fmax=None,
    title=None,
    figsize=None,
    dpi=200,
    show=True,
    fname=None,
):
    """Plot the segmented DSS diagnostics dashboard."""
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch

    seg_results = _get_segment_results(estimator)
    if not seg_results:
        return plot_dss_summary(
            estimator,
            data_before=data_before,
            data_after=data_after,
            sfreq=sfreq,
            channel_names=channel_names,
            info=info,
            title=title,
            figsize=figsize,
            dpi=dpi,
            show=show,
            fname=fname,
        )

    if sfreq is None:
        sfreq = getattr(estimator, "sfreq", None)
        if sfreq is None:
            bias = getattr(estimator, "bias", None)
            sfreq = getattr(bias, "sfreq", None)

    if info is None:
        info = _get_info(estimator)

    removed = _get_dss_removed(estimator)
    n_segments = len(seg_results)
    has_psd = data_before is not None and data_after is not None and sfreq is not None

    if removed is None and data_before is not None and data_after is not None:
        removed = data_before - data_after

    if fmax is None:
        fmax = min(100.0, sfreq / 2) if sfreq else 100.0

    per_seg_selected = [s.get("n_selected", s.get("n_removed", 0)) for s in seg_results]
    per_seg_eigenvalues = [s.get("eigenvalues", np.array([])) for s in seg_results]

    freqs = psd_before = psd_after = None
    if has_psd:
        nperseg = min(data_before.shape[1], int(sfreq * 2))
        freqs, psd_before = signal.welch(data_before, sfreq, nperseg=nperseg)
        _, psd_after = signal.welch(data_after, sfreq, nperseg=nperseg)

    C = COLORS
    F = FONTS
    fig, gs = _new_summary_grid(figsize=figsize, dpi=dpi, hspace=0.40)

    ax_a = fig.add_subplot(gs[0, 0])
    style_axes(ax_a, grid=True)

    ax_a.bar(
        range(n_segments),
        per_seg_selected,
        color=C["primary"],
        edgecolor="none",
        width=0.75,
    )
    mean_sel = np.mean(per_seg_selected)
    ax_a.axhline(mean_sel, color=C["accent"], ls="--", lw=0.9, zorder=4)

    ax_a.set_xlabel("Segment index", fontsize=F["label"])
    ax_a.set_ylabel("Components selected", fontsize=F["label"])
    ax_a.set_title(
        "(a)  Components selected per segment",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )
    ax_a.set_xlim(-0.6, n_segments - 0.4)
    ax_a.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    themed_legend(
        ax_a,
        loc="upper left",
        handles=[
            Patch(fc=C["primary"], label="Selected"),
            plt.Line2D(
                [],
                [],
                color=C["accent"],
                ls="--",
                lw=0.9,
                label=f"Mean ({mean_sel:.1f})",
            ),
        ],
    )

    ax_b = fig.add_subplot(gs[0, 1])
    style_axes(ax_b)

    max_n_eig = max((len(e) for e in per_seg_eigenvalues if len(e) > 0), default=0)

    if max_n_eig > 0:
        n_show_eig = min(max_n_eig, 10)
        eig_matrix = np.full((n_segments, n_show_eig), np.nan)
        for i, ev in enumerate(per_seg_eigenvalues):
            n = min(len(ev), n_show_eig)
            eig_matrix[i, :n] = ev[:n]

        im = ax_b.imshow(
            eig_matrix.T,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            origin="lower",
        )
        cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
        cbar.set_label("Eigenvalue", fontsize=F["tick"])
        cbar.ax.tick_params(labelsize=F["tick"] - 1)

        ax_b.set_xlabel("Segment index", fontsize=F["label"])
        ax_b.set_ylabel("Component", fontsize=F["label"])

        for i, ns in enumerate(per_seg_selected):
            if 0 < ns <= n_show_eig:
                ax_b.plot(i, ns - 0.5, "w_", markersize=6, markeredgewidth=1.5)
    else:
        ax_b.text(
            0.5,
            0.5,
            "No eigenvalues available",
            ha="center",
            va="center",
            transform=ax_b.transAxes,
            fontsize=F["label"],
            color=COLORS["placeholder"],
        )

    ax_b.set_title(
        "(b)  Eigenvalue heatmap",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )

    ax_c = fig.add_subplot(gs[1, 0])
    style_axes(ax_c)

    if sfreq is not None and sfreq > 0:
        total_dur = (
            seg_results[-1].get("end", seg_results[-1].get("end_sample", 0)) / sfreq
        )

        for _i, s in enumerate(seg_results):
            s_t = s.get("start", s.get("start_sample", 0)) / sfreq
            e_t = s.get("end", s.get("end_sample", 0)) / sfreq
            ns = s.get("n_selected", s.get("n_removed", 0))

            col = C["primary"] if ns > 0 else C["muted"]
            ax_c.barh(
                0,
                e_t - s_t,
                left=s_t,
                height=0.7,
                color=col,
                edgecolor="white",
                linewidth=0.25,
            )

            seg_frac = (e_t - s_t) / max(total_dur, 1e-9)
            if seg_frac > 0.018:
                tc = "white" if ns > 0 else C["text"]
                ax_c.text(
                    (s_t + e_t) / 2,
                    0,
                    str(ns),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=tc,
                    fontweight="bold",
                )

        ax_c.set_xlabel("Time (s)", fontsize=F["label"])
        ax_c.set_yticks([])
        ax_c.set_ylim(-0.5, 0.5)
        themed_legend(
            ax_c,
            loc="upper right",
            handles=[
                Patch(fc=C["primary"], label="Artifact cleaned"),
                Patch(fc=C["muted"], label="No artifact"),
            ],
        )
    else:
        ax_c.text(
            0.5,
            0.5,
            "sfreq required",
            ha="center",
            va="center",
            transform=ax_c.transAxes,
            fontsize=F["label"],
            color=COLORS["placeholder"],
        )

    ax_c.set_title(
        "(c)  Segment boundaries",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )

    ax_d = fig.add_subplot(gs[1, 1])
    _plot_removed_power_panel(
        fig,
        ax_d,
        removed,
        info=info,
        channel_names=channel_names,
        title="(d)  Removed artifact power",
        bar_color=COLORS["primary"],
        no_data_text="No removal data",
    )

    ax_e = fig.add_subplot(gs[2, 0])
    bias = getattr(estimator, "bias", None)
    bias_freq = getattr(bias, "freq", None) or getattr(bias, "line_freq", None)
    _plot_before_after_psd_panel(
        ax_e,
        freqs=freqs,
        psd_before=psd_before,
        psd_after=psd_after,
        line_freq=bias_freq,
        fmax=fmax,
        title="(e)  Power spectral density",
        fallback_values=_get_eigenvalues(estimator),
    )

    ax_f = fig.add_subplot(gs[2, 1])

    bias_name = _get_bias_name(estimator)
    total_selected = sum(per_seg_selected)

    rows = [
        ("Bias function", bias_name),
        ("Mode", "Segmented"),
        ("Number of segments", str(n_segments)),
        ("Total components selected", str(total_selected)),
        ("Per-segment min / max", f"{min(per_seg_selected)} / {max(per_seg_selected)}"),
        (
            "Per-segment mean +/- std",
            f"{np.mean(per_seg_selected):.1f} +/- {np.std(per_seg_selected):.1f}",
        ),
    ]

    all_max = [ev[0] for ev in per_seg_eigenvalues if len(ev) > 0]
    if all_max:
        rows.append(
            ("Max eigenvalue range", f"{min(all_max):.3f} - {max(all_max):.3f}")
        )

    smooth = getattr(estimator, "smooth", None)
    if smooth is not None:
        if isinstance(smooth, int):
            rows.append(("Smoothing", f"window={smooth} samples"))
        else:
            rows.append(("Smoothing", type(smooth).__name__))

    if has_psd and freqs is not None:
        bias = getattr(estimator, "bias", None)
        bias_freq = getattr(bias, "freq", None) or getattr(bias, "line_freq", None)
        if bias_freq is not None:
            idx = np.argmin(np.abs(freqs - bias_freq))
            pb = np.mean(psd_before[:, idx])
            pa = np.mean(psd_after[:, idx])
            if pa > 0 and pb > 0:
                reduction_db = 10 * np.log10(pb / pa)
                rows.append(
                    (f"Power reduction @ {bias_freq:.0f} Hz", f"{reduction_db:.1f} dB")
                )

    if removed is not None and np.any(removed != 0) and data_before is not None:
        total_var = np.var(data_before)
        removed_var = np.var(removed)
        if total_var > 0:
            pct = 100 * removed_var / total_var
            rows.append(("Variance removed", f"{pct:.4f} %"))

    _draw_summary_table(ax_f, rows)

    if title is None:
        title = f"Segmented DSS Diagnostics ({bias_name})"
    fig.suptitle(title, fontsize=F["suptitle"], fontweight="bold", color=C["text"])
    return _finalize_fig(fig, show=show, fname=fname)


def plot_dss_mode_comparison(
    bias,
    data,
    *,
    n_components=20,
    n_select="auto",
    selection_method="combined",
    smooth=None,
    max_prop_remove=0.2,
    min_select=1,
    line_freq=50.0,
    n_harmonics=3,
    title=None,
    figsize=(16, 5),
    show=True,
    fname=None,
):
    """Run and compare DSS in plain, smoothed, and segmented modes."""
    from ..dss.linear import DSS

    sfreq = data.info["sfreq"]
    data_orig = data.get_data()

    if smooth is None:
        smooth = int(sfreq / line_freq)

    bias_name = type(bias).__name__

    print(f"DSS + {bias_name} - 3-Way Comparison")
    print("=" * 65)

    dss_plain = DSS(
        bias=bias,
        n_components=n_components,
        n_select=n_select,
        selection_method=selection_method,
        return_type="raw",
    )
    raw_plain = dss_plain.fit_transform(data)
    data_plain = raw_plain.get_data()
    ev_plain = dss_plain.eigenvalues_
    n_plain = dss_plain.n_selected_
    print(
        f"  A) Plain DSS:          {n_plain} comp(s) | "
        f"max eigenvalue = {ev_plain[0]:.6f}"
    )

    dss_smooth = DSS(
        bias=bias,
        n_components=n_components,
        n_select=n_select,
        selection_method=selection_method,
        smooth=smooth,
        return_type="raw",
    )
    raw_smooth = dss_smooth.fit_transform(data)
    data_smooth = raw_smooth.get_data()
    ev_smooth = dss_smooth.eigenvalues_
    n_smooth = dss_smooth.n_selected_
    boost = ev_smooth[0] / ev_plain[0] if ev_plain[0] > 0 else float("inf")
    print(
        f"  B) + Smoothing:        {n_smooth} comp(s) | "
        f"max eigenvalue = {ev_smooth[0]:.6f} ({boost:.1f}?? boost)"
    )

    dss_seg = DSS(
        bias=bias,
        n_components=n_components,
        n_select=n_select,
        selection_method=selection_method,
        smooth=smooth,
        segmented=True,
        max_prop_remove=max_prop_remove,
        min_select=min_select,
    )
    raw_seg = dss_seg.fit_transform(data)
    data_seg = raw_seg.get_data()
    n_segments = len(dss_seg.segment_results_)
    seg_n_sel = [s["n_selected"] for s in dss_seg.segment_results_]
    seg_evals = [s["eigenvalues"][0] for s in dss_seg.segment_results_]
    total_removed = sum(seg_n_sel)
    print(
        f"  C) + Segmentation:     {n_segments} seg(s) | "
        f"n_removed: {min(seg_n_sel)}-{max(seg_n_sel)} "
        f"(total {total_removed}) | "
        f"eigenvalue range: {min(seg_evals):.4f}-{max(seg_evals):.4f}"
    )

    nperseg = int(sfreq * 4)
    f_psd, psd_orig = signal.welch(data_orig, fs=sfreq, nperseg=nperseg, axis=-1)
    _, psd_A = signal.welch(data_plain, fs=sfreq, nperseg=nperseg, axis=-1)
    _, psd_B = signal.welch(data_smooth, fs=sfreq, nperseg=nperseg, axis=-1)
    _, psd_C = signal.welch(data_seg, fs=sfreq, nperseg=nperseg, axis=-1)

    labels = ["(A) Plain", "(B) + Smooth", "(C) + Smooth + Seg"]
    cleaned_list = [data_plain, data_smooth, data_seg]
    psd_list = [psd_A, psd_B, psd_C]

    harmonics = [line_freq * k for k in range(1, n_harmonics + 1)]
    harm_labels = [f"SR@{int(h)}" for h in harmonics]

    header = f"{'Method':<22} " + "  ".join(f"{h:>7}" for h in harm_labels)
    header += f" {'Mean SR':>8} {'SD':>5} {'Var%':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    all_metrics = []
    for label, cleaned, psd_after in zip(labels, cleaned_list, psd_list):
        srs = [suppression_ratio(f_psd, psd_orig, psd_after, h) for h in harmonics]
        sr_mean = np.mean(srs)
        sd = spectral_distortion(
            f_psd,
            psd_orig,
            psd_after,
            line_freq=line_freq,
            n_harmonics=n_harmonics,
        )
        var_pct = variance_removed(data_orig, cleaned)
        metrics = {
            "sr_harmonics": srs,
            "sr_mean": sr_mean,
            "sd": sd,
            "var_pct": var_pct,
        }
        all_metrics.append(metrics)

        sr_str = "  ".join(f"{s:>7.1f}" for s in srs)
        print(f"{label:<22} {sr_str}  {sr_mean:>7.1f}  {sd:>4.2f}  {var_pct:>5.3f}")

    fig, axes = themed_figure(1, 2, figsize=figsize)

    ax = axes[0]
    fmask = f_psd <= 160
    ax.semilogy(
        f_psd[fmask],
        psd_orig.mean(0)[fmask],
        color=COLORS["before"],
        lw=2,
        alpha=0.4,
        label="Original",
    )
    ax.semilogy(
        f_psd[fmask],
        psd_A.mean(0)[fmask],
        color=COLORS["primary"],
        lw=1.2,
        alpha=0.7,
        label="(A) Plain",
    )
    ax.semilogy(
        f_psd[fmask],
        psd_B.mean(0)[fmask],
        color=COLORS["secondary"],
        lw=1.2,
        alpha=0.7,
        label="(B) + Smooth",
    )
    ax.semilogy(
        f_psd[fmask],
        psd_C.mean(0)[fmask],
        color=COLORS["accent"],
        lw=1.5,
        alpha=0.9,
        label="(C) + Smooth + Seg",
    )
    for k in range(1, n_harmonics + 1):
        ax.axvline(line_freq * k, color=COLORS["line_marker"], ls="--", alpha=0.3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(f"PSD: {bias_name} - 3 Modes")
    themed_legend(ax, fontsize=9)
    style_axes(ax, grid=True)

    ax = axes[1]
    zoom = (f_psd >= line_freq - 5) & (f_psd <= line_freq + 5)
    ax.semilogy(
        f_psd[zoom],
        psd_orig.mean(0)[zoom],
        color=COLORS["before"],
        lw=2,
        alpha=0.4,
        label="Original",
    )
    ax.semilogy(
        f_psd[zoom],
        psd_A.mean(0)[zoom],
        color=COLORS["primary"],
        lw=1.5,
        alpha=0.7,
        label="(A) Plain",
    )
    ax.semilogy(
        f_psd[zoom],
        psd_B.mean(0)[zoom],
        color=COLORS["secondary"],
        lw=1.5,
        alpha=0.7,
        label="(B) + Smooth",
    )
    ax.semilogy(
        f_psd[zoom],
        psd_C.mean(0)[zoom],
        color=COLORS["accent"],
        lw=2,
        alpha=0.9,
        label="(C) + Smooth + Seg",
    )
    ax.axvline(
        line_freq,
        color=COLORS["line_marker"],
        ls="--",
        alpha=0.5,
        label=f"{int(line_freq)} Hz",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(f"Zoom: {int(line_freq - 5)}-{int(line_freq + 5)} Hz")
    themed_legend(ax, fontsize=9)
    style_axes(ax, grid=True)

    if title is None:
        title = f"{bias_name} - Plain vs Smoothing vs Segmented"
    fig.suptitle(title, fontsize=13, y=1.02)
    fig = _finalize_fig(fig, show=show, fname=fname)

    results = {
        "plain": {
            "estimator": dss_plain,
            "cleaned_raw": raw_plain,
            "data": data_plain,
            "metrics": all_metrics[0],
        },
        "smooth": {
            "estimator": dss_smooth,
            "cleaned_raw": raw_smooth,
            "data": data_smooth,
            "metrics": all_metrics[1],
        },
        "segmented": {
            "estimator": dss_seg,
            "cleaned_raw": raw_seg,
            "data": data_seg,
            "metrics": all_metrics[2],
        },
    }
    return fig, results


def plot_zapline_cleaning_summary(
    data_before: np.ndarray,
    data_after: np.ndarray,
    estimator,
    sfreq: float,
    line_freq: float | None = None,
    show: bool = True,
    fname: str | None = None,
) -> plt.Figure:
    """Create a combined ZapLine cleaning summary."""
    fig, axes = themed_figure(2, 2, figsize=(12, 8))

    plot_psd_comparison(
        data_before,
        data_after,
        sfreq=sfreq,
        line_freq=line_freq,
        ax=axes[0, 0],
        show=False,
    )

    plot_component_score_curve(estimator, ax=axes[0, 1], show=False)
    plot_component_patterns(estimator, ax=axes[1, 0], show=False)

    ax = axes[1, 1]
    ax.axis("off")

    stats = []
    if line_freq is not None:
        stats.append(f"Line Frequency: {line_freq:.1f} Hz")

    n_removed = getattr(estimator, "n_removed_", 0)
    stats.append(f"Components Removed: {n_removed}")

    n_harmonics = getattr(estimator, "n_harmonics_", None)
    if n_harmonics is not None:
        stats.append(f"Harmonics: {n_harmonics}")

    if line_freq is not None:
        nperseg = min(data_before.shape[1], int(sfreq * 2))
        freqs, psd_before = signal.welch(data_before, sfreq, nperseg=nperseg)
        _, psd_after = signal.welch(data_after, sfreq, nperseg=nperseg)
        idx = np.argmin(np.abs(freqs - line_freq))
        power_before = np.mean(psd_before[:, idx])
        power_after = np.mean(psd_after[:, idx])
        if power_after > 0:
            reduction_db = 10 * np.log10(power_before / power_after)
            stats.append(f"Power Reduction: {reduction_db:.1f} dB")

    ax.text(
        0.1,
        0.8,
        "\n".join(stats),
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": COLORS["light_gray"], "alpha": 0.5},
    )
    ax.set_title("Summary Statistics")

    fig.suptitle("ZapLine Cleaning Summary", fontsize=14, fontweight="bold")

    return _finalize_fig(fig, show=show, fname=fname)


def plot_zapline_adaptive_summary(
    result,
    data_before=None,
    data_after=None,
    sfreq=None,
    channel_names=None,
    info=None,
    title=None,
    figsize=None,
    dpi=200,
    show=True,
    fname=None,
):
    """Plot the adaptive ZapLine diagnostics dashboard."""
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch

    chunk_info = _get_chunk_info(result)
    if not chunk_info:
        chunk_info = _fallback_chunk_info(result)

    if sfreq is None:
        sfreq = getattr(result, "sfreq", None)

    removed = _get_zapline_removed(result)
    n_chunks = len(chunk_info)
    has_psd = data_before is not None and data_after is not None and sfreq is not None
    line_freq = getattr(result, "line_freq", None)

    per_chunk_removed = [c.get("n_removed", 0) for c in chunk_info]
    fine_freqs = [c.get("fine_freq", c.get("frequency", 0)) for c in chunk_info]
    coarse_freqs = [c.get("frequency", 0) for c in chunk_info]
    artifact_present = [c.get("artifact_present", True) for c in chunk_info]

    freqs = psd_before = psd_after = None
    if has_psd:
        nperseg = min(data_before.shape[1], int(sfreq * 2))
        freqs, psd_before = signal.welch(data_before, sfreq, nperseg=nperseg)
        _, psd_after = signal.welch(data_after, sfreq, nperseg=nperseg)

    C = COLORS
    F = FONTS

    fig, gs = _new_summary_grid(figsize=figsize, dpi=dpi, hspace=0.40)

    ax_a = fig.add_subplot(gs[0, 0])
    style_axes(ax_a, grid=True)

    bar_colors = [C["primary"] if ap else C["muted"] for ap in artifact_present]
    ax_a.bar(
        range(n_chunks),
        per_chunk_removed,
        color=bar_colors,
        edgecolor="none",
        width=0.75,
    )
    mean_val = np.mean(per_chunk_removed)
    ax_a.axhline(mean_val, color=C["accent"], ls="--", lw=0.9, zorder=4)

    ax_a.set_xlabel("Segment index", fontsize=F["label"])
    ax_a.set_ylabel("Components removed", fontsize=F["label"])
    ax_a.set_title(
        "(a)  Components removed per segment",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )
    ax_a.set_xlim(-0.6, n_chunks - 0.4)
    ax_a.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    themed_legend(
        ax_a,
        loc="upper left",
        handles=[
            Patch(fc=C["primary"], label="Artifact present"),
            Patch(fc=C["muted"], label="No artifact"),
            plt.Line2D(
                [],
                [],
                color=C["accent"],
                ls="--",
                lw=0.9,
                label=f"Mean ({mean_val:.1f})",
            ),
        ],
    )

    ax_b = fig.add_subplot(gs[0, 1])
    style_axes(ax_b, grid=True)

    ax_b.plot(
        range(n_chunks),
        fine_freqs,
        color=C["primary"],
        marker="o",
        markersize=2.2,
        linewidth=0.7,
        label="Fine-tuned",
        zorder=3,
    )
    if any(abs(f - ff) > 1e-6 for f, ff in zip(coarse_freqs, fine_freqs)):
        ax_b.axhline(
            coarse_freqs[0],
            color=C["secondary"],
            ls="--",
            lw=1.0,
            label=f"Nominal ({coarse_freqs[0]:.0f} Hz)",
            zorder=2,
        )

    ax_b.set_xlabel("Segment index", fontsize=F["label"])
    ax_b.set_ylabel("Frequency (Hz)", fontsize=F["label"])
    ax_b.set_title(
        "(b)  Detected peak frequency",
        fontsize=F["title"],
        fontweight="semibold",
        loc="left",
        pad=6,
    )
    ax_b.set_xlim(-0.6, n_chunks - 0.4)

    freq_range = max(fine_freqs) - min(fine_freqs)
    freq_pad = max(0.02, freq_range * 0.20)
    ax_b.set_ylim(min(fine_freqs) - freq_pad, max(fine_freqs) + freq_pad)
    themed_legend(ax_b, loc="upper left")

    ax_c = fig.add_subplot(gs[1, 0])
    style_axes(ax_c)

    if sfreq is not None and sfreq > 0:
        total_dur = chunk_info[-1]["end"] / sfreq
        for _i, c in enumerate(chunk_info):
            s_t = c["start"] / sfreq
            e_t = c["end"] / sfreq
            ap = c.get("artifact_present", True)
            col = C["primary"] if ap else C["muted"]
            ax_c.barh(
                0,
                e_t - s_t,
                left=s_t,
                height=0.7,
                color=col,
                edgecolor="white",
                linewidth=0.25,
            )
            seg_frac = (e_t - s_t) / total_dur
            if seg_frac > 0.018:
                n_r = c.get("n_removed", 0)
                tc = "white" if ap else C["text"]
                ax_c.text(
                    (s_t + e_t) / 2,
                    0,
                    str(n_r),
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=tc,
                    fontweight="bold",
                )

        ax_c.set_xlabel("Time (s)", fontsize=F["label"])
        ax_c.set_yticks([])
        ax_c.set_ylim(-0.5, 0.5)
        ax_c.set_title(
            "(c)  Segment boundaries",
            fontsize=F["title"],
            fontweight="semibold",
            loc="left",
            pad=6,
        )
        themed_legend(
            ax_c,
            loc="upper right",
            handles=[
                Patch(fc=C["primary"], label="Artifact present"),
                Patch(fc=C["muted"], label="No artifact"),
            ],
        )
    else:
        ax_c.text(
            0.5,
            0.5,
            "sfreq required",
            ha="center",
            va="center",
            transform=ax_c.transAxes,
            fontsize=F["label"],
            color=COLORS["placeholder"],
        )
        ax_c.set_title(
            "(c)  Segment boundaries",
            fontsize=F["title"],
            fontweight="semibold",
            loc="left",
            pad=6,
        )

    ax_d = fig.add_subplot(gs[1, 1])
    _plot_removed_power_panel(
        fig,
        ax_d,
        removed,
        info=info,
        channel_names=channel_names,
        title="(d)  Removed artifact power",
        bar_color=COLORS["primary"],
        no_data_text="No removal data",
    )

    ax_e = fig.add_subplot(gs[2, 0])
    _plot_before_after_psd_panel(
        ax_e,
        freqs=freqs,
        psd_before=psd_before,
        psd_after=psd_after,
        line_freq=line_freq,
        fmax=100.0,
        title="(e)  Power spectral density",
        harmonic_count=3,
        fallback_values=getattr(result, "eigenvalues_", None),
    )

    ax_f = fig.add_subplot(gs[2, 1])

    total_removed = _get_zapline_n_removed(result)
    n_present = sum(1 for ap in artifact_present if ap)
    cleaned = _get_cleaned(result)

    rows = [
        ("Line frequency", f"{line_freq:.1f} Hz" if line_freq else "auto"),
        ("Harmonics", str(getattr(result, "n_harmonics", "None"))),
        ("Number of segments", str(n_chunks)),
        ("Segments with artifact", f"{n_present} / {n_chunks}"),
        ("Total components removed", str(total_removed)),
        (
            "Per-segment min / max",
            f"{min(per_chunk_removed)} / {max(per_chunk_removed)}",
        ),
        (
            "Per-segment mean +/- std",
            f"{np.mean(per_chunk_removed):.1f} +/- {np.std(per_chunk_removed):.1f}",
        ),
        ("Fine freq range", f"{min(fine_freqs):.2f} - {max(fine_freqs):.2f} Hz"),
    ]

    if has_psd and line_freq is not None and freqs is not None:
        idx = np.argmin(np.abs(freqs - line_freq))
        pb = np.mean(psd_before[:, idx])
        pa = np.mean(psd_after[:, idx])
        if pa > 0 and pb > 0:
            reduction_db = 10 * np.log10(pb / pa)
            rows.append(("Power reduction at line freq", f"{reduction_db:.1f} dB"))

    if cleaned is not None and removed is not None and np.any(removed != 0):
        total_var = np.var(cleaned + removed)
        removed_var = np.var(removed)
        if total_var > 0:
            pct = 100 * removed_var / total_var
            rows.append(("Variance removed", f"{pct:.4f} %"))

    _draw_summary_table(ax_f, rows)

    if title is None:
        title = "Adaptive ZapLine+ Diagnostics"
    fig.suptitle(title, fontsize=F["suptitle"], fontweight="bold", color=C["text"])
    return _finalize_fig(fig, show=show, fname=fname)


def plot_zapline_summary(
    result,
    data_before=None,
    data_after=None,
    sfreq=None,
    channel_names=None,
    info=None,
    max_components=4,
    title=None,
    figsize=None,
    dpi=200,
    show=True,
    fname=None,
):
    """Plot the standard ZapLine diagnostics dashboard."""
    if _is_adaptive(result):
        return plot_zapline_adaptive_summary(
            result,
            data_before=data_before,
            data_after=data_after,
            sfreq=sfreq,
            channel_names=channel_names,
            title=title,
            figsize=figsize,
            dpi=dpi,
            show=show,
            fname=fname,
        )

    if sfreq is None:
        sfreq = getattr(result, "sfreq", None)
    line_freq = getattr(result, "line_freq", None)
    eigenvalues = getattr(result, "eigenvalues_", None)
    patterns = getattr(result, "patterns_", None)
    n_removed = _get_zapline_n_removed(result)
    removed = _get_zapline_removed(result)
    sources = getattr(result, "sources_", None)
    has_psd = data_before is not None and data_after is not None and sfreq is not None

    freqs = psd_before = psd_after = None
    if has_psd:
        nperseg = min(data_before.shape[1], int(sfreq * 2))
        freqs, psd_before = signal.welch(data_before, sfreq, nperseg=nperseg)
        _, psd_after = signal.welch(data_after, sfreq, nperseg=nperseg)

    fig, gs = _new_summary_grid(figsize=figsize, dpi=dpi, hspace=0.42)

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_selection_eigenvalues_panel(
        ax_a,
        eigenvalues,
        n_removed,
        selected_label=f"Removed ({n_removed})",
        title="(a)  DSS eigenvalues",
    )

    _plot_patterns_panel(
        fig,
        gs[0, 1],
        patterns,
        info=info,
        channel_names=channel_names,
        max_components=max_components,
        title="(b)  Spatial mixing patterns",
    )

    ax_c = fig.add_subplot(gs[1, 0])
    _plot_removed_power_panel(
        fig,
        ax_c,
        removed,
        info=info,
        channel_names=channel_names,
        title="(c)  Removed artifact power",
        bar_color=COLORS["accent"],
    )

    ax_d = fig.add_subplot(gs[1, 1])
    _plot_source_trace_panel(
        ax_d,
        sources,
        sfreq=sfreq,
        title="(d)  Artifact sources (2 s)",
    )

    ax_e = fig.add_subplot(gs[2, 0])
    _plot_before_after_psd_panel(
        ax_e,
        freqs=freqs,
        psd_before=psd_before,
        psd_after=psd_after,
        line_freq=line_freq,
        fmax=100.0,
        title="(e)  Power spectral density",
        harmonic_count=getattr(result, "n_harmonics_", None) or 0,
    )

    ax_f = fig.add_subplot(gs[2, 1])

    _get_cleaned(result)
    rows = [
        ("Mode", "Standard"),
        ("Line frequency", f"{line_freq:.1f} Hz" if line_freq else "N/A"),
        ("Harmonics", str(getattr(result, "n_harmonics_", "N/A"))),
        ("Components removed", str(n_removed)),
        ("n_remove setting", str(getattr(result, "n_remove", "N/A"))),
        ("Threshold (sigma)", f"{getattr(result, 'threshold', 'N/A')}"),
        (
            "nkeep / rank",
            f"{getattr(result, 'nkeep', 'N/A')} / {getattr(result, 'rank', 'N/A')}",
        ),
    ]

    if eigenvalues is not None and eigenvalues.size > 0:
        rows.append(("Max eigenvalue", f"{eigenvalues[0]:.4f}"))
        if n_removed > 0 and n_removed < len(eigenvalues):
            rows.append(
                (
                    "Eigen gap (removed/next)",
                    f"{eigenvalues[n_removed - 1]:.4f} / {eigenvalues[n_removed]:.4f}",
                )
            )

    if has_psd and line_freq is not None and freqs is not None:
        idx = np.argmin(np.abs(freqs - line_freq))
        pb = np.mean(psd_before[:, idx])
        pa = np.mean(psd_after[:, idx])
        if pa > 0 and pb > 0:
            reduction_db = 10 * np.log10(pb / pa)
            rows.append(("Power reduction at line freq", f"{reduction_db:.1f} dB"))

    if removed is not None and np.any(removed != 0) and data_before is not None:
        total_var = np.var(data_before)
        removed_var = np.var(removed)
        if total_var > 0:
            pct = 100 * removed_var / total_var
            rows.append(("Variance removed", f"{pct:.4f} %"))

    _draw_summary_table(ax_f, rows)

    if title is None:
        title = "ZapLine Diagnostics"
    fig.suptitle(
        title,
        fontsize=FONTS["suptitle"],
        fontweight="bold",
        color=COLORS["text"],
    )
    return _finalize_fig(fig, show=show, fname=fname)


def plot_erp_signal_diagnostics(
    pipe_epochs,
    pipe_evokeds,
    *,
    channels=("Cz", "Pz"),
    dev_mask=None,
    std_mask=None,
    sfreq=250.0,
    erp_windows=None,
    fmax=45.0,
    subject="",
    pipe_order=None,
    pipe_colors=None,
    pipe_labels=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Plot the ERP signal diagnostics dashboard."""
    if pipe_order is None:
        pipe_order = sorted(pipe_epochs.keys())
    if figsize is None:
        figsize = (16, 14)

    n_ch = len(channels)
    fig, axes = themed_figure(3, n_ch, figsize=figsize)
    if n_ch == 1:
        axes = axes[:, np.newaxis]

    for col, ch_name in enumerate(channels):
        ax = axes[0, col]
        for ptag in pipe_order:
            ep = pipe_epochs[ptag]
            data = ep.get_data() if hasattr(ep, "get_data") else ep
            ci = _ch_index(ep, ch_name)
            mean_data = data.mean(axis=0)
            nperseg = min(256, data.shape[-1])
            freqs, psd = _welch(mean_data[ci], fs=sfreq, nperseg=nperseg)
            ax.semilogy(
                freqs,
                psd,
                color=_pipe_color(ptag, pipe_colors),
                lw=1.5,
                alpha=0.8,
                label=_pipe_label(ptag, pipe_labels),
            )
        ax.set_xlabel("Frequency (Hz)", fontsize=FONTS["label"])
        ax.set_ylabel("PSD (V^2/Hz)", fontsize=FONTS["label"])
        ax.set_title(f"Epoch-Averaged PSD at {ch_name}", fontsize=FONTS["title"])
        ax.set_xlim(0, fmax)
        themed_legend(ax)
        style_axes(ax)

    for col, ch_name in enumerate(channels):
        ax = axes[1, col]
        for ptag in pipe_order:
            ev = pipe_evokeds[ptag]
            ci = ev.ch_names.index(ch_name)
            ax.plot(
                ev.times * 1000,
                ev.data[ci] * 1e6,
                color=_pipe_color(ptag, pipe_colors),
                lw=1.8,
                alpha=0.85,
                label=_pipe_label(ptag, pipe_labels),
            )
        ax.axvline(0, color=COLORS["gray"], ls="--", alpha=0.5)
        ax.axhline(0, color=COLORS["gray"], alpha=0.3)
        _add_time_windows(
            ax,
            DEFAULT_ERP_WINDOWS if erp_windows is None else erp_windows,
            colors=_ERP_WIN_COLORS,
        )
        ax.set_xlabel("Time (ms)", fontsize=FONTS["label"])
        ax.set_ylabel("Amplitude (uV)", fontsize=FONTS["label"])
        ax.set_title(f"Evoked Overlay at {ch_name} (test set)", fontsize=FONTS["title"])
        themed_legend(ax)
        style_axes(ax)

    for col, ch_name in enumerate(channels):
        ax = axes[2, col]
        if dev_mask is None or std_mask is None:
            ax.text(
                0.5,
                0.5,
                "No condition masks provided",
                transform=ax.transAxes,
                ha="center",
                fontsize=FONTS["label"],
            )
            style_axes(ax)
            continue

        for ptag in pipe_order:
            ep = pipe_epochs[ptag]
            data = ep.get_data() if hasattr(ep, "get_data") else ep
            ci = _ch_index(ep, ch_name)

            if dev_mask.sum() < 3 or std_mask.sum() < 3:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient trials",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=FONTS["label"],
                )
                break

            dev_data = data[dev_mask, ci, :] * 1e6
            std_data = data[std_mask, ci, :] * 1e6
            diff = dev_data.mean(axis=0) - std_data.mean(axis=0)
            diff_se = np.sqrt(
                dev_data.var(axis=0) / dev_data.shape[0]
                + std_data.var(axis=0) / std_data.shape[0]
            )

            times_ms = _get_times_ms(ep)
            c = _pipe_color(ptag, pipe_colors)
            ax.plot(
                times_ms,
                diff,
                color=c,
                lw=1.8,
                alpha=0.85,
                label=_pipe_label(ptag, pipe_labels),
            )
            ax.fill_between(
                times_ms,
                diff - diff_se,
                diff + diff_se,
                color=c,
                alpha=0.15,
                lw=0,
            )

        ax.axvline(0, color=COLORS["gray"], ls="--", alpha=0.5)
        ax.axhline(0, color=COLORS["gray"], alpha=0.3)
        _add_time_windows(
            ax,
            DEFAULT_ERP_WINDOWS if erp_windows is None else erp_windows,
            colors=_ERP_WIN_COLORS,
        )
        ax.set_xlabel("Time (ms)", fontsize=FONTS["label"])
        ax.set_ylabel("Amplitude (uV)", fontsize=FONTS["label"])
        ax.set_title(
            f"Difference Wave at {ch_name} (test set)", fontsize=FONTS["title"]
        )
        themed_legend(ax)
        style_axes(ax)

    fig.suptitle(
        f"QA Level B - Signal Diagnostics ({subject}, test set only)"
        if subject
        else "QA Level B - Signal Diagnostics (test set only)",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )
    return _finalize_fig(fig, show=show, fname=fname)


def plot_erp_condition_interaction(
    diff_waves,
    diff_se,
    effect_sizes,
    times_ms,
    *,
    conditions=None,
    condition_labels=None,
    erp_windows=None,
    subject="",
    pipe_order=None,
    pipe_colors=None,
    pipe_labels=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Plot ERP condition-by-pipeline difference waves and effect sizes."""
    import pandas as pd

    if pipe_order is None:
        pipe_order = DEFAULT_PIPE_ORDER
    if conditions is None:
        conditions = sorted({k[0] for k in diff_waves})
    if condition_labels is None:
        condition_labels = {c: c.title() for c in conditions}

    n_conds = len(conditions)

    fig1, axes1 = themed_figure(1, n_conds, figsize=figsize or (6 * n_conds, 5))
    if n_conds == 1:
        axes1 = np.array([axes1])

    for i, cond in enumerate(conditions):
        ax = axes1.flat[i]
        for ptag in pipe_order:
            key = (cond, ptag)
            dw = diff_waves.get(key)
            se = diff_se.get(key)
            if dw is None:
                continue
            c = _pipe_color(ptag, pipe_colors)
            ax.plot(
                times_ms,
                dw,
                color=c,
                lw=1.8,
                alpha=0.85,
                label=_pipe_label(ptag, pipe_labels),
            )
            if se is not None:
                ax.fill_between(times_ms, dw - se, dw + se, color=c, alpha=0.15, lw=0)
        ax.axvline(0, color=COLORS["gray"], ls="--", alpha=0.5)
        ax.axhline(0, color=COLORS["gray"], alpha=0.3)
        _add_time_windows(
            ax,
            DEFAULT_ERP_WINDOWS if erp_windows is None else erp_windows,
            colors=_ERP_WIN_COLORS,
        )
        ax.set_xlabel("Time (ms)", fontsize=FONTS["label"])
        if i == 0:
            ax.set_ylabel("Amplitude (uV)", fontsize=FONTS["label"])
        ax.set_title(condition_labels.get(cond, cond), fontsize=FONTS["title"])
        themed_legend(ax)
        style_axes(ax)

    fig1.suptitle(
        f"Condition x Pipeline: Difference Waves ({subject})"
        if subject
        else "Condition x Pipeline: Difference Waves",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )

    fig2, (ax_strip, ax_line) = themed_figure(1, 2, figsize=(14, 5))

    try:
        sns = _try_import_seaborn()
        rows = []
        for cond in conditions:
            for ptag in pipe_order:
                g_val = effect_sizes.get((cond, ptag), np.nan)
                if not np.isnan(g_val):
                    rows.append(
                        {
                            "Condition": condition_labels.get(cond, cond),
                            "Pipeline": _pipe_label(ptag, pipe_labels),
                            "Hedges_g": g_val,
                        }
                    )
        df_g = pd.DataFrame(rows)
        if len(df_g):
            cond_order = [condition_labels.get(c, c) for c in conditions]
            palette = {
                _pipe_label(p, pipe_labels): _pipe_color(p, pipe_colors)
                for p in pipe_order
            }
            with _suppress_seaborn_plot_warnings():
                sns.stripplot(
                    data=df_g,
                    x="Condition",
                    y="Hedges_g",
                    hue="Pipeline",
                    order=cond_order,
                    hue_order=[_pipe_label(p, pipe_labels) for p in pipe_order],
                    palette=palette,
                    dodge=True,
                    size=9,
                    alpha=0.85,
                    ax=ax_strip,
                    zorder=5,
                )
    except ImportError:
        for i_c, cond in enumerate(conditions):
            for j_p, ptag in enumerate(pipe_order):
                g_val = effect_sizes.get((cond, ptag), np.nan)
                if not np.isnan(g_val):
                    offset = (j_p - len(pipe_order) / 2) * 0.15
                    ax_strip.scatter(
                        i_c + offset,
                        g_val,
                        color=_pipe_color(ptag, pipe_colors),
                        s=80,
                        zorder=5,
                        label=_pipe_label(ptag, pipe_labels) if i_c == 0 else "",
                    )
        ax_strip.set_xticks(range(len(conditions)))
        ax_strip.set_xticklabels([condition_labels.get(c, c) for c in conditions])

    ax_strip.axhline(0, color=COLORS["gray"], alpha=0.3)
    ax_strip.set_ylabel("Hedges' g", fontsize=FONTS["label"])
    ax_strip.set_title("Effect Size by Condition x Pipeline", fontsize=FONTS["title"])
    themed_legend(ax_strip)
    style_axes(ax_strip)

    for ptag in pipe_order:
        vals = [effect_sizes.get((c, ptag), 0.0) for c in conditions]
        ax_line.plot(
            range(n_conds),
            vals,
            "o-",
            color=_pipe_color(ptag, pipe_colors),
            lw=2,
            markersize=8,
            label=_pipe_label(ptag, pipe_labels),
        )
    ax_line.set_xticks(range(n_conds))
    ax_line.set_xticklabels(
        [condition_labels.get(c, c) for c in conditions],
        fontsize=FONTS["tick"],
    )
    ax_line.set_ylabel("Hedges' g", fontsize=FONTS["label"])
    ax_line.set_title("Condition x Pipeline Interaction", fontsize=FONTS["title"])
    ax_line.axhline(0, color=COLORS["gray"], alpha=0.3)
    themed_legend(ax_line)
    style_axes(ax_line)

    fig2.suptitle(
        f"Condition x Pipeline Effect-Size Interaction ({subject})"
        if subject
        else "Condition x Pipeline Effect-Size Interaction",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )

    if fname is not None:
        p = Path(fname)
        stem, sfx = p.stem, p.suffix
        _finalize_fig(fig1, show=False, fname=p.with_name(f"{stem}_diffwaves{sfx}"))
        _finalize_fig(fig2, show=False, fname=p.with_name(f"{stem}_interaction{sfx}"))
        if show:
            plt.show()
    else:
        _finalize_fig(fig1, show=show, fname=None)
        _finalize_fig(fig2, show=show, fname=None)

    return fig1, fig2


def plot_erp_endpoint_summary(
    df,
    metric_cols,
    metric_labels=None,
    *,
    null_distributions=None,
    null_metric="hedges_g",
    null_pipe="C2",
    slope_metric="hedges_g",
    slope_from="C0",
    slope_to="C2",
    group_col="pipeline",
    subject_col="subject",
    group_order=None,
    group_colors=None,
    group_labels=None,
    suptitle=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Plot the ERP endpoint summary storyboard."""
    import pandas as pd

    sns = _try_import_seaborn()

    if metric_labels is None:
        metric_labels = list(metric_cols)
    if group_order is None:
        group_order = sorted(df[group_col].unique())
    if group_labels is None:
        group_labels = {}

    n_met = len(metric_cols)
    n_sub = df[subject_col].nunique()
    if figsize is None:
        figsize = (n_met * 4 + 5, 5.5)

    fig = _new_summary_figure(figsize=figsize, dpi=200, constrained_layout=True)
    gs = fig.add_gridspec(1, n_met + 1, width_ratios=[1] * n_met + [1.2])

    pretty_order = [group_labels.get(g, g) for g in group_order]
    palette = {
        group_labels.get(g, g): _pipe_color(g, group_colors) for g in group_order
    }

    for i, (mk, ml) in enumerate(zip(metric_cols, metric_labels)):
        ax = fig.add_subplot(gs[0, i])
        style_axes(ax)

        rows = []
        for g in group_order:
            for val in df.loc[df[group_col] == g, mk].dropna():
                rows.append({"Group": group_labels.get(g, g), "value": val})
        if not rows:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue
        df_v = pd.DataFrame(rows)

        with _suppress_seaborn_plot_warnings():
            sns.violinplot(
                data=df_v,
                x="Group",
                y="value",
                hue="Group",
                order=pretty_order,
                hue_order=pretty_order,
                palette=palette,
                inner=None,
                linewidth=0.8,
                alpha=0.3,
                ax=ax,
                cut=0,
                density_norm="width",
                legend=False,
            )
            sns.stripplot(
                data=df_v,
                x="Group",
                y="value",
                hue="Group",
                order=pretty_order,
                hue_order=pretty_order,
                palette=palette,
                size=3,
                alpha=0.7,
                jitter=0.12,
                ax=ax,
                zorder=5,
                legend=False,
            )

        if (
            null_distributions is not None
            and mk == null_metric
            and null_pipe in null_distributions
        ):
            null_vals = null_distributions[null_pipe].get("g")
            if null_vals is not None and len(null_vals):
                try:
                    pipe_x = group_order.index(null_pipe)
                except ValueError:
                    pipe_x = None
                if pipe_x is not None:
                    q025, q975 = np.percentile(null_vals, [2.5, 97.5])
                    ax.fill_between(
                        [pipe_x - 0.35, pipe_x + 0.35],
                        q025,
                        q975,
                        color=COLORS["gray"],
                        alpha=0.25,
                        zorder=1,
                        label=f"Null 95% CI\n[{q025:+.2f}, {q975:+.2f}]",
                    )
                    themed_legend(ax, fontsize=6, loc="lower right")

        if mk == "auc":
            ax.axhline(0.5, color="k", ls=":", lw=0.7, alpha=0.5)

        ax.set_xlabel("")
        ax.set_ylabel(ml, fontsize=FONTS["label"])
        ax.tick_params(axis="x", labelsize=6, rotation=30)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        if i == 0:
            ax.set_title(
                f"A  Violin + Dots (N={n_sub})",
                fontweight="bold",
                fontsize=FONTS["tick"],
            )

    ax_slope = fig.add_subplot(gs[0, n_met])
    style_axes(ax_slope)

    if n_sub > 1:
        for sub in df[subject_col].unique():
            v_from = df.loc[
                (df[subject_col] == sub) & (df[group_col] == slope_from), slope_metric
            ]
            v_to = df.loc[
                (df[subject_col] == sub) & (df[group_col] == slope_to), slope_metric
            ]
            if len(v_from) and len(v_to):
                ax_slope.plot(
                    [0, 1],
                    [v_from.iloc[0], v_to.iloc[0]],
                    "o-",
                    color=_pipe_color(slope_to, group_colors),
                    alpha=0.25,
                    markersize=3,
                    lw=0.7,
                )

        mean_from = df.loc[df[group_col] == slope_from, slope_metric].mean()
        mean_to = df.loc[df[group_col] == slope_to, slope_metric].mean()
        lbl_from = group_labels.get(slope_from, slope_from)
        lbl_to = group_labels.get(slope_to, slope_to)
        ax_slope.plot(
            [0, 1],
            [mean_from, mean_to],
            "s-",
            color=_pipe_color(slope_to, group_colors),
            markersize=10,
            lw=3,
            zorder=10,
            label=f"Mean: {mean_from:.2f} -> {mean_to:.2f}",
        )

        if null_distributions is not None and null_pipe in null_distributions:
            null_g = null_distributions[null_pipe].get("g")
            if null_g is not None and len(null_g):
                q025, q975 = np.percentile(null_g, [2.5, 97.5])
                ax_slope.axhspan(
                    q025,
                    q975,
                    xmin=0.6,
                    xmax=1.0,
                    color=COLORS["gray"],
                    alpha=0.15,
                    label="Null 95% CI",
                )

        ax_slope.axhline(0, color=COLORS["gray"], ls="--", alpha=0.3)
        ax_slope.set_xticks([0, 1])
        ax_slope.set_xticklabels([lbl_from, lbl_to], fontsize=FONTS["tick"])
        ax_slope.set_ylabel(
            metric_labels[metric_cols.index(slope_metric)]
            if slope_metric in metric_cols
            else slope_metric,
            fontsize=FONTS["label"],
        )
        ax_slope.set_title(
            f"B  Paired Slopes\n({lbl_from} -> {lbl_to})",
            fontweight="bold",
            fontsize=FONTS["tick"],
        )
        themed_legend(ax_slope, fontsize=7, loc="upper left")
        ax_slope.grid(axis="y", alpha=0.3)
    else:
        ax_slope.text(
            0.5,
            0.5,
            ">= 2 subjects needed",
            transform=ax_slope.transAxes,
            ha="center",
            fontsize=FONTS["label"],
        )

    title = suptitle or "ERP Benchmark: Endpoint Summary with Null Control"
    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


def plot_erp_grand_condition_interaction(
    all_diff_waves,
    all_effect_sizes,
    times_ms,
    *,
    conditions=None,
    condition_labels=None,
    erp_windows=None,
    suptitle=None,
    pipe_order=None,
    pipe_colors=None,
    pipe_labels=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Plot group-level condition-by-pipeline interactions."""
    if pipe_order is None:
        pipe_order = DEFAULT_PIPE_ORDER
    if conditions is None:
        conditions = sorted({k[0] for k in all_diff_waves})
    if condition_labels is None:
        condition_labels = {c: c.title() for c in conditions}

    n_conds = len(conditions)

    fig1, axes1 = themed_figure(1, n_conds, figsize=figsize or (6 * n_conds, 5))
    if n_conds == 1:
        axes1 = np.array([axes1])

    for i, cond in enumerate(conditions):
        ax = axes1.flat[i]
        for ptag in pipe_order:
            key = (cond, ptag)
            dw_stack = all_diff_waves.get(key)
            if dw_stack is None:
                continue
            dw_stack = np.asarray(dw_stack)
            n_sub = dw_stack.shape[0]
            grand_mean = dw_stack.mean(axis=0)
            grand_sem = (
                dw_stack.std(axis=0, ddof=1) / np.sqrt(n_sub)
                if n_sub > 1
                else np.zeros_like(grand_mean)
            )
            c = _pipe_color(ptag, pipe_colors)
            ax.plot(
                times_ms,
                grand_mean,
                color=c,
                lw=1.8,
                alpha=0.85,
                label=_pipe_label(ptag, pipe_labels),
            )
            ax.fill_between(
                times_ms,
                grand_mean - grand_sem,
                grand_mean + grand_sem,
                color=c,
                alpha=0.15,
                lw=0,
            )
        ax.axvline(0, color=COLORS["gray"], ls="--", alpha=0.5)
        ax.axhline(0, color=COLORS["gray"], alpha=0.3)
        _add_time_windows(
            ax,
            DEFAULT_ERP_WINDOWS if erp_windows is None else erp_windows,
            colors=_ERP_WIN_COLORS,
        )
        ax.set_xlabel("Time (ms)", fontsize=FONTS["label"])
        if i == 0:
            ax.set_ylabel("Amplitude (uV)", fontsize=FONTS["label"])
        ax.set_title(condition_labels.get(cond, cond), fontsize=FONTS["title"])
        themed_legend(ax)
        style_axes(ax)

    fig1.suptitle(
        suptitle or "Grand-Average Difference Waves +/- Between-Subject SEM",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )

    fig2, (ax_bar, ax_line) = themed_figure(1, 2, figsize=(14, 5))

    x = np.arange(n_conds)
    bar_w = 0.8 / len(pipe_order)

    for j, ptag in enumerate(pipe_order):
        means, sems = [], []
        for cond in conditions:
            g_arr = np.asarray(all_effect_sizes.get((cond, ptag), []))
            means.append(g_arr.mean() if len(g_arr) else 0)
            sems.append(
                g_arr.std(ddof=1) / np.sqrt(len(g_arr)) if len(g_arr) > 1 else 0
            )
        c = _pipe_color(ptag, pipe_colors)
        ax_bar.bar(
            x + j * bar_w,
            means,
            bar_w,
            yerr=sems,
            color=c,
            alpha=0.8,
            edgecolor="white",
            lw=0.5,
            capsize=3,
            label=_pipe_label(ptag, pipe_labels),
        )

    ax_bar.set_xticks(x + bar_w * (len(pipe_order) - 1) / 2)
    ax_bar.set_xticklabels(
        [condition_labels.get(c, c) for c in conditions],
        fontsize=FONTS["tick"],
    )
    ax_bar.set_ylabel("Hedges' g (mean +/- SEM)", fontsize=FONTS["label"])
    ax_bar.set_title("Effect Size by Condition x Pipeline", fontsize=FONTS["title"])
    ax_bar.axhline(0, color=COLORS["gray"], alpha=0.3)
    themed_legend(ax_bar)
    style_axes(ax_bar)

    for ptag in pipe_order:
        means = []
        sems = []
        for cond in conditions:
            g_arr = np.asarray(all_effect_sizes.get((cond, ptag), []))
            means.append(g_arr.mean() if len(g_arr) else 0)
            sems.append(
                g_arr.std(ddof=1) / np.sqrt(len(g_arr)) if len(g_arr) > 1 else 0
            )
        means = np.array(means)
        sems = np.array(sems)
        c = _pipe_color(ptag, pipe_colors)
        ax_line.plot(
            range(n_conds),
            means,
            "o-",
            color=c,
            lw=2,
            markersize=8,
            label=_pipe_label(ptag, pipe_labels),
        )
        ax_line.fill_between(
            range(n_conds),
            means - sems,
            means + sems,
            color=c,
            alpha=0.15,
            lw=0,
        )
    ax_line.set_xticks(range(n_conds))
    ax_line.set_xticklabels(
        [condition_labels.get(c, c) for c in conditions],
        fontsize=FONTS["tick"],
    )
    ax_line.set_ylabel("Hedges' g", fontsize=FONTS["label"])
    ax_line.set_title("Condition x Pipeline Interaction", fontsize=FONTS["title"])
    ax_line.axhline(0, color=COLORS["gray"], alpha=0.3)
    themed_legend(ax_line)
    style_axes(ax_line)

    fig2.suptitle(
        "Group-Level Condition x Pipeline Effect-Size Interaction",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )

    if fname is not None:
        p = Path(fname)
        stem, sfx = p.stem, p.suffix
        _finalize_fig(fig1, show=False, fname=p.with_name(f"{stem}_diffwaves{sfx}"))
        _finalize_fig(fig2, show=False, fname=p.with_name(f"{stem}_interaction{sfx}"))
        if show:
            plt.show()
    else:
        _finalize_fig(fig1, show=show, fname=None)
        _finalize_fig(fig2, show=show, fname=None)

    return fig1, fig2
