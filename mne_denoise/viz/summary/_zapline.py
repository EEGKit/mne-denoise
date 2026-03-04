"""ZapLine summary dashboards.

Public functions
----------------
- :func:`plot_zapline_summary`
- :func:`plot_zapline_adaptive_summary`
- :func:`plot_zapline_cleaning_summary`
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from .._utils import (
    _fallback_chunk_info,
    _get_chunk_info,
    _get_cleaned,
    _get_zapline_n_removed,
    _get_zapline_removed,
    _is_adaptive,
)
from ..components import plot_component_patterns, plot_component_score_curve
from ..spectra import plot_psd_comparison
from ..theme import (
    _finalize_fig,
    style_axes,
    themed_figure,
    themed_legend,
)
from ._panels import (
    COLORS,
    FONTS,
    _draw_summary_table,
    _new_summary_grid,
    _plot_before_after_psd_panel,
    _plot_patterns_panel,
    _plot_removed_power_panel,
    _plot_selection_eigenvalues_panel,
    _plot_source_trace_panel,
)


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
