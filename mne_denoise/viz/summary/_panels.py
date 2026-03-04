"""Shared panel-painting helpers used by summary submodules.

These are internal building blocks; end-users should never import them
directly. The public API lives in :mod:`mne_denoise.viz.summary`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..theme import (
    COLORS,
    DEFAULT_DPI,
    DEFAULT_FIGSIZE,
    DEFAULT_PIPE_COLORS,
    FONTS,
    style_axes,
    themed_legend,
    use_theme,
)

# ── default look-up tables ──────────────────────────────────────────────────

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


# ── tiny look-up helpers ────────────────────────────────────────────────────


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


# ── figure / grid factories ─────────────────────────────────────────────────


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


# ── reusable panel painters ─────────────────────────────────────────────────


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
