"""Benchmark summary dashboards.

Public functions
----------------
- :func:`plot_denoising_summary`
- :func:`plot_metric_tradeoff_summary`
"""

from __future__ import annotations

from ..signals import _get_gfp_for_summary, plot_power_ratio_map
from ..spectra import plot_psd_comparison
from ..stats import plot_single_metric_comparison, plot_tradeoff_scatter
from ..theme import (
    _finalize_fig,
    style_axes,
    themed_figure,
    themed_legend,
    use_theme,
)
from ._panels import (
    COLORS,
    DEFAULT_METHOD_ORDER,
    FONTS,
    _new_summary_figure,
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
