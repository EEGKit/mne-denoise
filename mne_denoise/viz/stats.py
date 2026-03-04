"""Visualization helpers for grouped metrics and summary statistics.

This module contains reusable plots for method or pipeline comparisons based
on scalar metrics, paired subject-level trajectories, and grouped summaries.

This module contains:
1. Grouped metric bar charts.
2. Trade-off scatter plots for paired metrics.
3. Single-metric comparison plots across groups.
4. Subject-level metric slope plots.
5. Distributional metric violin plots.
6. Null-distribution histograms for observed statistics.
7. Per-subject forest plots with confidence intervals.
8. Per-harmonic attenuation grouped bars.

Authors: Sina Esmaeili <sina.esmaeili@umontreal.ca>
         Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>
"""

from __future__ import annotations

import numpy as np

from ..qa import peak_attenuation_db
from ._seaborn import _suppress_seaborn_plot_warnings, _try_import_seaborn
from .theme import (
    _STATS_STYLE,
    COLORS,
    FONTS,
    _finalize_fig,
    get_series_color,
    style_axes,
    themed_figure,
    themed_legend,
    use_theme,
)


def _group_color(group, index, group_colors=None):
    """Resolve a color for a named comparison group."""
    if group_colors and group in group_colors:
        return group_colors[group]
    return get_series_color(index)


def _group_label(group, group_labels=None):
    """Resolve a display label for a named comparison group."""
    if group_labels and group in group_labels:
        return group_labels[group]
    return group


def _resolve_group_order(df, group_col, group_order):
    """Resolve plotting order for grouped metric plots."""
    if group_order is not None:
        return list(group_order)
    return sorted(df[group_col].dropna().unique())


def _default_metric_label(metric_name):
    """Format a metric column name for display."""
    return metric_name.replace("_", " ").strip()


def _infer_numeric_metrics(df, exclude_cols):
    """Infer numeric metric columns while skipping grouping identifiers."""
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    metric_cols = [col for col in numeric_cols if col not in set(exclude_cols)]
    if not metric_cols:
        raise ValueError("Could not infer metric columns from the provided DataFrame.")
    return metric_cols


def _resolve_metric_cols(df, metric_cols, exclude_cols):
    """Resolve metric columns explicitly or by numeric-column inference."""
    if metric_cols is not None:
        return list(metric_cols)
    return _infer_numeric_metrics(df, exclude_cols=exclude_cols)


def _resolve_metric_labels(metric_cols, metric_labels):
    """Resolve display labels for a metric-column list."""
    if metric_labels is None:
        return [_default_metric_label(col) for col in metric_cols]
    metric_labels = list(metric_labels)
    if len(metric_labels) != len(metric_cols):
        raise ValueError("metric_labels must match metric_cols in length.")
    return metric_labels


def _resolve_metric_directions(metric_cols, lower_better):
    """Resolve optional optimization direction for each metric."""
    if lower_better is None:
        return [None] * len(metric_cols)
    lower_better = list(lower_better)
    if len(lower_better) != len(metric_cols):
        raise ValueError("lower_better must match metric_cols in length.")
    return lower_better


def _resolve_xy_columns(df, group_col, x_col, y_col):
    """Resolve x/y metric columns for scatter plots."""
    metric_cols = _infer_numeric_metrics(df, exclude_cols=[group_col])
    if x_col is None:
        x_col = metric_cols[0]
    if y_col is None:
        if len(metric_cols) < 2:
            raise ValueError(
                "Could not infer both x_col and y_col from the provided DataFrame."
            )
        y_col = metric_cols[1] if metric_cols[0] == x_col else metric_cols[0]
    return x_col, y_col


def _resolve_metric_specs(df, metric_specs, group_col, subject_col):
    """Resolve slope-plot metric specs explicitly or by numeric-column inference."""
    if metric_specs is not None:
        return list(metric_specs)

    metric_cols = _infer_numeric_metrics(df, exclude_cols=[group_col, subject_col])[:3]
    return [(col, _default_metric_label(col)) for col in metric_cols]


def plot_metric_bars(
    df,
    *,
    metric_cols=None,
    metric_labels=None,
    lower_better=None,
    group_col="group",
    group_order=None,
    group_colors=None,
    group_labels=None,
    title="Metric Comparison (group mean ± SEM)",
    fname=None,
    show=True,
):
    """Plot grouped bar charts for one or more scalar metrics.

    Parameters
    ----------
    df : DataFrame
        Table containing a grouping column and one or more metric columns.
    metric_cols : list of str | None
        Metric columns to visualize. If None, all numeric columns except the
        grouping column are used.
    metric_labels : list of str | None
        Axis labels corresponding to ``metric_cols``. If None, labels are
        derived from the column names.
    lower_better : list of bool | None
        Whether smaller values indicate better performance for each metric.
        If None, no "best" marker is added.
    group_col : str
        Column identifying the comparison group.
    group_order : list of str | None
        Order of groups along the x-axis.
    group_colors, group_labels : dict | None
        Optional color and label overrides keyed by group name.
    title : str
        Figure-level title.
    fname : path-like | None
        Optional output path.
    show : bool
        Whether to display the figure.
    """
    metric_cols = _resolve_metric_cols(df, metric_cols, exclude_cols=[group_col])
    metric_labels = _resolve_metric_labels(metric_cols, metric_labels)
    lower_better = _resolve_metric_directions(metric_cols, lower_better)
    group_order = _resolve_group_order(df, group_col, group_order)

    with use_theme():
        n_metrics = len(metric_cols)
        fig, axes = themed_figure(1, n_metrics, figsize=(4 * n_metrics, 5))
        if n_metrics == 1:
            axes = np.array([axes])

        for axis_index, (col, label, is_lower_better) in enumerate(
            zip(metric_cols, metric_labels, lower_better)
        ):
            ax = axes[axis_index]
            means, sems = [], []
            for group in group_order:
                vals = df.loc[df[group_col] == group, col].dropna()
                means.append(vals.mean() if len(vals) else np.nan)
                sems.append(vals.sem() if len(vals) > 1 else 0.0)

            x = np.arange(len(group_order))
            colors = [
                _group_color(group, idx, group_colors)
                for idx, group in enumerate(group_order)
            ]
            bars = ax.bar(
                x,
                means,
                yerr=sems,
                color=colors,
                edgecolor=COLORS["edge"],
                linewidth=_STATS_STYLE["bar_linewidth"],
                capsize=_STATS_STYLE["bar_capsize"],
                alpha=_STATS_STYLE["bar_alpha"],
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [_group_label(group, group_labels) for group in group_order],
                fontsize=FONTS["tick"],
            )
            ax.set_ylabel(label, fontsize=FONTS["label"])
            style_axes(ax, grid=True)

            for bar, mean_value in zip(bars, means):
                if np.isnan(mean_value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean_value,
                    f"{mean_value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=FONTS["annotation"],
                )

            finite_means = np.asarray(means, dtype=float)
            if is_lower_better is not None and np.isfinite(finite_means).any():
                best_index = (
                    int(np.nanargmin(finite_means))
                    if is_lower_better
                    else int(np.nanargmax(finite_means))
                )
                ax.annotate(
                    "★",
                    xy=(best_index, finite_means[best_index]),
                    fontsize=_STATS_STYLE["annotation_star_size"],
                    ha="center",
                    va="bottom",
                    color=COLORS["highlight"],
                )

        fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
        return _finalize_fig(fig, show=show, fname=fname)


def plot_tradeoff_scatter(
    df,
    *,
    x_col=None,
    y_col=None,
    group_col="group",
    group_order=None,
    group_colors=None,
    group_labels=None,
    x_label=None,
    y_label=None,
    title="Metric Trade-off",
    reference_x=None,
    reference_y=None,
    ax=None,
    fname=None,
    show=True,
):
    """Plot a grouped x/y trade-off scatter with optional group means."""
    x_col, y_col = _resolve_xy_columns(df, group_col, x_col, y_col)
    if x_label is None:
        x_label = _default_metric_label(x_col)
    if y_label is None:
        y_label = _default_metric_label(y_col)

    group_order = _resolve_group_order(df, group_col, group_order)

    with use_theme():
        finalize = ax is None
        if ax is None:
            fig, ax = themed_figure(1, 1, figsize=(8, 6))
            if isinstance(ax, np.ndarray):
                ax = ax.flat[0]
        else:
            fig = ax.figure

        for idx, group in enumerate(group_order):
            sub_df = df[df[group_col] == group]
            color = _group_color(group, idx, group_colors)
            label = _group_label(group, group_labels)
            ax.scatter(
                sub_df[x_col],
                sub_df[y_col],
                color=color,
                s=_STATS_STYLE["scatter_size"],
                alpha=_STATS_STYLE["scatter_alpha"],
                edgecolors=COLORS["edge"],
                linewidth=_STATS_STYLE["scatter_edge_linewidth"],
                label=label,
                zorder=3,
            )
            if len(sub_df) > 1:
                ax.scatter(
                    sub_df[x_col].mean(),
                    sub_df[y_col].mean(),
                    color=color,
                    s=_STATS_STYLE["mean_scatter_size"],
                    marker="*",
                    edgecolors=COLORS["edge"],
                    linewidth=_STATS_STYLE["mean_linewidth"] / 2,
                    zorder=4,
                )

        ax.set_xlabel(x_label, fontsize=FONTS["label"])
        ax.set_ylabel(y_label, fontsize=FONTS["label"])
        ax.set_title(title, fontsize=FONTS["title"], fontweight="bold")
        if reference_y is not None:
            ax.axhline(
                reference_y,
                color=COLORS["success"],
                ls=":",
                lw=_STATS_STYLE["reference_linewidth"],
                alpha=_STATS_STYLE["reference_alpha"],
            )
        if reference_x is not None:
            ax.axvline(
                reference_x,
                color=COLORS["accent"],
                ls=":",
                lw=_STATS_STYLE["reference_linewidth"],
                alpha=_STATS_STYLE["reference_alpha"],
            )
        themed_legend(ax)
        style_axes(ax, grid=True)

        if finalize:
            return _finalize_fig(fig, show=show, fname=fname)
        return fig


def plot_single_metric_comparison(
    df,
    *,
    metric_col=None,
    metric_label=None,
    group_col="group",
    subject_col="subject",
    group_order=None,
    group_colors=None,
    group_labels=None,
    title="Single-Metric Comparison",
    reference_value=None,
    reference_label="Reference",
    ax=None,
    fname=None,
    show=True,
):
    """Plot a single metric as bars or paired subject-level dots."""
    if metric_col is None:
        metric_col = _infer_numeric_metrics(df, exclude_cols=[group_col, subject_col])[
            0
        ]
    if metric_label is None:
        metric_label = _default_metric_label(metric_col)

    group_order = _resolve_group_order(df, group_col, group_order)

    with use_theme():
        finalize = ax is None
        if ax is None:
            fig, ax = themed_figure(1, 1, figsize=(8, 6))
            if isinstance(ax, np.ndarray):
                ax = ax.flat[0]
        else:
            fig = ax.figure

        multi_subject = df[subject_col].nunique() > 1

        if multi_subject:
            for subject in df[subject_col].unique():
                subject_vals = []
                for group in group_order:
                    values = df.loc[
                        (df[subject_col] == subject) & (df[group_col] == group),
                        metric_col,
                    ]
                    subject_vals.append(values.iloc[0] if len(values) else float("nan"))
                ax.plot(
                    range(len(group_order)),
                    subject_vals,
                    "o-",
                    color=COLORS["gray"],
                    alpha=_STATS_STYLE["subject_trace_alpha"],
                    markersize=_STATS_STYLE["subject_trace_marker_size"],
                )

            means = [
                df.loc[df[group_col] == group, metric_col].mean()
                for group in group_order
            ]
            ax.plot(
                range(len(group_order)),
                means,
                "s-",
                color=COLORS["before"],
                markersize=_STATS_STYLE["mean_marker_size"],
                lw=_STATS_STYLE["mean_linewidth"],
                label="Group mean",
                zorder=5,
            )
        else:
            metric_vals = []
            for group in group_order:
                values = df.loc[df[group_col] == group, metric_col]
                metric_vals.append(values.iloc[0] if len(values) else np.nan)
            x = np.arange(len(group_order))
            colors = [
                _group_color(group, idx, group_colors)
                for idx, group in enumerate(group_order)
            ]
            ax.bar(
                x,
                metric_vals,
                color=colors,
                edgecolor=COLORS["edge"],
                linewidth=_STATS_STYLE["bar_linewidth"],
                alpha=_STATS_STYLE["bar_alpha"],
            )
            for xi, metric_value in zip(x, metric_vals):
                if np.isnan(metric_value):
                    continue
                ax.text(
                    xi,
                    metric_value,
                    f"{metric_value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=FONTS["annotation"],
                )

        if reference_value is not None:
            ax.axhline(
                reference_value,
                color=COLORS["success"],
                ls="--",
                lw=_STATS_STYLE["reference_linewidth"],
                alpha=_STATS_STYLE["reference_alpha"],
                label=reference_label,
            )
        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(
            [_group_label(group, group_labels) for group in group_order],
            fontsize=FONTS["tick"],
        )
        ax.set_ylabel(metric_label, fontsize=FONTS["label"])
        ax.set_title(title, fontsize=FONTS["title"], fontweight="bold")
        themed_legend(ax)
        style_axes(ax, grid=True)

        if finalize:
            return _finalize_fig(fig, show=show, fname=fname)
        return fig


def plot_harmonic_attenuation(
    freqs_before,
    gm_before,
    cleaned_psds,
    harmonics_hz,
    *,
    subject="",
    series_order=None,
    series_colors=None,
    series_labels=None,
    title=None,
    fname=None,
    show=True,
):
    """Plot grouped per-harmonic attenuation bars for multiple series."""
    if series_order is None:
        series_order = sorted(cleaned_psds.keys())

    with use_theme():
        fig, ax = themed_figure(1, 1, figsize=(10, 5))
        if isinstance(ax, np.ndarray):
            ax = ax.flat[0]

        bar_width = 0.8 / max(len(series_order), 1)
        x = np.arange(len(harmonics_hz))

        for idx, series_name in enumerate(series_order):
            if series_name not in cleaned_psds:
                continue
            freqs_clean, gm_clean = cleaned_psds[series_name]
            attenuation = [
                peak_attenuation_db(freqs_before, gm_before, gm_clean, harmonic)
                for harmonic in harmonics_hz
            ]
            ax.bar(
                x + idx * bar_width,
                attenuation,
                bar_width,
                color=_group_color(series_name, idx, series_colors),
                edgecolor=COLORS["edge"],
                linewidth=_STATS_STYLE["bar_linewidth"],
                label=_group_label(series_name, series_labels),
                alpha=_STATS_STYLE["bar_alpha"],
            )

        ax.set_xticks(x + bar_width * (len(series_order) - 1) / 2)
        ax.set_xticklabels(
            [f"{harmonic:.0f} Hz" for harmonic in harmonics_hz],
            fontsize=FONTS["tick"],
        )
        ax.set_ylabel("Peak Attenuation (dB)", fontsize=FONTS["label"])
        if title is None:
            title = (
                f"Per-Harmonic Attenuation — {subject}"
                if subject
                else "Per-Harmonic Attenuation"
            )
        ax.set_title(title, fontsize=FONTS["title"], fontweight="bold")
        themed_legend(ax)
        style_axes(ax, grid=True)

        return _finalize_fig(fig, show=show, fname=fname)


def plot_metric_slopes(
    df,
    *,
    metric_cols=None,
    metric_labels=None,
    metric_specs=None,
    group_col="group",
    subject_col="subject",
    group_order=None,
    group_colors=None,
    group_labels=None,
    reference_lines=None,
    suptitle=None,
    title="Paired Subject-Level Comparison",
    fname=None,
    show=True,
):
    """Plot subject-level paired trajectories for one or more metrics."""
    if df[subject_col].nunique() < 2:
        return None

    if metric_specs is not None and (
        metric_cols is not None or metric_labels is not None
    ):
        raise ValueError(
            "Specify either metric_specs or metric_cols/metric_labels, not both."
        )

    if metric_specs is None:
        if metric_cols is not None:
            metric_cols = list(metric_cols)
            metric_labels = _resolve_metric_labels(metric_cols, metric_labels)
            metric_specs = list(zip(metric_cols, metric_labels))
        else:
            metric_specs = _resolve_metric_specs(
                df,
                metric_specs,
                group_col=group_col,
                subject_col=subject_col,
            )
    else:
        metric_specs = list(metric_specs)

    group_order = _resolve_group_order(df, group_col, group_order)

    with use_theme():
        fig, axes = themed_figure(
            1, len(metric_specs), figsize=(6 * len(metric_specs), 5)
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        tick_labels = [_group_label(group, group_labels) for group in group_order]

        for ax, (metric_col, metric_label) in zip(axes.flat, metric_specs):
            for subject in df[subject_col].unique():
                subject_vals = []
                for group in group_order:
                    values = df.loc[
                        (df[subject_col] == subject) & (df[group_col] == group),
                        metric_col,
                    ]
                    subject_vals.append(values.iloc[0] if len(values) else float("nan"))
                ax.plot(
                    range(len(group_order)),
                    subject_vals,
                    "o-",
                    color=COLORS["gray"],
                    alpha=_STATS_STYLE["subject_trace_alpha"],
                    markersize=_STATS_STYLE["subject_trace_marker_size"],
                )

            means = [
                df.loc[df[group_col] == group, metric_col].mean()
                for group in group_order
            ]
            ax.plot(
                range(len(group_order)),
                means,
                "s-",
                color=COLORS["before"],
                markersize=_STATS_STYLE["mean_marker_size"],
                lw=_STATS_STYLE["mean_linewidth"],
                label="Group mean",
                zorder=5,
            )

            ax.set_xticks(range(len(group_order)))
            ax.set_xticklabels(tick_labels, fontsize=FONTS["tick"])
            ax.set_ylabel(metric_label, fontsize=FONTS["label"])
            if reference_lines and metric_col in reference_lines:
                for y_val, style in reference_lines[metric_col]:
                    ax.axhline(y_val, **style)
            themed_legend(ax)
            style_axes(ax, grid=True)

        fig.suptitle(
            suptitle or title,
            fontsize=FONTS["suptitle"],
            fontweight="bold",
        )
        return _finalize_fig(fig, show=show, fname=fname)


def plot_metric_violins(
    df,
    metric_cols,
    metric_labels=None,
    *,
    group_col="group",
    subject_col="subject",
    group_order=None,
    group_colors=None,
    group_labels=None,
    reference_lines=None,
    show_paired=True,
    suptitle=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Plot violin + strip distributions with optional paired subject lines."""
    import pandas as pd

    sns = _try_import_seaborn()

    metric_cols = list(metric_cols)
    metric_labels = _resolve_metric_labels(metric_cols, metric_labels)
    group_order = _resolve_group_order(df, group_col, group_order)
    group_labels = {} if group_labels is None else dict(group_labels)

    n_metrics = len(metric_cols)
    n_subjects = df[subject_col].nunique()
    if figsize is None:
        figsize = (4 * n_metrics, 5.5)

    with use_theme():
        fig, axes = themed_figure(1, n_metrics, figsize=figsize)
        if n_metrics == 1:
            axes = np.array([axes])

        pretty_order = [_group_label(group, group_labels) for group in group_order]
        palette = {
            _group_label(group, group_labels): _group_color(group, idx, group_colors)
            for idx, group in enumerate(group_order)
        }

        for ax, metric_col, metric_label in zip(axes.flat, metric_cols, metric_labels):
            rows = []
            for group in group_order:
                for value in df.loc[df[group_col] == group, metric_col].dropna():
                    rows.append(
                        {
                            "Group": _group_label(group, group_labels),
                            "value": value,
                        }
                    )

            if not rows:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=FONTS["label"],
                )
                style_axes(ax)
                continue

            df_plot = pd.DataFrame(rows)

            with _suppress_seaborn_plot_warnings():
                sns.violinplot(
                    data=df_plot,
                    x="Group",
                    y="value",
                    hue="Group",
                    order=pretty_order,
                    hue_order=pretty_order,
                    palette=palette,
                    inner=None,
                    linewidth=_STATS_STYLE["reference_linewidth"],
                    alpha=_STATS_STYLE["subject_trace_alpha"],
                    ax=ax,
                    cut=0,
                    density_norm="width",
                    legend=False,
                )
                sns.stripplot(
                    data=df_plot,
                    x="Group",
                    y="value",
                    hue="Group",
                    order=pretty_order,
                    hue_order=pretty_order,
                    palette=palette,
                    size=_STATS_STYLE["strip_size"],
                    alpha=_STATS_STYLE["strip_alpha"],
                    jitter=_STATS_STYLE["strip_jitter"],
                    ax=ax,
                    zorder=5,
                    legend=False,
                )

            if show_paired and n_subjects > 1:
                for subject in df[subject_col].unique():
                    subject_values = []
                    for group in group_order:
                        value = df.loc[
                            (df[subject_col] == subject) & (df[group_col] == group),
                            metric_col,
                        ]
                        subject_values.append(
                            value.iloc[0] if len(value) else float("nan")
                        )
                    ax.plot(
                        range(len(group_order)),
                        subject_values,
                        "-",
                        color=COLORS["gray"],
                        alpha=_STATS_STYLE["paired_line_alpha"],
                        lw=_STATS_STYLE["paired_linewidth"],
                        zorder=1,
                    )

            base_values = df.loc[df[group_col] == group_order[0], metric_col].dropna()
            if len(base_values):
                ax.axhline(
                    base_values.mean(),
                    color=COLORS["gray"],
                    ls="--",
                    lw=_STATS_STYLE["reference_linewidth"],
                    alpha=_STATS_STYLE["reference_alpha"],
                )

            if reference_lines and metric_col in reference_lines:
                for y_val, style in reference_lines[metric_col]:
                    ax.axhline(y_val, **style)

            ax.set_xlabel("")
            ax.set_ylabel(metric_label, fontsize=FONTS["label"])
            ax.tick_params(axis="x", labelsize=FONTS["tick"], rotation=30)
            style_axes(ax, grid=True)
            ax.xaxis.grid(False)

        fig.suptitle(
            suptitle or f"Endpoint Metrics (N = {n_subjects})",
            fontsize=FONTS["suptitle"],
            fontweight="bold",
        )
        return _finalize_fig(fig, show=show, fname=fname)


def plot_null_distribution(
    null_values,
    observed,
    *,
    metric_label="Statistic",
    ci=95,
    n_bins=60,
    suptitle=None,
    series_color=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Plot a null-distribution histogram with observed statistic and CI."""
    null_values = np.asarray(null_values)
    if figsize is None:
        figsize = (8, 5)
    if series_color is None:
        series_color = COLORS["accent"]

    with use_theme():
        fig, ax = themed_figure(1, 1, figsize=figsize)

        ax.hist(
            null_values,
            bins=n_bins,
            color=COLORS["gray"],
            alpha=_STATS_STYLE["hist_alpha"],
            edgecolor=COLORS["separator"],
            linewidth=_STATS_STYLE["hist_linewidth"],
            density=True,
            zorder=2,
            label=f"Null (N = {len(null_values):,})",
        )

        alpha_tail = (100 - ci) / 2
        lo, hi = np.percentile(null_values, [alpha_tail, 100 - alpha_tail])
        ax.axvspan(
            lo,
            hi,
            color=COLORS["gray"],
            alpha=0.12,
            zorder=1,
            label=f"{ci}% CI [{lo:+.3f}, {hi:+.3f}]",
        )
        ax.axvline(
            lo,
            color=COLORS["gray"],
            ls=":",
            lw=_STATS_STYLE["reference_linewidth"],
            alpha=0.6,
        )
        ax.axvline(
            hi,
            color=COLORS["gray"],
            ls=":",
            lw=_STATS_STYLE["reference_linewidth"],
            alpha=0.6,
        )

        ax.axvline(
            observed,
            color=series_color,
            lw=2.5,
            ls="--",
            zorder=5,
            label=f"Observed = {observed:.3f}",
        )

        p_value = float(np.mean(np.abs(null_values) >= np.abs(observed)))

        ax.annotate(
            f"p = {p_value:.4f}",
            xy=(observed, ax.get_ylim()[1] * 0.92),
            fontsize=FONTS["annotation"],
            fontweight="bold",
            ha="left" if observed > np.median(null_values) else "right",
            va="top",
            color=series_color,
            xytext=(8, 0),
            textcoords="offset points",
        )

        ax.set_xlabel(metric_label, fontsize=FONTS["label"])
        ax.set_ylabel("Density", fontsize=FONTS["label"])
        themed_legend(ax, fontsize=_STATS_STYLE["legend_fontsize_small"])
        style_axes(ax)

        fig.suptitle(
            suptitle or f"Null Distribution - {metric_label}",
            fontsize=FONTS["suptitle"],
            fontweight="bold",
        )
        return _finalize_fig(fig, show=show, fname=fname), p_value


def plot_forest(
    df,
    metric_col,
    *,
    ci_col=None,
    se_col=None,
    group_col="group",
    subject_col="subject",
    target_group=None,
    baseline_group=None,
    group_colors=None,
    group_labels=None,
    metric_label=None,
    reference_line=0.0,
    suptitle=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Plot per-subject point estimates with confidence intervals and pooled mean."""
    groups = sorted(df[group_col].unique())
    if target_group is None:
        target_group = groups[-1]
    if metric_label is None:
        metric_label = _default_metric_label(metric_col).title()
    group_labels = {} if group_labels is None else dict(group_labels)

    df_target = df.loc[df[group_col] == target_group].copy()
    df_target = df_target.sort_values(metric_col, ascending=True).reset_index(drop=True)
    subjects = df_target[subject_col].values
    n_subjects = len(subjects)
    values = df_target[metric_col].values.astype(float)

    if ci_col is not None and ci_col in df_target.columns:
        half_width = df_target[ci_col].values.astype(float)
    elif se_col is not None and se_col in df_target.columns:
        half_width = 1.96 * df_target[se_col].values.astype(float)
    else:
        sd = values.std(ddof=1) if n_subjects > 1 else 1.0
        half_width = np.full(n_subjects, 1.96 * sd / np.sqrt(max(n_subjects, 1)))

    if figsize is None:
        figsize = (8, max(4, n_subjects * 0.35 + 2))

    with use_theme():
        fig, ax = themed_figure(1, 1, figsize=figsize)

        y_pos = np.arange(n_subjects)
        target_color = _group_color(
            target_group, groups.index(target_group), group_colors
        )
        target_label = _group_label(target_group, group_labels)

        if baseline_group is not None:
            df_base = df.loc[df[group_col] == baseline_group].copy()
            base_color = _group_color(
                baseline_group,
                groups.index(baseline_group),
                group_colors,
            )
            base_label = _group_label(baseline_group, group_labels)
            for i, subject in enumerate(subjects):
                base_value = df_base.loc[df_base[subject_col] == subject, metric_col]
                if len(base_value):
                    ax.plot(
                        base_value.iloc[0],
                        y_pos[i],
                        "o",
                        color=base_color,
                        markersize=_STATS_STYLE["forest_marker_size"],
                        alpha=0.35,
                        zorder=2,
                    )
            base_mean = df_base[metric_col].mean()
            ax.plot(
                base_mean,
                -1.2,
                "D",
                color=base_color,
                markersize=_STATS_STYLE["forest_baseline_mean_marker_size"],
                zorder=6,
                alpha=0.5,
                label=f"{base_label} mean = {base_mean:.3f}",
            )

        ax.errorbar(
            values,
            y_pos,
            xerr=half_width,
            fmt="o",
            color=target_color,
            ecolor=target_color,
            elinewidth=1.2,
            capsize=_STATS_STYLE["bar_capsize"],
            markersize=_STATS_STYLE["forest_marker_size"],
            alpha=_STATS_STYLE["bar_alpha"],
            zorder=4,
            label=target_label,
        )

        target_mean = values.mean()
        target_se = values.std(ddof=1) / np.sqrt(n_subjects) if n_subjects > 1 else 0
        ax.errorbar(
            target_mean,
            -1.2,
            xerr=1.96 * target_se,
            fmt="D",
            color=target_color,
            ecolor=target_color,
            elinewidth=_STATS_STYLE["mean_linewidth"],
            capsize=_STATS_STYLE["bar_capsize"] + 1,
            markersize=_STATS_STYLE["forest_pooled_marker_size"],
            zorder=6,
            label=f"Pooled mean = {target_mean:.3f}",
        )

        if reference_line is not None:
            ax.axvline(
                reference_line,
                color=COLORS["gray"],
                ls="--",
                lw=_STATS_STYLE["reference_linewidth"],
                alpha=_STATS_STYLE["reference_alpha"],
            )

        ax.set_yticks(list(y_pos) + [-1.2])
        ax.set_yticklabels(list(subjects) + ["Pooled"], fontsize=FONTS["tick"])
        ax.set_xlabel(metric_label, fontsize=FONTS["label"])
        ax.set_ylabel("")
        ax.invert_yaxis()
        style_axes(ax, grid=True)
        ax.yaxis.grid(False)
        themed_legend(
            ax, fontsize=_STATS_STYLE["legend_fontsize_small"], loc="lower right"
        )

        fig.suptitle(
            suptitle or f"Forest Plot - {_group_label(target_group, group_labels)}",
            fontsize=FONTS["suptitle"],
            fontweight="bold",
        )
        return _finalize_fig(fig, show=show, fname=fname)
