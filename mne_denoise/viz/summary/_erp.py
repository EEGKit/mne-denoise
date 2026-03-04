"""ERP summary dashboards.

Public functions
----------------
- :func:`plot_erp_signal_diagnostics`
- :func:`plot_erp_condition_interaction`
- :func:`plot_erp_endpoint_summary`
- :func:`plot_erp_grand_condition_interaction`
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch as _welch

from .._seaborn import _suppress_seaborn_plot_warnings, _try_import_seaborn
from ..signals import _add_time_windows, _ch_index, _get_times_ms
from ..theme import (
    _finalize_fig,
    style_axes,
    themed_figure,
    themed_legend,
)
from ._panels import (
    _ERP_WIN_COLORS,
    COLORS,
    DEFAULT_ERP_WINDOWS,
    DEFAULT_PIPE_ORDER,
    FONTS,
    _new_summary_figure,
    _pipe_color,
    _pipe_label,
)


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
