"""ERP benchmark visualizations for multi-pipeline comparisons.

Publication-quality figures for ERP denoising benchmarks (e.g., DSS vs
ICA vs baseline).  Designed to pair with the runabout ERP DSS benchmark
notebook, but general enough for any N-pipeline comparison.

Every ``plot_*`` function returns a :class:`matplotlib.figure.Figure`
and accepts an optional ``fname`` for disk persistence.  The functions
use the shared theme from :mod:`mne_denoise.viz._theme`.

Functions
---------
plot_erp_signal_diagnostics
    3Ã—2 QA dashboard: PSD, evoked overlay, difference waves at 2 channels.
plot_erp_condition_interaction
    Per-condition difference waves + Hedges' g effect-size interaction.
plot_erp_metric_violins
    Violin + swarm + paired lines for arbitrary metric columns.
plot_erp_endpoint_summary
    Multipanel storyboard: violins + null-CI overlay + paired slopes.
plot_erp_pipeline_slopes
    Multi-metric paired subject-level trajectories.
plot_erp_grand_average
    Group-mean evoked Â± between-subject SEM for each pipeline.
plot_erp_grand_condition_interaction
    Group-level condition Ã— pipeline interaction with between-subject error.
plot_erp_null_distribution
    Histogram of permutation null with observed statistic and CI.
plot_erp_forest
    Per-subject effect-size forest plot with confidence intervals.

Authors
-------
Sina Esmaeili â€” sina.esmaeili@umontreal.ca
Hamza Abdelhedi â€” hamza.abdelhedi@umontreal.ca
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch as _welch

from ._theme import COLORS, FONTS, _finalize_fig, pub_figure, pub_legend, style_axes

# Suppress seaborn FutureWarning about palette without hue (our calls DO pass
# hue, but seaborn â‰¤ 0.13 may still emit the warning internally).
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"Passing `palette` without assigning `hue`",
)
warnings.filterwarnings(
    "ignore",
    message=r"set_ticklabels\(\) should only be used",
)

# =====================================================================
# Default pipeline palette (colorblind-safe, Wong 2011)
# =====================================================================
DEFAULT_PIPE_COLORS: dict[str, str] = {
    "C0": COLORS["dark"],
    "C1": COLORS["green"],
    "C2": COLORS["red"],
}

DEFAULT_PIPE_LABELS: dict[str, str] = {
    "C0": "Baseline",
    "C1": "Paper (ICA)",
    "C2": "DSS (AverageBias)",
}

DEFAULT_PIPE_ORDER: list[str] = ["C0", "C1", "C2"]

# Fallback ERP windows (seconds) used for axvspan highlighting
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


# =====================================================================
# Helpers
# =====================================================================


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


def _add_erp_windows(ax, erp_windows=None, alpha=0.06):
    """Shade ERP time windows on an axes (expects x-axis in **ms**)."""
    if erp_windows is None:
        erp_windows = DEFAULT_ERP_WINDOWS
    for wname, (t0, t1) in erp_windows.items():
        color = _ERP_WIN_COLORS.get(wname, COLORS["gray"])
        ax.axvspan(t0 * 1000, t1 * 1000, alpha=alpha, color=color)


def _try_import_seaborn():
    """Import seaborn or raise a clear error."""
    try:
        import seaborn as sns

        return sns
    except ImportError as err:
        raise ImportError(
            "seaborn is required for this plotting function. "
            "Install it with:  pip install seaborn"
        ) from err


# =====================================================================
# plot_erp_signal_diagnostics â€” Cell 22 QA Level B
# =====================================================================


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
    """3Ã—2 QA dashboard: PSD, evoked overlay, difference waves at 2 channels.

    Rows: PSD | Evoked | Difference wave.
    Columns: one per channel in *channels*.

    Parameters
    ----------
    pipe_epochs : dict
        ``{pipe_tag: Epochs | ndarray}`` â€” epoched data per pipeline.
        If MNE Epochs, ``get_data()`` is called automatically.
    pipe_evokeds : dict
        ``{pipe_tag: Evoked}`` â€” grand-averaged evokeds per pipeline.
    channels : tuple of str
        Two channel names to display (default ``("Cz", "Pz")``).
    dev_mask : ndarray of bool | None
        Deviant-condition mask (over epochs in *pipe_epochs*).
        Required for difference-wave rows.
    std_mask : ndarray of bool | None
        Standard-condition mask.
    sfreq : float
        Sampling frequency (Hz).
    erp_windows : dict | None
        ``{name: (t0_s, t1_s)}`` time windows to shade.
    fmax : float
        Maximum frequency for PSD panels.
    subject : str
        Subject label for the suptitle.
    pipe_order : list of str | None
        Which pipelines to plot and in what order.
    pipe_colors, pipe_labels : dict | None
        Override palettes.
    figsize : tuple | None
    fname : str | Path | None
        Save figure to this path.
    show : bool

    Returns
    -------
    fig : Figure
    """
    if pipe_order is None:
        pipe_order = sorted(pipe_epochs.keys())
    if figsize is None:
        figsize = (16, 14)

    n_ch = len(channels)
    fig, axes = pub_figure(3, n_ch, figsize=figsize)
    if n_ch == 1:
        axes = axes[:, np.newaxis]

    # ---- Row 1: PSD ----
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
        ax.set_ylabel("PSD (VÂ²/Hz)", fontsize=FONTS["label"])
        ax.set_title(f"Epoch-Averaged PSD at {ch_name}", fontsize=FONTS["title"])
        ax.set_xlim(0, fmax)
        pub_legend(ax)
        style_axes(ax)

    # ---- Row 2: Evoked overlay ----
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
        _add_erp_windows(ax, erp_windows)
        ax.set_xlabel("Time (ms)", fontsize=FONTS["label"])
        ax.set_ylabel("Amplitude (ÂµV)", fontsize=FONTS["label"])
        ax.set_title(f"Evoked Overlay at {ch_name} (test set)", fontsize=FONTS["title"])
        pub_legend(ax)
        style_axes(ax)

    # ---- Row 3: Difference waves (deviant âˆ’ standard) ----
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
        _add_erp_windows(ax, erp_windows)
        ax.set_xlabel("Time (ms)", fontsize=FONTS["label"])
        ax.set_ylabel("Amplitude (ÂµV)", fontsize=FONTS["label"])
        ax.set_title(
            f"Difference Wave at {ch_name} (test set)", fontsize=FONTS["title"]
        )
        pub_legend(ax)
        style_axes(ax)

    fig.suptitle(
        f"QA Level B â€” Signal Diagnostics ({subject}, test set only)"
        if subject
        else "QA Level B â€” Signal Diagnostics (test set only)",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )
    return _finalize_fig(fig, show=show, fname=fname)


# =====================================================================
# plot_erp_condition_interaction â€” Cell 24
# =====================================================================


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
    """Per-condition difference waves + Hedges' g effect-size interaction.

    Produces **two** figures side-by-side:

    * **Figure 1** (1 Ã— n_conditions): difference-wave overlays per
      condition, with pipelines as traces.
    * **Figure 2** (1 Ã— 2): stripplot of *g* by condition+pipeline and
      a condition Ã— pipeline line interaction plot.

    Parameters
    ----------
    diff_waves : dict
        ``{(condition, pipe_tag): 1-D array}`` â€” difference-wave time
        series in ÂµV.
    diff_se : dict
        ``{(condition, pipe_tag): 1-D array}`` â€” standard error envelope.
    effect_sizes : dict
        ``{(condition, pipe_tag): float}`` â€” Hedges' *g* per cell.
    times_ms : ndarray
        Time vector in milliseconds.
    conditions : list of str | None
        Condition keys (order matters).
    condition_labels : dict | None
        Pretty labels ``{cond: label}``.
    erp_windows : dict | None
        Shaded ERP time windows.
    subject : str
    pipe_order, pipe_colors, pipe_labels : list/dict | None
    figsize : tuple | None
        Size for each figure. Default ``(18, 5)`` for fig1.
    fname : str | Path | None
        If given, ``"_diffwaves"`` and ``"_interaction"`` suffixes are
        appended before the extension.
    show : bool

    Returns
    -------
    fig_diff : Figure
        Difference-wave figure.
    fig_interact : Figure
        Effect-size interaction figure.
    """
    import pandas as pd

    if pipe_order is None:
        pipe_order = DEFAULT_PIPE_ORDER
    if conditions is None:
        conditions = sorted({k[0] for k in diff_waves})
    if condition_labels is None:
        condition_labels = {c: c.title() for c in conditions}

    n_conds = len(conditions)

    # ---- Figure 1: difference waves ----
    fig1, axes1 = pub_figure(1, n_conds, figsize=figsize or (6 * n_conds, 5))
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
                ax.fill_between(
                    times_ms,
                    dw - se,
                    dw + se,
                    color=c,
                    alpha=0.15,
                    lw=0,
                )
        ax.axvline(0, color=COLORS["gray"], ls="--", alpha=0.5)
        ax.axhline(0, color=COLORS["gray"], alpha=0.3)
        _add_erp_windows(ax, erp_windows)
        ax.set_xlabel("Time (ms)", fontsize=FONTS["label"])
        if i == 0:
            ax.set_ylabel("Amplitude (ÂµV)", fontsize=FONTS["label"])
        ax.set_title(condition_labels.get(cond, cond), fontsize=FONTS["title"])
        pub_legend(ax)
        style_axes(ax)

    fig1.suptitle(
        f"Condition Ã— Pipeline: Difference Waves ({subject})"
        if subject
        else "Condition Ã— Pipeline: Difference Waves",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )

    # ---- Figure 2: effect-size interaction ----
    fig2, (ax_strip, ax_line) = pub_figure(1, 2, figsize=(14, 5))

    # Left: strip plot (seaborn optional)
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
        # Fallback: simple scatter
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
    ax_strip.set_title("Effect Size by Condition Ã— Pipeline", fontsize=FONTS["title"])
    pub_legend(ax_strip)
    style_axes(ax_strip)

    # Right: interaction line plot
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
    ax_line.set_title("Condition Ã— Pipeline Interaction", fontsize=FONTS["title"])
    ax_line.axhline(0, color=COLORS["gray"], alpha=0.3)
    pub_legend(ax_line)
    style_axes(ax_line)

    fig2.suptitle(
        f"Condition Ã— Pipeline Effect-Size Interaction ({subject})"
        if subject
        else "Condition Ã— Pipeline Effect-Size Interaction",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )

    # Save with suffixes
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


# =====================================================================
# plot_erp_metric_violins â€” Cell 26  (generic for any benchmark)
# =====================================================================


def plot_erp_metric_violins(
    df,
    metric_cols,
    metric_labels=None,
    *,
    group_col="pipeline",
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
    """Violin + swarm + within-subject paired lines for arbitrary metrics.

    This is the ERP counterpart of :func:`plot_metric_bars` in
    ``benchmark.py``, but uses distributional plots rather than bars
    and is schema-agnostic (works with any grouping column and metric
    set).

    Parameters
    ----------
    df : DataFrame
        Long-form table with at least *group_col*, *subject_col*, and
        the columns listed in *metric_cols*.
    metric_cols : list of str
        Column names to plot (one panel per metric).
    metric_labels : list of str | None
        Y-axis labels.  Defaults to *metric_cols*.
    group_col : str
        Column that identifies the grouping (``"pipeline"`` or
        ``"method"``).
    subject_col : str
        Column that identifies the subject.
    group_order : list of str | None
        Order of groups on the x-axis.
    group_colors : dict | None
        ``{group_tag: color}`` overrides.
    group_labels : dict | None
        ``{group_tag: pretty_label}`` overrides.
    reference_lines : dict | None
        ``{metric_col: [(y, style_dict), ...]}``.  E.g.
        ``{"auc": [(0.5, {"color": "k", "ls": ":", "alpha": 0.5})]}``.
    show_paired : bool
        If *True*, draw faint within-subject connecting lines.
    suptitle : str | None
    figsize : tuple | None
    fname : str | Path | None
    show : bool

    Returns
    -------
    fig : Figure
    """
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
        figsize = (4 * n_met, 5.5)

    fig, axes = pub_figure(1, n_met, figsize=figsize)
    if n_met == 1:
        axes = np.array([axes])

    # Resolve pretty labels + palette
    pretty_order = [group_labels.get(g, g) for g in group_order]
    palette = {}
    for g in group_order:
        lbl = group_labels.get(g, g)
        palette[lbl] = _pipe_color(g, group_colors)

    for ax, mk, ml in zip(axes.flat, metric_cols, metric_labels):
        # Build long-form data
        rows = []
        for g in group_order:
            for val in df.loc[df[group_col] == g, mk].dropna():
                rows.append({"Group": group_labels.get(g, g), "value": val})
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
        df_v = pd.DataFrame(rows)

        # Violin
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
        # Swarm
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

        # Paired within-subject lines
        if show_paired and n_sub > 1:
            for sub in df[subject_col].unique():
                sub_vals = []
                for g in group_order:
                    v = df.loc[(df[subject_col] == sub) & (df[group_col] == g), mk]
                    sub_vals.append(v.iloc[0] if len(v) else float("nan"))
                ax.plot(
                    range(len(group_order)),
                    sub_vals,
                    "-",
                    color=COLORS["gray"],
                    alpha=0.1,
                    lw=0.4,
                    zorder=1,
                )

        # Baseline reference (group-mean of first group)
        base_vals = df.loc[df[group_col] == group_order[0], mk].dropna()
        if len(base_vals):
            ax.axhline(
                base_vals.mean(),
                color=COLORS["gray"],
                ls="--",
                lw=0.8,
                alpha=0.5,
            )

        # Custom reference lines
        if reference_lines and mk in reference_lines:
            for y_val, style in reference_lines[mk]:
                ax.axhline(y_val, **style)

        ax.set_xlabel("")
        ax.set_ylabel(ml, fontsize=FONTS["label"])
        ax.tick_params(axis="x", labelsize=FONTS["tick"], rotation=30)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        style_axes(ax)

    title = suptitle or f"Endpoint Metrics (N = {n_sub})"
    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


# =====================================================================
# plot_erp_endpoint_summary â€” Cell 30  (multipanel storyboard)
# =====================================================================


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
    """Multipanel storyboard: metric violins + null-CI overlay + paired slopes.

    Layout: ``n_metrics`` violin panels + one paired-slope panel on the
    right.  When *null_distributions* is provided, the 95 % CI from the
    permutation null is shaded on the matching metric panel.

    Parameters
    ----------
    df : DataFrame
        Long-form metrics table.
    metric_cols : list of str
    metric_labels : list of str | None
    null_distributions : dict | None
        ``{pipe_tag: {"g": ndarray, "auc": ndarray}}``.
    null_metric : str
        Which metric column receives the null overlay.
    null_pipe : str
        Which pipeline's null to display.
    slope_metric : str
        Metric shown in the paired-slope panel.
    slope_from, slope_to : str
        The two groups connected by slope lines.
    group_col, subject_col : str
    group_order, group_colors, group_labels : list/dict | None
    suptitle : str | None
    figsize : tuple | None
    fname : str | Path | None
    show : bool

    Returns
    -------
    fig : Figure
    """
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

    fig = plt.figure(figsize=figsize, dpi=200, constrained_layout=True)
    fig.set_facecolor("white")
    gs = fig.add_gridspec(1, n_met + 1, width_ratios=[1] * n_met + [1.2])

    pretty_order = [group_labels.get(g, g) for g in group_order]
    palette = {
        group_labels.get(g, g): _pipe_color(g, group_colors) for g in group_order
    }

    # ---- Panels A: Violin per metric ----
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

        # Null distribution overlay
        if (
            null_distributions is not None
            and mk == null_metric
            and null_pipe in null_distributions
        ):
            null_vals = null_distributions[null_pipe].get("g")
            if null_vals is not None and len(null_vals):
                # Find x position for null_pipe
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
                    pub_legend(ax, fontsize=6, loc="lower right")

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

    # ---- Panel B: Paired slope plot ----
    ax_slope = fig.add_subplot(gs[0, n_met])
    style_axes(ax_slope)

    if n_sub > 1:
        for sub in df[subject_col].unique():
            v_from = df.loc[
                (df[subject_col] == sub) & (df[group_col] == slope_from),
                slope_metric,
            ]
            v_to = df.loc[
                (df[subject_col] == sub) & (df[group_col] == slope_to),
                slope_metric,
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

        # Group means
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
            label=f"Mean: {mean_from:.2f} â†’ {mean_to:.2f}",
        )

        # Null CI band on destination end
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
            f"B  Paired Slopes\n({lbl_from} â†’ {lbl_to})",
            fontweight="bold",
            fontsize=FONTS["tick"],
        )
        pub_legend(ax_slope, fontsize=7, loc="upper left")
        ax_slope.grid(axis="y", alpha=0.3)
    else:
        ax_slope.text(
            0.5,
            0.5,
            "â‰¥ 2 subjects needed",
            transform=ax_slope.transAxes,
            ha="center",
            fontsize=FONTS["label"],
        )

    title = suptitle or "ERP Benchmark: Endpoint Summary with Null Control"
    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


# =====================================================================
# plot_erp_pipeline_slopes â€” Cell 32
# =====================================================================


def plot_erp_pipeline_slopes(
    df,
    metric_cols,
    metric_labels=None,
    *,
    group_col="pipeline",
    subject_col="subject",
    group_order=None,
    group_colors=None,
    group_labels=None,
    reference_lines=None,
    suptitle=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Multi-metric paired subject-level trajectory (slope) plots.

    One panel per metric.  Thin coloured lines connect each subject's
    values across pipelines; a bold black line shows the group mean.

    Parameters
    ----------
    df : DataFrame
    metric_cols : list of str
    metric_labels : list of str | None
    group_col, subject_col : str
    group_order, group_colors, group_labels : list/dict | None
    reference_lines : dict | None
        ``{metric_col: [(y, style_dict), ...]}``.
    suptitle : str | None
    figsize : tuple | None
    fname : str | Path | None
    show : bool

    Returns
    -------
    fig : Figure | None
        *None* if fewer than 2 subjects.
    """
    if metric_labels is None:
        metric_labels = list(metric_cols)
    if group_order is None:
        group_order = sorted(df[group_col].unique())
    if group_labels is None:
        group_labels = {}

    n_sub = df[subject_col].nunique()
    if n_sub < 2:
        return None

    n_met = len(metric_cols)
    if figsize is None:
        figsize = (n_met * 4, 5)

    fig, axes = pub_figure(1, n_met, figsize=figsize)
    if n_met == 1:
        axes = np.array([axes])

    x_labels = [group_labels.get(g, g) for g in group_order]

    for ax, mk, ml in zip(axes.flat, metric_cols, metric_labels):
        # Individual subjects
        for sub in df[subject_col].unique():
            sub_vals = []
            for g in group_order:
                v = df.loc[(df[subject_col] == sub) & (df[group_col] == g), mk]
                sub_vals.append(v.iloc[0] if len(v) else float("nan"))

            # Coloured segments connecting adjacent pipelines
            for j in range(len(group_order) - 1):
                ax.plot(
                    [j, j + 1],
                    [sub_vals[j], sub_vals[j + 1]],
                    color=_pipe_color(group_order[j + 1], group_colors),
                    alpha=0.15,
                    lw=0.6,
                    zorder=1,
                )
            # Dots at each pipeline
            for j, g in enumerate(group_order):
                if not np.isnan(sub_vals[j]):
                    ax.scatter(
                        j,
                        sub_vals[j],
                        color=_pipe_color(g, group_colors),
                        s=8,
                        alpha=0.4,
                        zorder=2,
                    )

        # Group mean
        means = [df.loc[df[group_col] == g, mk].mean() for g in group_order]
        ax.plot(
            range(len(group_order)),
            means,
            "s-",
            color=COLORS["dark"],
            markersize=9,
            lw=2.5,
            zorder=5,
            label="Group mean",
        )

        # Annotate mean values
        for j, m in enumerate(means):
            if not np.isnan(m):
                fmt = f"{m:.2f}" if abs(m) < 100 else f"{m:.0f}"
                ax.annotate(
                    fmt,
                    (j, m),
                    textcoords="offset points",
                    xytext=(0, 10),
                    fontsize=FONTS["annotation"],
                    ha="center",
                    fontweight="bold",
                )

        # Reference lines
        if reference_lines and mk in reference_lines:
            for y_val, style in reference_lines[mk]:
                ax.axhline(y_val, **style)

        ax.set_xticks(range(len(group_order)))
        ax.set_xticklabels(x_labels, fontsize=FONTS["tick"], rotation=30, ha="right")
        ax.set_ylabel(ml, fontsize=FONTS["label"])
        pub_legend(ax, fontsize=7)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.annotate(
            f"N = {n_sub}",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            fontsize=FONTS["annotation"],
            va="top",
        )
        style_axes(ax)

    title = suptitle or "Paired Subject-Level Slopes â€” Within-Subject Trajectories"
    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


# =====================================================================
# plot_erp_grand_average â€” group-mean evoked Â± between-subject SEM
# =====================================================================


def plot_erp_grand_average(
    all_evokeds,
    *,
    channels=("Cz", "Pz"),
    erp_windows=None,
    suptitle=None,
    pipe_order=None,
    pipe_colors=None,
    pipe_labels=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Group-mean evoked Â± between-subject SEM for each pipeline.

    Parameters
    ----------
    all_evokeds : dict
        ``{pipe_tag: list[Evoked]}``.  Each list has one Evoked per
        subject; the function computes the grand mean and between-subject
        SEM over these lists.
    channels : tuple of str
        Channel names to display (one column per channel).
    erp_windows : dict | None
        Shaded ERP time windows.  Default ``DEFAULT_ERP_WINDOWS``.
    suptitle : str | None
    pipe_order : list of str | None
    pipe_colors, pipe_labels : dict | None
    figsize : tuple | None
    fname : str | Path | None
    show : bool

    Returns
    -------
    fig : Figure
    """
    if pipe_order is None:
        pipe_order = sorted(all_evokeds.keys())
    n_ch = len(channels)
    if figsize is None:
        figsize = (8 * n_ch, 5)

    fig, axes = pub_figure(1, n_ch, figsize=figsize)
    if n_ch == 1:
        axes = np.array([axes])

    for col, ch_name in enumerate(channels):
        ax = axes.flat[col]
        for ptag in pipe_order:
            evoked_list = all_evokeds[ptag]
            n_sub = len(evoked_list)
            ci = evoked_list[0].ch_names.index(ch_name)
            times_ms = evoked_list[0].times * 1000

            # Stack subject data: (n_sub, n_times)
            stacked = np.array([ev.data[ci] * 1e6 for ev in evoked_list])
            grand_mean = stacked.mean(axis=0)
            grand_sem = (
                stacked.std(axis=0, ddof=1) / np.sqrt(n_sub)
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
            if n_sub > 1:
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
        _add_erp_windows(ax, erp_windows)
        ax.set_xlabel("Time (ms)", fontsize=FONTS["label"])
        ax.set_ylabel("Amplitude (ÂµV)", fontsize=FONTS["label"])
        ax.set_title(f"Grand Average at {ch_name}", fontsize=FONTS["title"])
        pub_legend(ax)
        style_axes(ax)

    n_total = len(next(iter(all_evokeds.values())))
    title = suptitle or f"Grand-Average Evoked Â± Between-Subject SEM (N = {n_total})"
    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


# =====================================================================
# plot_erp_grand_condition_interaction â€” group-level cond Ã— pipeline
# =====================================================================


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
    """Group-level condition Ã— pipeline interaction with between-subject error.

    Produces **two** figures:

    * **Figure 1** (1 Ã— n_conditions): Grand-average difference waves
      Â± between-subject SEM.
    * **Figure 2** (mean Â± SEM of Hedges' *g* per cell, strip + bars).

    Parameters
    ----------
    all_diff_waves : dict
        ``{(condition, pipe_tag): ndarray (n_subjects, n_times)}`` â€”
        per-subject difference-wave time series in ÂµV.
    all_effect_sizes : dict
        ``{(condition, pipe_tag): 1-D array (n_subjects,)}`` â€”
        per-subject Hedges' *g* values.
    times_ms : ndarray
        Time vector in milliseconds.
    conditions : list of str | None
    condition_labels : dict | None
    erp_windows : dict | None
    suptitle : str | None
    pipe_order, pipe_colors, pipe_labels : list/dict | None
    figsize : tuple | None
    fname : str | Path | None
    show : bool

    Returns
    -------
    fig_diff : Figure
    fig_effect : Figure
    """
    if pipe_order is None:
        pipe_order = DEFAULT_PIPE_ORDER
    if conditions is None:
        conditions = sorted({k[0] for k in all_diff_waves})
    if condition_labels is None:
        condition_labels = {c: c.title() for c in conditions}

    n_conds = len(conditions)

    # ---- Figure 1: grand-average diff waves ----
    fig1, axes1 = pub_figure(1, n_conds, figsize=figsize or (6 * n_conds, 5))
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
        _add_erp_windows(ax, erp_windows)
        ax.set_xlabel("Time (ms)", fontsize=FONTS["label"])
        if i == 0:
            ax.set_ylabel("Amplitude (ÂµV)", fontsize=FONTS["label"])
        ax.set_title(condition_labels.get(cond, cond), fontsize=FONTS["title"])
        pub_legend(ax)
        style_axes(ax)

    fig1.suptitle(
        suptitle or "Grand-Average Difference Waves Â± Between-Subject SEM",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )

    # ---- Figure 2: effect-size mean Â± SEM bars + strip ----
    fig2, (ax_bar, ax_line) = pub_figure(1, 2, figsize=(14, 5))

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
    ax_bar.set_ylabel("Hedges' g (mean Â± SEM)", fontsize=FONTS["label"])
    ax_bar.set_title("Effect Size by Condition Ã— Pipeline", fontsize=FONTS["title"])
    ax_bar.axhline(0, color=COLORS["gray"], alpha=0.3)
    pub_legend(ax_bar)
    style_axes(ax_bar)

    # Interaction line plot (means Â± SEM as error band)
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
    ax_line.set_title("Condition Ã— Pipeline Interaction", fontsize=FONTS["title"])
    ax_line.axhline(0, color=COLORS["gray"], alpha=0.3)
    pub_legend(ax_line)
    style_axes(ax_line)

    fig2.suptitle(
        "Group-Level Condition Ã— Pipeline Effect-Size Interaction",
        fontsize=FONTS["suptitle"],
        fontweight="bold",
    )

    # Dual-save with suffixes
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


# =====================================================================
# plot_erp_null_distribution â€” permutation null histogram
# =====================================================================


def plot_erp_null_distribution(
    null_values,
    observed,
    *,
    metric_label="Hedges' g",
    ci=95,
    n_bins=60,
    suptitle=None,
    pipe_color=None,
    figsize=None,
    fname=None,
    show=True,
):
    """Histogram of permutation null with observed statistic and CI.

    Parameters
    ----------
    null_values : ndarray
        1-D array of the null distribution (e.g. permuted Hedges' g).
    observed : float
        The observed (real) test statistic.
    metric_label : str
        X-axis label.
    ci : int
        Confidence interval percentage (default 95).
    n_bins : int
        Number of histogram bins.
    suptitle : str | None
    pipe_color : str | None
        Color for the observed marker.  Defaults to ``COLORS["red"]``.
    figsize : tuple | None
    fname : str | Path | None
    show : bool

    Returns
    -------
    fig : Figure
    p_value : float
        Two-tailed p-value: proportion of null values at least as
        extreme as *observed*.
    """
    null_values = np.asarray(null_values)
    if figsize is None:
        figsize = (8, 5)
    if pipe_color is None:
        pipe_color = COLORS["red"]

    fig, ax = pub_figure(1, 1, figsize=figsize)

    # Histogram
    ax.hist(
        null_values,
        bins=n_bins,
        color=COLORS["gray"],
        alpha=0.5,
        edgecolor="white",
        linewidth=0.5,
        density=True,
        zorder=2,
        label=f"Null (N = {len(null_values):,})",
    )

    # CI bounds
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
    ax.axvline(lo, color=COLORS["gray"], ls=":", lw=0.8, alpha=0.6)
    ax.axvline(hi, color=COLORS["gray"], ls=":", lw=0.8, alpha=0.6)

    # Observed
    ax.axvline(
        observed,
        color=pipe_color,
        lw=2.5,
        ls="--",
        zorder=5,
        label=f"Observed = {observed:.3f}",
    )

    # p-value (two-tailed)
    p_value = float(np.mean(np.abs(null_values) >= np.abs(observed)))

    ax.annotate(
        f"p = {p_value:.4f}",
        xy=(observed, ax.get_ylim()[1] * 0.92),
        fontsize=FONTS["annotation"],
        fontweight="bold",
        ha="left" if observed > np.median(null_values) else "right",
        va="top",
        color=pipe_color,
        xytext=(8, 0),
        textcoords="offset points",
    )

    ax.set_xlabel(metric_label, fontsize=FONTS["label"])
    ax.set_ylabel("Density", fontsize=FONTS["label"])
    pub_legend(ax, fontsize=8)
    style_axes(ax)

    title = suptitle or f"Permutation Null Distribution â€” {metric_label}"
    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname), p_value


# =====================================================================
# plot_erp_forest â€” per-subject effect-size forest plot
# =====================================================================


def plot_erp_forest(
    df,
    metric_col="hedges_g",
    *,
    ci_col=None,
    se_col=None,
    group_col="pipeline",
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
    """Per-subject effect-size forest plot with CI and pooled mean.

    Each subject is a row; horizontal lines show the 95 % CI (Â± 1.96
    SE).  The pooled group mean is drawn at the bottom as a diamond.

    Parameters
    ----------
    df : DataFrame
        Long-form metrics table.
    metric_col : str
        Column with the effect-size point estimate (default ``hedges_g``).
    ci_col : str | None
        Column with a pre-computed 95 % CI half-width.  If *None* and
        *se_col* is also *None*, the CI is estimated from the standard
        deviation of *metric_col* across subjects (very rough).
    se_col : str | None
        Column with the standard error per subject.
    group_col, subject_col : str
    target_group : str | None
        Pipeline to display.  Defaults to the last in sorted order.
    baseline_group : str | None
        If given, plot that pipeline side-by-side in a lighter shade.
    group_colors, group_labels : dict | None
    metric_label : str | None
    reference_line : float | None
        Vertical reference line (default 0.0).
    suptitle : str | None
    figsize : tuple | None
    fname : str | Path | None
    show : bool

    Returns
    -------
    fig : Figure
    """
    groups = sorted(df[group_col].unique())
    if target_group is None:
        target_group = groups[-1]
    if metric_label is None:
        metric_label = metric_col.replace("_", " ").title()
    if group_labels is None:
        group_labels = {}

    # Get target data
    df_t = df.loc[df[group_col] == target_group].copy()
    df_t = df_t.sort_values(metric_col, ascending=True).reset_index(drop=True)
    subjects = df_t[subject_col].values
    n_sub = len(subjects)
    vals = df_t[metric_col].values.astype(float)

    # CI half-widths
    if ci_col is not None and ci_col in df_t.columns:
        hw = df_t[ci_col].values.astype(float)
    elif se_col is not None and se_col in df_t.columns:
        hw = 1.96 * df_t[se_col].values.astype(float)
    else:
        # Rough estimate: pooled SD across subjects
        sd = vals.std(ddof=1) if n_sub > 1 else 1.0
        hw = np.full(n_sub, 1.96 * sd / np.sqrt(n_sub))

    if figsize is None:
        figsize = (8, max(4, n_sub * 0.35 + 2))

    fig, ax = pub_figure(1, 1, figsize=figsize)

    y_pos = np.arange(n_sub)
    t_color = _pipe_color(target_group, group_colors)
    t_label = group_labels.get(target_group, target_group)

    # Optional baseline group
    if baseline_group is not None:
        df_b = df.loc[df[group_col] == baseline_group].copy()
        b_color = _pipe_color(baseline_group, group_colors)
        b_label = group_labels.get(baseline_group, baseline_group)
        for i, sub in enumerate(subjects):
            bv = df_b.loc[df_b[subject_col] == sub, metric_col]
            if len(bv):
                ax.plot(
                    bv.iloc[0],
                    y_pos[i],
                    "o",
                    color=b_color,
                    markersize=5,
                    alpha=0.35,
                    zorder=2,
                )
        # Pooled baseline mean marker at bottom
        b_mean = df_b[metric_col].mean()
        ax.plot(
            b_mean,
            -1.2,
            "D",
            color=b_color,
            markersize=9,
            zorder=6,
            alpha=0.5,
            label=f"{b_label} mean = {b_mean:.3f}",
        )

    # Target group CIs
    ax.errorbar(
        vals,
        y_pos,
        xerr=hw,
        fmt="o",
        color=t_color,
        ecolor=t_color,
        elinewidth=1.2,
        capsize=3,
        markersize=5,
        alpha=0.85,
        zorder=4,
        label=t_label,
    )

    # Pooled target mean (diamond at bottom)
    t_mean = vals.mean()
    t_se = vals.std(ddof=1) / np.sqrt(n_sub) if n_sub > 1 else 0
    ax.errorbar(
        t_mean,
        -1.2,
        xerr=1.96 * t_se,
        fmt="D",
        color=t_color,
        ecolor=t_color,
        elinewidth=2,
        capsize=4,
        markersize=10,
        zorder=6,
        label=f"Pooled mean = {t_mean:.3f}",
    )

    # Reference line
    if reference_line is not None:
        ax.axvline(
            reference_line,
            color=COLORS["gray"],
            ls="--",
            lw=0.8,
            alpha=0.5,
        )

    # Aesthetics
    ax.set_yticks(list(y_pos) + [-1.2])
    ax.set_yticklabels(
        list(subjects) + ["Pooled"],
        fontsize=FONTS["tick"],
    )
    ax.set_xlabel(metric_label, fontsize=FONTS["label"])
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, zorder=0)
    pub_legend(ax, fontsize=7, loc="lower right")
    style_axes(ax)

    title = (
        suptitle or f"Forest Plot â€” {group_labels.get(target_group, target_group)}"
    )
    fig.suptitle(title, fontsize=FONTS["suptitle"], fontweight="bold")
    return _finalize_fig(fig, show=show, fname=fname)


# =====================================================================
# Private helpers
# =====================================================================


def _ch_index(epochs_or_array, ch_name):
    """Get channel index from Epochs or assume array ordering."""
    if hasattr(epochs_or_array, "ch_names"):
        return epochs_or_array.ch_names.index(ch_name)
    # Fallback â€” caller must provide integer directly
    return int(ch_name)


def _get_times_ms(epochs):
    """Return the time vector in milliseconds."""
    if hasattr(epochs, "times"):
        return epochs.times * 1000
    raise ValueError("Cannot determine time vector from epochs object.")
