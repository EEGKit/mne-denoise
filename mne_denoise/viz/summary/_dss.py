"""DSS summary dashboards.

Public functions
----------------
- :func:`plot_dss_summary`
- :func:`plot_dss_segmented_summary`
- :func:`plot_dss_mode_comparison`
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from ...qa import spectral_distortion, suppression_ratio, variance_removed
from .._utils import (
    _get_bias_name,
    _get_dss_removed,
    _get_eigenvalues,
    _get_info,
    _get_n_selected,
    _get_segment_results,
    _is_segmented,
)
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
    import inspect

    from ...dss.linear import DSS

    sfreq = data.info["sfreq"]
    data_orig = data.get_data()

    if smooth is None:
        smooth = int(sfreq / line_freq)

    bias_name = type(bias).__name__

    print(f"DSS + {bias_name} - 3-Way Comparison")
    print("=" * 65)

    dss_params = set(inspect.signature(DSS).parameters)

    def _make_dss(*, use_smooth=False, use_segmented=False):
        kwargs = {
            "bias": bias,
            "n_components": n_components,
            "return_type": "raw",
        }
        if "n_select" in dss_params:
            kwargs["n_select"] = n_select
        if "selection_method" in dss_params:
            kwargs["selection_method"] = selection_method
        if use_smooth and "smooth" in dss_params:
            kwargs["smooth"] = smooth
        if use_segmented and "segmented" in dss_params:
            kwargs["segmented"] = True
            if "max_prop_remove" in dss_params:
                kwargs["max_prop_remove"] = max_prop_remove
            if "min_select" in dss_params:
                kwargs["min_select"] = min_select
        return DSS(**kwargs)

    def _selected_count(est):
        n_sel = _get_n_selected(est)
        if n_sel > 0:
            return n_sel
        ev = getattr(est, "eigenvalues_", None)
        if ev is not None and len(ev) > 0:
            return len(ev)
        return 0

    compat_mode = not {"n_select", "selection_method", "smooth", "segmented"}.issubset(
        dss_params
    )
    if compat_mode:
        print("  Note: DSS backend lacks advanced mode args; using compatibility path.")

    dss_plain = _make_dss(use_smooth=False, use_segmented=False)
    raw_plain = dss_plain.fit_transform(data)
    data_plain = raw_plain.get_data() if hasattr(raw_plain, "get_data") else raw_plain
    ev_plain = np.asarray(getattr(dss_plain, "eigenvalues_", np.array([])))
    n_plain = _selected_count(dss_plain)
    ev_plain_max = ev_plain[0] if ev_plain.size > 0 else np.nan
    print(
        f"  A) Plain DSS:          {n_plain} comp(s) | "
        f"max eigenvalue = {ev_plain_max:.6f}"
    )

    dss_smooth = _make_dss(use_smooth=True, use_segmented=False)
    raw_smooth = dss_smooth.fit_transform(data)
    data_smooth = (
        raw_smooth.get_data() if hasattr(raw_smooth, "get_data") else raw_smooth
    )
    ev_smooth = np.asarray(getattr(dss_smooth, "eigenvalues_", np.array([])))
    n_smooth = _selected_count(dss_smooth)
    ev_smooth_max = ev_smooth[0] if ev_smooth.size > 0 else np.nan
    boost = (
        ev_smooth_max / ev_plain_max
        if np.isfinite(ev_plain_max) and ev_plain_max > 0
        else float("inf")
    )
    print(
        f"  B) + Smoothing:        {n_smooth} comp(s) | "
        f"max eigenvalue = {ev_smooth_max:.6f} ({boost:.1f}x boost)"
    )

    dss_seg = _make_dss(use_smooth=True, use_segmented=True)
    raw_seg = dss_seg.fit_transform(data)
    data_seg = raw_seg.get_data() if hasattr(raw_seg, "get_data") else raw_seg

    segment_results = getattr(dss_seg, "segment_results_", None)
    if segment_results:
        n_segments = len(segment_results)
        seg_n_sel = [
            s.get("n_selected", s.get("n_removed", 0)) for s in segment_results
        ]
        seg_evals = [
            s["eigenvalues"][0]
            for s in segment_results
            if "eigenvalues" in s and len(s["eigenvalues"]) > 0
        ]
    else:
        n_segments = 1
        seg_n_sel = [_selected_count(dss_seg)]
        ev_seg = np.asarray(getattr(dss_seg, "eigenvalues_", np.array([])))
        seg_evals = [ev_seg[0]] if ev_seg.size > 0 else [np.nan]

    seg_eval_min = np.nanmin(seg_evals) if len(seg_evals) > 0 else np.nan
    seg_eval_max = np.nanmax(seg_evals) if len(seg_evals) > 0 else np.nan
    total_removed = sum(seg_n_sel)
    print(
        f"  C) + Segmentation:     {n_segments} seg(s) | "
        f"n_removed: {min(seg_n_sel)}-{max(seg_n_sel)} "
        f"(total {total_removed}) | "
        f"eigenvalue range: {seg_eval_min:.4f}-{seg_eval_max:.4f}"
    )

    nperseg = min(int(sfreq * 4), data_orig.shape[-1])
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
    ax.set_ylabel("PSD (V\u00b2/Hz)")
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
    ax.set_ylabel("PSD (V\u00b2/Hz)")
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
