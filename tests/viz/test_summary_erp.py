"""Tests for ERP-related summary plots, grouped stats, and ERP I/O helpers."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

pd = pytest.importorskip("pandas")

# â”€â”€ Imports under test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

from mne_denoise.viz import (
    DEFAULT_PIPE_COLORS,
    DEFAULT_PIPE_LABELS,
    DEFAULT_PIPE_ORDER,
    plot_erp_condition_interaction,
    plot_erp_endpoint_summary,
    plot_erp_grand_condition_interaction,
    plot_erp_signal_diagnostics,
)
from mne_denoise.viz._seaborn import _try_import_seaborn
from mne_denoise.viz.erp_io import (
    ERPGroupData,
    aggregate_erp_results,
    load_subject_erp_results,
    save_subject_erp_results,
)
from mne_denoise.viz.stats import (
    plot_forest,
    plot_metric_slopes,
    plot_metric_violins,
    plot_null_distribution,
)
from mne_denoise.viz.summary import DEFAULT_ERP_WINDOWS, _pipe_color, _pipe_label
from mne_denoise.viz.theme import _finalize_fig

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  Shared Fixtures
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

N_EPOCHS = 20
N_CH = 5
N_TIMES = 250
SFREQ = 250.0
CH_NAMES = ["Fz", "Cz", "Pz", "Oz", "F3"]
PIPES = ["C0", "C1", "C2"]


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def mne_info():
    info = mne.create_info(CH_NAMES, SFREQ, "eeg")
    info.set_montage("standard_1020")
    return info


@pytest.fixture(scope="module")
def epochs_dict(mne_info, rng):
    """Dict of {pipe_tag: EpochsArray} for all three pipelines."""
    out = {}
    for ptag in PIPES:
        data = rng.standard_normal((N_EPOCHS, N_CH, N_TIMES))
        # Inject a mild signal to make metrics non-trivial
        t = np.linspace(-0.2, 0.8, N_TIMES)
        signal = np.exp(-((t - 0.3) ** 2) / 0.01) * (0.5 + 0.3 * PIPES.index(ptag))
        data[:, 1, :] += signal  # Cz
        data[:, 2, :] += signal * 0.8  # Pz
        ev = mne.EpochsArray(data, mne_info, tmin=-0.2, verbose=False)
        out[ptag] = ev
    return out


@pytest.fixture(scope="module")
def evokeds_dict(epochs_dict):
    """Dict of {pipe_tag: Evoked}."""
    return {ptag: ep.average() for ptag, ep in epochs_dict.items()}


@pytest.fixture(scope="module")
def condition_masks():
    """Deviant and standard boolean masks over 20 epochs."""
    dev = np.zeros(N_EPOCHS, dtype=bool)
    dev[:8] = True
    std = np.zeros(N_EPOCHS, dtype=bool)
    std[8:] = True
    return dev, std


@pytest.fixture(scope="module")
def times_ms(epochs_dict):
    return epochs_dict["C0"].times * 1000


@pytest.fixture(scope="module")
def diff_waves_and_effects(epochs_dict, condition_masks, times_ms):
    """Build diff_waves, diff_se, effect_sizes dicts for condition_interaction."""
    dev_mask, std_mask = condition_masks
    conditions = ["lab", "oval"]
    diff_waves = {}
    diff_se = {}
    effect_sizes = {}
    for cond in conditions:
        for ptag in PIPES:
            ep = epochs_dict[ptag]
            data = ep.get_data()
            ci = 2  # Pz
            dev_data = data[dev_mask, ci, :] * 1e6
            std_data = data[std_mask, ci, :] * 1e6
            diff = dev_data.mean(axis=0) - std_data.mean(axis=0)
            se = np.sqrt(
                dev_data.var(axis=0) / dev_mask.sum()
                + std_data.var(axis=0) / std_mask.sum()
            )
            diff_waves[(cond, ptag)] = diff
            diff_se[(cond, ptag)] = se
            effect_sizes[(cond, ptag)] = float(np.abs(diff).mean() / (se.mean() + 1e-9))
    return diff_waves, diff_se, effect_sizes, conditions


@pytest.fixture(scope="module")
def metrics_df():
    """Synthetic long-form metrics DataFrame (8 subjects x 3 pipes)."""
    rng_local = np.random.default_rng(99)
    rows = []
    for i in range(8):
        sub = f"sub-{i + 1:02d}"
        for ptag in PIPES:
            rows.append(
                {
                    "subject": sub,
                    "pipeline": ptag,
                    "hedges_g": rng_local.normal(0.3 * PIPES.index(ptag), 0.2),
                    "peak_lat_ms": rng_local.normal(300, 20),
                    "morph_r": rng_local.uniform(0.6, 1.0),
                    "morph_nrmse": rng_local.uniform(0.05, 0.3),
                    "split_half_r": rng_local.uniform(0.4, 0.9),
                    "auc": rng_local.uniform(0.45, 0.85),
                }
            )
    return pd.DataFrame(rows)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  A. Private helpers
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestPrivateHelpers:
    """Tests for private ERP helper utilities."""

    def test_pipe_color_default(self):
        assert _pipe_color("C0") == DEFAULT_PIPE_COLORS["C0"]

    def test_pipe_color_override(self):
        assert _pipe_color("C0", {"C0": "#ff0000"}) == "#ff0000"

    def test_pipe_color_unknown_falls_back(self):
        c = _pipe_color("UNKNOWN")
        assert isinstance(c, str)

    def test_pipe_label_default(self):
        assert _pipe_label("C0") == DEFAULT_PIPE_LABELS["C0"]

    def test_pipe_label_override(self):
        assert _pipe_label("C0", {"C0": "Custom"}) == "Custom"

    def test_pipe_label_unknown_falls_back(self):
        assert _pipe_label("ZZZ") == "ZZZ"

    def test_try_import_seaborn(self):
        sns = _try_import_seaborn()
        assert hasattr(sns, "violinplot")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  B. plot_erp_signal_diagnostics
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestPlotERPSignalDiagnostics:
    def test_basic(self, epochs_dict, evokeds_dict, condition_masks):
        dev, std = condition_masks
        fig = plot_erp_signal_diagnostics(
            epochs_dict,
            evokeds_dict,
            channels=("Cz", "Pz"),
            dev_mask=dev,
            std_mask=std,
            subject="sub-01",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_no_condition_masks(self, epochs_dict, evokeds_dict):
        """Diff-wave row should show placeholder text when masks are None."""
        fig = plot_erp_signal_diagnostics(
            epochs_dict,
            evokeds_dict,
            channels=("Cz",),
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_insufficient_trials(self, epochs_dict, evokeds_dict):
        """When dev_mask has < 3 deviant trials, show 'Insufficient'."""
        dev = np.zeros(N_EPOCHS, dtype=bool)
        dev[0] = True  # Only 1 deviant
        std = ~dev
        fig = plot_erp_signal_diagnostics(
            epochs_dict,
            evokeds_dict,
            dev_mask=dev,
            std_mask=std,
            channels=("Cz",),
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_fname_saves(self, epochs_dict, evokeds_dict, condition_masks, tmp_path):
        dev, std = condition_masks
        fpath = tmp_path / "diag.png"
        plot_erp_signal_diagnostics(
            epochs_dict,
            evokeds_dict,
            dev_mask=dev,
            std_mask=std,
            show=False,
            fname=str(fpath),
        )
        assert fpath.exists()
        assert fpath.stat().st_size > 0

    def test_custom_pipe_order(self, epochs_dict, evokeds_dict, condition_masks):
        dev, std = condition_masks
        fig = plot_erp_signal_diagnostics(
            epochs_dict,
            evokeds_dict,
            dev_mask=dev,
            std_mask=std,
            pipe_order=["C2", "C0"],
            pipe_colors={"C0": "#aaa", "C2": "#bbb"},
            pipe_labels={"C0": "Base", "C2": "DSS"},
            erp_windows={"P300": (0.25, 0.5)},
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_single_channel(self, epochs_dict, evokeds_dict, condition_masks):
        dev, std = condition_masks
        fig = plot_erp_signal_diagnostics(
            epochs_dict,
            evokeds_dict,
            channels=("Cz",),
            dev_mask=dev,
            std_mask=std,
            show=False,
        )
        assert isinstance(fig, plt.Figure)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  C. plot_erp_condition_interaction
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestPlotConditionInteraction:
    def test_basic(self, diff_waves_and_effects, times_ms):
        dw, se, es, conds = diff_waves_and_effects
        fig1, fig2 = plot_erp_condition_interaction(
            dw,
            se,
            es,
            times_ms,
            conditions=conds,
            show=False,
        )
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)

    def test_fname_dual_save(self, diff_waves_and_effects, times_ms, tmp_path):
        dw, se, es, conds = diff_waves_and_effects
        fpath = tmp_path / "cond.png"
        fig1, fig2 = plot_erp_condition_interaction(
            dw,
            se,
            es,
            times_ms,
            conditions=conds,
            fname=str(fpath),
            show=False,
        )
        assert (tmp_path / "cond_diffwaves.png").exists()
        assert (tmp_path / "cond_interaction.png").exists()

    def test_fname_dual_save_with_show(
        self, diff_waves_and_effects, times_ms, tmp_path
    ):
        dw, se, es, conds = diff_waves_and_effects
        fpath = tmp_path / "cond_show.png"
        with pytest.MonkeyPatch.context() as m:
            m.setattr(plt, "show", lambda *a, **kw: None)
            fig1, fig2 = plot_erp_condition_interaction(
                dw,
                se,
                es,
                times_ms,
                conditions=conds,
                fname=str(fpath),
                show=True,
            )
        assert (tmp_path / "cond_show_diffwaves.png").exists()

    def test_single_condition(self, diff_waves_and_effects, times_ms):
        dw, se, es, _ = diff_waves_and_effects
        # Filter to just "lab"
        dw1 = {k: v for k, v in dw.items() if k[0] == "lab"}
        se1 = {k: v for k, v in se.items() if k[0] == "lab"}
        es1 = {k: v for k, v in es.items() if k[0] == "lab"}
        fig1, fig2 = plot_erp_condition_interaction(
            dw1,
            se1,
            es1,
            times_ms,
            conditions=["lab"],
            show=False,
        )
        assert isinstance(fig1, plt.Figure)

    def test_missing_key_skipped(self, times_ms):
        """A missing (cond, pipe) key should be silently skipped."""
        dw = {("lab", "C0"): np.zeros(N_TIMES)}
        se = {("lab", "C0"): np.ones(N_TIMES) * 0.1}
        es = {("lab", "C0"): 0.5}
        fig1, fig2 = plot_erp_condition_interaction(
            dw,
            se,
            es,
            times_ms,
            conditions=["lab"],
            pipe_order=["C0", "C2"],
            show=False,
        )
        assert isinstance(fig1, plt.Figure)

    def test_no_se(self, diff_waves_and_effects, times_ms):
        """diff_se values of None should not crash."""
        dw, _, es, conds = diff_waves_and_effects
        se_none = dict.fromkeys(dw)
        fig1, fig2 = plot_erp_condition_interaction(
            dw,
            se_none,
            es,
            times_ms,
            conditions=conds,
            show=False,
        )
        assert isinstance(fig1, plt.Figure)

    def test_custom_labels(self, diff_waves_and_effects, times_ms):
        dw, se, es, conds = diff_waves_and_effects
        fig1, fig2 = plot_erp_condition_interaction(
            dw,
            se,
            es,
            times_ms,
            conditions=conds,
            condition_labels={"lab": "Laboratory", "oval": "Oval"},
            pipe_colors={"C0": "#111", "C1": "#222", "C2": "#333"},
            pipe_labels={"C0": "A", "C1": "B", "C2": "C"},
            subject="sub-99",
            show=False,
        )
        assert isinstance(fig1, plt.Figure)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  D. plot_metric_violins
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestPlotMetricViolins:
    def test_basic(self, metrics_df):
        fig = plot_metric_violins(
            metrics_df,
            ["hedges_g", "auc"],
            metric_labels=["Hedges' g", "AUC"],
            group_col="pipeline",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_single_metric(self, metrics_df):
        fig = plot_metric_violins(
            metrics_df,
            ["hedges_g"],
            group_col="pipeline",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_no_paired(self, metrics_df):
        fig = plot_metric_violins(
            metrics_df,
            ["hedges_g"],
            group_col="pipeline",
            show_paired=False,
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_reference_lines(self, metrics_df):
        fig = plot_metric_violins(
            metrics_df,
            ["auc"],
            group_col="pipeline",
            reference_lines={"auc": [(0.5, {"color": "k", "ls": ":", "alpha": 0.5})]},
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_custom_group(self, metrics_df):
        fig = plot_metric_violins(
            metrics_df,
            ["hedges_g"],
            group_col="pipeline",
            group_order=["C2", "C0"],
            group_colors={"C0": "#aaa", "C2": "#bbb"},
            group_labels={"C0": "Base", "C2": "DSS"},
            suptitle="Custom Title",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_empty_metric_col(self, metrics_df):
        """A metric column with all NaN should show 'No data' text."""
        df = metrics_df.copy()
        df["empty"] = np.nan
        fig = plot_metric_violins(df, ["empty"], group_col="pipeline", show=False)
        assert isinstance(fig, plt.Figure)

    def test_fname(self, metrics_df, tmp_path):
        fpath = tmp_path / "violins.png"
        plot_metric_violins(
            metrics_df,
            ["hedges_g"],
            group_col="pipeline",
            show=False,
            fname=str(fpath),
        )
        assert fpath.exists()

    def test_single_subject(self):
        """With 1 subject, paired lines should not be drawn."""
        df = pd.DataFrame(
            [
                {"subject": "sub-01", "pipeline": "C0", "hedges_g": 0.1},
                {"subject": "sub-01", "pipeline": "C2", "hedges_g": 0.8},
            ]
        )
        fig = plot_metric_violins(df, ["hedges_g"], group_col="pipeline", show=False)
        assert isinstance(fig, plt.Figure)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  E. plot_erp_endpoint_summary
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestPlotEndpointSummary:
    def test_basic(self, metrics_df):
        fig = plot_erp_endpoint_summary(
            metrics_df,
            ["hedges_g", "auc"],
            metric_labels=["Hedges' g", "AUC"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_with_null_distributions(self, metrics_df):
        rng = np.random.default_rng(7)
        null = {"C2": {"g": rng.normal(0, 0.1, 500)}}
        fig = plot_erp_endpoint_summary(
            metrics_df,
            ["hedges_g", "auc"],
            null_distributions=null,
            null_metric="hedges_g",
            null_pipe="C2",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_slope_panel(self, metrics_df):
        fig = plot_erp_endpoint_summary(
            metrics_df,
            ["hedges_g"],
            slope_metric="hedges_g",
            slope_from="C0",
            slope_to="C2",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_single_subject(self):
        df = pd.DataFrame(
            [
                {
                    "subject": "sub-01",
                    "pipeline": "C0",
                    "hedges_g": 0.1,
                    "auc": 0.5,
                },
                {
                    "subject": "sub-01",
                    "pipeline": "C2",
                    "hedges_g": 0.8,
                    "auc": 0.7,
                },
            ]
        )
        fig = plot_erp_endpoint_summary(
            df,
            ["hedges_g", "auc"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_auc_reference_line(self, metrics_df):
        """The auc panel should show a horizontal line at 0.5."""
        fig = plot_erp_endpoint_summary(
            metrics_df,
            ["auc"],
            metric_labels=["AUC"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_fname(self, metrics_df, tmp_path):
        fpath = tmp_path / "summary.png"
        plot_erp_endpoint_summary(
            metrics_df,
            ["hedges_g"],
            show=False,
            fname=str(fpath),
        )
        assert fpath.exists()

    def test_custom_labels(self, metrics_df):
        fig = plot_erp_endpoint_summary(
            metrics_df,
            ["hedges_g"],
            group_order=["C0", "C2"],
            group_colors={"C0": "#111", "C2": "#222"},
            group_labels={"C0": "Base", "C2": "DSS"},
            suptitle="Custom",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_empty_metric(self, metrics_df):
        df = metrics_df.copy()
        df["empty"] = np.nan
        fig = plot_erp_endpoint_summary(df, ["empty"], show=False)
        assert isinstance(fig, plt.Figure)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  F. plot_metric_slopes
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestPlotPipelineSlopes:
    def test_basic(self, metrics_df):
        fig = plot_metric_slopes(
            metrics_df,
            metric_cols=["hedges_g", "auc"],
            metric_labels=["Hedges' g", "AUC"],
            group_col="pipeline",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_single_metric(self, metrics_df):
        fig = plot_metric_slopes(
            metrics_df,
            metric_cols=["hedges_g"],
            group_col="pipeline",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_returns_none_single_subject(self):
        df = pd.DataFrame(
            [
                {"subject": "sub-01", "pipeline": "C0", "hedges_g": 0.1},
                {"subject": "sub-01", "pipeline": "C2", "hedges_g": 0.8},
            ]
        )
        fig = plot_metric_slopes(
            df,
            metric_cols=["hedges_g"],
            group_col="pipeline",
            show=False,
        )
        assert fig is None

    def test_reference_lines(self, metrics_df):
        fig = plot_metric_slopes(
            metrics_df,
            metric_cols=["auc"],
            group_col="pipeline",
            reference_lines={"auc": [(0.5, {"color": "k", "ls": ":"})]},
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_custom_labels(self, metrics_df):
        fig = plot_metric_slopes(
            metrics_df,
            metric_cols=["hedges_g"],
            group_col="pipeline",
            group_order=["C0", "C2"],
            group_colors={"C0": "#aaa", "C2": "#bbb"},
            group_labels={"C0": "Base", "C2": "DSS"},
            suptitle="Custom Slopes",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_fname(self, metrics_df, tmp_path):
        fpath = tmp_path / "slopes.png"
        plot_metric_slopes(
            metrics_df,
            metric_cols=["hedges_g"],
            group_col="pipeline",
            show=False,
            fname=str(fpath),
        )
        assert fpath.exists()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  H. plot_erp_grand_condition_interaction
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestPlotGrandConditionInteraction:
    @pytest.fixture()
    def group_data(self, rng):
        """Build group-level diff_waves and effect_sizes for 5 subjects."""
        n_sub = 5
        conditions = ["lab", "oval"]
        all_dw = {}
        all_es = {}
        for cond in conditions:
            for ptag in PIPES:
                all_dw[(cond, ptag)] = rng.standard_normal((n_sub, N_TIMES))
                all_es[(cond, ptag)] = rng.normal(0.5, 0.2, n_sub)
        return all_dw, all_es, conditions

    def test_basic(self, group_data, times_ms):
        all_dw, all_es, conds = group_data
        fig1, fig2 = plot_erp_grand_condition_interaction(
            all_dw,
            all_es,
            times_ms,
            conditions=conds,
            show=False,
        )
        assert isinstance(fig1, plt.Figure)
        assert isinstance(fig2, plt.Figure)

    def test_single_condition(self, group_data, times_ms):
        all_dw, all_es, _ = group_data
        dw1 = {k: v for k, v in all_dw.items() if k[0] == "lab"}
        es1 = {k: v for k, v in all_es.items() if k[0] == "lab"}
        fig1, fig2 = plot_erp_grand_condition_interaction(
            dw1,
            es1,
            times_ms,
            conditions=["lab"],
            show=False,
        )
        assert isinstance(fig1, plt.Figure)

    def test_fname_dual_save(self, group_data, times_ms, tmp_path):
        all_dw, all_es, conds = group_data
        fpath = tmp_path / "grand_cond.png"
        plot_erp_grand_condition_interaction(
            all_dw,
            all_es,
            times_ms,
            conditions=conds,
            fname=str(fpath),
            show=False,
        )
        assert (tmp_path / "grand_cond_diffwaves.png").exists()
        assert (tmp_path / "grand_cond_interaction.png").exists()

    def test_fname_with_show(self, group_data, times_ms, tmp_path):
        all_dw, all_es, conds = group_data
        fpath = tmp_path / "grand_cond_s.png"
        with pytest.MonkeyPatch.context() as m:
            m.setattr(plt, "show", lambda *a, **kw: None)
            plot_erp_grand_condition_interaction(
                all_dw,
                all_es,
                times_ms,
                conditions=conds,
                fname=str(fpath),
                show=True,
            )
        assert (tmp_path / "grand_cond_s_diffwaves.png").exists()

    def test_missing_key(self, times_ms, rng):
        """Missing (cond, pipe) keys should be silently skipped."""
        all_dw = {("lab", "C0"): rng.standard_normal((3, N_TIMES))}
        all_es = {("lab", "C0"): np.array([0.5, 0.6, 0.7])}
        fig1, fig2 = plot_erp_grand_condition_interaction(
            all_dw,
            all_es,
            times_ms,
            conditions=["lab"],
            pipe_order=["C0", "C2"],
            show=False,
        )
        assert isinstance(fig1, plt.Figure)

    def test_single_subject_no_sem(self, times_ms, rng):
        """N=1 subject should produce zero SEM band."""
        all_dw = {("lab", "C0"): rng.standard_normal((1, N_TIMES))}
        all_es = {("lab", "C0"): np.array([0.5])}
        fig1, fig2 = plot_erp_grand_condition_interaction(
            all_dw,
            all_es,
            times_ms,
            conditions=["lab"],
            pipe_order=["C0"],
            show=False,
        )
        assert isinstance(fig1, plt.Figure)

    def test_custom_labels(self, group_data, times_ms):
        all_dw, all_es, conds = group_data
        fig1, fig2 = plot_erp_grand_condition_interaction(
            all_dw,
            all_es,
            times_ms,
            conditions=conds,
            condition_labels={"lab": "Lab", "oval": "Oval"},
            pipe_colors={"C0": "#111", "C1": "#222", "C2": "#333"},
            pipe_labels={"C0": "A", "C1": "B", "C2": "C"},
            suptitle="Custom Title",
            show=False,
        )
        assert isinstance(fig1, plt.Figure)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  I. plot_null_distribution
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestPlotNullDistribution:
    def test_basic(self, rng):
        null = rng.normal(0, 0.1, 500)
        fig, pval = plot_null_distribution(null, observed=0.8, show=False)
        assert isinstance(fig, plt.Figure)
        assert 0.0 <= pval <= 1.0

    def test_observed_in_null_range(self, rng):
        null = rng.normal(0, 1.0, 500)
        fig, pval = plot_null_distribution(null, observed=0.0, show=False)
        assert pval > 0.1  # should be non-significant

    def test_extreme_observed(self, rng):
        null = rng.normal(0, 0.01, 500)
        fig, pval = plot_null_distribution(null, observed=10.0, show=False)
        assert pval < 0.01

    def test_custom_options(self, rng):
        null = rng.normal(0, 0.5, 300)
        fig, pval = plot_null_distribution(
            null,
            observed=0.3,
            metric_label="Effect Size",
            ci=99,
            n_bins=30,
            suptitle="Custom Null",
            series_color="#ff0000",
            figsize=(10, 6),
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_fname(self, rng, tmp_path):
        null = rng.normal(0, 0.1, 100)
        fpath = tmp_path / "null.png"
        plot_null_distribution(
            null,
            observed=0.5,
            show=False,
            fname=str(fpath),
        )
        assert fpath.exists()

    def test_observed_below_median(self, rng):
        """Annotation ha changes when observed < median."""
        null = rng.normal(1.0, 0.1, 500)
        fig, pval = plot_null_distribution(null, observed=0.5, show=False)
        assert isinstance(fig, plt.Figure)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  J. plot_forest
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestPlotForest:
    def test_basic(self, metrics_df):
        fig = plot_forest(
            metrics_df,
            metric_col="hedges_g",
            group_col="pipeline",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_with_baseline_group(self, metrics_df):
        fig = plot_forest(
            metrics_df,
            metric_col="hedges_g",
            group_col="pipeline",
            target_group="C2",
            baseline_group="C0",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_with_se_col(self, metrics_df):
        df = metrics_df.copy()
        df["hedges_g_se"] = 0.1
        fig = plot_forest(
            df,
            metric_col="hedges_g",
            group_col="pipeline",
            se_col="hedges_g_se",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_with_ci_col(self, metrics_df):
        df = metrics_df.copy()
        df["hedges_g_ci"] = 0.2
        fig = plot_forest(
            df,
            metric_col="hedges_g",
            group_col="pipeline",
            ci_col="hedges_g_ci",
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_custom_options(self, metrics_df):
        fig = plot_forest(
            metrics_df,
            metric_col="hedges_g",
            group_col="pipeline",
            target_group="C2",
            group_colors={"C0": "#aaa", "C2": "#bbb"},
            group_labels={"C0": "Base", "C2": "DSS"},
            metric_label="Custom Label",
            reference_line=0.5,
            suptitle="Custom Forest",
            figsize=(10, 8),
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_no_reference_line(self, metrics_df):
        fig = plot_forest(
            metrics_df,
            metric_col="hedges_g",
            group_col="pipeline",
            reference_line=None,
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_fname(self, metrics_df, tmp_path):
        fpath = tmp_path / "forest.png"
        plot_forest(
            metrics_df,
            metric_col="hedges_g",
            group_col="pipeline",
            show=False,
            fname=str(fpath),
        )
        assert fpath.exists()

    def test_single_subject(self):
        df = pd.DataFrame(
            [
                {"subject": "sub-01", "pipeline": "C0", "hedges_g": 0.1},
                {"subject": "sub-01", "pipeline": "C2", "hedges_g": 0.8},
            ]
        )
        fig = plot_forest(
            df,
            metric_col="hedges_g",
            group_col="pipeline",
            show=False,
        )
        assert isinstance(fig, plt.Figure)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  K. _finalize_fig â€” mkdir fix
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestFinalizeFigMkdir:
    def test_creates_nested_directory(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c" / "fig.png"
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        _finalize_fig(fig, show=False, fname=deep)
        assert deep.exists()

    def test_no_fname_no_error(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        ret = _finalize_fig(fig, show=False, fname=None)
        assert ret is fig

    def test_show_false_fname_closes(self, tmp_path):
        """show=False + fname should close the figure."""
        fpath = tmp_path / "closed.png"
        fig, ax = plt.subplots()
        ax.plot([1, 2])
        _finalize_fig(fig, show=False, fname=fpath)
        assert fpath.exists()

    def test_tight_false(self, tmp_path):
        fpath = tmp_path / "notight.png"
        fig, ax = plt.subplots()
        ax.plot([1, 2])
        ret = _finalize_fig(fig, show=False, fname=fpath, tight=False)
        assert ret is fig
        assert fpath.exists()


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  L. erp_io â€” save / load / aggregate
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


class TestERPIO:
    @pytest.fixture()
    def sample_data(self, rng, mne_info):
        """Create sample per-subject data for IO round-trip."""
        pipe_evokeds = {}
        for ptag in PIPES:
            data = rng.standard_normal((N_CH, N_TIMES))
            ev = mne.EvokedArray(data, mne_info, tmin=-0.2, verbose=False)
            pipe_evokeds[ptag] = ev

        times_ms = np.linspace(-200, 800, N_TIMES)
        diff_waves = {
            ("lab", "C0"): rng.standard_normal(N_TIMES),
            ("lab", "C2"): rng.standard_normal(N_TIMES),
            ("oval", "C0"): rng.standard_normal(N_TIMES),
            ("oval", "C2"): rng.standard_normal(N_TIMES),
        }
        effect_sizes = {
            ("lab", "C0"): 0.1,
            ("lab", "C2"): 0.9,
            ("oval", "C0"): 0.15,
            ("oval", "C2"): 0.85,
        }
        df = pd.DataFrame(
            [
                {
                    "subject": "sub-01",
                    "pipeline": "C0",
                    "hedges_g": 0.1,
                    "auc": 0.5,
                },
                {
                    "subject": "sub-01",
                    "pipeline": "C2",
                    "hedges_g": 0.8,
                    "auc": 0.7,
                },
            ]
        )
        return {
            "pipe_evokeds": pipe_evokeds,
            "times_ms": times_ms,
            "diff_waves": diff_waves,
            "effect_sizes": effect_sizes,
            "metrics_df": df,
        }

    def test_save_creates_files(self, sample_data, tmp_path):
        out_dir = tmp_path / "sub-01" / "erp_dss"
        save_subject_erp_results(
            out_dir,
            subject="sub-01",
            **sample_data,
        )
        assert (out_dir / "metrics.tsv").exists()
        assert (out_dir / "evokeds.npz").exists()
        assert (out_dir / "diff_waves.npz").exists()
        assert (out_dir / "erp_io_meta.json").exists()

    def test_load_round_trip(self, sample_data, tmp_path):
        out_dir = tmp_path / "sub-01" / "erp_dss"
        save_subject_erp_results(
            out_dir,
            subject="sub-01",
            **sample_data,
        )
        r = load_subject_erp_results(out_dir)
        assert r["subject"] == "sub-01"
        assert sorted(r["pipe_evokeds"].keys()) == PIPES
        assert r["pipe_evokeds"]["C0"].shape == (N_CH, N_TIMES)
        assert sorted(r["ch_names"]["C0"]) == sorted(CH_NAMES)
        assert r["times_ms"].shape == (N_TIMES,)
        assert ("lab", "C0") in r["diff_waves"]
        assert ("lab", "C2") in r["effect_sizes"]
        assert len(r["df"]) == 2

    def test_save_with_metrics_dict(self, tmp_path):
        out_dir = tmp_path / "sub-02" / "erp_dss"
        save_subject_erp_results(
            out_dir,
            subject="sub-02",
            metrics_dict={"C0": {"hedges_g": 0.1}, "C2": {"hedges_g": 0.9}},
        )
        r = load_subject_erp_results(out_dir)
        assert len(r["df"]) == 2
        assert "hedges_g" in r["df"].columns

    def test_save_with_pipe_order(self, sample_data, tmp_path):
        out_dir = tmp_path / "sub-01" / "erp_dss"
        save_subject_erp_results(
            out_dir,
            subject="sub-01",
            pipe_order=["C0", "C1", "C2"],
            **sample_data,
        )
        with open(out_dir / "erp_io_meta.json") as f:
            meta = json.load(f)
        assert meta["pipe_order"] == ["C0", "C1", "C2"]

    def test_save_raw_arrays(self, rng, tmp_path):
        """Evokeds can be plain ndarray instead of MNE objects."""
        out_dir = tmp_path / "sub-03" / "erp_dss"
        save_subject_erp_results(
            out_dir,
            subject="sub-03",
            pipe_evokeds={"C0": rng.standard_normal((5, 100))},
            times_ms=np.linspace(-200, 800, 100),
        )
        r = load_subject_erp_results(out_dir)
        assert r["pipe_evokeds"]["C0"].shape == (5, 100)

    def test_save_evokeds_derives_times(self, mne_info, tmp_path):
        """When times_ms is None but Evoked has .times, auto-derived."""
        out_dir = tmp_path / "sub-04" / "erp_dss"
        data = np.zeros((N_CH, N_TIMES))
        ev = mne.EvokedArray(data, mne_info, tmin=-0.2, verbose=False)
        save_subject_erp_results(
            out_dir,
            subject="sub-04",
            pipe_evokeds={"C0": ev},
            times_ms=None,
        )
        r = load_subject_erp_results(out_dir)
        assert "times_ms" in r
        assert r["times_ms"][0] == pytest.approx(-200.0, abs=1.0)

    def test_aggregate_multi_subject(self, sample_data, tmp_path, rng):
        """Aggregate 3 subjects' results."""
        for i in range(3):
            sub = f"sub-{i + 1:02d}"
            out_dir = tmp_path / sub / "erp_dss"
            # Use raw arrays to avoid sharing MNE objects
            sd = {
                "pipe_evokeds": {
                    p: rng.standard_normal((N_CH, N_TIMES)) for p in PIPES
                },
                "times_ms": sample_data["times_ms"],
                "diff_waves": {
                    k: rng.standard_normal(N_TIMES) for k in sample_data["diff_waves"]
                },
                "effect_sizes": {
                    k: float(rng.uniform(0, 1)) for k in sample_data["effect_sizes"]
                },
                "metrics_df": pd.DataFrame(
                    [
                        {
                            "subject": sub,
                            "pipeline": "C0",
                            "hedges_g": 0.1 * i,
                        },
                        {
                            "subject": sub,
                            "pipeline": "C2",
                            "hedges_g": 0.8 + 0.1 * i,
                        },
                    ]
                ),
            }
            save_subject_erp_results(out_dir, subject=sub, **sd)

        agg = aggregate_erp_results(tmp_path)
        assert isinstance(agg, ERPGroupData)
        assert len(agg.df) == 6  # 3 subs x 2 pipes
        assert len(agg.all_evokeds["C0"]) == 3
        assert agg.all_diff_waves[("lab", "C0")].shape == (3, N_TIMES)
        assert agg.all_effect_sizes[("lab", "C0")].shape == (3,)
        assert agg.times_ms is not None

    def test_aggregate_auto_discover(self, sample_data, tmp_path, rng):
        """subjects=None should auto-discover sub-* dirs."""
        for i in range(2):
            sub = f"sub-{i + 1:02d}"
            out_dir = tmp_path / sub / "erp_dss"
            save_subject_erp_results(
                out_dir,
                subject=sub,
                pipe_evokeds={p: rng.standard_normal((N_CH, N_TIMES)) for p in PIPES},
                times_ms=sample_data["times_ms"],
            )

        agg = aggregate_erp_results(tmp_path)
        assert len(agg.all_evokeds["C0"]) == 2

    def test_aggregate_empty(self, tmp_path):
        agg = aggregate_erp_results(tmp_path)
        assert len(agg.df) == 0
        assert agg.times_ms is None

    def test_aggregate_explicit_subjects(self, sample_data, tmp_path, rng):
        """Passing explicit subjects list."""
        for i in range(3):
            sub = f"sub-{i + 1:02d}"
            out_dir = tmp_path / sub / "erp_dss"
            save_subject_erp_results(
                out_dir,
                subject=sub,
                metrics_dict={"C0": {"hedges_g": 0.1 * i}},
            )

        # Only aggregate 2 of 3
        agg = aggregate_erp_results(tmp_path, subjects=["sub-01", "sub-03"])
        assert len(agg.df) == 2

    def test_load_empty_dir(self, tmp_path):
        """Loading from empty dir should return minimal result."""
        out_dir = tmp_path / "sub-empty" / "erp_dss"
        out_dir.mkdir(parents=True)
        r = load_subject_erp_results(out_dir)
        assert "subject" in r
        assert "df" not in r

    def test_aggregate_skips_missing(self, sample_data, tmp_path, rng):
        """Subjects with no erp_dss dir should be skipped."""
        # Create sub-01 properly
        out_dir = tmp_path / "sub-01" / "erp_dss"
        save_subject_erp_results(
            out_dir,
            subject="sub-01",
            metrics_dict={"C0": {"hedges_g": 0.1}},
        )
        # sub-02 has no erp_dss
        (tmp_path / "sub-02").mkdir()

        agg = aggregate_erp_results(tmp_path, subjects=["sub-01", "sub-02"])
        assert len(agg.df) == 1

    def test_erp_group_data_fields(self):
        assert ERPGroupData._fields == (
            "df",
            "all_evokeds",
            "all_diff_waves",
            "all_effect_sizes",
            "times_ms",
        )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  M. Default constants are exported
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•


def test_default_constants():
    """Verify DEFAULT_PIPE_* constants are well-formed."""
    assert isinstance(DEFAULT_PIPE_COLORS, dict)
    assert isinstance(DEFAULT_PIPE_LABELS, dict)
    assert isinstance(DEFAULT_PIPE_ORDER, list)
    assert set(DEFAULT_PIPE_ORDER) == set(DEFAULT_PIPE_COLORS.keys())
    assert isinstance(DEFAULT_ERP_WINDOWS, dict)
    for w_name, (t0, t1) in DEFAULT_ERP_WINDOWS.items():
        assert t0 < t1
