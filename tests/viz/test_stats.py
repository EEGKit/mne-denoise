"""Tests for grouped metric and statistical visualization helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from mne_denoise.viz import (
    DEFAULT_METHOD_COLORS,
    DEFAULT_METHOD_LABELS,
    DEFAULT_METHOD_ORDER,
    plot_harmonic_attenuation,
    plot_metric_bars,
    plot_metric_slopes,
    plot_metric_tradeoff_summary,
    plot_single_metric_comparison,
    plot_tradeoff_scatter,
)

# =====================================================================
# Synthetic data fixtures
# =====================================================================


@pytest.fixture
def freqs():
    return np.arange(0, 200, 0.5)


@pytest.fixture
def gm_psd(freqs):
    """Synthetic geometric-mean PSD with a peak at 50 Hz."""
    psd = np.ones_like(freqs) * 1e-6
    for h in [50, 100, 150]:
        mask = (freqs >= h - 1) & (freqs <= h + 1)
        psd[mask] = 1e-3
    return psd


@pytest.fixture
def gm_psd_clean(freqs):
    """Cleaned PSD with reduced peaks."""
    psd = np.ones_like(freqs) * 1e-6
    for h in [50, 100, 150]:
        mask = (freqs >= h - 1) & (freqs <= h + 1)
        psd[mask] = 1e-5
    return psd


@pytest.fixture
def cleaned_psds(freqs, gm_psd_clean):
    """Dict of cleaned PSDs for multiple methods."""
    return {
        "M1": (freqs, gm_psd_clean),
        "M2": (freqs, gm_psd_clean * 1.1),
    }


@pytest.fixture
def single_subject_df():
    """DataFrame for a single subject with method metrics."""
    rows = []
    for method in ["M0", "M1", "M2"]:
        rows.append(
            {
                "subject": "sub-01",
                "method": method,
                "R_f0": np.random.uniform(0.7, 1.2),
                "peak_attenuation_db": np.random.uniform(5, 20),
                "below_noise_pct": np.random.uniform(-5, 5),
                "overclean_proportion": np.random.uniform(0, 0.3),
                "underclean_proportion": np.random.uniform(0, 0.3),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def multi_subject_df():
    """DataFrame for multiple subjects."""
    rng = np.random.default_rng(42)
    rows = []
    for sub in ["sub-01", "sub-02", "sub-03"]:
        for method in ["M0", "M1", "M2"]:
            rows.append(
                {
                    "subject": sub,
                    "method": method,
                    "R_f0": rng.uniform(0.7, 1.2),
                    "peak_attenuation_db": rng.uniform(5, 20),
                    "below_noise_pct": rng.uniform(-5, 5),
                    "overclean_proportion": rng.uniform(0, 0.3),
                    "underclean_proportion": rng.uniform(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


# =====================================================================
# Helper tests
# =====================================================================


class TestMethodDefaults:
    def test_method_defaults_shape(self):
        assert isinstance(DEFAULT_METHOD_LABELS, dict)
        assert isinstance(DEFAULT_METHOD_ORDER, list)
        assert set(DEFAULT_METHOD_ORDER).issubset(DEFAULT_METHOD_COLORS.keys())


class TestPlotMetricBars:
    def test_basic(self, single_subject_df):
        fig = plot_metric_bars(single_subject_df, group_col="method", show=False)
        assert isinstance(fig, plt.Figure)

    def test_custom_order(self, single_subject_df):
        fig = plot_metric_bars(
            single_subject_df,
            group_col="method",
            group_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotTradeoffScatter:
    def test_single_subject(self, single_subject_df):
        fig = plot_tradeoff_scatter(single_subject_df, group_col="method", show=False)
        assert isinstance(fig, plt.Figure)

    def test_multi_subject(self, multi_subject_df):
        fig = plot_tradeoff_scatter(
            multi_subject_df,
            group_col="method",
            group_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotSingleMetricComparison:
    def test_single_subject_bar(self, single_subject_df):
        fig = plot_single_metric_comparison(
            single_subject_df,
            group_col="method",
            group_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_multi_subject_paired(self, multi_subject_df):
        fig = plot_single_metric_comparison(
            multi_subject_df,
            group_col="method",
            group_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotHarmonicAttenuation:
    def test_basic(self, freqs, gm_psd, cleaned_psds):
        fig = plot_harmonic_attenuation(
            freqs,
            gm_psd,
            cleaned_psds,
            harmonics_hz=[50.0, 100.0],
            series_order=["M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_with_subject(self, freqs, gm_psd, cleaned_psds):
        fig = plot_harmonic_attenuation(
            freqs,
            gm_psd,
            cleaned_psds,
            harmonics_hz=[50.0],
            subject="sub-01",
            series_order=["M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotMetricSlopes:
    def test_multi_subject(self, multi_subject_df):
        fig = plot_metric_slopes(
            multi_subject_df,
            group_col="method",
            group_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_single_subject_returns_none(self, single_subject_df):
        result = plot_metric_slopes(single_subject_df, group_col="method", show=False)
        assert result is None


class TestPlotMetricTradeoffSummary:
    def test_single_subject(self, single_subject_df):
        fig = plot_metric_tradeoff_summary(
            single_subject_df,
            method_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_multi_subject(self, multi_subject_df):
        fig = plot_metric_tradeoff_summary(
            multi_subject_df,
            method_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)
