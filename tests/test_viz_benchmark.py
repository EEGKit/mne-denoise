"""Tests for mne_denoise.viz.benchmark plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from mne_denoise.viz.benchmark import (
    DEFAULT_METHOD_COLORS,
    DEFAULT_METHOD_LABELS,
    _method_color,
    _method_label,
    plot_harmonic_attenuation,
    plot_metric_bars,
    plot_paired_metrics,
    plot_psd_gallery,
    plot_qc_psd,
    plot_r_comparison,
    plot_subject_psd_overlay,
    plot_tradeoff_and_r,
    plot_tradeoff_scatter,
)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


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
        rows.append({
            "subject": "sub-01",
            "method": method,
            "R_f0": np.random.uniform(0.7, 1.2),
            "peak_attenuation_db": np.random.uniform(5, 20),
            "below_noise_pct": np.random.uniform(-5, 5),
            "overclean_proportion": np.random.uniform(0, 0.3),
            "underclean_proportion": np.random.uniform(0, 0.3),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def multi_subject_df():
    """DataFrame for multiple subjects."""
    rng = np.random.default_rng(42)
    rows = []
    for sub in ["sub-01", "sub-02", "sub-03"]:
        for method in ["M0", "M1", "M2"]:
            rows.append({
                "subject": sub,
                "method": method,
                "R_f0": rng.uniform(0.7, 1.2),
                "peak_attenuation_db": rng.uniform(5, 20),
                "below_noise_pct": rng.uniform(-5, 5),
                "overclean_proportion": rng.uniform(0, 0.3),
                "underclean_proportion": rng.uniform(0, 0.3),
            })
    return pd.DataFrame(rows)


# =====================================================================
# Helper tests
# =====================================================================

class TestMethodHelpers:
    def test_method_color_known(self):
        for m in DEFAULT_METHOD_COLORS:
            assert _method_color(m) == DEFAULT_METHOD_COLORS[m]

    def test_method_color_unknown(self):
        c = _method_color("UNKNOWN_XYZ")
        assert isinstance(c, str)

    def test_method_color_custom(self):
        custom = {"M1": "#ff0000"}
        assert _method_color("M1", custom) == "#ff0000"

    def test_method_label_known(self):
        for m in DEFAULT_METHOD_LABELS:
            assert _method_label(m) == DEFAULT_METHOD_LABELS[m]

    def test_method_label_unknown(self):
        assert _method_label("MY_METHOD") == "MY_METHOD"

    def test_method_label_custom(self):
        custom = {"M1": "Custom Label"}
        assert _method_label("M1", custom) == "Custom Label"


# =====================================================================
# Plot function tests
# =====================================================================

class TestPlotQcPsd:
    def test_basic(self, freqs, gm_psd, gm_psd_clean):
        fig = plot_qc_psd(freqs, gm_psd, freqs, gm_psd_clean, show=False)
        assert isinstance(fig, plt.Figure)

    def test_with_method_tag(self, freqs, gm_psd, gm_psd_clean):
        fig = plot_qc_psd(
            freqs, gm_psd, freqs, gm_psd_clean,
            method_tag="M1", subject="sub-01", show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_with_metrics_dict(self, freqs, gm_psd, gm_psd_clean):
        metrics = {
            "harmonics_hz": [50, 100, 150, 200],
            "attenuation_per_harmonic_db": [10.0, 8.0, 6.0],
            "R_per_harmonic": [0.8, 0.9, 0.95],
        }
        fig = plot_qc_psd(
            freqs, gm_psd, freqs, gm_psd_clean,
            metrics_dict=metrics, show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_custom_harmonics(self, freqs, gm_psd, gm_psd_clean):
        fig = plot_qc_psd(
            freqs, gm_psd, freqs, gm_psd_clean,
            harmonics_hz=[60.0, 120.0], show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_save(self, freqs, gm_psd, gm_psd_clean, tmp_path):
        fpath = tmp_path / "qc.png"
        plot_qc_psd(
            freqs, gm_psd, freqs, gm_psd_clean,
            fname=str(fpath), show=False,
        )
        assert fpath.exists()


class TestPlotPsdGallery:
    def test_basic(self, freqs, gm_psd, cleaned_psds):
        fig = plot_psd_gallery(freqs, gm_psd, cleaned_psds, show=False)
        assert isinstance(fig, plt.Figure)

    def test_with_order(self, freqs, gm_psd, cleaned_psds):
        fig = plot_psd_gallery(
            freqs, gm_psd, cleaned_psds,
            method_order=["M2", "M1"], subject="sub-01", show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_missing_method(self, freqs, gm_psd, cleaned_psds):
        """Method in order but not in cleaned_psds → 'no data' panels."""
        fig = plot_psd_gallery(
            freqs, gm_psd, cleaned_psds,
            method_order=["M1", "MISSING"], show=False,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotSubjectPsdOverlay:
    def test_basic(self, freqs, gm_psd, cleaned_psds):
        fig = plot_subject_psd_overlay(
            freqs, gm_psd, cleaned_psds, show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_with_params(self, freqs, gm_psd, cleaned_psds):
        fig = plot_subject_psd_overlay(
            freqs, gm_psd, cleaned_psds,
            line_freq=50.0, n_harmonics=2, subject="sub-01", show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_skip_m0(self, freqs, gm_psd):
        """M0 should be skipped in overlay."""
        cleaned = {
            "M0": (freqs, np.ones_like(freqs) * 1e-6),
            "M1": (freqs, np.ones_like(freqs) * 1e-6),
        }
        fig = plot_subject_psd_overlay(
            freqs, np.ones_like(freqs) * 1e-5, cleaned, show=False,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotMetricBars:
    def test_basic(self, single_subject_df):
        fig = plot_metric_bars(single_subject_df, show=False)
        assert isinstance(fig, plt.Figure)

    def test_custom_order(self, single_subject_df):
        fig = plot_metric_bars(
            single_subject_df,
            method_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotTradeoffScatter:
    def test_single_subject(self, single_subject_df):
        fig = plot_tradeoff_scatter(single_subject_df, show=False)
        assert isinstance(fig, plt.Figure)

    def test_multi_subject(self, multi_subject_df):
        fig = plot_tradeoff_scatter(
            multi_subject_df,
            method_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotRComparison:
    def test_single_subject_bar(self, single_subject_df):
        fig = plot_r_comparison(
            single_subject_df,
            method_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_multi_subject_paired(self, multi_subject_df):
        fig = plot_r_comparison(
            multi_subject_df,
            method_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotHarmonicAttenuation:
    def test_basic(self, freqs, gm_psd, cleaned_psds):
        fig = plot_harmonic_attenuation(
            freqs, gm_psd, cleaned_psds,
            harmonics_hz=[50.0, 100.0], show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_with_subject(self, freqs, gm_psd, cleaned_psds):
        fig = plot_harmonic_attenuation(
            freqs, gm_psd, cleaned_psds,
            harmonics_hz=[50.0], subject="sub-01", show=False,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotPairedMetrics:
    def test_multi_subject(self, multi_subject_df):
        fig = plot_paired_metrics(
            multi_subject_df,
            method_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_single_subject_returns_none(self, single_subject_df):
        result = plot_paired_metrics(single_subject_df, show=False)
        assert result is None


class TestPlotTradeoffAndR:
    def test_single_subject(self, single_subject_df):
        fig = plot_tradeoff_and_r(
            single_subject_df,
            method_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_multi_subject(self, multi_subject_df):
        fig = plot_tradeoff_and_r(
            multi_subject_df,
            method_order=["M0", "M1", "M2"],
            show=False,
        )
        assert isinstance(fig, plt.Figure)
