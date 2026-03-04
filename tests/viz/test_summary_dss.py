"""Tests for DSS-related summary helpers and plots."""

import matplotlib.pyplot as plt
import mne
import numpy as np

from mne_denoise.dss.denoisers import LineNoiseBias
from mne_denoise.viz import (
    plot_dss_mode_comparison,
    plot_dss_segmented_summary,
    plot_dss_summary,
)
from mne_denoise.viz._utils import (
    _get_bias_name,
    _get_dss_removed,
    _get_eigenvalues,
    _get_n_selected,
    _get_segment_results,
    _is_segmented,
)

# =====================================================================
# Mock estimator classes
# =====================================================================


class MockDSS:
    """Minimal mock for a fitted, non-segmented DSS estimator."""

    def __init__(self, n_ch=8, n_comp=4, n_selected=2, sfreq=250.0):
        self.eigenvalues_ = np.sort(np.random.rand(n_comp))[::-1]
        self.patterns_ = np.random.randn(n_ch, n_comp)
        self.n_selected_ = n_selected
        self.n_components = n_comp
        self.selection_method = "combined"
        self.sfreq = sfreq
        self.removed_ = np.random.randn(n_ch, 500) * 0.01
        self.sources_ = np.random.randn(n_comp, 500)
        self.bias = MockBias(sfreq=sfreq)
        self.smooth = None

    def transform(self, data):
        return data

    def get_params(self):
        return {}


class MockSegmentedDSS:
    """Mock for segmented DSS estimator."""

    def __init__(self, n_ch=8, n_segments=4, sfreq=250.0):
        self.sfreq = sfreq
        self.segmented = True
        self.n_components = 10
        self.selection_method = "combined"
        self.smooth = 5
        self.bias = MockBias(sfreq=sfreq)
        self.removed_ = np.random.randn(n_ch, 2000) * 0.01
        self.n_selected_ = 3

        seg_len = 500
        self.segment_results_ = []
        for i in range(n_segments):
            self.segment_results_.append(
                {
                    "n_selected": np.random.randint(1, 4),
                    "eigenvalues": np.sort(np.random.rand(10))[::-1],
                    "start": i * seg_len,
                    "end": (i + 1) * seg_len,
                }
            )


class MockBias:
    """Mock bias function."""

    def __init__(self, sfreq=250.0, freq=50.0):
        self.sfreq = sfreq
        self.freq = freq


class MockBiasCustom:
    """A custom bias for testing _get_bias_name fallback."""

    pass


class MockBiasNamed:
    """Mock bias with a recognized class-name for renaming."""

    pass


def _make_line_noise_raw(
    n_ch: int = 6, n_times: int = 1200, sfreq: float = 200.0, line_freq: float = 50.0
):
    """Create a small Raw object with synthetic line-noise contamination."""
    rng = np.random.default_rng(123)
    times = np.arange(n_times) / sfreq
    line = np.sin(2 * np.pi * line_freq * times)
    noise = 0.05 * rng.standard_normal((n_ch, n_times))
    mix = rng.uniform(0.5, 1.5, size=(n_ch, 1))
    data = noise + mix * line
    info = mne.create_info(
        [f"EEG{i}" for i in range(n_ch)], sfreq=sfreq, ch_types="eeg"
    )
    return mne.io.RawArray(data, info, verbose=False)


# =====================================================================
# Helper function tests
# =====================================================================


class TestGetNSelected:
    def test_from_n_selected(self):
        est = MockDSS(n_selected=3)
        assert _get_n_selected(est) == 3

    def test_from_n_removed(self):
        class E:
            n_removed_ = 2

        assert _get_n_selected(E()) == 2

    def test_fallback_zero(self):
        class E:
            pass

        assert _get_n_selected(E()) == 0


class TestGetEigenvalues:
    def test_with_eigenvalues(self):
        est = MockDSS()
        ev = _get_eigenvalues(est)
        assert ev is not None
        assert len(ev) == 4

    def test_without_eigenvalues(self):
        class E:
            pass

        assert _get_eigenvalues(E()) is None


class TestGetSegmentResults:
    def test_segmented(self):
        est = MockSegmentedDSS()
        sr = _get_segment_results(est)
        assert len(sr) == 4

    def test_non_segmented(self):
        est = MockDSS()
        sr = _get_segment_results(est)
        assert sr == []

    def test_adaptive_fallback(self):
        class E:
            adaptive_results_ = {"chunk_info": [{"n_removed": 1}]}

        sr = _get_segment_results(E())
        assert len(sr) == 1


class TestIsSegmented:
    def test_segmented_flag(self):
        est = MockSegmentedDSS()
        assert _is_segmented(est) is True

    def test_non_segmented(self):
        est = MockDSS()
        assert _is_segmented(est) is False


class TestGetRemoved:
    def test_with_removed(self):
        est = MockDSS()
        r = _get_dss_removed(est)
        assert r is not None

    def test_without(self):
        class E:
            pass

        assert _get_dss_removed(E()) is None

    def test_removed_attr(self):
        class E:
            removed = np.array([1, 2])

        assert _get_dss_removed(E()) is not None


class TestGetBiasName:
    def test_known_bias(self):
        class BandpassBias:
            pass

        class E:
            bias = BandpassBias()

        assert _get_bias_name(E()) == "Bandpass"

    def test_unknown_bias(self):
        class E:
            bias = MockBiasCustom()

        assert _get_bias_name(E()) == "MockBiasCustom"

    def test_no_bias(self):
        class E:
            pass

        assert _get_bias_name(E()) == "Unknown"


# =====================================================================
# Plot function tests
# =====================================================================


class TestPlotDssSummary:
    def test_basic_no_data(self):
        """Minimal call with just an estimator, no PSD data."""
        est = MockDSS()
        fig = plot_dss_summary(est, show=False)
        assert isinstance(fig, plt.Figure)

    def test_with_psd_data(self):
        """With data_before/after for PSD panel."""
        est = MockDSS(n_ch=8, sfreq=250.0)
        rng = np.random.default_rng(42)
        data_before = rng.standard_normal((8, 500))
        data_after = data_before * 0.9
        fig = plot_dss_summary(
            est,
            data_before=data_before,
            data_after=data_after,
            sfreq=250.0,
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_with_channel_names(self):
        est = MockDSS(n_ch=8)
        ch_names = [f"Ch{i}" for i in range(8)]
        fig = plot_dss_summary(est, channel_names=ch_names, show=False)
        assert isinstance(fig, plt.Figure)

    def test_custom_title(self):
        est = MockDSS()
        fig = plot_dss_summary(est, title="My Title", show=False)
        assert isinstance(fig, plt.Figure)

    def test_no_eigenvalues(self):
        """Estimator with no eigenvalues → placeholder text."""

        class E:
            pass

        est = E()
        est.sfreq = 250.0
        fig = plot_dss_summary(est, show=False)
        assert isinstance(fig, plt.Figure)

    def test_no_patterns(self):
        """Estimator with no patterns → placeholder text."""
        est = MockDSS()
        est.patterns_ = None
        fig = plot_dss_summary(est, show=False)
        assert isinstance(fig, plt.Figure)

    def test_no_removed(self):
        """Estimator with no removed data → placeholder text."""
        est = MockDSS()
        est.removed_ = None
        fig = plot_dss_summary(est, show=False)
        assert isinstance(fig, plt.Figure)

    def test_no_sources(self):
        """Estimator with no sources → placeholder text."""
        est = MockDSS()
        est.sources_ = None
        fig = plot_dss_summary(est, show=False)
        assert isinstance(fig, plt.Figure)

    def test_delegates_to_segmented(self):
        """Segmented estimator → delegates to plot_dss_segmented_summary."""
        est = MockSegmentedDSS()
        fig = plot_dss_summary(est, show=False)
        assert isinstance(fig, plt.Figure)

    def test_save(self, tmp_path):
        est = MockDSS()
        fpath = tmp_path / "dss_summary.png"
        plot_dss_summary(est, fname=str(fpath), show=False)
        assert fpath.exists()

    def test_with_data_computes_removed(self):
        """When removed is None but data_before/after given, compute it."""
        est = MockDSS(n_ch=4, sfreq=250.0)
        est.removed_ = None
        est.removed = None
        data_before = np.random.randn(4, 500)
        data_after = data_before * 0.8
        fig = plot_dss_summary(
            est,
            data_before=data_before,
            data_after=data_after,
            sfreq=250.0,
            show=False,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotDssSegmentedSummary:
    def test_basic(self):
        est = MockSegmentedDSS()
        fig = plot_dss_segmented_summary(est, show=False)
        assert isinstance(fig, plt.Figure)

    def test_with_psd(self):
        est = MockSegmentedDSS(n_ch=8, sfreq=250.0)
        data_before = np.random.randn(8, 2000)
        data_after = data_before * 0.9
        fig = plot_dss_segmented_summary(
            est,
            data_before=data_before,
            data_after=data_after,
            sfreq=250.0,
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_no_segment_results_fallback(self):
        """If no segment_results_, falls back to standard summary."""
        est = MockDSS()
        est.segmented = False
        fig = plot_dss_segmented_summary(est, show=False)
        assert isinstance(fig, plt.Figure)

    def test_with_channel_names(self):
        est = MockSegmentedDSS(n_ch=8)
        ch_names = [f"Ch{i}" for i in range(8)]
        fig = plot_dss_segmented_summary(
            est,
            channel_names=ch_names,
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_custom_title_and_figsize(self):
        est = MockSegmentedDSS()
        fig = plot_dss_segmented_summary(
            est,
            title="Custom",
            figsize=(12, 8),
            show=False,
        )
        assert isinstance(fig, plt.Figure)


class TestPlotDssModeComparison:
    def test_mode_comparison_basic(self):
        raw = _make_line_noise_raw()
        bias = LineNoiseBias(
            freq=50.0, sfreq=raw.info["sfreq"], method="fft", n_harmonics=3
        )

        fig, results = plot_dss_mode_comparison(
            bias,
            raw,
            n_components=6,
            n_select="auto",
            selection_method="combined",
            line_freq=50.0,
            n_harmonics=3,
            show=False,
        )

        assert isinstance(fig, plt.Figure)
        assert set(results) == {"plain", "smooth", "segmented"}
        for mode in ("plain", "smooth", "segmented"):
            assert "metrics" in results[mode]
            assert "data" in results[mode]
            assert "cleaned_raw" in results[mode]
            assert "estimator" in results[mode]

    def test_mode_comparison_save(self, tmp_path):
        raw = _make_line_noise_raw()
        bias = LineNoiseBias(
            freq=50.0, sfreq=raw.info["sfreq"], method="fft", n_harmonics=2
        )
        out = tmp_path / "dss_mode_comparison.png"

        fig, _ = plot_dss_mode_comparison(
            bias,
            raw,
            n_components=4,
            n_harmonics=2,
            show=False,
            fname=str(out),
        )

        assert isinstance(fig, plt.Figure)
        assert out.exists()
