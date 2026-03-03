"""Tests for advanced ZapLine visualization (adaptive summary, standard summary, helpers)."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from mne_denoise.viz.zapline import (
    _get_chunk_info,
    _get_cleaned,
    _get_n_removed,
    _get_removed,
    _is_adaptive,
    plot_adaptive_summary,
    plot_zapline_analytics,
    plot_zapline_summary,
)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


# =====================================================================
# Mock helpers
# =====================================================================

def _make_chunk_info(n_chunks=4, sfreq=500, n_times=5000, line_freq=50.0):
    """Create synthetic chunk_info list for adaptive mode."""
    chunk_len = n_times // n_chunks
    info = []
    rng = np.random.RandomState(42)
    for i in range(n_chunks):
        info.append({
            "n_removed": rng.randint(1, 4),
            "fine_freq": line_freq + rng.uniform(-0.3, 0.3),
            "frequency": line_freq,
            "artifact_present": rng.random() > 0.2,
            "start": i * chunk_len,
            "end": (i + 1) * chunk_len,
        })
    return info


class MockStandardZapLine:
    """Mock standard (non-adaptive) ZapLine result."""

    def __init__(self, n_channels=8, n_times=2500, sfreq=500.0, line_freq=50.0):
        rng = np.random.RandomState(0)
        self.sfreq = sfreq
        self.line_freq = line_freq
        self.n_harmonics = 1
        self.n_harmonics_ = 1
        self.n_remove = 2
        self.threshold = 1.0
        self.nkeep = 6
        self.rank = 8
        self.n_removed_ = 2
        self.eigenvalues_ = np.sort(rng.rand(8))[::-1]
        self.patterns_ = rng.randn(n_channels, 4)
        self.removed = rng.randn(n_channels, n_times) * 0.01
        self.cleaned = rng.randn(n_channels, n_times)
        self.sources_ = rng.randn(4, n_times) * 0.1
        # Not adaptive
        self.adaptive = False


class MockAdaptiveZapLine:
    """Mock adaptive ZapLine result."""

    def __init__(self, n_channels=8, n_times=5000, sfreq=500.0, line_freq=50.0):
        rng = np.random.RandomState(1)
        self.sfreq = sfreq
        self.line_freq = line_freq
        self.n_harmonics = 2
        self.n_harmonics_ = 2
        self.n_remove = 2
        self.threshold = 1.0
        self.nkeep = 6
        self.rank = 8
        self.n_removed_ = 6  # total across chunks
        self.adaptive = True
        self.eigenvalues_ = np.sort(rng.rand(8))[::-1]
        self.patterns_ = rng.randn(n_channels, 4)
        self.removed = rng.randn(n_channels, n_times) * 0.01
        self.cleaned = rng.randn(n_channels, n_times)
        self.sources_ = rng.randn(4, n_times) * 0.1
        self.adaptive_results_ = {
            "chunk_info": _make_chunk_info(4, sfreq, n_times, line_freq),
            "removed": self.removed,
            "cleaned": self.cleaned,
        }


# =====================================================================
# Helper function tests
# =====================================================================


class TestGetNRemoved:
    """Tests for _get_n_removed."""

    def test_n_removed_attr(self):
        obj = MockStandardZapLine()
        assert _get_n_removed(obj) == 2

    def test_n_removed_fallback(self):
        class R:
            n_removed_ = None
            n_removed = 5
        assert _get_n_removed(R()) == 5

    def test_dict_input(self):
        assert _get_n_removed({"n_removed": 3}) == 3

    def test_missing(self):
        class R:
            pass
        assert _get_n_removed(R()) == 0

    def test_none_values(self):
        class R:
            n_removed_ = None
            n_removed = None
        assert _get_n_removed(R()) == 0


class TestGetRemoved:
    """Tests for _get_removed."""

    def test_direct_attr(self):
        obj = MockStandardZapLine()
        result = _get_removed(obj)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_adaptive_fallback(self):
        obj = MockAdaptiveZapLine()
        # Remove direct attr to force adaptive path
        obj.removed = None
        result = _get_removed(obj)
        assert result is not None

    def test_dict_input(self):
        data = np.ones((3, 10))
        result = _get_removed({"removed": data})
        assert result is data

    def test_missing(self):
        class R:
            pass
        assert _get_removed(R()) is None


class TestGetCleaned:
    """Tests for _get_cleaned."""

    def test_direct_attr(self):
        obj = MockStandardZapLine()
        result = _get_cleaned(obj)
        assert result is not None

    def test_adaptive_fallback(self):
        obj = MockAdaptiveZapLine()
        obj.cleaned = None
        result = _get_cleaned(obj)
        assert result is not None

    def test_dict_input(self):
        data = np.ones((3, 10))
        result = _get_cleaned({"cleaned": data})
        assert result is data

    def test_missing(self):
        class R:
            pass
        assert _get_cleaned(R()) is None


class TestIsAdaptive:
    """Tests for _is_adaptive."""

    def test_standard(self):
        obj = MockStandardZapLine()
        assert not _is_adaptive(obj)

    def test_adaptive_flag(self):
        obj = MockAdaptiveZapLine()
        assert _is_adaptive(obj)

    def test_adaptive_results_attr(self):
        class R:
            adaptive = False
            adaptive_results_ = {"chunk_info": []}
        assert _is_adaptive(R())

    def test_dict_with_chunk_info(self):
        assert _is_adaptive({"chunk_info": [{"n_removed": 1}]})

    def test_dict_without_chunk_info(self):
        assert not _is_adaptive({"other": 1})

    def test_plain_object(self):
        class R:
            pass
        assert not _is_adaptive(R())


class TestGetChunkInfo:
    """Tests for _get_chunk_info."""

    def test_adaptive(self):
        obj = MockAdaptiveZapLine()
        chunks = _get_chunk_info(obj)
        assert len(chunks) == 4

    def test_dict(self):
        chunks = _get_chunk_info({"chunk_info": [{"n_removed": 1}]})
        assert len(chunks) == 1

    def test_missing(self):
        class R:
            pass
        assert _get_chunk_info(R()) == []

    def test_none_adaptive_results(self):
        class R:
            adaptive_results_ = None
        assert _get_chunk_info(R()) == []


# =====================================================================
# plot_zapline_analytics tests
# =====================================================================


class TestPlotZaplineAnalytics:
    """Tests for the legacy analytics dashboard."""

    def test_basic_standard(self):
        obj = MockStandardZapLine()
        fig = plot_zapline_analytics(obj, sfreq=500, show=False)
        assert isinstance(fig, plt.Figure)

    def test_adaptive_mode(self):
        obj = MockAdaptiveZapLine()
        fig = plot_zapline_analytics(obj, sfreq=500, show=False)
        assert isinstance(fig, plt.Figure)

    def test_missing_eigenvalues(self):
        obj = MockStandardZapLine()
        obj.eigenvalues_ = np.array([])
        fig = plot_zapline_analytics(obj, sfreq=500, show=False)
        assert isinstance(fig, plt.Figure)

    def test_no_removed(self):
        obj = MockStandardZapLine()
        obj.removed = None
        fig = plot_zapline_analytics(obj, sfreq=500, show=False)
        assert isinstance(fig, plt.Figure)


# =====================================================================
# plot_adaptive_summary tests
# =====================================================================


class TestPlotAdaptiveSummary:
    """Tests for the adaptive ZapLine dashboard."""

    def test_basic(self):
        obj = MockAdaptiveZapLine()
        fig = plot_adaptive_summary(obj, show=False)
        assert isinstance(fig, plt.Figure)

    def test_with_psd_data(self):
        obj = MockAdaptiveZapLine()
        rng = np.random.RandomState(10)
        n_ch, n_t = 8, 5000
        before = rng.randn(n_ch, n_t)
        after = rng.randn(n_ch, n_t) * 0.9
        fig = plot_adaptive_summary(
            obj, data_before=before, data_after=after, sfreq=500, show=False
        )
        assert isinstance(fig, plt.Figure)

    def test_no_chunk_info_fallback(self):
        """When chunk_info is empty, should fall back to plot_zapline_analytics."""
        obj = MockAdaptiveZapLine()
        obj.adaptive_results_ = {"chunk_info": []}
        fig = plot_adaptive_summary(obj, show=False)
        assert isinstance(fig, plt.Figure)

    def test_custom_title(self):
        obj = MockAdaptiveZapLine()
        fig = plot_adaptive_summary(obj, title="Custom Title", show=False)
        assert isinstance(fig, plt.Figure)

    def test_with_channel_names(self):
        obj = MockAdaptiveZapLine()
        names = [f"CH{i}" for i in range(8)]
        fig = plot_adaptive_summary(obj, channel_names=names, show=False)
        assert isinstance(fig, plt.Figure)

    def test_no_sfreq_fallback(self):
        """When sfreq is None, segment timeline shows placeholder."""
        obj = MockAdaptiveZapLine()
        obj.sfreq = None
        fig = plot_adaptive_summary(obj, sfreq=None, show=False)
        assert isinstance(fig, plt.Figure)

    def test_no_removed_data(self):
        obj = MockAdaptiveZapLine()
        obj.removed = None
        obj.adaptive_results_["removed"] = None
        fig = plot_adaptive_summary(obj, show=False)
        assert isinstance(fig, plt.Figure)

    def test_save_to_file(self, tmp_path):
        obj = MockAdaptiveZapLine()
        fpath = tmp_path / "adaptive.png"
        fig = plot_adaptive_summary(obj, show=False, fname=str(fpath))
        assert isinstance(fig, plt.Figure)
        assert fpath.exists()

    def test_eigenvalues_fallback_no_psd(self):
        """When no PSD data and eigenvalues exist, panel (e) shows eigenvalues."""
        obj = MockAdaptiveZapLine()
        fig = plot_adaptive_summary(obj, show=False)
        assert isinstance(fig, plt.Figure)


# =====================================================================
# plot_zapline_summary tests
# =====================================================================


class TestPlotZaplineSummary:
    """Tests for the standard ZapLine summary dashboard."""

    def test_basic(self):
        obj = MockStandardZapLine()
        fig = plot_zapline_summary(obj, show=False)
        assert isinstance(fig, plt.Figure)

    def test_with_psd_data(self):
        obj = MockStandardZapLine()
        rng = np.random.RandomState(10)
        n_ch, n_t = 8, 2500
        before = rng.randn(n_ch, n_t)
        after = rng.randn(n_ch, n_t) * 0.9
        fig = plot_zapline_summary(
            obj, data_before=before, data_after=after, sfreq=500, show=False
        )
        assert isinstance(fig, plt.Figure)

    def test_delegates_to_adaptive(self):
        """When result is adaptive, delegates to plot_adaptive_summary."""
        obj = MockAdaptiveZapLine()
        fig = plot_zapline_summary(obj, show=False)
        assert isinstance(fig, plt.Figure)

    def test_with_channel_names(self):
        obj = MockStandardZapLine()
        names = [f"EEG{i}" for i in range(8)]
        fig = plot_zapline_summary(obj, channel_names=names, show=False)
        assert isinstance(fig, plt.Figure)

    def test_no_eigenvalues(self):
        obj = MockStandardZapLine()
        obj.eigenvalues_ = None
        fig = plot_zapline_summary(obj, show=False)
        assert isinstance(fig, plt.Figure)

    def test_no_patterns(self):
        obj = MockStandardZapLine()
        obj.patterns_ = None
        fig = plot_zapline_summary(obj, show=False)
        assert isinstance(fig, plt.Figure)

    def test_empty_eigenvalues(self):
        obj = MockStandardZapLine()
        obj.eigenvalues_ = np.array([])
        fig = plot_zapline_summary(obj, show=False)
        assert isinstance(fig, plt.Figure)

    def test_no_sources(self):
        obj = MockStandardZapLine()
        obj.sources_ = None
        fig = plot_zapline_summary(obj, show=False)
        assert isinstance(fig, plt.Figure)

    def test_sources_no_sfreq(self):
        """Sources exist but no sfreq — shows placeholder."""
        obj = MockStandardZapLine()
        obj.sfreq = None
        fig = plot_zapline_summary(obj, sfreq=None, show=False)
        assert isinstance(fig, plt.Figure)

    def test_no_removed(self):
        obj = MockStandardZapLine()
        obj.removed = None
        fig = plot_zapline_summary(obj, show=False)
        assert isinstance(fig, plt.Figure)

    def test_custom_title_and_figsize(self):
        obj = MockStandardZapLine()
        fig = plot_zapline_summary(
            obj, title="My ZapLine", figsize=(14, 10), show=False
        )
        assert isinstance(fig, plt.Figure)

    def test_max_components(self):
        obj = MockStandardZapLine()
        fig = plot_zapline_summary(obj, max_components=2, show=False)
        assert isinstance(fig, plt.Figure)

    def test_save_to_file(self, tmp_path):
        obj = MockStandardZapLine()
        fpath = tmp_path / "zapline_summary.png"
        fig = plot_zapline_summary(obj, show=False, fname=str(fpath))
        assert isinstance(fig, plt.Figure)
        assert fpath.exists()

    def test_with_psd_and_line_freq_harmonics(self):
        """PSD panel with line_freq triggers harmonic vertical lines."""
        obj = MockStandardZapLine()
        obj.n_harmonics_ = 3
        rng = np.random.RandomState(10)
        n_ch, n_t = 8, 2500
        before = rng.randn(n_ch, n_t)
        after = rng.randn(n_ch, n_t) * 0.9
        fig = plot_zapline_summary(
            obj, data_before=before, data_after=after, sfreq=500, show=False
        )
        assert isinstance(fig, plt.Figure)

    def test_many_channels_no_channel_names(self):
        """Test stem plot x-axis ticks with >20 channels, no names."""
        obj = MockStandardZapLine(n_channels=30)
        obj.patterns_ = np.random.RandomState(0).randn(30, 4)
        obj.removed = np.random.RandomState(0).randn(30, 2500) * 0.01
        fig = plot_zapline_summary(obj, show=False)
        assert isinstance(fig, plt.Figure)
