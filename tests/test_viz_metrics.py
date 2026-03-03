"""Tests for mne_denoise.viz._metrics module."""

import numpy as np

from mne_denoise.viz._metrics import (
    _spectral_distortion,
    _suppression_ratio,
    _variance_removed,
)


class TestSuppressionRatio:
    """Tests for _suppression_ratio."""

    def test_basic(self):
        """Halving power at target → ~3 dB."""
        f = np.arange(0, 100, 0.5)
        psd_before = np.ones((3, len(f))) * 0.01
        psd_after = np.ones((3, len(f))) * 0.01
        mask = (f >= 49) & (f <= 51)
        psd_before[:, mask] = 1.0
        psd_after[:, mask] = 0.5
        sr = _suppression_ratio(f, psd_before, psd_after, 50.0, bw=2.0)
        expected = 10 * np.log10(1.0 / 0.5)
        np.testing.assert_allclose(sr, expected, atol=0.5)

    def test_full_suppression(self):
        """Complete suppression → inf."""
        f = np.arange(0, 100, 0.5)
        psd_before = np.ones((2, len(f)))
        psd_after = np.zeros((2, len(f)))
        sr = _suppression_ratio(f, psd_before, psd_after, 50.0)
        assert sr == np.inf

    def test_no_change(self):
        """Same PSD → 0 dB."""
        f = np.arange(0, 100, 0.5)
        psd = np.ones((2, len(f)))
        sr = _suppression_ratio(f, psd, psd, 50.0)
        np.testing.assert_allclose(sr, 0.0, atol=1e-10)


class TestSpectralDistortion:
    """Tests for _spectral_distortion."""

    def test_no_distortion(self):
        """Identical PSDs outside harmonics → 0 distortion."""
        f = np.arange(0, 200, 0.5)
        psd = np.ones((2, len(f)))
        sd = _spectral_distortion(f, psd, psd, line_freq=50.0, n_harm=3, bw=2.0)
        np.testing.assert_allclose(sd, 0.0, atol=1e-10)

    def test_with_distortion(self):
        """Modified spectrum outside harmonics → positive distortion."""
        f = np.arange(0, 200, 0.5)
        psd_before = np.ones((2, len(f)))
        psd_after = np.ones((2, len(f))) * 2.0  # doubled everywhere
        sd = _spectral_distortion(f, psd_before, psd_after, line_freq=50.0)
        assert sd > 0

    def test_returns_float(self):
        """Result should be a scalar float."""
        f = np.arange(0, 200, 0.5)
        psd = np.ones((2, len(f)))
        result = _spectral_distortion(f, psd, psd)
        assert isinstance(result, (float, np.floating))


class TestVarianceRemoved:
    """Tests for _variance_removed."""

    def test_half_variance(self):
        """Removing half variance → 50%."""
        rng = np.random.default_rng(42)
        data_before = rng.standard_normal((4, 1000))
        # data_after has half the amplitude → quarter variance
        data_after = data_before * 0.5
        pct = _variance_removed(data_before, data_after)
        # var(0.5*x) = 0.25*var(x), so removed = 1 - 0.25 = 0.75 → 75%
        np.testing.assert_allclose(pct, 75.0, atol=1.0)

    def test_no_removal(self):
        """Same data → 0% removed."""
        rng = np.random.RandomState(0)
        data = rng.randn(3, 100)
        pct = _variance_removed(data, data)
        np.testing.assert_allclose(pct, 0.0, atol=1e-10)

    def test_full_removal(self):
        """Zero output → 100% removed."""
        data = np.random.randn(3, 100)
        pct = _variance_removed(data, np.zeros_like(data))
        np.testing.assert_allclose(pct, 100.0, atol=1e-10)
