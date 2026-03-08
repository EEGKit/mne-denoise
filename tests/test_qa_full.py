"""Tests for the full mne_denoise.qa metrics module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from mne_denoise.viz import (
    below_noise_distortion_db,
    compute_all_qa_metrics,
    geometric_mean_psd,
    geometric_mean_psd_ratio,
    noise_surround_ratio,
    overclean_proportion,
    underclean_proportion,
)

# ── Shared helpers ───────────────────────────────────────────────────


def _freqs(fmax=100.0, df=0.5):
    """Return a simple frequency vector."""
    return np.arange(0, fmax, df)


def _flat_psd(freqs, n_channels=None, value=1.0):
    """Flat PSD at *value* µV²/Hz."""
    if n_channels is None:
        return np.full_like(freqs, value)
    return np.full((n_channels, len(freqs)), value)


def _add_peak(psd, freqs, center, bw, height):
    """Add a Gaussian-ish peak to *psd* (in-place)."""
    mask = (freqs >= center - bw) & (freqs <= center + bw)
    if psd.ndim == 1:
        psd[mask] += height
    else:
        psd[:, mask] += height
    return psd


# ── noise_surround_ratio ────────────────────────────────────────────


class TestNoiseSurroundRatio:
    """Tests for noise_surround_ratio."""

    def test_flat_spectrum_returns_one(self):
        """Flat spectrum → ratio ≈ 1."""
        freqs = _freqs()
        psd = _flat_psd(freqs, n_channels=4)
        result = noise_surround_ratio(freqs, psd, 50.0)
        np.testing.assert_allclose(result, 1.0, atol=0.01)

    def test_residual_peak_gt_one(self):
        """A residual peak at target → ratio > 1."""
        freqs = _freqs()
        psd = _flat_psd(freqs, n_channels=4)
        _add_peak(psd, freqs, 50.0, 2.0, 10.0)
        result = noise_surround_ratio(freqs, psd, 50.0)
        assert np.all(result > 1.0)

    def test_1d_input(self):
        """1-D PSD should return a scalar."""
        freqs = _freqs()
        psd = _flat_psd(freqs)
        result = noise_surround_ratio(freqs, psd, 50.0)
        assert np.ndim(result) == 0

    def test_custom_bandwidths(self):
        """Custom peak_bw / surround_bw should work without error."""
        freqs = _freqs()
        psd = _flat_psd(freqs, n_channels=2)
        result = noise_surround_ratio(freqs, psd, 50.0, peak_bw=1.0, surround_bw=3.0)
        np.testing.assert_allclose(result, 1.0, atol=0.01)


# ── below_noise_distortion_db ───────────────────────────────────────


class TestBelowNoiseDistortionDb:
    """Tests for below_noise_distortion_db."""

    def test_identical_psd_zero_distortion(self):
        """Same PSD → 0 dB distortion."""
        freqs = _freqs()
        psd = _flat_psd(freqs, n_channels=3)
        result = below_noise_distortion_db(freqs, psd, psd)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_distortion_increases_with_change(self):
        """Larger broadband change → larger distortion."""
        freqs = _freqs()
        psd_before = _flat_psd(freqs, n_channels=3, value=1.0)
        psd_after_small = _flat_psd(freqs, n_channels=3, value=0.9)
        psd_after_large = _flat_psd(freqs, n_channels=3, value=0.1)
        d_small = below_noise_distortion_db(freqs, psd_before, psd_after_small)
        d_large = below_noise_distortion_db(freqs, psd_before, psd_after_large)
        assert np.all(d_large > d_small)

    def test_exclude_freq_reduces_distortion(self):
        """Excluding the noisy band should lower measured distortion."""
        freqs = _freqs()
        psd_before = _flat_psd(freqs, n_channels=2)
        psd_after = psd_before.copy()
        # Only change at 50 Hz band
        _add_peak(psd_after, freqs, 50.0, 5.0, -0.9)
        d_incl = below_noise_distortion_db(freqs, psd_before, psd_after)
        d_excl = below_noise_distortion_db(
            freqs, psd_before, psd_after, exclude_freq=50.0
        )
        assert np.all(d_excl < d_incl)

    def test_1d_input(self):
        """1-D PSD should return a scalar."""
        freqs = _freqs()
        psd = _flat_psd(freqs)
        result = below_noise_distortion_db(freqs, psd, psd)
        assert np.ndim(result) == 0

    def test_n_harmonics_excludes_more(self):
        """More harmonics excluded → less measured distortion."""
        freqs = _freqs(fmax=200.0)
        psd_before = _flat_psd(freqs, n_channels=2)
        psd_after = psd_before.copy()
        # Add changes at 50, 100, 150 Hz
        for hf in [50.0, 100.0, 150.0]:
            _add_peak(psd_after, freqs, hf, 5.0, -0.8)
        d0 = below_noise_distortion_db(
            freqs,
            psd_before,
            psd_after,
            exclude_freq=50.0,
            n_harmonics=0,
            fmax=180.0,
        )
        d2 = below_noise_distortion_db(
            freqs,
            psd_before,
            psd_after,
            exclude_freq=50.0,
            n_harmonics=2,
            fmax=180.0,
        )
        assert np.all(d2 <= d0)


# ── overclean_proportion ────────────────────────────────────────────


class TestOvercleanProportion:
    """Tests for overclean_proportion."""

    def test_no_overcleaning(self):
        """Identical spectral floor → proportion = 0."""
        freqs = _freqs()
        psd = _flat_psd(freqs, n_channels=5)
        result = overclean_proportion(freqs, psd, psd, 50.0)
        assert result == pytest.approx(0.0)

    def test_all_overcleaned(self):
        """Surround heavily suppressed → proportion = 1."""
        freqs = _freqs()
        psd_before = _flat_psd(freqs, n_channels=4, value=10.0)
        psd_after = _flat_psd(freqs, n_channels=4, value=0.01)
        result = overclean_proportion(freqs, psd_before, psd_after, 50.0)
        assert result == pytest.approx(1.0)

    def test_1d_input(self):
        """1-D PSD should return a scalar (0 or 1)."""
        freqs = _freqs()
        psd = _flat_psd(freqs)
        result = overclean_proportion(freqs, psd, psd, 50.0)
        assert result == pytest.approx(0.0)

    def test_threshold_sensitivity(self):
        """Lower threshold → more channels flagged."""
        freqs = _freqs()
        psd_before = _flat_psd(freqs, n_channels=10, value=2.0)
        psd_after = _flat_psd(freqs, n_channels=10, value=1.0)
        p_strict = overclean_proportion(
            freqs, psd_before, psd_after, 50.0, threshold_db=10.0
        )
        p_loose = overclean_proportion(
            freqs, psd_before, psd_after, 50.0, threshold_db=0.1
        )
        assert p_loose >= p_strict


# ── underclean_proportion ───────────────────────────────────────────


class TestUndercleanProportion:
    """Tests for underclean_proportion."""

    def test_fully_cleaned(self):
        """Flat spectrum → no channels under-cleaned."""
        freqs = _freqs()
        psd = _flat_psd(freqs, n_channels=5)
        result = underclean_proportion(freqs, psd, 50.0)
        assert result == pytest.approx(0.0)

    def test_residual_peak(self):
        """Prominent residual peak → proportion > 0."""
        freqs = _freqs()
        psd = _flat_psd(freqs, n_channels=5)
        _add_peak(psd, freqs, 50.0, 2.0, 50.0)
        result = underclean_proportion(freqs, psd, 50.0)
        assert result > 0.0

    def test_1d_input(self):
        """1-D PSD should return a scalar."""
        freqs = _freqs()
        psd = _flat_psd(freqs)
        result = underclean_proportion(freqs, psd, 50.0)
        assert np.ndim(result) == 0


# ── geometric_mean_psd_ratio ────────────────────────────────────────


class TestGeometricMeanPsdRatio:
    """Tests for geometric_mean_psd_ratio."""

    def test_identical_psd(self):
        """Same PSD → ratio = 1."""
        freqs = _freqs()
        psd = _flat_psd(freqs, n_channels=3)
        result = geometric_mean_psd_ratio(freqs, psd, psd)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_halved_psd(self):
        """PSD halved everywhere → ratio ≈ 0.5."""
        freqs = _freqs()
        psd_before = _flat_psd(freqs, n_channels=3, value=2.0)
        psd_after = _flat_psd(freqs, n_channels=3, value=1.0)
        result = geometric_mean_psd_ratio(freqs, psd_before, psd_after)
        np.testing.assert_allclose(result, 0.5, atol=1e-6)

    def test_1d_input(self):
        """1-D PSD should return a scalar."""
        freqs = _freqs()
        psd = _flat_psd(freqs)
        result = geometric_mean_psd_ratio(freqs, psd, psd)
        assert np.ndim(result) == 0
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_custom_frequency_range(self):
        """Custom fmin/fmax limits the comparison band."""
        freqs = _freqs()
        psd_before = _flat_psd(freqs, n_channels=2, value=1.0)
        psd_after = psd_before.copy()
        # Modify only outside [10, 30]
        psd_after[:, freqs < 10] = 0.1
        psd_after[:, freqs > 30] = 0.1
        result = geometric_mean_psd_ratio(
            freqs, psd_before, psd_after, fmin=10.0, fmax=30.0
        )
        np.testing.assert_allclose(result, 1.0, atol=1e-6)


# ── geometric_mean_psd ──────────────────────────────────────────────


class TestGeometricMeanPsd:
    """Tests for geometric_mean_psd (Raw-level helper)."""

    def _mock_raw(self, n_channels=4, n_freqs=200, fmax=100.0):
        """Create a mock Raw with a compute_psd method."""
        freqs = np.linspace(0, fmax, n_freqs)
        data = np.ones((n_channels, n_freqs))
        psd_obj = MagicMock()
        psd_obj.freqs = freqs
        psd_obj.get_data.return_value = data
        raw = MagicMock()
        raw.compute_psd.return_value = psd_obj
        return raw, freqs, data

    def test_returns_freqs_and_1d(self):
        """Should return (freqs, 1d_array)."""
        raw, freqs, _ = self._mock_raw()
        f, gm = geometric_mean_psd(raw)
        np.testing.assert_array_equal(f, freqs)
        assert gm.ndim == 1
        assert len(gm) == len(freqs)

    def test_flat_spectrum_gm_equals_value(self):
        """Flat spectrum at value v → gm = v."""
        raw, _, _ = self._mock_raw()
        _, gm = geometric_mean_psd(raw)
        np.testing.assert_allclose(gm, 1.0, atol=1e-6)

    def test_custom_fmax(self):
        """fmax should be forwarded to compute_psd."""
        raw, _, _ = self._mock_raw()
        geometric_mean_psd(raw, fmax=50.0)
        raw.compute_psd.assert_called_with(fmax=50.0, verbose=False)


# ── compute_all_qa_metrics ──────────────────────────────────────────


class TestComputeAllQaMetrics:
    """Tests for the convenience wrapper compute_all_qa_metrics."""

    def _mock_raw_pair(self, n_channels=4, n_freqs=200, fmax=100.0):
        """Create a pair of mock Raw objects with identical flat PSDs."""
        freqs = np.linspace(0, fmax, n_freqs)
        data_before = np.ones((n_channels, n_freqs))
        data_after = np.ones((n_channels, n_freqs))
        # Add a line-noise peak at 50 Hz to before, remove it in after
        mask50 = (freqs >= 48) & (freqs <= 52)
        data_before[:, mask50] = 10.0
        data_after[:, mask50] = 1.0  # cleaned

        def _make_raw(data):
            psd_obj = MagicMock()
            psd_obj.freqs = freqs
            psd_obj.get_data.return_value = data.copy()
            raw = MagicMock()
            raw.compute_psd.return_value = psd_obj
            return raw

        return _make_raw(data_before), _make_raw(data_after)

    def test_returns_expected_keys(self):
        """Output dict has all required keys."""
        raw_b, raw_a = self._mock_raw_pair()
        result = compute_all_qa_metrics(raw_b, raw_a, line_freq=50.0)
        expected_keys = {
            "peak_attenuation_db",
            "R_f0",
            "below_noise_distortion_db",
            "below_noise_pct",
            "overclean_proportion",
            "underclean_proportion",
            "geometric_mean_psd_ratio",
            "harmonics_hz",
            "per_harmonic_attenuation_db",
            "per_harmonic_R",
        }
        assert expected_keys == set(result.keys())

    def test_attenuation_positive_for_cleaned(self):
        """If peak was removed, attenuation should be positive."""
        raw_b, raw_a = self._mock_raw_pair()
        result = compute_all_qa_metrics(raw_b, raw_a, line_freq=50.0)
        assert result["peak_attenuation_db"] > 0

    def test_r_f0_near_one_after_cleaning(self):
        """Cleaned spectrum has flat residual → R_f0 ≈ 1."""
        raw_b, raw_a = self._mock_raw_pair()
        result = compute_all_qa_metrics(raw_b, raw_a, line_freq=50.0)
        assert result["R_f0"] == pytest.approx(1.0, abs=0.5)

    def test_harmonics_hz_length(self):
        """harmonics_hz list length = n_harmonics + 1."""
        raw_b, raw_a = self._mock_raw_pair()
        result = compute_all_qa_metrics(raw_b, raw_a, line_freq=50.0, n_harmonics=2)
        assert len(result["harmonics_hz"]) == 3
        assert result["harmonics_hz"] == [50.0, 100.0, 150.0]

    def test_per_harmonic_lists_match_harmonics(self):
        """Per-harmonic lists should have same length as harmonics_hz."""
        raw_b, raw_a = self._mock_raw_pair()
        result = compute_all_qa_metrics(raw_b, raw_a, line_freq=50.0, n_harmonics=1)
        assert len(result["per_harmonic_attenuation_db"]) == 2
        assert len(result["per_harmonic_R"]) == 2

    def test_scalar_values_are_float(self):
        """All non-list values should be plain floats."""
        raw_b, raw_a = self._mock_raw_pair()
        result = compute_all_qa_metrics(raw_b, raw_a, line_freq=50.0)
        for key, val in result.items():
            if isinstance(val, list):
                continue
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"

    def test_below_noise_pct_alias(self):
        """below_noise_pct should equal below_noise_distortion_db."""
        raw_b, raw_a = self._mock_raw_pair()
        result = compute_all_qa_metrics(raw_b, raw_a, line_freq=50.0)
        assert result["below_noise_pct"] == result["below_noise_distortion_db"]

    def test_proportions_in_0_1(self):
        """overclean and underclean proportions should be in [0, 1]."""
        raw_b, raw_a = self._mock_raw_pair()
        result = compute_all_qa_metrics(raw_b, raw_a, line_freq=50.0)
        assert 0.0 <= result["overclean_proportion"] <= 1.0
        assert 0.0 <= result["underclean_proportion"] <= 1.0
