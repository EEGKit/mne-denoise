"""Tests for mne_denoise.qa module."""

import numpy as np
import pytest

import mne_denoise.qa as qa
from mne_denoise.qa import (
    peak_attenuation_db,
    spectral_distortion,
    suppression_ratio,
    variance_removed,
)


# --- peak_attenuation_db ---

def test_peak_attenuation_1d_basic():
    """Peak is halved → ~3 dB attenuation."""
    freqs = np.arange(0, 100, 0.5)
    psd_before = np.ones_like(freqs) * 0.01
    psd_after = np.ones_like(freqs) * 0.01
    # Add peak at 50 Hz
    mask = (freqs >= 48) & (freqs <= 52)
    psd_before[mask] = 1.0
    psd_after[mask] = 0.5
    result = peak_attenuation_db(freqs, psd_before, psd_after, 50.0)
    assert isinstance(result, (float, np.floating))
    expected = 10 * np.log10(1.0 / 0.5)
    np.testing.assert_allclose(result, expected, atol=0.01)


def test_peak_attenuation_2d_input():
    """Test with (n_channels, n_freqs) shaped PSD."""
    freqs = np.arange(0, 100, 0.5)
    n_ch = 4
    psd_before = np.ones((n_ch, len(freqs))) * 0.01
    psd_after = np.ones((n_ch, len(freqs))) * 0.01
    mask = (freqs >= 48) & (freqs <= 52)
    psd_before[:, mask] = 1.0
    psd_after[:, mask] = 0.1
    result = peak_attenuation_db(freqs, psd_before, psd_after, 50.0)
    assert result.shape == (n_ch,)
    expected = 10 * np.log10(1.0 / 0.1)
    np.testing.assert_allclose(result, expected, atol=0.01)


def test_peak_attenuation_out_of_range():
    """Target outside frequency range → NaN (1D) or array of NaN (2D)."""
    freqs = np.arange(0, 50, 0.5)
    psd_before = np.ones_like(freqs)
    psd_after = np.ones_like(freqs)
    
    # 1D
    assert np.isnan(peak_attenuation_db(freqs, psd_before, psd_after, 100.0))
    
    # 2D
    psd_before_2d = np.ones((3, len(freqs)))
    psd_after_2d = np.ones((3, len(freqs)))
    result = peak_attenuation_db(freqs, psd_before_2d, psd_after_2d, 100.0)
    assert result.shape == (3,)
    assert np.all(np.isnan(result))


def test_peak_attenuation_no_change():
    """Same PSD before/after → 0 dB attenuation."""
    freqs = np.arange(0, 100, 0.5)
    psd = np.ones_like(freqs)
    result = peak_attenuation_db(freqs, psd, psd, 50.0)
    np.testing.assert_allclose(result, 0.0, atol=1e-10)


def test_peak_attenuation_custom_bandwidth():
    """Test custom bandwidth parameter."""
    freqs = np.arange(0, 100, 0.5)
    psd_before = np.ones_like(freqs)
    psd_after = np.ones_like(freqs)
    psd_before[(freqs >= 49) & (freqs <= 51)] = 10.0
    psd_after[(freqs >= 49) & (freqs <= 51)] = 1.0
    result = peak_attenuation_db(freqs, psd_before, psd_after, 50.0, bandwidth=1.0)
    assert result > 0


# --- suppression_ratio ---

def test_suppression_ratio_basic():
    """Test suppression_ratio with 1D and 2D input."""
    freqs = np.arange(0, 100, 0.5)
    psd_before = np.ones_like(freqs)
    psd_after = np.ones_like(freqs) * 0.1
    # Average before=1.0, after=0.1 -> Ratio=10 -> 10 dB
    res = suppression_ratio(freqs, psd_before, psd_after, 50.0)
    np.testing.assert_allclose(res, 10.0)

    # 2D input (averages across channels first)
    psd_before_2d = np.ones((2, len(freqs)))
    psd_after_2d = np.ones((2, len(freqs))) * 0.1
    res_2d = suppression_ratio(freqs, psd_before_2d, psd_after_2d, 50.0)
    np.testing.assert_allclose(res_2d, 10.0)


def test_suppression_ratio_out_of_range():
    """Target frequency outside range → NaN."""
    freqs = np.arange(0, 50, 0.5)
    psd = np.ones_like(freqs)
    assert np.isnan(suppression_ratio(freqs, psd, psd, 100.0))


def test_suppression_ratio_full_suppression():
    """Zero power after → Inf."""
    freqs = np.arange(0, 100, 0.5)
    psd_before = np.ones_like(freqs)
    psd_after = np.zeros_like(freqs)
    assert suppression_ratio(freqs, psd_before, psd_after, 50.0) == np.inf


# --- spectral_distortion ---

def test_spectral_distortion_basic():
    """Identical PSDs outside harmonics → 0 distortion."""
    freqs = np.arange(0, 200, 0.5)
    psd = np.ones((2, len(freqs)))
    # Exclude harmonics of 50 Hz
    res = spectral_distortion(freqs, psd, psd, line_freq=50.0, n_harmonics=3)
    np.testing.assert_allclose(res, 0.0, atol=1e-10)

    # Modified spectrum outside harmonics → positive distortion
    psd_after = psd.copy()
    psd_after[:, (freqs > 10) & (freqs < 20)] *= 2.0
    res_dist = spectral_distortion(freqs, psd, psd_after, line_freq=50.0)
    assert res_dist > 0


def test_spectral_distortion_no_safe_freqs():
    """No frequencies left after filtering/exclusion → 0.0."""
    freqs = np.linspace(48, 52, 10)
    psd = np.ones_like(freqs)
    # Exclude 50 +/- 2*bandwidth. bandwidth=5 -> exclude 40-60. freqs is 48-52.
    res = spectral_distortion(freqs, psd, psd, line_freq=50.0, bandwidth=5.0)
    assert res == 0.0


# --- variance_removed ---

def test_variance_removed_basic():
    """Test basic variance removal calculation."""
    x = np.array([1.0, -1.0, 1.0, -1.0])
    # Halving amplitude -> 1/4 variance -> 75% removed
    assert variance_removed(x, 0.5 * x) == 75.0


def test_variance_removed_zero_var():
    """Input with zero variance → 0.0."""
    x = np.zeros(10)
    assert variance_removed(x, x) == 0.0


# --- Re-export ---

def test_reexport_matches():
    """Ensure direct import matches the module attribute."""
    assert peak_attenuation_db is qa.peak_attenuation_db
