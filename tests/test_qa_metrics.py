"""Tests for peak_attenuation_db (canonical in mne_denoise.qa.metrics)."""

import numpy as np

from mne_denoise.qa import peak_attenuation_db as _peak_atten_qa
from mne_denoise.viz import peak_attenuation_db


class TestPeakAttenuationDb:
    """Tests for peak_attenuation_db."""

    def test_1d_basic_attenuation(self):
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

    def test_2d_input(self):
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

    def test_out_of_range_1d(self):
        """Target outside frequency range → NaN."""
        freqs = np.arange(0, 50, 0.5)
        psd_before = np.ones_like(freqs)
        psd_after = np.ones_like(freqs)
        result = peak_attenuation_db(freqs, psd_before, psd_after, 100.0)
        assert np.isnan(result)

    def test_out_of_range_2d(self):
        """Target outside frequency range for 2D → array of NaN."""
        freqs = np.arange(0, 50, 0.5)
        psd_before = np.ones((3, len(freqs)))
        psd_after = np.ones((3, len(freqs)))
        result = peak_attenuation_db(freqs, psd_before, psd_after, 100.0)
        assert result.shape == (3,)
        assert np.all(np.isnan(result))

    def test_no_attenuation(self):
        """Same PSD before/after → 0 dB attenuation."""
        freqs = np.arange(0, 100, 0.5)
        psd = np.ones_like(freqs)
        result = peak_attenuation_db(freqs, psd, psd, 50.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_custom_bandwidth(self):
        """Test custom bandwidth parameter."""
        freqs = np.arange(0, 100, 0.5)
        psd_before = np.ones_like(freqs)
        psd_after = np.ones_like(freqs)
        psd_before[(freqs >= 49) & (freqs <= 51)] = 10.0
        psd_after[(freqs >= 49) & (freqs <= 51)] = 1.0
        result = peak_attenuation_db(freqs, psd_before, psd_after, 50.0, bandwidth=1.0)
        assert result > 0

    def test_reexport_matches(self):
        """Ensure qa re-export matches viz canonical import."""
        assert peak_attenuation_db is _peak_atten_qa
