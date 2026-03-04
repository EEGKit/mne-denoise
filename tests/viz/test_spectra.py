"""Tests for mne_denoise.viz.spectra functions."""

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

from mne_denoise.viz import (
    plot_component_psd_comparison,
    plot_narrowband_score_scan,
    plot_psd_comparison,
    plot_psd_gallery,
    plot_psd_overlay,
    plot_psd_zoom_comparison,
    plot_spectrogram_comparison,
    plot_time_frequency_mask,
)


def test_plot_narrowband_score_scan():
    """Test narrowband scan visualization."""
    frequencies = np.arange(5, 30, 0.5)
    eigenvalues = np.random.randn(len(frequencies)) * 0.1 + 0.5

    peak_idx = np.argmin(np.abs(frequencies - 10))
    eigenvalues[peak_idx] = 2.0
    eigenvalues[peak_idx - 1 : peak_idx + 2] += np.array([0.5, 0, 0.5])

    fig = plot_narrowband_score_scan(frequencies, eigenvalues, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_narrowband_score_scan(
        frequencies,
        eigenvalues,
        peak_freq=frequencies[peak_idx],
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_narrowband_score_scan(
        frequencies,
        eigenvalues,
        true_freqs=[10, 22],
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_narrowband_score_scan(
        frequencies,
        eigenvalues,
        peak_freq=10.0,
        true_freqs=[10, 22],
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig, ax = plt.subplots()
    fig_ret = plot_narrowband_score_scan(frequencies, eigenvalues, ax=ax, show=False)
    assert fig_ret is fig

    with pytest.raises(ValueError, match="matching first dimensions"):
        plot_narrowband_score_scan(frequencies, eigenvalues[:-1], show=False)


def test_plot_time_frequency_mask():
    """Test TF mask visualization."""
    n_freqs, n_times = 10, 20
    mask = np.random.rand(n_freqs, n_times)
    times = np.arange(n_times)
    freqs = np.arange(n_freqs)

    fig = plot_time_frequency_mask(mask, times, freqs, show=False)
    assert isinstance(fig, plt.Figure)

    fig, ax = plt.subplots()
    fig_ret = plot_time_frequency_mask(mask, times, freqs, ax=ax, show=False)
    assert fig_ret is fig

    with pytest.raises(ValueError, match="mask shape must match"):
        plot_time_frequency_mask(mask[:, :-1], times, freqs, show=False)

    with pytest.raises(ValueError, match="mask must be a 2D array"):
        plot_time_frequency_mask(mask[0], times, freqs, show=False)


def test_plot_psd_comparison(fitted_dss, synthetic_data):
    """Test PSD comparison visualization."""
    epochs_clean = fitted_dss.transform(synthetic_data)
    raw_before = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    raw_after = mne.io.RawArray(epochs_clean.get_data()[0], synthetic_data.info)

    fig, ax = plt.subplots()
    fig_ret = plot_psd_comparison(synthetic_data, epochs_clean, ax=ax, show=False)
    assert fig_ret is fig

    fig = plot_psd_comparison(raw_before, raw_after, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_psd_comparison(synthetic_data, epochs_clean, average=False, show=False)
    assert isinstance(fig, plt.Figure)

    array_before = synthetic_data.get_data()[0]
    array_after = epochs_clean.get_data()[0]
    fig = plot_psd_comparison(
        array_before,
        array_after,
        sfreq=synthetic_data.info["sfreq"],
        line_freq=10.0,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    with pytest.raises(ValueError, match="sfreq must be provided"):
        plot_psd_comparison(array_before, array_after, show=False)


def test_plot_psd_comparison_with_zapline_arrays(zapline_data, fitted_zapline):
    """Test PSD comparison with ZapLine-style array inputs."""
    data, sfreq = zapline_data
    cleaned = fitted_zapline.transform(data)

    fig = plot_psd_comparison(data, cleaned, sfreq=sfreq, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_psd_comparison(
        data,
        cleaned,
        sfreq=sfreq,
        line_freq=50.0,
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig_ext, custom_ax = plt.subplots()
    ret_fig = plot_psd_comparison(
        data,
        cleaned,
        sfreq=sfreq,
        ax=custom_ax,
        show=False,
    )
    assert ret_fig is fig_ext

    fig = plot_psd_comparison(data, cleaned, sfreq=sfreq, fmax=150.0, show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_component_psd_comparison(fitted_dss, synthetic_data):
    """Test component PSD comparison visualization."""
    sources = fitted_dss.transform(synthetic_data)

    fig = plot_component_psd_comparison(
        synthetic_data, sources, sfreq=100, peak_freq=10, show=False
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_component_psd_comparison(synthetic_data, sources, sfreq=100, show=False)
    assert isinstance(fig, plt.Figure)

    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    sources_raw = fitted_dss.transform(raw)
    fig = plot_component_psd_comparison(
        raw, sources_raw, sfreq=100, peak_freq=10, show=False
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_component_psd_comparison(
        raw, sources_raw, sfreq=100, fmin=5, fmax=20, show=False
    )
    assert isinstance(fig, plt.Figure)


def test_plot_psd_zoom_comparison():
    """Test PSD comparison with zoom panels."""
    freqs = np.arange(0, 200, 0.5)
    psd_before = np.ones_like(freqs) * 1e-6
    psd_after = np.ones_like(freqs) * 1e-5

    fig = plot_psd_zoom_comparison(
        freqs,
        psd_before,
        freqs,
        psd_after,
        series_name="M1",
        title="sub-01",
        zoom_freqs=[50.0, 100.0],
        zoom_annotations=["atten=10 dB", "atten=8 dB"],
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_psd_gallery():
    """Test PSD gallery visualization."""
    freqs = np.arange(0, 200, 0.5)
    psd_before = np.ones_like(freqs) * 1e-6
    series_psds = {
        "M1": (freqs, np.ones_like(freqs) * 1e-5),
        "M2": (freqs, np.ones_like(freqs) * 1.1e-5),
    }

    fig = plot_psd_gallery(
        freqs, psd_before, series_psds, zoom_freqs=[50.0], show=False
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_psd_gallery(
        freqs,
        psd_before,
        series_psds,
        zoom_freqs=[50.0, 100.0],
        series_order=["M2", "M1"],
        title="sub-01",
        show=False,
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_psd_gallery(
        freqs,
        psd_before,
        series_psds,
        zoom_freqs=[50.0, 100.0],
        series_order=["M1", "MISSING"],
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_psd_overlay():
    """Test PSD overlay visualization."""
    freqs = np.arange(0, 200, 0.5)
    psd_before = np.ones_like(freqs) * 1e-5
    series_psds = {
        "M0": (freqs, np.ones_like(freqs) * 1e-6),
        "M1": (freqs, np.ones_like(freqs) * 1.2e-6),
    }

    fig = plot_psd_overlay(freqs, psd_before, series_psds, focus_freq=50.0, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_psd_overlay(
        freqs,
        psd_before,
        series_psds,
        focus_freq=50.0,
        n_harmonics=2,
        title="sub-01",
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_spectrogram_comparison(fitted_dss, synthetic_data):
    """Test spectrogram comparison visualization."""
    epochs_clean = fitted_dss.transform(synthetic_data)
    fig = plot_spectrogram_comparison(
        synthetic_data, epochs_clean, fmin=1, fmax=20, n_freqs=5, show=False
    )
    assert isinstance(fig, plt.Figure)

    with pytest.raises(ValueError, match="n_freqs must be at least 2"):
        plot_spectrogram_comparison(
            synthetic_data, epochs_clean, fmin=1, fmax=20, n_freqs=1, show=False
        )
