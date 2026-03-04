"""Tests for general summary-level denoising plots."""

import matplotlib.pyplot as plt
import mne
import pytest

from mne_denoise.viz import (
    plot_denoising_summary,
)


def test_comparison_viz_show(fitted_dss, synthetic_data):
    """Test show=True code paths for the denoising summary plot."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
        epochs_clean = fitted_dss.transform(synthetic_data)
        plot_denoising_summary(synthetic_data, epochs_clean, show=True)


def test_denoising_summary(fitted_dss, synthetic_data):
    """Test denoising summary plots."""
    epochs_clean = fitted_dss.transform(synthetic_data)
    raw_orig = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    raw_clean = mne.io.RawArray(epochs_clean.get_data()[0], synthetic_data.info)

    fig = plot_denoising_summary(raw_orig, raw_clean, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_denoising_summary(synthetic_data, epochs_clean, show=False)
    assert isinstance(fig, plt.Figure)


def test_fname_parameter_comparison(fitted_dss, synthetic_data, tmp_path):
    """Test fname parameter on the denoising summary plot."""
    epochs_clean = fitted_dss.transform(synthetic_data)

    summary_path = tmp_path / "summary.png"
    fig = plot_denoising_summary(
        synthetic_data, epochs_clean, show=False, fname=str(summary_path)
    )
    assert isinstance(fig, plt.Figure)
    assert summary_path.exists()
