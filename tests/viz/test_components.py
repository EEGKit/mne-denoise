"""Tests for mne_denoise.viz.components functions."""

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

from mne_denoise.dss import DSS
from mne_denoise.viz import (
    plot_component_epochs_image,
    plot_component_patterns,
    plot_component_score_curve,
    plot_component_spectrogram,
    plot_component_summary,
    plot_component_time_series,
)


def test_plot_component_score_curve(fitted_dss):
    """Test score curve plotting."""
    fig = plot_component_score_curve(fitted_dss, mode="raw", show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_component_score_curve(fitted_dss, mode="cumulative", show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_component_score_curve(fitted_dss, mode="ratio", show=False)
    assert isinstance(fig, plt.Figure)

    fig, ax = plt.subplots()
    plot_component_score_curve(fitted_dss, ax=ax, show=False)

    with pytest.raises(ValueError, match="mode must be one of"):
        plot_component_score_curve(fitted_dss, mode="bad-mode", show=False)

    class NoScoreEst:
        pass

    with pytest.raises(ValueError, match="does not expose component scores"):
        plot_component_score_curve(NoScoreEst(), show=False)


def test_plot_component_patterns(fitted_dss):
    """Test topomap plotting."""
    fig = plot_component_patterns(fitted_dss, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_component_patterns(fitted_dss, n_components=2, show=False)
    assert isinstance(fig, plt.Figure)

    class MockEst:
        patterns_ = np.zeros((5, 3))

    fig = plot_component_patterns(MockEst(), show=False)
    assert isinstance(fig, plt.Figure)

    fig, ax = plt.subplots()
    fig_ret = plot_component_patterns(MockEst(), ax=ax, show=False)
    assert fig_ret is fig


def test_component_primitives_support_zapline(fitted_zapline):
    """Component primitives should also work with fitted ZapLine estimators."""
    fig = plot_component_score_curve(fitted_zapline, show=False)
    assert isinstance(fig, plt.Figure)

    fig_ext, custom_ax = plt.subplots()
    ret_fig = plot_component_score_curve(fitted_zapline, ax=custom_ax, show=False)
    assert ret_fig is fig_ext

    fig = plot_component_patterns(fitted_zapline, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_component_patterns(fitted_zapline, n_components=2, show=False)
    assert isinstance(fig, plt.Figure)

    fig_ext, custom_ax = plt.subplots()
    ret_fig = plot_component_patterns(fitted_zapline, ax=custom_ax, show=False)
    assert ret_fig is fig_ext


def test_plot_component_summary(fitted_dss, synthetic_data):
    """Test component summary dashboard."""
    fig = plot_component_summary(
        fitted_dss, data=synthetic_data, n_components=2, show=False
    )
    assert isinstance(fig, plt.Figure)

    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    fig = plot_component_summary(fitted_dss, data=raw, n_components=1, show=False)
    assert isinstance(fig, plt.Figure)

    plot_component_summary(
        fitted_dss, data=synthetic_data, n_components=[0, 2], show=False
    )

    def dummy_bias(d):
        return d

    DSS(n_components=3, bias=dummy_bias)
    with pytest.raises(ValueError, match="Data must be provided"):
        plot_component_summary(fitted_dss, data=None, show=False)
        plot_component_summary(fitted_dss, data=None, show=False)


def test_plot_component_epochs_image(fitted_dss, synthetic_data):
    """Test component image plotting."""
    fig = plot_component_epochs_image(
        fitted_dss, data=synthetic_data, n_components=2, show=False
    )
    assert isinstance(fig, plt.Figure)

    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    fig = plot_component_epochs_image(fitted_dss, data=raw, n_components=1, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_component_epochs_image(
        fitted_dss,
        data=synthetic_data,
        n_components=[0, 2],
        show=False,
    )
    assert isinstance(fig, plt.Figure)


def test_plot_component_time_series(fitted_dss, synthetic_data):
    """Test stacked time series plotting."""
    fig = plot_component_time_series(fitted_dss, data=synthetic_data, show=False)
    assert isinstance(fig, plt.Figure)

    with pytest.raises(ValueError):
        plot_component_time_series(fitted_dss, data=None, show=False)

    class MockEst:
        sources_ = np.random.randn(3, 100)
        eigenvalues_ = np.array([1.0, 0.5, 0.1])

        def get_params(self):
            return {}

    fig = plot_component_time_series(MockEst(), show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_component_spectrogram():
    """Test component TFR plotting."""
    sfreq = 100.0
    n_times = 200

    comp_1d = np.random.randn(n_times)
    fig = plot_component_spectrogram(comp_1d, sfreq=sfreq, show=False)
    assert isinstance(fig, plt.Figure)

    comp_2d = np.random.randn(1, n_times)
    fig = plot_component_spectrogram(
        comp_2d, sfreq=sfreq, freqs=np.arange(1, 10), show=False
    )
    assert isinstance(fig, plt.Figure)

    fig, ax = plt.subplots()
    ret_fig = plot_component_spectrogram(comp_1d, sfreq=sfreq, ax=ax, show=False)
    assert ret_fig is fig


def test_fname_parameter_components(fitted_dss, tmp_path):
    """Test fname parameter on a component plot function."""
    fpath = tmp_path / "scores.png"
    fig = plot_component_score_curve(
        fitted_dss,
        mode="raw",
        show=False,
        fname=str(fpath),
    )
    assert isinstance(fig, plt.Figure)
    assert fpath.exists()

    fpath = tmp_path / "patterns.png"
    fig = plot_component_patterns(
        fitted_dss,
        show=False,
        fname=str(fpath),
    )
    assert isinstance(fig, plt.Figure)
    assert fpath.exists()
