"""Tests for signal-domain visualization helpers and plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

from mne_denoise.viz import (
    plot_channel_time_course_comparison,
    plot_evoked_gfp_comparison,
    plot_grand_average_evokeds,
    plot_power_ratio_map,
    plot_signal_overlay,
)
from mne_denoise.viz.signals import _add_time_windows, _ch_index, _get_times_ms


@pytest.fixture(scope="module")
def signal_evokeds(synthetic_data):
    """Build a small grouped-evoked dict for generic group plotting tests."""
    base = synthetic_data.average()
    groups = {}
    for group_idx, group in enumerate(["C0", "C1", "C2"]):
        evoked_list = []
        for sub_idx in range(4):
            evoked = base.copy()
            evoked.data = evoked.data.copy() + 0.05 * group_idx + 0.01 * sub_idx
            evoked_list.append(evoked)
        groups[group] = evoked_list
    return groups


def test_signal_viz_show(fitted_dss, synthetic_data):
    """Test show=True code paths for signal plots."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
        epochs_clean = fitted_dss.transform(synthetic_data)
        plot_evoked_gfp_comparison(synthetic_data, epochs_clean, show=True)
        plot_channel_time_course_comparison(synthetic_data, epochs_clean, show=True)
        plot_power_ratio_map(synthetic_data, epochs_clean, show=True)
        plot_signal_overlay(synthetic_data, epochs_clean, show=True)


def test_signal_comparisons(fitted_dss, synthetic_data):
    """Test the public signal comparison primitives."""
    epochs_clean = fitted_dss.transform(synthetic_data)
    raw_orig = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    raw_clean = mne.io.RawArray(epochs_clean.get_data()[0], synthetic_data.info)

    fig = plot_channel_time_course_comparison(
        synthetic_data, epochs_clean, picks=[0], show=False
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_channel_time_course_comparison(
        raw_orig, raw_clean, picks=[0], start=10, stop=50, show=False
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_channel_time_course_comparison(
        synthetic_data, epochs_clean, picks=[0, 1], show=False
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_power_ratio_map(
        synthetic_data, epochs_clean, info=synthetic_data.info, show=False
    )
    assert isinstance(fig, plt.Figure)

    fig, ax = plt.subplots()
    plot_power_ratio_map(raw_orig, raw_clean, ax=ax, show=False)

    fig = plot_power_ratio_map(synthetic_data, epochs_clean, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_signal_overlay(synthetic_data, epochs_clean, show=False)
    assert isinstance(fig, plt.Figure)

    fig = plot_signal_overlay(
        synthetic_data, epochs_clean, scale_after=True, show=False
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_signal_overlay(
        synthetic_data, epochs_clean, start=0.1, stop=0.5, show=False
    )
    assert isinstance(fig, plt.Figure)

    fig = plot_signal_overlay(raw_orig, raw_clean, show=False)
    assert isinstance(fig, plt.Figure)


def test_plot_evoked_gfp_comparison(synthetic_data):
    """Test evoked comparison with and without bootstrap CI."""
    epochs_before = synthetic_data
    epochs_after = synthetic_data.copy()

    fig = plot_evoked_gfp_comparison(epochs_before, epochs_after, n_boot=10, show=False)
    assert isinstance(fig, plt.Figure)

    ev_before = epochs_before.average()
    ev_after = epochs_after.average()
    fig = plot_evoked_gfp_comparison(ev_before, ev_after, ci=0.95, show=False)
    assert isinstance(fig, plt.Figure)

    fig, ax = plt.subplots()
    plot_evoked_gfp_comparison(ev_before, ev_after, ax=ax, show=False)


def test_signal_private_helpers(synthetic_data):
    """Test moved signal helpers shared with ERP plots."""
    ax = plt.figure().add_subplot(111)
    _add_time_windows(ax, {"P300": (0.25, 0.50)})
    assert len(ax.patches) == 1

    idx = _ch_index(synthetic_data, "Pz")
    assert idx == synthetic_data.ch_names.index("Pz")

    idx = _ch_index(np.zeros((5, 10)), "2")
    assert idx == 2

    times_ms = _get_times_ms(synthetic_data)
    assert np.allclose(times_ms, synthetic_data.times * 1000)

    with pytest.raises(ValueError, match="Cannot determine times"):
        _get_times_ms(np.zeros((5, 10)))


class TestPlotGrandAverageEvokeds:
    def test_basic(self, signal_evokeds):
        fig = plot_grand_average_evokeds(signal_evokeds, show=False)
        assert isinstance(fig, plt.Figure)

    def test_single_channel(self, signal_evokeds):
        fig = plot_grand_average_evokeds(
            signal_evokeds,
            channels=("Cz",),
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_single_subject(self, synthetic_data):
        single = {group: [synthetic_data.average()] for group in ["C0", "C1", "C2"]}
        fig = plot_grand_average_evokeds(single, show=False)
        assert isinstance(fig, plt.Figure)

    def test_custom_options(self, signal_evokeds):
        fig = plot_grand_average_evokeds(
            signal_evokeds,
            channels=("Pz",),
            time_windows={"P300": (0.25, 0.5)},
            suptitle="Custom Grand Average",
            group_order=["C2", "C0"],
            group_colors={"C0": "#aaa", "C2": "#bbb"},
            group_labels={"C0": "Base", "C2": "DSS"},
            figsize=(10, 6),
            show=False,
        )
        assert isinstance(fig, plt.Figure)

    def test_fname(self, signal_evokeds, tmp_path):
        fpath = tmp_path / "grand_avg.png"
        plot_grand_average_evokeds(signal_evokeds, show=False, fname=str(fpath))
        assert fpath.exists()
