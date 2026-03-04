"""Tests for mne_denoise.icanclean module."""

from __future__ import annotations

import numpy as np
import pytest

from mne_denoise.icanclean import ICanClean
from mne_denoise.icanclean.cca import canonical_correlation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng():
    """Shared random generator."""
    return np.random.default_rng(42)


@pytest.fixture()
def synthetic_dual_layer(rng):
    """Create synthetic dual-layer data with a known artifact.

    Returns (data, primary_idx, ref_idx, sfreq, truth) where truth contains
    the clean brain signal for validation.
    """
    sfreq = 250.0
    duration = 10.0
    n_times = int(sfreq * duration)
    n_primary = 16
    n_ref = 4
    t = np.arange(n_times) / sfreq

    # Brain signal: sum of alpha + theta
    brain = np.zeros((n_primary, n_times))
    for i in range(n_primary):
        phase = rng.uniform(0, 2 * np.pi)
        brain[i] = (
            0.5 * np.sin(2 * np.pi * 10 * t + phase)
            + 0.3 * np.sin(2 * np.pi * 6 * t + phase * 0.7)
        )

    # Artifact sources: correlated across primary AND reference
    n_artifacts = 2
    artifact_sources = rng.standard_normal((n_artifacts, n_times)) * 2.0

    mixing_primary = rng.standard_normal((n_primary, n_artifacts)) * 0.8
    mixing_ref = rng.standard_normal((n_ref, n_artifacts)) * 1.0

    artifact_primary = mixing_primary @ artifact_sources
    artifact_ref = mixing_ref @ artifact_sources

    # Reference also has its own noise
    ref_noise = rng.standard_normal((n_ref, n_times)) * 0.3

    data_primary = brain + artifact_primary
    data_ref = artifact_ref + ref_noise

    data = np.vstack([data_primary, data_ref])
    primary_idx = list(range(n_primary))
    ref_idx = list(range(n_primary, n_primary + n_ref))

    truth = {"brain": brain, "artifact_primary": artifact_primary}
    return data, primary_idx, ref_idx, sfreq, truth


# ---------------------------------------------------------------------------
# CCA tests
# ---------------------------------------------------------------------------


class TestCanonicalCorrelation:
    """Tests for the CCA implementation."""

    def test_basic_shapes(self, rng):
        """CCA returns correct shapes."""
        n, px, py = 200, 8, 4
        X = rng.standard_normal((n, px))
        Y = rng.standard_normal((n, py))
        A, B, R, U, V = canonical_correlation(X, Y)

        d = min(px, py)
        assert A.shape == (px, d)
        assert B.shape == (py, d)
        assert R.shape == (d,)
        assert U.shape == (n, d)
        assert V.shape == (n, d)

    def test_correlations_descending(self, rng):
        """Canonical correlations are sorted descending."""
        X = rng.standard_normal((300, 10))
        Y = rng.standard_normal((300, 6))
        _, _, R, _, _ = canonical_correlation(X, Y)
        assert np.all(np.diff(R) <= 1e-10)  # descending

    def test_correlations_bounded(self, rng):
        """Canonical correlations are in [0, 1]."""
        X = rng.standard_normal((200, 8))
        Y = rng.standard_normal((200, 5))
        _, _, R, _, _ = canonical_correlation(X, Y)
        assert np.all(R >= -1e-10)
        assert np.all(R <= 1.0 + 1e-10)

    def test_perfect_correlation(self):
        """Perfectly correlated signals give R ≈ 1."""
        t = np.linspace(0, 1, 500)
        X = np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
        Y = X @ np.array([[0.5, 0.3], [-0.2, 0.8]])  # linear transform
        _, _, R, _, _ = canonical_correlation(X, Y)
        np.testing.assert_allclose(R[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(R[1], 1.0, atol=1e-6)

    def test_uncorrelated(self, rng):
        """Independent signals give low correlations."""
        X = rng.standard_normal((5000, 8))
        Y = rng.standard_normal((5000, 4))
        _, _, R, _, _ = canonical_correlation(X, Y)
        # With many samples, random correlations should be small
        assert R.max() < 0.15

    def test_unit_variance(self, rng):
        """Canonical variates have unit variance (ddof=1)."""
        X = rng.standard_normal((300, 6))
        Y = rng.standard_normal((300, 4))
        _, _, _, U, V = canonical_correlation(X, Y)
        np.testing.assert_allclose(U.std(axis=0, ddof=1), 1.0, atol=1e-10)
        np.testing.assert_allclose(V.std(axis=0, ddof=1), 1.0, atol=1e-10)

    def test_mismatched_samples_raises(self, rng):
        """Different n_samples raises ValueError."""
        X = rng.standard_normal((100, 5))
        Y = rng.standard_normal((80, 3))
        with pytest.raises(ValueError, match="same number of samples"):
            canonical_correlation(X, Y)

    def test_rank_deficient(self, rng):
        """Handles rank-deficient input gracefully."""
        X = rng.standard_normal((100, 4))
        # Y has only 2 independent columns out of 5
        base = rng.standard_normal((100, 2))
        Y = np.column_stack([base, base @ rng.standard_normal((2, 3))])
        A, B, R, U, V = canonical_correlation(X, Y)
        assert R.shape[0] <= 2  # rank of Y is 2


# ---------------------------------------------------------------------------
# ICanClean tests (numpy)
# ---------------------------------------------------------------------------


class TestICanCleanNumpy:
    """Tests for ICanClean with numpy arrays."""

    def test_basic_cleaning(self, synthetic_dual_layer):
        """ICanClean reduces artifact power."""
        data, primary_idx, ref_idx, sfreq, truth = synthetic_dual_layer

        icc = ICanClean(
            sfreq=sfreq,
            ref_channels=ref_idx,
            primary_channels=primary_idx,
            segment_len=2.0,
            overlap=0.5,
            threshold=0.5,
            verbose=False,
        )
        cleaned = icc.fit_transform(data)

        # Artifact power should decrease
        artifact = truth["artifact_primary"]
        residual_before = np.var(data[primary_idx] - truth["brain"])
        residual_after = np.var(cleaned[primary_idx] - truth["brain"])
        assert residual_after < residual_before, (
            f"Artifact power did not decrease: {residual_after:.4f} >= {residual_before:.4f}"
        )

    def test_output_shape(self, synthetic_dual_layer):
        """Output has same shape as input."""
        data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
        icc = ICanClean(
            sfreq=sfreq, ref_channels=ref_idx, verbose=False
        )
        cleaned = icc.fit_transform(data)
        assert cleaned.shape == data.shape

    def test_ref_channels_unchanged(self, synthetic_dual_layer):
        """Reference channels are not modified."""
        data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
        icc = ICanClean(
            sfreq=sfreq,
            ref_channels=ref_idx,
            primary_channels=primary_idx,
            verbose=False,
        )
        cleaned = icc.fit_transform(data)
        np.testing.assert_array_equal(cleaned[ref_idx], data[ref_idx])

    def test_qc_attributes(self, synthetic_dual_layer):
        """QC attributes are populated after cleaning."""
        data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
        icc = ICanClean(
            sfreq=sfreq, ref_channels=ref_idx, verbose=False
        )
        icc.fit_transform(data)

        assert icc.n_windows_ > 0
        assert icc.correlations_.shape[0] == icc.n_windows_
        assert icc.n_removed_.shape == (icc.n_windows_,)
        assert len(icc.removed_idx_) == icc.n_windows_
        assert len(icc.filters_) == icc.n_windows_
        assert len(icc.patterns_) == icc.n_windows_

    def test_no_ref_channels_raises(self, rng):
        """Missing ref_channels on numpy input raises ValueError."""
        data = rng.standard_normal((10, 1000))
        icc = ICanClean(sfreq=250.0, verbose=False)
        with pytest.raises(ValueError, match="ref_channels must be provided"):
            icc.fit_transform(data)

    def test_window_too_long_raises(self, rng):
        """Window longer than data raises ValueError."""
        data = rng.standard_normal((10, 100))
        icc = ICanClean(
            sfreq=250.0,
            ref_channels=[8, 9],
            segment_len=10.0,  # 2500 samples > 100
            verbose=False,
        )
        with pytest.raises(ValueError, match="exceeds data length"):
            icc.fit_transform(data)

    def test_auto_threshold(self, synthetic_dual_layer):
        """Auto threshold mode runs without error."""
        data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
        icc = ICanClean(
            sfreq=sfreq,
            ref_channels=ref_idx,
            threshold="auto",
            verbose=False,
        )
        cleaned = icc.fit_transform(data)
        assert cleaned.shape == data.shape

    def test_max_reject_fraction(self, synthetic_dual_layer):
        """max_reject_fraction caps the number of removed components."""
        data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
        icc = ICanClean(
            sfreq=sfreq,
            ref_channels=ref_idx,
            threshold=0.01,  # very low -> would remove everything
            max_reject_fraction=0.25,
            verbose=False,
        )
        icc.fit_transform(data)
        n_comp = min(len(primary_idx), len(ref_idx))
        max_allowed = max(1, int(0.25 * n_comp))
        assert np.all(icc.n_removed_ <= max_allowed)

    def test_zero_overlap(self, synthetic_dual_layer):
        """overlap=0 works (non-overlapping windows)."""
        data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
        icc = ICanClean(
            sfreq=sfreq,
            ref_channels=ref_idx,
            overlap=0.0,
            verbose=False,
        )
        cleaned = icc.fit_transform(data)
        assert cleaned.shape == data.shape


# ---------------------------------------------------------------------------
# ICanClean tests (MNE)
# ---------------------------------------------------------------------------


class TestICanCleanMNE:
    """Tests for ICanClean with MNE objects."""

    @pytest.fixture()
    def raw_with_refs(self, rng):
        """Create MNE Raw with EEG + EOG channels."""
        mne = pytest.importorskip("mne")
        sfreq = 256.0
        n_times = int(sfreq * 8)
        n_eeg = 12
        n_eog = 2
        t = np.arange(n_times) / sfreq

        # Brain
        brain = np.zeros((n_eeg, n_times))
        for i in range(n_eeg):
            brain[i] = 0.5 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))

        # Shared artifact
        art_src = rng.standard_normal((1, n_times)) * 3
        art_eeg = rng.standard_normal((n_eeg, 1)) @ art_src
        art_eog = rng.standard_normal((n_eog, 1)) @ art_src

        eeg_data = brain + art_eeg
        eog_data = art_eog + rng.standard_normal((n_eog, n_times)) * 0.2

        ch_names = [f"EEG{i+1:03d}" for i in range(n_eeg)] + [
            f"EOG{i+1}" for i in range(n_eog)
        ]
        ch_types = ["eeg"] * n_eeg + ["eog"] * n_eog

        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(np.vstack([eeg_data, eog_data]), info, verbose=False)
        return raw, brain

    def test_mne_raw_cleaning(self, raw_with_refs):
        """ICanClean works on MNE Raw and returns Raw."""
        mne = pytest.importorskip("mne")
        raw, _ = raw_with_refs

        icc = ICanClean(
            sfreq=raw.info["sfreq"],
            ref_channels=["EOG1", "EOG2"],
            segment_len=2.0,
            threshold=0.5,
            verbose=False,
        )
        raw_clean = icc.fit_transform(raw)
        assert isinstance(raw_clean, mne.io.RawArray)
        assert raw_clean.get_data().shape == raw.get_data().shape

    def test_mne_prefix_detection(self, rng):
        """Prefix-based channel detection works."""
        mne = pytest.importorskip("mne")
        sfreq = 250.0
        n_times = int(sfreq * 6)
        n_scalp = 8
        n_noise = 4

        ch_names = [f"1-EEG{i}" for i in range(n_scalp)] + [
            f"2-NSE{i}" for i in range(n_noise)
        ]
        ch_types = ["eeg"] * (n_scalp + n_noise)
        info = mne.create_info(ch_names, sfreq, ch_types)
        data = rng.standard_normal((n_scalp + n_noise, n_times))
        raw = mne.io.RawArray(data, info, verbose=False)

        icc = ICanClean(
            sfreq=sfreq,
            primary_prefix="1-",
            ref_prefix="2-",
            verbose=False,
        )
        raw_clean = icc.fit_transform(raw)
        assert isinstance(raw_clean, mne.io.RawArray)
        assert icc.primary_channels_ == [f"1-EEG{i}" for i in range(n_scalp)]
        assert icc.ref_channels_ == [f"2-NSE{i}" for i in range(n_noise)]

    def test_mne_artifact_reduction(self, raw_with_refs):
        """ICanClean reduces artifact variance on MNE Raw."""
        raw, brain = raw_with_refs
        n_eeg = brain.shape[0]

        icc = ICanClean(
            sfreq=raw.info["sfreq"],
            ref_channels=["EOG1", "EOG2"],
            threshold=0.5,
            verbose=False,
        )
        raw_clean = icc.fit_transform(raw)

        before = raw.get_data()[:n_eeg]
        after = raw_clean.get_data()[:n_eeg]

        var_before = np.var(before - brain)
        var_after = np.var(after - brain)
        assert var_after < var_before


# ---------------------------------------------------------------------------
# Viz smoke tests
# ---------------------------------------------------------------------------


class TestICanCleanViz:
    """Smoke tests for ICanClean visualization functions."""

    @pytest.fixture()
    def fitted_icanclean(self, synthetic_dual_layer):
        data, primary_idx, ref_idx, sfreq, _ = synthetic_dual_layer
        icc = ICanClean(
            sfreq=sfreq, ref_channels=ref_idx, verbose=False
        )
        icc.fit_transform(data)
        return icc, data, sfreq, primary_idx

    def test_plot_correlation_scores(self, fitted_icanclean):
        """plot_correlation_scores runs without error."""
        import matplotlib
        matplotlib.use("Agg")
        from mne_denoise.viz.icanclean import plot_correlation_scores

        icc, _, _, _ = fitted_icanclean
        fig = plot_correlation_scores(icc, show=False)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_removal_summary(self, fitted_icanclean):
        """plot_removal_summary runs without error."""
        import matplotlib
        matplotlib.use("Agg")
        from mne_denoise.viz.icanclean import plot_removal_summary

        icc, _, _, _ = fitted_icanclean
        fig = plot_removal_summary(icc, show=False)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_psd_comparison(self, fitted_icanclean):
        """plot_psd_comparison runs without error."""
        import matplotlib
        matplotlib.use("Agg")
        from mne_denoise.viz.icanclean import plot_psd_comparison

        icc, data, sfreq, primary_idx = fitted_icanclean
        cleaned = icc.fit_transform(data)  # re-run to get cleaned
        # Actually we already have cleaned from fixture; just use data arrays
        fig = plot_psd_comparison(
            data[primary_idx], cleaned[primary_idx], sfreq, show=False
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_timeseries_comparison(self, fitted_icanclean):
        """plot_timeseries_comparison runs without error."""
        import matplotlib
        matplotlib.use("Agg")
        from mne_denoise.viz.icanclean import plot_timeseries_comparison

        icc, data, sfreq, primary_idx = fitted_icanclean
        cleaned = icc.fit_transform(data)
        fig = plot_timeseries_comparison(
            data[primary_idx], cleaned[primary_idx], sfreq,
            channel=0, tmax=2.0, show=False,
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
