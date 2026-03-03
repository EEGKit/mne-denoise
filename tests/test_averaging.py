"""Tests for AverageBias._apply_datasets method."""

import numpy as np
import pytest

from mne_denoise.dss.denoisers.averaging import AverageBias


class TestApplyDatasets:
    """Tests for _apply_datasets (axis='datasets')."""

    def test_uniform_average(self):
        """Uniform averaging: each slice should equal the grand mean."""
        rng = np.random.RandomState(0)
        data = rng.randn(4, 3, 50)  # 4 datasets, 3 channels, 50 times
        bias = AverageBias(axis="datasets")
        biased = bias.apply(data)
        expected_mean = data.mean(axis=0)
        for i in range(4):
            np.testing.assert_allclose(biased[i], expected_mean)

    def test_weighted_average(self):
        """Weighted averaging: result should reflect weights."""
        rng = np.random.RandomState(1)
        data = rng.randn(3, 2, 20)
        weights = np.array([1.0, 2.0, 3.0])
        bias = AverageBias(axis="datasets", weights=weights)
        biased = bias.apply(data)
        norm_w = weights / weights.sum()
        expected = np.tensordot(norm_w, data, axes=(0, 0))
        for i in range(3):
            np.testing.assert_allclose(biased[i], expected)

    def test_wrong_ndim(self):
        """2D input should raise ValueError."""
        bias = AverageBias(axis="datasets")
        with pytest.raises(ValueError, match="requires 3D"):
            bias.apply(np.ones((3, 10)))

    def test_weights_length_mismatch(self):
        """Weights with wrong length should raise."""
        data = np.ones((3, 2, 10))
        bias = AverageBias(axis="datasets", weights=np.ones(5))
        with pytest.raises(ValueError, match="weights length"):
            bias.apply(data)

    def test_output_shape(self):
        """Output shape should match input."""
        data = np.random.randn(5, 4, 30)
        bias = AverageBias(axis="datasets")
        assert bias.apply(data).shape == data.shape

    def test_is_copy(self):
        """Output should be an independent copy."""
        data = np.random.randn(3, 2, 10)
        bias = AverageBias(axis="datasets")
        biased = bias.apply(data)
        biased[0, 0, 0] = 999
        assert data[0, 0, 0] != 999


class TestAverageBiasInit:
    """Test __init__ validation."""

    def test_invalid_axis(self):
        with pytest.raises(ValueError, match="axis must be"):
            AverageBias(axis="invalid")
