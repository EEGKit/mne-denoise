"""Tests for internal visualization utilities."""

import mne
import numpy as np
import pytest

from mne_denoise.viz._utils import (
    _get_components,
    _get_filters,
    _get_info,
    _get_patterns,
    _get_scores,
    _handle_picks,
)


def test_viz_utils(fitted_dss, synthetic_data):
    """Test internal utility functions."""
    assert isinstance(_get_info(fitted_dss), mne.Info)
    assert _get_info(fitted_dss, synthetic_data.info) is synthetic_data.info

    class MockEst:
        pass

    est = MockEst()
    assert _get_info(est) is None
    est._mne_info = synthetic_data.info
    assert _get_info(est) is synthetic_data.info

    picks = _handle_picks(synthetic_data.info, picks=None)
    assert len(picks) == 5
    picks = _handle_picks(synthetic_data.info, picks=[0, 1])
    assert len(picks) == 2

    patterns = _get_patterns(fitted_dss)
    assert patterns.shape == (5, 3)

    with pytest.raises(ValueError):
        _get_patterns(MockEst())

    filters = _get_filters(fitted_dss)
    assert filters.shape == (3, 5)
    with pytest.raises(ValueError):
        _get_filters(MockEst())

    scores = _get_scores(fitted_dss)
    assert len(scores) == 3

    est_iter = MockEst()
    est_iter.convergence_info_ = {"dummy": 1}
    assert _get_scores(est_iter) is None
    assert _get_scores(MockEst()) is None

    sources = _get_components(fitted_dss, synthetic_data)
    assert sources.shape == (3, 200, 10)

    est_cached = MockEst()
    est_cached.sources_ = np.zeros((3, 200, 10))
    assert _get_components(est_cached, data=None) is est_cached.sources_

    with pytest.raises(ValueError, match="Data must be provided"):
        _get_components(fitted_dss, data=None)

    raw = mne.io.RawArray(synthetic_data.get_data()[0], synthetic_data.info)
    sources_raw = _get_components(fitted_dss, raw)
    assert sources_raw.ndim == 2
