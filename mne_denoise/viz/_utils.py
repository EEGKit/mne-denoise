"""Internal utility functions for visualization.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

import mne
import numpy as np


def _get_info(estimator, info=None):
    """Resolve MNE info from estimator or argument."""
    if info is not None:
        return info
    if hasattr(estimator, "info_") and estimator.info_ is not None:
        return estimator.info_
    if hasattr(estimator, "_mne_info") and estimator._mne_info is not None:
        return estimator._mne_info
    return None


def _get_patterns(estimator):
    """Extract spatial patterns from estimator."""
    if hasattr(estimator, "patterns_") and estimator.patterns_ is not None:
        return estimator.patterns_
    raise ValueError(
        "Estimator does not have a 'patterns_' attribute or it is None. "
        "Make sure the estimator is fitted."
    )


def _get_filters(estimator):
    """Extract spatial filters from estimator."""
    if hasattr(estimator, "filters_") and estimator.filters_ is not None:
        return estimator.filters_
    raise ValueError("Estimator does not have a 'filters_' attribute or it is None.")


def _get_scores(estimator):
    """Extract scores (eigenvalues or similar) from estimator.

    Checks for both eigenvalues_ (DSS) and scores_ (ZapLine).
    """
    # Check for eigenvalues_ (LinearDSS)
    if hasattr(estimator, "eigenvalues_") and estimator.eigenvalues_ is not None:
        ev = estimator.eigenvalues_
        if isinstance(ev, np.ndarray) and ev.size > 0:
            return ev
    # Check for scores_ (ZapLine)
    if hasattr(estimator, "scores_") and estimator.scores_ is not None:
        sc = estimator.scores_
        if isinstance(sc, np.ndarray) and sc.size > 0:
            return sc
    # For iterative DSS, we might not have a simple "score"
    if hasattr(estimator, "convergence_info_"):
        return None
    return None


def _get_components(estimator, data=None):
    """
    Get component time series.

    If 'sources_' (cached) is present (e.g. IterativeDSS), use it.
    Otherwise, compute from data using 'transform'.
    """
    # 1. Try cached sources if no new data provided
    if data is None:
        if hasattr(estimator, "sources_") and estimator.sources_ is not None:
            return estimator.sources_
        raise ValueError(
            "Data must be provided if sources are not cached in the estimator."
        )

    # 2. Compute sources for the provided data
    # Check if we need to force 'return_type' for LinearDSS
    if hasattr(estimator, "return_type"):
        old_return_type = estimator.return_type
        try:
            estimator.return_type = "sources"
            sources = estimator.transform(data)
        finally:
            estimator.return_type = old_return_type
    else:
        # Assumed to be IterativeDSS or similar which transforms to sources
        sources = estimator.transform(data)

    # 3. Standardize shape to (n_components, n_times, [n_epochs])
    # LinearDSS with MNE Epochs input returns (n_epochs, n_components, n_times)
    # We want dimension 0 to be components for easier plotting.

    if (
        isinstance(data, mne.BaseEpochs)
        and sources.ndim == 3
        and sources.shape[1] == _get_filters(estimator).shape[0]
    ):
        # Transpose to (n_comp, n_times, n_epochs)
        sources = np.transpose(sources, (1, 2, 0))

    return sources


def _get_n_selected(estimator):
    """Return number of components auto-selected (``n_selected_``)."""
    if hasattr(estimator, "n_selected_") and estimator.n_selected_ is not None:
        return estimator.n_selected_
    if hasattr(estimator, "n_removed_") and estimator.n_removed_ is not None:
        return estimator.n_removed_
    return 0


def _get_eigenvalues(estimator):
    """Eigenvalue vector, or ``None``."""
    return _get_scores(estimator)


def _get_segment_results(estimator):
    """Return ``segment_results_`` list-of-dict or ``[]``."""
    if hasattr(estimator, "segment_results_") and estimator.segment_results_:
        return estimator.segment_results_
    if hasattr(estimator, "adaptive_results_") and estimator.adaptive_results_:
        return estimator.adaptive_results_.get("chunk_info", [])
    return []


def _is_segmented(estimator):
    """Check whether the result is from segmented / adaptive mode."""
    if hasattr(estimator, "segmented") and estimator.segmented:
        return True
    return bool(_get_segment_results(estimator))


def _get_dss_removed(estimator):
    """Return removed artifact array (n_channels, n_times) or ``None``."""
    if hasattr(estimator, "removed") and estimator.removed is not None:
        return estimator.removed
    if hasattr(estimator, "removed_") and estimator.removed_ is not None:
        return estimator.removed_
    return None


def _get_bias_name(estimator):
    """Human-readable bias function name."""
    bias = getattr(estimator, "bias", None)
    if bias is None:
        return "Unknown"
    name = type(bias).__name__
    rename = {
        "BandpassBias": "Bandpass",
        "TrialAverageBias": "Trial Average",
        "SmoothingBias": "Smoothing",
        "PeakFilterBias": "Peak Filter",
        "CombFilterBias": "Comb Filter",
        "NotchBias": "Notch",
    }
    return rename.get(name, name)


def _get_zapline_n_removed(result):
    """Extract ``n_removed`` from a ZapLine result, handling both modes."""
    n_removed = 0
    if hasattr(result, "n_removed_") and result.n_removed_ is not None:
        n_removed = result.n_removed_
    elif hasattr(result, "n_removed") and result.n_removed is not None:
        n_removed = result.n_removed
    elif isinstance(result, dict):
        n_removed = result.get("n_removed", 0) or 0
    return n_removed


def _get_zapline_removed(result):
    """Extract removed data from a ZapLine result, handling both modes."""
    if hasattr(result, "removed") and result.removed is not None:
        return result.removed
    if hasattr(result, "adaptive_results_") and result.adaptive_results_ is not None:
        return result.adaptive_results_.get("removed")
    if isinstance(result, dict):
        return result.get("removed")
    return None


def _get_cleaned(result):
    """Extract cleaned data from a ZapLine result, handling both modes."""
    if hasattr(result, "cleaned") and result.cleaned is not None:
        return result.cleaned
    if hasattr(result, "adaptive_results_") and result.adaptive_results_ is not None:
        return result.adaptive_results_.get("cleaned")
    if isinstance(result, dict):
        return result.get("cleaned")
    return None


def _is_adaptive(result):
    """Check whether the result is from adaptive mode."""
    if hasattr(result, "adaptive") and result.adaptive:
        return True
    if hasattr(result, "adaptive_results_") and result.adaptive_results_ is not None:
        return True
    return bool(isinstance(result, dict) and "chunk_info" in result)


def _get_chunk_info(result):
    """Extract per-chunk info from an adaptive ZapLine result."""
    if hasattr(result, "adaptive_results_") and result.adaptive_results_ is not None:
        return result.adaptive_results_.get("chunk_info", [])
    if isinstance(result, dict):
        return result.get("chunk_info", [])
    return []


def _fallback_chunk_info(result):
    """Synthesize a single adaptive chunk when detailed chunk info is unavailable."""
    line_freq = getattr(result, "line_freq", None)
    removed = _get_zapline_removed(result)
    cleaned = _get_cleaned(result)
    n_times = 1
    for data in (removed, cleaned):
        if isinstance(data, np.ndarray) and data.ndim >= 2:
            n_times = data.shape[-1]
            break
    return [
        {
            "n_removed": _get_zapline_n_removed(result),
            "fine_freq": 0.0 if line_freq is None else line_freq,
            "frequency": 0.0 if line_freq is None else line_freq,
            "artifact_present": _get_zapline_n_removed(result) > 0,
            "start": 0,
            "end": n_times,
        }
    ]


def _handle_picks(info, picks=None):
    """Wrap mne.pick_types/pick_channels using public API."""
    if picks is None:
        return mne.pick_types(
            info, meg=True, eeg=True, seeg=True, ecog=True, fnirs=True, exclude="bads"
        )
    # Use public API for picking
    if isinstance(picks, str):
        if picks == "all":
            return np.arange(len(info["ch_names"]))
        # Use pick_types for string type specifiers
        return mne.pick_types(info, **{picks: True}, exclude="bads")
    elif isinstance(picks, list | np.ndarray):
        # Could be channel names or indices
        if len(picks) > 0 and isinstance(picks[0], str):
            return mne.pick_channels(info["ch_names"], include=picks)
        else:
            return np.asarray(picks)
    return np.asarray(picks)
