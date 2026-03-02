"""Spectral QA metrics for line-noise removal benchmarks."""

from __future__ import annotations

import numpy as np


def peak_attenuation_db(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    target_freq: float,
    bandwidth: float = 2.0,
) -> np.ndarray:
    """Attenuation (dB) of the spectral peak at *target_freq*.

    Positive values indicate the peak was reduced.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_before : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD before cleaning.
    psd_after : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD after cleaning.
    target_freq : float
        Centre frequency of the peak (Hz).
    bandwidth : float
        Half-bandwidth (Hz) around *target_freq* to search for the peak.

    Returns
    -------
    attenuation : array
        Per-channel (or scalar) attenuation in dB.
    """
    mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
    if not mask.any():
        # Target frequency outside the frequency range (e.g. above Nyquist)
        return np.nan if psd_before.ndim == 1 else np.full(psd_before.shape[0], np.nan)
    if psd_before.ndim == 1:
        peak_before = psd_before[mask].max()
        peak_after = psd_after[mask].max()
    else:
        peak_before = psd_before[:, mask].max(axis=1)
        peak_after = psd_after[:, mask].max(axis=1)
    return 10.0 * np.log10(peak_before / np.maximum(peak_after, 1e-30))


# TODO: Implement and export the remaining QA metrics used in the
# benchmark scripts.  The following are referenced in
# ``scripts/run_line_noise_benchmark.py`` but not yet shipped:
#
#   - noise_surround_ratio (R_f0)
#   - below_noise_distortion (below_noise_pct)
#   - overclean_proportion
#   - underclean_proportion
#   - geometric_mean_psd_ratio
#   - compute_all_qa_metrics (convenience wrapper)
