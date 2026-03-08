"""Spectral QA metric helpers for denoising comparison plots.

These private utilities compute per-channel or aggregate metrics used by
the DSS comparison dashboard (:func:`~mne_denoise.viz.dss.plot_dss_comparison`)
and potentially other benchmark visualisations.
"""

from __future__ import annotations

import numpy as np


def _suppression_ratio(f, psd_before, psd_after, freq, bw=2.0):
    """Suppression ratio (dB) at a single target frequency."""
    mask = (f >= freq - bw) & (f <= freq + bw)
    pb = psd_before.mean(0)[mask].mean()
    pa = psd_after.mean(0)[mask].mean()
    if pa <= 0:
        return np.inf
    return 10 * np.log10(pb / pa)


def _spectral_distortion(f, psd_before, psd_after, line_freq=50.0, n_harm=3, bw=2.0):
    """Spectral distortion (dB RMS) at non-harmonic frequencies."""
    safe = np.ones(len(f), dtype=bool)
    for k in range(1, n_harm + 1):
        safe &= ~((f >= line_freq * k - bw * 2) & (f <= line_freq * k + bw * 2))
    safe &= (f >= 2) & (f <= 160)
    if not safe.any():
        return 0.0
    ratio = psd_after.mean(0)[safe] / psd_before.mean(0)[safe]
    return np.sqrt(np.mean((10 * np.log10(ratio)) ** 2))


def _variance_removed(data_before, data_after):
    """Percentage of total variance removed."""
    return 100 * (1 - np.var(data_after) / np.var(data_before))
