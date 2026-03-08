"""Spectral QA metrics for line-noise removal benchmarks.

This module provides both low-level metrics that operate on pre-computed
PSD arrays as well as high-level convenience helpers that accept
:class:`~mne.io.BaseRaw` objects directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mne

_EPS = 1e-30  # floor to avoid log(0)

__all__ = [
    "peak_attenuation_db",
    "noise_surround_ratio",
    "below_noise_distortion_db",
    "overclean_proportion",
    "underclean_proportion",
    "geometric_mean_psd_ratio",
    "geometric_mean_psd",
    "compute_all_qa_metrics",
]


# ──────────────────────────────────────────────────────────────────────
# Low-level metrics (operate on pre-computed PSD arrays)
# ──────────────────────────────────────────────────────────────────────


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
        return np.nan if psd_before.ndim == 1 else np.full(psd_before.shape[0], np.nan)
    if psd_before.ndim == 1:
        peak_before = psd_before[mask].max()
        peak_after = psd_after[mask].max()
    else:
        peak_before = psd_before[:, mask].max(axis=1)
        peak_after = psd_after[:, mask].max(axis=1)
    return 10.0 * np.log10(peak_before / np.maximum(peak_after, _EPS))


def noise_surround_ratio(
    freqs: np.ndarray,
    psd_after: np.ndarray,
    target_freq: float,
    peak_bw: float = 2.0,
    surround_bw: float = 5.0,
) -> np.ndarray:
    """Ratio of residual peak power to surrounding spectral floor.

    Values near 1 indicate the peak has been fully removed; values > 1
    indicate a residual peak remains.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_after : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD after cleaning.
    target_freq : float
        Centre frequency of the line-noise peak (Hz).
    peak_bw : float
        Half-bandwidth (Hz) of the peak region.
    surround_bw : float
        Half-bandwidth (Hz) of the surrounding region (measured from the
        outer edge of *peak_bw*).

    Returns
    -------
    ratio : array
        Per-channel ratio.  Shape ``(n_channels,)`` for 2-D input,
        scalar for 1-D input.
    """
    peak_mask = (freqs >= target_freq - peak_bw) & (freqs <= target_freq + peak_bw)
    surr_mask = (
        (freqs >= target_freq - surround_bw) & (freqs < target_freq - peak_bw)
    ) | ((freqs > target_freq + peak_bw) & (freqs <= target_freq + surround_bw))

    if psd_after.ndim == 1:
        peak_power = psd_after[peak_mask].mean() if peak_mask.any() else 0.0
        surr_power = psd_after[surr_mask].mean() if surr_mask.any() else _EPS
        return peak_power / max(surr_power, _EPS)

    peak_power = (
        psd_after[:, peak_mask].mean(axis=1)
        if peak_mask.any()
        else np.zeros(psd_after.shape[0])
    )
    surr_power = (
        psd_after[:, surr_mask].mean(axis=1)
        if surr_mask.any()
        else np.full(psd_after.shape[0], _EPS)
    )
    return peak_power / np.maximum(surr_power, _EPS)


def below_noise_distortion_db(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    exclude_freq: float | None = None,
    exclude_bw: float = 5.0,
    fmin: float = 1.0,
    fmax: float = 45.0,
    n_harmonics: int = 0,
) -> np.ndarray:
    """Broadband spectral distortion (dB) outside the line-noise band.

    Computed as the mean of ``|10 * log10(psd_after / psd_before)|``
    across non-target frequencies.  Lower values indicate less
    collateral distortion.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_before : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD before cleaning.
    psd_after : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD after cleaning.
    exclude_freq : float | None
        Fundamental line-noise frequency to exclude (together with its
        harmonics).  If ``None`` no exclusion is applied.
    exclude_bw : float
        Half-bandwidth (Hz) to exclude around each harmonic.
    fmin, fmax : float
        Frequency range for the broadband comparison.
    n_harmonics : int
        Number of harmonics of *exclude_freq* to also exclude
        (0 = fundamental only).

    Returns
    -------
    distortion : array
        Per-channel (or scalar) distortion in dB.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if exclude_freq is not None:
        for h in range(1, n_harmonics + 2):
            hf = exclude_freq * h
            mask &= ~((freqs >= hf - exclude_bw) & (freqs <= hf + exclude_bw))
    if not mask.any():
        return 0.0 if psd_before.ndim == 1 else np.zeros(psd_before.shape[0])
    if psd_before.ndim == 1:
        ratio = np.log10(psd_after[mask] / np.maximum(psd_before[mask], _EPS))
        return float(np.mean(np.abs(ratio)) * 10.0)
    ratio = np.log10(psd_after[:, mask] / np.maximum(psd_before[:, mask], _EPS))
    return np.mean(np.abs(ratio), axis=1) * 10.0


def overclean_proportion(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    target_freq: float,
    bandwidth: float = 2.0,
    threshold_db: float = 3.0,
) -> float:
    """Fraction of channels where the spectral floor is over-suppressed.

    The spectral floor is defined as the region immediately surrounding
    the line-noise peak.  If it is attenuated by more than *threshold_db*
    that channel is counted as over-cleaned.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_before : array of shape (n_channels, n_freqs)
        PSD before cleaning.
    psd_after : array of shape (n_channels, n_freqs)
        PSD after cleaning.
    target_freq : float
        Centre frequency (Hz) of the line-noise peak.
    bandwidth : float
        Half-bandwidth (Hz) used for peak identification.  The surround
        region spans ``[target - 2*bw, target - bw]`` and
        ``[target + bw, target + 2*bw]``.
    threshold_db : float
        Attenuation threshold in dB above which a channel is considered
        over-cleaned.

    Returns
    -------
    proportion : float
        Value in ``[0, 1]``.
    """
    surr_mask = (
        (freqs >= target_freq - bandwidth * 2) & (freqs < target_freq - bandwidth)
    ) | ((freqs > target_freq + bandwidth) & (freqs <= target_freq + bandwidth * 2))
    if not surr_mask.any():
        return 0.0
    if psd_before.ndim == 1:
        floor_before = psd_before[surr_mask].mean()
        floor_after = psd_after[surr_mask].mean()
        atten_db = 10.0 * np.log10(floor_before / max(floor_after, _EPS))
        return float(atten_db > threshold_db)
    floor_before = psd_before[:, surr_mask].mean(axis=1)
    floor_after = psd_after[:, surr_mask].mean(axis=1)
    atten_db = 10.0 * np.log10(floor_before / np.maximum(floor_after, _EPS))
    return float((atten_db > threshold_db).mean())


def underclean_proportion(
    freqs: np.ndarray,
    psd_after: np.ndarray,
    target_freq: float,
    peak_bw: float = 2.0,
    surround_bw: float = 5.0,
    threshold_ratio: float = 2.0,
) -> float:
    """Fraction of channels where the line-noise peak remains prominent.

    A channel is counted as under-cleaned when its
    :func:`noise_surround_ratio` exceeds *threshold_ratio*.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_after : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD after cleaning.
    target_freq : float
        Centre frequency (Hz) of the line-noise peak.
    peak_bw : float
        Half-bandwidth of the peak region.
    surround_bw : float
        Half-bandwidth of the surrounding region.
    threshold_ratio : float
        Ratio above which a channel is considered under-cleaned.

    Returns
    -------
    proportion : float
        Value in ``[0, 1]``.
    """
    nsr = noise_surround_ratio(freqs, psd_after, target_freq, peak_bw, surround_bw)
    if np.ndim(nsr) == 0:
        return float(nsr > threshold_ratio)
    return float((nsr > threshold_ratio).mean())


def geometric_mean_psd_ratio(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    fmin: float = 1.0,
    fmax: float = 45.0,
) -> np.ndarray:
    """Geometric mean of ``psd_after / psd_before`` across broadband.

    Values near 1 indicate minimal broadband change; values < 1
    indicate net power reduction.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_before : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD before cleaning.
    psd_after : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD after cleaning.
    fmin, fmax : float
        Frequency range.

    Returns
    -------
    gm_ratio : array
        Per-channel geometric-mean ratio.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return 1.0 if psd_before.ndim == 1 else np.ones(psd_before.shape[0])
    if psd_before.ndim == 1:
        ratio = psd_after[mask] / np.maximum(psd_before[mask], _EPS)
        return float(np.exp(np.mean(np.log(np.maximum(ratio, _EPS)))))
    ratio = psd_after[:, mask] / np.maximum(psd_before[:, mask], _EPS)
    return np.exp(np.mean(np.log(np.maximum(ratio, _EPS)), axis=1))


# ──────────────────────────────────────────────────────────────────────
# High-level helpers (operate on Raw objects)
# ──────────────────────────────────────────────────────────────────────


def _compute_psd_pair(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    fmax: float = 125.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(freqs, psd_before, psd_after)`` from two Raw objects."""
    psd_b = raw_before.compute_psd(fmax=fmax, verbose=False)
    psd_a = raw_after.compute_psd(fmax=fmax, verbose=False)
    return psd_b.freqs, psd_b.get_data(), psd_a.get_data()


def geometric_mean_psd(
    raw: mne.io.BaseRaw,
    fmax: float = 125.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Geometric-mean PSD across channels of a Raw object.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The Raw object.
    fmax : float
        Maximum frequency for the PSD computation.

    Returns
    -------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    gm_psd : array of shape (n_freqs,)
        Geometric mean of the PSD across channels.
    """
    psd = raw.compute_psd(fmax=fmax, verbose=False)
    data = psd.get_data()  # (n_channels, n_freqs)
    gm = np.exp(np.mean(np.log(np.maximum(data, _EPS)), axis=0))
    return psd.freqs, gm


def compute_all_qa_metrics(
    raw_before: mne.io.BaseRaw,
    raw_after: mne.io.BaseRaw,
    line_freq: float = 50.0,
    n_harmonics: int = 0,
    fmax: float = 125.0,
) -> dict:
    """Compute all QA metrics for a line-noise removal benchmark.

    This is a convenience wrapper that calls the individual metric
    functions and returns their results as a flat dictionary.

    Parameters
    ----------
    raw_before : mne.io.BaseRaw
        Raw recording before cleaning.
    raw_after : mne.io.BaseRaw
        Raw recording after cleaning.
    line_freq : float
        Fundamental line-noise frequency (Hz).
    n_harmonics : int
        Number of harmonics above the fundamental to evaluate
        (0 = fundamental only).
    fmax : float
        Maximum frequency for PSD computation.

    Returns
    -------
    metrics : dict
        Dictionary with keys:

        * ``peak_attenuation_db`` – median across channels (float).
        * ``R_f0`` – median noise-surround ratio (float).
        * ``below_noise_distortion_db`` – median broadband distortion
          (float).
        * ``overclean_proportion`` – fraction of over-cleaned channels
          (float).
        * ``underclean_proportion`` – fraction of under-cleaned channels
          (float).
        * ``geometric_mean_psd_ratio`` – median broadband GM ratio
          (float).
        * ``harmonics_hz`` – list of harmonic frequencies evaluated.
        * ``per_harmonic_attenuation_db`` – list of per-harmonic median
          attenuations.
        * ``per_harmonic_R`` – list of per-harmonic median
          noise-surround ratios.
    """
    freqs, psd_b, psd_a = _compute_psd_pair(raw_before, raw_after, fmax=fmax)

    harmonics = [line_freq * h for h in range(1, n_harmonics + 2)]

    # Per-harmonic metrics
    per_h_atten: list[float] = []
    per_h_r: list[float] = []
    for hf in harmonics:
        atten = peak_attenuation_db(freqs, psd_b, psd_a, hf)
        nsr = noise_surround_ratio(freqs, psd_a, hf)
        per_h_atten.append(float(np.nanmedian(atten)))
        per_h_r.append(float(np.nanmedian(nsr)))

    # Broadband metrics
    distort = below_noise_distortion_db(
        freqs,
        psd_b,
        psd_a,
        exclude_freq=line_freq,
        n_harmonics=n_harmonics,
    )
    oc = overclean_proportion(freqs, psd_b, psd_a, line_freq)
    uc = underclean_proportion(freqs, psd_a, line_freq)
    gmr = geometric_mean_psd_ratio(freqs, psd_b, psd_a)

    return {
        "peak_attenuation_db": per_h_atten[0],
        "R_f0": per_h_r[0],
        "below_noise_distortion_db": float(np.median(distort)),
        "below_noise_pct": float(np.median(distort)),  # alias
        "overclean_proportion": float(oc),
        "underclean_proportion": float(uc),
        "geometric_mean_psd_ratio": float(np.median(gmr)),
        "harmonics_hz": harmonics,
        "per_harmonic_attenuation_db": per_h_atten,
        "per_harmonic_R": per_h_r,
    }
