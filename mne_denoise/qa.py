"""Quality assurance metrics for denoising evaluation.

This module contains:
1. `peak_attenuation_db`: Measure how strongly a narrow-band spectral peak is
   reduced after denoising.
2. `suppression_ratio`: Ratio (dB) of power before vs after at a target frequency.
3. `spectral_distortion`: RMS distortion (dB) at non-harmonic frequencies.
4. `variance_removed`: Percentage of total variance removed.

The QA layer is intended to hold reusable scalar or vector metrics that
quantify artifact suppression and signal preservation independently of any
plotting code.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

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

    This metric compares the strongest spectral value in a narrow frequency
    window before and after cleaning. A search mask is built around
    ``target_freq`` using ``bandwidth``, the maximum PSD value is extracted
    from that window in both inputs, and attenuation is reported as:

    .. math:: 10 \\log_{10}(P_{before} / P_{after})

    Positive values indicate the peak was reduced, values near zero indicate
    little change, and negative values indicate the peak increased after
    cleaning.

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

    Notes
    -----
    For 1D input, a scalar attenuation value is returned. For 2D input, the
    metric is computed independently for each channel and returned as a
    1D array of shape ``(n_channels,)``.

    If the target window lies completely outside the sampled frequency range,
    the function returns ``np.nan`` for 1D input or an all-``np.nan`` array
    for 2D input.

    Examples
    --------
    Estimate attenuation for a single PSD:

    >>> import numpy as np
    >>> from mne_denoise.qa import peak_attenuation_db
    >>> freqs = np.arange(0, 100, 0.5)
    >>> psd_before = np.ones_like(freqs) * 0.01
    >>> psd_after = np.ones_like(freqs) * 0.01
    >>> mask = (freqs >= 49) & (freqs <= 51)
    >>> psd_before[mask] = 1.0
    >>> psd_after[mask] = 0.25
    >>> peak_attenuation_db(freqs, psd_before, psd_after, 50.0)
    6.020...

    Compute the metric channel-wise for multi-channel PSDs:

    >>> psd_before_2d = np.vstack([psd_before, psd_before])
    >>> psd_after_2d = np.vstack([psd_after, psd_after])
    >>> peak_attenuation_db(freqs, psd_before_2d, psd_after_2d, 50.0).shape
    (2,)
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


def suppression_ratio(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    target_freq: float,
    bandwidth: float = 2.0,
) -> float:
    """Suppression ratio (dB) at a single target frequency.

    This metric summarizes how much power was reduced within a frequency
    window centered on ``target_freq``. Unlike :func:`peak_attenuation_db`,
    which compares the strongest peak in the window, this function first
    averages PSD values across channels (for 2D input) and then computes the
    mean power inside the selected band before and after cleaning:

    .. math:: 10 \\log_{10}(\\bar{P}_{before} / \\bar{P}_{after})

    Positive values indicate attenuation of the target band, values near zero
    indicate little change, and ``np.inf`` indicates complete suppression of
    the band under the current numerical precision.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_before : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD before cleaning.
    psd_after : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSD after cleaning.
    target_freq : float
        Target frequency (Hz).
    bandwidth : float
        Half-bandwidth (Hz) around target_freq.

    Returns
    -------
    ratio : float
        Suppression ratio in dB.

    Notes
    -----
    For 2D PSD input, channels are averaged before the band-power ratio is
    computed, so the result is always a single scalar.

    If the requested target window lies completely outside the sampled
    frequency range, the function returns ``np.nan``.

    Examples
    --------
    Estimate suppression around a 50 Hz band:

    >>> import numpy as np
    >>> from mne_denoise.qa import suppression_ratio
    >>> freqs = np.arange(0, 100, 0.5)
    >>> psd_before = np.ones((2, len(freqs))) * 0.01
    >>> psd_after = np.ones((2, len(freqs))) * 0.01
    >>> mask = (freqs >= 49) & (freqs <= 51)
    >>> psd_before[:, mask] = 1.0
    >>> psd_after[:, mask] = 0.5
    >>> suppression_ratio(freqs, psd_before, psd_after, 50.0)
    3.010...
    """
    mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
    if not mask.any():
        return np.nan

    # Average across channels if 2D
    pb = psd_before.mean(axis=0) if psd_before.ndim == 2 else psd_before
    pa = psd_after.mean(axis=0) if psd_after.ndim == 2 else psd_after

    pb_mean = pb[mask].mean()
    pa_mean = pa[mask].mean()

    if pa_mean <= 0:
        return np.inf
    return 10.0 * np.log10(pb_mean / pa_mean)


def spectral_distortion(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    line_freq: float = 50.0,
    n_harmonics: int = 3,
    bandwidth: float = 2.0,
) -> float:
    """Spectral distortion (dB RMS) at non-harmonic frequencies.

    This measures how much the cleaning process changed the spectrum
    outside of the target line-noise frequencies. Harmonic neighborhoods are
    excluded, channels are averaged if the PSD input is 2D, and the remaining
    spectrum is summarized with the RMS of the log-power ratio:

    .. math:: \\sqrt{\\mathrm{mean}\\left(\\left(10 \\log_{10}(P_{after}/P_{before})\\right)^2\\right)}

    Values near zero indicate good spectral preservation outside the artifact
    bands, while larger values indicate stronger off-target spectral changes.

    Parameters
    ----------
    freqs : array of shape (n_freqs,)
        Frequency vector.
    psd_before, psd_after : array of shape (n_channels, n_freqs) or (n_freqs,)
        PSDs before and after cleaning.
    line_freq : float
        Fundamental line frequency (Hz).
    n_harmonics : int
        Number of harmonics to exclude.
    bandwidth : float
        Base exclusion bandwidth (Hz) around each harmonic. The current
        implementation excludes an interval of ``harmonic ± 2 * bandwidth``.

    Returns
    -------
    distortion : float
        RMS distortion in dB.

    Notes
    -----
    Evaluation is further restricted to the range 2 to 160 Hz, matching the
    current implementation used by the DSS visualization diagnostics.

    If no frequency bins remain after harmonic exclusion and range filtering,
    the function returns ``0.0``.

    Examples
    --------
    Identical spectra outside the excluded harmonic regions yield zero
    distortion:

    >>> import numpy as np
    >>> from mne_denoise.qa import spectral_distortion
    >>> freqs = np.arange(0, 200, 0.5)
    >>> psd = np.ones((2, len(freqs)))
    >>> spectral_distortion(freqs, psd, psd, line_freq=50.0, n_harmonics=3)
    0.0
    """
    safe = np.ones(len(freqs), dtype=bool)
    for k in range(1, n_harmonics + 1):
        target = line_freq * k
        safe &= ~((freqs >= target - bandwidth * 2) & (freqs <= target + bandwidth * 2))

    # Restrict to a reasonable range for evaluation
    safe &= (freqs >= 2) & (freqs <= 160)

    if not safe.any():
        return 0.0

    pb = psd_before.mean(axis=0) if psd_before.ndim == 2 else psd_before
    pa = psd_after.mean(axis=0) if psd_after.ndim == 2 else psd_after

    ratio = pa[safe] / np.maximum(pb[safe], 1e-30)
    return np.sqrt(np.mean((10.0 * np.log10(ratio)) ** 2))


def variance_removed(data_before: np.ndarray, data_after: np.ndarray) -> float:
    """Percentage of total variance removed.

    This metric compares the total variance of the cleaned data to the total
    variance of the original data:

    .. math:: 100 \\times \\left(1 - \\mathrm{var}(X_{after}) / \\mathrm{var}(X_{before})\\right)

    Values near ``0`` indicate little change in overall variance, ``100``
    indicates complete removal of variance, and negative values indicate that
    the cleaned data has higher variance than the original.

    Parameters
    ----------
    data_before, data_after : array
        Data before and after cleaning.

    Returns
    -------
    pct : float
        Percentage of variance removed.

    Notes
    -----
    If ``data_before`` has zero variance, the function returns ``0.0`` to
    avoid division by zero and to indicate that no meaningful percentage
    removal can be computed.

    Examples
    --------
    Halving signal amplitude leaves one quarter of the original variance, so
    75% of the variance is removed:

    >>> import numpy as np
    >>> from mne_denoise.qa import variance_removed
    >>> x = np.array([1.0, -1.0, 1.0, -1.0])
    >>> variance_removed(x, 0.5 * x)
    75.0
    """
    var_before = np.var(data_before)
    if var_before == 0:
        return 0.0
    return 100.0 * (1.0 - np.var(data_after) / var_before)

# TODO: Implement and export the remaining QA metrics used in the
# benchmark scripts. The following are referenced in
# ``scripts/run_line_noise_benchmark.py`` but not yet shipped:
#
#   - noise_surround_ratio (R_f0)
#   - below_noise_distortion (below_noise_pct)
#   - overclean_proportion
#   - underclean_proportion
#   - geometric_mean_psd_ratio
#   - compute_all_qa_metrics (convenience wrapper)
