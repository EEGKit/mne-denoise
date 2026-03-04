"""ICanClean: Reference-based artifact removal using CCA.

This module implements the ICanClean algorithm for removing motion and muscle
artifacts from EEG/MEG recordings by exploiting canonical correlations between
primary (scalp) channels and reference noise sensors (EOG, EMG, accelerometers,
or a second electrode layer).

The algorithm works by:
1. Dividing data into overlapping sliding windows
2. Computing CCA between primary and reference channels per window
3. Identifying components with high canonical correlation (likely artifacts)
4. Removing artifact-dominated components via least-squares projection
5. Reconstructing the cleaned signal using overlap-add

This module contains:
1. ``ICanClean``: Scikit-learn / MNE-Python compatible Transformer.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Downey, R. J., & Ferris, D. P. (2022). The iCanClean Algorithm:
       How to Remove Artifacts using Reference Noise Recordings.
       arXiv:2201.11798.

.. [2] Downey, R. J., & Ferris, D. P. (2023). iCanClean Removes Motion,
       Muscle, Eye, and Line-Noise Artifacts from Phantom EEG. Sensors,
       23(19), 8214. doi:10.3390/s23198214.

.. [3] Gonsisko, C. B., Ferris, D. P., & Downey, R. J. (2023). iCanClean
       Improves ICA of Mobile Brain Imaging with EEG. Sensors, 23(2), 928.
       doi:10.3390/s23020928.

.. [4] Nordin, A. D., Hairston, W. D., & Ferris, D. P. (2018).
       Dual-electrode motion artifact cancellation for mobile
       electroencephalography. J Neural Eng, 15(5), 056024.
       doi:10.1088/1741-2552/aad7d7.

.. [5] Hotelling, H. (1936). Relations between two sets of variates.
       Biometrika, 28(3/4), 321-377.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
from scipy import linalg as la
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import extract_data_from_mne, reconstruct_mne_object
from .cca import canonical_correlation

# Optional MNE support
try:
    import mne
    from mne.epochs import BaseEpochs
    from mne.evoked import Evoked
    from mne.io import BaseRaw

    _HAS_MNE = True
except ImportError:
    mne = None
    _HAS_MNE = False

logger = logging.getLogger(__name__)


class ICanClean(BaseEstimator, TransformerMixin):
    r"""ICanClean Transformer for reference-based artifact removal.

    Implements the ICanClean algorithm [1]_ using Canonical Correlation
    Analysis (CCA) between primary EEG/MEG channels and reference noise
    sensors to identify and remove artifact-dominated components.

    The cleaning pipeline for each sliding window:

    1. **CCA decomposition**: Compute canonical correlations between primary
       channels :math:`X` and reference channels :math:`Y`.
    2. **Component selection**: Identify components whose squared canonical
       correlation :math:`R^2` exceeds ``threshold``.
    3. **Artifact projection**: Subtract the least-squares projection of
       artifact-correlated canonical variates from :math:`X`.
    4. **Overlap-add**: Reconstruct the full time series from cleaned windows.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    ref_channels : list of str | None, default=None
        Names of reference noise channels. If ``None``, channels are detected
        automatically using ``ref_prefix``.
    primary_channels : list of str | None, default=None
        Names of primary (scalp) channels to clean. If ``None``, all channels
        not in ``ref_channels`` are used (or detected via ``primary_prefix``).
    ref_prefix : str | None, default=None
        Prefix to auto-detect reference channels (e.g. ``'2-'``, ``'EOG'``).
        Only used when ``ref_channels`` is ``None``.
    primary_prefix : str | None, default=None
        Prefix to auto-detect primary channels (e.g. ``'1-'``).
        Only used when ``primary_channels`` is ``None``.
    exclude_pattern : str | None, default=None
        Pattern to exclude channels from both layers (e.g. ``'EXG'``).
    segment_len : float, default=2.0
        Sliding window duration in seconds.
    overlap : float, default=0.5
        Overlap between consecutive windows as a fraction in [0, 1).
    threshold : float | 'auto', default=0.7
        :math:`R^2` threshold for component rejection.
        If ``'auto'``, uses an adaptive threshold based on the 95th percentile
        of the running correlation distribution.
    max_reject_fraction : float, default=0.5
        Safety cap: at most this fraction of canonical components can be
        removed per window.
    verbose : bool, default=True
        Whether to log progress information.

    Attributes
    ----------
    correlations_ : ndarray, shape (n_windows, d)
        Squared canonical correlations per window.
        ``d = min(rank(primary), rank(reference))``.
    n_removed_ : ndarray, shape (n_windows,)
        Number of components removed per window.
    removed_idx_ : list of ndarray
        Indices of rejected components per window.
    filters_ : list of ndarray
        CCA coefficient matrices ``A`` (primary) per window.
    patterns_ : list of ndarray
        CCA coefficient matrices ``B`` (reference) per window.
    n_windows_ : int
        Total number of windows processed.
    primary_channels_ : list of str
        Primary channel names used during cleaning.
    ref_channels_ : list of str
        Reference channel names used during cleaning.

    See Also
    --------
    mne_denoise.dss.DSS : Denoising Source Separation.
    mne_denoise.zapline.ZapLine : Line noise removal.
    mne_denoise.icanclean.cca.canonical_correlation : CCA utility.

    Notes
    -----
    When ``threshold='auto'``, the adaptive threshold is computed as the
    95th percentile of all :math:`R^2` values accumulated so far. For the
    first 10 windows (insufficient statistics), a conservative default of
    0.95 is used.

    The ``max_reject_fraction`` parameter prevents the algorithm from
    removing too many components in a single window, which would distort
    the signal. This is especially important for short windows or noisy
    reference sensors.

    Examples
    --------
    Basic usage with MNE Raw object (explicit reference channels):

    >>> from mne_denoise.icanclean import ICanClean
    >>> icanclean = ICanClean(
    ...     sfreq=raw.info["sfreq"],
    ...     ref_channels=["EOG1", "EOG2", "EMG1"],
    ... )
    >>> raw_clean = icanclean.fit_transform(raw)

    Dual-layer EEG with prefix-based detection:

    >>> icanclean = ICanClean(
    ...     sfreq=256.0,
    ...     primary_prefix="1-",
    ...     ref_prefix="2-",
    ...     exclude_pattern="EXG",
    ...     segment_len=2.0,
    ...     overlap=0.5,
    ...     threshold=0.85,
    ... )
    >>> raw_clean = icanclean.fit_transform(raw)
    >>> print(f"Removed {icanclean.n_removed_.mean():.1f} components on average")

    NumPy array interface:

    >>> import numpy as np
    >>> primary = np.random.randn(32, 5000)  # (n_primary, n_times)
    >>> reference = np.random.randn(4, 5000)  # (n_ref, n_times)
    >>> data = np.vstack([primary, reference])
    >>> icanclean = ICanClean(
    ...     sfreq=250.0,
    ...     ref_channels=list(range(32, 36)),  # last 4 channels
    ... )
    >>> cleaned = icanclean.fit_transform(data)

    References
    ----------
    .. [1] Downey, R. J., & Ferris, D. P. (2022). The iCanClean Algorithm:
           How to Remove Artifacts using Reference Noise Recordings.
           arXiv:2201.11798.

    .. [2] Downey, R. J., & Ferris, D. P. (2023). iCanClean Removes Motion,
           Muscle, Eye, and Line-Noise Artifacts from Phantom EEG. Sensors,
           23(19), 8214. doi:10.3390/s23198214.

    .. [3] Gonsisko, C. B., Ferris, D. P., & Downey, R. J. (2023). iCanClean
           Improves ICA of Mobile Brain Imaging with EEG. Sensors, 23(2), 928.
           doi:10.3390/s23020928.
    """

    def __init__(
        self,
        sfreq: float,
        ref_channels: list[str] | list[int] | None = None,
        primary_channels: list[str] | list[int] | None = None,
        ref_prefix: str | None = None,
        primary_prefix: str | None = None,
        exclude_pattern: str | None = None,
        segment_len: float = 2.0,
        overlap: float = 0.5,
        threshold: float | str = 0.7,
        max_reject_fraction: float = 0.5,
        verbose: bool = True,
    ):
        self.sfreq = float(sfreq)
        self.ref_channels = ref_channels
        self.primary_channels = primary_channels
        self.ref_prefix = ref_prefix
        self.primary_prefix = primary_prefix
        self.exclude_pattern = exclude_pattern
        self.segment_len = segment_len
        self.overlap = overlap
        self.threshold = threshold
        self.max_reject_fraction = max_reject_fraction
        self.verbose = verbose

    def fit(self, X: Any, y=None) -> "ICanClean":
        """Fit is a no-op; included for sklearn compatibility.

        The actual computation happens in :meth:`transform` since ICanClean
        operates on a sliding-window basis and does not learn a single global
        decomposition.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Input data (unused aside from validation).
        y : None
            Ignored.

        Returns
        -------
        self : ICanClean
        """
        return self

    def transform(self, X: Any, y=None) -> Any:
        """Apply ICanClean artifact removal.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Input data to clean. Accepted formats:

            - MNE ``Raw``: channels are resolved by name.
            - MNE ``Epochs``: each epoch is cleaned individually.
            - ndarray, shape ``(n_channels, n_times)``: channel indices in
              ``ref_channels`` / ``primary_channels`` are used directly.

        y : None
            Ignored.

        Returns
        -------
        X_clean : Raw | Epochs | ndarray
            Cleaned data in the same format as the input.
        """
        data, sfreq_data, mne_type, orig_inst = extract_data_from_mne(X)
        sfreq = sfreq_data if sfreq_data is not None else self.sfreq

        if mne_type == "epochs":
            # Epochs: (n_epochs, n_channels, n_times)
            cleaned_epochs = np.empty_like(data)
            for i in range(data.shape[0]):
                cleaned_epochs[i] = self._clean_continuous(
                    data[i], sfreq, orig_inst
                )
            return reconstruct_mne_object(
                cleaned_epochs, orig_inst, mne_type, verbose=False
            )
        else:
            # Raw / Evoked / array: (n_channels, n_times)
            cleaned = self._clean_continuous(data, sfreq, orig_inst)
            return reconstruct_mne_object(
                cleaned, orig_inst, mne_type, verbose=False
            )

    def fit_transform(self, X: Any, y=None, **fit_params) -> Any:
        """Fit and apply ICanClean in one step.

        Parameters
        ----------
        X : Raw | Epochs | ndarray
            Input data.
        y : None
            Ignored.
        **fit_params
            Ignored.

        Returns
        -------
        X_clean : Raw | Epochs | ndarray
            Cleaned data.
        """
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _resolve_channels(
        self, data: np.ndarray, orig_inst: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve primary and reference channel indices.

        Returns
        -------
        primary_idx : ndarray of int
        ref_idx : ndarray of int
        """
        n_channels = data.shape[0]

        # If MNE object, use channel names
        if _HAS_MNE and orig_inst is not None and hasattr(orig_inst, "ch_names"):
            ch_names = list(orig_inst.ch_names)
            return self._resolve_channels_by_name(ch_names)

        # Numpy array path: require explicit integer indices
        if self.ref_channels is not None:
            ref_idx = np.asarray(self.ref_channels, dtype=int)
        else:
            raise ValueError(
                "ref_channels must be provided (as integer indices) "
                "when using numpy arrays without an MNE object."
            )

        if self.primary_channels is not None:
            primary_idx = np.asarray(self.primary_channels, dtype=int)
        else:
            # Everything not in ref_channels
            all_idx = set(range(n_channels))
            primary_idx = np.array(
                sorted(all_idx - set(ref_idx.tolist())), dtype=int
            )

        return primary_idx, ref_idx

    def _resolve_channels_by_name(
        self, ch_names: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve channels by name (MNE path)."""

        def _should_exclude(name: str) -> bool:
            if self.exclude_pattern and self.exclude_pattern in name:
                return True
            return False

        # Reference channels
        if self.ref_channels is not None:
            ref_names = [
                ch for ch in self.ref_channels
                if ch in ch_names and not _should_exclude(ch)
            ]
        elif self.ref_prefix is not None:
            ref_names = [
                ch for ch in ch_names
                if ch.startswith(self.ref_prefix) and not _should_exclude(ch)
            ]
        else:
            raise ValueError(
                "Either ref_channels or ref_prefix must be provided "
                "to identify reference channels."
            )

        # Primary channels
        if self.primary_channels is not None:
            primary_names = [
                ch for ch in self.primary_channels
                if ch in ch_names and not _should_exclude(ch)
            ]
        elif self.primary_prefix is not None:
            primary_names = [
                ch for ch in ch_names
                if ch.startswith(self.primary_prefix) and not _should_exclude(ch)
            ]
        else:
            # Everything not in ref
            ref_set = set(ref_names)
            primary_names = [
                ch for ch in ch_names
                if ch not in ref_set and not _should_exclude(ch)
            ]

        if not ref_names:
            raise ValueError(
                f"No reference channels found. ref_channels={self.ref_channels}, "
                f"ref_prefix={self.ref_prefix!r}"
            )
        if not primary_names:
            raise ValueError(
                f"No primary channels found. primary_channels={self.primary_channels}, "
                f"primary_prefix={self.primary_prefix!r}"
            )

        # Store resolved names
        self.primary_channels_ = primary_names
        self.ref_channels_ = ref_names

        primary_idx = np.array([ch_names.index(ch) for ch in primary_names], dtype=int)
        ref_idx = np.array([ch_names.index(ch) for ch in ref_names], dtype=int)

        return primary_idx, ref_idx

    def _clean_continuous(
        self,
        data: np.ndarray,
        sfreq: float,
        orig_inst: Any,
    ) -> np.ndarray:
        """Core sliding-window CCA cleaning on a 2-D array.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
        sfreq : float
        orig_inst : MNE object or None

        Returns
        -------
        data_out : ndarray, shape (n_channels, n_times)
        """
        primary_idx, ref_idx = self._resolve_channels(data, orig_inst)

        data_primary = data[primary_idx, :]  # (n_primary, n_times)
        data_ref = data[ref_idx, :]          # (n_ref, n_times)

        n_times = data_primary.shape[1]
        n_primary = data_primary.shape[0]

        # Window parameters
        win_samples = int(self.segment_len * sfreq)
        step_samples = max(1, int(round(win_samples * (1 - self.overlap))))

        if win_samples > n_times:
            raise ValueError(
                f"Window length ({win_samples} samples = {self.segment_len}s) "
                f"exceeds data length ({n_times} samples)."
            )

        # Overlap-add buffers
        cleaned = np.zeros_like(data_primary)
        weights = np.zeros(n_times, dtype=np.float64)
        window_fn = np.ones(win_samples, dtype=np.float64)

        # Window start positions
        starts = np.arange(0, n_times - win_samples + 1, step_samples)

        # Per-window QC storage
        all_corr: list[np.ndarray] = []
        all_n_removed: list[int] = []
        all_removed_idx: list[np.ndarray] = []
        all_filters: list[np.ndarray] = []
        all_patterns: list[np.ndarray] = []

        # Running R\u00b2 accumulator for adaptive threshold
        running_r2: list[float] = []

        for start in starts:
            end = start + win_samples

            X_win = data_primary[:, start:end].T  # (samples, n_primary)
            Y_win = data_ref[:, start:end].T       # (samples, n_ref)

            try:
                A, B, R, U, V = canonical_correlation(X_win, Y_win)
            except Exception as exc:
                if self.verbose:
                    logger.warning("CCA failed for window at %d: %s", start, exc)
                cleaned[:, start:end] += data_primary[:, start:end] * window_fn
                weights[start:end] += window_fn
                all_corr.append(np.array([]))
                all_n_removed.append(0)
                all_removed_idx.append(np.array([], dtype=int))
                all_filters.append(np.array([]))
                all_patterns.append(np.array([]))
                continue

            r2 = (R ** 2).astype(np.float64)
            running_r2.extend(r2.tolist())

            # Determine threshold
            if self.threshold == "auto":
                thr = self._adaptive_threshold(running_r2)
            else:
                thr = float(self.threshold)

            # Identify artifact components (strict >)
            bad_mask = r2 > thr

            # Safety cap
            max_bad = max(1, int(self.max_reject_fraction * len(r2)))
            if bad_mask.sum() > max_bad:
                # Keep only the top-correlation ones
                order = np.argsort(r2)[::-1]
                bad_mask[:] = False
                bad_mask[order[:max_bad]] = True

            bad_idx = np.where(bad_mask)[0]

            all_corr.append(r2)
            all_n_removed.append(int(bad_idx.size))
            all_removed_idx.append(bad_idx)
            all_filters.append(A)
            all_patterns.append(B)

            if bad_idx.size > 0:
                # Least-squares projection removal (matches MATLAB cleanXwith='X')
                noise_sources = U[:, bad_idx]  # (samples, k)
                X_mc = X_win - X_win.mean(axis=0, keepdims=True)
                Z_mc = noise_sources - noise_sources.mean(axis=0, keepdims=True)

                beta, *_ = la.lstsq(Z_mc, X_mc, lapack_driver="gelsy")
                X_clean_win = X_win - Z_mc @ beta

                cleaned[:, start:end] += X_clean_win.T * window_fn
            else:
                cleaned[:, start:end] += X_win.T * window_fn

            weights[start:end] += window_fn

        # Normalise overlap-add
        mask = weights > 0
        cleaned[:, mask] /= weights[mask]

        # Handle edges with no coverage (shouldn't happen with rectangular window)
        if not mask.all():
            cleaned[:, ~mask] = data_primary[:, ~mask]

        # Store fitted attributes
        self.n_windows_ = len(starts)
        self.correlations_ = _pad_ragged(all_corr)  # (n_windows, d)
        self.n_removed_ = np.array(all_n_removed, dtype=int)
        self.removed_idx_ = all_removed_idx
        self.filters_ = all_filters
        self.patterns_ = all_patterns

        if self.verbose:
            total_removed = self.n_removed_.sum()
            pct_windows = (
                (self.n_removed_ > 0).sum() / self.n_windows_ * 100
                if self.n_windows_ > 0
                else 0
            )
            logger.info(
                "ICanClean: %d windows, %.1f%% had removals, "
                "%.1f components removed on average",
                self.n_windows_,
                pct_windows,
                total_removed / max(self.n_windows_, 1),
            )

        # Write cleaned data back
        data_out = data.copy()
        data_out[primary_idx, :] = cleaned
        return data_out

    @staticmethod
    def _adaptive_threshold(running_r2: list[float]) -> float:
        """Compute adaptive R\u00b2 threshold from running distribution."""
        if len(running_r2) > 10:
            return float(np.percentile(running_r2, 95))
        return 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_ragged(arrays: list[np.ndarray]) -> np.ndarray:
    """Stack a list of possibly different-length 1-D arrays into a 2-D array.

    Shorter rows are padded with NaN. Returns shape ``(n_rows, max_len)``.
    """
    if not arrays or all(a.size == 0 for a in arrays):
        return np.empty((len(arrays), 0), dtype=np.float64)
    max_len = max(a.size for a in arrays)
    out = np.full((len(arrays), max_len), np.nan, dtype=np.float64)
    for i, a in enumerate(arrays):
        if a.size > 0:
            out[i, : a.size] = a
    return out
