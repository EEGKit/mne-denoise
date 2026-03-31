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
        Sliding window duration in seconds (the "clean window").
    overlap : float, default=0.0
        Overlap between consecutive windows as a fraction in [0, 1).
        Default 0.0 matches the MATLAB reference (non-overlapping windows).
    threshold : float | 'auto', default=0.7
        :math:`R^2` threshold for component rejection.
        If ``'auto'``, uses an adaptive threshold based on the 95th percentile
        of the running correlation distribution.
    max_reject_fraction : float, default=0.5
        Safety cap: at most this fraction of canonical components can be
        removed per window.
    reref_primary : bool | str, default=False
        Apply average re-referencing to primary channels *for CCA only*
        (the original data is used for cleaning). Matches MATLAB's
        ``rerefX='yes-temp'``. Accepts ``True`` / ``'fullrank'`` (preserves
        rank, divides by n+1) or ``'loserank'`` (standard avg ref, loses
        1 rank).
    reref_ref : bool | str, default=False
        Same as ``reref_primary`` but for reference channels.
    stats_segment_len : float | None, default=None
        Duration (seconds) of the broader "stats window" for CCA
        computation. If ``None`` or equal to ``segment_len``, the same
        window is used for CCA and cleaning. When larger, CCA is computed
        on the broader window but only the inner ``segment_len`` portion
        is cleaned — matching MATLAB's ``statsWindow`` / ``cleanWindow``
        distinction.
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
        mode: str = "sliding",
        clean_with: str = "X",
        segment_len: float = 2.0,
        overlap: float = 0.0,
        threshold: float | str = 0.7,
        max_reject_fraction: float = 0.5,
        reref_primary: bool | str = False,
        reref_ref: bool | str = False,
        stats_segment_len: float | None = None,
        filter_ref: tuple | None = None,
        pseudo_ref: bool = False,
        global_threshold: float | str | None = None,
        global_clean_with: str | None = None,
        global_max_reject_fraction: float | None = None,
        verbose: bool = True,
    ):
        if mode not in ("sliding", "global", "hybrid", "calibrated"):
            raise ValueError(
                f"mode must be 'sliding', 'global', 'hybrid', or "
                f"'calibrated', got {mode!r}"
            )
        if clean_with not in ("X", "Y", "both"):
            raise ValueError(
                f"clean_with must be 'X', 'Y', or 'both', got {clean_with!r}"
            )
        if not (0 <= overlap < 1):
            raise ValueError(
                f"overlap must be in [0, 1), got {overlap}"
            )
        if not (0 <= max_reject_fraction <= 1):
            raise ValueError(
                f"max_reject_fraction must be in [0, 1], got {max_reject_fraction}"
            )

        self.sfreq = float(sfreq)
        self.ref_channels = ref_channels
        self.primary_channels = primary_channels
        self.ref_prefix = ref_prefix
        self.primary_prefix = primary_prefix
        self.exclude_pattern = exclude_pattern
        self.mode = mode
        self.clean_with = clean_with
        self.segment_len = segment_len
        self.overlap = overlap
        self.threshold = threshold
        self.max_reject_fraction = max_reject_fraction
        self.reref_primary = reref_primary
        self.reref_ref = reref_ref
        self.stats_segment_len = stats_segment_len
        self.filter_ref = filter_ref
        self.pseudo_ref = pseudo_ref
        self.global_threshold = global_threshold
        self.global_clean_with = global_clean_with
        self.global_max_reject_fraction = global_max_reject_fraction
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
            # Accumulate QC across epochs instead of overwriting
            epoch_corrs: list[np.ndarray] = []
            epoch_n_removed: list[int] = []
            epoch_removed_idx: list[np.ndarray] = []
            epoch_filters: list[np.ndarray] = []
            epoch_patterns: list[np.ndarray] = []
            epoch_window_counts: list[int] = []
            # For hybrid mode, also accumulate global QC
            epoch_global_corrs: list[np.ndarray] = []
            epoch_global_n_removed: list[int] = []
            epoch_global_removed_idx: list[np.ndarray] = []

            for i in range(data.shape[0]):
                cleaned_epochs[i] = self._clean_continuous(
                    data[i], sfreq, orig_inst
                )
                # Collect per-epoch QC (top-level = sliding pass in hybrid)
                epoch_corrs.extend(
                    [self.correlations_[j] for j in range(self.n_windows_)]
                )
                epoch_n_removed.extend(self.n_removed_.tolist())
                epoch_removed_idx.extend(self.removed_idx_)
                epoch_filters.extend(self.filters_)
                epoch_patterns.extend(self.patterns_)
                epoch_window_counts.append(self.n_windows_)
                # Collect hybrid global QC if present
                if hasattr(self, "global_correlations_"):
                    epoch_global_corrs.extend(
                        [self.global_correlations_[j]
                         for j in range(self.global_correlations_.shape[0])]
                    )
                    epoch_global_n_removed.extend(
                        self.global_n_removed_.tolist()
                    )
                    epoch_global_removed_idx.extend(
                        self.global_removed_idx_
                    )

            # Aggregate QC across all epochs
            self.correlations_ = _pad_ragged(epoch_corrs)
            self.n_removed_ = np.array(epoch_n_removed, dtype=int)
            self.removed_idx_ = epoch_removed_idx
            self.filters_ = epoch_filters
            self.patterns_ = epoch_patterns
            self.n_windows_ = len(epoch_corrs)
            self.epoch_window_counts_ = epoch_window_counts
            # Aggregate hybrid global QC
            if epoch_global_corrs:
                self.global_correlations_ = _pad_ragged(epoch_global_corrs)
                self.global_n_removed_ = np.array(
                    epoch_global_n_removed, dtype=int
                )
                self.global_removed_idx_ = epoch_global_removed_idx

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
        """Orchestrate CCA cleaning based on mode.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
        sfreq : float
        orig_inst : MNE object or None

        Returns
        -------
        data_out : ndarray, shape (n_channels, n_times)
        """
        # Pseudo-reference mode: use primary channels as both X and Y
        # for CCA. The Y copy gets filtered (e.g. notch out brain band).
        n_orig = data.shape[0]
        if self.pseudo_ref:
            # Resolve primary channels; ref = same channels.
            # Accept either primary_channels or ref_channels as input.
            try:
                primary_idx, _ = self._resolve_channels(data, orig_inst)
            except ValueError:
                primary_idx = np.array([], dtype=int)
            if primary_idx.size == 0:
                try:
                    _, primary_idx = self._resolve_channels(data, orig_inst)
                except ValueError:
                    pass
            if primary_idx.size == 0:
                # Last resort: use all channels
                primary_idx = np.arange(data.shape[0])
            # Create filtered pseudo-reference from primary data
            pseudo_data = data[primary_idx, :].copy()
            pseudo_data = _filter_channels(pseudo_data, self.filter_ref,
                                           sfreq)
            data = np.vstack([data, pseudo_data])
            ref_idx = np.arange(n_orig, data.shape[0])
        else:
            primary_idx, ref_idx = self._resolve_channels(data, orig_inst)

        if self.filter_ref is not None and not self.pseudo_ref:
            # Non-pseudo: filter ref channels in a copy (ref != primary)
            data = data.copy()
            data[ref_idx, :] = _filter_channels(
                data[ref_idx, :], self.filter_ref, sfreq
            )

        if self.mode == "global":
            data_out = self._run_single_pass(
                data, primary_idx, ref_idx, sfreq,
                threshold=self.threshold,
                clean_with=self.clean_with,
                max_reject_fraction=self.max_reject_fraction,
                use_windows=False,
            )
        elif self.mode == "sliding":
            data_out = self._run_single_pass(
                data, primary_idx, ref_idx, sfreq,
                threshold=self.threshold,
                clean_with=self.clean_with,
                max_reject_fraction=self.max_reject_fraction,
                use_windows=True,
            )
        elif self.mode == "hybrid":
            # Global pass first
            g_thr = self.global_threshold if self.global_threshold is not None else self.threshold
            g_cw = self.global_clean_with if self.global_clean_with is not None else self.clean_with
            g_mrf = self.global_max_reject_fraction if self.global_max_reject_fraction is not None else self.max_reject_fraction

            data_after_global = self._run_single_pass(
                data, primary_idx, ref_idx, sfreq,
                threshold=g_thr,
                clean_with=g_cw,
                max_reject_fraction=g_mrf,
                use_windows=False,
            )
            # Store global-pass QC
            self.global_correlations_ = self.correlations_
            self.global_n_removed_ = self.n_removed_
            self.global_removed_idx_ = self.removed_idx_
            self.global_filters_ = self.filters_
            self.global_patterns_ = self.patterns_

            # Sliding pass on globally-cleaned data
            data_out = self._run_single_pass(
                data_after_global, primary_idx, ref_idx, sfreq,
                threshold=self.threshold,
                clean_with=self.clean_with,
                max_reject_fraction=self.max_reject_fraction,
                use_windows=True,
            )
            # Store sliding-pass QC (also becomes top-level)
            self.sliding_correlations_ = self.correlations_
            self.sliding_n_removed_ = self.n_removed_
            self.sliding_removed_idx_ = self.removed_idx_
            self.sliding_filters_ = self.filters_
            self.sliding_patterns_ = self.patterns_
        elif self.mode == "calibrated":
            data_out = self._run_calibrated_pass(
                data, primary_idx, ref_idx, sfreq,
                threshold=self.threshold,
                max_reject_fraction=self.max_reject_fraction,
            )
        else:
            raise ValueError(f"Unknown mode {self.mode!r}")

        # Strip pseudo-reference rows if they were appended
        if self.pseudo_ref and data_out.shape[0] > n_orig:
            data_out = data_out[:n_orig]

        return data_out

    def _run_single_pass(
        self,
        data: np.ndarray,
        primary_idx: np.ndarray,
        ref_idx: np.ndarray,
        sfreq: float,
        *,
        threshold: float | str,
        clean_with: str,
        max_reject_fraction: float,
        use_windows: bool,
    ) -> np.ndarray:
        """Run one CCA cleaning pass (either global or sliding windows).

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_times)
        primary_idx, ref_idx : ndarray of int
        sfreq : float
        threshold : float or 'auto'
        clean_with : 'X', 'Y', or 'both'
        max_reject_fraction : float
        use_windows : bool
            If True, use sliding windows. If False, single global pass.

        Returns
        -------
        data_out : ndarray, shape (n_channels, n_times)
        """
        data_primary = data[primary_idx, :]
        data_ref = data[ref_idx, :]
        n_times = data_primary.shape[1]

        if use_windows:
            win_samples = int(self.segment_len * sfreq)
            step_samples = max(1, int(round(win_samples * (1 - self.overlap))))

            if win_samples > n_times:
                raise ValueError(
                    f"Window length ({win_samples} samples = {self.segment_len}s) "
                    f"exceeds data length ({n_times} samples)."
                )

            starts = list(np.arange(0, n_times - win_samples + 1, step_samples))
            # Ensure terminal window covers the last sample
            last_possible = n_times - win_samples
            if starts and starts[-1] < last_possible:
                starts.append(last_possible)
        else:
            # Global: one window covering the whole recording
            win_samples = n_times
            starts = [0]

        # Overlap-add buffers
        cleaned = np.zeros_like(data_primary)
        weights = np.zeros(n_times, dtype=np.float64)
        window_fn = np.ones(win_samples, dtype=np.float64)

        # Per-window QC
        all_corr: list[np.ndarray] = []
        all_n_removed: list[int] = []
        all_removed_idx: list[np.ndarray] = []
        all_filters: list[np.ndarray] = []
        all_patterns: list[np.ndarray] = []
        running_r2: list[float] = []

        # Broader stats window support (Bug 6: MATLAB statsWindow)
        use_stats_win = (
            self.stats_segment_len is not None
            and self.stats_segment_len > self.segment_len
            and use_windows
        )
        if use_stats_win:
            stats_win_samples = int(self.stats_segment_len * sfreq)
        else:
            stats_win_samples = None

        for start in starts:
            end = start + win_samples
            if end > n_times:
                end = n_times

            actual_len = end - start
            wfn = window_fn[:actual_len]

            # -- Stats window: broader context for CCA (Bug 6) --
            if use_stats_win:
                extra = stats_win_samples - actual_len
                extra_pre = extra // 2
                extra_post = extra - extra_pre
                s_start = start - extra_pre
                s_end = end + extra_post
                # Clamp to data boundaries, redistribute excess
                if s_start < 0:
                    s_end = min(n_times, s_end - s_start)
                    s_start = 0
                if s_end > n_times:
                    s_start = max(0, s_start - (s_end - n_times))
                    s_end = n_times
                inner_offset = start - s_start
            else:
                s_start, s_end = start, end
                inner_offset = 0

            # -- Extract data: original for cleaning, processed for CCA --
            # Bug 1: X_orig/X_temp separation — CCA may use re-referenced
            # data but projection always targets original data.
            X_orig = data_primary[:, s_start:s_end].T  # (stats_len, n_primary)
            Y_orig = data_ref[:, s_start:s_end].T      # (stats_len, n_ref)

            # Bug 2: optionally re-reference for CCA only
            X_cca = _apply_reref(X_orig, self.reref_primary)
            Y_cca = _apply_reref(Y_orig, self.reref_ref)

            try:
                A, B, R, U, V = canonical_correlation(X_cca, Y_cca)
            except Exception as exc:
                if self.verbose:
                    logger.warning("CCA failed for window at %d: %s", start, exc)
                cleaned[:, start:end] += data_primary[:, start:end] * wfn
                weights[start:end] += wfn
                all_corr.append(np.array([]))
                all_n_removed.append(0)
                all_removed_idx.append(np.array([], dtype=int))
                all_filters.append(np.array([]))
                all_patterns.append(np.array([]))
                continue

            r2 = (R ** 2).astype(np.float64)
            running_r2.extend(r2.tolist())

            # Threshold
            if threshold == "auto":
                thr = self._adaptive_threshold(running_r2)
            else:
                thr = float(threshold)

            bad_mask = r2 > thr

            # Safety cap: max_reject_fraction=0 means remove nothing;
            # otherwise allow at least 1 if any components exceed threshold
            if max_reject_fraction == 0:
                max_bad = 0
            else:
                max_bad = max(1, int(max_reject_fraction * len(r2)))
            if bad_mask.sum() > max_bad:
                order = np.argsort(r2)[::-1]
                bad_mask[:] = False
                if max_bad > 0:
                    bad_mask[order[:max_bad]] = True

            bad_idx = np.where(bad_mask)[0]

            all_corr.append(r2)
            all_n_removed.append(int(bad_idx.size))
            all_removed_idx.append(bad_idx)
            all_filters.append(A)
            all_patterns.append(B)

            if bad_idx.size > 0:
                # Select noise sources based on clean_with mode
                if clean_with == "X":
                    noise_sources = U[:, bad_idx]
                elif clean_with == "Y":
                    noise_sources = V[:, bad_idx]
                else:  # "both" — average of U and V (matches MATLAB 'XY' mode)
                    noise_sources = (U[:, bad_idx] + V[:, bad_idx]) / 2

                # Bug 1: project noise out of ORIGINAL (not re-referenced)
                # data over the full stats window, matching MATLAB's
                # iCanClean_cleanChansWithNoiseSources(X_localWindow, ...).
                X_mc = X_orig - X_orig.mean(axis=0, keepdims=True)
                Z_mc = noise_sources - noise_sources.mean(axis=0, keepdims=True)

                beta, *_ = la.lstsq(Z_mc, X_mc, lapack_driver="gelsy")
                X_clean_full = X_orig - Z_mc @ beta

                # Extract the inner clean-window portion
                X_clean_win = X_clean_full[inner_offset:inner_offset + actual_len]
                cleaned[:, start:end] += X_clean_win.T * wfn
            else:
                X_inner = X_orig[inner_offset:inner_offset + actual_len]
                cleaned[:, start:end] += X_inner.T * wfn

            weights[start:end] += wfn

        # Normalise overlap-add
        mask = weights > 0
        cleaned[:, mask] /= weights[mask]
        if not mask.all():
            cleaned[:, ~mask] = data_primary[:, ~mask]

        # Store QC attributes (top-level, will be overwritten by hybrid second pass)
        self.n_windows_ = len(starts)
        self.correlations_ = _pad_ragged(all_corr)
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

        data_out = data.copy()
        data_out[primary_idx, :] = cleaned
        return data_out

    def _run_calibrated_pass(
        self,
        data: np.ndarray,
        primary_idx: np.ndarray,
        ref_idx: np.ndarray,
        sfreq: float,
        *,
        threshold: float | str,
        max_reject_fraction: float,
    ) -> np.ndarray:
        """Calibrated mode: global CCA decomposition, per-window cleaning.

        Matches MATLAB ``calcCCAonWholeData=true``:

        1. Compute CCA on the **entire** recording → global A, B.
        2. Compute global mixing matrix (``fakeWinv``).
        3. For each sliding window:
           - Project local data through global A, B → local U, V.
           - Compute **local** pairwise correlations.
           - Threshold on local R² → select bad components.
           - Reconstruct noise via ``fakeWinv[:, bad] @ (X_mc @ A[:, bad]).T``
             and subtract from original.

        This is far more effective than per-window CCA because the global
        spatial filters are stable, while local thresholding adapts to
        which artifacts are active in each window.
        """
        data_primary = data[primary_idx, :]
        data_ref = data[ref_idx, :]
        n_times = data_primary.shape[1]

        # ---- Step 1: Global CCA on entire recording ----
        X_global = data_primary.T  # (T, n_primary)
        Y_global = data_ref.T      # (T, n_ref)

        X_cca = _apply_reref(X_global, self.reref_primary)
        Y_cca = _apply_reref(Y_global, self.reref_ref)

        A, B, R_global, U_global, V_global = canonical_correlation(
            X_cca, Y_cca
        )
        d = A.shape[1]
        if d == 0:
            logger.warning("CCA returned 0 components — returning data as-is")
            return data.copy()

        # ---- Step 2: Global mixing matrix ----
        # MATLAB: fakeWinv = mrdivide(X_mc', U')
        # Python equivalent: lstsq(U, X_mc) → (d, n_ch) → transpose
        X_global_mc = X_global - X_global.mean(axis=0, keepdims=True)
        fakeWinv_T, *_ = la.lstsq(
            U_global, X_global_mc, lapack_driver="gelsy"
        )
        fakeWinv = fakeWinv_T.T  # (n_primary, d)

        # Store global calibration for inspection
        self.global_A_ = A
        self.global_B_ = B
        self.global_R_ = R_global
        self.fakeWinv_ = fakeWinv

        if self.verbose:
            logger.info(
                "Calibrated mode: global CCA → %d components, "
                "top-5 R²: %s",
                d,
                np.round(R_global[:min(5, d)] ** 2, 3),
            )

        # ---- Step 3: Sliding-window cleaning ----
        win_samples = int(self.segment_len * sfreq)
        step_samples = max(1, int(round(win_samples * (1 - self.overlap))))

        if win_samples > n_times:
            raise ValueError(
                f"Window length ({win_samples} samples) exceeds "
                f"data length ({n_times} samples)."
            )

        starts = list(
            np.arange(0, n_times - win_samples + 1, step_samples)
        )
        last_possible = n_times - win_samples
        if starts and starts[-1] < last_possible:
            starts.append(last_possible)

        cleaned = np.zeros_like(data_primary)
        weights = np.zeros(n_times, dtype=np.float64)
        window_fn = np.ones(win_samples, dtype=np.float64)

        all_corr: list[np.ndarray] = []
        all_n_removed: list[int] = []
        all_removed_idx: list[np.ndarray] = []
        all_filters: list[np.ndarray] = []
        all_patterns: list[np.ndarray] = []
        running_r2: list[float] = []

        for start in starts:
            end = start + win_samples
            if end > n_times:
                end = n_times
            actual_len = end - start
            wfn = window_fn[:actual_len]

            # Local data (original, for cleaning)
            X_local = data_primary[:, start:end].T  # (win, n_primary)
            Y_local = data_ref[:, start:end].T      # (win, n_ref)

            # Project through global A, B → local canonical variates
            X_local_mc = X_local - X_local.mean(axis=0, keepdims=True)
            Y_local_mc = Y_local - Y_local.mean(axis=0, keepdims=True)
            U_local = X_local_mc @ A  # (win, d)
            V_local = Y_local_mc @ B  # (win, d)

            # Local pairwise correlations: R_local[i] = corr(U[:,i], V[:,i])
            U_zm = U_local - U_local.mean(axis=0, keepdims=True)
            V_zm = V_local - V_local.mean(axis=0, keepdims=True)
            u_norm = np.sqrt(np.sum(U_zm ** 2, axis=0))
            v_norm = np.sqrt(np.sum(V_zm ** 2, axis=0))
            denom = u_norm * v_norm
            denom[denom == 0] = 1.0
            R_local = np.sum(U_zm * V_zm, axis=0) / denom

            r2 = np.clip(R_local ** 2, 0.0, 1.0)
            running_r2.extend(r2.tolist())

            # Threshold
            if threshold == "auto":
                thr = self._adaptive_threshold(running_r2)
            else:
                thr = float(threshold)

            bad_mask = r2 > thr

            # Safety cap
            if max_reject_fraction == 0:
                max_bad = 0
            else:
                max_bad = max(1, int(max_reject_fraction * len(r2)))
            if bad_mask.sum() > max_bad:
                order = np.argsort(r2)[::-1]
                bad_mask[:] = False
                if max_bad > 0:
                    bad_mask[order[:max_bad]] = True

            bad_idx = np.where(bad_mask)[0]

            all_corr.append(r2)
            all_n_removed.append(int(bad_idx.size))
            all_removed_idx.append(bad_idx)
            all_filters.append(A)
            all_patterns.append(B)

            if bad_idx.size > 0:
                # MATLAB calibrated cleaning:
                #   activations = X_mc * A(:, bad)  — project into bad subspace
                #   noise_est = fakeWinv(:, bad) * activations'  — back to channels
                #   X_clean = X - noise_est'
                activations = X_local_mc @ A[:, bad_idx]       # (win, n_bad)
                noise_est = (fakeWinv[:, bad_idx] @ activations.T).T  # (win, n_ch)
                X_clean_win = X_local - noise_est

                cleaned[:, start:end] += X_clean_win.T * wfn
            else:
                cleaned[:, start:end] += X_local.T * wfn

            weights[start:end] += wfn

        # Normalise overlap-add
        mask = weights > 0
        cleaned[:, mask] /= weights[mask]
        if not mask.all():
            cleaned[:, ~mask] = data_primary[:, ~mask]

        # Store QC
        self.n_windows_ = len(starts)
        self.correlations_ = _pad_ragged(all_corr)
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
                "Calibrated: %d windows, %.1f%% had removals, "
                "%.1f components removed on average",
                self.n_windows_,
                pct_windows,
                total_removed / max(self.n_windows_, 1),
            )

        data_out = data.copy()
        data_out[primary_idx, :] = cleaned
        return data_out

    @staticmethod
    def _adaptive_threshold(running_r2: list[float]) -> float:
        """Compute adaptive R^2 threshold from running distribution."""
        if len(running_r2) > 10:
            return float(np.percentile(running_r2, 95))
        return 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_channels(
    data: np.ndarray,
    filter_spec: tuple | None,
    sfreq: float,
) -> np.ndarray:
    """Apply a filter to channel data (n_channels, n_times).

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    filter_spec : tuple (type, freqs) or None
        ``('notch', (lo, hi))``: bandstop — remove lo–hi Hz (brain band),
        keep outside (artifact frequencies). Matches MATLAB ``filtYtype='Notch'``.
        ``('hp', freq)``: high-pass.
        ``('lp', freq)``: low-pass.
        ``('bp', (lo, hi))``: band-pass.
    sfreq : float

    Returns
    -------
    filtered : ndarray, same shape
    """
    if filter_spec is None:
        return data
    from scipy.signal import butter, sosfiltfilt

    ftype, ffreqs = filter_spec
    if ftype == "notch":
        lo, hi = ffreqs
        sos = butter(4, [lo, hi], btype="bandstop", fs=sfreq, output="sos")
    elif ftype == "hp":
        sos = butter(4, ffreqs, btype="high", fs=sfreq, output="sos")
    elif ftype == "lp":
        sos = butter(4, ffreqs, btype="low", fs=sfreq, output="sos")
    elif ftype == "bp":
        lo, hi = ffreqs
        sos = butter(4, [lo, hi], btype="band", fs=sfreq, output="sos")
    else:
        raise ValueError(
            f"filter type must be 'notch', 'hp', 'lp', or 'bp', "
            f"got {ftype!r}"
        )
    out = data.copy()
    for i in range(out.shape[0]):
        out[i] = sosfiltfilt(sos, out[i])
    return out


def _apply_reref(data: np.ndarray, reref: bool | str) -> np.ndarray:
    """Apply average re-reference across channels for CCA input.

    Matches MATLAB ``iCanClean_reref``.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_channels)
    reref : bool or str
        ``False``: no re-referencing.
        ``True`` or ``'fullrank'``: full-rank average re-reference
        (divides by n+1, preserves rank — MATLAB default).
        ``'loserank'``: standard average re-reference (divides by n,
        loses 1 rank).

    Returns
    -------
    data_reref : ndarray, shape (n_samples, n_channels)
    """
    if reref is False:
        return data
    n_ch = data.shape[1]
    if reref is True or reref == "fullrank":
        # eye(n) - ones(n)/(n+1) — MATLAB iCanClean_reref 'fullrank'
        ref = np.eye(n_ch) - np.ones((n_ch, n_ch)) / (n_ch + 1)
    elif reref == "loserank":
        # eye(n) - ones(n)/n — standard average reference
        ref = np.eye(n_ch) - np.ones((n_ch, n_ch)) / n_ch
    else:
        raise ValueError(
            f"reref must be False, True, 'fullrank', or 'loserank', "
            f"got {reref!r}"
        )
    # ref is symmetric, so data @ ref == data @ ref.T
    return data @ ref


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
