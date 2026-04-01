"""Core canonical correlation analysis for iCanClean.

This module contains:
1. ``canonical_correlation``: The core canonical correlation analysis solver
   used by iCanClean [1]_ [2]_.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Downey, R. J., & Ferris, D. P. (2022). The iCanClean Algorithm:
       How to Remove Artifacts using Reference Noise Recordings.
       arXiv:2201.11798.
.. [2] Hotelling, H. (1936). Relations between two sets of variates.
       Biometrika, 28(3/4), 321-377.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as la


def canonical_correlation(
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute canonical correlation analysis between two matrices.

    This implements the low-level canonical correlation analysis (CCA) solver
    used by iCanClean [1]_. The algorithm identifies pairs of linear
    combinations of ``X`` and ``Y`` that are maximally correlated with each
    other, ordered by decreasing canonical correlation [2]_.

    The computation proceeds as follows:

    1. Mean-center ``X`` and ``Y``.
    2. Compute rank-revealing QR decompositions with column pivoting.
    3. Compute the singular value decomposition of :math:`Q_x^T Q_y`.
    4. Back-solve through the ``R`` factors to obtain canonical coefficients.
    5. Normalize the canonical variates to unit variance.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features_x)
        First data matrix (e.g. EEG scalp channels).
    Y : ndarray, shape (n_samples, n_features_y)
        Second data matrix (e.g. reference noise channels).

    Returns
    -------
    A : ndarray, shape (n_features_x, d)
        Coefficients for X canonical variates. ``d = min(rank(X), rank(Y))``.
    B : ndarray, shape (n_features_y, d)
        Coefficients for Y canonical variates.
    R : ndarray, shape (d,)
        Canonical correlations in descending order.
    U : ndarray, shape (n_samples, d)
        Canonical variates for X, unit-variance normalized (ddof=1).
    V : ndarray, shape (n_samples, d)
        Canonical variates for Y, unit-variance normalized (ddof=1).

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.icanclean._cca import canonical_correlation
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((200, 8))
    >>> Y = rng.standard_normal((200, 4))
    >>> A, B, R, U, V = canonical_correlation(X, Y)
    >>> R.shape
    (4,)

    See Also
    --------
    mne_denoise.icanclean.compute_icanclean : Core iCanClean cleaning pass.
    mne_denoise.icanclean.ICanClean : Estimator interface for iCanClean.

    References
    ----------
    .. [1] Downey, R. J., & Ferris, D. P. (2022). The iCanClean Algorithm: How to
           Remove Artifacts using Reference Noise Recordings. arXiv:2201.11798.
    .. [2] Hotelling, H. (1936). Relations between two sets of variates.
           Biometrika, 28(3/4), 321-377.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples, "
            f"got {X.shape[0]} and {Y.shape[0]}"
        )

    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    Qx, Rx, Px = la.qr(Xc, mode="economic", pivoting=True)
    Qy, Ry, Py = la.qr(Yc, mode="economic", pivoting=True)

    eps = np.finfo(np.float64).eps

    tolx = eps * max(Xc.shape) * (np.abs(np.diag(Rx)).max() if Rx.size else 0.0)
    toly = eps * max(Yc.shape) * (np.abs(np.diag(Ry)).max() if Ry.size else 0.0)
    rx = int(np.sum(np.abs(np.diag(Rx)) > tolx)) if Rx.size else 0
    ry = int(np.sum(np.abs(np.diag(Ry)) > toly)) if Ry.size else 0

    if rx == 0 or ry == 0:
        d = 0
        return (
            np.empty((X.shape[1], d), dtype=np.float64),
            np.empty((Y.shape[1], d), dtype=np.float64),
            np.empty(d, dtype=np.float64),
            np.empty((X.shape[0], d), dtype=np.float64),
            np.empty((Y.shape[0], d), dtype=np.float64),
        )

    Qx = Qx[:, :rx]
    Rx = Rx[:rx, :rx]
    Qy = Qy[:, :ry]
    Ry = Ry[:ry, :ry]

    Ux, s, VyT = la.svd(Qx.T @ Qy, full_matrices=False)
    Vy = VyT.T

    R = np.clip(s, 0.0, 1.0)

    Ex = np.eye(X.shape[1])[:, Px][:, :rx]
    Ey = np.eye(Y.shape[1])[:, Py][:, :ry]

    A = Ex @ la.solve(Rx, Ux)
    B = Ey @ la.solve(Ry, Vy)

    U = Xc @ A
    V = Yc @ B

    def _unit_var(
        Z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        std = Z.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        return Z / std, 1.0 / std

    U, su = _unit_var(U)
    V, sv = _unit_var(V)
    A = (A * su).astype(np.float64)
    B = (B * sv).astype(np.float64)

    return A, B, R.astype(np.float64), U.astype(np.float64), V.astype(np.float64)
