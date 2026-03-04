"""CCA decomposition utilities for ICanClean.

This module provides a MATLAB-compatible canonical correlation analysis (CCA)
implementation used by the ICanClean algorithm. The implementation mirrors
MATLAB's ``canoncorr`` function using rank-revealing QR with column pivoting
followed by SVD.

Authors: Sina Esmaeili (sina.esmaeili@umontreal.ca)
         Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)

References
----------
.. [1] Hotelling, H. (1936). Relations between two sets of variates.
       Biometrika, 28(3/4), 321-377.

.. [2] Downey, R. J., & Ferris, D. P. (2022). The iCanClean Algorithm:
       How to Remove Artifacts using Reference Noise Recordings.
       arXiv:2201.11798.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg as la


def canonical_correlation(
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute canonical correlation analysis between two matrices.

    MATLAB-compatible implementation of ``canoncorr`` using rank-revealing QR
    with column pivoting followed by SVD on the cross-correlation of the
    Q-factors.

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
        Canonical variates for X, unit-variance normalised (ddof=1).
    V : ndarray, shape (n_samples, d)
        Canonical variates for Y, unit-variance normalised (ddof=1).

    Notes
    -----
    The algorithm follows MATLAB's ``canoncorr`` exactly:

    1. Mean-center both matrices.
    2. Rank-revealing QR with column pivoting on each.
    3. SVD of :math:`Q_x^T Q_y`.
    4. Back-solve through R-factors to get coefficients.
    5. Normalise canonical variates to unit variance (ddof=1).

    Examples
    --------
    >>> import numpy as np
    >>> from mne_denoise.icanclean.cca import canonical_correlation
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((200, 8))
    >>> Y = rng.standard_normal((200, 4))
    >>> A, B, R, U, V = canonical_correlation(X, Y)
    >>> R.shape
    (4,)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X and Y must have the same number of samples, "
            f"got {X.shape[0]} and {Y.shape[0]}"
        )

    # Mean-center
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # Rank-revealing QR with pivoting
    Qx, Rx, Px = la.qr(Xc, mode="economic", pivoting=True)
    Qy, Ry, Py = la.qr(Yc, mode="economic", pivoting=True)

    eps = np.finfo(np.float64).eps

    # Determine numerical ranks
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

    # Truncate to numerical rank
    Qx = Qx[:, :rx]
    Rx = Rx[:rx, :rx]
    Qy = Qy[:, :ry]
    Ry = Ry[:ry, :ry]

    # SVD of cross-correlation of Q-factors
    Ux, s, VyT = la.svd(Qx.T @ Qy, full_matrices=False)
    Vy = VyT.T

    # Clip correlations to [0, 1] for numerical safety
    R = np.clip(s, 0.0, 1.0)

    # Undo column pivots -> full-space coefficient matrices
    Ex = np.eye(X.shape[1])[:, Px][:, :rx]
    Ey = np.eye(Y.shape[1])[:, Py][:, :ry]

    A = Ex @ la.solve(Rx, Ux)
    B = Ey @ la.solve(Ry, Vy)

    # Canonical variates
    U = Xc @ A
    V = Yc @ B

    # Normalise to unit variance (ddof=1) and adjust coefficients
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
