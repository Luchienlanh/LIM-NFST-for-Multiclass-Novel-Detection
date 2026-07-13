"""Pearson sample mapping used by the revised LIM-NFST paper."""

from __future__ import annotations

import numpy as np


def center_normalize(X):
    """Center and L2-normalize every sample (paper Eq. 20).

    The repository represents samples by rows, whereas the paper represents
    them by columns. Consequently, centering is performed across features
    (``axis=1``) independently for each sample.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array with samples in rows")

    centered = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(
            "Pearson mapping is undefined for a sample whose features are all equal"
        )
    return centered / norms
