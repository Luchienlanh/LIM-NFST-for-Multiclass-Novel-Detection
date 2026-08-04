from __future__ import annotations

import numpy as np


def center_and_normalize(X):
    """Center and L2-normalize each sample independently."""
    X = np.asarray(X, dtype=np.float64)

    centered_X = X - X.mean(axis=1, keepdims=True)
    row_norms = np.linalg.norm(centered_X, axis=1, keepdims=True)

    # A constant row has norm zero. Keeping its divisor at one returns a
    # zero row instead of NaN.
    row_norms[row_norms == 0.0] = 1.0
    return centered_X / row_norms


def center_projection(projected_X):
    """Move every projected sample onto the centered-simplex hyperplane."""
    projected_X = np.asarray(projected_X, dtype=np.float64)
    return projected_X - projected_X.mean(axis=1, keepdims=True)
