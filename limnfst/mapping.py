from __future__ import annotations

import numpy as np


def center_normalize(X):
    """Center and L2-normalize every sample (paper Eq. 20).

    The repository represents samples by rows, whereas the paper represents
    them by columns. Consequently, centering is performed across features
    (``axis=1``) independently for each sample.
    """
    X = np.asarray(X, dtype=np.float64)

    centered = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / norms
