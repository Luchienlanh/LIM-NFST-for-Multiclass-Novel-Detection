"""Algorithm 2 distance and novelty decision for revised LIM-NFST."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def dist_to_basepoints(projected_samples, base_points):
    """Return squared Euclidean distances used by Algorithm 2 steps 3-4."""
    projected_samples = np.asarray(projected_samples, dtype=np.float64)
    base_points = np.asarray(base_points, dtype=np.float64)
    return cdist(projected_samples, base_points, metric="sqeuclidean")


def novelty_score(distance_matrix):
    return np.min(np.asarray(distance_matrix, dtype=np.float64), axis=1)


def nearest_idx(distance_matrix):
    return np.argmin(np.asarray(distance_matrix, dtype=np.float64), axis=1)


def novelty_mask(distance_matrix, threshold):
    """Apply the scalar squared-distance threshold from Algorithm 2."""
    threshold = float(threshold)
    return novelty_score(distance_matrix) > threshold


def get_threshold(delta):
    """Return Algorithm 2 line 5 threshold: (delta/2)^2 = 0.25 delta^2.

    The prose Eq. 35 in the supplied PDF prints ``0.5 delta^2`` despite using a
    squared norm. Algorithm 2 explicitly prints ``0.25 delta^2``; the latter is
    also the squared radius of the non-overlapping balls of radius delta/2.
    """
    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a finite positive edge length")
    return 0.25 * delta**2


def predict_dist(distance_matrix, classes, threshold, novel_label):
    indices = nearest_idx(distance_matrix)
    predictions = np.asarray(classes, dtype=object)[indices].copy()
    predictions[novelty_mask(distance_matrix, threshold)] = novel_label
    return predictions
