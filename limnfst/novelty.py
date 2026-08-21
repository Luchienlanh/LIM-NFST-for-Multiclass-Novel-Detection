from __future__ import annotations

import numpy as np


def reference_cloud_threshold(leave_one_out_scores, quantile):
    """Return one finite class radius from leave-one-out cloud distances."""
    scores = np.asarray(leave_one_out_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("leave_one_out_scores must be a non-empty vector.")
    if not np.isfinite(scores).all():
        raise ValueError("leave_one_out_scores contains NaN or infinite values.")

    threshold = float(np.quantile(scores, quantile))
    return max(threshold, 1e-12)
