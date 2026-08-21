from __future__ import annotations

import numpy as np


def reference_cloud_threshold(leave_one_out_scores, quantile):
    return max(float(np.quantile(leave_one_out_scores, quantile)), 1e-12)
