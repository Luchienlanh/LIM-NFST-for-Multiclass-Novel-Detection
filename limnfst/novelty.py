import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist

# def dist_to_basepoints(y, base_points):
#     N = len(y)   # num sample
#     n = np.unique(N)  # num class
#     D = np.zeros(N, n)
    
#     for i in range(N):
#         for j in range(n):
#             D[i, j] = euclidean_distances(y[i], base_points[j])
            
#     return D


def dist_to_basepoints(y, base_points):
    return cdist(y, base_points, metric='euclidean')


def novelty_score(Distance_mtrx):
    score = np.min(Distance_mtrx, axis=1)
    return score

def nearest_idx(Distance_mtrx):
    idx = np.argmin(Distance_mtrx, axis=1)
    return idx

def predict_dist(Distance_mtrx, classes, threshold, novel_label):
    scores = novelty_score(Distance_mtrx)
    idx = nearest_idx(Distance_mtrx)
    
    predictions = np.asarray(classes, dtype=object)[idx]
    
    predictions[scores > threshold] = novel_label           
            
    return predictions

# def get_threshold(base_points):
#     if len(base_points) < 2:
#         return 0.0

#     base_dist = dist_to_basepoints(base_points, base_points)
#     np.fill_diagonal(base_dist, np.inf)
#     min_base_dist = float(np.min(base_dist))

#     if not np.isfinite(min_base_dist):
#         return 0.0

#     return 0.5 * min_base_dist


# Train-score calibration kept only for experiments; paper LIM uses base-point geometry.
def get_threshold(train_scores, percentile=95):
    return float(np.percentile(train_scores, percentile))

