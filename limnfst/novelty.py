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

def novelty_mask(Distance_mtrx, threshold):
    scores = novelty_score(Distance_mtrx)
    threshold = np.asarray(threshold, dtype=np.float64)

    if threshold.ndim == 0:
        return scores > float(threshold)

    idx = nearest_idx(Distance_mtrx)
    return scores > threshold[idx]

def predict_dist(Distance_mtrx, classes, threshold, novel_label):
    idx = nearest_idx(Distance_mtrx)
    
    predictions = np.asarray(classes, dtype=object)[idx]
    
    predictions[novelty_mask(Distance_mtrx, threshold)] = novel_label           
            
    return predictions

# def get_threshold(base_points):
#     base_points = np.asarray(base_points, dtype=np.float64)
#     if len(base_points) == 0:
#         return 0.0
#     if len(base_points) < 2:
#         return 0.0
# 
#     base_dist = cdist(base_points, base_points, metric="euclidean")
#     np.fill_diagonal(base_dist, np.inf)
#     min_base_dist = float(np.min(base_dist))
#     if not np.isfinite(min_base_dist):
#         return 0.0
# 
#     return 0.5 * min_base_dist
# 
# 
# # Train-score calibration kept only for experiments; paper LIM uses base-point geometry.
# def get_threshold(train_scores, percentile=95):
#     return float(np.percentile(train_scores, percentile))

def get_threshold(train_scores, beta=3.0):
    train_scores = np.asarray(train_scores, dtype=np.float64)
    mean_score = np.mean(train_scores)
    std_score = np.std(train_scores)
    return float(mean_score + beta * std_score)

