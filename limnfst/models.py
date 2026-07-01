from __future__ import annotations
import numpy as np
from limnfst.mapping import center_normalize
from limnfst.novelty import *
from limnfst.nfst import *

class LIM_NFST:
    def __init__(
        self,
        eps=1e-6,
        novel_label=-1,
        subspace="qw",
    ):
        if subspace != "qw":
            raise ValueError("The implicit LIM-NFST implementation only supports subspace='qw'")
        self.eps = eps
        self.novel_label = novel_label
        self.subspace = subspace
        self.X_train_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.projection_matrix_: np.ndarray | None = None
        self.threshold_: float = 0
        
    def fit(self, X, y, collect_diagnostics=False):
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.X_train_ = X
        self.classes_ = np.unique(y)
        
        X_train_norm = center_normalize(X)
        
        Q_w, R_w = get_Q_w(X_train_norm, y, self.eps)
        if Q_w.shape[1] == 0:
            raise ValueError("Woodbury null subspace is empty")
        
        n_components = min(len(self.classes_) - 1, Q_w.shape[1])
        theta_matrix = get_project(Q_w, R_w, y, n_components=n_components)
        projection_matrix = X_train_norm.T @ theta_matrix
        Y_train = X_train_norm @ projection_matrix
        base_points = np.vstack([Y_train[y == cls].mean(axis=0) for cls in self.classes_])
        D_train = dist_to_basepoints(Y_train, base_points)
        train_scores = novelty_score(D_train)
        threshold = get_threshold(base_points)
        train_closed_pred = self.classes_[nearest_idx(D_train)]
        
        self.X_train_norm_ = X_train_norm
        self.theta_ = theta_matrix
        self.projection_matrix_ = projection_matrix
        self.base_points_ = base_points
        self.Y_train_ = Y_train
        self.train_scores_ = train_scores
        self.train_closed_pred_ = train_closed_pred
        self.threshold_ = threshold
        self.subspace_shape_ = Q_w.shape
        self.classes_ = np.unique(y)
        self.diagnostics_ = self._build_diagnostics(X_train_norm, y, Q_w, R_w) if collect_diagnostics else None
        
        return self

    def _build_diagnostics(self, X_norm, y, Q_w, R_w):
        classes, counts = np.unique(y, return_counts=True)
        if len(self.base_points_) > 1:
            base_dist = dist_to_basepoints(self.base_points_, self.base_points_)
            np.fill_diagonal(base_dist, np.inf)
            min_base_dist = float(np.min(base_dist))
        else:
            min_base_dist = None

        return {
            "algorithm": "lim-nfst-implicit",
            "subspace": self.subspace,
            "n_train": int(X_norm.shape[0]),
            "n_features": int(X_norm.shape[1]),
            "n_classes": int(len(classes)),
            "class_counts": {str(cls): int(count) for cls, count in zip(classes, counts)},
            "eps": float(self.eps),
            "basis_label": "Q_w",
            "basis_shape": [int(dim) for dim in Q_w.shape],
            "qr_r_shape": [int(dim) for dim in R_w.shape],
            "theta_shape": [int(dim) for dim in self.theta_.shape],
            "projection_shape": [int(dim) for dim in self.projection_matrix_.shape],
            "base_points_shape": [int(dim) for dim in self.base_points_.shape],
            "threshold_rule": "0.5 * min_{i != j} ||base_i - base_j||_2",
            "threshold": float(self.threshold_),
            "min_basepoint_distance": min_base_dist,
            "train_score_min": float(np.min(self.train_scores_)) if len(self.train_scores_) else None,
            "train_score_mean": float(np.mean(self.train_scores_)) if len(self.train_scores_) else None,
            "train_score_max": float(np.max(self.train_scores_)) if len(self.train_scores_) else None,
        }
    
    def transform(self, X):
        X_test = center_normalize(X)
        Y_test = X_test @ self.projection_matrix_
        return Y_test       
        
    def predict(self, X):
        Y_test = self.transform(X)
        D = dist_to_basepoints(Y_test, self.base_points_)
        scores = novelty_score(D)
        idx = nearest_idx(D)
        pred = self.classes_[idx]
        pred[scores > self.threshold_] = self.novel_label
        return pred

    def predict_closed(self, X):
        Y_test = self.transform(X)
        D = dist_to_basepoints(Y_test, self.base_points_)
        idx = nearest_idx(D)
        return self.classes_[idx]
        
