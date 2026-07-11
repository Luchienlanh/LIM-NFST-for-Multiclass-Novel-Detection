from __future__ import annotations
import numpy as np
from limnfst.mapping import center_normalize
from limnfst.novelty import *
from limnfst.nfst import *

class LIM_NFST:
    def __init__(
        self,
        eps=1e-4,
        novel_label=-1,
        subspace="qw",
        beta=3.0,
    ):
        if subspace != "qw":
            raise ValueError("The implicit LIM-NFST implementation only supports subspace='qw'")
        self.eps = eps
        self.novel_label = novel_label
        self.subspace = subspace
        self.beta = beta
        self.X_train_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.projection_matrix_: np.ndarray | None = None
        self.threshold_: np.ndarray | float = 0
        
    def fit(self, X, y, collect_diagnostics=False):
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.X_train_ = X
        self.classes_ = np.unique(y)
        
        X_train_norm = center_normalize(X)
        
        Q_w, R_w = get_Q_w(X_train_norm, y, self.eps)
        if Q_w.shape[1] == 0:
            raise ValueError("Woodbury null subspace is empty")
        
        # n_components = min(len(self.classes_) - 1, Q_w.shape[1])
        # theta_matrix = get_project(Q_w, X_train_norm, y, self.eps, n_components=n_components)
        theta_matrix = get_project(Q_w, R_w, y, dist=2.0)
        projection_matrix = X_train_norm.T @ theta_matrix
        
                # --- PHƯƠNG PHÁP 1: Z-SCORE SCALING ---
        projection_matrix = X_train_norm.T @ theta_matrix
        
        # 1. Chiếu thử tập train (unregularized) để tính toán độ lệch chuẩn của từng chiều con
        Y_train_raw = X_train_norm @ projection_matrix
        
        # 2. Tính độ lệch chuẩn từng chiều (std của từng cột)
        dim_std = np.std(Y_train_raw, axis=0)
        # Tránh chia cho 0 (nếu có chiều nào đó biến động bằng 0, ta đặt std = 1.0)
        dim_std = np.where(dim_std > 0, dim_std, 1.0)
        
        # 3. Scale ma trận chiếu bằng cách chia cho độ lệch chuẩn
        # (Phép chia này hoàn toàn TUYẾN TÍNH vì chia cho hằng số)
        projection_matrix_scaled = projection_matrix / dim_std
        
        # 4. Sử dụng ma trận chiếu đã scale để tính base points và các tọa độ huấn luyện mới
        Y_train = X_train_norm @ projection_matrix_scaled + self.eps * (theta_matrix / dim_std)
        base_points = np.vstack([Y_train[y == cls].mean(axis=0) for cls in self.classes_])
        
        # Lưu các giá trị đã scale vào thuộc tính của lớp
        self.projection_matrix_ = projection_matrix_scaled
        
        # Y_train = X_train_norm @ projection_matrix + self.eps * theta_matrix
        # base_points = np.vstack([Y_train[y == cls].mean(axis=0) for cls in self.classes_])
        
        # Calculate threshold based on variance of test-like projections
        Y_train_trans = X_train_norm @ projection_matrix
        D_train_trans = dist_to_basepoints(Y_train_trans, base_points)
        train_scores_trans = novelty_score(D_train_trans)
        
        D_train = dist_to_basepoints(Y_train, base_points)
        train_scores = novelty_score(D_train)
        train_nearest_idx = nearest_idx(D_train)
        threshold = get_threshold(train_scores_trans, beta=self.beta)
        train_closed_pred = self.classes_[train_nearest_idx]
        
        self.X_train_norm_ = X_train_norm
        self.theta_ = theta_matrix
        self.projection_matrix_ = projection_matrix
        self.base_points_ = base_points
        self.Y_train_ = Y_train
        self.train_scores_ = train_scores
        self.train_nearest_idx_ = train_nearest_idx
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
        threshold = np.asarray(self.threshold_, dtype=np.float64)
        if threshold.ndim == 0:
            threshold_values = [float(threshold)]
        else:
            threshold_values = [float(value) for value in threshold]

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
            "threshold_rule": f"variance-based threshold: mean + {self.beta} * std",
            "threshold": threshold_values,
            "threshold_min": float(np.min(threshold)) if threshold.size else None,
            "threshold_mean": float(np.mean(threshold)) if threshold.size else None,
            "threshold_max": float(np.max(threshold)) if threshold.size else None,
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
        pred[novelty_mask(D, self.threshold_)] = self.novel_label
        return pred

    def predict_closed(self, X):
        Y_test = self.transform(X)
        D = dist_to_basepoints(Y_test, self.base_points_)
        idx = nearest_idx(D)
        return self.classes_[idx]
        
