from __future__ import annotations

import numpy as np

from limnfst.mapping import center_normalize
from limnfst.nfst import (
    align_to_simplex,
    get_Q_w,
    get_project,
    psi_eps_times,
)
from limnfst.novelty import (
    dist_to_basepoints,
    get_threshold,
    nearest_idx,
    novelty_score as minimum_novelty_score,
    novelty_mask,
)


class LIM_NFST:
    """Algorithms 1-2 from the revised LIM-NFST paper."""

    def __init__(self, eps=1e-4, delta=2.0, novel_label=-1, subspace="qw"):
        self.eps = float(eps)
        self.delta = float(delta)
        self.novel_label = novel_label
        self.subspace = subspace
        self.projection_matrix_ = None

    def fit(self, X, y, collect_diagnostics=False):
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y)
        classes = np.unique(y)
        
        # Algorithm 1, step 1: Pearson sample normalization.
        X_norm = center_normalize(X)

        # Steps 2-3: Woodbury Q, removal of the trivial direction, then QR.
        Q_w, R = get_Q_w(X_norm, y, self.eps)

        # Steps 4-6: positive EVD directions produce Theta_init.
        theta_init, eigenvalues, V, R_pinv = get_project(
            Q_w,
            R,
            y,
            n_components=len(classes) - 1,
            return_eigenvalues=True,
        )
        initial_centroids = (R_pinv @ V).T

        # Steps 7-11: map initial centroids to a regular simplex of edge delta.
        theta, target_base_points, initial_centroids, alignment = align_to_simplex(
            theta_init,
            X_norm,
            y,
            self.eps,
            self.delta,
            initial_centroids=initial_centroids,
        )
        
        projection_matrix = X_norm.T @ theta
        base_points = target_base_points
        threshold = get_threshold(self.delta)

        self.X_train_ = X
        self.X_train_norm_ = X_norm
        self.y_train_ = y
        self.y_fit_ = y
        self.classes_ = classes
        self.Q_w_ = Q_w
        self.R_w_ = R
        self.R_w_pinv = R_pinv
        self.between_eigenvectors_ = V
        self.theta_init_ = theta_init
        self.theta_ = theta
        self.between_eigenvalues_ = eigenvalues
        self.initial_centroids_ = initial_centroids
        self.alignment_matrix_ = alignment
        self.projection_matrix_ = projection_matrix
        self.base_points_ = base_points
        self.threshold_ = float(threshold)
        self.n_features_in_ = X.shape[1]
        self.subspace_shape_ = Q_w.shape
        self.diagnostics_ = self._build_diagnostics() if collect_diagnostics else None
        return self

    def _build_diagnostics(self):
        c = len(self.classes_)
        edge_squared = dist_to_basepoints(self.base_points_, self.base_points_)
        off_diagonal = edge_squared[~np.eye(c, dtype=bool)]
        centroid_error = np.linalg.norm(
            self.empirical_base_points_ - self.base_points_, axis=1
        )

        projected_within = np.zeros(
            (c - 1, c - 1), dtype=np.float64
        )
        for class_index, cls in enumerate(self.classes_):
            residual = self.Y_train_[self.y_fit_ == cls] - self.base_points_[class_index]
            projected_within += residual.T @ residual

        train_accuracy = float(
            np.mean(self.train_closed_pred_ == self.y_fit_) * 100
        )
        _, counts = np.unique(self.y_fit_, return_counts=True)
        return {
            "algorithm": "lim-nfst-revised-paper-algorithm-1-2",
            "subspace": self.subspace,
            "n_train": int(len(self.y_fit_)),
            "n_features": int(self.n_features_in_),
            "n_classes": int(c),
            "class_counts": {
                str(cls): int(count)
                for cls, count in zip(self.classes_, counts)
            },
            "eps": self.eps,
            "delta": self.delta,
            "basis_label": "Q_w",
            "basis_shape": list(map(int, self.Q_w_.shape)),
            "qr_r_shape": list(map(int, self.R_w_.shape)),
            "theta_init_shape": list(map(int, self.theta_init_.shape)),
            "theta_shape": list(map(int, self.theta_.shape)),
            "projection_shape": list(map(int, self.projection_matrix_.shape)),
            "base_points_shape": list(map(int, self.base_points_.shape)),
            "between_eigenvalues": self.between_eigenvalues_.astype(float).tolist(),
            "threshold_rule": "Algorithm 2: squared distance > 0.25 * delta^2",
            "threshold": self.threshold_,
            "threshold_min": self.threshold_,
            "threshold_mean": self.threshold_,
            "threshold_max": self.threshold_,
            "min_basepoint_distance": float(np.sqrt(np.min(off_diagonal))),
            "min_basepoint_squared_distance": float(np.min(off_diagonal)),
            "max_basepoint_edge_error": float(
                np.max(np.abs(np.sqrt(off_diagonal) - self.delta))
            ),
            "max_centroid_alignment_error": float(np.max(centroid_error)),
            "train_score_min": float(np.min(self.train_scores_)),
            "train_score_mean": float(np.mean(self.train_scores_)),
            "train_score_max": float(np.max(self.train_scores_)),
            "train_closed_set_accuracy": train_accuracy,
            "projected_within_scatter_fro": float(
                np.linalg.norm(projected_within, ord="fro")
            ),
            "max_train_to_own_base_distance": float(
                np.sqrt(np.max(self.train_scores_))
            ),
            "paper_invariants": {
                "dimension_is_c_minus_1": bool(self.theta_.shape[1] == c - 1),
                "simplex_is_centered": bool(
                    np.allclose(self.base_points_.mean(axis=0), 0.0, atol=1e-9)
                ),
                "all_simplex_edges_equal_delta": bool(
                    np.allclose(off_diagonal, self.delta**2, rtol=1e-8, atol=1e-10)
                ),
                "threshold_is_quarter_delta_squared": bool(
                    np.isclose(self.threshold_, 0.25 * self.delta**2)
                ),
            },
        }

    def transform(self, X):
        """Algorithm 2 steps 1-2"""
        X = np.asarray(X, dtype=np.float64)
        return center_normalize(X) @ self.projection_matrix_

    def novelty_score(self, X):
        """Return the squared distance to the nearest learned base point.

        Larger values indicate a more novel sample.  ``predict`` assigns the
        novel label exactly when this score is greater than ``threshold_``.
        """
        Y_test = self.transform(X)
        distances = dist_to_basepoints(Y_test, self.base_points_)
        return minimum_novelty_score(distances)

    def predict(self, X):
        Y_test = self.transform(X)
        distances = dist_to_basepoints(Y_test, self.base_points_)
        idx = nearest_idx(distances)
        prediction = np.asarray(self.classes_, dtype=object)[idx].copy()
        prediction[novelty_mask(distances, self.threshold_)] = self.novel_label
        return prediction

    def predict_closed(self, X):
        Y_test = self.transform(X)
        distances = dist_to_basepoints(Y_test, self.base_points_)
        return self.classes_[nearest_idx(distances)]
