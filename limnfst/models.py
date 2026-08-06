from __future__ import annotations

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split

from .mapping import center_and_normalize, center_projection
from .nfst import train_lim_projection


class LIM_NFST:
    def __init__(
        self,
        epsilon=1e-4,
        reference_size=0.20,
        number_of_neighbors=5,
        novelty_quantile=0.95,
        random_state=42,
        use_rff=False,
        rff_components=256,
        rff_gamma_multiplier=1.0,
    ):
        self.epsilon = float(epsilon)
        self.reference_size = float(reference_size)
        self.number_of_neighbors = int(number_of_neighbors)
        self.novelty_quantile = float(novelty_quantile)
        self.random_state = int(random_state)
        self.use_rff = bool(use_rff)
        self.rff_components = int(rff_components)
        self.rff_gamma_multiplier = float(rff_gamma_multiplier)

        # This remains None until fit() finishes learning the projection.
        self.projection_matrix_ = None
        self.rff_mapper_ = None
        self.rff_gamma_ = None

    def _calculate_rff_gamma(self, X):
        """Calculate gamma='scale', then apply the configured multiplier."""
        variance = float(np.var(X))
        if not np.isfinite(variance) or variance <= 0.0:
            raise ValueError(
                "Cannot calculate RFF gamma because training variance is "
                f"{variance}."
            )

        scale_gamma = 1.0 / (X.shape[1] * variance)
        return scale_gamma * self.rff_gamma_multiplier

    def _fit_rff(self, X):
        """Fit the optional RFF map and return the model input features."""
        if not self.use_rff:
            self.rff_mapper_ = None
            self.rff_gamma_ = None
            return X

        if self.rff_components < 1:
            raise ValueError("rff_components must be at least one.")
        if self.rff_gamma_multiplier <= 0.0:
            raise ValueError("rff_gamma_multiplier must be greater than zero.")

        self.rff_gamma_ = self._calculate_rff_gamma(X)
        self.rff_mapper_ = RBFSampler(
            gamma=self.rff_gamma_,
            n_components=self.rff_components,
            random_state=self.random_state,
        )
        return self.rff_mapper_.fit_transform(X)

    def _apply_rff(self, X):
        """Apply the fitted RFF map, or keep the original features."""
        if not self.use_rff:
            return X
        if self.rff_mapper_ is None:
            raise RuntimeError("Call fit before applying RFF.")
        return self.rff_mapper_.transform(X)

    # def _check_parameters(self):
    #     if self.epsilon <= 0.0:
    #         raise ValueError("epsilon must be greater than zero.")
    #     if not 0.0 < self.reference_size < 1.0:
    #         raise ValueError("reference_size must be between zero and one.")
    #     if self.number_of_neighbors < 1:
    #         raise ValueError("number_of_neighbors must be at least one.")
    #     if not 0.0 < self.novelty_quantile < 1.0:
    #         raise ValueError("novelty_quantile must be between zero and one.")

    def fit(self, X, y):
        """Learn the LIM projection and build one reference cloud per class."""
        # self._check_parameters()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]
        X = self._fit_rff(X)
        self.model_features_in_ = X.shape[1]

        # if X.ndim != 2:
        #     raise ValueError("X must be a two-dimensional matrix.")
        # if len(X) != len(y):
        #     raise ValueError("X and y must contain the same number of samples.")
        # if not np.isfinite(X).all():
        #     raise ValueError("X contains NaN or infinite values.")

        # X_fit is used to learn Theta. X_reference is never used to learn it.
        X_fit, X_reference, y_fit, y_reference = train_test_split(
            X,
            y,
            test_size=self.reference_size,
            stratify=y,
            random_state=self.random_state,
        )

        X_fit_normalized = center_and_normalize(X_fit)
        classes, theta, projection_matrix, base_points = train_lim_projection(
            X_fit_normalized,
            y_fit,
            self.epsilon,
        )

        self.classes_ = classes
        self.theta_ = theta
        self.projection_matrix_ = projection_matrix
        self.base_points_ = base_points
        self.X_fit_ = X_fit
        self.y_fit_ = y_fit
        self.X_reference_ = X_reference
        self.y_reference_ = y_reference

        # X_reference is already in the optional RFF space at this point.
        self.reference_projection_ = self._project_model_features(
            X_reference
        )

        self.reference_points_ = []
        self.reference_thresholds_ = []

        for class_label in self.classes_:
            class_reference_points = self.reference_projection_[
                y_reference == class_label
            ]

            # At least two points are required for leave-one-out calibration.
            # if len(class_reference_points) < 2:
            #     raise ValueError(
            #         f"Class {class_label!r} needs at least two reference samples."
            #     )

            threshold = self._calculate_reference_threshold(
                class_reference_points
            )

            self.reference_points_.append(class_reference_points)
            self.reference_thresholds_.append(threshold)

        self.reference_thresholds_ = np.asarray(
            self.reference_thresholds_,
            dtype=np.float64,
        )
        return self

    def _project_model_features(self, X):
        """Project features that are already in the model input space."""
        X_normalized = center_and_normalize(X)
        projected_X = X_normalized @ self.projection_matrix_
        return center_projection(projected_X)

    def transform(self, X):
        """Apply optional RFF and project into the c-dimensional LIM space."""
        if self.projection_matrix_ is None:
            raise RuntimeError("Call fit before transform.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        X = self._apply_rff(X)
        return self._project_model_features(X)

    def _mean_nearest_distance(
        self,
        projected_X,
        reference_points,
        exclude_same_sample=False,
    ):
        """Mean squared distance to the k nearest points in one class."""
        differences = (
            projected_X[:, np.newaxis, :]
            - reference_points[np.newaxis, :, :]
        )
        squared_distances = np.sum(differences * differences, axis=2)

        if exclude_same_sample:
            # During calibration each reference point must not match itself.
            np.fill_diagonal(squared_distances, np.inf)
            available_neighbors = len(reference_points) - 1
        else:
            available_neighbors = len(reference_points)

        number_of_neighbors = min(
            self.number_of_neighbors,
            available_neighbors,
        )
        sorted_distances = np.sort(squared_distances, axis=1)
        nearest_distances = sorted_distances[:, :number_of_neighbors]
        return nearest_distances.mean(axis=1)

    def _calculate_reference_threshold(self, class_reference_points):
        """Calibrate a class novelty threshold by leave-one-out distances."""
        leave_one_out_scores = self._mean_nearest_distance(
            class_reference_points,
            class_reference_points,
            exclude_same_sample=True,
        )
        threshold = np.quantile(
            leave_one_out_scores,
            self.novelty_quantile,
        )
        return max(float(threshold), 1e-12)

    def reference_scores(self, X):
        """Return one reference distance per sample and class."""
        projected_X = self.transform(X)
        scores_by_class = []

        for class_reference_points in self.reference_points_:
            class_scores = self._mean_nearest_distance(
                projected_X,
                class_reference_points,
            )
            scores_by_class.append(class_scores)

        return np.column_stack(scores_by_class)

    def predict_closed(self, X):
        """Choose the class with the smallest reference-cloud distance."""
        scores = self.reference_scores(X)
        class_indices = np.argmin(scores, axis=1)
        return self.classes_[class_indices]

    def novelty_scores(self, X):
        """Return predicted-class distance divided by its fitted threshold."""
        scores = self.reference_scores(X)
        predicted_indices = np.argmin(scores, axis=1)
        row_indices = np.arange(len(scores))

        predicted_scores = scores[row_indices, predicted_indices]
        predicted_thresholds = self.reference_thresholds_[predicted_indices]
        return predicted_scores / predicted_thresholds

    def predict_open(self, X, novel_label=-1):
        """Replace predictions whose novelty score exceeds one."""
        predictions = np.asarray(self.predict_closed(X), dtype=object)
        predictions[self.novelty_scores(X) > 1.0] = novel_label
        return predictions
