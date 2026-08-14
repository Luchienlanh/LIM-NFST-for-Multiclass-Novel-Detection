"""Competitor models used by the LIM-NFST comparison script."""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import KernelCenterer


def compute_kernel(X, Y=None, kernel="rbf"):
    """Compute the kernel used by the local kNFST baseline."""
    if kernel is None or kernel == "none":
        return X @ (Y.T if Y is not None else X.T)
    return pairwise_kernels(X, Y, metric=kernel)


def sparse_within_class_matrix(y):
    classes, counts = np.unique(y, return_counts=True)
    inverse_counts = dict(zip(classes, 1.0 / counts))
    number_of_samples = len(y)
    inverse_values = np.vectorize(inverse_counts.get)(y)
    rows, columns = np.meshgrid(
        np.arange(number_of_samples),
        np.arange(number_of_samples),
        indexing="ij",
    )
    same_class = y[rows] == y[columns]
    rows = rows[same_class]
    columns = columns[same_class]
    return csr_matrix(
        (inverse_values[rows], (rows, columns)),
        shape=(number_of_samples, number_of_samples),
    )


def null_space(matrix, epsilon=1e-12):
    _, singular_values, right_vectors = svd(matrix)
    return right_vectors[singular_values <= epsilon].T


def learn_knfst(kernel_matrix, y):
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("KNFST requires at least two classes.")

    number_of_rows, number_of_columns = kernel_matrix.shape
    centered_kernel = KernelCenterer().fit_transform(kernel_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(centered_kernel)
    positive = eigenvalues > 1e-12
    eigenvectors = eigenvectors[:, positive]
    eigenvalues = eigenvalues[positive]
    basis = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues))

    within_class = sparse_within_class_matrix(y)
    mean_matrix = np.ones((number_of_columns, number_of_columns))
    mean_matrix /= number_of_columns
    centered_basis = (np.eye(number_of_columns) - mean_matrix) @ basis
    difference = np.eye(number_of_rows, number_of_columns) - within_class
    helper = centered_basis.T @ kernel_matrix @ difference
    within_scatter = helper @ helper.T
    eigenvectors = null_space(within_scatter)

    if eigenvectors.shape[1] < 1:
        _, fallback_vectors = np.linalg.eigh(within_scatter)
        eigenvectors = fallback_vectors[:, :1]

    projection = centered_basis @ eigenvectors
    centroids = []
    for class_label in classes:
        class_kernel = kernel_matrix[:, y == class_label]
        centroids.append(np.mean(class_kernel.T @ projection, axis=0))
    return projection, np.asarray(centroids).real


def softmax(values):
    values = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(values)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


class KNFST:
    """Kernel NFST baseline using nearest class centroid in null space."""

    def __init__(self, kernel="rbf"):
        self.kernel = kernel
        self.X_train = None
        self.projection = None
        self.centroids = None

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        kernel_matrix = compute_kernel(self.X_train, kernel=self.kernel)
        self.projection, self.centroids = learn_knfst(kernel_matrix, y)
        return self

    def distances(self, X):
        test_kernel = compute_kernel(self.X_train, np.asarray(X), self.kernel)
        projected = (test_kernel.T @ self.projection).real
        return np.linalg.norm(
            projected[:, None, :] - self.centroids[None, :, :],
            axis=2,
        )

    def predict(self, X):
        return np.argmin(self.distances(X), axis=1)

    def predict_proba(self, X):
        return softmax(-self.distances(X))
