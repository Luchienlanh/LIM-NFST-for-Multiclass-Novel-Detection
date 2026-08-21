from __future__ import annotations

import numpy as np


def make_class_indicator(y, classes):
    """Create the n_samples x n_classes label-indicator matrix."""
    class_indicator = np.zeros((len(y), len(classes)), dtype=np.float64)

    for class_index, class_label in enumerate(classes):
        class_indicator[y == class_label, class_index] = 1.0

    return class_indicator


def calculate_q_basis(X_normalized, y, classes, epsilon):
    """Calculate the row-centered Q matrix and its QR basis."""
    class_indicator = make_class_indicator(y, classes)
    n_features = X_normalized.shape[1]

    system_matrix = (
        epsilon * np.eye(n_features, dtype=np.float64)
        + X_normalized.T @ X_normalized
    )
    right_hand_side = X_normalized.T @ class_indicator
    correction = np.linalg.solve(system_matrix, right_hand_side)

    Q = (class_indicator - X_normalized @ correction) / epsilon
    Q_centered = Q - Q.mean(axis=1, keepdims=True)

    full_Q_basis, full_R = np.linalg.qr(Q_centered, mode="reduced")
    number_of_directions = len(classes) - 1

    Q_basis = full_Q_basis[:, :number_of_directions]
    R = full_R[:number_of_directions, :]
    return Q_basis, R


def calculate_between_scatter(reduced_labels, y, classes):
    """Calculate between-class scatter without building an n x n matrix."""
    global_mean = reduced_labels.mean(axis=0)
    number_of_directions = reduced_labels.shape[1]
    between_scatter = np.zeros(
        (number_of_directions, number_of_directions),
        dtype=np.float64,
    )

    for class_label in classes:
        class_rows = reduced_labels[y == class_label]
        class_mean = class_rows.mean(axis=0)
        between_scatter += len(class_rows) * np.outer(class_mean, class_mean)

    between_scatter -= len(y) * np.outer(global_mean, global_mean)
    return (between_scatter + between_scatter.T) / 2.0


def calculate_initial_projection(Q_basis, R, y, classes):
    """Calculate Theta_init and its initial class-centroid matrix M."""
    R_pseudoinverse = np.linalg.pinv(R)
    class_indicator = make_class_indicator(y, classes)
    reduced_labels = class_indicator @ R_pseudoinverse
    between_scatter = calculate_between_scatter(
        reduced_labels,
        y,
        classes,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(between_scatter)
    descending_order = np.argsort(eigenvalues)[::-1]
    number_of_directions = len(classes) - 1
    selected_indices = descending_order[:number_of_directions]
    selected_vectors = eigenvectors[:, selected_indices]

    theta_initial = Q_basis @ selected_vectors
    initial_centroids = (R_pseudoinverse @ selected_vectors).T
    return theta_initial, initial_centroids


def align_projection_to_simplex(theta_initial, initial_centroids):
    """Align Theta_init to H_c = I_c - 11.T/c."""
    centroid_gram = initial_centroids @ initial_centroids.T
    
    alignment = np.linalg.solve(
        centroid_gram,
        initial_centroids,
    ).T

    theta = theta_initial @ alignment.T
    return theta


def train_lim_projection(X_normalized, y, epsilon):
    """Run the LIM-NFST training phase and return its projection data."""
    y = np.asarray(y)
    classes = np.unique(y)

    Q_basis, R = calculate_q_basis(
        X_normalized,
        y,
        classes,
        epsilon,
    )
    theta_initial, initial_centroids = calculate_initial_projection(
        Q_basis,
        R,
        y,
        classes,
    )
    theta = align_projection_to_simplex(
        theta_initial,
        initial_centroids,
    )

    projection_matrix = X_normalized.T @ theta
    return classes, theta, projection_matrix
