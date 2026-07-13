"""Algorithm 1 of the revised LIM-NFST paper.

All arrays in this implementation store samples by rows. The paper writes the
normalized training matrix as ``X_tilde`` with samples by columns, so the paper
matrix ``X_tilde.T`` corresponds to ``X_norm`` below.
"""

from __future__ import annotations

import numpy as np


def get_E(y):
    """Return the n x c class-indicator matrix E."""
    y = np.asarray(y)
    classes = np.unique(y)
    E = np.zeros((len(y), len(classes)), dtype=np.float64)
    for column, cls in enumerate(classes):
        E[y == cls, column] = 1.0
    return E


def get_W_H(y):
    """Return the paper's within-class averaging matrix W and centering H."""
    y = np.asarray(y)
    n = len(y)
    W = np.zeros((n, n), dtype=np.float64)
    for cls in np.unique(y):
        indices = np.flatnonzero(y == cls)
        W[np.ix_(indices, indices)] = 1.0 / len(indices)
    H = np.full((n, n), 1.0 / n, dtype=np.float64)
    return W, H


def psi_eps_times(X_norm, matrix, eps):
    """Compute (Psi + eps I) @ matrix without constructing the n x n Psi."""
    X_norm = np.asarray(X_norm, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    return X_norm @ (X_norm.T @ matrix) + eps * matrix


def get_Q_w(X_norm, y, eps):
    """Algorithm 1 steps 2-3: Woodbury solution, row-centering and QR.

    The revised paper permits omitting the global ``1 / eps`` factor in Q:
    QR removes that common scale and omission avoids magnifying round-off.
    The returned subspace is therefore mathematically identical to Eq. 27.
    """
    X_norm = np.asarray(X_norm, dtype=np.float64)
    y = np.asarray(y)
    if X_norm.ndim != 2 or len(X_norm) != len(y):
        raise ValueError("X_norm must be 2-D with one row per label")
    if eps <= 0:
        raise ValueError("eps must be positive")

    E = get_E(y)
    n, d = X_norm.shape
    c = E.shape[1]
    if c < 2:
        raise ValueError("LIM-NFST requires at least two known classes")

    # Eq. 27 in row-major form:
    # Q = eps^-1 [E - Xn (eps I_d + Xn.T Xn)^-1 Xn.T E].
    rhs = X_norm.T @ E
    correction = np.linalg.solve(
        eps * np.eye(d, dtype=np.float64) + X_norm.T @ X_norm,
        rhs,
    )
    Q_without_global_scale = E - X_norm @ correction

    # E 1_c = 1_n creates one trivial constant direction. Algorithm 1 removes
    # it by centering every row across the c class-code columns.
    Q_centered = Q_without_global_scale - Q_without_global_scale.mean(
        axis=1, keepdims=True
    )
    required_rank = c - 1
    numerical_rank = int(np.linalg.matrix_rank(Q_centered))
    if numerical_rank < required_rank:
        raise ValueError(
            f"row-centered Q has rank {numerical_rank}; paper requires c-1={required_rank}"
        )

    Q_full, R_full = np.linalg.qr(Q_centered, mode="reduced")
    Q_w = Q_full[:, :required_rank]
    R = R_full[:required_rank, :]
    return Q_w, R


def _reduced_between_scatter(E_w, y):      # note
    """Compute E_w.T (W-H) E_w without materializing W or H."""
    E_w = np.asarray(E_w, dtype=np.float64)
    y = np.asarray(y)
    global_mean = E_w.mean(axis=0)
    scatter = np.zeros((E_w.shape[1], E_w.shape[1]), dtype=np.float64)
    for cls in np.unique(y):
        rows = E_w[y == cls]
        class_mean = rows.mean(axis=0)
        scatter += len(rows) * np.outer(class_mean, class_mean)
    scatter -= len(y) * np.outer(global_mean, global_mean)
    return 0.5 * (scatter + scatter.T)


def get_project(Q_w, R, y, n_components=None, *, return_eigenvalues=False):
    """Algorithm 1 steps 4-6: reduced scatter EVD and Theta_init."""
    Q_w = np.asarray(Q_w, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    y = np.asarray(y)
    c = len(np.unique(y))
    required = c - 1
    requested = required if n_components is None else int(n_components)
    if requested != required:
        raise ValueError(
            f"the revised paper fixes the projection dimension at c-1={required}"
        )
    if Q_w.shape[1] != required or R.shape != (required, c):
        raise ValueError(
            f"expected Q_w (n, {required}) and R ({required}, {c}); "
            f"received {Q_w.shape} and {R.shape}"
        )

    # R is (c-1) x c after removal of the trivial direction, so the paper's
    # R^-1 is its Moore-Penrose right inverse in an implementation.
    E_w = get_E(y) @ np.linalg.pinv(R)   # note
    reduced_scatter = _reduced_between_scatter(E_w, y)
    eigenvalues, eigenvectors = np.linalg.eigh(reduced_scatter)
    order = np.argsort(eigenvalues)[::-1]
    selected_values = eigenvalues[order[:required]]
    tolerance = (
        max(reduced_scatter.shape)
        * np.finfo(np.float64).eps
        * max(float(np.max(np.abs(eigenvalues))), 1.0)
    )
    if np.any(selected_values <= tolerance):
        raise ValueError(
            "reduced between-class scatter does not have c-1 positive eigenvalues: "
            f"{eigenvalues.tolist()}"
        )

    V = eigenvectors[:, order[:required]]
    theta_init = Q_w @ V
    if return_eigenvalues:
        return theta_init, selected_values
    return theta_init


def regular_simplex_targets(n_classes, delta):
    """Construct the (c-1) x c centered regular simplex T of edge delta."""
    c = int(n_classes)
    delta = float(delta)
    if c < 2:
        raise ValueError("a regular class simplex requires at least two vertices")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be a finite positive edge length")

    centering = np.eye(c) - np.ones((c, c), dtype=np.float64) / c
    values, vectors = np.linalg.eigh(centering)
    basis = vectors[:, values > 0.5]
    if basis.shape != (c, c - 1):
        raise RuntimeError("failed to construct the centered simplex basis")

    # Columns of basis.T have pairwise distance sqrt(2); rescale to delta.
    return (delta / np.sqrt(2.0)) * basis.T


def align_to_simplex(theta_init, X_norm, y, eps, delta):
    """Algorithm 1 steps 7-11: isometric centroid alignment.

    Returns final Theta, target base points (one row per class), initial
    centroid matrix M and alignment matrix A.
    """
    theta_init = np.asarray(theta_init, dtype=np.float64)
    X_norm = np.asarray(X_norm, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    dimension = len(classes) - 1
    if theta_init.shape != (len(y), dimension):
        raise ValueError(
            f"theta_init must have shape ({len(y)}, {dimension}); got {theta_init.shape}"
        )

    Y_init = psi_eps_times(X_norm, theta_init, eps)
    # Paper notation uses centroids as columns: M=[mu_1,...,mu_c].
    M = np.vstack([Y_init[y == cls].mean(axis=0) for cls in classes]).T
    if np.linalg.matrix_rank(M) < dimension:
        raise ValueError("initial centroid matrix M is not full row rank")

    T = regular_simplex_targets(len(classes), delta)
    gram = M @ M.T
    # A = T M.T (M M.T)^-1, evaluated with a linear solve.
    A = np.linalg.solve(gram, M @ T.T).T
    theta = theta_init @ A.T
    return theta, T.T, M, A


def cal_base_point(theta_mtrx, psi_eps, y):
    """Compute class centroids of Psi_eps @ Theta (diagnostic helper)."""
    theta_mtrx = np.asarray(theta_mtrx, dtype=np.float64)
    psi_eps = np.asarray(psi_eps, dtype=np.float64)
    y = np.asarray(y)
    projected = psi_eps @ theta_mtrx
    return np.vstack(
        [projected[y == cls].mean(axis=0) for cls in np.unique(y)]
    )
