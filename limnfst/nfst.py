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
    """Algorithm 1 steps 2-3: Woodbury solution, row-centering and QR."""
    X_norm = np.asarray(X_norm, dtype=np.float64)
    y = np.asarray(y)
    E = get_E(y)
    n, d = X_norm.shape
    c = E.shape[1]

    # Q = eps^-1 [E - Xn (eps I_d + Xn.T Xn)^-1 Xn.T E].
    
    correction = np.linalg.solve(
        eps * np.eye(d, dtype=np.float64) + X_norm.T @ X_norm,
        X_norm.T @ E,
    )
    Q = (E - X_norm @ correction) / eps

    Q_centered = Q - Q.mean(axis=1, keepdims=True)
    
    max_rank = c - 1

    Q_full, R_full = np.linalg.qr(Q_centered, mode="reduced")
    Q_w = Q_full[:, :max_rank]
    R = R_full[:max_rank, :]
    return Q_w, R


def between_scatter(E_w, y):      # alternative method to compute Sb
    """Compute E_w.T (W-H) E_w without materializing W or H for minimize memory usage."""
    E_w = np.asarray(E_w, dtype=np.float64)
    y = np.asarray(y)
    global_mean = E_w.mean(axis=0)
    scatter = np.zeros((E_w.shape[1], E_w.shape[1]), dtype=np.float64)
    for cls in np.unique(y):
        rows = E_w[y == cls]
        class_mean = rows.mean(axis=0)
        scatter += len(rows) * np.outer(class_mean, class_mean)
    scatter -= len(y) * np.outer(global_mean, global_mean)
    return 0.5 * (scatter + scatter.T)  # đảm bảo đối xứng



def get_project(Q_w, R, y, n_components=None, *, return_eigenvalues=False):
    """Algorithm 1 steps 4-6: reduced scatter EVD and Theta_init."""
    Q_w = np.asarray(Q_w, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    y = np.asarray(y)
    c = len(np.unique(y))
    required = c - 1
    
    R_pinv = np.linalg.pinv(R)
    E_w = get_E(y) @ R_pinv   
    reduced_scatter = between_scatter(E_w, y)    # Sb = E_w.T (W-H) E_w
    eigenvalues, eigenvectors = np.linalg.eigh(reduced_scatter)
    order = np.argsort(eigenvalues)[::-1]
    selected_values = eigenvalues[order[:required]]
    # tolerance = (             # kiểm tra đủ c - 1 eigenvalues > 0, chưa cần dùng
    #     max(reduced_scatter.shape)
    #     * np.finfo(np.float64).eps
    #     * max(float(np.max(np.abs(eigenvalues))), 1.0)
    # )

    V = eigenvectors[:, order[:required]]
    theta_init = Q_w @ V
    if return_eigenvalues:
        return theta_init, selected_values, V, R_pinv
    return theta_init


def regular_simplex_targets(n_classes, delta):
    """Construct the (c-1) x c centered regular simplex T of edge delta."""
    c = int(n_classes)
    delta = float(delta)

    centering = np.eye(c) - np.ones((c, c), dtype=np.float64) / c
    values, vectors = np.linalg.eigh(centering)
    basis = vectors[:, values > 0.5]

    # Columns of basis.T have pairwise distance sqrt(2); rescale to delta.
    return (delta / np.sqrt(2.0)) * basis.T


def align_to_simplex(theta_init, X_norm, y, eps, delta, initial_centroids=None):
    """Algorithm 1 steps 7-11: isometric centroid alignment.

    Returns final Theta, target base points (one row per class), initial
    centroid matrix M and alignment matrix A.
    """
    theta_init = np.asarray(theta_init, dtype=np.float64)
    X_norm = np.asarray(X_norm, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)

    if initial_centroids is None:
        Y_init = psi_eps_times(X_norm, theta_init, eps)
        # Paper notation uses centroids as columns: M=[mu_1,...,mu_c].
        M = np.vstack([Y_init[y == cls].mean(axis=0) for cls in classes]).T
    else:
        M = np.asarray(initial_centroids, dtype=np.float64)
        
    
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
