import numpy as np


def get_E(y):
    y = np.asarray(y)
    classes = np.unique(y)
    E = np.zeros((len(y), len(classes)), dtype=np.float64)
    
    for i, cls in enumerate(classes):
        E[y == cls, i] = 1.0
        
    return E

def get_W_H(y):
    y = np.array(y)
    n = len(y)
    W = np.zeros((n, n), dtype=float)
    
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        W[np.ix_(idx, idx)] = 1 / len(idx)
        
    H = np.ones((n, n), dtype=float) / n
    
    return W, H

def within_scatter(y, Psi_eps):
    W, _ = get_W_H(y)
    I = np.eye(Psi_eps.shape[0])
    return Psi_eps @ (I - W) @ Psi_eps.T
    
def between_scatter(y, Psi_eps):
    W, H = get_W_H(y)
    return Psi_eps @ (W - H) @ Psi_eps.T

# def get_Z(S):
#     S = np.asarray(S, dtype=np.float64)
#     R, pivot_cols = _rref(S)
#     rank = len(pivot_cols)

#     Z_raw = _nullspace_basis_from_rref(R, pivot_cols)
#     Z = _modified_gram_schmidt_columns(Z_raw)  # null space
#     Z_perp = _modified_gram_schmidt_columns(R[:rank].T)  # anti null space
#     return Z, Z_perp

def get_Q_w(X_norm, y, eps):
    if eps <= 0:
        raise ValueError("eps must be positive")
    
    E = get_E(y)
    n, d = X_norm.shape
    
    G = X_norm.T @ X_norm
    R = X_norm.T @ E
    A = np.linalg.solve(G + eps * np.eye(d), R)
    
    Q = (E - X_norm @ A) / eps
    Q = Q - Q.mean(axis=1, keepdims=True)
    
    Q_w, upper = np.linalg.qr(Q, mode='reduced')
    diag = np.abs(np.diag(upper))
    
    if diag.size == 0:
        return np.empty((n, 0))
    
    tol = max(Q.shape) * diag.max() * np.finfo(np.float64).eps
    rank = min(int(np.sum(diag > tol)), len(np.unique(y)) - 1)
    
    return Q_w[:, :rank], upper[:rank, :]


def get_project(Q_w, R, y, n_components=None):
    if n_components is None:
        n_components = Q_w.shape[1]
    n_components = min(n_components, Q_w.shape[1])
    
    if n_components == 0:
        return np.empty((Q_w.shape[0], 0))

    W, H = get_W_H(y)
    E = get_E(y)
    E_w = E @ np.linalg.pinv(R)
    S_b_prj = E_w.T @ (W - H) @ E_w
    S_b_prj = (S_b_prj + S_b_prj.T) / 2.0
    
    eigvals, eigvecs = np.linalg.eigh(S_b_prj)
    idx = np.argsort(eigvals)[::-1]
    A = eigvecs[:, idx[:n_components]]
    
    theta_mtrx = Q_w @ A
    
    return theta_mtrx

def cal_base_point(theta_mtrx, X, y):
    classes = np.unique(y)
    base_points = []

    for cls in classes:
        class_mean = X[y == cls].mean(axis=0)
        base_points.append((X @ class_mean) @ theta_mtrx)

    return np.vstack(base_points)
    
    


    


    
