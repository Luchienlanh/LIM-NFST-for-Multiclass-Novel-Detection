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

# def get_Q_w(X_norm, y, eps):
#     if eps <= 0:
#         raise ValueError("eps must be positive")

#     E = get_E(y)
#     n, d = X_norm.shape
    
#     # --- Bước 1: Giải hệ Woodbury (Giữ nguyên dùng solve là tốt nhất) ---
#     G = X_norm.T @ X_norm
#     R = X_norm.T @ E
#     A = np.linalg.solve(G + eps * np.eye(d), R)
    
#     # --- Bước 2: Loại bỏ phép chia cho eps ---
#     # Thay vì: Q = (E - X_norm @ A) / eps
#     # Ta dùng:
#     Q = E - X_norm @ A 
#     # Lý do: Hệ số 1/eps là hằng số tỉ lệ, sẽ bị triệt tiêu bởi QR. 
#     # Loại bỏ nó giúp tránh phóng đại sai số máy tính lên 1 triệu lần (nếu eps=1e-6).

#     # --- Bước 3: Sửa lại Centering (Chuẩn hóa theo Cột) ---
#     # Thay vì: Q = Q - Q.mean(axis=1, keepdims=True) (Trung bình hàng)
#     # Ta dùng:
#     Q = Q - Q.mean(axis=0) 
#     # Lý do: Để trực giao với vector hằng số 1_n (điều kiện Z_t_perp), 
#     # tổng mỗi cột của Q phải bằng 0. Do đó ta phải trừ trung bình cột (axis=0).

#     # --- Bước 4: QR và trích xuất Rank ---
#     Q_w, upper = np.linalg.qr(Q, mode='reduced')
#     diag = np.abs(np.diag(upper))
    
#     if diag.size == 0:
#         return np.empty((n, 0))
    
#     # Tính toán ngưỡng sai số để xác định rank thực tế
#     tol = max(Q.shape) * diag.max() * np.finfo(np.float64).eps
    
#     # NFST bắt buộc lấy tối đa c-1 chiều phân biệt
#     rank = min(int(np.sum(diag > tol)), len(np.unique(y)) - 1)
    
#     return Q_w[:, :rank], upper[:rank, :]

def get_Q_w(X_norm, y, eps):
    if eps <= 0:
        raise ValueError("eps must be positive")

    E = get_E(y)
    E_centered = E - E.mean(axis=0, keepdims=True)
    n, d = X_norm.shape

    G = X_norm.T @ X_norm
    R = X_norm.T @ E_centered
    A = np.linalg.solve(G + eps * np.eye(d), R)

    Q = (E_centered - X_norm @ A) / eps

    Q_w, upper = np.linalg.qr(Q, mode="reduced")

    diag = np.abs(np.diag(upper))
    if diag.size == 0:
        return np.empty((n, 0)), np.empty((0, E.shape[1]))

    tol = max(Q.shape) * diag.max() * np.finfo(np.float64).eps
    rank = min(int(np.sum(diag > tol)), len(np.unique(y)) - 1)

    return Q_w[:, :rank], upper[:rank, :]


def _psi_eps_times(X_norm, V, eps):
    return X_norm @ (X_norm.T @ V) + eps * V


def _projected_between_scatter(Z, y):
    y = np.asarray(y)
    Z = np.asarray(Z, dtype=np.float64)
    n, r = Z.shape
    global_mean = Z.mean(axis=0)
    scatter = np.zeros((r, r), dtype=np.float64)

    for cls in np.unique(y):
        Z_cls = Z[y == cls]
        mean = Z_cls.mean(axis=0)
        scatter += len(Z_cls) * np.outer(mean, mean)

    scatter -= n * np.outer(global_mean, global_mean)
    return scatter


def get_project(Q_w, X_norm, y, eps, n_components=None):
    if n_components is None:
        n_components = Q_w.shape[1]
    n_components = min(n_components, Q_w.shape[1])
    
    if n_components == 0:
        return np.empty((Q_w.shape[0], 0))

    E_w = _psi_eps_times(X_norm, Q_w, eps)
    S_b_prj = _projected_between_scatter(E_w, y)
    S_b_prj = (S_b_prj + S_b_prj.T) / 2.0
    
    eigvals, eigvecs = np.linalg.eigh(S_b_prj)
    
    idx = np.argsort(eigvals)[::-1]
    top_idx = idx[:n_components]
    
    V = eigvecs[:, top_idx]
    L = eigvals[top_idx]
    
    # 4. WHITENING: Đây là chìa khóa để cách đều
    # A = V * (1 / sqrt(L))
    A = V @ np.diag(1.0 / np.sqrt(L + 1e-12)) # Thêm eps nhỏ để tránh chia cho 0
    
    theta_mtrx = Q_w @ A
    # idx = np.argsort(eigvals)[::-1]
    # A = eigvecs[:, idx[:n_components]]
    
    # theta_mtrx = Q_w @ A
    
    return theta_mtrx

def get_project(Q_w, R, y, dist=1.0):
    classes = np.unique(y)
    c = len(classes)
    
    # 1. Tạo Target Points cách đều (Simplex)
    # Tạo c điểm cách đều nhau trong không gian c-1 chiều
    I = np.eye(c)
    targets = I - np.mean(I, axis=0) # Centering
    targets, _ = np.linalg.qr(targets.T) # Rút về c-1 chiều trực chuẩn
    targets = targets.T * dist # Chỉnh khoảng cách theo ý muốn
    
    # 2. Tính tọa độ Centroids hiện tại trong Null Space
    E = get_E(y)
    # Tọa độ centroids trong Ew (không gian chưa trực chuẩn)
    # M = (E.T @ E)^-1 @ E.T @ Ew @ A_qr (từ QR của Q)
    # Đơn giản nhất: Lấy trung bình các mẫu đã chiếu lên Q_w theo từng lớp
    E_w = E @ np.linalg.pinv(R)
    current_centroids = []
    for cls in classes:
        m = E_w[y == cls].mean(axis=0)
        current_centroids.append(m)
    current_centroids = np.vstack(current_centroids).T # (c-1) x c
    
    # 3. Tính ma trận ánh xạ A
    # A @ current_centroids = targets => A = targets @ pinv(current_centroids)
    A = targets @ np.linalg.pinv(current_centroids)
    
    # Ma trận chiếu cuối cùng
    return Q_w @ A.T

def cal_base_point(theta_mtrx, X, y):
    classes = np.unique(y)
    base_points = []

    for cls in classes:
        class_mean = X[y == cls].mean(axis=0)
        base_points.append((X @ class_mean) @ theta_mtrx)

    return np.vstack(base_points)
    
    


    


    
