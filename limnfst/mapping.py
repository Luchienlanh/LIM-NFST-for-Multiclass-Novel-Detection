import numpy as np


def center_normalize(X):
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - X_mean
    norm = np.linalg.norm(X_centered, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    X_norm = X_centered / norm
    return X_norm

# def get_Psi(X, epsilon=1e-4):
#     X_normalize = center_normalize(X)
    
#     Psi = X_normalize @ X_normalize.T
    
#     I = np.eye(Psi.shape[0])
    
#     Psi_eps = Psi + epsilon * I
#     return Psi_eps 
    