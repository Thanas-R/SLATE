"""
algebra.py
Linear algebra core:
  1. Matrix properties (Rank, Norms, Condition number)
  2. Orthonormal basis computation via QR decomposition (Gram-Schmidt)
  3. Orthogonal subspace projection (P = B @ B^T)
  4. Dimensionality reduction from n dimensions to 2D and 3D

This module uses standard library functions from NumPy to minimize
boilerplate and focus on mathematical clarity.
"""

import numpy as np


def compute_orthonormal_basis(H: np.ndarray, k: int = 3):
    """
    Constructs an orthonormal basis B in R^(n x k) for the subspace occupied by hidden states.
    
    Mathematical theory:
      1. Center the state vectors: H_centered = H - mean(H).
      2. The covariance or scatter of state vectors is proportional to H_centered^T @ H_centered.
      3. Using QR decomposition on H_centered^T in R^(n x T):
           H_centered^T = Q @ R
         where Q in R^(n x n) is an orthogonal matrix satisfying Q^T @ Q = I_n,
         and R is upper triangular.
      4. The first k columns of Q form an orthonormal basis matrix B in R^(n x k):
           B = Q[:, :k]
           B^T @ B = I_k
    """
    num_timesteps, hidden_dim = H.shape
    mean_vector = np.mean(H, axis=0, keepdims=True)
    H_centered = H - mean_vector
    
    # Standard library QR decomposition
    Q, R = np.linalg.qr(H_centered.T)
    
    # Select the top k orthonormal basis vectors
    k = min(k, hidden_dim)
    B = Q[:, :k]
    
    # Verify orthonormality: ||B^T @ B - I_k|| should be near machine epsilon (1e-15)
    identity_k = np.eye(k)
    orthonormality_error = float(np.linalg.norm(B.T @ B - identity_k))
    
    return {
        "basis_matrix_B": B,
        "mean_vector": mean_vector.squeeze(),
        "orthonormality_error": orthonormality_error,
        "k": k
    }


def project_states(H: np.ndarray, B: np.ndarray):
    """
    Projects high-dimensional hidden state vectors onto a k-dimensional subspace.
    
    Mathematical theory:
      - Given state h_t in R^n and orthonormal basis B in R^(n x k):
          Coordinates in subspace:  z_t = B^T @ (h_t - mean) in R^k
          Reconstructed vector:     h_hat_t = mean + B @ z_t in R^n
      - The orthogonal projection operator P in R^(n x n) is:
          P = B @ (B^T @ B)^(-1) @ B^T = B @ B^T  (since B^T @ B = I_k)
        P is symmetric (P^T = P) and idempotent (P^2 = P).
      - Energy / Variance Retained:
          Ratio of projected variance to total variance:
          Percentage = (sum ||z_t||^2 / sum ||h_t - mean||^2) * 100%
    """
    num_timesteps, hidden_dim = H.shape
    mean_vector = np.mean(H, axis=0, keepdims=True)
    H_centered = H - mean_vector
    
    # 1. Coordinate projection: Z in R^(T x k)
    Z = H_centered @ B
    
    # 2. Reconstructed states back in original space R^(T x n)
    H_reconstructed = Z @ B.T + mean_vector
    
    # 3. Residual error per timestep: e_t = ||h_t - h_hat_t||_2
    residuals = np.linalg.norm(H - H_reconstructed, axis=1)
    mean_residual = float(np.mean(residuals))
    
    # 4. Variance retained
    total_variance = float(np.sum(H_centered ** 2))
    projected_variance = float(np.sum(Z ** 2))
    variance_retained_pct = (projected_variance / (total_variance + 1e-12)) * 100.0
    
    return {
        "Z": Z,
        "residuals": residuals.tolist(),
        "mean_residual": round(mean_residual, 5),
        "variance_retained_pct": round(variance_retained_pct, 2)
    }


def analyze_matrix_structure(M: np.ndarray, name: str = "Matrix"):
    """
    Calculates fundamental linear algebraic properties of matrix M:
      - Rank (number of linearly independent rows / columns)
      - Frobenius Norm: ||M||_F = sqrt(sum M_ij^2)
      - Spectral Norm: ||M||_2 = maximum singular value
      - Condition Number: kappa(M) = sigma_max / sigma_min (measures numerical sensitivity)
    """
    rows, cols = M.shape
    
    # Direct library calls for linear algebra metrics
    rank = int(np.linalg.matrix_rank(M))
    frobenius_norm = float(np.linalg.norm(M, 'fro'))
    
    # Singular Value Decomposition (SVD) for spectral norm and singular spectrum
    U, singular_values, Vt = np.linalg.svd(M, full_matrices=False)
    spectral_norm = float(singular_values[0]) if len(singular_values) > 0 else 0.0
    condition_number = float(np.linalg.cond(M)) if len(singular_values) > 0 else 1.0
    
    return {
        "name": name,
        "shape": [rows, cols],
        "rank": rank,
        "full_rank": bool(rank == min(rows, cols)),
        "frobenius_norm": round(frobenius_norm, 4),
        "spectral_norm": round(spectral_norm, 4),
        "condition_number": round(condition_number, 2),
        "singular_values": [round(float(s), 4) for s in singular_values[:8]]
    }
