"""
dynamics.py
Discrete dynamical system modeling using Least-Squares estimation:
  Actual nonlinear neural system: h_(t+1) = tanh(W @ h_t + U @ x_t + b)
  Linear surrogate model:         h_(t+1) approx A @ h_t

This module finds the optimal linear transition operator A by solving
the overdetermined system: H_next approx H_prev @ A^T
via NumPy's least-squares solver (Moore-Penrose pseudoinverse).
"""

import numpy as np


def fit_linear_surrogate_transition(H: np.ndarray):
    """
    Fits a linear discrete dynamical operator A in R^(n x n) using Least Squares.
    
    Mathematical formulation:
      Let consecutive hidden states be partitioned into:
        H_prev = H[0 : T-1] in R^((T-1) x n)  (antecedent states)
        H_next = H[1 : T]   in R^((T-1) x n)  (consequent states)
        
      We seek matrix A in R^(n x n) minimizing the Frobenius error:
        min_A ||H_next - H_prev @ A^T||_F^2
        
      From the normal equations:
        H_prev^T @ H_prev @ A^T = H_prev^T @ H_next
        A^T = (H_prev^T @ H_prev)^(-1) @ H_prev^T @ H_next = H_prev^+ @ H_next
        A   = (H_prev^+ @ H_next)^T
      where H_prev^+ is the Moore-Penrose pseudoinverse.
    """
    num_timesteps, hidden_dim = H.shape
    H_prev = H[:-1]  # shape (T-1, n)
    H_next = H[1:]   # shape (T-1, n)
    
    # Direct library call to solve least-squares overdetermined system
    sol, _, _, _ = np.linalg.lstsq(H_prev, H_next, rcond=None)
    A = sol.T  # shape (n x n)
    
    # 1-step prediction across the sequence: h_pred = (A @ h_t^T)^T = H_prev @ A^T
    H_pred_1step = H_prev @ A.T
    
    # Residual error per step: e_t = ||h_(t+1) - A @ h_t||_2
    step_errors = np.linalg.norm(H_next - H_pred_1step, axis=1)
    mean_step_error = float(np.mean(step_errors))
    
    # Coefficient of determination (R^2): proportion of variance explained by linear model
    total_ss = np.sum((H_next - np.mean(H_next, axis=0)) ** 2)
    residual_ss = np.sum((H_next - H_pred_1step) ** 2)
    r2_fit = float(1.0 - (residual_ss / (total_ss + 1e-12)))
    
    return {
        "A": A,
        "step_errors": step_errors.tolist(),
        "mean_step_error": round(mean_step_error, 5),
        "r2_fit_score": round(r2_fit, 4),
        "H_pred_1step": H_pred_1step
    }


def simulate_surrogate_rollout(A: np.ndarray, h_0: np.ndarray, steps: int = 50):
    """
    Simulates recursive autonomous forward rollout using only the linear surrogate:
      h_(t+1) = A @ h_t
    This reveals the asymptotic stability and trajectory of the linearized model.
    """
    hidden_dim = len(h_0)
    rollout = np.zeros((steps, hidden_dim), dtype=np.float32)
    h_curr = h_0.copy()
    
    for t in range(steps):
        rollout[t] = h_curr
        h_curr = A @ h_curr
        
    return rollout
