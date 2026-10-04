"""
rnn_engine.py
Simple and transparent mathematical recurrence for sequential data.
This file implements an RNN purely as a discrete recurrence relation:
  h_(t+1) = tanh(W @ h_t + U @ x_t + b)
  y_t     = V @ h_t + c

No deep learning frameworks or backpropagation are required.
All operations are standard matrix multiplications using NumPy.
"""

import numpy as np


def generate_recurrent_matrices(input_dim: int = 5, hidden_dim: int = 8, seed: int = 42):
    """
    Constructs the transition matrices for the dynamical system.
    
    Parameters:
      input_dim:  Number of features in each input vector x_t (dimension d).
      hidden_dim: Number of units in the hidden state vector h_t (dimension n).
      seed:       Random seed for reproducibility.
      
    Returns:
      W: Recurrent transition matrix in R^(n x n).
      U: Input weight matrix in R^(n x d).
      b: Hidden bias vector in R^n.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Initialize random recurrent transition matrix W
    W_raw = rng.normal(0.0, 1.0, size=(hidden_dim, hidden_dim))
    
    # 2. Scale W so its spectral radius (largest absolute eigenvalue) is < 1.
    # By Perron-Frobenius and dynamical stability theory, keeping the spectral
    # radius around 0.90 ensures bounded, non-explosive memory over long time horizons.
    eigenvalues = np.linalg.eigvals(W_raw)
    spectral_radius = np.max(np.abs(eigenvalues))
    W = (W_raw / spectral_radius) * 0.90
    
    # 3. Input projection matrix U mapping input features x_t into state space
    U = rng.normal(0.0, 0.35, size=(hidden_dim, input_dim))
    
    # 4. Constant bias vector b
    b = rng.normal(0.0, 0.05, size=(hidden_dim,))
    
    return W, U, b


def rnn_step(h_t: np.ndarray, x_t: np.ndarray, W: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Executes a single discrete dynamical update step:
      h_(t+1) = tanh(W @ h_t + U @ x_t + b)
      
    Mathematical explanation:
      - W @ h_t propagates previous memory forward in time.
      - U @ x_t injects new environmental observation at time t.
      - b shifts the operating point.
      - tanh squashes all activations into the bounded interval (-1, +1).
    """
    return np.tanh(W @ h_t + U @ x_t + b)


def run_rnn(X: np.ndarray, W: np.ndarray, U: np.ndarray, b: np.ndarray, V: np.ndarray = None, c: float = 0.0):
    """
    Iterates the dynamical system across all time steps in data matrix X.
    
    Parameters:
      X: Data matrix in R^(T x d) containing T sequential observations.
      W: Recurrent matrix in R^(n x n).
      U: Input matrix in R^(n x d).
      b: Hidden bias vector in R^n.
      V: Output readout vector in R^n (optional).
      c: Output scalar bias (optional).
      
    Returns:
      H: Hidden state matrix in R^(T x n) storing each state vector h_t.
      y_preds: Output predictions vector in R^T.
    """
    num_timesteps, feature_dim = X.shape
    hidden_dim = W.shape[0]
    
    # Pre-allocate hidden state matrix H
    H = np.zeros((num_timesteps, hidden_dim), dtype=np.float32)
    h_curr = np.zeros(hidden_dim, dtype=np.float32)
    
    # Sequential forward iteration
    for t in range(num_timesteps):
        h_curr = rnn_step(h_curr, X[t], W, U, b)
        H[t] = h_curr
        
    # Compute output prediction via linear readout: y = H @ V + c
    if V is not None:
        y_preds = (H @ V) + c
    else:
        y_preds = np.zeros(num_timesteps, dtype=np.float32)
        
    return H, y_preds


def fit_linear_readout(H: np.ndarray, y: np.ndarray):
    """
    Calculates the output prediction weights V using Least Squares:
      y approx H @ V + c
    
    This solves the normal equations: (H_aug^T @ H_aug) @ [V, c]^T = H_aug^T @ y
    using np.linalg.lstsq.
    """
    num_timesteps, hidden_dim = H.shape
    
    # Augment H with a column of ones for the scalar bias c
    H_augmented = np.hstack([H, np.ones((num_timesteps, 1))])
    
    # Solve least squares
    solution, residuals, rank, singular_values = np.linalg.lstsq(H_augmented, y, rcond=None)
    
    V = solution[:hidden_dim]
    c = float(solution[hidden_dim])
    
    return V, c
