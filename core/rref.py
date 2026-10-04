"""
rref.py
Reduced Row Echelon Form (RREF) and Latent Dimension Dependency Analysis.
Calls SymPy's standard Matrix.rref() to compute:
  1. Reduced Row Echelon Form of hidden state samples
  2. Numerical rank
  3. Pivot columns (linearly independent representation dimensions)
  4. Free columns (redundant dimensions)
"""

import numpy as np
import sympy as sp


def compute_rref_sympy(M: np.ndarray):
    """
    Computes RREF using SymPy's exact rational row reduction.
    
    Returns:
      rref_matrix: Matrix in reduced row echelon form.
      pivot_cols: Indices of pivot columns (linearly independent basis features).
      free_cols: Indices of non-pivot columns (redundant features).
      rank: Total number of pivots.
    """
    rows, cols = M.shape
    
    # Convert numpy array to SymPy Matrix for standard RREF reduction
    sympy_matrix = sp.Matrix(np.round(M, 4))
    
    # Direct library call to compute RREF and pivot column indices
    rref_sym, pivot_cols = sympy_matrix.rref()
    
    # Convert back to float numpy array for downstream processing
    rref_matrix = np.array(rref_sym).astype(float)
    pivot_cols = list(pivot_cols)
    free_cols = [c for c in range(cols) if c not in pivot_cols]
    rank = len(pivot_cols)
    redundancy_pct = (len(free_cols) / cols) * 100.0 if cols > 0 else 0.0
    
    return {
        "rref_matrix": rref_matrix,
        "pivot_cols": pivot_cols,
        "free_cols": free_cols,
        "rank": rank,
        "redundancy_pct": round(redundancy_pct, 1)
    }


def analyze_latent_dependencies(H: np.ndarray, num_samples: int = 16):
    """
    Extracts sample hidden states across the timeline and performs RREF
    to evaluate whether certain hidden dimensions are redundant.
    """
    num_timesteps, hidden_dim = H.shape
    sample_indices = np.linspace(0, num_timesteps - 1, min(num_samples, num_timesteps), dtype=int)
    M_sample = H[sample_indices]
    
    result = compute_rref_sympy(M_sample)
    
    return {
        "sampled_timesteps": sample_indices.tolist(),
        "input_shape": [int(num_samples), int(hidden_dim)],
        "rank": result["rank"],
        "pivot_dimensions": result["pivot_cols"],
        "redundant_dimensions": result["free_cols"],
        "redundancy_pct": result["redundancy_pct"],
        "rref_preview": result["rref_matrix"][:min(10, num_samples)].tolist()
    }
