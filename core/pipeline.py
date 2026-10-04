"""
pipeline.py
Scientific research coordinator for SLATE.
Binds data representation, recurrent state generation, QR orthonormal basis,
least-squares surrogate modeling, spectral stability, and RREF redundancy analysis.
"""

import numpy as np
from data.dataset_loader import MatrixDataLoader
from core.rnn_engine import generate_recurrent_matrices, run_rnn, fit_linear_readout
from core.algebra import compute_orthonormal_basis, project_states, analyze_matrix_structure
from core.dynamics import fit_linear_surrogate_transition, simulate_surrogate_rollout
from core.spectral import compute_spectral_properties
from core.rref import analyze_latent_dependencies


class SLATEPipeline:
    """
    Main research pipeline connecting all mathematical steps:
      X (Data Matrix)
        -> H (State Matrix via h_(t+1) = tanh(Wh + Ux + b))
        -> Q, R (Orthonormal Basis via QR Decomposition)
        -> Z (Projected 2D & 3D Coordinates via Z = H @ Q_k)
        -> A (Linear Surrogate via Least Squares: A = (H_prev^+ @ H_next)^T)
        -> Lambda, v (Eigenvalues and Spectral Radius)
        -> RREF (Rank and Latent Redundancy via SymPy)
    """
    def __init__(self, data_path="data/air_quality.csv", hidden_dim=8):
        self.data_path = data_path
        self.hidden_dim = hidden_dim
        self.loader = MatrixDataLoader(data_path)
        self.results = {}
        
    def run_experiment(self, recurrent_scale: float = 1.0):
        """
        Executes the mathematical analysis pipeline.
        
        Parameters:
          recurrent_scale: Factor scaling the recurrent weights.
                           1.0 = baseline stable dynamics (rho approx 0.90)
                           >1.1 = induces instability / explosive modes (rho > 1.0)
        """
        # 1. Dataset Matrix Representation: X in R^(T x d) and target y in R^T
        X_raw = self.loader.X_raw
        X_norm = self.loader.X
        y_norm = self.loader.y
        y_raw = self.loader.y_raw
        num_timesteps, feature_dim = X_norm.shape
        
        matrix_X_info = analyze_matrix_structure(X_norm, name="Data Matrix X")
        
        # 2. Recurrent System Matrices
        W, U, b = generate_recurrent_matrices(
            input_dim=feature_dim,
            hidden_dim=self.hidden_dim,
            seed=42
        )
        
        # Apply scaling if requested to test stability behavior
        if recurrent_scale != 1.0:
            W = W * recurrent_scale
            
        # 3. Generate Hidden-State Sequence: h_(t+1) = tanh(W @ h_t + U @ x_t + b)
        H, _ = run_rnn(X_norm, W, U, b)
        
        # 4. Fit output prediction weights V using Least Squares
        V, c = fit_linear_readout(H, y_norm)
        _, y_preds_norm = run_rnn(X_norm, W, U, b, V, c)
        
        # Denormalize target predictions back to physical units (ug/m^3)
        y_preds = (y_preds_norm * self.loader.sigma_y) + self.loader.mu_y
        
        matrix_H_info = analyze_matrix_structure(H, name="Hidden-State Matrix H")
        
        # 5. Orthonormal Basis via QR Decomposition and Subspace Projection
        # 2D projection
        basis_2d = compute_orthonormal_basis(H, k=2)
        proj_2d = project_states(H, basis_2d["basis_matrix_B"])
        
        # 3D projection
        basis_3d = compute_orthonormal_basis(H, k=3)
        proj_3d = project_states(H, basis_3d["basis_matrix_B"])
        
        # 6. Least-Squares Linear Surrogate Dynamics: H_next approx H_prev @ A^T
        dyn_results = fit_linear_surrogate_transition(H)
        A = dyn_results["A"]
        
        # Project surrogate 1-step predictions into 2D for direct comparison
        H_surrogate_1step = dyn_results["H_pred_1step"]
        proj_surrogate_2d = (H_surrogate_1step - basis_2d["mean_vector"]) @ basis_2d["basis_matrix_B"]
        
        # Multi-step autonomous rollout
        H_rollout = simulate_surrogate_rollout(A, H[0], steps=min(60, num_timesteps))
        proj_rollout_2d = (H_rollout - basis_2d["mean_vector"]) @ basis_2d["basis_matrix_B"]
        
        # 7. Spectral Analysis of Linear Transition Matrix A
        spectral_info = compute_spectral_properties(A)
        
        # 8. Latent Dependency and Redundancy Analysis via RREF (SymPy)
        rref_info = analyze_latent_dependencies(H, num_samples=16)
        
        # 9. Predictive Evaluation Metrics
        pred_rmse = float(np.sqrt(np.mean((y_raw - y_preds) ** 2)))
        pred_mae = float(np.mean(np.abs(y_raw - y_preds)))
        y_variance = float(np.var(y_raw))
        pred_r2 = float(1.0 - (np.mean((y_raw - y_preds) ** 2) / (y_variance + 1e-12)))
        
        # Package full mathematical results
        self.results = {
            "meta": {
                "timesteps_T": int(num_timesteps),
                "feature_dim_d": int(feature_dim),
                "hidden_dim_n": int(self.hidden_dim),
                "target_name": self.loader.target_col,
                "features": self.loader.feature_cols
            },
            "matrices": {
                "X_info": matrix_X_info,
                "H_info": matrix_H_info,
                "W_shape": list(W.shape),
                "U_shape": list(U.shape),
                "A_transition": A.tolist()
            },
            "projection": {
                "variance_retained_2d": proj_2d["variance_retained_pct"],
                "variance_retained_3d": proj_3d["variance_retained_pct"],
                "orthonormality_error": basis_2d["orthonormality_error"],
                "coords_2d": proj_2d["Z"].tolist(),
                "coords_3d": proj_3d["Z"].tolist(),
                "reconstruction_residuals": proj_2d["residuals"]
            },
            "dynamics": {
                "r2_fit_score": dyn_results["r2_fit_score"],
                "mean_step_error": dyn_results["mean_step_error"],
                "step_errors": dyn_results["step_errors"],
                "coords_surrogate_2d": proj_surrogate_2d.tolist(),
                "coords_rollout_2d": proj_rollout_2d.tolist()
            },
            "spectral": spectral_info,
            "rref": rref_info,
            "predictions": {
                "actual_y": y_raw.tolist(),
                "predicted_y": y_preds.tolist(),
                "rmse": round(pred_rmse, 3),
                "mae": round(pred_mae, 3),
                "r2": round(pred_r2, 4)
            },
            "raw_samples": {
                "X_preview": X_raw[:100].tolist(),
                "H_preview": H[:100].tolist()
            }
        }
        return self.results
