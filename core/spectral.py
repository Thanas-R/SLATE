"""
spectral.py
Spectral analysis of the discrete linear transition operator A:
  A @ v_i = lambda_i * v_i

This module computes eigenvalues, eigenvectors, and the spectral radius rho(A),
evaluating dynamical stability relative to the complex unit circle |lambda| = 1.
"""

import numpy as np


def compute_spectral_properties(A: np.ndarray):
    """
    Computes the eigen-decomposition and spectral stability of transition matrix A.
    
    Mathematical theory:
      In a discrete dynamical system: h_(t+1) = A @ h_t,
      the state after t iterations is given by: h_t = A^t @ h_0.
      If A is diagonalizable (A = V @ Lambda @ V^(-1)), then:
        h_t = sum_i c_i * (lambda_i)^t * v_i
        
      The magnitude |lambda_i| controls asymptotic behavior:
        1. |lambda_i| < 1 : Mode decays to zero exponentially as t -> infinity (stable sink).
        2. |lambda_i| = 1 : Mode maintains persistent oscillation without decay (marginally stable).
        3. |lambda_i| > 1 : Mode amplifies exponentially (unstable source / explosion).
        
      The spectral radius is defined as:
        rho(A) = max_i |lambda_i|
      The system is asymptotically stable if and only if rho(A) < 1.
    """
    # Direct standard library eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Sort eigenvalues in descending order of magnitude
    sort_indices = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]
    
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    
    # Classify dynamical stability
    if spectral_radius < 0.99:
        stability_class = "STABLE (Decaying)"
        stability_desc = "All eigenvalues reside strictly inside the unit circle (|lambda| < 1). Perturbations decay asymptotically."
    elif spectral_radius <= 1.01:
        stability_class = "MARGINALLY STABLE (Attractor)"
        stability_desc = "Dominant eigenvalue rests on the unit circle boundary (|lambda| approx 1.0). Sustained periodic memory."
    else:
        stability_class = "UNSTABLE (Expanding)"
        stability_desc = "One or more eigenvalues breach the unit circle (|lambda| > 1). Dynamics amplify exponentially."
        
    modes = []
    for i, lam in enumerate(eigenvalues):
        real_part = float(np.real(lam))
        imag_part = float(np.imag(lam))
        magnitude = float(np.abs(lam))
        angle_rad = float(np.angle(lam))
        angle_deg = float(np.degrees(angle_rad))
        
        # Oscillation period in discrete steps: T = 2*pi / |theta|
        period = (2.0 * np.pi / abs(angle_rad)) if abs(angle_rad) > 1e-4 else float('inf')
        
        if magnitude < 0.99:
            mode_type = "Contracting"
        elif magnitude <= 1.01:
            mode_type = "Persistent"
        else:
            mode_type = "Expanding"
            
        modes.append({
            "index": i + 1,
            "real": round(real_part, 4),
            "imag": round(imag_part, 4),
            "magnitude": round(magnitude, 4),
            "angle_deg": round(angle_deg, 1),
            "period_steps": round(period, 1) if period != float('inf') else "Aperiodic",
            "mode_type": mode_type
        })
        
    trace_val = float(np.trace(A))
    det_val = float(np.linalg.det(A))
    
    return {
        "spectral_radius": round(spectral_radius, 4),
        "stability_class": stability_class,
        "stability_desc": stability_desc,
        "trace": round(trace_val, 4),
        "determinant": round(det_val, 4),
        "modes": modes
    }
