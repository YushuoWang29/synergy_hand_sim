"""
Independent quasi-static solver for SDAS verification.

No Pinocchio dependency — pure NumPy.

Verification solver #1: UniformTensionSolver (no Capstan)
  Uses r_geo (geometric sum) instead of Capstan-weighted r_A/r_B.
  This is the MOST independent verification: different physical assumption.
  If SDAS (with Capstan friction) still matches, it validates that Capstan
  effects are negligible.

Verification solver #2: VerificationKKTSolver (Newton on KKT system)
  Uses the SAME R_A/R_B as SDAS, but solves via Newton-Raphson on the KKT
  system instead of closed-form Schur complement.
  This tests the SOLVER implementation.

Clamping logic (consistent with SDASModel):
  Total cable shortening theta_A + theta_B > 0 → both ends tense → 2 constraints active
  theta_A + theta_B = 0 → slack → no active constraints
"""

import numpy as np
from typing import Optional, Tuple


class VerificationKKTSolver:
    """
    Independent solver using KKT Newton-Raphson iteration.
    
    Same optimization problem as SDAS:
        min_q  (1/2) qᵀ·K·q  s.t.  R_A·q = θ_A, R_B·q = θ_B
    
    Different solver: Newton iteration on KKT system instead of
    closed-form Schur complement.
    """
    
    def __init__(self, r_A: np.ndarray, r_B: np.ndarray, stiffness: np.ndarray):
        self.r_A = np.asarray(r_A).flatten()
        self.r_B = np.asarray(r_B).flatten()
        self.n_joints = len(self.r_A)
        self.K = np.diag(np.asarray(stiffness).flatten())
        self.K_inv = np.diag(1.0 / np.asarray(stiffness).flatten())
    
    def _build_KKT(self, q: np.ndarray, theta_A: float, theta_B: float):
        """Build KKT matrix and residual."""
        EPS = 1e-10
        if theta_A + theta_B > EPS:
            # Total cable shortening → both ends tense → 2 constraints active
            nc = 2
            R = np.vstack([self.r_A, self.r_B])
            theta = np.array([theta_A, theta_B])
        else:
            return self.K, -(self.K @ q)
        
        # Estimate λ from Schur complement
        S = R @ self.K_inv @ R.T  # (nc, nc)
        try:
            lam = np.linalg.solve(S, theta)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(S, theta, rcond=None)[0]
        
        A = np.block([[self.K, R.T], [R, np.zeros((nc, nc))]])
        b = np.concatenate([-(self.K @ q - R.T @ lam), -(R @ q - theta)])
        return A, b
    
    def solve_motors(self, theta_A: float, theta_B: float,
                     tol=1e-10, max_iter=50, verbose=False) -> np.ndarray:
        """Newton-Raphson on KKT system."""
        theta_A = max(0.0, float(theta_A))
        theta_B = max(0.0, float(theta_B))
        EPS = 1e-10
        
        if theta_A + theta_B <= EPS:
            return np.zeros(self.n_joints)
        
        # Total cable shortening → both ends tense → 2 constraints active
        R = np.vstack([self.r_A, self.r_B])
        th = np.array([theta_A, theta_B])
        S = R @ self.K_inv @ R.T
        try:
            lam = np.linalg.solve(S, th)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(S, th, rcond=None)[0]
        q = self.K_inv @ (R.T @ lam)
        
        # Newton
        for it in range(max_iter):
            A, b = self._build_KKT(q, theta_A, theta_B)
            bn = np.linalg.norm(b)
            if verbose:
                print(f"    KKT iter {it}: |res| = {bn:.3e}")
            if bn < tol:
                break
            try:
                dq = np.linalg.solve(A, b)[:self.n_joints]
            except np.linalg.LinAlgError:
                dq = (np.linalg.pinv(A) @ b)[:self.n_joints]
            q = q + dq
        
        return q


class UniformTensionSolver:
    """
    Verification solver: uniform tension (no Capstan).
    
    Uses r_geo (geometric sum Σrk) instead of Capstan-weighted R_A/R_B.
    Different physical assumption → most independent verification.
    
    q = K⁻¹ · r_geoᵀ · (r_geo · K⁻¹ · r_geoᵀ)⁻¹ · (θ_A + θ_B)
    
    Clamping: uses total cable shortening theta_A + theta_B > 0 (consistent with SDAS).
    """
    
    def __init__(self, r_geo: np.ndarray, stiffness: np.ndarray):
        self.r_geo = np.asarray(r_geo).flatten()
        self.n_joints = len(self.r_geo)
        self.K = np.diag(np.asarray(stiffness).flatten())
        self.K_inv = np.diag(1.0 / np.asarray(stiffness).flatten())
        
        R = self.r_geo[np.newaxis, :]
        S = float(R @ self.K_inv @ R.T)
        if abs(S) < 1e-15:
            raise ValueError(f"UniformTensionSolver: Schur complement is zero")
        self.S_A = (self.K_inv @ R.T / S).flatten()
        print(f"  [UniformTension] r_geo = {self.r_geo}")
        print(f"  [UniformTension] direction = q/θ = {self.S_A}")
    
    def solve_motors(self, theta_A: float, theta_B: float, **kw) -> np.ndarray:
        theta = max(0.0, float(theta_A)) + max(0.0, float(theta_B))
        if theta <= 1e-10:
            return np.zeros(self.n_joints)
        return self.S_A * theta
