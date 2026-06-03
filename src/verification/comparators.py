"""
Comparison utilities for SDAS verification.

Provides:
    - ComparisonResult: structured data class for comparison results
    - compare_at_sigma: run SDAS vs verification solver at each sigma value
"""

import numpy as np
from typing import List, Optional, Dict


class ComparisonResult:
    """
    Stores the comparison between SDAS and verification solver.
    """
    
    def __init__(
        self,
        sigma_values: np.ndarray,
        q_sdas: np.ndarray,
        q_verif: np.ndarray,
        joint_names: List[str],
        description: str = "",
    ):
        """
        Parameters
        ----------
        sigma_values : ndarray, shape (n_steps,)
        q_sdas : ndarray, shape (n_steps, n_joints)
        q_verif : ndarray, shape (n_steps, n_joints)
        joint_names : list of str
        description : str
        """
        self.sigma_values = sigma_values
        self.q_sdas = q_sdas
        self.q_verif = q_verif
        self.joint_names = joint_names
        self.description = description
        self.n_joints = len(joint_names)
        
        # Per-joint errors
        self.max_errors = np.max(np.abs(q_sdas - q_verif), axis=0)
        self.mean_errors = np.mean(np.abs(q_sdas - q_verif), axis=0)
        self.rmse = np.sqrt(np.mean((q_sdas - q_verif)**2, axis=0))
        
        # Overall metrics
        self.global_max_error = np.max(self.max_errors)
        self.global_mean_error = np.mean(self.mean_errors)
        self.global_rmse = np.sqrt(np.mean((q_sdas - q_verif)**2))
    
    def summary(self) -> str:
        """Text summary of comparison."""
        lines = [
            "=" * 60,
            "Comparison Summary",
            "=" * 60,
            f"  Method: {self.description}",
            f"  Sigma range: [{self.sigma_values[0]:.3f}, {self.sigma_values[-1]:.3f}]",
            f"  Steps: {len(self.sigma_values)}",
            f"  Joints: {self.n_joints}",
            "",
            "--- Overall Metrics ---",
            f"  Global max error:  {self.global_max_error:.6f} rad ({np.degrees(self.global_max_error):.4f}°)",
            f"  Global mean error: {self.global_mean_error:.6f} rad ({np.degrees(self.global_mean_error):.4f}°)",
            f"  Global RMSE:       {self.global_rmse:.6f} rad ({np.degrees(self.global_rmse):.4f}°)",
            "",
            "--- Per-Joint Max Errors (sorted descending) ---",
        ]
        
        # Sort by max error descending
        sorted_idx = np.argsort(-self.max_errors)
        for idx in sorted_idx:
            deg = np.degrees(self.max_errors[idx])
            lines.append(
                f"  {self.joint_names[idx]:20s}: "
                f"max={self.max_errors[idx]:.6f} rad ({deg:.4f}°), "
                f"rmse={self.rmse[idx]:.6f} rad"
            )
        
        return "\n".join(lines)
    
    def max_joint_angle_table(self) -> str:
        """
        Generate table comparing max joint angles at highest sigma.
        Useful for checking if the solution "shape" (which joints move most)
        is consistent between methods.
        """
        i = -1  # last sigma value
        sdas_max = np.max(np.abs(self.q_sdas[i]))
        verif_max = np.max(np.abs(self.q_verif[i]))
        
        lines = [
            f"--- Max Joint Angles at σ={self.sigma_values[i]:.3f} ---",
            f"  {'Joint':20s} {'SDAS (rad)':>12s} {'Verif (rad)':>12s} {'Diff (rad)':>12s} {'Diff/Deg':>10s}",
            "  " + "-" * 66,
        ]
        
        for j in range(self.n_joints):
            diff = np.abs(self.q_sdas[i][j] - self.q_verif[i][j])
            lines.append(
                f"  {self.joint_names[j]:20s} "
                f"{self.q_sdas[i][j]:12.6f} "
                f"{self.q_verif[i][j]:12.6f} "
                f"{diff:12.6f} "
                f"{np.degrees(diff):9.4f}°"
            )
        
        return "\n".join(lines)


def compare_at_sigma(
    sdas_model,
    verification_solver,
    sigma_values: np.ndarray,
    sigma_f: float = 0.0,
    joint_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> ComparisonResult:
    """
    Run SDAS vs verification solver at each sigma value.
    
    Parameters
    ----------
    sdas_model : SDASModel
        The SDAS model (reference implementation).
    verification_solver : object
        Any solver with solve_motors(theta_A, theta_B) returning q.
    sigma_values : ndarray
        Array of sigma values to test.
    sigma_f : float
        Differential synergy value. Parameter name kept for backward compatibility;
        internally treated as sigma_d.
    joint_names : list of str, optional
        Joint names for display.
    verbose : bool
        Print per-step output.
    
    Returns
    -------
    ComparisonResult
    """
    n_steps = len(sigma_values)
    n_joints = sdas_model.n_joints
    
    q_sdas = np.zeros((n_steps, n_joints))
    q_verif = np.zeros((n_steps, n_joints))
    
    for i, sigma in enumerate(sigma_values):
        theta_A = sigma + sigma_f
        theta_B = sigma - sigma_f
        
        # SDAS - use solve_motors (direct theta_A, theta_B input)
        q_s = sdas_model.solve_motors(theta_A, theta_B).flatten()
        q_sdas[i] = q_s
        
        # Verification solver
        q_v = verification_solver.solve_motors(theta_A, theta_B, verbose=False)
        q_verif[i] = q_v
        
        if verbose and (i % max(1, n_steps // 10) == 0):
            err = np.max(np.abs(q_s - q_v))
            print(f"  σ={sigma:.4f}: |q_sdas - q_verif|_∞ = {err:.6e}")
    
    if joint_names is None:
        joint_names = [f"Joint_{j}" for j in range(n_joints)]
    
    return ComparisonResult(
        sigma_values=sigma_values,
        q_sdas=q_sdas,
        q_verif=q_verif,
        joint_names=joint_names,
    )
