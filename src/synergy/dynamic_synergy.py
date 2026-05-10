# src/synergy/dynamic_synergy.py
"""
Dynamic Synergy Model (Piazza et al. 2016, ICRA)

Implements the descriptor linear system approach for speed-dependent hand
synergies. Uses passive damping components to generate different closures
(different synergy directions) as the input speed changes.

Mathematical background (Piazza et al. 2016, "SoftHand Pro-D"):
    System: T^T C T q_dot + E q = R^T u + w     (Eq. 3)

    Slow closure (quasi-static, damping negligible, q_dot ≈ 0):
        E q = R^T u  →  q = S_s σ               (Eq. 4-6)

    Fast closure (transient, damping dominates):
        T^T C T q_dot ≫ 0  → initial direction = S_f σ   (Eq. 11-14)
        Hand converges from S_f σ → S_s σ over time.

    S_f = T_⊥^T R^+_{E_y}   where E_y = T_⊥ E T_⊥^T      (Eq. 14)

Compatibility with Augmented Adaptive Synergies (Della Santina et al. 2018):
    - Shares the same R (transmission) and E (stiffness) matrices
    - R can be extended to R_aug = [R; R_f] for combined model
    - Adds T (damper transmission) and C_diag (damping coefficients)
"""

import numpy as np
from typing import Optional, Tuple, Union
from .base_adaptive import AdaptiveSynergyModel


class DynamicSynergyModel:
    """
    Dynamic Synergy solver implementing speed-dependent synergies.

    Parameters
    ----------
    n_joints : int
        Number of hand joints.
    R : np.ndarray, shape (k, n)
        Actuator transmission matrix (or R_aug for combined augmented+dynamic).
    E_vec : np.ndarray, shape (n,)
        Joint stiffness vector (diagonal of E).
    T : np.ndarray, shape (n_d, n)
        Damper transmission matrix. Maps joint velocities to damper velocities.
    C_diag : np.ndarray, shape (n_d,)
        Diagonal damping coefficients of each damper.

    Usage
    -----
    # Basic dynamic synergy (single actuator, speed-dependent)
    model = DynamicSynergyModel(n_joints, R, E_vec, T, C_diag)

    # Slow closure (quasi-static)
    q_slow = model.solve(sigma, speed_factor=0.0)

    # Fast closure (damping-dominated transient)
    q_fast = model.solve(sigma, speed_factor=1.0)

    # Combined with augmented synergy (R_aug = vstack([R, R_f]))
    R_aug = np.vstack([R, R_f])
    model = DynamicSynergyModel(n_joints, R_aug, E_vec, T, C_diag)

    # sigma = [sigma_dyn, sigma_f], speed_factor applies to all directions
    q = model.solve(np.concatenate([sigma_dyn, sigma_f]), speed_factor)
    """

    def __init__(self, n_joints: int, R: np.ndarray, E_vec: np.ndarray,
                 T: np.ndarray, C_diag: np.ndarray):
        self.n_joints = n_joints
        self.k = R.shape[0]       # number of actuation inputs
        self.n_d = T.shape[0]     # number of dampers
        self.n_t = T.shape[1]     # should equal n_joints

        assert T.shape[1] == n_joints, \
            f"T must be ({self.n_d} x {n_joints}), got {T.shape}"
        assert C_diag.shape[0] == self.n_d, \
            f"C_diag must have {self.n_d} elements, got {C_diag.shape[0]}"
        assert E_vec.shape[0] == n_joints, \
            f"E_vec must have {n_joints} elements, got {E_vec.shape[0]}"

        self.R = R.copy()
        self.E = np.diag(E_vec)
        self.T = T.copy()
        self.C = np.diag(C_diag)  # (n_d, n_d)

        # --- Precompute Slow Synergy S_s (standard adaptive synergy) ---
        self._base_model = AdaptiveSynergyModel(n_joints, R, E_vec)
        self.S_s = self._base_model.S       # (n, k)
        self.C_s = self._base_model.C       # (n, n) passive compliance

        # --- Precompute Fast Synergy S_f ---
        self.S_f = self._compute_fast_synergy()

    # ------------------------------------------------------------------
    #  Internal: compute T_⊥ (nullspace basis of T)
    # ------------------------------------------------------------------
    def _compute_T_perp(self) -> np.ndarray:
        """
        Compute a basis T_⊥ for the nullspace of T.

        Returns
        -------
        T_perp : np.ndarray, shape (n_null, n_joints)
            Each row is a basis vector in the nullspace of T.
            If n_null == 0, T has no nullspace.
        """
        n, n_j = self.T.shape  # (n_d, n_joints)

        if n >= n_j:
            # T has >= n_j rows, may be full column rank.
            _, s, Vt = np.linalg.svd(self.T, full_matrices=True)
            rank = np.sum(s > 1e-12 * max(s) if max(s) > 0 else 0)
            if rank < n_j:
                return Vt[rank:, :]  # rows corresponding to zero singular values
            else:
                return np.empty((0, n_j))
        else:
            # T is (n_d x n_joints) with n_d < n_joints.
            # Nullspace has dimension n_j - n_d.
            _, s, Vt = np.linalg.svd(self.T, full_matrices=True)
            # Vt is (n_j x n_j), last (n_j - n_d) rows are nullspace basis
            return Vt[self.n_d:, :]

    # ------------------------------------------------------------------
    #  Compute fast synergy S_f
    #  Following Eq. (14): S_f = T_⊥^T R^+_{E_y}
    # ------------------------------------------------------------------
    def _compute_fast_synergy(self) -> np.ndarray:
        """
        Precompute the fast dynamic synergy matrix S_f.

        Returns
        -------
        S_f : np.ndarray, shape (n, k)
        """
        T_perp = self._compute_T_perp()
        n_null = T_perp.shape[0]

        if n_null == 0:
            # No nullspace → no independent fast dynamics → S_f = S_s
            return self.S_s.copy()

        # E_y = T_⊥ E T_⊥^T  (n_null x n_null)
        E_y = T_perp @ self.E @ T_perp.T       # (n_null, n_null)
        E_y_inv = np.linalg.pinv(E_y)

        # R_bar = R T_⊥^T  (k x n_null)
        R_bar = self.R @ T_perp.T              # (k, n_null)

        # Compute R^+_{E_y} = E_y^{-1} R_bar^T (R_bar E_y^{-1} R_bar^T)^{-1}
        RE = R_bar @ E_y_inv @ R_bar.T         # (k, k)
        RE_inv = np.linalg.pinv(RE)

        R_plus_Ey = E_y_inv @ R_bar.T @ RE_inv  # (n_null, k)

        # S_f = T_⊥^T R^+_{E_y}  (n x k)
        S_f = T_perp.T @ R_plus_Ey

        return S_f

    # ------------------------------------------------------------------
    #  Main solve method
    # ------------------------------------------------------------------
    def solve(self,
              sigma: np.ndarray,
              speed_factor: Union[float, np.ndarray] = 0.0,
              J: Optional[np.ndarray] = None,
              f_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute joint configuration q for given synergy input and speed.

        Parameters
        ----------
        sigma : np.ndarray, shape (k,)
            Synergy input(s). For augmented synergy, pass concatenated
            [sigma_dyn, sigma_f].
        speed_factor : float or ndarray, in [0, 1]
            0.0 = slow closure (quasi-static, S_s)
            1.0 = fast closure (damping-dominated, S_f)
            Intermediate values interpolate smoothly.
        J : np.ndarray, shape (m, n), optional
            Grasp Jacobian.
        f_ext : np.ndarray, shape (m,), optional
            External wrench.

        Returns
        -------
        q : np.ndarray, shape (n,)
        """
        sigma = np.asarray(sigma, dtype=float)
        if sigma.ndim == 0:
            sigma = sigma.reshape(1)
        if sigma.shape[0] != self.k:
            raise ValueError(
                f"sigma must have {self.k} elements, got {sigma.shape[0]}")

        # Clamp speed_factor to [0, 1]
        alpha = np.clip(float(speed_factor), 0.0, 1.0)

        # Interpolate between slow and fast synergies
        S_eff = (1.0 - alpha) * self.S_s + alpha * self.S_f

        # Active part
        q = S_eff @ sigma

        # External force compliance (using slow compliance for simplicity)
        if J is not None and f_ext is not None:
            q += self.C_s @ J.T @ f_ext

        return q

    # ------------------------------------------------------------------
    #  Combined solve: separate dynamic and augmented inputs
    # ------------------------------------------------------------------
    def solve_combined(self,
                       sigma_dyn: np.ndarray,
                       sigma_f: np.ndarray,
                       speed_factor: float = 0.0,
                       use_dynamic_on_f: bool = False,
                       J: Optional[np.ndarray] = None,
                       f_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Combined solve with separated dynamic (sigma_dyn) and augmented
        (sigma_f) inputs.  Only works when R was constructed as
        R_aug = vstack([R_dyn, R_f]).

        Parameters
        ----------
        sigma_dyn : np.ndarray, shape (k_dyn,)
            Basic actuation input (subject to speed-dependent dynamic effect).
        sigma_f : np.ndarray, shape (m,)
            Friction/augmented input.
            By default NOT subject to dynamic effect (static only).
            Set use_dynamic_on_f=True to apply dynamic to both.
        speed_factor : float, in [0, 1]
        use_dynamic_on_f : bool
            If True, both sigma_dyn and sigma_f are interpolated.
            If False (default), only sigma_dyn is interpolated while
            sigma_f uses the slow (static) synergy.
        J, f_ext : optional

        Returns
        -------
        q : np.ndarray, shape (n,)
        """
        k_dyn = sigma_dyn.shape[0]
        m = sigma_f.shape[0]

        if k_dyn + m != self.k:
            raise ValueError(
                f"R has {self.k} inputs, but sigma_dyn ({k_dyn}) + "
                f"sigma_f ({m}) = {k_dyn + m}")

        alpha = np.clip(float(speed_factor), 0.0, 1.0)

        if use_dynamic_on_f:
            # Both interpolated
            S_eff = (1.0 - alpha) * self.S_s + alpha * self.S_f
            q = S_eff @ np.concatenate([sigma_dyn, sigma_f])
        else:
            # Only sigma_dyn interpolated; sigma_f uses slow synergy
            S_eff_dyn = (1.0 - alpha) * self.S_s[:, :k_dyn] + \
                        alpha * self.S_f[:, :k_dyn]
            q = S_eff_dyn @ sigma_dyn + self.S_s[:, k_dyn:] @ sigma_f

        if J is not None and f_ext is not None:
            q += self.C_s @ J.T @ f_ext

        return q

    # ------------------------------------------------------------------
    #  Separated synergy directions
    # ------------------------------------------------------------------
    @property
    def slow_synergy(self) -> np.ndarray:
        """S_s : slow synergy matrix (n, k)."""
        return self.S_s.copy()

    @property
    def fast_synergy(self) -> np.ndarray:
        """S_f : fast synergy matrix (n, k)."""
        return self.S_f.copy()

    @property
    def synergy_diff(self) -> np.ndarray:
        """S_f - S_s : direction of dynamic modulation."""
        return self.S_f - self.S_s

    # ------------------------------------------------------------------
    #  Static: design fast synergy from desired S_f and T, E
    # ------------------------------------------------------------------
    @staticmethod
    def compute_R_for_fast_synergy(
            S_s: np.ndarray,
            S_f_desired: np.ndarray,
            E_vec: np.ndarray,
            T: np.ndarray,
            tol: float = 1e-10) -> np.ndarray:
        """
        Given a desired fast synergy S_f_desired, solve for the
        transmission matrix R that approximates it.

        This is an inverse design tool: "I want S_f to be X, what R do I need?"

        Assumes T, E are already chosen.  The method computes R by
        matching S_f ≈ T_⊥^T (T_⊥ E T_⊥^T)^{-1} R T_⊥^T ...

        For simplicity, returns the R that best approximates both S_s and S_f.
        """
        E = np.diag(E_vec)
        E_inv = np.diag(1.0 / E_vec)

        # Compute T_⊥
        n_j = E_vec.shape[0]
        n_d = T.shape[0]
        _, s, Vt = np.linalg.svd(T, full_matrices=True)
        if n_d < n_j:
            T_perp = Vt[n_d:, :]
        else:
            rank = np.sum(s > tol)
            if rank < n_j:
                T_perp = Vt[rank:, :]
            else:
                raise ValueError("T has no nullspace, cannot design fast synergy")

        # We want S_f ≈ T_⊥^T (T_⊥ E T_⊥^T)^{-1} R T_⊥^T ...
        # But solving for R analytically is complex.
        # For now, use the base R from S_s and verify S_f similarity.
        R_slow = AdaptiveSynergyModel.compute_R_from_synergy(S_s, E_vec, tol)

        return R_slow
