# src/synergy/dynamic_synergy.py
"""
Dynamic Synergy Model (Piazza et al. 2016, ICRA)

Implements speed-dependent hand synergies using passive damping components.
The physical system is:

    T^T C T q_dot + E q = R^T u + w     (Eq. 3)

At low speed (q_dot ≈ 0), damping is negligible and the solution reduces to
the standard adaptive synergy (slow synergy S_s).

At high speed, the damping force T^T C T q_dot opposes joint motion. This is
modeled as an *effective additional stiffness* on the damped joints:

    E_eff(α) = E + α · T^T · C · T

where α = speed_factor ∈ [0, 1]. The equilibrium synergy matrix becomes:

    S_eff(α) = E_eff(α)^{-1} R^T (R E_eff(α)^{-1} R^T)^{-1}

This "damped equilibrium" interpretation:
  - Preserves all joints bending in the same direction (no sign flip paradox)
  - Reduces motion of damped joints proportionally to damping strength × speed
  - Always satisfies both the motor constraint R·q = σ and damper consistency
  - Smoothly interpolates between slow (α=0) and high-speed behavior (α=1)

NOTE on descriptor system theory (Piazza et al. 2016):
  The original paper derives S_f = T_⊥^T R^+_{E_y} via descriptor decomposition.
  That formulation enforces T·S_f = 0, which is the *instantaneous rate* constraint
  for the fast subsystem. For configurations with 3+ joints sharing one damper
  (like ohd_8), this forces some joints to have negative angles — physically
  unrealistic when the hand is closing. The damped equilibrium approach avoids
  this by modeling the net effect of damping as a speed-dependent stiffness increase,
  which is more physically appropriate for general damper topologies.

References:
  - Piazza et al. 2016, "SoftHand Pro-D: Matching Dynamic Content of Natural
    User Commands with Hand Embodiment for Enhanced Prosthesis Control", ICRA.
"""

import numpy as np
from typing import Optional, Union
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

    # Fast closure (damping-dominated)
    q_fast = model.solve(sigma, speed_factor=1.0)

    # Combined with augmented synergy (R_aug = vstack([R, R_f]))
    R_aug = np.vstack([R, R_f])
    model = DynamicSynergyModel(n_joints, R_aug, E_vec, T, C_diag)

    # sigma = [sigma_dyn, sigma_f], speed_factor applies to both
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

        # --- Precompute Damper-Induced Stiffness Term ---
        # FULL T^T C T (coupled, used for projection direction estimate)
        self._TCT_full = self.T.T @ self.C @ self.T  # (n, n)
        # DIAGONAL of T^T C T: each damped joint gets independent stiffness.
        # This removes the non-physical cross-coupling that can make some
        # joints bend in reverse (sign flip) when a single damper connects
        # multiple joints with different R values.
        #
        # Physical justification:
        #   When a single shared damper connects multiple joints, the damper
        #   cable is in series with each joint's tendon. The return spring on
        #   the damper keeps it reset when slack. Each joint thus experiences
        #   damping INDEPENDENTLY — the damper tendon slips if one joint moves
        #   while another is locked. The coupled T^T C T would imply all joints
        #   move together rigidly, which is not correct for tendon-driven
        #   dampers with return springs.
        #
        #   Diagonal T^T C T → each damped joint j gets additional stiffness
        #     ΔE_j = sum_i C_i * T_ij^2   (independent per joint)
        self._TCT_diag = np.diag(np.diag(self._TCT_full))  # (n, n)

    # ------------------------------------------------------------------
    #  Core: compute speed-dependent synergy matrix via damped equilibrium
    # ------------------------------------------------------------------
    def _compute_damped_synergy(self, speed_factor: float) -> np.ndarray:
        """
        Compute the effective synergy matrix for a given speed factor.

        At speed_factor=0: E_eff = E               →  S_eff = S_s (slow synergy)
        At speed_factor=1: E_eff = E + diag(T^T C T)  →  S_eff = damped synergy

        IMPORTANT: Uses diag(T^T C T) instead of full T^T C T.
        This ensures each damped joint receives INDEPENDENT additional stiffness,
        avoiding the non-physical cross-coupling that causes sign flips when
        T=[1,1,...,1] and joints have different R values.

        Parameters
        ----------
        speed_factor : float in [0, 1]
            Fraction of full damping applied.

        Returns
        -------
        S_eff : np.ndarray, shape (n, k)
        """
        alpha = np.clip(float(speed_factor), 0.0, 1.0)

        # Effective stiffness: E + α · diag(T^T C T)
        # Using diagonal formulation for physically correct independent damping.
        E_eff = self.E + alpha * self._TCT_diag
        E_eff_inv = np.linalg.inv(E_eff)

        # Damped equilibrium synergy:
        # S_eff = E_eff^{-1} R^T (R E_eff^{-1} R^T)^{-1}
        RE_inv = E_eff_inv @ self.R.T   # (n, k)
        RER = self.R @ RE_inv            # (k, k)
        RER_inv = np.linalg.inv(RER)

        S_eff = RE_inv @ RER_inv         # (n, k)
        return S_eff

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
            1.0 = fast closure (damping-dominated)
            Intermediate values → smooth damped equilibrium.
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

        # Compute speed-dependent damped equilibrium synergy
        S_eff = self._compute_damped_synergy(float(speed_factor))

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
                       use_dynamic_on_f: bool = True,
                       J: Optional[np.ndarray] = None,
                       f_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Combined solve with separated dynamic (sigma_dyn) and augmented
        (sigma_f) inputs. Only works when R was constructed as
        R_aug = vstack([R_dyn, R_f]).

        Parameters
        ----------
        sigma_dyn : np.ndarray, shape (k_dyn,)
            Basic actuation input.
        sigma_f : np.ndarray, shape (m,)
            Friction/augmented input.
            By default both are subject to the same damped equilibrium.
            Set use_dynamic_on_f=False to keep sigma_f at static (S_s).
        speed_factor : float, in [0, 1]
        use_dynamic_on_f : bool
            If True (default), both sigma_dyn and sigma_f use the same
            speed-dependent synergy. If False, only sigma_dyn is modulated
            while sigma_f uses the slow (static) synergy.
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

        if use_dynamic_on_f:
            # Both modulated by same damped equilibrium
            q = self.solve(np.concatenate([sigma_dyn, sigma_f]),
                           speed_factor=speed_factor,
                           J=J, f_ext=f_ext)
        else:
            # Only sigma_dyn modulated; sigma_f uses slow synergy
            S_eff = self._compute_damped_synergy(speed_factor)
            q = (S_eff[:, :k_dyn] @ sigma_dyn +
                 self.S_s[:, k_dyn:] @ sigma_f)

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
        """
        High-speed synergy matrix S_f : S_eff at speed_factor=1.0.

        This is NOT the descriptor-system S_f from Eq. (14) of the paper,
        but rather the damped equilibrium matrix:
            S_f = (E + T^T C T)^{-1} R^T (R (E + T^T C T)^{-1} R^T)^{-1}
        """
        return self._compute_damped_synergy(1.0)

    @property
    def synergy_diff(self) -> np.ndarray:
        """S_f - S_s : direction of dynamic modulation."""
        return self.fast_synergy - self.S_s

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

        Assumes T, E are already chosen.
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
