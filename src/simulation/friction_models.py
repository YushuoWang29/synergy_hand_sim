# src/simulation/friction_models.py
"""
Friction models for tendon-driven hands.

Implementations of:
1. Viscous friction — linear velocity dependence (Eq.39)
2. Hayward-Armstrong static friction — virtual angle memory state (Eq.37-38)
3. Capstan tension distribution — exponential tension decay along pulley path

These models correspond to the Lambda, Sigma, and z-state dynamics in
Della Santina et al. (2018), Eq.37-42.

References
----------
Della Santina et al. "Toward Dexterous Manipulation With Augmented Adaptive
Synergies" IEEE TRO 2018.
"""

import numpy as np
from typing import Optional, Tuple


class CapstanTensionDistribution:
    """
    Capstan friction model for tension distribution along a tendon path.

    Given tension T_in at one end, the tension after wrapping around
    a pulley with friction coefficient mu and wrap angle theta is:
        T_out = T_in * exp(-mu * theta)

    For a path with N elements, the effective decay from the start is:
        T_k = T_0 * exp(-beta * k)
    where beta = mu * theta_eff is the effective decay per element.

    This class precomputes tension distributions for both motor A and B,
    which is used to compute R and R_f matrices with the Capstan correction.

    Parameters
    ----------
    n_elements : int
        Number of elements in the tendon path.
    beta : float
        Effective Capstan decay per element. Default 0.09.
    """

    def __init__(self, n_elements: int, beta: float = 0.09):
        self.n = n_elements
        self.beta = beta

        # Precompute tension distributions
        k = np.arange(n_elements, dtype=float)
        self.T_A = np.exp(-beta * k)          # Tension from Motor A
        self.T_B = np.exp(-beta * (n_elements - 1 - k))  # Tension from Motor B

    def get_sigma_weights(self) -> np.ndarray:
        r"""
        Compute sigma-mode weights :math:`w_k = T_A[k] + T_B[k]`.

        Both motors pull together -> symmetric sum.
        """
        return self.T_A + self.T_B

    def get_sigma_f_weights(self) -> np.ndarray:
        r"""
        Compute sigma_f-mode weights :math:`w_k = T_A[k] - T_B[k]`.

        Motors pull differentially -> signed gradient.
        """
        return self.T_A - self.T_B

    def get_dead_zone_mask(self, threshold: float = 0.05) -> np.ndarray:
        """
        Compute dead zone mask where tension from both sides is below threshold.

        Elements in the dead zone are "frozen" and do not contribute to sigma_f.
        """
        return (self.T_A < threshold) & (self.T_B < threshold)

    def get_tension_at(self, k: int, T_0: float = 1.0) -> float:
        """
        Get tension at element k given initial tension T_0 at motor A.
        """
        return T_0 * np.exp(-self.beta * k)

    @staticmethod
    def from_elements(elements: list, beta: float = 0.09) -> 'CapstanTensionDistribution':
        """Create from list of element IDs."""
        return CapstanTensionDistribution(len(elements), beta)


class HaywardArmstrongFriction:
    """
    Hayward-Armstrong static friction model (Eq.37-38).

    Maintains a virtual angle z_j for each pulley. When the real angle
    theta_j exceeds the interval [z_j - Delta_max_j, z_j + Delta_max_j],
    z_j updates to track theta_j.

    Friction torque:
        tau_friction_j = (theta_j - z_j) * kappa_j / r_j^2

    This implements the "elastic stiction" model where the friction element
    behaves like a spring-damper with a limited elastic range.

    Attributes
    ----------
    n_pulleys : int
        Number of pulleys (excluding boundary row).
    z : np.ndarray, shape (n_pulleys,)
        Current virtual angle state.
    delta_max : np.ndarray, shape (n_pulleys,)
        Static friction range for each pulley.
    kappa : np.ndarray, shape (n_pulleys,)
        Static friction stiffness for each pulley.
    theta_prev : np.ndarray, shape (n_pulleys,)
        Previous real angle (for zero-crossing detection).
    """

    def __init__(self, n_pulleys: int,
                 delta_max: Optional[np.ndarray] = None,
                 kappa: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        n_pulleys : int
            Number of pulley elements.
        delta_max : np.ndarray, optional
            Static friction range per pulley. Default 1e-4 * ones.
        kappa : np.ndarray, optional
            Static friction stiffness per pulley. Default 0.3 * ones.
        """
        self.n_pulleys = n_pulleys
        self.delta_max = (delta_max if delta_max is not None
                          else np.ones(n_pulleys) * 1e-4)
        self.kappa = (kappa if kappa is not None
                      else np.ones(n_pulleys) * 0.3)
        self.z = np.zeros(n_pulleys)
        self.theta_prev = np.zeros(n_pulleys)

    def update(self, theta: np.ndarray, dt: float) -> np.ndarray:
        """
        Update virtual angle state z (Eq.37).

        Parameters
        ----------
        theta : np.ndarray, shape (n_pulleys,)
            Current real pulley angles.
        dt : float
            Time step (unused in update rule, kept for API consistency).

        Returns
        -------
        z_new : np.ndarray, shape (n_pulleys,)
            Updated virtual angles.
        """
        z_new = self.z.copy()

        # Eq.37: update rule for virtual angles
        below = theta <= self.z - self.delta_max
        above = theta >= self.z + self.delta_max

        z_new[below] = theta[below] + self.delta_max[below]
        z_new[above] = theta[above] - self.delta_max[above]
        # z_new[sticking] = self.z[sticking]  (unchanged)

        self.z = z_new
        self.theta_prev = theta.copy()
        return z_new

    def get_friction_torque(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute static friction torque (Eq.38).

        tau_friction_j = (theta_j - z_j) * kappa_j

        Note: Does not divide by r_j^2 here because the caller
        (e.g., build_static_friction_matrix) handles radius conversion
        via the Sigma matrix.

        Parameters
        ----------
        theta : np.ndarray, shape (n_pulleys,)
            Current real pulley angles.

        Returns
        -------
        tau : np.ndarray, shape (n_pulleys,)
            Static friction torque.
        """
        return (theta - self.z) * self.kappa

    def get_energy(self, theta: np.ndarray) -> float:
        """
        Compute stored static friction potential energy.

        E = 0.5 * sum_j kappa_j * (theta_j - z_j)^2

        Parameters
        ----------
        theta : np.ndarray, shape (n_pulleys,)
            Current real pulley angles.

        Returns
        -------
        energy : float
            Total stored energy.
        """
        return 0.5 * float(np.sum(self.kappa * (theta - self.z)**2))

    def reset(self):
        """Reset all virtual angles to zero."""
        self.z = np.zeros(self.n_pulleys)
        self.theta_prev = np.zeros(self.n_pulleys)

    def is_sticking(self, theta: np.ndarray) -> np.ndarray:
        """
        Detect which elements are in the sticking regime.

        Returns
        -------
        sticking : np.ndarray, shape (n_pulleys,), bool
            True for elements in the sticking regime.
        """
        return (theta >= self.z - self.delta_max) & (theta <= self.z + self.delta_max)

    def get_state(self) -> dict:
        """
        Return full state dict for serialization.
        """
        return {
            'z': self.z.copy(),
            'delta_max': self.delta_max.copy(),
            'kappa': self.kappa.copy(),
            'theta_prev': self.theta_prev.copy(),
        }

    def set_state(self, state: dict):
        """Restore state from dict."""
        self.z = state['z'].copy()
        self.delta_max = state['delta_max'].copy()
        self.kappa = state['kappa'].copy()
        self.theta_prev = state['theta_prev'].copy()
