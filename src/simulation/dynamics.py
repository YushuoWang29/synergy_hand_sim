# src/simulation/dynamics.py
"""
Dynamics assembler for the full robot dynamics (Eq.43).

Implements the equations of motion:
    M(q) q_ddot + C(q, q_dot) q_dot + g(q) + tau_f(q, q_dot) = Q(q, u)

where:
    M(q): Inertia matrix (diagonal approximation for origami links)
    C: Coriolis/centrifugal (zero for planar, single-DOF-per-joint)
    g: Gravity torque
    tau_f: Joint friction (viscous + Coulomb)
    Q(q, u): Transmission force (Eq.44)

References
----------
Della Santina et al. (2018) TRO, Eq.43-44.
"""

import numpy as np
from typing import Callable, Optional, Tuple


class DynamicsAssembler:
    """
    Assembles robot dynamics: M(q) q_ddot = Q(q, u) + tau_g(q) - B q_dot - tau_Coulomb(q_dot) - g(q)

    Parameters
    ----------
    n_joints : int
        Number of joints.
    inertia_matrix : ndarray, optional
        Joint-space inertia matrix (n, n). Default identity.
    joint_damping : ndarray, optional
        Viscous damping coefficients (n,). Default zeros.
    joint_coulomb : ndarray, optional
        Coulomb friction magnitudes (n,). Default zeros.
    gravity_vector : ndarray, optional
        Gravity vector (3,). Default [0, 0, -9.81].
    """

    def __init__(
        self,
        n_joints: int,
        inertia_matrix: Optional[np.ndarray] = None,
        joint_damping: Optional[np.ndarray] = None,
        joint_coulomb: Optional[np.ndarray] = None,
        joint_stiffness: Optional[np.ndarray] = None,
        gravity_vector: Optional[np.ndarray] = None,
    ):
        self.n_joints = n_joints

        # Default: unit inertia
        if inertia_matrix is not None:
            self.inertia_matrix = inertia_matrix
        else:
            self.inertia_matrix = np.eye(n_joints) * 0.001  # 1e-3 kg*m^2

        self.inertia_inv = np.linalg.inv(self.inertia_matrix)

        # Viscous damping
        self.joint_damping = (
            np.zeros(n_joints) if joint_damping is None else np.asarray(joint_damping)
        )

        # Coulomb friction
        self.joint_coulomb = (
            np.zeros(n_joints) if joint_coulomb is None else np.asarray(joint_coulomb)
        )

        # Joint elastic stiffness (Nm/rad)
        if joint_stiffness is not None:
            self.joint_stiffness = np.asarray(joint_stiffness)
        else:
            self.joint_stiffness = np.ones(n_joints) * 10.0

        # Gravity
        if gravity_vector is not None:
            self.gravity = np.asarray(gravity_vector)
        else:
            self.gravity = np.array([0.0, 0.0, -9.81])

    def compute_elastic_torque(self, q: np.ndarray) -> np.ndarray:
        """
        Compute joint elastic restoring torque (linear spring).

        tau_elastic = K * q
        
        Parameters
        ----------
        q : ndarray, shape (n,)
            Joint positions.

        Returns
        -------
        tau_elastic : ndarray, shape (n,)
            Elastic restoring torque.
        """
        return self.joint_stiffness * q

    def compute_friction_torque(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """
        Compute joint friction torque (viscous + Coulomb).

        tau_f = B * q_dot + tau_coulomb * tanh(q_dot / eps)

        Parameters
        ----------
        q : ndarray, shape (n,)
            Joint positions.
        q_dot : ndarray, shape (n,)
            Joint velocities.

        Returns
        -------
        tau_f : ndarray, shape (n,)
            Friction torque.
        """
        eps = 1e-4
        viscous = self.joint_damping * q_dot
        coulomb = self.joint_coulomb * np.tanh(q_dot / eps)
        return viscous + coulomb

    def compute_inverse_inertia(self, q: np.ndarray) -> np.ndarray:
        """
        Compute M(q)^{-1}.

        Parameters
        ----------
        q : ndarray, shape (n,)
            Joint positions.

        Returns
        -------
        M_inv : ndarray, shape (n, n)
        """
        return self.inertia_inv

    def compute_coriolis(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis/centrifugal torque C(q, q_dot) q_dot.

        For planar origami hand with 1-DOF-per-link joints,
        Coriolis effects are negligible. Returns zero.

        Parameters
        ----------
        q : ndarray, shape (n,)
        q_dot : ndarray, shape (n,)

        Returns
        -------
        C_qdot : ndarray, shape (n,)
        """
        return np.zeros(self.n_joints)

    def compute_acceleration(
        self,
        q: np.ndarray,
        q_dot: np.ndarray,
        u: np.ndarray,
        Q_mat: np.ndarray,
        tau_g: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute q_ddot from the equations of motion (Eq.43).

        M(q) q_ddot = Q(q) * u - C(q, q_dot) q_dot - B q_dot - tau_coulomb(q_dot) - tau_g(q) - K q

        Parameters
        ----------
        q : ndarray, shape (n,)
            Joint positions.
        q_dot : ndarray, shape (n,)
            Joint velocities.
        u : ndarray, shape (3,)
            Control input [tau_M, s, s_dot].
        Q_mat : ndarray, shape (n, 3)
            Transmission matrix.
        tau_g : ndarray, shape (n,), optional
            Gravity torque. Zero if None.

        Returns
        -------
        q_ddot : ndarray, shape (n,)
            Joint accelerations.
        """
        # Transmission force
        tau_t = Q_mat @ u

        # Friction
        tau_f = self.compute_friction_torque(q, q_dot)

        # Coriolis (zero in simplified case)
        tau_cor = self.compute_coriolis(q, q_dot)

        # Gravity
        if tau_g is None:
            tau_g = np.zeros(self.n_joints)

        # Elastic joint torque (K * q)
        tau_elastic = self.compute_elastic_torque(q)

        # Net torque: transmission - coriolis - friction - gravity - elastic
        tau_net = tau_t - tau_cor - tau_f - tau_g - tau_elastic

        # Acceleration
        M_inv = self.compute_inverse_inertia(q)
        q_ddot = M_inv @ tau_net

        return q_ddot
