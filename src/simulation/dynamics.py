# src/simulation/dynamics.py
"""
Dynamics assembler for the full robot dynamics (Eq.43).

Implements the equations of motion:
    M(q) q_ddot + C(q, q_dot) q_dot + g(q) + tau_f(q, q_dot) = Q(q, u)

where:
    M(q): Inertia matrix
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


def compute_joint_space_inertia(design, rbs) -> np.ndarray:
    """
    Compute a diagonal joint-space inertia matrix from RigidBodySystem.

    For each joint, aggregate the reflected inertia of all downstream faces:
        I_j = Σ m_i · d_i² + I_cm_face
    where d_i is the perpendicular distance from face i's centroid to the fold line.

    This captures the dominant inertial coupling along the kinematic tree.

    Parameters
    ----------
    design : OrigamiHandDesign
        The origami hand design (needs face_parent, faces, fold_lines).
    rbs : RigidBodySystem
        Pre-computed rigid body parameters (link_inertias).

    Returns
    -------
    M : ndarray, shape (n_joints, n_joints)
        Diagonal inertia matrix.
    """
    from src.models.transmission_builder import get_joint_list

    joints, jid_to_idx = get_joint_list(design)
    n = len(joints)
    M_diag = np.zeros(n)

    if n == 0:
        return np.eye(1) * 0.001

    # Build parent→children map
    face_children = {fid: [] for fid in design.faces}
    for child, parent in design.face_parent.items():
        if parent in face_children:
            face_children[parent].append(child)
        else:
            face_children[parent] = [child]

    def get_all_descendants(face_id):
        result = []
        for child in face_children.get(face_id, []):
            result.append(child)
            result.extend(get_all_descendants(child))
        return result

    for joint in joints:
        j_idx = jid_to_idx.get(joint.fold_line_id)
        if j_idx is None:
            continue

        fold = design.fold_lines.get(joint.fold_line_id)
        if fold is None:
            M_diag[j_idx] = 0.001
            continue

        # Determine child face (the one that moves relative to parent)
        child_face_id = joint.face_b_id  # default
        if design.face_parent.get(joint.face_b_id) == joint.face_a_id:
            child_face_id = joint.face_b_id
        elif design.face_parent.get(joint.face_a_id) == joint.face_b_id:
            child_face_id = joint.face_a_id
        elif not design.face_parent:
            # No kinematic tree - use both faces
            child_face_id = joint.face_b_id

        # All downstream faces
        downstream = [child_face_id] + get_all_descendants(child_face_id)

        # Fold line geometry (convert mm → m)
        fl_start = np.array([fold.start.x, fold.start.y]) * 1e-3
        fl_end = np.array([fold.end.x, fold.end.y]) * 1e-3
        fl_dir = fl_end - fl_start
        fl_len = np.linalg.norm(fl_dir)
        if fl_len < 1e-10:
            fl_len = 0.01
        fl_dir = fl_dir / fl_len

        total_inertia = 0.0
        for fid in downstream:
            if fid not in rbs.link_inertias:
                continue
            li = rbs.link_inertias[fid]

            # Face centroid in 2D (mm → m)
            face = design.faces.get(fid)
            if face is None:
                continue
            centroid = np.array([face.centroid.x, face.centroid.y]) * 1e-3

            # Perpendicular distance from centroid to fold line
            vec = centroid - fl_start
            perp_dist = abs(vec[0] * fl_dir[1] - vec[1] * fl_dir[0])

            # Steiner term: m * d²
            total_inertia += li.mass * perp_dist**2

        # Add the child face's own rotational inertia about the fold axis
        if child_face_id in rbs.link_inertias:
            li_child = rbs.link_inertias[child_face_id]
            # For a thin plate in the XY plane, I_xx and I_yy are in-plane
            # Use the smaller in-plane inertia as a conservative estimate
            total_inertia += min(li_child.inertia[0, 0], li_child.inertia[1, 1])

        M_diag[j_idx] = max(total_inertia, 1e-8)  # ensure positive

    return np.diag(M_diag)


def compute_joint_stiffness(design) -> np.ndarray:
    """
    Read joint stiffness from design fold_lines.

    Each joint corresponds to a fold line with known stiffness.

    Parameters
    ----------
    design : OrigamiHandDesign

    Returns
    -------
    K : ndarray, shape (n_joints,)
        Joint stiffness values (Nm/rad).
    """
    from src.models.transmission_builder import get_joint_list

    joints, jid_to_idx = get_joint_list(design)
    n = len(joints)
    K = np.ones(n) * 10.0  # fallback

    for joint in joints:
        j_idx = jid_to_idx.get(joint.fold_line_id)
        if j_idx is None:
            continue
        fold = design.fold_lines.get(joint.fold_line_id)
        if fold is not None:
            K[j_idx] = fold.stiffness

    return K


class DynamicsAssembler:
    """
    Assembles robot dynamics: M(q) q_ddot = Q(q, u) + tau_g(q) - B q_dot - tau_Coulomb(q_dot) - tau_elastic(q)

    Parameters
    ----------
    n_joints : int
        Number of joints.
    inertia_matrix : ndarray, optional
        Joint-space inertia matrix (n, n). If None, computed from design.
    joint_damping : ndarray, optional
        Viscous damping coefficients (n,). Default zeros.
    joint_coulomb : ndarray, optional
        Coulomb friction magnitudes (n,). Default zeros.
    joint_stiffness : ndarray, optional
        Joint stiffness (n,). If None, read from design.
    gravity_vector : ndarray, optional
        Gravity vector (3,). Default [0, 0, -9.81].
    design : OrigamiHandDesign, optional
        Design reference for computing q-dependent quantities.
    rigid_body_system : RigidBodySystem, optional
        Rigid body parameters for inertia computation.
    friction_model : HaywardArmstrongFriction, optional
        Static friction model (pulley-level).
    """

    def __init__(
        self,
        n_joints: int,
        inertia_matrix: Optional[np.ndarray] = None,
        joint_damping: Optional[np.ndarray] = None,
        joint_coulomb: Optional[np.ndarray] = None,
        joint_stiffness: Optional[np.ndarray] = None,
        gravity_vector: Optional[np.ndarray] = None,
        design=None,
        rigid_body_system=None,
        friction_model=None,
    ):
        self.n_joints = n_joints
        self.design = design
        self.rbs = rigid_body_system
        self.friction_model = friction_model

        # ---- Inertia matrix ----
        if inertia_matrix is not None:
            self.inertia_matrix = inertia_matrix
        elif rigid_body_system is not None and design is not None:
            # Compute from RigidBodySystem
            self.inertia_matrix = compute_joint_space_inertia(design, rigid_body_system)
            print(f"  [Dynamics] Inertia from RigidBodySystem:\n"
                  f"    diag = {np.diag(self.inertia_matrix)}")
        else:
            # Fallback: unit inertia scaled by 1e-3
            self.inertia_matrix = np.eye(n_joints) * 0.001
            print(f"  [Dynamics] Using default inertia = 0.001 * I")

        try:
            self.inertia_inv = np.linalg.inv(self.inertia_matrix)
        except np.linalg.LinAlgError:
            self.inertia_inv = np.linalg.pinv(self.inertia_matrix)

        # ---- Viscous damping ----
        self.joint_damping = (
            np.zeros(n_joints) if joint_damping is None else np.asarray(joint_damping)
        )

        # ---- Coulomb friction ----
        self.joint_coulomb = (
            np.zeros(n_joints) if joint_coulomb is None else np.asarray(joint_coulomb)
        )

        # ---- Joint elastic stiffness ----
        if joint_stiffness is not None:
            self.joint_stiffness = np.asarray(joint_stiffness)
        elif design is not None:
            self.joint_stiffness = compute_joint_stiffness(design)
            print(f"  [Dynamics] Stiffness from design:\n"
                  f"    K = {self.joint_stiffness}")
        else:
            self.joint_stiffness = np.ones(n_joints) * 10.0

        # ---- Gravity (default: off, set to zero) ----
        if gravity_vector is not None:
            self.gravity = np.asarray(gravity_vector)
        else:
            self.gravity = np.array([0.0, 0.0, 0.0])  # gravity off by default

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
        Compute M(q)^{-1}. Currently q-independent.

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
