# src/simulation/transmission_force.py
"""
Tendon transmission force matrix Q(u) computation.

Implements the Q(q) matrix from Della Santina et al. (2018) Eq.44:
    Q(q) = R^T M^{-1} [0.5 M e_v | Sigma N e_v | -Lambda e_v]

Input u = [tau_M, s, s_dot]^T:
    Column 1 (0.5 M e_v):     Motor pulling force tau_M
    Column 2 (Sigma N e_v):    Sliding displacement s (static friction memory)
    Column 3 (-Lambda e_v):    Sliding velocity s_dot (viscous friction)

References
----------
Della Santina et al. (2018) TRO, Eq.13-14, Eq.38-44.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


def build_M_matrix(elements: List[int]) -> np.ndarray:
    """
    Build the M matrix (Eq.13).

    M in R^{(m+1) x (m+1)}:
        M[i,i] = -1 (i=0..m-1)
        M[i,i+1] = +1 (i=0..m-1)
        M[m,0] = +1
        M[m,m] = +1

    Physical meaning: tension balance along the tendon path:
        M * T + V(v) + e * tau_M = 0

    Parameters
    ----------
    elements : list[int]
        Element IDs along the tendon path.

    Returns
    -------
    M : ndarray, shape (m+1, m+1)
    """
    m = len(elements)
    M = np.zeros((m + 1, m + 1))
    for i in range(m):
        M[i, i] = -1.0
        M[i, i + 1] = 1.0
    M[m, 0] = 1.0
    M[m, m] = 1.0
    return M


def build_R_bar_matrix(elements: List[int],
                       design,
                       jid_to_idx: Dict[int, int]) -> np.ndarray:
    """
    Build the R_bar matrix (Eq.13 notation).

    R_bar in R^{(m+1) x n}, where R_bar[seg_idx, joint_idx] = r_k
    if segment k is on joint j, otherwise 0.

    Note: The paper's R (Eq.13) is R = -R_bar^T e_v at the merged level.
    Here R_bar retains the per-segment structure.

    Parameters
    ----------
    elements : list[int]
        Element IDs along the path.
    design : OrigamiHandDesign
    jid_to_idx : dict
        Mapping from fold_line_id to joint index.

    Returns
    -------
    R_bar : ndarray, shape (m+1, n_joints)
    """
    from src.models.origami_design import is_pulley_id, is_hole_id

    n_joints = len(jid_to_idx)
    m = len(elements)
    R_bar = np.zeros((m + 1, n_joints))

    for seg_idx, eid in enumerate(elements):
        j_idx = None
        r_val = 0.0

        if is_pulley_id(eid) and eid in design.pulleys:
            pulley = design.pulleys[eid]
            if pulley.attached_fold_line_id is not None:
                j_idx = jid_to_idx.get(pulley.attached_fold_line_id)
            r_val = pulley.radius
        elif is_hole_id(eid) and eid in design.holes:
            hole = design.holes[eid]
            if hole.attached_fold_line_id is not None:
                j_idx = jid_to_idx.get(hole.attached_fold_line_id)
            r_val = hole.plate_offset

        if j_idx is not None and r_val > 0:
            R_bar[seg_idx, j_idx] = r_val

    return R_bar


def build_viscous_damping_matrix(elements: List[int],
                                 design,
                                 tendon_viscous_damping: float = 0.001) -> np.ndarray:
    """
    Build the viscous friction diagonal matrix Lambda (Eq.38-39).

    Lambda in R^{(m+1) x (m+1)}:
        Lambda[i,i] = c_i / r_i^2   where c_i = r_i^2 * tendon_viscous_damping

    For hole elements, equivalent viscous friction is used.

    Parameters
    ----------
    elements : list[int]
        Element IDs along the path.
    design : OrigamiHandDesign
    tendon_viscous_damping : float
        Viscous damping coefficient.

    Returns
    -------
    Lambda : ndarray, shape (m+1, m+1)
    """
    from src.models.origami_design import is_pulley_id, is_hole_id

    m = len(elements)
    Lambda_mat = np.zeros((m + 1, m + 1))

    for seg_idx, eid in enumerate(elements):
        c_i = 0.0
        r_i = 1.0

        if is_pulley_id(eid) and eid in design.pulleys:
            pulley = design.pulleys[eid]
            r_i = pulley.radius
            c_i = r_i**2 * tendon_viscous_damping
        elif is_hole_id(eid) and eid in design.holes:
            hole = design.holes[eid]
            r_i = hole.plate_offset
            c_i = r_i * hole.friction_coefficient * 0.01 * tendon_viscous_damping

        if c_i > 0 and r_i > 0:
            Lambda_mat[seg_idx, seg_idx] = c_i / r_i**2

    return Lambda_mat


def build_static_friction_matrix(elements: List[int],
                                 design,
                                 kappa_ratio: float = 0.3) -> np.ndarray:
    """
    Build the static friction matrix Sigma (Eq.38-39).

    Sigma in R^{(m+1) x (m+1)}:
        Sigma[i,i] = kappa_i / r_i^2
    where kappa_i is the static friction stiffness.

    Parameters
    ----------
    elements : list[int]
        Element IDs along the path.
    design : OrigamiHandDesign
    kappa_ratio : float
        Ratio of static friction stiffness to joint stiffness.

    Returns
    -------
    Sigma : ndarray, shape (m+1, m+1)
    """
    from src.models.origami_design import is_pulley_id, is_hole_id

    m = len(elements)
    Sigma = np.zeros((m + 1, m + 1))

    for seg_idx, eid in enumerate(elements):
        kappa_i = 0.0
        r_i = 1.0

        if is_pulley_id(eid) and eid in design.pulleys:
            pulley = design.pulleys[eid]
            r_i = pulley.radius
            kappa_i = kappa_ratio * 1.0
        elif is_hole_id(eid) and eid in design.holes:
            hole = design.holes[eid]
            r_i = hole.plate_offset
            kappa_i = kappa_ratio * hole.friction_coefficient * 0.1

        if kappa_i > 0 and r_i > 0:
            Sigma[seg_idx, seg_idx] = kappa_i / r_i**2

    return Sigma


def build_N_matrix(elements: List[int], design) -> np.ndarray:
    """
    Build the angle conversion matrix N (Eq.41).

    N in R^{(m+1) x (m+1)}:
        N = diag(1/r_i)

    Converts tendon segment displacement to pulley rotation angle:
        theta = N (M^{-1} R_bar q - e_v s)

    Parameters
    ----------
    elements : list[int]
        Element IDs along the path.
    design : OrigamiHandDesign

    Returns
    -------
    N : ndarray, shape (m+1, m+1)
    """
    from src.models.origami_design import is_pulley_id, is_hole_id

    m = len(elements)
    N = np.zeros((m + 1, m + 1))

    for seg_idx, eid in enumerate(elements):
        r_i = 1.0
        if is_pulley_id(eid) and eid in design.pulleys:
            r_i = design.pulleys[eid].radius
        elif is_hole_id(eid) and eid in design.holes:
            r_i = design.holes[eid].plate_offset
        if r_i > 0:
            N[seg_idx, seg_idx] = 1.0 / r_i

    return N


def compute_Q_matrix(design) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the three columns of the Q matrix (Eq.44).

    For each tendon:
        Q = R^T M^{-1} [0.5 M e_v | Sigma N e_v | -Lambda e_v]

    Returns three column vectors of shape (n_joints,) each.

    Returns
    -------
    Q_tauM : (n_joints,)  — maps tau_M to joint torques
    Q_s : (n_joints,)     — maps s (sliding) to joint torques
    Q_sdot : (n_joints,)  — maps s_dot to joint torques
    """
    from src.models.transmission_builder import get_joint_list
    from src.models.origami_design import is_hole_id, is_pulley_id

    joints, jid_to_idx = get_joint_list(design)
    n_joints = len(joints)

    if len(design.tendons) == 0:
        return np.zeros(n_joints), np.zeros(n_joints), np.zeros(n_joints)

    Q_tauM_list = []
    Q_s_list = []
    Q_sdot_list = []

    for tendon in design.tendons.values():
        elements = [eid for eid in tendon.pulley_sequence
                    if eid >= 0 or (eid <= -100 and eid > -200)]
        if len(elements) == 0:
            continue

        M = build_M_matrix(elements)
        R_bar = build_R_bar_matrix(elements, design, jid_to_idx)
        Lambda = build_viscous_damping_matrix(elements, design)
        Sigma = build_static_friction_matrix(elements, design)
        N = build_N_matrix(elements, design)

        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        e_v = np.ones(len(elements) + 1)

        # R^T maps segment tensions T to joint torques.
        # Physical: tau_joint[j] = sum(r_k * T_k) over all segments k on joint j
        # Positive tension T_k produces positive (flexion) torque.
        # R_T = R_bar^T (shape n x (m+1))
        R_T = R_bar.T  # shape (n, m+1)

        # Three input directions
        Q_tauM = R_T @ M_inv @ (0.5 * M @ e_v)
        Q_s = R_T @ M_inv @ (Sigma @ N @ e_v)
        Q_sdot = R_T @ M_inv @ (-Lambda @ e_v)

        Q_tauM_list.append(Q_tauM)
        Q_s_list.append(Q_s)
        Q_sdot_list.append(Q_sdot)

    if len(Q_tauM_list) == 0:
        return np.zeros(n_joints), np.zeros(n_joints), np.zeros(n_joints)

    # SUM across tendons: each tendon contributes independently to joint torque.
    # AVERAGE would dilute multi-tendon designs (e.g., 5 tendons → 1/5 torque each).
    # Physical: multiple tendons pull simultaneously, torques add.
    return (np.sum(Q_tauM_list, axis=0),
            np.sum(Q_s_list, axis=0),
            np.sum(Q_sdot_list, axis=0))
