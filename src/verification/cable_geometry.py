"""
Independent cable geometry extraction — no Della Santina dependencies.

Extracts pure geometric transmission vectors from OrigamiHandDesign:
    r_A[j] = Σ r_k · exp(-β · d_Ak)    Classic Capstan from Motor A side
    r_B[j] = Σ r_k · exp(-β · d_Bk)    Classic Capstan from Motor B side

Restriction: Uses classic Euler-Capstan formula T_out = T_in · exp(-μθ)
directly, NOT the Della Santina M-matrix formulation.

Imports only:
    - origami_design (data structures, no math models)
    - numpy (linear algebra)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from src.models.origami_design import (
    OrigamiHandDesign,
    is_pulley_id,
    is_hole_id,
)


DEFAULT_BETA = 0.09


def get_joint_indices(design: OrigamiHandDesign) -> Tuple[List, Dict[int, int]]:
    """
    Extract joint list and mapping fold_line_id -> joint index.
    Pure geometric query — no transmission math.
    """
    if not design.face_parent:
        design.build_topology()

    joints = design.joints
    jid_to_idx = {}
    for i, j in enumerate(joints):
        jid_to_idx[j.fold_line_id] = i
    return joints, jid_to_idx


def _get_element_radius(eid: int, design: OrigamiHandDesign) -> float:
    if is_pulley_id(eid) and eid in design.pulleys:
        return design.pulleys[eid].radius
    elif is_hole_id(eid) and eid in design.holes:
        return design.holes[eid].plate_offset
    return 0.0


def _get_element_joint_idx(eid: int, design: OrigamiHandDesign,
                            jid_to_idx: Dict[int, int]) -> Optional[int]:
    if is_pulley_id(eid) and eid in design.pulleys:
        p = design.pulleys[eid]
        if p.attached_fold_line_id is not None:
            return jid_to_idx.get(p.attached_fold_line_id)
    elif is_hole_id(eid) and eid in design.holes:
        h = design.holes[eid]
        if h.attached_fold_line_id is not None:
            return jid_to_idx.get(h.attached_fold_line_id)
    return None


def compute_R_vectors(
    design: OrigamiHandDesign,
    beta: float = DEFAULT_BETA
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute transmission vectors using classic Capstan formula.

    Returns
    -------
    r_geo : (n_joints,) uniform (no Capstan) transmission
    r_A   : (n_joints,) Capstan from Motor A
    r_B   : (n_joints,) Capstan from Motor B
    """
    joints, jid_to_idx = get_joint_indices(design)
    n = len(joints)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    r_geo_sum = np.zeros(n)
    r_A_sum = np.zeros(n)
    r_B_sum = np.zeros(n)

    if len(design.tendons) == 0:
        return r_geo_sum, r_A_sum, r_B_sum

    n_tendons = 0
    for tendon in design.tendons.values():
        elements = [eid for eid in tendon.pulley_sequence
                    if eid >= 0 or is_hole_id(eid)]
        N = len(elements)
        if N == 0:
            continue
        n_tendons += 1
        for k, eid in enumerate(elements):
            r = _get_element_radius(eid, design)
            if r <= 0:
                continue
            j_idx = _get_element_joint_idx(eid, design, jid_to_idx)
            if j_idx is None:
                continue
            r_geo_sum[j_idx] += r
            r_A_sum[j_idx] += r * np.exp(-beta * float(k))
            r_B_sum[j_idx] += r * np.exp(-beta * float(N - 1 - k))

    if n_tendons > 1:
        r_geo_sum /= n_tendons
        r_A_sum /= n_tendons
        r_B_sum /= n_tendons

    return r_geo_sum, r_A_sum, r_B_sum
