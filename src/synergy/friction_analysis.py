#!/usr/bin/env python3
"""
Friction model analysis and computation for augmented adaptive synergy.

Provides three physically-meaningful models for computing R and R_f:

Model 1: Coulomb (paper Section III)
    R[Σr] — uniform geometric, correct for steady-state synergy
    R_f[M^{-1}V_maxe_v] — Coulomb friction coupling

Model 2: Capstan (physically corrected)
    R_capstan[j] = Σ r_k · (T_A[k] + T_B[k])    — U-shaped, captures dynamic friction
    Rf_capstan[j] = Σ r_k · (T_A[k] - T_B[k])   — signed gradient with dead zone
    where T_A[k] = exp(-β·k), T_B[k] = exp(-β·(N-1-k))

Model 3: Normalized combination
    R from Model 1 + R_f normalized to R's scale + dead zone
    This is the recommended practical model.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.models.origami_design import (OrigamiHandDesign, is_pulley_id, is_hole_id)
from src.models.transmission_builder import get_joint_list


def get_element_radius(eid: int, design: OrigamiHandDesign) -> float:
    """Get the effective radius of any element type."""
    if is_pulley_id(eid) and eid in design.pulleys:
        return design.pulleys[eid].radius
    elif is_hole_id(eid) and eid in design.holes:
        return design.holes[eid].plate_offset
    return 0.0


def get_element_joint_idx(eid: int, design: OrigamiHandDesign,
                           jid_to_idx: Dict[int, int]) -> Optional[int]:
    """Get the joint index for an element."""
    if is_pulley_id(eid) and eid in design.pulleys:
        p = design.pulleys[eid]
        if p.attached_fold_line_id is not None:
            return jid_to_idx.get(p.attached_fold_line_id)
    elif is_hole_id(eid) and eid in design.holes:
        h = design.holes[eid]
        if h.attached_fold_line_id is not None:
            return jid_to_idx.get(h.attached_fold_line_id)
    return None


def compute_capstan_R_Rf(
    design: OrigamiHandDesign,
    beta: float = 0.09,
    deadzone_frac: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute R and R_f using the Capstan tension decay model.

    Physical model:
        T_from_A[k] = exp(-β · k)                   (tension from Motor A)
        T_from_B[k] = exp(-β · (N-1-k))             (tension from Motor B)
        
        R_capstan[j] = Σ r_k · (T_A[k] + T_B[k])    σ mode: both pull together
        Rf_capstan[j] = Σ r_k · (T_A[k] - T_B[k])   σ_f mode: differential

    β: effective friction per element.
      β=0.09 → after 42 elements, T ≈ exp(-3.78) ≈ 0.023 (reasonable attenuation)
      β=0.3  → after 10 elements, T ≈ exp(-3.0) ≈ 0.05 (strong attenuation)

    Returns:
        R     : Capstan-weighted R (num_tendons × n_joints)
        Rf    : Capstan-weighted R_f (num_tendons × n_joints)
        info  : dict with additional analysis
    """
    joints, jid_to_idx = get_joint_list(design)
    n = len(joints)
    n_t = len(design.tendons)

    R_rows = []
    Rf_rows = []
    info = {'beta': beta, 'deadzone_frac': deadzone_frac}

    for tendon in design.tendons.values():
        elements = [eid for eid in tendon.pulley_sequence
                    if eid >= 0 or is_hole_id(eid)]
        N = len(elements)
        if N == 0:
            R_rows.append(np.zeros(n))
            Rf_rows.append(np.zeros(n))
            continue

        # Tension from each motor, normalized so max weight = 1
        T_A = np.exp(-beta * np.arange(N))
        T_B = np.exp(-beta * (N - 1 - np.arange(N)))

        R_row = np.zeros(n)
        Rf_row = np.zeros(n)

        for k, eid in enumerate(elements):
            r = get_element_radius(eid, design)
            if r <= 0:
                continue
            j_idx = get_element_joint_idx(eid, design, jid_to_idx)
            if j_idx is None:
                continue

            # σ mode: sum of tensions = both motors pulling together
            R_row[j_idx] += r * (T_A[k] + T_B[k])
            # σ_f mode: difference = motor A pulls while motor B opposes
            Rf_row[j_idx] += r * (T_A[k] - T_B[k])

        R_rows.append(R_row)
        Rf_rows.append(Rf_row)

    R = np.array(R_rows)
    Rf = np.array(Rf_rows)

    # Store raw Rf before dead zone
    info['Rf_raw'] = Rf.copy()
    info['R_capstan'] = R.copy()

    return R, Rf, info


def apply_dead_zone(Rf: np.ndarray, threshold: float = None,
                    deadzone_frac: float = 0.05) -> np.ndarray:
    """Apply dead zone: zero out elements below threshold."""
    Rf_out = Rf.copy()
    if threshold is None:
        max_abs = np.max(np.abs(Rf))
        if max_abs > 0:
            threshold = deadzone_frac * max_abs
        else:
            return Rf_out
    for i in range(Rf.shape[0]):
        for j in range(Rf.shape[1]):
            if abs(Rf[i, j]) < threshold:
                Rf_out[i, j] = 0.0
    return Rf_out


def normalize_Rf_to_R(Rf: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Normalize R_f so its mean absolute magnitude matches R's mean.

    This is critical because R and R_f have different physical origins:
    - R ≈ Σ(r_j) has magnitude ~ r̄ × (pulleys_per_joint)
    - R_f ≈ Σ(r_j·ΔT) has magnitude ~ r̄ × β × (path_length_factor)

    For long paths (42 elements), R_f can be 5× larger than R, causing
    the σ_f mode to dominate the R_aug SVD and corrupt σ mode.
    """
    R_mean = np.mean(np.abs(R[R != 0])) if np.any(R != 0) else 1.0
    Rf_mean = np.mean(np.abs(Rf[Rf != 0])) if np.any(Rf != 0) else 1.0
    if Rf_mean > 0 and R_mean > 0:
        scale = R_mean / Rf_mean
        Rf = Rf * scale
    return Rf


def compute_coulomb_Rf(
    design: OrigamiHandDesign,
    normalize: bool = True,
    deadzone_frac: float = 0.05
) -> np.ndarray:
    """
    Compute R_f using paper's Coulomb friction model (formula 20/25).
    With normalization and dead zone fixes.

    This is the recommended practical model.
    """
    joints, jid_to_idx = get_joint_list(design)
    n = len(joints)

    Rf_list = []
    for tendon in design.tendons.values():
        elements = [eid for eid in tendon.pulley_sequence
                    if eid >= 0 or is_hole_id(eid)]
        m = len(elements)
        if m == 0:
            Rf_list.append(np.zeros(n))
            continue

        # AR matrix (m+1, n)
        AR = np.zeros((m + 1, n))
        for seg_idx, eid in enumerate(elements):
            r = get_element_radius(eid, design)
            j_idx = get_element_joint_idx(eid, design, jid_to_idx)
            if j_idx is not None and r > 0:
                AR[seg_idx, j_idx] = r

        # M matrix (m+1, m+1)
        M = np.zeros((m + 1, m + 1))
        for i in range(m):
            M[i, i] = -1.0
            M[i, i + 1] = 1.0
        M[m, 0] = 1.0
        M[m, m] = 1.0

        # V_max diagonal
        V_max = np.zeros((m + 1, m + 1))
        for seg_idx, eid in enumerate(elements):
            if is_pulley_id(eid) and eid in design.pulleys:
                V_max[seg_idx, seg_idx] = design.pulleys[eid].friction_coefficient
            elif is_hole_id(eid) and eid in design.holes:
                V_max[seg_idx, seg_idx] = design.holes[eid].friction_coefficient

        e_v = np.ones(m + 1)
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        Rf_vec = -AR.T @ M_inv @ V_max @ e_v
        Rf_list.append(Rf_vec)

    Rf = np.array(Rf_list)

    # Dead zone
    if deadzone_frac > 0:
        Rf = apply_dead_zone(Rf, deadzone_frac=deadzone_frac)

    # Normalize to scale of R if requested
    if normalize:
        from src.models.transmission_builder import compute_R as compute_R_geometric
        R_geo = compute_R_geometric(design)
        Rf = normalize_Rf_to_R(Rf, R_geo)

    return Rf


def analyze_R_Rf(
    design: OrigamiHandDesign,
    use_capstan: bool = True,
    beta: float = 0.09,
    normalize: bool = True,
    deadzone_frac: float = 0.05
) -> dict:
    """
    Comprehensive analysis of R and R_f for a given design.

    Returns dict with:
        'R': effective R matrix
        'Rf': effective R_f matrix
        'R_geometric': geometric R (paper formula 31)
        'sigma_direction': the σ mode motion pattern
        'sigma_f_direction': the σ_f mode motion pattern
        'physical_checks': dict of physical sanity checks
    """
    from src.models.transmission_builder import compute_R as compute_R_geometric

    joints, jid_to_idx = get_joint_list(design)
    n = len(joints)

    # Get geometric R
    R_geo = compute_R_geometric(design)

    if use_capstan:
        R, Rf_raw, info = compute_capstan_R_Rf(design, beta=beta,
                                                 deadzone_frac=deadzone_frac)
        Rf_dead = apply_dead_zone(Rf_raw, deadzone_frac=deadzone_frac)

        # Normalize Rf to R's scale
        if normalize:
            Rf_norm = normalize_Rf_to_R(Rf_dead, R)
        else:
            Rf_norm = Rf_dead

        result_R = R
        result_Rf = Rf_norm
    else:
        Rf_coulomb = compute_coulomb_Rf(design, normalize=False,
                                         deadzone_frac=deadzone_frac)
        if normalize:
            Rf_norm = normalize_Rf_to_R(Rf_coulomb, R_geo)
        else:
            Rf_norm = Rf_coulomb
        result_R = R_geo
        result_Rf = Rf_norm

    # Average multiple tendons
    drive_indices = [i for (i, t) in enumerate(design.tendons.values())
                     if -1 in t.pulley_sequence or -2 in t.pulley_sequence]
    if len(drive_indices) < len(list(design.tendons.values())):
        result_R = result_R[drive_indices]
        result_Rf = result_Rf[drive_indices]

    if result_R.shape[0] > 1:
        result_R = result_R.mean(axis=0, keepdims=True)
        result_Rf = result_Rf.mean(axis=0, keepdims=True)

    # Physical checks
    checks = {}
    if n >= 3:
        flat_R = result_R.flatten()
        flat_Rf = result_Rf.flatten()
        mid = n // 2

        # Check U-shape in R
        is_ushape = flat_R[mid] < flat_R[0] * 0.9 and flat_R[mid] < flat_R[-1] * 0.9
        ratio_mid_end = min(flat_R[mid] / max(flat_R[0], 1e-10),
                            flat_R[mid] / max(flat_R[-1], 1e-10))
        checks['R_U_shape'] = bool(is_ushape)
        checks['R_mid_to_outer_ratio'] = float(ratio_mid_end)
        checks['R_min_idx'] = int(np.argmin(flat_R))

        # Check signed gradient in Rf
        if np.any(flat_Rf != 0):
            checks['Rf_left_sign'] = np.sign(flat_Rf[0])
            checks['Rf_right_sign'] = np.sign(flat_Rf[-1])
            checks['Rf_polarized'] = bool(flat_Rf[0] * flat_Rf[-1] < 0)
            checks['Rf_dead_zone_count'] = int(np.sum(np.abs(flat_Rf) < 1e-10))
        else:
            checks['Rf_all_zero'] = True

    return {
        'R': result_R,
        'Rf': result_Rf,
        'R_geometric': R_geo,
        'sigma_direction': result_R.flatten() / np.linalg.norm(result_R),
        'sigma_f_direction': result_Rf.flatten() / (np.linalg.norm(result_Rf) + 1e-10),
        'physical_checks': checks,
        'n_joints': n,
        'n_tendons': len(design.tendons),
    }
