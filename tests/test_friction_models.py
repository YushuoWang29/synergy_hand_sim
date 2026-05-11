#!/usr/bin/env python3
"""Comprehensive test: validate friction models against physical expectations.

Tests:
1. σ mode: R should show U-shape for long paths (Capstan model)
2. σ_f mode: Rf should show signed gradient + dead zone in middle
3. Symmetric paths: Rf should be zero (e.g., single finger ohd_1)
4. Multi-tendon averaging should not corrupt synergy directions
5. No MuJoCo windows — pure numerical analysis
"""
import sys; sys.path.insert(0, '.')
import numpy as np
from src.models.origami_design import OrigamiHandDesign
from src.models.transmission_builder import compute_R as compute_R_old
from src.models.transmission_builder import compute_Rf as compute_Rf_old
from src.models.transmission_builder import get_joint_list
from src.synergy.friction_analysis import (compute_capstan_R_Rf, compute_coulomb_Rf,
                                           normalize_Rf_to_R, apply_dead_zone)
from src.synergy.augmented_adaptive import AugmentedAdaptiveSynergyModel


def test_design(name: str, ohd_path: str):
    print(f"\n{'='*72}")
    print(f"  TEST: {name}")
    print(f"{'='*72}")

    design = OrigamiHandDesign.load(ohd_path)
    joints, jid_to_idx = get_joint_list(design)
    n = len(joints)
    print(f"  Joints: {n}")

    drive_tendons = [i for (i, t) in enumerate(design.tendons.values())
                     if -1 in t.pulley_sequence or -2 in t.pulley_sequence]
    print(f"  Drive tendons: {drive_tendons}")

    # ─── Old model ───
    R_old = compute_R_old(design)
    Rf_old = compute_Rf_old(design)

    # Average only drive tendons
    di = drive_tendons
    if R_old.shape[0] > 1:
        R_old = R_old[di].mean(axis=0, keepdims=True) if len(di) > 1 else R_old[di]
    if Rf_old.shape[0] > 1:
        rf_sel = Rf_old[di]
        Rf_old = rf_sel.mean(axis=0, keepdims=True) if len(di) > 1 else rf_sel

    # ─── Capstan model (beta=0.09) ───
    R_cap, Rf_cap_raw, _ = compute_capstan_R_Rf(design, beta=0.09, deadzone_frac=0.0)
    if R_cap.shape[0] > 1:
        R_cap = R_cap[di].mean(axis=0, keepdims=True)
        Rf_cap_raw = Rf_cap_raw[di].mean(axis=0, keepdims=True)
    Rf_cap_dead = apply_dead_zone(Rf_cap_raw, deadzone_frac=0.03)
    Rf_cap_norm = normalize_Rf_to_R(Rf_cap_dead, R_cap)

    # ─── Coulomb model with fixes ───
    Rf_coul_fixed = compute_coulomb_Rf(design, normalize=True, deadzone_frac=0.03)
    if Rf_coul_fixed.shape[0] > 1:
        Rf_coul_fixed = Rf_coul_fixed[di].mean(axis=0, keepdims=True)

    # ─── Display comparison ───
    def fmt(x):
        return np.array2string(x.flatten(), precision=3, suppress_small=True)
    print(f"\n  R_old (uniform):     {fmt(R_old)}")
    print(f"  R_cap (beta=0.09):   {fmt(R_cap)}")
    print(f"  Rf_old (raw):        {fmt(Rf_old)}")
    print(f"  Rf_cap (raw):        {fmt(Rf_cap_raw)}")
    print(f"  Rf_cap (fixed):      {fmt(Rf_cap_norm)}")
    print(f"  Rf_coul (fixed):     {fmt(Rf_coul_fixed)}")

    # ─── Physical checks ───
    def check_phys(label, arr):
        f = arr.flatten()
        if len(f) < 3:
            return
        mid = len(f) // 2
        is_u = bool(f[mid] < f[0] and f[mid] < f[-1])
        u_ratio = min(f[mid] / max(f[0], 1e-10), f[mid] / max(f[-1], 1e-10))
        has_n = np.any(f < -1e-8)
        has_p = np.any(f > 1e-8)
        pol = has_n and has_p
        dead = int(np.sum(np.abs(f) < 1e-8))
        print(f"    [{label:15s}] U={is_u}(r={u_ratio:.3f}) pol={pol} dead={dead}")

    print(f"\n  Physical checks:")
    check_phys("R_old", R_old)
    check_phys("R_cap", R_cap)
    check_phys("Rf_old", Rf_old)
    check_phys("Rf_cap_raw", Rf_cap_raw)
    check_phys("Rf_cap_fixed", Rf_cap_norm)
    check_phys("Rf_coul_fixed", Rf_coul_fixed)

    # ─── Augmented model test ───
    E_vec = np.array([design.fold_lines[j.fold_line_id].stiffness for j in joints])

    try:
        m_old = AugmentedAdaptiveSynergyModel(n, R_old, Rf_old, E_vec)
        s = m_old.S_aug
        print(f"\n  SVD old Rf:  sigma={s[:,0].round(3)}  sigma_f={s[:,1].round(3)}")
        print(f"    sigma all+? {np.all(s[:,0] > -1e-6)} (min={s[:,0].min():.3f})")
    except Exception as e:
        print(f"  Old model FAILED: {e}")

    try:
        m_fix = AugmentedAdaptiveSynergyModel(n, R_old, Rf_coul_fixed, E_vec)
        sf = m_fix.S_aug
        print(f"  SVD fixed Rf: sigma={sf[:,0].round(3)}  sigma_f={sf[:,1].round(3)}")
        print(f"    sigma all+? {np.all(sf[:,0] > -1e-6)} (min={sf[:,0].min():.3f})")
    except Exception as e:
        print(f"  Fixed model FAILED: {e}")

    # ─── Verdict ───
    print(f"\n  VERDICT:")
    nj = len(R_old.flatten())
    if nj >= 8:
        rf = R_cap.flatten()
        ro = R_old.flatten()
        m = nj // 2
        old_u = ro[m] < ro[0]
        cap_u = rf[m] < rf[0]
        print(f"  Old uniform={not old_u}  Capstan U={cap_u}")
        if cap_u and not old_u:
            print(f"  ✓ Capstan correctly captures U-shape")

    if "ohd_1" in name:
        mx = np.max(np.abs(Rf_coul_fixed.flatten()))
        print(f"  Symmetric path |Rf|_max = {mx:.6f}",
              "✓ ~0" if mx < 0.01 else "⚠ expected ~0")


# ─── Run ───
designs = [
    ("ohd_1 (single finger)",          "models/ohd test/ohd_1.ohd"),
    ("ohd_2 (3 fingers route A)",      "models/ohd test/ohd_2.ohd"),
    ("ohd_3 (3 fingers route B)",      "models/ohd test/ohd_3.ohd"),
    ("ohd_4 (4 fingers, 4 tendons)",   "models/ohd test/ohd_4.ohd"),
    ("ohd_6 (5 fingers, 1 tendon)",    "models/ohd test/ohd_6.ohd"),
    ("ohd_7 (5 fingers, hole)",        "models/ohd test/ohd_7.ohd"),
    ("ohd_8 (5 fingers, hole+damper)", "models/ohd test/ohd_8.ohd"),
]

for name, path in designs:
    test_design(name, path)

print(f"\n{'='*72}")
print(f"  ALL TESTS COMPLETE — no MuJoCo windows opened")
print(f"{'='*72}")
