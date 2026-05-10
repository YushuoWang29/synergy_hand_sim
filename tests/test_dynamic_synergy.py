"""
Test Dynamic Synergy Model
==========================
Tests the DynamicSynergyModel together with AugmentedAdaptiveSynergyModel.

Covers:
1. Unit test: DynamicSynergyModel with simple 2-joint hand
2. Unit test: Fast vs slow synergy directions
3. Unit test: Combined augmented + dynamic synergy
4. Unit test: Smooth interpolation from slow to fast
5. Integration test: From .ohd design with dampers
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import json
import os
import time

from src.synergy.base_adaptive import AdaptiveSynergyModel
from src.synergy.augmented_adaptive import AugmentedAdaptiveSynergyModel
from src.synergy.dynamic_synergy import DynamicSynergyModel
from src.models.origami_design import (
    OrigamiHandDesign, FoldLine, FoldType, Point2D, OrigamiFace,
    JointConnection, Pulley, Hole, Tendon, Actuator, Damper
)
from src.models.transmission_builder import (
    compute_R, compute_Rf, compute_damper_T,
    build_synergy_model, build_dynamic_synergy_model
)
from src.interactive.mujoco_simulator import MuJoCoSimulator


def test_01_dynamic_synergy_basic():
    """Basic unit test: 2-joint hand with 1 damper."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Dynamic Synergy (2 joints, 1 damper)")
    print("=" * 60)

    n_joints = 2
    # Transmission: sigma drives both joints equally
    R = np.array([[1.0, 1.0]])           # (1, 2)
    # Damper: only affects joint 0 (thumb)
    T = np.array([[1.0, 0.0]])            # (1, 2)
    C_diag = np.array([1.0])              # 1 damper
    E_vec = np.array([1.0, 1.0])          # equal stiffness

    model = DynamicSynergyModel(n_joints, R, E_vec, T, C_diag)

    print(f"  S_s (slow)  = {model.S_s.flatten()}")
    print(f"  S_f (fast)  = {model.S_f.flatten()}")

    # Slow closure: both joints should close equally
    q_slow = model.solve(np.array([1.0]), speed_factor=0.0)
    print(f"  q_slow      = {q_slow}")

    # Fast closure: joint 0 (damper) should move less than joint 1
    q_fast = model.solve(np.array([1.0]), speed_factor=1.0)
    print(f"  q_fast      = {q_fast}")

    # Interpolated
    q_mid = model.solve(np.array([1.0]), speed_factor=0.5)
    print(f"  q_mid(0.5)  = {q_mid}")

    # Verify: slow should be equal closure
    assert np.allclose(q_slow[0], q_slow[1], atol=1e-6), \
        f"Slow closure should be symmetric: {q_slow}"
    # Verify: fast should have joint 0 < joint 1 (damper resists)
    assert q_fast[0] < q_fast[1], \
        f"Fast closure: damper joint should move less: {q_fast}"
    # Verify: mid should be between slow and fast
    assert q_mid[0] > q_fast[0] and q_mid[0] < q_slow[0], \
        f"Mid closure should interpolate: {q_mid}"

    print("  ✓ PASSED")
    return model


def test_02_dynamic_synergy_separate_directions():
    """Test that S_s and S_f produce qualitatively different postures."""
    print("\n" + "=" * 60)
    print("TEST 2: Separate synergy directions")
    print("=" * 60)

    # SoftHand Pro-D style: 4 joints (thumb, index, middle, ring)
    n_joints = 4
    # R: all joints close together
    R = np.array([[1.0, 1.0, 1.0, 1.0]])
    # T: damper connected to thumb distal+proximal
    T = np.array([[1.0, 0.0, 0.0, 0.0]])
    C_diag = np.array([5.0])
    E_vec = np.array([1.0, 1.0, 1.0, 1.0])

    model = DynamicSynergyModel(n_joints, R, E_vec, T, C_diag)

    print(f"  S_s = {model.S_s.flatten()}")
    print(f"  S_f = {model.S_f.flatten()}")
    print(f"  Δ   = S_f - S_s = {model.synergy_diff.flatten()}")

    # Slow: all fingers close equally → power grasp
    q_slow = model.solve(np.array([1.0]), speed_factor=0.0)
    print(f"  q_slow (power grasp) = {q_slow}")

    # Fast: thumb opens relative → pinch
    q_fast = model.solve(np.array([1.0]), speed_factor=1.0)
    print(f"  q_fast (pinch)       = {q_fast}")

    # Power grasp: all positive, roughly equal
    assert np.all(q_slow > 0), "Slow should close all joints"
    # Pinch: thumb moves differently from others
    assert not np.allclose(q_fast, q_slow, atol=1e-3), \
        "Fast and slow should differ: thumb behavior changes"

    print("  ✓ PASSED")
    return model


def test_03_combined_augmented_dynamic():
    """Test combining DynamicSynergyModel with augmented inputs."""
    print("\n" + "=" * 60)
    print("TEST 3: Augmented + Dynamic (combined R_aug)")
    print("=" * 60)

    n_joints = 3
    # R: basic synergy (close all)
    R = np.array([[1.0, 1.0, 1.0]])
    # R_f: friction sliding (thumb vs others)
    R_f = np.array([[1.0, -0.5, -0.5]])
    R_aug = np.vstack([R, R_f])

    # Dampers
    T = np.array([[1.0, 0.0, 0.0]])  # damper on joint 0
    C_diag = np.array([2.0])
    E_vec = np.array([1.0, 1.0, 1.0])

    model = DynamicSynergyModel(n_joints, R_aug, E_vec, T, C_diag)

    print("  Slow synergies:")
    print(f"    S_s[0] = {model.S_s[:, 0]} (basic closure)")
    print(f"    S_s[1] = {model.S_s[:, 1]} (augmented slip)")

    print("  Fast synergies:")
    print(f"    S_f[0] = {model.S_f[:, 0]} (basic closure)")
    print(f"    S_f[1] = {model.S_f[:, 1]} (augmented slip)")

    # Test 1: Only basic synergy, slow vs fast
    sigma_dyn = np.array([1.0])
    sigma_f = np.array([0.0])

    q_slow = model.solve(np.concatenate([sigma_dyn, sigma_f]), speed_factor=0.0)
    q_fast = model.solve(np.concatenate([sigma_dyn, sigma_f]), speed_factor=1.0)
    print(f"  Basic only: slow={q_slow}  fast={q_fast}")

    # Slow: all joints should be positive
    assert np.all(q_slow > 0), "Slow basic: all joints closed"

    # Test 2: Basic + augmented, slow vs fast
    sigma_dyn = np.array([1.0])
    sigma_f = np.array([0.3])

    q_slow2 = model.solve(np.concatenate([sigma_dyn, sigma_f]), speed_factor=0.0)
    q_fast2 = model.solve(np.concatenate([sigma_dyn, sigma_f]), speed_factor=1.0)
    print(f"  Basic+Aug: slow={q_slow2}  fast={q_fast2}")

    # With augmented input, thumb (joint 0) should be more closed
    assert q_slow2[0] > q_slow[0], \
        "Augmented input should increase thumb closure"

    # Test solve_combined: with use_dynamic_on_f=True should match solve()
    q_combined = model.solve_combined(
        sigma_dyn, sigma_f, speed_factor=0.5, use_dynamic_on_f=True)
    q_via_solve = model.solve(
        np.concatenate([sigma_dyn, sigma_f]), speed_factor=0.5)
    print(f"  Combined(0.5, dynamic_on_f=True): {q_combined}")
    print(f"  Via solve():                       {q_via_solve}")
    assert np.allclose(q_combined, q_via_solve, atol=1e-6), \
        "solve_combined(use_dynamic_on_f=True) should equal solve()"

    # Test solve_combined with use_dynamic_on_f=False
    q_combined_static = model.solve_combined(
        sigma_dyn, sigma_f, speed_factor=0.5, use_dynamic_on_f=False)
    print(f"  Combined(0.5, dynamic_on_f=False): {q_combined_static}")
    # With dynamic_on_f=False, only sigma_dyn is speed-interpolated
    # sigma_f uses S_s (slow synergy). So this should differ from solve().
    assert not np.allclose(q_combined_static, q_via_solve, atol=1e-6), \
        "solve_combined(dynamic_on_f=False) should differ from solve()"
    print("  ✓ solve_combined modes verified")


    print("  ✓ PASSED")
    return model


def test_04_smooth_transition():
    """Test smooth interpolation from speed_factor=0 to 1."""
    print("\n" + "=" * 60)
    print("TEST 4: Smooth transition slow→fast")
    print("=" * 60)

    n_joints = 3
    R = np.array([[1.0, 1.0, 1.0]])
    T = np.array([[1.0, 0.5, 0.0]])
    C_diag = np.array([3.0])
    E_vec = np.array([2.0, 1.0, 1.0])

    model = DynamicSynergyModel(n_joints, R, E_vec, T, C_diag)

    speeds = np.linspace(0, 1, 11)
    q_hist = []
    for alpha in speeds:
        q = model.solve(np.array([1.0]), speed_factor=alpha)
        q_hist.append(q)
        print(f"  α={alpha:.1f}  q={q}")

    q_hist = np.array(q_hist)

    # Verify monotonic change in joint 0 (damper joint)
    assert np.all(np.diff(q_hist[:, 0]) <= 0) or np.all(np.diff(q_hist[:, 0]) >= 0), \
        "Joint 0 should change monotonically with speed"
    # First and last should differ
    assert not np.allclose(q_hist[0], q_hist[-1], atol=1e-3), \
        "Slow and fast closures should differ"

    print("  ✓ PASSED")


def test_05_design_with_dampers():
    """
    Integration test: Build a synthetic OrigamiHandDesign with dampers,
    then compute R, Rf, T, and build the DynamicSynergyModel.
    """
    print("\n" + "=" * 60)
    print("TEST 5: From OrigamiHandDesign (with dampers)")
    print("=" * 60)

    design = OrigamiHandDesign("test_dynamic", material_thickness=2.0)

    # Fold lines (simple 4-joint hand)
    fold_lines = [
        FoldLine(1, Point2D(0, 0), Point2D(0, 10), FoldType.VALLEY, stiffness=1.0),
        FoldLine(2, Point2D(10, 0), Point2D(10, 10), FoldType.VALLEY, stiffness=1.0),
        FoldLine(3, Point2D(20, 0), Point2D(20, 10), FoldType.VALLEY, stiffness=1.0),
        FoldLine(4, Point2D(30, 0), Point2D(30, 10), FoldType.VALLEY, stiffness=1.0),
    ]
    for fl in fold_lines:
        design.add_fold_line(fl)

    # Pulleys (for tendons)
    pulleys = [
        Pulley(101, Point2D(2, 2), radius=3.5, friction_coefficient=0.3,
               attached_fold_line_id=1),
        Pulley(102, Point2D(12, 2), radius=3.5, friction_coefficient=0.3,
               attached_fold_line_id=2),
        Pulley(103, Point2D(22, 2), radius=3.5, friction_coefficient=0.3,
               attached_fold_line_id=3),
        Pulley(104, Point2D(32, 2), radius=3.5, friction_coefficient=0.3,
               attached_fold_line_id=4),
        Pulley(105, Point2D(2, 8), radius=3.5, friction_coefficient=0.3,
               attached_fold_line_id=1),
        Pulley(106, Point2D(12, 8), radius=3.5, friction_coefficient=0.3,
               attached_fold_line_id=2),
        Pulley(107, Point2D(22, 8), radius=3.5, friction_coefficient=0.3,
               attached_fold_line_id=3),
        Pulley(108, Point2D(32, 8), radius=3.5, friction_coefficient=0.3,
               attached_fold_line_id=4),
    ]
    for p in pulleys:
        design.add_pulley(p)

    # Dampers: one damper on joint 1 (fold_line 1) with damping coeff 5.0
    design.add_damper(Damper(
        id=1,
        position=Point2D(5, 5),
        attached_fold_line_ids=[1],
        transmission_ratios=[1.0],
        damping_coefficient=5.0,
        name="thumb_damper"
    ))

    # Tendon: all pulleys + actuators
    design.add_tendon(Tendon(1, [101, 105, 102, 106, 103, 107, 104, 108]))

    # Compute matrices
    R = compute_R(design)
    Rf = compute_Rf(design)
    T = compute_damper_T(design)
    E_vec = np.array([fl.stiffness for fl in fold_lines])

    print(f"  R  = {R}")
    print(f"  Rf = {Rf}")
    print(f"  T  = {T}")
    print(f"  E  = {E_vec}")

    assert R.shape[0] == 1, f"Expected 1 tendon row, got {R.shape[0]}"
    assert T.shape[0] == 1, f"Expected 1 damper row, got {T.shape[0]}"
    assert T[0, 0] == 1.0, "Damper should be connected to joint 0 (fold_line 1)"

    # Build dynamic synergy model
    model = DynamicSynergyModel(
        n_joints=len(fold_lines),
        R=R,
        E_vec=E_vec,
        T=T,
        C_diag=np.array([5.0])
    )

    print(f"  S_s = {model.S_s.flatten()}")
    print(f"  S_f = {model.S_f.flatten()}")

    q_slow = model.solve(np.array([1.0]), speed_factor=0.0)
    q_fast = model.solve(np.array([1.0]), speed_factor=1.0)
    print(f"  q_slow = {q_slow}")
    print(f"  q_fast = {q_fast}")

    # With damper on joint 0, fast closure should reduce joint 0 relative to others
    ratio_slow = q_slow[0] / (q_slow[1:].mean() + 1e-10)
    ratio_fast = q_fast[0] / (q_fast[1:].mean() + 1e-10)
    print(f"  thumb/others ratio: slow={ratio_slow:.3f}  fast={ratio_fast:.3f}")
    assert ratio_fast < ratio_slow, \
        "Fast synergy should reduce damper joint relative motion"

    print("  ✓ PASSED")


def test_06_build_dynamic_synergy_model():
    """Integration test for build_dynamic_synergy_model() factory function."""
    print("\n" + "=" * 60)
    print("TEST 6: build_dynamic_synergy_model factory")
    print("=" * 60)

    design = OrigamiHandDesign("test_dynamic_factory", material_thickness=2.0)

    fold_lines = [
        FoldLine(1, Point2D(0, 0), Point2D(0, 10), FoldType.VALLEY, stiffness=1.0),
        FoldLine(2, Point2D(10, 0), Point2D(10, 10), FoldType.VALLEY, stiffness=1.0),
        FoldLine(3, Point2D(20, 0), Point2D(20, 10), FoldType.VALLEY, stiffness=1.0),
    ]
    for fl in fold_lines:
        design.add_fold_line(fl)

    for pid, (x, fold_id) in enumerate([(2, 1), (12, 2), (22, 3),
                                          (8, 1), (18, 2), (28, 3)], start=201):
        design.add_pulley(Pulley(pid, Point2D(x, 2), radius=3.5,
                                  friction_coefficient=0.3,
                                  attached_fold_line_id=fold_id))

    # Damper on fold_line 1
    design.add_damper(Damper(
        id=1,
        position=Point2D(5, 5),
        attached_fold_line_ids=[1],
        transmission_ratios=[1.0],
        damping_coefficient=3.0
    ))

    design.add_tendon(Tendon(1, [201, 202, 203, 204, 205, 206]))

    # Test without augmented
    model_basic, info = build_dynamic_synergy_model(
        design, use_augmented=False, speed_factor=0.5)
    print(f"  Basic model: {model_basic}")
    print(f"  R_aug shape: {model_basic.R.shape} (should be (1, 3))")
    assert model_basic.R.shape == (1, 3), \
        f"Expected (1, 3), got {model_basic.R.shape}"

    # Test with augmented (should be (2, 3) when Rf is computed)
    design.add_tendon(Tendon(2, [201, 202, 203, 204, 205, 206]))  # two tendons
    model_aug, info2 = build_dynamic_synergy_model(
        design, use_augmented=True, speed_factor=0.0)
    print(f"  Augmented model R_aug shape: {model_aug.R.shape}")

    # Test solve
    q_slow = model_basic.solve(np.array([1.0]), speed_factor=0.0)
    q_fast = model_basic.solve(np.array([1.0]), speed_factor=1.0)
    print(f"  Factory: q_slow={q_slow}  q_fast={q_fast}")

    print("  ✓ PASSED")


def test_07_mujoco_visualization():
    """
    Optional: visualize dynamic synergy in MuJoCo.
    Only runs if URDF exists. Use an existing .ohd file if available.
    """
    print("\n" + "=" * 60)
    print("TEST 7: MuJoCo visualization (optional)")
    print("=" * 60)

    # Try loading an existing .ohd file
    ohd_path = "test_hole_design.ohd"
    if not os.path.exists(ohd_path):
        print("  SKIP: no .ohd file found")
        return

    with open(ohd_path, 'r') as f:
        design_data = json.load(f)
    design = OrigamiHandDesign.from_dict(design_data)

    # Try building synergy
    R = compute_R(design)
    Rf = compute_Rf(design)
    T_mat = compute_damper_T(design)
    E_vec = np.array([
        design.fold_lines[j.fold_line_id].stiffness
        for j in get_joint_list(design)[0]
    ])
    C_diag = np.array([dm.damping_coefficient for dm in design.dampers.values()])

    if T_mat.size == 0:
        print("  No dampers in design. Testing augmented only.")

    print(f"  R  shape: {R.shape}")
    print(f"  T  shape: {T_mat.shape}")

    # Just test the matrix computations
    if T_mat.size > 0:
        model = DynamicSynergyModel(R.shape[1], R, E_vec, T_mat, C_diag)
        q_slow = model.solve(np.array([1.0]), speed_factor=0.0)
        q_fast = model.solve(np.array([1.0]), speed_factor=1.0)
        print(f"  q_slow = {q_slow}")
        print(f"  q_fast = {q_fast}")

    print("  ✓ PASSED (matrix computation only)")


# ====== Add import for get_joint_list used in test_07 ======
from src.models.transmission_builder import get_joint_list


if __name__ == "__main__":
    print("=" * 60)
    print("Dynamic Synergy Test Suite")
    print("=" * 60)

    test_01_dynamic_synergy_basic()
    test_02_dynamic_synergy_separate_directions()
    test_03_combined_augmented_dynamic()
    test_04_smooth_transition()
    test_05_design_with_dampers()
    test_06_build_dynamic_synergy_model()
    test_07_mujoco_visualization()

    print("\n" + "=" * 60)
    print("All tests PASSED ✓")
    print("=" * 60)
