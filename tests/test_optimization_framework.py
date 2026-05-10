"""
Test the computational design optimization framework.

This test demonstrates:
1. Loading an OHD design
2. Setting up the design space with editable variables
3. Running optimization to achieve a target synergy direction
4. Validating the results

Usage:
    python tests/test_optimization_framework.py [--ohd models/ohd_test/ohd_5.ohd]
"""

import sys
sys.path.insert(0, '.')

import os
import numpy as np
import argparse

from src.models.origami_design import OrigamiHandDesign
from src.models.transmission_builder import build_synergy_model, get_joint_list
from src.optimization.design_space import DesignSpace, DesignVariable, VariableType
from src.optimization.design_evaluator import DesignEvaluator
from src.optimization.objective_functions import (
    DirectionTarget, SpeedDependentTarget, CompositeObjective, normalize_direction
)
from src.optimization.optimization_engine import (
    OptimizationEngine, AlgorithmType, OptimizationResult
)


def test_direction_optimization(ohd_path: str, quick: bool = True):
    """
    Test the full optimization pipeline:
    Target: Achieve specific synergy directions (Dir0, Dir1) by
    tuning fold stiffness and hole parameters.
    
    This test uses a known base design and tries to match target
    synergy directions derived from the original design's S_aug.
    """
    print("=" * 70)
    print("TEST: Synergy Direction Optimization")
    print("=" * 70)
    
    # 1. Load design
    print(f"\n1. Loading design from: {ohd_path}")
    design = OrigamiHandDesign.load(ohd_path)
    design.summary()
    
    # 2. Get joint info
    joints, jid_to_idx = get_joint_list(design)
    n_joints = len(joints)
    n_holes = len(design.holes)
    print(f"\n  n_joints = {n_joints}, n_holes = {n_holes}")
    
    if n_joints < 2:
        print("  SKIP: Need at least 2 joints for meaningful optimization")
        return None
    
    # 3. Compute baseline synergy model to get target directions
    print(f"\n2. Computing baseline synergy model...")
    model, info = build_synergy_model(design)
    S_aug = model.S_aug
    
    print(f"  S_aug shape = {S_aug.shape}")
    for i in range(S_aug.shape[1]):
        print(f"    Direction {i}: {normalize_direction(S_aug[:, i])}")
    
    # 4. Define target directions (based on current design, slightly modified)
    #    Target: make Dir0 more uniform across joints
    #    and Dir1 more differentiated
    target_dir0 = np.ones(n_joints)  # Uniform motion of all joints
    target_dir1 = None
    if S_aug.shape[1] >= 2:
        target_dir1 = np.zeros(n_joints)
        target_dir1[0] = 1.0          # Joint 0 moves opposite to others
        target_dir1[1:] = -0.5
    
    print(f"\n3. Target directions:")
    print(f"    Dir0 (sigma):  {normalize_direction(target_dir0)}")
    if target_dir1 is not None:
        print(f"    Dir1 (sigma_f): {target_dir1} (nominal, not normalized)")
    
    # 5. Set up design space
    print(f"\n4. Setting up design space...")
    space = DesignSpace(n_joints, n_holes)
    
    # Add fold stiffness as editable (all joints)
    for i in range(n_joints):
        space.add_fold_stiffness(i, low=0.3, high=5.0, default=1.0)
    
    # Add hole distances to crease
    for i in range(n_holes):
        space.add_hole_distance_to_crease(i, low=2.0, high=12.0, default=5.0)
        space.add_hole_plate_offset(i, low=1.0, high=6.0, default=3.0)
    
    # Add damping if dampers exist
    n_dampers = len(design.dampers)
    for i in range(n_dampers):
        space.add_damping(i, low=0.1, high=15.0, default=2.0)
    
    print(space.summary())
    
    # 6. Create evaluator and objective
    print(f"\n5. Creating evaluator and objective...")
    evaluator = DesignEvaluator(design)
    
    objective = DirectionTarget(
        target_dir0=target_dir0,
        target_dir1=target_dir1,
        weight_dir0=1.0,
        weight_dir1=1.0 if target_dir1 is not None else 0.0,
        weight_ortho=0.2,
    )
    
    # 7. Run optimization
    print(f"\n6. Running optimization...")
    engine = OptimizationEngine(
        objective_fn=objective,
        evaluator=evaluator,
        param_names=space.continuous_variable_names,
        bounds=space.continuous_bounds,
        use_dynamic=(n_dampers > 0),
    )
    
    if quick:
        # Quick run: random sampling + limited DE
        result = engine.run_two_phase(
            n_explore=50,
            de_kwargs={'maxiter': 20, 'popsize': 10},
            verbose=True
        )
    else:
        # Full run
        result = engine.run(
            algorithm=AlgorithmType.DIFFERENTIAL_EVOLUTION,
            maxiter=80,
            popsize=20,
            verbose=True
        )
    
    # 8. Print results
    print(f"\n7. Results:")
    print(result.summary())
    
    # 9. Verify: compute S_aug with best params and compare
    print(f"\n8. Verification:")
    best_params = result.best_params
    eval_result = evaluator.evaluate(best_params, use_dynamic=(n_dampers > 0))
    S_best = eval_result.get('S_aug')
    
    if S_best is not None:
        print(f"  Best S_aug:")
        for i in range(S_best.shape[1]):
            print(f"    Dir {i}: {normalize_direction(S_best[:, i])}")
        
        # Compute cosine similarity with targets
        d0_best = normalize_direction(S_best[:, 0])
        cos0 = np.dot(d0_best, normalize_direction(target_dir0))
        print(f"  Dir0 vs target cosine similarity: {cos0:.6f} (1.0 = perfect)")
        
        if target_dir1 is not None and S_best.shape[1] >= 2:
            d1_best = normalize_direction(S_best[:, 1])
            cos1 = np.dot(d1_best, normalize_direction(target_dir1))
            print(f"  Dir1 vs target cosine similarity: {cos1:.6f} (1.0 = perfect)")
    
    print("\n✓ Direction optimization test completed")
    return result


def test_speed_dependent_optimization(ohd_path: str):
    """
    Test optimization for speed-dependent synergy (dynamic model).
    Only runs if the design has dampers.
    """
    print("\n" + "=" * 70)
    print("TEST: Speed-Dependent Synergy Optimization")
    print("=" * 70)
    
    design = OrigamiHandDesign.load(ohd_path)
    joints, jid_to_idx = get_joint_list(design)
    n_joints = len(joints)
    n_dampers = len(design.dampers)
    
    if n_dampers == 0:
        print("  SKIP: Design has no dampers. Speed-dependent optimization requires dampers.")
        return None
    
    print(f"\n1. Design has {n_dampers} dampers, n_joints = {n_joints}")
    
    # Target: slow closure = all joints move uniformly
    #         fast closure = only distal joints move (more localized)
    target_slow = np.ones(n_joints)  # Uniform
    target_fast = np.zeros(n_joints)
    target_fast[-min(2, n_joints):] = 1.0  # Only distal joints
    
    print(f"\n2. Targets:")
    print(f"    Slow direction: {normalize_direction(target_slow)}")
    print(f"    Fast direction: {normalize_direction(target_fast)}")
    
    # Setup design space
    space = DesignSpace(n_joints, len(design.holes))
    for i in range(n_joints):
        space.add_fold_stiffness(i, low=0.3, high=5.0)
    for i in range(n_dampers):
        space.add_damping(i, low=0.1, high=20.0, default=5.0)
    for i in range(len(design.holes)):
        space.add_hole_distance_to_crease(i, low=2.0, high=12.0)
        space.add_hole_plate_offset(i, low=1.0, high=6.0)
    
    evaluator = DesignEvaluator(design)
    
    objective = SpeedDependentTarget(
        target_slow=target_slow,
        target_fast=target_fast,
        weight_slow=1.0,
        weight_fast=1.5,
        weight_separation=0.5,
    )
    
    engine = OptimizationEngine(
        objective_fn=objective,
        evaluator=evaluator,
        param_names=space.continuous_variable_names,
        bounds=space.continuous_bounds,
        use_dynamic=True,
    )
    
    print(f"\n3. Running quick optimization...")
    result = engine.run_two_phase(
        n_explore=30,
        de_kwargs={'maxiter': 15, 'popsize': 8},
        verbose=True
    )
    
    print(f"\n4. Results:")
    print(result.summary())
    
    # Verification
    best_params = result.best_params
    eval_result = evaluator.evaluate(best_params, use_dynamic=True)
    S_s = eval_result.get('S_s')
    S_f = eval_result.get('S_f')
    
    if S_s is not None and S_f is not None:
        print(f"\n  Slow synergy direction: {normalize_direction(S_s[:, 0])}")
        print(f"  Fast synergy direction: {normalize_direction(S_f[:, 0])}")
        sep = np.linalg.norm(S_f - S_s) / max(np.linalg.norm(S_s), 1e-10)
        print(f"  Separation ||S_f - S_s||/||S_s||: {sep:.4f}")
    
    print("\n✓ Speed-dependent optimization test completed")
    return result


def main():
    parser = argparse.ArgumentParser(description="Test optimization framework")
    parser.add_argument("--ohd", default="models/ohd test/ohd_5.ohd",
                        help="Path to .ohd design file")
    parser.add_argument("--quick", action="store_true", default=True,
                        help="Run quick test (fewer iterations)")
    parser.add_argument("--full", action="store_false", dest="quick",
                        help="Run full optimization")
    parser.add_argument("--test-speed", action="store_true",
                        help="Also test speed-dependent optimization")
    args = parser.parse_args()
    
    ohd_path = os.path.abspath(args.ohd)
    if not os.path.exists(ohd_path):
        print(f"Error: Design file not found: {ohd_path}")
        print("Available designs:")
        import glob
        for f in sorted(glob.glob("models/**/*.ohd", recursive=True)):
            print(f"  {f}")
        return 1
    
    result = test_direction_optimization(ohd_path, quick=args.quick)
    
    if args.test_speed:
        result2 = test_speed_dependent_optimization(ohd_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
