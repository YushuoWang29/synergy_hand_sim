#!/usr/bin/env python3
"""
Computational Design Optimization for Origami Hand Synergies.

Usage:
    python scripts/run_optimization.py --ohd path/to/design.ohd --mode direction
    python scripts/run_optimization.py --ohd path/to/design.ohd --mode speed
    python scripts/run_optimization.py --ohd path/to/design.ohd --mode both

Modes:
    direction: Match target synergy directions (σ and σ_f)
    speed: Match target slow vs fast synergy behavior (requires dampers)
    both: Run both optimizations sequentially

Editable variables controlled during optimization:
    - fold_stiffness_i: Joint stiffness [0.1, 10.0]
    - hole_distance_i: Hole distance to crease [2.0, 15.0]
    - hole_offset_i: Hole plate offset (height h) [0.5, 8.0]
    - damping_i: Damping coefficient [0.0, 20.0]
"""

import sys
sys.path.insert(0, '.')

import os
import argparse
import numpy as np

from src.models.origami_design import OrigamiHandDesign
from src.models.transmission_builder import (
    build_synergy_model, build_dynamic_synergy_model, get_joint_list
)
from src.optimization.design_space import DesignSpace
from src.optimization.design_evaluator import DesignEvaluator
from src.optimization.objective_functions import (
    DirectionTarget, SpeedDependentTarget, CompositeObjective, normalize_direction
)
from src.optimization.optimization_engine import (
    OptimizationEngine, AlgorithmType
)


def build_design_space(design, include_holes=True, include_damping=True):
    """Build a DesignSpace from the loaded design."""
    joints, _ = get_joint_list(design)
    n_joints = len(joints)
    n_holes = len(design.holes)
    n_dampers = len(design.dampers)
    
    space = DesignSpace(n_joints, n_holes)
    
    # Fold stiffness
    for i in range(n_joints):
        space.add_fold_stiffness(i, low=0.1, high=10.0, default=1.0)
    
    # Hole parameters
    if include_holes:
        for i in range(n_holes):
            space.add_hole_distance_to_crease(i, low=2.0, high=15.0, default=5.0)
            space.add_hole_plate_offset(i, low=1.0, high=8.0, default=3.0)
    
    # Damping
    if include_damping:
        for i in range(n_dampers):
            space.add_damping(i, low=0.0, high=20.0, default=2.0)
    
    return space


def run_direction_optimization(design, ohd_path, output_dir, de_kwargs=None):
    """Optimize for synergy direction matching."""
    print("\n" + "=" * 70)
    print("DIRECTION OPTIMIZATION")
    print("Target: Match σ (Dir0) and σ_f (Dir1) synergy directions")
    print("=" * 70)
    
    joints, _ = get_joint_list(design)
    n_joints = len(joints)
    n_dampers = len(design.dampers)
    
    # Get baseline synergy
    model, info = build_synergy_model(design)
    S_aug = model.S_aug
    
    print(f"\nBaseline S_aug:")
    for i in range(S_aug.shape[1]):
        print(f"  Dir {i}: {normalize_direction(S_aug[:, i])}")
    
    # === TARGET DEFINITION ===
    # Customize these targets based on your design goals
    # Example: Dir0 = all joints flex (uniform), Dir1 = thumb opposition
    
    target_dir0 = np.ones(n_joints)  # All joints close together
    
    target_dir1 = None
    if S_aug.shape[1] >= 2:
        target_dir1 = np.ones(n_joints)
        target_dir1[0] = 2.0  # More motion in joint 0
        target_dir1[1:] = -1.0  # Others extend
    
    print(f"\nTarget direction 0 (σ): {normalize_direction(target_dir0)}")
    if target_dir1 is not None:
        print(f"Target direction 1 (σ_f): {normalize_direction(target_dir1)}")
    
    # Setup
    space = build_design_space(design)
    print(f"\nDesign space: {len(space.continuous_variable_names)} variables")
    
    evaluator = DesignEvaluator(design)
    objective = DirectionTarget(
        target_dir0=target_dir0,
        target_dir1=target_dir1,
        weight_dir0=1.0,
        weight_dir1=1.0 if target_dir1 is not None else 0.0,
        weight_ortho=0.2,
    )
    
    engine = OptimizationEngine(
        objective_fn=objective,
        evaluator=evaluator,
        param_names=space.continuous_variable_names,
        bounds=space.continuous_bounds,
        use_dynamic=(n_dampers > 0),
    )
    
    de_kwargs = de_kwargs or {}
    de_kwargs.setdefault('maxiter', 80)
    de_kwargs.setdefault('popsize', 20)
    
    print("\nRunning optimization (Differential Evolution)...")
    result = engine.run_differential_evolution(**de_kwargs, verbose=True)
    
    print("\n" + result.summary())
    
    # Save result
    result_dir = output_dir
    os.makedirs(result_dir, exist_ok=True)
    
    # Parameter summary
    param_path = os.path.join(result_dir, "direction_best_params.txt")
    with open(param_path, 'w') as f:
        f.write(f"Direction Optimization Results\n")
        f.write(f"Cost: {result.best_cost:.6f}\n")
        f.write(f"Algorithm: {result.algorithm}\n\n")
        f.write("Best Parameters:\n")
        for k, v in sorted(result.best_params.items()):
            f.write(f"  {k} = {v:.6f}\n")
    print(f"\nParameters saved to: {param_path}")
    
    # Verification
    print("\nVerification with best parameters:")
    eval_result = evaluator.evaluate(result.best_params, use_dynamic=(n_dampers > 0))
    S_best = eval_result.get('S_aug')
    
    if S_best is not None:
        print(f"  Best S_aug:")
        for i in range(S_best.shape[1]):
            print(f"    Dir {i}: {normalize_direction(S_best[:, i])}")
        
        cos0 = np.dot(normalize_direction(S_best[:, 0]),
                     normalize_direction(target_dir0))
        print(f"  Dir0 cosine similarity: {cos0:.4f}")
        
        if target_dir1 is not None and S_best.shape[1] >= 2:
            cos1 = np.dot(normalize_direction(S_best[:, 1]),
                         normalize_direction(target_dir1))
            print(f"  Dir1 cosine similarity: {cos1:.4f}")
    
    return result


def run_speed_optimization(design, ohd_path, output_dir, de_kwargs=None):
    """Optimize for speed-dependent synergy."""
    print("\n" + "=" * 70)
    print("SPEED-DEPENDENT OPTIMIZATION")
    print("Target: Different synergy at slow vs fast speeds")
    print("=" * 70)
    
    joints, _ = get_joint_list(design)
    n_joints = len(joints)
    n_dampers = len(design.dampers)
    
    if n_dampers == 0:
        print("ERROR: No dampers found! Speed-dependent optimization requires dampers.")
        print("Add dampers to your design first.")
        return None
    
    # Get baseline dynamic synergy
    dyn_model, dyn_info = build_dynamic_synergy_model(design)
    S_s = dyn_model.S_s
    S_f = dyn_model.fast_synergy
    
    print(f"\nBaseline slow synergy: {normalize_direction(S_s[:, 0])}")
    print(f"Baseline fast synergy: {normalize_direction(S_f[:, 0])}")

    print(f"Separation: ||S_f - S_s||/||S_s|| = {np.linalg.norm(S_f - S_s) / max(np.linalg.norm(S_s), 1e-10):.4f}")
    
    # === TARGET DEFINITION ===
    # Example: slow=power grasp (uniform), fast=pinch (distal joints)
    
    target_slow = np.ones(n_joints)
    target_fast = np.zeros(n_joints)
    target_fast[-min(3, n_joints):] = 1.0  # Distal joints
    
    print(f"\nTarget slow (uniform grasp): {normalize_direction(target_slow)}")
    print(f"Target fast (distal pinch): {normalize_direction(target_fast)}")
    
    # Setup
    space = build_design_space(design)
    print(f"\nDesign space: {len(space.continuous_variable_names)} variables")
    
    evaluator = DesignEvaluator(design)
    objective = SpeedDependentTarget(
        target_slow=target_slow,
        target_fast=target_fast,
        weight_slow=1.0,
        weight_fast=1.5,  # Prioritize fast mode
        weight_separation=0.3,
    )
    
    engine = OptimizationEngine(
        objective_fn=objective,
        evaluator=evaluator,
        param_names=space.continuous_variable_names,
        bounds=space.continuous_bounds,
        use_dynamic=True,
    )
    
    de_kwargs = de_kwargs or {}
    de_kwargs.setdefault('maxiter', 80)
    de_kwargs.setdefault('popsize', 20)
    
    print("\nRunning optimization...")
    result = engine.run_two_phase(
        n_explore=200,
        de_kwargs=de_kwargs,
        verbose=True
    )
    
    print("\n" + result.summary())
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    param_path = os.path.join(output_dir, "speed_best_params.txt")
    with open(param_path, 'w') as f:
        f.write(f"Speed-Dependent Optimization Results\n")
        f.write(f"Cost: {result.best_cost:.6f}\n")
        f.write(f"Algorithm: {result.algorithm}\n\n")
        f.write("Best Parameters:\n")
        for k, v in sorted(result.best_params.items()):
            f.write(f"  {k} = {v:.6f}\n")
    print(f"\nParameters saved to: {param_path}")
    
    # Verification
    print("\nVerification with best parameters:")
    eval_result = evaluator.evaluate(result.best_params, use_dynamic=True)
    S_s_best = eval_result.get('S_s')
    S_f_best = eval_result.get('S_f')
    
    if S_s_best is not None and S_f_best is not None:
        print(f"  Best slow:  {normalize_direction(S_s_best[:, 0])}")
        print(f"  Best fast:  {normalize_direction(S_f_best[:, 0])}")
        sep = np.linalg.norm(S_f_best - S_s_best) / max(np.linalg.norm(S_s_best), 1e-10)
        print(f"  Separation: {sep:.4f}")
        
        cos_slow = np.dot(normalize_direction(S_s_best[:, 0]),
                         normalize_direction(target_slow))
        cos_fast = np.dot(normalize_direction(S_f_best[:, 0]),
                         normalize_direction(target_fast))
        print(f"  Slow similarity: {cos_slow:.4f}")
        print(f"  Fast similarity: {cos_fast:.4f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Optimize origami hand design parameters for synergy targets"
    )
    parser.add_argument("--ohd", required=True,
                        help="Path to .ohd design file")
    parser.add_argument("--mode", choices=['direction', 'speed', 'both'],
                        default='direction',
                        help="Optimization mode")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: models/<name>/optimization/)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (fewer iterations)")
    args = parser.parse_args()
    
    # Load design
    ohd_path = os.path.abspath(args.ohd)
    if not os.path.exists(ohd_path):
        print(f"ERROR: Design file not found: {ohd_path}")
        return 1
    
    hand_name = os.path.splitext(os.path.basename(ohd_path))[0]
    output_dir = args.output or os.path.join(
        os.path.dirname(ohd_path), f"{hand_name}_optimization"
    )
    
    print(f"Loading design: {ohd_path}")
    design = OrigamiHandDesign.load(ohd_path)
    design.summary()
    
    de_kwargs = {'maxiter': 20, 'popsize': 10} if args.quick else None
    
    if args.mode in ('direction', 'both'):
        run_direction_optimization(design, ohd_path, output_dir, de_kwargs)
    
    if args.mode in ('speed', 'both'):
        run_speed_optimization(design, ohd_path, output_dir, de_kwargs)
    
    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
