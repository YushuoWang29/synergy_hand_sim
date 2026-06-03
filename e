#!/usr/bin/env python3
"""
SDAS verification script — independent cross-validation.

Usage:
    # Uniform tension comparison (tests whether Capstan matters)
    python scripts/run_verification.py tests/test_3.dxf --mode uniform

    # KKT Newton comparison (independent solver implementation)
    python scripts/run_verification.py tests/test_3.dxf --mode kkt

    # Differential mode
    python scripts/run_verification.py tests/test_3.dxf --sigma_f 0.1
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import argparse
import os

from src.models.origami_parser import OrigamiParser
from src.synergy.sdas_model import SDASModel
from src.verification.pinocchio_solver import (
    VerificationKKTSolver,
    UniformTensionSolver,
    compute_transmission_vectors,
)
from src.verification.comparators import compare_at_sigma


def get_joint_names(design) -> list:
    """Get human-readable joint names from design."""
    names = []
    for joint in design.joints:
        fl = design.fold_lines.get(joint.fold_line_id)
        if fl:
            names.append(f"L{joint.fold_line_id}_{fl.fold_type.value}")
        else:
            names.append(f"J{joint.id}")
    return names


def main():
    parser = argparse.ArgumentParser(
        description="SDAS Verification — Independent Cross-Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dxf_file", help="DXF hand model file")
    parser.add_argument("--mode", choices=["uniform", "kkt"],
                        default="uniform",
                        help="Verification mode (default: uniform)")
    parser.add_argument("--beta", type=float, default=0.09,
                        help="Capstan friction coefficient (default: 0.09)")
    parser.add_argument("--max-sigma", type=float, default=1.0,
                        help="Maximum sigma value (default: 1.0 rad)")
    parser.add_argument("--n-steps", type=int, default=30,
                        help="Number of sigma steps (default: 30)")
    parser.add_argument("--sigma-f", type=float, default=0.0,
                        help="Sigma_f differential mode (default: 0.0)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    print("=" * 60)
    print("SDAS Verification — Independent Cross-Validation")
    print("=" * 60)

    # 1. Load design
    print(f"\n[1] Loading design: {args.dxf_file}")
    parser_dxf = OrigamiParser(point_tolerance=2.0)
    design = parser_dxf.parse(args.dxf_file)
    n_joints = len(design.joints)
    print(f"    Joints: {n_joints}")

    # 2. Compute transmission vectors (pure Euler-Capstan formula)
    print(f"\n[2] Computing transmission vectors (Euler-Capstan)...")
    r_geo, r_A, r_B = compute_transmission_vectors(design, beta=args.beta)
    print(f"    r_geo (geometric sum):        {r_geo}")
    print(f"    r_A   (Capstan from Motor A): {r_A}")
    print(f"    r_B   (Capstan from Motor B): {r_B}")

    # 3. Build SDAS model (reference implementation)
    print(f"\n[3] Building SDAS model (Schur complement)...")
    stiffness = np.array(design.get_joint_stiffness_list())
    print(f"    Stiffness: {stiffness}")

    sdas = SDASModel(n_joints, r_A, r_B, stiffness)
    joint_names = get_joint_names(design)

    # 4. Build verification solver
    print(f"\n[4] Building verification solver...")
    if args.mode == "uniform":
        solver = UniformTensionSolver(r_geo=r_geo, stiffness=stiffness)
        description = "Uniform Tension (geometric sum Σrk, no Capstan)"
    elif args.mode == "kkt":
        solver = VerificationKKTSolver(r_A=r_A, r_B=r_B, stiffness=stiffness)
        description = "KKT Newton (same R_A/R_B as SDAS, different solver)"
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # 5. Run comparison
    print(f"\n[5] Running comparison...")
    print(f"    SDAS vs {description}")
    print(f"    beta = {args.beta}, sigma_f = {args.sigma_f}")

    sigma_values = np.linspace(0.001, args.max_sigma, args.n_steps)
    result = compare_at_sigma(
        sdas_model=sdas,
        verification_solver=solver,
        sigma_values=sigma_values,
        sigma_f=args.sigma_f,
        joint_names=joint_names,
        verbose=args.verbose,
    )

    # 6. Results
    print("\n" + result.summary())
    print("\n" + result.max_joint_angle_table())


if __name__ == "__main__":
    main()
