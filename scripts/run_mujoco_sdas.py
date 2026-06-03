#!/usr/bin/env python3
"""
Run a no-damper MuJoCo SDAS simulation from an .ohd simulation file.

Examples:
    python scripts/run_mujoco_sdas.py "models/ohd test/mujoco_sdas_step.ohd"
    python scripts/run_mujoco_sdas.py "models/ohd test/three_finger_gripper_b.ohd" --out outputs/manual_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.mujoco_sdas import load_ohd_simulation, run_ohd_mujoco_simulation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MuJoCo numerical simulation with SDAS/no-damper control."
    )
    parser.add_argument("ohd_file", help="Hand-design .ohd or simulation-definition .ohd.")
    parser.add_argument("--out", default=None, help="Override output directory.")
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration.")
    parser.add_argument("--dt", type=float, default=None, help="Override MuJoCo timestep.")
    args = parser.parse_args()

    if args.duration is None and args.dt is None:
        result = run_ohd_mujoco_simulation(args.ohd_file, args.out)
    else:
        config = load_ohd_simulation(args.ohd_file)
        if args.out:
            config.output_dir = Path(args.out).resolve()
        if args.duration is not None:
            config.duration = args.duration
        if args.dt is not None:
            config.dt = args.dt
        from src.simulation.mujoco_sdas import MuJoCoSDASSimulator

        result = MuJoCoSDASSimulator(config).run()

    print(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()

