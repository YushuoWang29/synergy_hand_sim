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

from src.simulation.mujoco_sdas import (
    MuJoCoSDASSimulator,
    VideoConfig,
    load_ohd_simulation,
)
from src.simulation.mujoco_sdas_presets import apply_object_override, tuple_or_default


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MuJoCo numerical simulation with SDAS/no-damper control."
    )
    parser.add_argument("ohd_file", help="Hand-design .ohd or simulation-definition .ohd.")
    parser.add_argument("--out", default=None, help="Override output directory.")
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration.")
    parser.add_argument("--dt", type=float, default=None, help="Override MuJoCo timestep.")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override analysis step count. If used, dt is computed as duration / steps.",
    )
    parser.add_argument(
        "--object",
        choices=["keep", "none", "box", "cylinder", "sphere", "scanned_mug"],
        default="keep",
        help="Override grasp object preset. Default keeps the object defined in .ohd.",
    )
    parser.add_argument(
        "--object-pos",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Object position in MuJoCo world coordinates.",
    )
    parser.add_argument(
        "--object-size",
        nargs="+",
        type=float,
        default=None,
        help="Object size tuple. Example: box uses 3 values, sphere uses 1, cylinder uses 2.",
    )
    parser.add_argument("--object-scale", type=float, default=None, help="Scale for scanned object assets.")
    parser.add_argument("--object-mass", type=float, default=None, help="Object mass.")
    parser.add_argument("--fixed-object", action="store_true", help="Create the selected object without a freejoint.")
    parser.add_argument("--contact", action="store_true", help="Enable contact even if the selected .ohd has none.")
    parser.add_argument("--video", action="store_true", help="Export an animated GIF of the simulation process.")
    parser.add_argument("--video-fps", type=float, default=24.0, help="Animated GIF frame rate.")
    parser.add_argument("--keep-video-frames", action="store_true", help="Keep intermediate rendered video frames.")
    args = parser.parse_args()

    config = load_ohd_simulation(args.ohd_file)
    if args.out:
        config.output_dir = Path(args.out).resolve()
    if args.duration is not None:
        config.duration = args.duration
    if args.steps is not None:
        if args.steps <= 0:
            raise ValueError("--steps must be positive.")
        config.dt = config.duration / float(args.steps)
    elif args.dt is not None:
        config.dt = args.dt

    if args.object == "none":
        apply_object_override(config, "none", force_contact=args.contact)
    elif args.object != "keep":
        position = tuple_or_default(args.object_pos, (0.045, 0.125, 0.0785))
        if len(position) != 3:
            raise ValueError("--object-pos must contain exactly 3 values.")
        apply_object_override(
            config,
            args.object,
            position,  # type: ignore[arg-type]
            args.object_size,
            args.object_scale,
            args.object_mass,
            args.fixed_object,
            force_contact=True,
        )
    elif args.contact:
        apply_object_override(config, "keep", force_contact=True)
    if args.video:
        config.video = VideoConfig(
            enabled=True,
            fps=args.video_fps,
            keep_frames=args.keep_video_frames,
            format="gif",
        )

    result = MuJoCoSDASSimulator(config).run()

    print(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()
