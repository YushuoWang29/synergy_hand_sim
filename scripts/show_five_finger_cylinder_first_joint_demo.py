#!/usr/bin/env python3
"""Open the final five-finger horizontal-cylinder demo in MuJoCo viewer."""

from __future__ import annotations

import csv
from pathlib import Path

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = (
    ROOT
    / "docs"
    / "Physical Simulation"
    / "poster"
    / "generated_figures"
    / "five_finger_horizontal_cylinder_radius2p5_first_joint_sweep"
    / "fixed_firstjoint_y145_z130"
)
XML_PATH = DEMO_DIR / "fixed_firstjoint_y145_z130.xml"
LOG_PATH = DEMO_DIR / "fixed_firstjoint_y145_z130_log.csv"


def load_last_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        last: dict[str, str] | None = None
        for row in rows:
            last = row
    if last is None:
        raise RuntimeError(f"No data rows found in {path}")
    return last


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    final = load_last_row(LOG_PATH)

    for name, value in final.items():
        if not name.startswith("q_joint_"):
            continue
        joint_name = name.removeprefix("q_")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)

    mujoco.mj_forward(model, data)
    print("Loaded final demo pose:")
    print(f"  XML: {XML_PATH}")
    print(f"  log: {LOG_PATH}")
    print("Close the MuJoCo window to return to the terminal.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
