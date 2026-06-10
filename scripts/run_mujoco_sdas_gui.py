#!/usr/bin/env python3
"""
Launch the graphical MuJoCo SDAS numerical simulation interface.

Usage:
    python scripts/run_mujoco_sdas_gui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interactive.mujoco_sdas_launcher import run_gui


if __name__ == "__main__":
    run_gui()
