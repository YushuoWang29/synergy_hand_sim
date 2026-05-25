# src/visualization/__init__.py
"""
Visualization tools for the synergy hand simulation project.

Only MuJoCo-based visualization is retained.
MeshCat-based visualization (origami_visualizer) has been removed.
"""
from .simulation_visualizer import SimulationVisualizer, visualize_simulation
from .mujoco_animator import (
    SimulationMuJoCoAnimator,
    animate_ohd_trajectory,
    animate_from_npz,
)
