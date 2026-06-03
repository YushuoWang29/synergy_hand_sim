# src/simulation/__init__.py
"""
Numerical Simulation Framework for Synergy-Based Tendon-Driven Hands.

Implements the three-phase mechanics simulation architecture:

    Phase 1: Quasi-static force balance (nonlinear algebraic solver)
    Phase 2: Full dynamics (ODE integration, RK4/scipy)
    Phase 3: Dynamics

Based on Della Santina et al. (2018) TRO, Eq.43-44.

Modules
-------
config            Simulation configuration dataclass.
rigid_body        Rigid body inertia parameters and estimation.
transmission_force  Tendon transmission force matrix Q(q) computation.
friction_models   Hayward-Armstrong static friction and viscous friction.
dynamics          Dynamics equation assembly.
integrator        Numerical ODE integrators (RK4, Euler, scipy wrappers).
quasi_static      Quasi-static force balance solver (Phase 1).
simulator         Top-level simulation engine integrating all components.
io                Simulation result serialization and logging.
"""

from .config import SimulationConfig
from .rigid_body import LinkInertia, RigidBodySystem, estimate_link_inertia
from .transmission_force import (
    build_M_matrix, build_R_bar_matrix,
    build_viscous_damping_matrix, build_static_friction_matrix,
    build_N_matrix, compute_Q_matrix
)
from .friction_models import HaywardArmstrongFriction, CapstanTensionDistribution
from .dynamics import DynamicsAssembler
from .integrator import ODEState, DynamicsODE, Integrator
from .quasi_static import QuasiStaticSolver, QuasiStaticResult
from .simulator import HandSimulator, SimulationTrajectory
from .io import SimulationWriter, SimulationReader
from .mujoco_sdas import (
    MuJoCoSDASConfig, MuJoCoSDASSimulator, SimulationRunResult,
    load_ohd_simulation, run_ohd_mujoco_simulation,
)

__all__ = [
    "SimulationConfig",
    "LinkInertia", "RigidBodySystem", "estimate_link_inertia",
    "build_M_matrix", "build_R_bar_matrix",
    "build_viscous_damping_matrix", "build_static_friction_matrix",
    "build_N_matrix", "compute_Q_matrix",
    "HaywardArmstrongFriction", "CapstanTensionDistribution",
    "DynamicsAssembler",
    "ODEState", "DynamicsODE", "Integrator",
    "QuasiStaticSolver", "QuasiStaticResult",
    "HandSimulator", "SimulationTrajectory",
    "SimulationWriter", "SimulationReader",
    "MuJoCoSDASConfig", "MuJoCoSDASSimulator", "SimulationRunResult",
    "load_ohd_simulation", "run_ohd_mujoco_simulation",
]
