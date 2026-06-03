# src/simulation/config.py
"""
Simulation configuration dataclass.

Contains all physical parameters and control settings for the three-phase
mechanics simulation framework.

See Also
--------
docs/numerical_simulation.md : Detailed documentation of all parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np


@dataclass
class SimulationConfig:
    """
    Complete simulator configuration with all physical parameters.

    Parameters
    ----------
    dt : float
        Integration time step (s). Default 1e-4.
    t_end : float
        Total simulation duration (s). Default 5.0.
    method : str
        Integration method: 'RK4', 'Euler', 'scipy_ode'. Default 'RK4'.

    # Control inputs
    sigma_func : Optional[str]
        Expression for sigma(t), e.g. "min(t*2, 1.0)".
    sigma_f_func : Optional[str]
        Expression for sigma_f(t).
    tau_M_func : Optional[str]
        Expression for tau_M(t) - motor pulling force.
    sliding_s_func : Optional[str]
        Expression for s(t) - tendon sliding displacement.

    # Capstan friction (tendon-pulley interface)
    beta_capstan : float
        Capstan decay coefficient per element. Default 0.09.
    beta_hole : float
        Friction decay for hole-type tendons. Default 0.09.

    # Static friction (Hayward-Armstrong)
    use_static_friction : bool
        Enable static friction memory effect (Eq.37). Default True.
    delta_max_ratio : float
        Delta_max / r (static friction range ratio). Default 1e-4.
    kappa_ratio : float
        Static friction stiffness kappa = kappa_ratio * k_joint. Default 0.3.

    # Joint viscous friction
    joint_viscous_damping : float
        Diagonal joint damping (N*m*s/rad). Default 0.01.
    tendon_viscous_damping : float
        Tendon viscous damping factor. Default 0.001.

    # Elastic parameters
    use_nonlinear_spring : bool
        Use nonlinear spring model (Appendix B). Default False.

    # Numerical control
    quasi_static_tol : float
        Quasi-static solver convergence tolerance. Default 1e-8.
    quasi_static_max_iter : int
        Maximum iterations for quasi-static solver. Default 50.
    verbose : int
        0 = silent, 1 = summary, 2 = detailed. Default 1.

    # Recording
    record_every : int
        Record every N steps. Default 10.
    output_path : Optional[str]
        Result save path (.npz). Default None.

    # Gravity
    gravity : np.ndarray
        Gravity vector (m/s^2). Default [0, 0, -9.81].

    # External forces
    external_forces : dict
        {body_name: (force_vector, application_point)}.

    # Phase control
    phase : int
        Simulation phase (1=quasistatic, 2=dynamic, 3=dynamic).
        Default 2.
    """

    # ---- Time integration ----
    dt: float = 1e-4
    t_end: float = 5.0
    method: str = 'RK4'

    # ---- Control inputs ----
    sigma_func: Optional[str] = None
    sigma_f_func: Optional[str] = None
    tau_M_func: Optional[str] = None
    sliding_s_func: Optional[str] = None

    # ---- Capstan friction parameters ----
    beta_capstan: float = 0.09
    beta_hole: float = 0.09

    # ---- Static friction (Hayward-Armstrong) ----
    use_static_friction: bool = True
    delta_max_ratio: float = 1e-4
    kappa_ratio: float = 0.3

    # ---- Joint viscous friction ----
    joint_viscous_damping: float = 0.01
    tendon_viscous_damping: float = 0.001

    # ---- Elastic parameters ----
    use_nonlinear_spring: bool = False

    # ---- Numerical control ----
    quasi_static_tol: float = 1e-8
    quasi_static_max_iter: int = 50
    verbose: int = 1

    # ---- Result output ----
    record_every: int = 10
    output_path: Optional[str] = None

    # ---- Gravity (默认为零，不开启重力) ----
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))


    # ---- External forces ----
    external_forces: dict = field(default_factory=dict)

    # ---- Phase control ----
    phase: int = 2

    @property
    def is_quasistatic(self) -> bool:
        """Return True if Phase 1 (quasi-static)."""
        return self.phase == 1

    @property
    def is_dynamic(self) -> bool:
        """Return True if Phase 2 or 3 (dynamic)."""
        return self.phase >= 2
