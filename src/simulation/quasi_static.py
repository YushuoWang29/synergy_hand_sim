# src/simulation/quasi_static.py
"""
Quasi-static force balance solver (Phase 1).

Solves the nonlinear algebraic system:
    f(q) = Q * u - g(q) - tau_elastic(q) - tau_friction(q) = 0

Using scipy.optimize.root (default method: hybr).

References
----------
Della Santina et al. (2018) TRO, Eq.43 with q_dot = q_ddot = 0.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class QuasiStaticResult:
    """
    Result of a quasi-static solve.

    Attributes
    ----------
    q : np.ndarray, shape (n_joints,)
        Equilibrium joint positions.
    converged : bool
        Whether the solver converged.
    residual : np.ndarray, shape (n_joints,)
        Final force residual.
    n_iterations : int
        Number of solver iterations.
    info : dict
        Additional solver info.
    """
    q: np.ndarray
    converged: bool = False
    residual: np.ndarray = None
    n_iterations: int = 0
    info: dict = None


class QuasiStaticSolver:
    """
    Quasi-static force balance solver.

    Solves: f(q) = Q * u - tau_g(q) - tau_elastic(q) - tau_friction(q, 0) = 0

    Parameters
    ----------
    n_joints : int
        Number of joints.
    compute_transmission_torque : Callable[[np.ndarray], np.ndarray]
        Function: q -> Q * u (transmission torque at given q).
    compute_gravity_torque : Callable[[np.ndarray], np.ndarray], optional
        Function: q -> tau_g(q).
    compute_elastic_torque : Callable[[np.ndarray], np.ndarray], optional
        Function: q -> tau_elastic(q) (e.g. joint stiffness).
    compute_friction_torque : Callable[[np.ndarray], np.ndarray], optional
        Function: q -> tau_friction(q, 0).
    tol : float
        Solver tolerance. Default 1e-8.
    max_iter : int
        Maximum solver iterations. Default 50.
    verbose : int
        Verbosity level. Default 0.
    """

    def __init__(
        self,
        n_joints: int,
        compute_transmission_torque: Callable[[np.ndarray], np.ndarray],
        compute_gravity_torque: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        compute_elastic_torque: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        compute_friction_torque: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        tol: float = 1e-8,
        max_iter: int = 50,
        verbose: int = 0,
    ):
        self.n_joints = n_joints
        self.compute_transmission_torque = compute_transmission_torque
        self.compute_gravity_torque = compute_gravity_torque or (lambda q: np.zeros(n_joints))
        self.compute_elastic_torque = compute_elastic_torque or (lambda q: np.zeros(n_joints))
        self.compute_friction_torque = compute_friction_torque or (lambda q: np.zeros(n_joints))
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def residual(self, q: np.ndarray) -> np.ndarray:
        """
        Compute force residual f(q) = Q*u - g(q) - tau_elastic(q) - tau_friction(q).

        Parameters
        ----------
        q : ndarray, shape (n_joints,)

        Returns
        -------
        f : ndarray, shape (n_joints,)
        """
        tau_t = self.compute_transmission_torque(q)
        tau_g = self.compute_gravity_torque(q)
        tau_e = self.compute_elastic_torque(q)
        tau_f = self.compute_friction_torque(q)
        return tau_t - tau_g - tau_e - tau_f

    def solve(self, q0: np.ndarray,
              method: str = 'hybr') -> QuasiStaticResult:
        """
        Solve for equilibrium joint positions.

        Parameters
        ----------
        q0 : ndarray, shape (n_joints,)
            Initial guess.
        method : str
            scipy.optimize.root method. Default 'hybr' (modified Powell).

        Returns
        -------
        QuasiStaticResult
        """
        from scipy.optimize import root

        if self.verbose > 0:
            print(f"[QuasiStaticSolver] Solving with {method} "
                  f"(tol={self.tol}, max_iter={self.max_iter})")

        sol = root(
            self.residual,
            q0,
            method=method,
            tol=self.tol,
            options={'maxfev': self.max_iter, 'xtol': self.tol},
        )

        # Extract solution
        q = sol.x
        residual_val = self.residual(q)
        converged = sol.success

        if self.verbose > 0:
            print(f"[QuasiStaticSolver] {'Converged' if converged else 'Not converged'}: "
                  f"|residual|={np.linalg.norm(residual_val):.2e}, "
                  f"iterations={sol.nfev}")

        return QuasiStaticResult(
            q=q,
            converged=converged,
            residual=residual_val,
            n_iterations=sol.nfev,
            info={'success': sol.success, 'status': sol.status,
                  'message': sol.message},
        )
