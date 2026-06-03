# src/simulation/integrator.py
"""
Numerical ODE integrators for rigid-body dynamics.

Provides:
    - ODEState: state vector container (q, q_dot, z, t)
    - DynamicsODE: ODE right-hand side construction
    - Integrator: RK4, Euler, scipy RK45 wrappers with timeout
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Any
import time
from copy import deepcopy



@dataclass
class ODEState:
    """Full state of the dynamical system."""
    q: np.ndarray       # Joint positions (n_joints,)
    q_dot: np.ndarray   # Joint velocities (n_joints,)
    z: np.ndarray = None  # Static friction virtual angles (n_pulleys,)
    t: float = 0.0

    def as_vector(self) -> np.ndarray:
        """Flatten state to vector for integrator."""
        parts = [self.q, self.q_dot]
        if self.z is not None:
            parts.append(self.z)
        return np.concatenate(parts)

    @classmethod
    def from_vector(cls, x: np.ndarray, n_joints: int, n_z: int = 0) -> 'ODEState':
        """Reconstruct from flattened vector."""
        q = x[:n_joints]
        q_dot = x[n_joints:2 * n_joints]
        z = x[2 * n_joints:2 * n_joints + n_z] if n_z > 0 else None
        return cls(q=q, q_dot=q_dot, z=z)

    def copy(self) -> 'ODEState':
        """Create a deep copy."""
        return ODEState(
            q=self.q.copy(),
            q_dot=self.q_dot.copy(),
            z=self.z.copy() if self.z is not None else None,
            t=self.t
        )


class DynamicsODE:
    """
    ODE right-hand side (RHS) function for rigid-body dynamics.

    State vector x = [q; q_dot; z] (z is optional static friction state).

    RHS:
        dq/dt = q_dot
        d(q_dot)/dt = B(q)^{-1} (Q(q) u - W q_dot - Gamma)
        dz/dt = 0  (virtual angles update at event rate, not continuous)
    """

    def __init__(self, assembler, compute_inputs_fn: Callable,
                 compute_gravity_fn: Callable = None,
                 z_update_fn: Callable = None,
                 n_z: int = 0):
        self.assembler = assembler
        self.compute_inputs_fn = compute_inputs_fn
        self.compute_gravity_fn = compute_gravity_fn
        self.z_update_fn = z_update_fn
        self.n_z = n_z  # static friction state dimension

    def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
        n_joints = self.assembler.n_joints
        state = ODEState.from_vector(x, n_joints, n_z=self.n_z)
        state.t = t

        q, q_dot = state.q, state.q_dot


        # Compute control inputs
        u = self.compute_inputs_fn(t, q, q_dot)

        # Compute Q matrix (three-column)
        Q_tauM, Q_s, Q_sdot = self.compute_q_matrix(q)
        Q_matrix = np.column_stack([Q_tauM, Q_s, Q_sdot])

        # Gravity
        tau_gravity = None
        if self.compute_gravity_fn is not None:
            tau_gravity = self.compute_gravity_fn(q)

        # Acceleration
        q_ddot = self.assembler.compute_acceleration(
            q, q_dot, u, Q_matrix,
            tau_g=tau_gravity,
        )

        # Assemble derivative
        dx_part = [q_dot, q_ddot]
        if state.z is not None:
            dx_part.append(np.zeros_like(state.z))

        return np.concatenate(dx_part)

    def compute_q_matrix(self, q: np.ndarray):
        """Compute Q matrix columns (can be overridden for q-dependence)."""
        raise NotImplementedError(
            "Subclass must implement compute_q_matrix or set it externally."
        )


class Integrator:
    """
    Numerical ODE integrator with timeout.

    Methods:
        - 'RK4': Classic 4th-order Runge-Kutta (fixed step)
        - 'Euler': Forward Euler (fixed step)
        - 'scipy_ode': Adaptive step using scipy.integrate.solve_ivp (RK45)
    """

    def __init__(self, dt: float = 1e-4, method: str = 'RK4',
                 verbose: int = 1):
        self.dt = dt
        self.method = method
        self.verbose = verbose
        self._timeout = 300.0  # max wall-clock seconds
        self._start_time = None
        self.post_step_callback = None  # Callable(t, x)
        self._internal_copy = True  # always copy x before passing to callback


    def integrate(self, rhs: Callable, t_span: tuple,
                  x0: np.ndarray, dt_fixed: float = None,
                  timeout: float = None) -> tuple:
        """
        Integrate ODE from t_span[0] to t_span[1].

        Parameters
        ----------
        rhs : Callable(t, x) -> dx/dt
        t_span : (t0, t1)
        x0 : initial state vector
        dt_fixed : fixed step size (overrides self.dt)
        timeout : max wall-clock seconds

        Returns
        -------
        ts : list of timestamps
        xs : list of state vectors
        """
        dt = dt_fixed if dt_fixed is not None else self.dt
        timeout = timeout or self._timeout
        self._start_time = time.time()

        if self.method == 'scipy_ode':
            return self._integrate_scipy(rhs, t_span, x0, timeout=timeout)

        # Fixed-step methods
        t0, t1 = t_span
        n_steps = int((t1 - t0) / dt) + 1

        ts = [t0]
        xs = [x0.copy()]

        t = t0
        x = x0.copy()

        for step in range(n_steps):
            # Timeout check
            if time.time() - self._start_time > timeout:
                if self.verbose > 0:
                    print(f"[Integrator] Timeout at t={t:.4f}, step={step}")
                break

            if self.method == 'RK4':
                x = self._rk4_step(rhs, t, x, dt)
            elif self.method == 'Euler':
                x = self._euler_step(rhs, t, x, dt)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            t += dt

            # ============================================================
            # 调用后步回调（用于更新静摩擦状态 z、记录等）
            # ============================================================
            if self.post_step_callback is not None:
                self.post_step_callback(t, x)

            if t > t1 - 0.5 * dt:
                break

            ts.append(t)
            xs.append(x.copy())


        return ts, xs

    def _rk4_step(self, rhs, t, x, dt):
        """Classic RK4 step."""
        k1 = rhs(t, x)
        k2 = rhs(t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = rhs(t + dt, x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _euler_step(self, rhs, t, x, dt):
        """Forward Euler step."""
        return x + dt * rhs(t, x)

    def _integrate_scipy(self, rhs, t_span, x0, timeout):
        """Adaptive integration via scipy."""
        from scipy.integrate import solve_ivp

        result = solve_ivp(
            rhs, t_span, x0,
            method='RK45',
            max_step=self.dt * 10,
            rtol=1e-6,
            atol=1e-8,
            events=None,
        )
        return result.t.tolist(), result.y.T.tolist()
