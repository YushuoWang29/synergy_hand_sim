"""
Top-level simulation engine integrating all components.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import time

from .config import SimulationConfig
from .rigid_body import RigidBodySystem
from .dynamics import DynamicsAssembler
from .integrator import ODEState, DynamicsODE, Integrator
from .quasi_static import QuasiStaticSolver, QuasiStaticResult
from .transmission_force import compute_Q_matrix


@dataclass
class SimulationTrajectory:
    t: np.ndarray = None
    q: np.ndarray = None
    q_dot: np.ndarray = None
    q_ddot: np.ndarray = None
    inputs: np.ndarray = None
    energy_kinetic: np.ndarray = None
    energy_potential: np.ndarray = None
    energy_dissipated: np.ndarray = None
    residuals: np.ndarray = None
    info: dict = field(default_factory=dict)

    @property
    def n_steps(self):
        return len(self.t) if self.t is not None else 0

    @property
    def n_joints(self):
        return self.q.shape[1] if self.q is not None else 0

    def final_state(self):
        return ODEState(q=self.q[-1], q_dot=self.q_dot[-1], t=self.t[-1])

    def get_angle(self, joint_idx: int) -> np.ndarray:
        return self.q[:, joint_idx]

    def to_dict(self) -> dict:
        return dict(t=self.t, q=self.q, q_dot=self.q_dot, q_ddot=self.q_ddot,
                    inputs=self.inputs, energy_kinetic=self.energy_kinetic,
                    energy_potential=self.energy_potential,
                    energy_dissipated=self.energy_dissipated,
                    residuals=self.residuals, info=self.info)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class HandSimulator:
    def __init__(self, design, config=None, rigid_body_system=None,
                 dynamics_assembler=None):
        self.design = design
        self.config = config or SimulationConfig()

        self.rbs = rigid_body_system or RigidBodySystem.from_design(design)
        self.assembler = dynamics_assembler or DynamicsAssembler(
            n_joints=self.rbs.n_joints,
            joint_damping=self.rbs.joint_damping,
            joint_coulomb=self.rbs.joint_coulomb_friction,
        )
        self.Q_tauM, self.Q_s, self.Q_sdot = compute_Q_matrix(design)
        self._integrator = Integrator(dt=self.config.dt,
                                       method=self.config.method,
                                       verbose=self.config.verbose)

    def _compute_inputs(self, t, q, q_dot):
        """Construct 3-component input vector u = [tau_M * sigma, sigma, sigma_f].
        
        tau_M  : motor torque magnitude
        sigma  : synergy activation (0→1 ramp)
        sigma_f: differential activation
        
        Q columns: Q_tauM * (tau_M * sigma) + Q_s * sigma + Q_sdot * sigma_f
        """
        sigma = min(t * 2.0, 1.0) if self.config.sigma_func is None else \
            eval(self.config.sigma_func, {'t': t, 'q': q, 'np': np})
        sigma_f = 0.0 if self.config.sigma_f_func is None else \
            eval(self.config.sigma_f_func, {'t': t, 'q': q, 'np': np})
        tau_M = 5.0 if self.config.tau_M_func is None else \
            eval(self.config.tau_M_func, {'t': t, 'q': q, 'np': np})
        return np.array([tau_M * sigma, sigma, sigma_f])

    def _compute_gravity(self, q):
        g = self.config.gravity
        tau_g = np.zeros(self.rbs.n_joints)
        for fid, inertia in self.rbs.link_inertias.items():
            if inertia.mass <= 0:
                continue
            for j in range(min(self.rbs.n_joints, 3)):
                tau_g[j] += inertia.mass * g[2] * 0.01 * (j + 1) * np.cos(q[min(j, len(q)-1)])
        return tau_g

    def run_quasistatic(self, q0, u=None):
        if u is None:
            u = np.array([5.0, 0.0, 0.0])
        Q = np.column_stack([self.Q_tauM, self.Q_s, self.Q_sdot])

        def tau_t(q):
            return Q @ u

        solver = QuasiStaticSolver(
            n_joints=self.rbs.n_joints,
            compute_transmission_torque=tau_t,
            compute_gravity_torque=self._compute_gravity,
            compute_elastic_torque=self.assembler.compute_elastic_torque,
            tol=self.config.quasi_static_tol,
            max_iter=self.config.quasi_static_max_iter,
            verbose=self.config.verbose,
        )
        return solver.solve(q0)

    def run(self, q0=None, q_dot0=None, timeout=300.0):
        n_joints = self.rbs.n_joints
        q0 = q0 if q0 is not None else np.zeros(n_joints)
        q_dot0 = q_dot0 if q_dot0 is not None else np.zeros(n_joints)

        def inputs_fn(t, q, qd):
            return self._compute_inputs(t, q, qd)

        def qmat_fn(q):
            return self.Q_tauM, self.Q_s, self.Q_sdot

        ode = DynamicsODE(
            assembler=self.assembler,
            compute_inputs_fn=inputs_fn,
            compute_gravity_fn=self._compute_gravity,
        )
        ode.compute_q_matrix = qmat_fn

        state = ODEState(q=q0, q_dot=q_dot0, t=0.0)
        t_span = (0.0, self.config.t_end)

        start = time.time()
        ts, xs = self._integrator.integrate(ode, t_span, state.as_vector(),
                                             timeout=timeout)
        elapsed = time.time() - start

        record_every = max(1, self.config.record_every)
        ts_arr = np.array(ts[::record_every])
        xs_arr = np.array([x.copy() for x in xs[::record_every]])

        q_arr = xs_arr[:, :n_joints]
        qd_arr = xs_arr[:, n_joints:2*n_joints]

        kinetic = np.zeros(len(ts_arr))
        if qd_arr.shape[1] == self.assembler.inertia_matrix.shape[0]:
            kinetic = 0.5 * np.sum(qd_arr @ self.assembler.inertia_matrix * qd_arr, axis=1)

        return SimulationTrajectory(
            t=ts_arr, q=q_arr, q_dot=qd_arr,
            inputs=np.zeros((len(ts_arr), 3)),
            energy_kinetic=kinetic,
            energy_potential=np.zeros(len(ts_arr)),
            energy_dissipated=np.zeros(len(ts_arr)),
            info=dict(n_steps_total=len(ts), n_joints=n_joints,
                      dt=self.config.dt, method=self.config.method,
                      phase=self.config.phase, elapsed_s=elapsed))
