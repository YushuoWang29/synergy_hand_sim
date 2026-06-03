"""
Top-level simulation engine integrating all components.
"""

import numpy as np
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field
import time

from .config import SimulationConfig
from .rigid_body import RigidBodySystem
from .dynamics import DynamicsAssembler
from .integrator import ODEState, DynamicsODE, Integrator
from .quasi_static import QuasiStaticSolver, QuasiStaticResult
from .transmission_force import compute_Q_matrix
from .friction_models import HaywardArmstrongFriction, CapstanTensionDistribution



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

        # ============================================================
        # 修复 1: 将 design + rbs 传入 DynamicsAssembler，
        # 让它可以从设计读取真实的惯性矩阵和关节刚度。
        # ============================================================
        self.assembler = dynamics_assembler or DynamicsAssembler(
            n_joints=self.rbs.n_joints,
            joint_damping=self.rbs.joint_damping,
            joint_coulomb=self.rbs.joint_coulomb_friction,
            design=design,
            rigid_body_system=self.rbs,
        )

        self.Q_tauM, self.Q_s, self.Q_sdot = compute_Q_matrix(design)

        # ============================================================
        # 修复 4: 创建 Hayward-Armstrong 静摩擦模型实例
        # 统计所有腱绳路径上的 pulley/hole 元素总数
        # ============================================================
        self.friction_model = None
        self._n_z = 0
        if self.config.use_static_friction:
            total_elements = 0
            for tendon in design.tendons.values():
                for eid in tendon.pulley_sequence:
                    if (hasattr(design, 'pulleys') and eid in design.pulleys) or \
                       (hasattr(design, 'holes') and eid in design.holes):
                        total_elements += 1
            if total_elements > 0:
                self._n_z = total_elements
                self.friction_model = HaywardArmstrongFriction(
                    n_pulleys=total_elements,
                    delta_max=np.ones(total_elements) * self.config.delta_max_ratio,
                    kappa=np.ones(total_elements) * self.config.kappa_ratio,
                )
                print(f"  [HandSimulator] Static friction active: "
                      f"{total_elements} elements, delta_max={self.config.delta_max_ratio}, "
                      f"kappa={self.config.kappa_ratio}")

        self._integrator = Integrator(dt=self.config.dt,
                                       method=self.config.method,
                                       verbose=self.config.verbose)

    def _get_pulley_angles(self, q: np.ndarray) -> np.ndarray:
        """
        Map joint angles q to pulley angles theta for friction state update.

        Uses the R_bar matrix structure: each pulley/hole element's angle
        is its radius * joint angle for the joint it's attached to.

        Returns flatten array of pulley angles matching friction_model ordering.
        """
        from src.models.transmission_builder import get_joint_list
        from src.models.origami_design import is_pulley_id, is_hole_id

        joints, jid_to_idx = get_joint_list(self.design)
        theta_list = []
        for tendon in self.design.tendons.values():
            for eid in tendon.pulley_sequence:
                if is_pulley_id(eid) and eid in self.design.pulleys:
                    p = self.design.pulleys[eid]
                    if p.attached_fold_line_id is not None:
                        j_idx = jid_to_idx.get(p.attached_fold_line_id)
                        if j_idx is not None and j_idx < len(q):
                            theta_list.append(q[j_idx] * p.radius)
                        else:
                            theta_list.append(0.0)
                    else:
                        theta_list.append(0.0)
                elif is_hole_id(eid) and eid in self.design.holes:
                    h = self.design.holes[eid]
                    if h.attached_fold_line_id is not None:
                        j_idx = jid_to_idx.get(h.attached_fold_line_id)
                        if j_idx is not None and j_idx < len(q):
                            theta_list.append(q[j_idx] * h.plate_offset)
                        else:
                            theta_list.append(0.0)
                    else:
                        theta_list.append(0.0)
        return np.array(theta_list)


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
            n_z=self._n_z,
        )
        ode.compute_q_matrix = qmat_fn


        # ============================================================
        # 修复 4: 将静摩擦状态 z 加入 ODE 状态向量
        # ============================================================
        z0 = (
            np.zeros(self._n_z)
            if self.friction_model is not None and self._n_z > 0
            else np.zeros(0)
        )
        state = ODEState(q=q0, q_dot=q_dot0, z=z0 if len(z0) > 0 else None, t=0.0)

        t_span = (0.0, self.config.t_end)

        # 后步回调：在每个积分步后更新摩擦模型 z 状态
        if self.friction_model is not None and self._n_z > 0:
            def post_step_callback(t, x):
                q_curr = x[:n_joints]
                theta = self._get_pulley_angles(q_curr)
                if len(theta) == self._n_z:
                    self.friction_model.update(theta, self.config.dt)
                # 回写更新后的 z 到状态向量
                # 状态向量 = [q (n_joints), q_dot (n_joints), z (n_z)]
                z_start = 2 * n_joints
                x[z_start:z_start + self._n_z] = self.friction_model.z
            self._integrator.post_step_callback = post_step_callback


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


