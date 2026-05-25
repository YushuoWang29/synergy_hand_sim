# src/simulation/mujoco_physics.py
"""
MuJoCo 原生物理引擎封装 (简化版)
==================================

注意：MuJoCo 物理引擎处理 thin-shell 折纸模型时因惯性极小
      (STL 网格 → ~1e-6 kg·m²) 而出现数值不稳定。
      
当前策略(cf. run_numerical_simulation.py):
  - Phase 3 动力学: 使用稳定的自定义求解器 HandSimulator
  - 可视化: 使用 SimulationMuJoCoAnimator 播放预计算轨迹
  - 详见 scripts/run_numerical_simulation.py
"""

import os, sys, time, tempfile, shutil
from typing import Optional, Dict, List, Tuple
import numpy as np

from .config import SimulationConfig
from .transmission_force import compute_Q_matrix
from .rigid_body import RigidBodySystem


class MuJoCoPhysicsEngine:
    """
    基于 MuJoCo 的物理仿真引擎（简化存根）。
    
    保留原有接口以确保兼容性，但此引擎已知对于折纸 thin-shell
    模型存在数值稳定性问题。建议使用 HandSimulator (custom solver)
    进行物理计算，配合 SimulationMuJoCoAnimator 进行可视化。
    """

    def __init__(self, design, config: SimulationConfig,
                 urdf_path: Optional[str] = None,
                 joint_stiffness: float = 0.05,
                 joint_damping_ratio: float = 1.0):
        self.design = design
        self.config = config
        Q_tauM, Q_s, Q_sdot = compute_Q_matrix(self.design)
        self.Q_tauM = Q_tauM
        self.Q_s = Q_s
        self.Q_sdot = Q_sdot
        self.n_joints = len(Q_tauM)
        self.n_tendons = len(design.tendons)
        self._joint_stiffness = joint_stiffness
        self._joint_damping_ratio = joint_damping_ratio
        self._urdf_path = urdf_path
        self._temp_dir = None
        self._setup_urdf()
        self._mjcf_path = None
        self.model = None
        self.data = None
        self._joint_qposadr: List[int] = []

    def _setup_urdf(self):
        if self._urdf_path is not None and os.path.exists(self._urdf_path):
            return
        from src.models.origami_design import OrigamiHandDesign
        from src.models.origami_to_urdf import export_urdf
        ohd_path = self.design.source_path if hasattr(self.design, 'source_path') else None
        ohd_name = "hand"
        if ohd_path:
            ohd_name = os.path.splitext(os.path.basename(ohd_path))[0]
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        standard = os.path.join(project_root, "models", ohd_name, f"{ohd_name}.urdf")
        if os.path.exists(standard):
            self._urdf_path = standard
            return
        self._temp_dir = tempfile.mkdtemp(prefix=f"mujoco_physics_{ohd_name}_")
        self._urdf_path = os.path.join(self._temp_dir, f"{ohd_name}.urdf")
        print(f"  [MuJoCoPhysics] 正在导出 URDF: {self._urdf_path}")
        if not self.design.faces:
            self.design.build_topology()
        if self.design.root_face_id is None and self.design.faces:
            self.design.set_root_face(list(self.design.faces.keys())[0])
            self.design.build_face_tree()
        export_urdf(self.design, self._urdf_path, thickness=3.0)
        print(f"  [MuJoCoPhysics] URDF 导出完成")

    def _compute_u(self, t, q, q_dot):
        sigma = eval(self.config.sigma_func, {'t': t, 'q': q, 'np': np}) if self.config.sigma_func else min(t * 2.0, 1.0)
        sigma_f = eval(self.config.sigma_f_func, {'t': t, 'q': q, 'np': np}) if self.config.sigma_f_func else 0.0
        tau_M = eval(self.config.tau_M_func, {'t': t, 'q': q, 'np': np}) if self.config.tau_M_func else 5.0
        return np.array([tau_M * sigma, sigma, sigma_f])

    def _compute_joint_torques(self, u, q=None, q_dot=None):
        mm_to_m = 0.001
        tau_drive = (self.Q_tauM * u[0] + self.Q_s * u[1] + self.Q_sdot * u[2]) * mm_to_m
        if q is not None and self._joint_stiffness > 0:
            tau_drive += -self._joint_stiffness * q
        if q_dot is not None and self._joint_damping_ratio > 0:
            tau_drive += -self._joint_damping_ratio * 0.01 * q_dot
        return tau_drive

    def _build_mjcf(self):
        import xml.etree.ElementTree as ET
        tree = ET.parse(self._urdf_path)
        root = tree.getroot()
        robot_name = root.get('name', 'hand')
        mesh_dir = os.path.join(os.path.dirname(self._urdf_path), "meshes")
        mesh_abs = mesh_dir.replace('\\', '/')
        links = {}
        for le in root.findall('link'): links[le.get('name')] = le
        joints = []
        for je in root.findall('joint'):
            joints.append({
                'name': je.get('name'), 'type': 'hinge',
                'parent': je.find('parent').get('link'),
                'child': je.find('child').get('link'),
                'origin': je.find('origin'), 'axis': je.find('axis'),
                'limit': je.find('limit'),
            })
        all_child_names = {j['child'] for j in joints}
        root_body_name = None
        for j in joints:
            if j['parent'] not in all_child_names: root_body_name = j['parent']; break
        if root_body_name is None: root_body_name = joints[0]['parent'] if joints else list(links.keys())[0]
        children_map = {}
        for j in joints: children_map.setdefault(j['parent'], []).append((j['child'], j))
        def _mesh_name(name):
            le = links.get(name)
            if le is None: return None
            v = le.find('visual')
            if v is None: return None
            g = v.find('geometry')
            if g is None: return None
            return g.find('mesh')
        def _joint_axis_xyz(ji):
            return ji['axis'].get('xyz', '0 0 1') if ji['axis'] is not None else '0 0 1'
        def _joint_range(ji):
            limit = ji['limit']
            return f'{limit.get("lower", "-1.57")} {limit.get("upper", "1.57")}' if limit is not None else '-1.57 1.57'
        def _joint_origin_xyz(ji):
            if ji['origin'] is not None:
                s = ji['origin'].get('xyz', '0 0 0')
                parts = s.split()
                return ' '.join(f'{float(v)/1000.0:.9f}' for v in parts)
            return '0 0 0'
        joint_names = []
        lines = [
            '<?xml version="1.0"?>',
            f'<mujoco model="{robot_name}_physics">',
            '  <asset>',
            '    <texture name="checker_tex" type="2d" builtin="checker" rgb1="0.25 0.26 0.28" rgb2="0.35 0.36 0.38" width="512" height="512"/>',
            '    <material name="ground_mat" texture="checker_tex" texrepeat="6 6" reflectance="0.15"/>',
            '    <material name="origami_plastic" rgba="0.65 0.82 0.95 1.0" specular="0.4" shininess="0.3"/>',
            '    <material name="origami_accent" rgba="0.75 0.85 0.98 1.0" specular="0.5" shininess="0.4"/>',
        ]
        for name in links:
            me = _mesh_name(name)
            if me is not None:
                fn = me.get('filename', '')
                if fn.startswith('meshes/'): fn = mesh_abs + '/' + fn[7:]
                elif not os.path.isabs(fn): fn = os.path.join(os.path.dirname(self._urdf_path), fn).replace('\\', '/')
                lines.append(f'    <mesh name="mesh_{name}" file="{fn}" scale="0.001 0.001 0.001"/>')
        lines += [
            '  </asset>',
            '  <compiler angle="radian" meshdir=""/>',
            '  <option gravity="0 0 0" cone="elliptic" impratio="1"/>',
            '  <size njmax="1000" nconmax="200"/>',
            '  <visual>',
            '    <quality offsamples="8" shadowsize="8192"/>',
            '    <headlight ambient="0.15 0.15 0.18" diffuse="0.5 0.5 0.55" specular="0.3 0.3 0.3"/>',
            '    <map fogstart="5.0" fogend="15.0" haze="0.2"/>',
            '  </visual>',
            '  <worldbody>',
            '    <light name="key_light" directional="true" pos="2 1 3" dir="-2 -1 -3" diffuse="0.65 0.6 0.55" ambient="0.0 0.0 0.0" specular="0.5 0.5 0.5" castshadow="true"/>',
            '    <light name="fill_light" directional="true" pos="-2 -1 2" dir="2 1 -2" diffuse="0.35 0.4 0.5" ambient="0.0 0.0 0.0" specular="0.2 0.2 0.4" castshadow="true"/>',
            '    <light name="back_light" directional="true" pos="0 -3 2" dir="0 3 -2" diffuse="0.25 0.25 0.3" ambient="0.0 0.0 0.0" specular="0.6 0.6 0.7" castshadow="true"/>',
        ]
        _material_toggle = [0]
        _hand_z_offset = 0.1
        _body_depth: Dict[str, int] = {}
        def _assign_depth(name, depth=0):
            if name not in _body_depth:
                _body_depth[name] = depth
                if name in children_map:
                    for cn, _ in children_map[name]: _assign_depth(cn, depth + 1)
        _assign_depth(root_body_name, 0)
        max_depth = max(_body_depth.values()) if _body_depth else 0
        all_bits_mask = (1 << (max_depth + 1)) - 1
        ground_bit = 1 << (max_depth + 1)
        lines.append(f'    <geom type="plane" name="ground" size="1 1 0.02" pos="0 0 0" material="ground_mat" contype="{ground_bit}" conaffinity="{ground_bit}"/>')
        def add_body(name, joint_info=None, indent=4):
            nonlocal joint_names
            pf = ' ' * indent
            pos = _joint_origin_xyz(joint_info) if joint_info is not None else f'0 0 {_hand_z_offset}'
            lines.append(f'{pf}<body name="{name}" pos="{pos}">')
            if joint_info is not None:
                jn = joint_info["name"]
                joint_names.append(jn)
                lines.append(f'{pf}  <joint name="{jn}" type="hinge" pos="0 0 0" axis="{_joint_axis_xyz(joint_info)}" range="{_joint_range(joint_info)}" damping="0.1" stiffness="0.0"/>')
            me = _mesh_name(name)
            if me is not None:
                _material_toggle[0] += 1
                mat = 'origami_accent' if (_material_toggle[0] % 2 == 0) else 'origami_plastic'
                depth = _body_depth.get(name, 0)
                self_bit = 1 << depth
                exclude_bits = self_bit
                if depth - 1 >= 0: exclude_bits |= (1 << (depth - 1))
                exclude_bits |= (1 << (depth + 1))
                conaffinity = all_bits_mask & ~exclude_bits
                if conaffinity == 0: conaffinity = 1
                lines.append(f'{pf}  <geom type="mesh" mesh="mesh_{name}" pos="0 0 0" material="{mat}" contype="{self_bit}" conaffinity="{conaffinity}" friction="0.8 0.005 0.0001" margin="0.015" solref="0.0003 2" solimp="0.999 0.999 0.0001"/>')
            if name in children_map:
                for cn, ji in children_map[name]: add_body(cn, ji, indent + 2)
            lines.append(f'{pf}</body>')
        add_body(root_body_name, None, indent=4)
        lines += ['  </worldbody>', '  <actuator>']
        for jn in joint_names:
            lines.append(f'    <motor name="motor_{jn}" joint="{jn}" ctrlrange="-50 50" ctrllimited="true"/>')
        lines += ['  </actuator>', '</mujoco>']
        self._joint_names = joint_names
        return '\n'.join(lines)

    def _load_model(self):
        import mujoco
        mjcf_xml = self._build_mjcf()
        tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='w', encoding='utf-8')
        tmp.write(mjcf_xml); tmp.close()
        self._mjcf_path = tmp.name
        self.model = mujoco.MjModel.from_xml_path(self._mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity = (0.0, 0.0, 0.0)
        self.model.opt.tolerance = 1e-8; self.model.opt.iterations = 200
        self.model.opt.timestep = self.config.dt
        self.model.vis.quality.offsamples = 8; self.model.vis.quality.shadowsize = 8192
        E = self.model.stat.extent; C = self.model.stat.center
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ground')
        if gid >= 0:
            self.model.geom_pos[gid] = (C[0], C[1], 0.0)
            self.model.geom_size[gid] = (E * 8.0, E * 8.0, 0.02)
        self._joint_qposadr = []
        for jname in self._joint_names:
            for i in range(self.model.njnt):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if name == jname: self._joint_qposadr.append(self.model.jnt_qposadr[i]); break
        self._motor_actuator_indices = []
        for jn in self._joint_names:
            act_name = f"motor_{jn}"
            found = False
            for i in range(self.model.nu):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if name == act_name: self._motor_actuator_indices.append(i); found = True; break
            if not found: self._motor_actuator_indices.append(-1)
        print(f"  [MuJoCoPhysics] 模型: nbody={self.model.nbody}, njnt={self.model.njnt}, nu={self.model.nu}")

    def step(self, u, q_current=None, qd_current=None):
        import mujoco
        tau_joint = self._compute_joint_torques(u, q_current, qd_current)
        for i, act_idx in enumerate(self._motor_actuator_indices):
            if act_idx >= 0 and i < len(tau_joint): self.data.ctrl[act_idx] = float(tau_joint[i])
        mujoco.mj_step(self.model, self.data)
        return (self.data.qpos[self._joint_qposadr].copy(),
                self.data.qvel[self._joint_qposadr].copy(),
                self.data.ncon)

    def run(self, q0=None, q_dot0=None, timeout=60.0):
        import mujoco
        from .simulator import SimulationTrajectory
        if self.model is None: self._load_model()
        q0 = np.zeros(self.n_joints) if q0 is None else np.asarray(q0)
        q_dot0 = np.zeros(self.n_joints) if q_dot0 is None else np.asarray(q_dot0)
        for i, adr in enumerate(self._joint_qposadr):
            if i < len(q0): self.data.qpos[adr] = float(q0[i])
        for i, adr in enumerate(self._joint_qposadr):
            if i < len(q_dot0): self.data.qvel[adr] = float(q_dot0[i])
        mujoco.mj_forward(self.model, self.data)
        n_steps = int(self.config.t_end / self.config.dt) + 1
        record_every = max(1, self.config.record_every)
        ts, qs, qds, us, ncons = [0.0], [self.data.qpos[self._joint_qposadr].copy()], [self.data.qvel[self._joint_qposadr].copy()], [np.zeros(3)], [self.data.ncon]
        start_time = time.time(); sim_t = 0.0; step = 0
        print(f"\n  开始仿真: {n_steps} 步, dt={self.config.dt:.2e}s")
        while sim_t < self.config.t_end:
            if time.time() - start_time > timeout: print(f"\n  超时 ({timeout:.0f}s)"); break
            q_cur = self.data.qpos[self._joint_qposadr].copy()
            qd_cur = self.data.qvel[self._joint_qposadr].copy()
            u = self._compute_u(sim_t, q_cur, qd_cur)
            q_new, qd_new, ncon = self.step(u, q_cur, qd_cur)
            sim_t += self.config.dt; step += 1
            if step % record_every == 0:
                ts.append(sim_t); qs.append(q_new.copy()); qds.append(qd_new.copy()); us.append(u.copy()); ncons.append(ncon)
        elapsed = time.time() - start_time
        print(f"  仿真完成: t={sim_t*1000:.1f}ms, steps={step}")
        return SimulationTrajectory(
            t=np.array(ts), q=np.array(qs), q_dot=np.array(qds),
            inputs=np.array(us) if len(us) > 0 else None,
            energy_kinetic=np.zeros(len(ts)), energy_potential=np.zeros(len(ts)),
            energy_dissipated=np.zeros(len(ts)),
            info=dict(phase=self.config.phase, method='MuJoCo', dt=self.config.dt, n_steps_total=step, elapsed_s=elapsed))

    def close(self):
        if hasattr(self, '_mjcf_path') and self._mjcf_path and os.path.exists(self._mjcf_path):
            try: os.unlink(self._mjcf_path)
            except OSError: pass
        if self._temp_dir and os.path.exists(self._temp_dir):
            try: shutil.rmtree(self._temp_dir)
            except OSError: pass

    def run_with_viewer(self, q0=None, q_dot0=None, timeout=60.0, show_left_ui=True, show_right_ui=True, playback_speed=0.2, print_contacts=True):
        """MuJoCo 实时物理仿真（注意：使用前请确认数值稳定性）。"""
        import mujoco
        from mujoco import viewer
        if self.model is None: self._load_model()
        q0 = np.zeros(self.n_joints) if q0 is None else np.asarray(q0)
        q_dot0 = np.zeros(self.n_joints) if q_dot0 is None else np.asarray(q_dot0)
        for i, adr in enumerate(self._joint_qposadr):
            if i < len(q0): self.data.qpos[adr] = float(q0[i])
        for i, adr in enumerate(self._joint_qposadr):
            if i < len(q_dot0): self.data.qvel[adr] = float(q_dot0[i])
        mujoco.mj_forward(self.model, self.data)
        v = viewer.launch_passive(self.model, self.data, show_left_ui=show_left_ui, show_right_ui=show_right_ui)
        center = self.model.stat.center; extent = self.model.stat.extent
        v.cam.lookat[:] = center; v.cam.distance = 2.5 * extent; v.cam.elevation = -30; v.cam.azimuth = 45
        print(f"\n  [MuJoCoPhysics] 实时查看器 (t_end={self.config.t_end:.3f}s, playback={playback_speed:.1f}x)")
        sim_t = 0.0; step = 0; start_time = time.time()
        try:
            while v.is_running():
                if time.time() - start_time > timeout: break
                if sim_t < self.config.t_end:
                    q_cur = self.data.qpos[self._joint_qposadr].copy()
                    qd_cur = self.data.qvel[self._joint_qposadr].copy()
                    ut = self._compute_u(sim_t, q_cur, qd_cur)
                    tau_joint = self._compute_joint_torques(ut, q_cur, qd_cur)
                    for i, act_idx in enumerate(self._motor_actuator_indices):
                        if act_idx >= 0 and i < len(tau_joint): self.data.ctrl[act_idx] = float(tau_joint[i])
                    mujoco.mj_step(self.model, self.data); sim_t += self.config.dt; step += 1; v.sync()
                    if playback_speed > 0:
                        sleep_needed = sim_t / playback_speed - (time.time() - start_time)
                        if sleep_needed > 0.0005: time.sleep(sleep_needed)
                else:
                    print(f"\n  仿真完成. 1.5s 后关闭...")
                    hold_end = time.time() + 1.5
                    while time.time() < hold_end and v.is_running(): v.sync(); time.sleep(0.05)
                    break
        except KeyboardInterrupt: pass
        finally: v.close()
        print(f"  实时仿真: {step} 步")

    def __del__(self):
        self.close()
