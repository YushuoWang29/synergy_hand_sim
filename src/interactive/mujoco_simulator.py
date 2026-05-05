# src/interactive/mujoco_simulator.py
"""
基于 MuJoCo 的交互式 URDF 仿真器

增强功能：四滑块协同模式（Motor A, B, σ, σ_f）
互联动：拖动 A/B 自动更新 σ/σ_f，拖动 σ/σ_f 自动更新 A/B

MuJoCo 查看器右侧面板有两类滑块：
  - "Joint" 滑块（灰色/只读）：仅显示 data.qpos，不可编辑
  - "Position" 滑块（可交互）：对应 MJCF 中 <position actuator> 的 data.ctrl
"""

import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import mujoco
from mujoco import viewer
from typing import Optional, Callable


class MuJoCoSimulator:
    """基于 MuJoCo 的折纸手交互式仿真器"""

    def __init__(self, urdf_path: str, mesh_dir: str, show_left_ui: bool = True,
                 show_right_ui: bool = True,
                 synergy_callback: Optional[Callable] = None,
                 synergy_motor_names: Optional[list] = None,
                 ctrl_range_deg: float = 720.0,
                 synergy_with_sigma_sliders: bool = False):
        """
        Args:
            urdf_path: URDF 文件路径
            mesh_dir: STL mesh 文件目录（绝对路径）
            show_left_ui: 是否显示查看器左侧 UI
            show_right_ui: 是否显示查看器右侧 UI
            synergy_callback: 协同回调函数 ctrl_radians -> qpos_radians dict
            synergy_motor_names: 电机滑块名称列表
            ctrl_range_deg: 滑块控制范围（度），默认 ±720°
            synergy_with_sigma_sliders: 是否启用 σ/σ_f 额外滑块
        """
        self.urdf_path = os.path.abspath(urdf_path)
        self.mesh_dir = os.path.abspath(mesh_dir)
        self._tmp_xml = None
        self.synergy_callback = synergy_callback
        self.synergy_motor_names = synergy_motor_names or []
        self.ctrl_range_rad = np.radians(ctrl_range_deg)
        self.synergy_with_sigma_sliders = synergy_with_sigma_sliders

        # 将 URDF 转换为 MJCF XML 并加载
        self._load_model()

        # 建立关节名 → qpos 地址映射
        self.joint_name_to_index: dict[str, int] = {}
        self.joint_name_to_qposadr: dict[str, int] = {}
        self.qpos_joint_names: list[str] = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and not name.startswith('_slider_'):
                self.joint_name_to_index[name] = i
                self.joint_name_to_qposadr[name] = self.model.jnt_qposadr[i]
                self.qpos_joint_names.append(name)

        # 记录 slider joints + actuator 索引
        self._slider_actuator_indices: dict[str, int] = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self._slider_actuator_indices[name] = i

        # 初始化 ctrl = 0
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0

        # 保存上一帧 ctrl 值，用于检测哪个滑块被拖动
        self._prev_ctrl = self.data.ctrl.copy()

        # 启动 MuJoCo 查看器
        self._init_viewer(show_left_ui, show_right_ui)

    # ===================================================================
    # URDF → MJCF 转换
    # ===================================================================

    def _urdf_to_mjcf(self):
        """
        将 URDF 转换为 MJCF XML 字符串。
        在 synergy 模式中，使用 dummy body+joint 承载 actuator 滑块，
        kp=0 使其不产生任何力矩。
        在四滑块模式中，创建 4 个滑块：Motor A, Motor B, σ, σ_f。
        """
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        robot_name = root.get('name', 'hand')
        mesh_abs = self.mesh_dir.replace('\\', '/')

        links = {}
        for le in root.findall('link'):
            links[le.get('name')] = le

        joints = []
        for je in root.findall('joint'):
            joints.append({
                'name': je.get('name'),
                'type': 'hinge' if je.get('type', 'revolute') == 'revolute' else 'hinge',
                'parent': je.find('parent').get('link'),
                'child': je.find('child').get('link'),
                'origin': je.find('origin'),
                'axis': je.find('axis'),
                'limit': je.find('limit'),
            })

        all_child_names = {j['child'] for j in joints}
        root_body_name = None
        for j in joints:
            if j['parent'] not in all_child_names:
                root_body_name = j['parent']
                break
        if root_body_name is None:
            root_body_name = joints[0]['parent'] if joints else list(links.keys())[0]

        children_map = {}
        for j in joints:
            children_map.setdefault(j['parent'], []).append((j['child'], j))

        def _mesh_name(name):
            le = links.get(name)
            if le is None:
                return None
            v = le.find('visual')
            if v is None:
                return None
            g = v.find('geometry')
            if g is None:
                return None
            m = g.find('mesh')
            if m is None:
                return None
            return f'mesh_{name}'

        def _joint_axis_xyz(ji):
            if ji['axis'] is not None:
                return ji['axis'].get('xyz', '0 0 1')
            return '0 0 1'

        def _joint_range(ji):
            limit = ji['limit']
            if limit is not None:
                return f'{limit.get("lower", "-1.57")} {limit.get("upper", "1.57")}'
            return '-1.57 1.57'

        def _joint_origin_xyz(ji):
            if ji['origin'] is not None:
                return ji['origin'].get('xyz', '0 0 0')
            return '0 0 0'

        joint_names = []
        lines = []
        lines.append('<?xml version="1.0"?>')
        lines.append(f'<mujoco model="{robot_name}">')

        # asset
        lines.append('  <asset>')
        for name in links:
            mn = _mesh_name(name)
            if mn is not None:
                le = links[name]
                me = le.find('visual').find('geometry').find('mesh')
                fn = me.get('filename', '')
                if fn.startswith('meshes/'):
                    fn = mesh_abs + '/' + fn[7:]
                elif not os.path.isabs(fn):
                    fn = os.path.join(
                        os.path.dirname(self.urdf_path), fn
                    ).replace('\\', '/')
                lines.append(f'    <mesh name="{mn}" file="{fn}"/>')
        lines.append('  </asset>')

        lines.append('  <compiler angle="radian"/>')
        lines.append('  <option gravity="0 0 0"/>')
        lines.append('  <worldbody>')
        lines.append('    <geom type="plane" size="500 500 0.1" pos="0 0 -20" rgba="1 1 1 1"/>')

        def add_body(name, joint_info=None, indent=4):
            nonlocal joint_names
            pf = ' ' * indent
            if joint_info is not None:
                pos = _joint_origin_xyz(joint_info)
            else:
                pos = '0 0 0'
            lines.append(f'{pf}<body name="{name}" pos="{pos}">')

            if joint_info is not None:
                jn = joint_info["name"]
                joint_names.append(jn)
                lines.append(
                    f'{pf}  <joint name="{jn}" type="hinge" '
                    f'pos="0 0 0" axis="{_joint_axis_xyz(joint_info)}" '
                    f'range="{_joint_range(joint_info)}"/>'
                )

            mn = _mesh_name(name)
            if mn is not None:
                lines.append(f'{pf}  <geom type="mesh" mesh="{mn}" pos="0 0 0" '
                             f'rgba="0.6 0.85 1.0 1.0"/>')

            if name in children_map:
                for child_name, ji in children_map[name]:
                    add_body(child_name, ji, indent + 2)

            lines.append(f'{pf}</body>')

        add_body(root_body_name, None, indent=4)

        # ========== 协同模式：添加 dummy sliders ==========
        if self.synergy_callback is not None:
            cr_slider = f"{-self.ctrl_range_rad:.4f} {self.ctrl_range_rad:.4f}"

            if self.synergy_with_sigma_sliders:
                # 四滑块模式：Motor A, Motor B, σ, σ_f
                slider_names = ["Motor A", "Motor B", "σ (同动量)", "σ_f (差动量)"]
            else:
                # 双滑块模式：仅 Motor A, Motor B
                slider_names = self.synergy_motor_names or ["Motor A", "Motor B"]

            for i, sname in enumerate(slider_names):
                lines.append(f'    <body name="_slider_body_{i}" pos="0 0 0">')
                lines.append(f'      <joint name="_slider_{i}" type="slide" '
                             f'axis="1 0 0" pos="0 0 0" range="{cr_slider}"/>')
                # 不可见、无碰撞的 geom
                lines.append(f'      <geom type="sphere" size="0.001" rgba="0 0 0 0" '
                             f'contype="0" conaffinity="0"/>')
                lines.append(f'    </body>')
        else:
            # 直接模式：无额外滑块
            pass

        lines.append('  </worldbody>')

        # ========== Actuator ==========
        lines.append('  <actuator>')
        if self.synergy_callback is not None:
            cr = f"{-self.ctrl_range_rad:.4f} {self.ctrl_range_rad:.4f}"

            if self.synergy_with_sigma_sliders:
                # 四滑块模式
                slider_names = ["Motor A", "Motor B", "Sigma", "Sigma_f"]
            else:
                slider_names = self.synergy_motor_names or ["Motor A", "Motor B"]

            for i, mname in enumerate(slider_names):
                lines.append(
                    f'    <position name="{mname}" joint="_slider_{i}" '
                    f'kp="0" ctrlrange="{cr}"/>'
                )
        else:
            # 直接模式：每个关节一个 position actuator
            cr_direct = '-3.14 3.14'
            for jn in joint_names:
                lines.append(
                    f'    <position name="act_{jn}" joint="{jn}" '
                    f'kp="5" ctrlrange="{cr_direct}"/>'
                )
        lines.append('  </actuator>')

        lines.append('</mujoco>')
        return '\n'.join(lines)

    # ===================================================================
    # 模型加载 + 查看器
    # ===================================================================

    def _load_model(self):
        mjcf_xml = self._urdf_to_mjcf()

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='w', encoding='utf-8')
        tmp.write(mjcf_xml)
        tmp.close()
        self._tmp_xml = tmp.name

        self.model = mujoco.MjModel.from_xml_path(self._tmp_xml)
        self.data = mujoco.MjData(self.model)

        # 关闭重力
        self.model.opt.gravity = (0.0, 0.0, 0.0)
        self.model.vis.headlight.ambient[:] = (0.5, 0.5, 0.5)

        print(f"MuJoCo model loaded: nq={self.model.nq}, nv={self.model.nv}, "
              f"nbody={self.model.nbody}, njnt={self.model.njnt}, "
              f"ngeom={self.model.ngeom}, nmesh={self.model.nmesh}, "
              f"nu={self.model.nu} (== num actuators == num ctrl sliders)")

        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                print(f"  Joint [{i}]: '{name}' -> qpos[{self.model.jnt_qposadr[i]}]")

        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                print(f"  Actuator [{i}]: '{name}'")

    def _init_viewer(self, show_left_ui: bool, show_right_ui: bool):
        """启动 MuJoCo 原生 GLFW 查看器"""
        mujoco.mj_forward(self.model, self.data)

        self.viewer = viewer.launch_passive(
            self.model, self.data,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui
        )
        self._auto_align_camera()
        print("MuJoCo viewer launched.")

    def _auto_align_camera(self):
        """自动对准相机到模型中心"""
        center = self.model.stat.center
        extent = self.model.stat.extent
        self.viewer.cam.lookat[:] = center
        self.viewer.cam.distance = 2.5 * extent
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 45
        self.viewer.cam.trackbodyid = -1

    # ===================================================================
    # 公共接口
    # ===================================================================

    def set_joint_angle(self, joint_name: str, angle_deg: float):
        """设置关节角度（度），并同步到查看器"""
        if joint_name not in self.joint_name_to_qposadr:
            print(f"Warning: joint '{joint_name}' not found")
            return
        adr = self.joint_name_to_qposadr[joint_name]
        rad = np.radians(angle_deg)
        self.data.qpos[adr] = rad

        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    def get_joint_angle(self, joint_name: str) -> float:
        if joint_name not in self.joint_name_to_qposadr:
            return 0.0
        return np.degrees(self.data.qpos[self.joint_name_to_qposadr[joint_name]])

    def set_joint_angles(self, angle_dict: dict):
        """批量设置关节角度 {joint_name: angle_deg}"""
        for name, deg in angle_dict.items():
            if name in self.joint_name_to_qposadr:
                adr = self.joint_name_to_qposadr[name]
                rad = np.radians(deg)
                self.data.qpos[adr] = rad

        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    def reset(self):
        """重置所有关节到零位"""
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0
        self._prev_ctrl = self.data.ctrl.copy()
        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()
        print("Model reset to neutral position.")

    def close(self):
        """关闭查看器，清理临时文件"""
        if hasattr(self, 'viewer'):
            self.viewer.close()
        if hasattr(self, '_tmp_xml') and self._tmp_xml and os.path.exists(self._tmp_xml):
            try:
                os.unlink(self._tmp_xml)
            except OSError:
                pass

    def run(self, timeout: float = None):
        """
        进入主循环（阻塞），运行仿真直到查看器窗口关闭或超时。

        协同模式（四滑块）：
          滑块索引：0=Motor A, 1=Motor B, 2=σ, 3=σ_f
          每帧检测哪对滑块变动：
          - 若 A/B 变动为主 → 计算 σ/σ_f 从动
          - 若 σ/σ_f 变动为主 → 计算 A/B 从动
          然后调用 synergy_callback(A, B) → 设置 qpos

        协同模式（双滑块）：
          滑块索引：0=Motor A, 1=Motor B
          直接调用 synergy_callback(A, B)

        直接模式：
          直接复制 data.ctrl → data.qpos
        """
        import time
        start_time = time.time()

        print("Simulation running. Close the MuJoCo viewer window to exit.")

        # 设置四滑块互联动
        num_ctrl = self.model.nu

        try:
            while self.viewer.is_running():
                if self.synergy_callback is not None and num_ctrl >= 2:
                    ctrl = self.data.ctrl.copy()
                    prev = self._prev_ctrl

                    if num_ctrl >= 4 and self.synergy_with_sigma_sliders:
                        # ---- 四滑块互联动 ----
                        # 检测变化量
                        delta_ab = max(abs(ctrl[0] - prev[0]), abs(ctrl[1] - prev[1]))
                        delta_sig = max(abs(ctrl[2] - prev[2]), abs(ctrl[3] - prev[3]))

                        # 容差：防止数值噪声触发切换
                        # 记录最后更新方式，避免抖动
                        if not hasattr(self, '_last_active_pair'):
                            self._last_active_pair = 'AB'

                        if delta_ab > delta_sig + 1e-8:
                            # 用户在拖动 A/B → A,B 为主，σ,σ_f 从动
                            motor_a = ctrl[0]
                            motor_b = ctrl[1]
                            sigma = (motor_a + motor_b) / 2.0
                            sigma_f = (motor_a - motor_b) / 2.0
                            ctrl[2] = sigma
                            ctrl[3] = sigma_f
                            self._last_active_pair = 'AB'
                        elif delta_sig > delta_ab + 1e-8:
                            # 用户在拖动 σ/σ_f → σ,σ_f 为主，A,B 从动
                            sigma = ctrl[2]
                            sigma_f = ctrl[3]
                            ctrl[0] = sigma + sigma_f
                            ctrl[1] = sigma - sigma_f
                            self._last_active_pair = 'SIG'
                        else:
                            # 无显著变化或两者同时变化：保持上次模式
                            if self._last_active_pair == 'AB':
                                sigma = (ctrl[0] + ctrl[1]) / 2.0
                                sigma_f = (ctrl[0] - ctrl[1]) / 2.0
                                ctrl[2] = sigma
                                ctrl[3] = sigma_f
                            else:
                                sigma = ctrl[2]
                                sigma_f = ctrl[3]
                                ctrl[0] = sigma + sigma_f
                                ctrl[1] = sigma - sigma_f

                        # 将更新后的 ctrl 写回
                        self.data.ctrl[:] = ctrl
                        self._prev_ctrl = ctrl.copy()

                        # 使用 A,B 驱动手指
                        theta1 = ctrl[0]
                        theta2 = ctrl[1]
                    else:
                        # ---- 双滑块模式 ----
                        theta1 = ctrl[0]
                        theta2 = ctrl[1]
                        self._prev_ctrl = ctrl.copy()

                    # 通过协同模型计算关节角度
                    qpos_dict = self.synergy_callback(theta1, theta2)
                    for jname, rad in qpos_dict.items():
                        if jname in self.joint_name_to_qposadr:
                            adr = self.joint_name_to_qposadr[jname]
                            self.data.qpos[adr] = rad
                else:
                    # 直接模式：ctrl → qpos
                    if self.model.nu > 0:
                        self.data.qpos[:self.model.nu] = self.data.ctrl[:]
                    self._prev_ctrl = self.data.ctrl.copy()

                mujoco.mj_forward(self.model, self.data)
                self.viewer.sync()

                if timeout and (time.time() - start_time) > timeout:
                    print(f"Simulation timed out after {timeout:.0f}s.")
                    break
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.close()
