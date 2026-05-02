# src/interactive/mujoco_simulator.py
"""
基于 MuJoCo 的交互式 URDF 仿真器

使用 MuJoCo 3.x 原生 GLFW 查看器替代 MeshCat。
- 更高质量的渲染（光照、阴影、材质）
- 更流畅的相机控制（滚轮缩放范围无限制）
- 右侧面板提供关节角度控制滑块

MuJoCo 查看器右侧面板有两类滑块：
  - "Joint" 滑块（灰色/只读）：仅显示 data.qpos，不可编辑
  - "Position" 滑块（可交互）：对应 MJCF 中 <position actuator> 的 data.ctrl

本模块在 MJCF 中为每个关节添加 position actuator，
使右侧面板显示可交互的 "Position" 滑块。
"""

import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import mujoco
from mujoco import viewer


class MuJoCoSimulator:
    """基于 MuJoCo 的折纸手交互式仿真器"""

    def __init__(self, urdf_path: str, mesh_dir: str, show_left_ui: bool = True,
                 show_right_ui: bool = True):
        """
        Args:
            urdf_path: URDF 文件路径
            mesh_dir: STL mesh 文件目录（绝对路径）
            show_left_ui: 是否显示查看器左侧 UI（信息面板）
            show_right_ui: 是否显示查看器右侧 UI（关节滑块控制面板）
        """
        self.urdf_path = os.path.abspath(urdf_path)
        self.mesh_dir = os.path.abspath(mesh_dir)
        self._tmp_xml = None

        # 将 URDF 转换为 MJCF XML 并加载
        self._load_model()

        # 建立关节名 → qpos 地址映射
        self.joint_name_to_index: dict[str, int] = {}
        self.joint_name_to_qposadr: dict[str, int] = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_name_to_index[name] = i
                self.joint_name_to_qposadr[name] = self.model.jnt_qposadr[i]

        # 初始化 ctrl = 0（同步 qpos 初始值）
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0

        # 启动 MuJoCo 查看器
        self._init_viewer(show_left_ui, show_right_ui)

    # ===================================================================
    # URDF → MJCF 转换
    # ===================================================================

    def _urdf_to_mjcf(self):
        """
        将 URDF 转换为 MJCF XML 字符串。

        关键映射规则：
        1. 子 body pos = joint origin xyz（子 link 原点在父坐标系中的位置）
        2. joint 放在子 body 内部，pos="0 0 0"
        3. joint axis 从 URDF 继承
        4. 添加 <position actuator> 使右侧面板出现可交互的滑块
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
                # 仅使用一个 geom（group=0 默认可见），STL 内置法线决定正反面
                lines.append(f'{pf}  <geom type="mesh" mesh="{mn}" pos="0 0 0" '
                             f'rgba="0.65 0.65 0.7 1.0"/>')

            if name in children_map:
                for child_name, ji in children_map[name]:
                    add_body(child_name, ji, indent + 2)

            lines.append(f'{pf}</body>')

        add_body(root_body_name, None, indent=4)
        lines.append('  </worldbody>')

        # actuator: position servo —— 让右侧面板出现"Position"（可交互）滑块
        if joint_names:
            lines.append('  <actuator>')
            for jn in joint_names:
                lines.append(f'    <position name="act_{jn}" joint="{jn}" '
                             f'kp="5" ctrlrange="-3.14 3.14"/>')
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

        print(f"MuJoCo model loaded: nq={self.model.nq}, nv={self.model.nv}, "
              f"nbody={self.model.nbody}, njnt={self.model.njnt}, "
              f"ngeom={self.model.ngeom}, nmesh={self.model.nmesh}, "
              f"nu={self.model.nu} (== num actuators == num ctrl sliders)")
        print(f"  extent={self.model.stat.extent:.2f}")
        print(f"  center=({self.model.stat.center[0]:.2f}, "
              f"{self.model.stat.center[1]:.2f}, {self.model.stat.center[2]:.2f})")

        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                print(f"  Joint [{i}]: '{name}' -> qpos[{self.model.jnt_qposadr[i]}]")

    def _init_viewer(self, show_left_ui: bool, show_right_ui: bool):
        """启动 MuJoCo 原生 GLFW 查看器"""
        mujoco.mj_forward(self.model, self.data)

        self.viewer = viewer.launch_passive(
            self.model, self.data,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui
        )
        # 自动对准相机
        self._auto_align_camera()
        print("MuJoCo viewer launched.")
        print("  - 鼠标左键拖拽: 旋转视角")
        print("  - 鼠标滚轮: 缩放（拉远无限制）")
        print("  - 鼠标右键拖拽: 平移")
        print("  - 右侧面板调节关节角度:")
        print("      「Position」滑块: 可交互拖动控制关节角度")
        print("      「Joint」滑块: 只读显示（灰色），忽略即可")

    def _auto_align_camera(self):
        """自动对准相机到模型中心，距离适配模型大小"""
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

        # 也同步 ctrl（如果有 actuator）
        act_name = f"act_{joint_name}"
        for i in range(self.model.nu):
            an = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if an == act_name:
                self.data.ctrl[i] = rad
                break

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

                act_name = f"act_{name}"
                for i in range(self.model.nu):
                    an = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    if an == act_name:
                        self.data.ctrl[i] = rad
                        break

        mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    def reset(self):
        """重置所有关节到零位"""
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0
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

        MuJoCo 查看器右侧面板有两类滑块：
        - "Position" 滑块（可交互）：对应 position actuator，拖动它设置 data.ctrl
        - "Joint" 滑块（只读灰色）：仅显示 data.qpos

        手动将 ctrl（滑块值）复制到 qpos，然后执行 mj_forward 更新场景。
        不使用 mj_step，避免动力学计算引发的崩溃。

        Args:
            timeout: 超时秒数，如果为 None 则无限等待直到窗口关闭。
        """
        import time
        start_time = time.time()

        if timeout:
            print(f"Simulation running (timeout={timeout:.0f}s). "
                  f"Close the MuJoCo viewer window or wait for timeout to exit.")
        else:
            print("Simulation running. Close the MuJoCo viewer window to exit.")

        try:
            while self.viewer.is_running():
                # 手动复制 ctrl → qpos（滑块值 → 关节位置）
                # 查看器右侧的 "Position" 滑块直接修改 data.ctrl
                # 我们将其转移到 data.qpos 中，再执行前向运动学
                if self.model.nu > 0:
                    self.data.qpos[:self.model.nu] = self.data.ctrl[:]
                mujoco.mj_forward(self.model, self.data)
                self.viewer.sync()
                if timeout and (time.time() - start_time) > timeout:
                    print(f"Simulation timed out after {timeout:.0f}s.")
                    break
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            self.close()


