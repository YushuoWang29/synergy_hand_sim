# src/visualization/mujoco_animator.py
"""
MuJoCo 3D 动画引擎 - 用于力学数值仿真结果的可视化
=====================================================

将仿真轨迹 (Phase 1/2/3) 输出的关节角度序列 q(t) 在 MuJoCo 3D 查看器中
以动画形式回放。场景设置、打光、材质、背景与 src/interactive/mujoco_simulator.py
完全一致（3 点照明、棋盘格地面、暖灰背景、亚克力质感材质）。

工作流:
    .ohd ──► export_urdf() ──► .urdf + .stl ──► MuJoCo ──► 动画

用法:
    from src.visualization.mujoco_animator import SimulationMuJoCoAnimator

    animator = SimulationMuJoCoAnimator(ohd_path="hand.ohd")
    animator.load_trajectory(t=t_arr, q=q_arr)
    animator.animate(playback_speed=1.0)
"""

import os, sys, time, tempfile, shutil, json
import numpy as np
from typing import Optional, Dict, List, Tuple


class SimulationMuJoCoAnimator:
    """
    基于 MuJoCo 的数值仿真动画播放器。

    将 Phase 1/2/3 数值仿真得到的关节角度轨迹 q(t) 在 MuJoCo 3D 查看器中
    以动画形式播放。

    Parameters
    ----------
    ohd_path : str
        .ohd 设计文件路径。
    urdf_dir : str, optional
        URDF 文件目录。若未提供，自动在临时目录中生成。
    dt_playback : float
        播放帧间隔（秒），默认 0.033 (≈30 FPS)。
    show_left_ui : bool
        是否显示查看器左侧 UI。
    show_right_ui : bool
        是否显示查看器右侧 UI。
    """

    def __init__(self, ohd_path: str, urdf_dir: Optional[str] = None,
                 dt_playback: float = 0.033,
                 show_left_ui: bool = True,
                 show_right_ui: bool = True):
        self.ohd_path = os.path.abspath(ohd_path)
        self.ohd_name = os.path.splitext(os.path.basename(self.ohd_path))[0]
        self.dt_playback = dt_playback
        self.show_left_ui = show_left_ui
        self.show_right_ui = show_right_ui

        # URDF 导出控制
        self._urdf_dir = urdf_dir
        self._temp_dir = None
        self._urdf_path = None
        self._cleanup_temp = False

        # MuJoCo 模型
        self.model = None
        self.data = None
        self.viewer = None
        self._qpos_joint_indices: List[int] = []
        self._joint_names: List[str] = []

        # 轨迹数据
        self.t = None
        self.q = None
        self.n_joints_traj = 0
        self._n_frames = 0

        # 导出 URDF
        self._setup_urdf()

    # ================================================================
    # URDF 设置
    # ================================================================

    def _setup_urdf(self):
        """
        设置 URDF 文件路径。若 urdf_dir 未指定，则在临时目录中导出 URDF。
        """
        if self._urdf_dir is not None:
            urdf_dir = os.path.abspath(self._urdf_dir)
            self._urdf_path = os.path.join(urdf_dir, f"{self.ohd_name}.urdf")
            if not os.path.exists(self._urdf_path):
                self._export_urdf(urdf_dir)
        else:
            # 查找已有的标准位置：models/<ohd_name>/<ohd_name>.urdf
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            standard_urdf = os.path.join(project_root, "models",
                                         self.ohd_name, f"{self.ohd_name}.urdf")
            if os.path.exists(standard_urdf):
                self._urdf_path = standard_urdf
                print(f"  [MuJoCoAnimator] 使用现有 URDF: {standard_urdf}")
            else:
                self._temp_dir = tempfile.mkdtemp(prefix=f"mujoco_{self.ohd_name}_")
                self._urdf_path = os.path.join(self._temp_dir,
                                               f"{self.ohd_name}.urdf")
                self._export_urdf(self._temp_dir)
                self._cleanup_temp = True

    def _export_urdf(self, output_dir: str):
        """从 .ohd 文件导出 URDF + STL 网格。"""
        from src.models.origami_design import OrigamiHandDesign
        from src.models.origami_to_urdf import export_urdf

        print(f"\n  [MuJoCoAnimator] 从 .ohd 导出 URDF...")
        print(f"    设计文件: {self.ohd_path}")
        print(f"    输出目录: {output_dir}")

        design = OrigamiHandDesign.load(self.ohd_path)

        if not design.faces or len(design.faces) == 0:
            print("    重建面片拓扑...")
            design.build_topology()

        if design.root_face_id is None and design.faces:
            design.set_root_face(sorted(design.faces.keys())[0])
            design.build_face_tree()

        design.summary()

        urdf_path = os.path.join(output_dir, f"{self.ohd_name}.urdf")
        export_urdf(design, urdf_path, thickness=3.0)
        print(f"  [MuJoCoAnimator] URDF 导出完成: {urdf_path}")

        return urdf_path

    # ================================================================
    # 轨迹加载
    # ================================================================

    def load_trajectory(self, t: np.ndarray, q: np.ndarray):
        """
        加载仿真轨迹数据。

        Parameters
        ----------
        t : np.ndarray, shape (N,)
            时间序列。
        q : np.ndarray, shape (N, n_joints)
            关节角度序列（弧度）。
        """
        self.t = np.asarray(t, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64)
        self._n_frames = len(self.t)

        if self.q.ndim == 1:
            self.q = self.q.reshape(-1, 1)

        self.n_joints_traj = self.q.shape[1]
        print(f"  [MuJoCoAnimator] 轨迹加载: {self._n_frames} 帧, "
              f"{self.n_joints_traj} 个关节")
        print(f"    时间范围: [{self.t[0]:.4f}, {self.t[-1]:.4f}] s")

        if hasattr(self, '_joint_names') and self._joint_names:
            n_urdf_joints = len(self._joint_names)
            if n_urdf_joints != self.n_joints_traj:
                print(f"  ⚠ 警告: 轨迹关节数 ({self.n_joints_traj}) "
                      f"≠ URDF 关节数 ({n_urdf_joints})")
                print(f"    将自动取 min 进行适配。")

    def load_trajectory_from_npz(self, npz_path: str):
        """
        从 .npz 文件加载仿真轨迹。

        Parameters
        ----------
        npz_path : str
            仿真结果 .npz 文件路径。
        """
        from src.visualization.simulation_visualizer import load_trajectory_simple
        data, meta = load_trajectory_simple(npz_path)
        t = data.get('t', [])
        q = data.get('q', [])
        if len(t) < 2 or q.size == 0:
            raise ValueError(f"轨迹文件 {npz_path} 缺少有效数据")
        self.load_trajectory(t, q)
        return meta

    # ================================================================
    # MuJoCo 模型加载 — MJCF 构建
    # ================================================================

    def _build_mjcf(self) -> str:
        """
        将 URDF → MJCF XML。

        场景设置与 MuJoCoSimulator._urdf_to_mjcf() 完全一致：
          - 3 点照明（主光 + 补光 + 背光）
          - 暖灰背景（haze / fog rgba）
          - 棋盘格地面（动态缩放）
          - 亚克力/ABS 塑料质感材质（origami_plastic / origami_accent 交替）
          - 面片上抬 z=100 以避免与地面穿插
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(self._urdf_path)
        root = tree.getroot()
        robot_name = root.get('name', self.ohd_name)

        mesh_dir = os.path.join(os.path.dirname(self._urdf_path), "meshes")
        mesh_abs = mesh_dir.replace('\\', '/')

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

        # ==================== asset ====================
        lines.append('  <asset>')
        # 棋盘格地面纹理
        lines.append('    <texture name="checker_tex" type="2d" builtin="checker" '
                     'rgb1="0.25 0.26 0.28" rgb2="0.35 0.36 0.38" '
                     'width="512" height="512"/>')
        # 地面材质
        lines.append('    <material name="ground_mat" texture="checker_tex" '
                     'texrepeat="6 6" reflectance="0.15"/>')
        # 折纸手材质：亚克力/ABS塑料质感（与 MuJoCoSimulator 一致）
        lines.append('    <material name="origami_plastic" '
                     'rgba="0.65 0.82 0.95 1.0" '
                     'specular="0.4" shininess="0.3" '
                     'emission="0.0" texrepeat="1 1"/>')
        lines.append('    <material name="origami_accent" '
                     'rgba="0.75 0.85 0.98 1.0" '
                     'specular="0.5" shininess="0.4" '
                     'emission="0.0" texrepeat="1 1"/>')
        # 网格
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
                        os.path.dirname(self._urdf_path), fn
                    ).replace('\\', '/')
                lines.append(f'    <mesh name="{mn}" file="{fn}"/>')
        lines.append('  </asset>')

        # ==================== compiler / option ====================
        # contype=1/conaffinity=1 on each geom enables collision between
        # different bodies of the same model. The 'collision' attribute on
        # <compiler> is only supported in MuJoCo 3.1+, so we skip it here
        # for compatibility with MuJoCo 2.x / 3.0.
        lines.append('  <compiler angle="radian"/>')
        lines.append('  <option gravity="0 0 0" cone="elliptic" impratio="1"/>')

        # ==================== 可视化配置 ====================
        lines.append('  <visual>')
        # 背景色：暖灰色渐变（与 MuJoCoSimulator 一致）
        lines.append('    <rgba haze="0.25 0.26 0.28 1.0" '
                     'fog="0.25 0.26 0.28 1.0"/>')
        # 渲染质量
        lines.append('    <quality offsamples="8" shadowsize="8192"/>')
        # 前照灯
        lines.append('    <headlight ambient="0.15 0.15 0.18" '
                     'diffuse="0.5 0.5 0.55" '
                     'specular="0.3 0.3 0.3"/>')
        lines.append('    <map fogstart="5.0" fogend="15.0" haze="0.2"/>')
        lines.append('  </visual>')

        # ==================== worldbody ====================
        lines.append('  <worldbody>')

        # ---- 3 点照明（与 MuJoCoSimulator 一致） ----
        # 主光：左上侧暖色
        lines.append('    <light name="key_light" directional="true" '
                     'pos="2 1 3" dir="-2 -1 -3" '
                     'diffuse="0.65 0.6 0.55" ambient="0.0 0.0 0.0" '
                     'specular="0.5 0.5 0.5" castshadow="true"/>')
        # 补光：右下侧冷色
        lines.append('    <light name="fill_light" directional="true" '
                     'pos="-2 -1 2" dir="2 1 -2" '
                     'diffuse="0.35 0.4 0.5" ambient="0.0 0.0 0.0" '
                     'specular="0.2 0.2 0.4" castshadow="true"/>')
        # 背光：后上方勾勒边缘
        lines.append('    <light name="back_light" directional="true" '
                     'pos="0 -3 2" dir="0 3 -2" '
                     'diffuse="0.25 0.25 0.3" ambient="0.0 0.0 0.0" '
                     'specular="0.6 0.6 0.7" castshadow="true"/>')

        # ---- 棋盘格地面（占位，加载后动态缩放） ----
        lines.append('    <geom type="plane" name="_bg_ground" '
                     'size="1 1 0.02" pos="0 0 0" '
                     'material="ground_mat" '
                     'contype="0" conaffinity="0" group="1"/>')

        # ---- 手模型的面片 ---- 
        _material_toggle = [0]  # 闭包中交替材质
        _hand_z_offset = 100.0

        def add_body(name, joint_info=None, indent=4):
            nonlocal joint_names
            pf = ' ' * indent
            if joint_info is not None:
                pos = _joint_origin_xyz(joint_info)
            else:
                # 根 body：加 z 偏移，使手整体上抬
                orig_pos = '0 0 0'
                px, py, pz = [float(v) for v in orig_pos.split()]
                pos = f'{px} {py} {pz + _hand_z_offset}'
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
                # 交替使用两个材质（与 MuJoCoSimulator 一致）
                _material_toggle[0] += 1
                mat = 'origami_accent' if (_material_toggle[0] % 2 == 0) \
                       else 'origami_plastic'
                # contype=1 conaffinity=1: 启用碰撞检测，允许与其他 geom 碰撞
                lines.append(f'{pf}  <geom type="mesh" mesh="{mn}" '
                             f'pos="0 0 0" material="{mat}" '
                             f'contype="1" conaffinity="1"/>')

            if name in children_map:
                for child_name, ji in children_map[name]:
                    add_body(child_name, ji, indent + 2)

            lines.append(f'{pf}</body>')

        add_body(root_body_name, None, indent=4)
        lines.append('  </worldbody>')

        # ==================== 无 actuator ====================
        lines.append('  <actuator/>')

        lines.append('</mujoco>')

        self._joint_names = joint_names
        return '\n'.join(lines)

    # ================================================================
    # MuJoCo 模型加载 + 后处理
    # ================================================================

    def _load_mujoco_model(self):
        """构建 MJCF XML，加载到 MuJoCo，并应用运行时场景设置。"""
        import mujoco

        mjcf_xml = self._build_mjcf()

        tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False,
                                          mode='w', encoding='utf-8')
        tmp.write(mjcf_xml)
        tmp.close()
        self._tmp_xml = tmp.name

        self.model = mujoco.MjModel.from_xml_path(self._tmp_xml)
        self.data = mujoco.MjData(self.model)

        # ---- 运行时场景后置调整（与 MuJoCoSimulator._load_model 一致） ----
        # 关闭重力
        self.model.opt.gravity = (0.0, 0.0, 0.0)

        # 渲染质量
        self.model.vis.quality.offsamples = 8
        self.model.vis.quality.shadowsize = 8192

        # 线宽
        self.model.vis.global_.linewidth = 1.5

        # 棋盘格地面：定位在模型中心，z=0，大小扩大 10 倍
        E = self.model.stat.extent
        C = self.model.stat.center
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                '_bg_ground')
        if gid >= 0:
            self.model.geom_pos[gid] = (C[0], C[1], 0.0)
            self.model.geom_size[gid] = (E * 8.0, E * 8.0, 0.02)

        # ---- 关节索引映射 ----
        self._qpos_joint_indices = []
        for jname in self._joint_names:
            for i in range(self.model.njnt):
                name = mujoco.mj_id2name(self.model,
                                         mujoco.mjtObj.mjOBJ_JOINT, i)
                if name == jname:
                    self._qpos_joint_indices.append(
                        self.model.jnt_qposadr[i])
                    break

        print(f"  [MuJoCoAnimator] 模型加载完成: "
              f"nq={self.model.nq}, nv={self.model.nv}, "
              f"nbody={self.model.nbody}, njnt={self.model.njnt}, "
              f"ngeom={self.model.ngeom}")
        print(f"    关节映射: {len(self._qpos_joint_indices)} 个关节")

    # ================================================================
    # 查看器启动
    # ================================================================

    def _launch_viewer(self):
        """启动 MuJoCo 被动查看器，并启用 FOG 背景渲染标志。"""
        import mujoco
        from mujoco import viewer

        mujoco.mj_forward(self.model, self.data)

        self.viewer = viewer.launch_passive(
            self.model, self.data,
            show_left_ui=self.show_left_ui,
            show_right_ui=self.show_right_ui
        )

        # ---- 启用 FOG 渲染标志（与 MuJoCoSimulator 一致） ----
        # 默认 mjRND_FOG 在场景标志中为 0（关闭），导致背景为纯黑
        # 启用后 model.vis.rgba.fog 作为背景色生效
        self._enable_bg_each_frame = False
        if hasattr(self.viewer, 'user_scn'):
            try:
                self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
                self._enable_bg_each_frame = True
            except Exception:
                pass

        # ---- 自动对齐相机 ----
        center = self.model.stat.center
        extent = self.model.stat.extent
        self.viewer.cam.lookat[:] = center
        self.viewer.cam.distance = 2.5 * extent
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 45
        self.viewer.cam.trackbodyid = -1

        print("  [MuJoCoAnimator] 查看器已启动。")

    # ================================================================
    # 动画播放
    # ================================================================

    def animate(self, playback_speed: float = 1.0,
                loop: bool = False, max_frames: int = 500,
                timeout: float = 60.0,
                show_progress: bool = True,
                report_contacts: bool = True,
                report_contact_interval: int = 5):
        """
        播放轨迹动画。

        Parameters
        ----------
        playback_speed : float
            播放速度倍率。1.0 = 实时（与物理时间同步）。
            2.0 = 2 倍速。
        loop : bool
            是否循环播放。
        max_frames : int
            最大播放帧数。
        timeout : float
            超时秒数。
        show_progress : bool
            是否在控制台显示进度。
        report_contacts : bool
            是否报告碰撞检测信息。
        report_contact_interval : int
            每 N 个进度报告输出一次详细碰撞信息。
        """
        import mujoco
        import time as _time

        if self.t is None or self.q is None:
            print("[MuJoCoAnimator] 错误: 未加载轨迹。请先调用 "
                  "load_trajectory()。")
            return

        if self.model is None:
            self._load_mujoco_model()

        if self.viewer is None:
            self._launch_viewer()

        # ---- 启用查看器中的接触点可视化 ----
        if report_contacts and hasattr(self.viewer, 'user_scn'):
            try:
                # mjRND_REFLECTION=6 - 接触点可视化通常在遮挡物中显示
                # 实际上 MuJoCo 通过 opt.enablecontact 和 scn.flags 控制
                # 对于用户场景，我们启用 CONTACT 标志
                self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
                # 设置模型选项以渲染接触点
                self.model.vis.map.force = 0.01  # 接触力缩放
                self.model.vis.scale.forcewidth = 0.005  # 力线宽度
                self.model.opt.enableflags = (
                    self.model.opt.enableflags | mujoco.mjtEnableBit.mjENBL_CONTACT
                )
                print("  [MuJoCoAnimator] 接触点可视化已启用")
            except Exception as e:
                print(f"  [MuJoCoAnimator] 接触可视化设置警告: {e}")

        n_total = len(self.t)
        n_play = min(n_total, max_frames)
        step_every = max(1, n_total // n_play)

        frame_indices = list(range(0, n_total, step_every))

        n_urdf_joints = len(self._qpos_joint_indices)
        n_traj_joints = self.n_joints_traj
        n_use = min(n_urdf_joints, n_traj_joints)

        dt_real = self.dt_playback / max(playback_speed, 0.01)

        print(f"\n  [MuJoCoAnimator] 播放 {len(frame_indices)} 帧, "
              f"物理时长 {self.t[-1]:.3f}s, "
              f"播放帧率 {1/max(dt_real, 0.001):.0f} FPS")
        if playback_speed != 1.0:
            print(f"    播放速度: {playback_speed:.2f}x")
        print("    关闭 MuJoCo 查看器窗口退出。")
        print()

        t_start = _time.time()
        frame_count = 0
        max_contacts_seen = 0
        last_contact_report = -1

        try:
            while self.viewer.is_running():
                for idx in frame_indices:
                    if _time.time() - t_start > timeout:
                        if show_progress:
                            print("  [MuJoCoAnimator] 播放超时。")
                        return

                    # 每帧确保 FOG 渲染标志开启（与 MuJoCoSimulator 一致）
                    if self._enable_bg_each_frame \
                       and hasattr(self.viewer, 'user_scn'):
                        self.viewer.user_scn.flags[
                            mujoco.mjtRndFlag.mjRND_FOG] = 1

                    # 设置关节角度
                    for j in range(n_use):
                        qpos_adr = self._qpos_joint_indices[j]
                        self.data.qpos[qpos_adr] = float(self.q[idx, j])

                    # ============================
                    # 碰撞检测关键步骤：
                    # 使用 mj_step1 替代 mj_forward。
                    # mj_forward 只计算运动学（body 位置/速度），不进行碰撞检测。
                    # mj_step1 完成碰撞检测（填充 data.contact, data.ncon）
                    # 同时也计算加速度、约束等。
                    # 注意：我们会在下一帧覆盖 qpos，所以 mj_step1 的计算结果
                    # 不会影响轨迹。
                    # ============================
                    mujoco.mj_step1(self.model, self.data)
                    self.viewer.sync()

                    # ============================
                    # 碰撞检测报告
                    # ============================
                    ncon = self.data.ncon
                    if ncon > max_contacts_seen:
                        max_contacts_seen = ncon

                    should_report = (show_progress and report_contacts
                                     and frame_count % max(1, len(frame_indices) // report_contact_interval) == 0
                                     and frame_count != last_contact_report)
                    if should_report:
                        last_contact_report = frame_count
                        pct = frame_count / len(frame_indices) * 100
                        phys_t = self.t[idx]
                        elapsed = _time.time() - t_start

                        if ncon > 0:
                            # 列出前几个碰撞对
                            contact_details = []
                            max_show = min(3, ncon)
                            for ci in range(max_show):
                                c = self.data.contact[ci]
                                g1_name = mujoco.mj_id2name(
                                    self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                                g2_name = mujoco.mj_id2name(
                                    self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                                dist = c.dist
                                contact_details.append(
                                    f"{g1_name}↔{g2_name}(d={dist:.4f})")
                            contact_str = "; ".join(contact_details)
                            if ncon > max_show:
                                contact_str += f"; ... +{ncon-max_show} more"
                            print(f"    [{pct:3.0f}%] t={phys_t:.4f}s "
                                  f"⚡ 碰撞: {ncon} 处 | {contact_str} "
                                  f"(实时 {elapsed:.1f}s)")
                        else:
                            print(f"    [{pct:3.0f}%] t={phys_t:.4f}s "
                                  f"✓ 无碰撞 (实时 {elapsed:.1f}s)")

                    # Frame pacing
                    t_frame = _time.time()
                    sleep = max(0, dt_real - (_time.time() - t_frame))
                    if sleep > 0:
                        _time.sleep(sleep)

                    frame_count += 1

                if not loop:
                    print(f"\n  [MuJoCoAnimator] 动画播放完成。"
                          f"最大同时碰撞数: {max_contacts_seen}")
                    hold_timeout = max(timeout, 10.0)
                    print(f"    将在 {hold_timeout:.0f}s 后自动关闭查看器。")
                    hold_start = _time.time()
                    while self.viewer.is_running():
                        # 保持 FOG 标志
                        if self._enable_bg_each_frame \
                           and hasattr(self.viewer, 'user_scn'):
                            self.viewer.user_scn.flags[
                                mujoco.mjtRndFlag.mjRND_FOG] = 1
                        # 保持碰撞检测
                        mujoco.mj_step1(self.model, self.data)
                        self.viewer.sync()
                        _time.sleep(0.05)
                        if _time.time() - hold_start > hold_timeout:
                            print(f"  [MuJoCoAnimator] 保持超时 ({hold_timeout:.0f}s)，自动关闭。")
                            # 强制关闭查看器
                            if hasattr(self.viewer, 'close'):
                                self.viewer.close()
                            break
                    break

        except KeyboardInterrupt:
            print("\n  [MuJoCoAnimator] 用户中断。")
        finally:
            self.close()
            print(f"  [MuJoCoAnimator] 动画结束。最大同时碰撞数: {max_contacts_seen}")

    # ================================================================
    # 清理
    # ================================================================

    def close(self):
        """关闭查看器，清理临时文件。"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

        if hasattr(self, '_tmp_xml') and self._tmp_xml \
           and os.path.exists(self._tmp_xml):
            try:
                os.unlink(self._tmp_xml)
            except OSError:
                pass

        if self._cleanup_temp and self._temp_dir \
           and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
                print(f"  [MuJoCoAnimator] 临时文件已清理: "
                      f"{self._temp_dir}")
            except OSError as e:
                print(f"  [MuJoCoAnimator] 临时文件清理失败: {e}")

    def __del__(self):
        self.close()


# ================================================================
#  便捷函数
# ================================================================

def animate_ohd_trajectory(ohd_path: str, t: np.ndarray, q: np.ndarray,
                           playback_speed: float = 1.0, loop: bool = False,
                           max_frames: int = 500, **kwargs):
    """
    便捷函数：加载 .ohd 设计并播放仿真轨迹动画。

    Parameters
    ----------
    ohd_path : str
        .ohd 设计文件路径。
    t : np.ndarray
        时间序列。
    q : np.ndarray
        关节角度序列。
    playback_speed : float
        播放速度倍率。
    loop : bool
        是否循环播放。
    max_frames : int
        最大播放帧数。
    **kwargs
        传递给 SimulationMuJoCoAnimator 的其他参数。
    """
    animator = SimulationMuJoCoAnimator(ohd_path=ohd_path, **kwargs)
    animator.load_trajectory(t=t, q=q)
    animator.animate(playback_speed=playback_speed, loop=loop,
                     max_frames=max_frames)
    return animator


def animate_from_npz(ohd_path: str, npz_path: str,
                     playback_speed: float = 1.0, **kwargs):
    """
    便捷函数：从 .npz 文件加载轨迹并播放动画。

    Parameters
    ----------
    ohd_path : str
        .ohd 设计文件路径。
    npz_path : str
        仿真结果 .npz 文件路径。
    playback_speed : float
        播放速度倍率。
    **kwargs
        传递给 SimulationMuJoCoAnimator 的其他参数。
    """
    animator = SimulationMuJoCoAnimator(ohd_path=ohd_path, **kwargs)
    animator.load_trajectory_from_npz(npz_path)
    animator.animate(playback_speed=playback_speed)
    return animator
