# src/interactive/pinocchio_simulator.py
"""
基于 Pinocchio 的交互式 URDF 仿真器
左侧：2D CAD 视图（点击选折痕）
右侧：3D MeshCat 视图（Pinocchio 实时更新）
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PyQt5 import QtWidgets, QtCore
from typing import Dict, Optional
import time
import os

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from src.models.origami_design import OrigamiHandDesign, FoldType
from src.interactive.cad_viewer import CADViewer


class PinocchioSimulator(QtWidgets.QMainWindow):
    """基于 Pinocchio 的折纸手交互式仿真器"""

    def __init__(self, design: OrigamiHandDesign, urdf_path: str, mesh_dir: str):
        super().__init__()

        self.design = design
        self.urdf_path = urdf_path
        self.mesh_dir = mesh_dir

        # 加载 Pinocchio 模型
        self._load_model()

        # 当前关节角度（弧度）
        self.joint_angles: Dict[int, float] = {}
        # 初始化所有实际关节为 0
        for i in range(1, self.model.njoints):
            if self.model.joints[i].nq == 1:
                self.joint_angles[i] = 0.0
        # Pinocchio 配置向量
        self.q = pin.neutral(self.model)

        # 选中的折痕 → Pinocchio joint index 映射
        self.selected_line_id: Optional[int] = None
        self._line_to_pin_joint: Dict[int, int] = {}  # line_id → pin_joint_idx

        # 建立映射：从 URDF joint name 反推 line_id
        # URDF joint 名格式：joint_{joint_id}，joint_id 对应 design.joints 的 id
        for design_joint in self.design.joints:
            fold_line_id = design_joint.fold_line_id
            urdf_joint_name = f"joint_{design_joint.id}"
            # 在 Pinocchio model 中搜索这个 joint name
            for pin_j in range(1, self.model.njoints):
                if self.model.names[pin_j] == urdf_joint_name:
                    self._line_to_pin_joint[fold_line_id] = pin_j
                    break

        # 3D 可视化器
        self._init_3d_view()

        # 初始化 UI
        self._init_ui()

        # 定时刷新 3D
        self._last_q = self.q.copy()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_3d)
        self.timer.start(50)  # 20 fps

    def _load_model(self):
        """加载 Pinocchio 模型"""
        # 如果 mesh 路径是相对的，先转换为绝对路径
        urdf_dir = os.path.dirname(os.path.abspath(self.urdf_path))
        mesh_abs = os.path.abspath(self.mesh_dir)

        # 读取 URDF 内容，替换相对 mesh 路径
        with open(self.urdf_path, 'r') as f:
            urdf_content = f.read()
        urdf_content = urdf_content.replace('meshes/', mesh_abs.replace('\\', '/') + '/')

        # 写临时文件
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.urdf', delete=False, mode='w')
        tmp.write(urdf_content)
        tmp.close()
        self._tmp_urdf = tmp.name

        # 用 Pinocchio 加载
        self.model = pin.buildModelFromUrdf(self._tmp_urdf)
        self.data = self.model.createData()
        self.visual_model = pin.buildGeomFromUrdf(
            self.model, self._tmp_urdf, pin.GeometryType.VISUAL
        )

        print(f"Pinocchio model loaded: {self.model.njoints} joints, {self.model.nq} DoF")
        for i in range(1, self.model.njoints):
            name = self.model.names[i]
            nq = self.model.joints[i].nq
            if nq == 1:
                print(f"  [{i}] {name} (revolute)")

    def _init_3d_view(self):
        """初始化 MeshCat 3D 视图"""
        # 创建空的 collision model（我们不需要碰撞检测）
        collision_model = pin.GeometryModel()
        
        self.viz = MeshcatVisualizer(self.model, collision_model, self.visual_model)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()
        self.viz.display(self.q)

    def _init_ui(self):
        """初始化界面"""
        self.setWindowTitle("Origami Pinocchio Simulator")
        self.setMinimumSize(1200, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # 左侧：2D CAD 面板
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)

        title = QtWidgets.QLabel("2D CAD View")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        left_layout.addWidget(title)

        self.cad_viewer = CADViewer(design=self.design)
        self.cad_viewer.set_select_callback(self._on_line_selected)

        self.canvas = FigureCanvasQTAgg(self.cad_viewer.get_figure())
        self.canvas.setMinimumSize(500, 500)
        left_layout.addWidget(self.canvas)

        toolbar = NavigationToolbar2QT(self.canvas, self)
        left_layout.addWidget(toolbar)
        main_layout.addWidget(left_panel, stretch=1)

        # 右侧：控制面板
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        info = QtWidgets.QLabel("Control Panel")
        info.setStyleSheet("font-weight: bold; font-size: 14px;")
        info.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(info)

        self.selection_label = QtWidgets.QLabel("Click on a red/blue fold line to select it")
        self.selection_label.setWordWrap(True)
        self.selection_label.setStyleSheet(
            "padding: 10px; background: #e8f5e9; border: 1px solid #4caf50; "
            "border-radius: 5px; font-size: 12px;"
        )
        right_layout.addWidget(self.selection_label)

        # 角度控制
        angle_group = QtWidgets.QGroupBox("Joint Angle Control")
        angle_layout = QtWidgets.QVBoxLayout(angle_group)

        self.angle_label = QtWidgets.QLabel("Select a fold line first")
        self.angle_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.angle_label.setAlignment(QtCore.Qt.AlignCenter)
        angle_layout.addWidget(self.angle_label)

        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(QtWidgets.QLabel("-180°"))
        self.angle_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.angle_slider.setMinimum(-180)
        self.angle_slider.setMaximum(180)
        self.angle_slider.setValue(0)
        self.angle_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.angle_slider.setTickInterval(15)
        self.angle_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.angle_slider)
        slider_layout.addWidget(QtWidgets.QLabel("+180°"))
        angle_layout.addLayout(slider_layout)

        input_layout = QtWidgets.QHBoxLayout()
        input_layout.addWidget(QtWidgets.QLabel("Angle:"))
        self.angle_input = QtWidgets.QDoubleSpinBox()
        self.angle_input.setMinimum(-180)
        self.angle_input.setMaximum(180)
        self.angle_input.setValue(0)
        self.angle_input.setSuffix("°")
        self.angle_input.setDecimals(1)
        input_layout.addWidget(self.angle_input)

        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_angle)
        input_layout.addWidget(apply_btn)
        angle_layout.addLayout(input_layout)
        right_layout.addWidget(angle_group)

        # 所有关节状态
        joints_group = QtWidgets.QGroupBox("All Joints")
        joints_layout = QtWidgets.QVBoxLayout(joints_group)
        self.joints_text = QtWidgets.QTextEdit()
        self.joints_text.setReadOnly(True)
        self.joints_text.setMaximumHeight(200)
        self.joints_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        joints_layout.addWidget(self.joints_text)

        reset_btn = QtWidgets.QPushButton("Reset All (R)")
        reset_btn.clicked.connect(self._reset_all)
        joints_layout.addWidget(reset_btn)
        right_layout.addWidget(joints_group)

        # 快捷键
        shortcuts = QtWidgets.QLabel(
            "Keys:\n  ↑↓ : ±5°\n  ←→ : ±1°\n  Enter : Apply\n  R : Reset selected\n  Esc : Deselect"
        )
        shortcuts.setStyleSheet(
            "padding: 8px; background: #fff3e0; border-radius: 5px; font-size: 11px;"
        )
        right_layout.addWidget(shortcuts)
        right_layout.addStretch()
        main_layout.addWidget(right_panel, stretch=0)

        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas.keyPressEvent = self._on_key_press
        self._update_joints_display()

    # ========== 事件处理 ==========

    def _on_line_selected(self, line_id: int, fold_type: FoldType):
        self.selected_line_id = line_id
        pin_joint = self._line_to_pin_joint.get(line_id)

        if pin_joint is not None:
            # 从 Pinocchio 配置向量读取当前角度
            idx_q = self.model.idx_qs[pin_joint]
            angle_rad = self.q[idx_q]
            angle_deg = np.degrees(angle_rad)

            type_str = "MOUNTAIN" if fold_type == FoldType.MOUNTAIN else "VALLEY"
            self.selection_label.setText(
                f"<b>Line {line_id}</b> ({type_str}) → <b>Pin Joint {pin_joint}</b> | "
                f"Current: {angle_deg:.1f}°"
            )

            self.angle_slider.blockSignals(True)
            self.angle_slider.setValue(int(angle_deg))
            self.angle_slider.blockSignals(False)

            self.angle_input.blockSignals(True)
            self.angle_input.setValue(angle_deg)
            self.angle_input.blockSignals(False)

            self.angle_label.setText(f"{angle_deg:.1f}°")

    def _on_slider_changed(self, value):
        self._set_angle(float(value))

    def _apply_angle(self):
        self._set_angle(self.angle_input.value())
        self.angle_slider.blockSignals(True)
        self.angle_slider.setValue(int(self.angle_input.value()))
        self.angle_slider.blockSignals(False)

    def _set_angle(self, angle_deg: float):
        if self.selected_line_id is None:
            return
        pin_joint = self._line_to_pin_joint.get(self.selected_line_id)
        if pin_joint is None:
            return

        idx_q = self.model.idx_qs[pin_joint]
        self.q[idx_q] = np.radians(angle_deg)
        self.angle_label.setText(f"{angle_deg:.1f}°")
        self._update_joints_display()

    def _update_joints_display(self):
        lines = []
        for line_id, pin_j in sorted(self._line_to_pin_joint.items()):
            idx_q = self.model.idx_qs[pin_j]
            deg = np.degrees(self.q[idx_q])
            fl = self.design.fold_lines[line_id]
            sym = "▲" if fl.fold_type == FoldType.MOUNTAIN else "▼"
            marker = " ◀" if line_id == self.selected_line_id else ""
            lines.append(f"{sym} L{line_id} → PJ{pin_j} | {deg:+6.1f}°{marker}")
        self.joints_text.setText("\n".join(lines))

    def _update_3d(self):
        if not np.array_equal(self.q, self._last_q):
            self.viz.display(self.q)
            self._last_q = self.q.copy()

    def _on_key_press(self, event):
        if self.selected_line_id is None:
            return
        pin_joint = self._line_to_pin_joint.get(self.selected_line_id)
        if pin_joint is None:
            return

        idx_q = self.model.idx_qs[pin_joint]
        current = np.degrees(self.q[idx_q])
        key = event.key()

        if key == QtCore.Qt.Key_Up:
            new = current + 5
        elif key == QtCore.Qt.Key_Down:
            new = current - 5
        elif key == QtCore.Qt.Key_Right:
            new = current + 1
        elif key == QtCore.Qt.Key_Left:
            new = current - 1
        elif key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self._apply_angle()
            return
        elif key == QtCore.Qt.Key_R:
            new = 0
        elif key == QtCore.Qt.Key_Escape:
            self.selected_line_id = None
            self.selection_label.setText("Deselected. Click a fold line to select.")
            self.angle_label.setText("--")
            self._update_joints_display()
            return
        else:
            return

        new = max(-180, min(180, new))
        self._set_angle(new)
        self.angle_slider.blockSignals(True)
        self.angle_slider.setValue(int(new))
        self.angle_slider.blockSignals(False)
        self.angle_input.blockSignals(True)
        self.angle_input.setValue(new)
        self.angle_input.blockSignals(False)

    def _reset_all(self):
        self.q = pin.neutral(self.model)
        self.selected_line_id = None
        self.selection_label.setText("Reset. Click a fold line to select.")
        self.angle_label.setText("--")
        self._update_joints_display()

    def closeEvent(self, event):
        self.timer.stop()
        # 清理临时 URDF
        try:
            os.unlink(self._tmp_urdf)
        except:
            pass
        event.accept()