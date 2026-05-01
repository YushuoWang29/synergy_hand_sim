# src/interactive/origami_simulator.py

import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # 必须在import pyplot之前

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PyQt5 import QtWidgets, QtCore, QtGui
from typing import Dict, Optional
import time

from src.models.origami_design import OrigamiHandDesign, FoldType
from src.models.origami_kinematics import OrigamiForwardKinematics
from src.visualization.origami_visualizer import OrigamiVisualizer
from src.interactive.cad_viewer import CADViewer


class OrigamiSimulator(QtWidgets.QMainWindow):
    """折纸手交互式仿真器"""
    
    def __init__(self, design: OrigamiHandDesign):
        super().__init__()
        
        self.design = design
        self.fk = OrigamiForwardKinematics(design)
        
        # 当前关节角度
        self.joint_angles: Dict[int, float] = {}
        for joint in design.joints:
            self.joint_angles[joint.id] = 0.0
        
        # 选中的折痕
        self.selected_line_id: Optional[int] = None
        
        # 折痕→关节映射
        self._line_to_joint_map: Dict[int, int] = {}
        for joint in design.joints:
            self._line_to_joint_map[joint.fold_line_id] = joint.id
        
        # 3D可视化器
        self.viz_3d = OrigamiVisualizer()
        
        # 初始化UI
        self._init_ui()
        self._init_3d_view()
        
        # 定时刷新3D视图
        self._last_angles = {}
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_3d)
        self.timer.start(50)
    
    def _init_ui(self):
        """初始化界面"""
        self.setWindowTitle("Origami Hand Simulator")
        self.setMinimumSize(1200, 700)
        
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        
        # 左侧：CAD面板
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
        
        # 选中信息
        self.selection_label = QtWidgets.QLabel("Click on a red/blue fold line to select it")
        self.selection_label.setWordWrap(True)
        self.selection_label.setStyleSheet(
            "padding: 10px; background: #e8f5e9; border: 1px solid #4caf50; "
            "border-radius: 5px; font-size: 12px;"
        )
        right_layout.addWidget(self.selection_label)
        
        # 角度控制组
        angle_group = QtWidgets.QGroupBox("Joint Angle Control")
        angle_layout = QtWidgets.QVBoxLayout(angle_group)
        
        self.angle_label = QtWidgets.QLabel("Select a fold line first")
        self.angle_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.angle_label.setAlignment(QtCore.Qt.AlignCenter)
        angle_layout.addWidget(self.angle_label)
        
        # 滑块
        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(QtWidgets.QLabel("-90°"))
        self.angle_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.angle_slider.setMinimum(-90)
        self.angle_slider.setMaximum(90)
        self.angle_slider.setValue(0)
        self.angle_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.angle_slider.setTickInterval(15)
        self.angle_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.angle_slider)
        slider_layout.addWidget(QtWidgets.QLabel("+90°"))
        angle_layout.addLayout(slider_layout)
        
        # 精确输入
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
        self.joints_text.setMaximumHeight(300)
        self.joints_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        joints_layout.addWidget(self.joints_text)
        
        reset_btn = QtWidgets.QPushButton("Reset All (R)")
        reset_btn.clicked.connect(self._reset_all)
        joints_layout.addWidget(reset_btn)
        right_layout.addWidget(joints_group)
        
        # 快捷键提示
        shortcuts = QtWidgets.QLabel(
            "Keys:\n"
            "  ↑↓ : ±5°\n"
            "  ←→ : ±1°\n"
            "  Enter : Apply input\n"
            "  R : Reset selected\n"
            "  Esc : Deselect"
        )
        shortcuts.setStyleSheet(
            "padding: 8px; background: #fff3e0; border-radius: 5px; font-size: 11px;"
        )
        right_layout.addWidget(shortcuts)
        right_layout.addStretch()
        main_layout.addWidget(right_panel, stretch=0)
        
        # 键盘事件
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas.keyPressEvent = self._on_key_press
        
        self._update_joints_display()
    
    def _init_3d_view(self):
        """初始化3D视图"""
        self.viz_3d.open_browser()
        time.sleep(1.5)
        self.viz_3d.display_hand(self.fk, self.joint_angles)
        self._last_angles = self.joint_angles.copy()
        
    
    def _on_line_selected(self, line_id, fold_type):
        """选中折痕回调"""
        self.selected_line_id = line_id
        joint_id = self._line_to_joint_map.get(line_id)
        
        if joint_id is not None:
            angle_deg = np.degrees(self.joint_angles[joint_id])
            type_str = "MOUNTAIN" if fold_type == FoldType.MOUNTAIN else "VALLEY"
            self.selection_label.setText(
                f"<b>Line {line_id}</b> ({type_str}) → <b>Joint {joint_id}</b> | "
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
    
    def _set_angle(self, angle_deg):
        """设置当前选中关节的角度"""
        if self.selected_line_id is None:
            return
        joint_id = self._line_to_joint_map.get(self.selected_line_id)
        if joint_id is None:
            return
        
        self.joint_angles[joint_id] = np.radians(angle_deg)
        self.angle_label.setText(f"{angle_deg:.1f}°")
        self._update_joints_display()
    
    def _update_joints_display(self):
        lines = []
        for j in self.design.joints:
            deg = np.degrees(self.joint_angles.get(j.id, 0))
            fl = self.design.fold_lines[j.fold_line_id]
            sym = "▲" if fl.fold_type == FoldType.MOUNTAIN else "▼"
            marker = " ◀" if j.fold_line_id == self.selected_line_id else ""
            lines.append(f"{sym} J{j.id} | L{j.fold_line_id} | {deg:+6.1f}°{marker}")
        self.joints_text.setText("\n".join(lines))
    
    def _update_3d(self):
        if self.joint_angles != self._last_angles:
            self.viz_3d.display_hand(self.fk, self.joint_angles)
            self._last_angles = self.joint_angles.copy()
    
    def _on_key_press(self, event):
        if self.selected_line_id is None:
            return
        
        joint_id = self._line_to_joint_map.get(self.selected_line_id)
        if joint_id is None:
            return
        
        current = np.degrees(self.joint_angles.get(joint_id, 0))
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
        for jid in self.joint_angles:
            self.joint_angles[jid] = 0.0
        self.selected_line_id = None
        self.selection_label.setText("Reset. Click a fold line to select.")
        self.angle_label.setText("--")
        self._update_joints_display()
    
    def closeEvent(self, event):
        self.timer.stop()
        event.accept()