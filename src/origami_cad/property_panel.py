from PyQt5 import QtWidgets, QtCore
from src.models.origami_design import FoldType, FoldLine

class PropertyPanel(QtWidgets.QWidget):
    stiffness_changed = QtCore.pyqtSignal(float)
    actuator_type_changed = QtCore.pyqtSignal(int)   # 0 or 1
    pulley_friction_changed = QtCore.pyqtSignal(float)
    pulley_radius_changed = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("属性"))
        self.info_label = QtWidgets.QLabel("未选中")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # 线段刚度编辑
        self.stiff_widget = QtWidgets.QWidget()
        stiff_l = QtWidgets.QHBoxLayout(self.stiff_widget)
        stiff_l.addWidget(QtWidgets.QLabel("刚度 (N·m/rad):"))
        self.stiff_spin = QtWidgets.QDoubleSpinBox()
        self.stiff_spin.setRange(0.0, 1000.0)
        self.stiff_spin.valueChanged.connect(self.stiffness_changed.emit)
        stiff_l.addWidget(self.stiff_spin)
        layout.addWidget(self.stiff_widget)

        # 驱动器类型编辑
        self.act_widget = QtWidgets.QWidget()
        act_l = QtWidgets.QHBoxLayout(self.act_widget)
        act_l.addWidget(QtWidgets.QLabel("类型:"))
        self.act_type_combo = QtWidgets.QComboBox()
        self.act_type_combo.addItems(["A (蓝色)", "B (红色)"])
        self.act_type_combo.currentIndexChanged.connect(self.actuator_type_changed.emit)
        act_l.addWidget(self.act_type_combo)
        layout.addWidget(self.act_widget)

        # 滑轮属性编辑
        self.pulley_widget = QtWidgets.QWidget()
        pulley_l = QtWidgets.QFormLayout(self.pulley_widget)
        self.friction_spin = QtWidgets.QDoubleSpinBox()
        self.friction_spin.setRange(0.0, 10.0)
        self.friction_spin.setDecimals(3)
        self.friction_spin.valueChanged.connect(self.pulley_friction_changed.emit)
        pulley_l.addRow("摩擦系数:", self.friction_spin)

        self.radius_spin = QtWidgets.QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 100.0)
        self.radius_spin.setDecimals(2)
        self.radius_spin.valueChanged.connect(self.pulley_radius_changed.emit)
        pulley_l.addRow("半径 (mm):", self.radius_spin)

        self.pulley_fold_label = QtWidgets.QLabel("无")
        pulley_l.addRow("关联折痕:", self.pulley_fold_label)
        layout.addWidget(self.pulley_widget)

        # 默认隐藏
        self.stiff_widget.hide()
        self.act_widget.hide()
        self.pulley_widget.hide()
        layout.addStretch()

    def update_for_item(self, item_type: str, data=None):
        """根据选中项类型更新面板"""
        self.stiff_widget.hide()
        self.act_widget.hide()
        self.pulley_widget.hide()
        if item_type == 'line' and data:
            self.info_label.setText(f"线段 ID: {data.id}\n类型: {data.fold_type.name}")
            self.stiff_spin.blockSignals(True)
            self.stiff_spin.setValue(data.stiffness)
            self.stiff_spin.setEnabled(data.fold_type != FoldType.OUTLINE)
            self.stiff_spin.blockSignals(False)
            self.stiff_widget.show()
        elif item_type == 'actuator':
            self.info_label.setText(f"驱动器 ({'A' if data==0 else 'B'})")
            self.act_type_combo.blockSignals(True)
            self.act_type_combo.setCurrentIndex(data)
            self.act_type_combo.blockSignals(False)
            self.act_widget.show()
        elif item_type == 'pulley':
            # data 是 (Pulley 对象, 折痕信息)
            pulley_obj, fold_info = data
            self.info_label.setText(f"滑轮 ID: {pulley_obj.id}")
            self.friction_spin.blockSignals(True)
            self.friction_spin.setValue(pulley_obj.friction_coefficient)
            self.friction_spin.blockSignals(False)
            self.radius_spin.blockSignals(True)
            self.radius_spin.setValue(pulley_obj.radius)
            self.radius_spin.blockSignals(False)
            self.pulley_fold_label.setText(
                f"折痕 {pulley_obj.attached_fold_line_id}" 
                if pulley_obj.attached_fold_line_id is not None else "无")
            self.pulley_widget.show()
        else:
            self.info_label.setText("未选中")
