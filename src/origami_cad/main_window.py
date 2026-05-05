# src/origami_cad/main_window.py
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from .cad_graphics_scene import CadGraphicsScene, DrawMode
from .cad_graphics_view import CadGraphicsView
from .property_panel import PropertyPanel
from src.models.origami_design import OrigamiHandDesign, FoldLine, Point2D, FoldType, Pulley, Tendon

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Origami Hand CAD")
        self.setMinimumSize(1200, 800)

        self.design = OrigamiHandDesign(name="untitled")

        self.scene = CadGraphicsScene(self)
        self.view = CadGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)

        # 属性面板
        self.property_panel = PropertyPanel()
        dock = QtWidgets.QDockWidget("属性", self)
        dock.setWidget(self.property_panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        self._create_toolbar()
        self._create_menus()

        # 信号连接
        self.scene.line_selected.connect(self._on_line_selected)
        self.scene.pulley_selected.connect(self._on_pulley_selected)
        self.scene.actuator_selected.connect(self._on_actuator_selected)
        self.property_panel.stiffness_changed.connect(self._on_stiffness_changed)
        self.property_panel.actuator_type_changed.connect(self._on_actuator_type_changed)
        self.property_panel.pulley_friction_changed.connect(self._on_pulley_friction_changed)
        self.property_panel.pulley_radius_changed.connect(self._on_pulley_radius_changed)

        # 存储当前选中的图元
        self._selected_actuator_item = None
        self._selected_pulley_id = None

    def _create_toolbar(self):
        toolbar = self.addToolBar("工具")
        # 使用 QActionGroup 保证互斥
        self.tool_group = QtWidgets.QActionGroup(self)
        self.tool_group.setExclusive(True)

        def add_tool(name, mode, shortcut=None):
            action = toolbar.addAction(name)
            action.setCheckable(True)
            action.setData(mode)
            if shortcut:
                action.setShortcut(shortcut)
            self.tool_group.addAction(action)
            action.triggered.connect(lambda checked, m=mode: self._set_mode(m) if checked else None)
            return action

        self.select_action = add_tool("选择", DrawMode.SELECT, None)
        self.outline_action = add_tool("轮廓线", DrawMode.DRAW_OUTLINE, None)
        self.mountain_action = add_tool("峰折", DrawMode.DRAW_MOUNTAIN, None)
        self.valley_action = add_tool("谷折", DrawMode.DRAW_VALLEY, None)
        self.pulley_action = add_tool("滑轮", DrawMode.PLACE_PULLEY, None)
        self.tendon_action = add_tool("腱绳", DrawMode.DRAW_TENDON, None)
        self.actuator_action = add_tool("驱动器", DrawMode.PLACE_ACTUATOR, None)

        self.select_action.setChecked(True)

        toolbar.addSeparator()
        delete_action = toolbar.addAction("删除选中")
        delete_action.triggered.connect(self._delete_selected)

    def _set_mode(self, mode: DrawMode):
        self.scene.set_draw_mode(mode)
        # 确保视图获取焦点以便接收键盘
        self.view.setFocus()
        # 对于需要坐标输入的绘图模式，启动命令模式浮动输入框
        key_map = {
            DrawMode.DRAW_OUTLINE: Qt.Key_L,
            DrawMode.DRAW_MOUNTAIN: Qt.Key_M,
            DrawMode.DRAW_VALLEY: Qt.Key_V,
            DrawMode.PLACE_PULLEY: Qt.Key_P,
            DrawMode.PLACE_ACTUATOR: Qt.Key_A,
        }
        if mode in key_map:
            self.scene._start_command_mode(key_map[mode])

    def _delete_selected(self):
        # 调用场景的删除方法
        self.scene._delete_selected_lines()

    def _on_line_selected(self, line_id: int):
        fl = self.scene.get_line_by_id(line_id)
        if fl:
            self.property_panel.update_for_item('line', fl)
        else:
            self.property_panel.update_for_item('none')

    def _on_pulley_selected(self, pid: int):
        # 从场景中获取 pulley 对象
        pulley_obj = None
        for item, p in self.scene._pulleys.items():
            if p.id == pid:
                pulley_obj = p
                break
        if pulley_obj:
            # 获取关联折痕信息（从场景图元中查找）
            fold_info = None
            if pulley_obj.attached_fold_line_id is not None:
                fold_info = self.scene.get_line_by_id(pulley_obj.attached_fold_line_id)
            self.property_panel.update_for_item('pulley', (pulley_obj, fold_info))
            # 存储当前选中的滑轮ID，供后续修改
            self._selected_pulley_id = pid
        else:
            self.property_panel.update_for_item('none')
            self._selected_pulley_id = None

    def _on_actuator_selected(self, atype: int):
        self.property_panel.update_for_item('actuator', atype)
        # 记住选中的驱动器图元（需要遍历找出）
        for item in self.scene.selectedItems():
            if isinstance(item, QtWidgets.QGraphicsEllipseItem) and item in self.scene._actuator_graphics:
                self._selected_actuator_item = item
                break

    def _on_stiffness_changed(self, value: float):
        selected = self.scene.selectedItems()
        for item in selected:
            if isinstance(item, QtWidgets.QGraphicsLineItem) and item in self.scene._lines:
                self.scene._lines[item].stiffness = value

    def _on_actuator_type_changed(self, atype: int):
        """属性面板驱动器类型变化时修改场景中的驱动器"""
        if self._selected_actuator_item is not None:
            self.scene.set_actuator_type(self._selected_actuator_item, atype)
        else:
            # 回退：从 selectedItems 查找
            for item in self.scene.selectedItems():
                if isinstance(item, QtWidgets.QGraphicsEllipseItem) and item in self.scene._actuator_graphics:
                    self.scene.set_actuator_type(item, atype)
                    self._selected_actuator_item = item
                    break

    def _on_pulley_friction_changed(self, value: float):
        if self._selected_pulley_id is not None:
            self.scene.set_pulley_properties(self._selected_pulley_id, friction=value, radius=None)

    def _on_pulley_radius_changed(self, value: float):
        if self._selected_pulley_id is not None:
            self.scene.set_pulley_properties(self._selected_pulley_id, friction=None, radius=value)

    def _create_menus(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("文件")
        file_menu.addAction("新建", self._new_file)
        file_menu.addAction("打开 .ohd", self._open_file)
        file_menu.addAction("保存 .ohd", self._save_file)
        file_menu.addSeparator()
        file_menu.addAction("导出 DXF", self._export_dxf)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)

    def _new_file(self):
        self.design = OrigamiHandDesign(name="untitled")
        self.scene.clear_lines()
        # 清除滑轮、腱绳、驱动器等
        self.scene._pulleys.clear()
        self.scene._tendons.clear()
        self.scene._actuator_graphics.clear()
        self.scene._tendon_graphics.clear()
        self.scene._selected_actuator_item = None

    def _open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开设计", "", "Origami Hand Design (*.ohd)")
        if not path:
            return
        try:
            self.design = OrigamiHandDesign.load(path)
            self.scene.clear_lines()
            self.scene.load_lines(list(self.design.fold_lines.values()))
            # 加载滑轮、驱动器、腱绳（注意顺序：腱绳依赖前两者已加载）
            self.scene.load_pulleys(self.design.pulleys)
            self.scene.load_actuators(self.design.actuator_positions)
            self.scene.load_tendons(self.design.tendons)
            if self.design.fold_lines:
                self.scene._next_line_id = max(self.design.fold_lines.keys(), default=-1) + 1
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"无法打开: {e}")

    def _save_file(self):
        # 从场景更新设计
        self.design.fold_lines.clear()
        for fl in self.scene.get_all_fold_lines():
            self.design.add_fold_line(fl)
        # 滑轮
        self.design.pulleys.clear()
        for item, pulley in self.scene._pulleys.items():
            self.design.pulleys[pulley.id] = pulley
        # 肌腱
        self.design.tendons = self.scene._tendons
        # 驱动器位置
        act_list = []
        for item, (atype, pos) in self.scene._actuator_graphics.items():
            act_list.append({"type": atype, "x": pos.x(), "y": pos.y()})
        self.design.actuator_positions = act_list

        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存设计", "", "Origami Hand Design (*.ohd)")
        if path:
            self.design.save(path)

    def _export_dxf(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出 DXF", "", "DXF Files (*.dxf)")
        if not path:
            return
        import ezdxf
        doc = ezdxf.new()
        msp = doc.modelspace()
        for fl in self.scene.get_all_fold_lines():
            color = {FoldType.OUTLINE: 7, FoldType.MOUNTAIN: 1, FoldType.VALLEY: 5}.get(fl.fold_type, 7)
            msp.add_line((fl.start.x, fl.start.y), (fl.end.x, fl.end.y), dxfattribs={'color': color})
        doc.saveas(path)
