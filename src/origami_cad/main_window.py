# src/origami_cad/main_window.py
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer
from .cad_graphics_scene import CadGraphicsScene, DrawMode
from .cad_graphics_view import CadGraphicsView
from .property_panel import PropertyPanel
from src.models.origami_design import OrigamiHandDesign, FoldLine, Point2D, FoldType, Pulley, Tendon, HOLE_ID_OFFSET, Damper

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
        self._property_dock = QtWidgets.QDockWidget("属性", self)
        self._property_dock.setWidget(self.property_panel)
        self._property_dock.setObjectName("propertyDock")
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._property_dock)

        self._create_toolbar()
        self._create_menus()

        # 状态栏
        self.statusBar().showMessage("准备就绪")

        # 信号连接
        self.scene.line_selected.connect(self._on_line_selected)
        self.scene.pulley_selected.connect(self._on_pulley_selected)
        self.scene.hole_selected.connect(self._on_hole_selected)
        self.scene.actuator_selected.connect(self._on_actuator_selected)
        self.scene.damper_selected.connect(self._on_damper_selected)
        self.scene.draw_mode_changed.connect(self._on_draw_mode_changed)
        self.scene.mouse_moved.connect(self._on_mouse_moved)
        self.property_panel.stiffness_changed.connect(self._on_stiffness_changed)
        self.property_panel.actuator_type_changed.connect(self._on_actuator_type_changed)
        self.property_panel.pulley_friction_changed.connect(self._on_pulley_friction_changed)
        self.property_panel.pulley_radius_changed.connect(self._on_pulley_radius_changed)
        self.property_panel.hole_plate_offset_changed.connect(self._on_hole_plate_offset_changed)
        self.property_panel.hole_friction_changed.connect(self._on_hole_friction_changed)
        self.property_panel.damper_damping_changed.connect(self._on_damper_damping_changed)

        # 存储当前选中的图元
        self._selected_actuator_item = None
        self._selected_pulley_id = None
        self._selected_hole_id = None

    # ========== 状态栏回调 ==========
    def _on_draw_mode_changed(self, mode: DrawMode):
        """绘图模式改变时更新状态栏"""
        mode_names = {
            DrawMode.SELECT: "选择模式",
            DrawMode.DRAW_OUTLINE: "绘制轮廓线",
            DrawMode.DRAW_MOUNTAIN: "绘制峰折线",
            DrawMode.DRAW_VALLEY: "绘制谷折线",
            DrawMode.PLACE_PULLEY: "放置滑轮",
            DrawMode.PLACE_HOLE: "放置孔",
            DrawMode.PLACE_DAMPER: "放置阻尼器",
            DrawMode.DRAW_TENDON: "绘制腱绳",
            DrawMode.PLACE_ACTUATOR: "放置驱动器",
        }
        name = mode_names.get(mode, "未知模式")
        self.statusBar().showMessage(f"模式: {name}")

    def _on_mouse_moved(self, x: float, y: float):
        """鼠标移动时更新状态栏坐标"""
        # 保留当前模式文本
        mode_names = {
            DrawMode.SELECT: "选择",
            DrawMode.DRAW_OUTLINE: "轮廓",
            DrawMode.DRAW_MOUNTAIN: "峰折",
            DrawMode.DRAW_VALLEY: "谷折",
            DrawMode.PLACE_PULLEY: "滑轮",
            DrawMode.PLACE_HOLE: "孔",
            DrawMode.PLACE_DAMPER: "阻尼器",
            DrawMode.DRAW_TENDON: "腱绳",
            DrawMode.PLACE_ACTUATOR: "驱动器",
        }
        mode_name = mode_names.get(self.scene.draw_mode, "未知")
        self.statusBar().showMessage(f"模式: {mode_name}  |  坐标: ({x:.1f}, {y:.1f})")

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

        self.select_action = add_tool("选择 [Esc]", DrawMode.SELECT, Qt.Key_Escape)
        self.outline_action = add_tool("轮廓线 [L]", DrawMode.DRAW_OUTLINE, Qt.Key_L)
        self.mountain_action = add_tool("峰折 [M]", DrawMode.DRAW_MOUNTAIN, Qt.Key_M)
        self.valley_action = add_tool("谷折 [V]", DrawMode.DRAW_VALLEY, Qt.Key_V)
        self.pulley_action = add_tool("滑轮 [P]", DrawMode.PLACE_PULLEY, Qt.Key_P)
        self.hole_action = add_tool("孔(橙色) [H]", DrawMode.PLACE_HOLE, Qt.Key_H)
        self.damper_action = add_tool("阻尼器(浅绿) [D]", DrawMode.PLACE_DAMPER, Qt.Key_D)
        self.tendon_action = add_tool("腱绳 [T]", DrawMode.DRAW_TENDON, Qt.Key_T)
        self.actuator_action = add_tool("驱动器 [A]", DrawMode.PLACE_ACTUATOR, Qt.Key_A)

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

    def _on_hole_selected(self, hid: int):
        """孔被选中时的处理"""
        hole_obj = None
        for item, h in self.scene._holes.items():
            if h.id == hid:
                hole_obj = h
                break
        if hole_obj:
            fold_info = None
            if hole_obj.attached_fold_line_id is not None:
                fold_info = self.scene.get_line_by_id(hole_obj.attached_fold_line_id)
            self.property_panel.update_for_item('hole', (hole_obj, fold_info))
            self._selected_hole_id = hid
        else:
            self.property_panel.update_for_item('none')
            self._selected_hole_id = None

    def _on_hole_plate_offset_changed(self, value: float):
        if self._selected_hole_id is not None:
            for item, h in self.scene._holes.items():
                if h.id == self._selected_hole_id:
                    h.plate_offset = value
                    break

    def _on_hole_friction_changed(self, value: float):
        if self._selected_hole_id is not None:
            for item, h in self.scene._holes.items():
                if h.id == self._selected_hole_id:
                    h.friction_coefficient = value
                    break

    def _on_actuator_selected(self, atype: int):
        self.property_panel.update_for_item('actuator', atype)
        # 记住选中的驱动器图元（需要遍历找出）
        for item in self.scene.selectedItems():
            if isinstance(item, QtWidgets.QGraphicsEllipseItem) and item in self.scene._actuator_graphics:
                self._selected_actuator_item = item
                break

    def _on_damper_selected(self, did: int):
        """阻尼器被选中时的处理"""
        damper_obj = None
        for item, d in self.scene._dampers.items():
            if d.id == did:
                damper_obj = d
                break
        if damper_obj:
            fold_infos = []
            for fid in damper_obj.attached_fold_line_ids:
                fl = self.scene.get_line_by_id(fid)
                if fl:
                    fold_infos.append(fl)
            self.property_panel.update_for_item('damper', (damper_obj, fold_infos))
        else:
            self.property_panel.update_for_item('none')

    def _on_damper_damping_changed(self, value: float):
        """阻尼器阻尼系数变化时更新场景中的阻尼器"""
        for item in self.scene.selectedItems():
            if isinstance(item, QtWidgets.QGraphicsEllipseItem) and item in self.scene._dampers:
                self.scene._dampers[item].damping_coefficient = value
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

        # 视图菜单 - 用于开关属性面板等
        view_menu = menu.addMenu("视图")
        self._toggle_property_action = view_menu.addAction("属性")
        self._toggle_property_action.setCheckable(True)
        self._toggle_property_action.setChecked(True)
        self._toggle_property_action.triggered.connect(self._toggle_property_panel)

    def _toggle_property_panel(self, visible: bool):
        """切换属性面板的显示/隐藏"""
        self._property_dock.setVisible(visible)
        self._toggle_property_action.setChecked(visible)

    def _new_file(self):
        self.design = OrigamiHandDesign(name="untitled")
        self.scene.clear_all()

    def _open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开设计", "", "Origami Hand Design (*.ohd)")
        if not path:
            return
        try:
            self.design = OrigamiHandDesign.load(path)
            # 清除前一个文件的所有图元（包括折痕线、滑轮、孔、阻尼器、驱动器、腱绳等）
            self.scene.clear_all()
            self._selected_actuator_item = None
            # 加载新文件的内容
            # 注意顺序：阻尼器必须在腱绳之前加载，
            # 因为 load_tendons 的路径后处理需要按 ID 查找所有图元
            self.scene.load_lines(list(self.design.fold_lines.values()))
            self.scene.load_pulleys(self.design.pulleys)
            self.scene.load_holes(self.design.holes)
            self.scene.load_actuators(self.design.actuator_positions)
            self.scene.load_dampers(self.design.dampers)
            self.scene.load_tendons(self.design.tendons)
            if self.design.fold_lines:
                self.scene._next_line_id = max(self.design.fold_lines.keys(), default=-1) + 1
            if self.design.holes:
                self.scene._hole_id = min(self.design.holes.keys(), default=HOLE_ID_OFFSET) - 1
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
        # 孔
        self.design.holes.clear()
        for item, hole in self.scene._holes.items():
            self.design.holes[hole.id] = hole
        # 阻尼器
        self.design.dampers.clear()
        for item, damper in self.scene._dampers.items():
            self.design.dampers[damper.id] = damper
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
