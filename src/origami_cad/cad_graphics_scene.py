# src/origami_cad/cad_graphics_scene.py
"""
重制版 QGraphicsScene：支持命令行/鼠标混合绘制，互斥工具栏，驱动器，腱绳端点保存。
"""

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QLineF, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QPen, QPainter, QColor, QBrush, QPainterPath, QPainterPathStroker
from PyQt5.QtWidgets import (QGraphicsScene, QGraphicsLineItem,
                             QGraphicsEllipseItem, QGraphicsProxyWidget, QLineEdit,
                             QGraphicsItem, QGraphicsRectItem)
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple
from src.models.origami_design import (FoldLine, Point2D, FoldType, Pulley, Tendon, Hole, Damper,
                                        is_hole_id, is_actuator_id, is_damper_id,
                                        HOLE_ID_OFFSET, DAMPER_ID_OFFSET)

class DrawMode(Enum):
    SELECT = auto()
    DRAW_OUTLINE = auto()
    DRAW_MOUNTAIN = auto()
    DRAW_VALLEY = auto()
    PLACE_PULLEY = auto()
    DRAW_TENDON = auto()
    PLACE_ACTUATOR = auto()
    PLACE_HOLE = auto()
    PLACE_DAMPER = auto()

# 键盘映射到绘线类型
KEY_TO_LINE_TYPE = {
    Qt.Key_L: FoldType.OUTLINE,
    Qt.Key_M: FoldType.MOUNTAIN,
    Qt.Key_V: FoldType.VALLEY,
}
KEY_TO_PULLEY_MODE = Qt.Key_P
KEY_TO_TENDON_MODE = Qt.Key_T
KEY_TO_ACTUATOR_MODE = Qt.Key_A
KEY_TO_DAMPER_MODE = Qt.Key_D

class ClickableLineItem(QGraphicsLineItem):
    """重写 shape() 使点击判定范围变宽，方便选中"""
    def shape(self):
        path = QPainterPath()
        path.moveTo(self.line().p1())
        path.lineTo(self.line().p2())
        pen = QPen(Qt.black, 8)  # 8px 宽的点击判定范围
        return QPainterPathStroker(pen).createStroke(path)

class CadGraphicsScene(QGraphicsScene):
    line_added = pyqtSignal(FoldLine)
    line_selected = pyqtSignal(int)
    line_deleted = pyqtSignal(int)
    pulley_selected = pyqtSignal(int)    # pulley id
    hole_selected = pyqtSignal(int)      # hole id
    actuator_selected = pyqtSignal(int)  # 0 or 1 (type)
    damper_selected = pyqtSignal(int)    # damper id
    mouse_moved = pyqtSignal(float, float)    # x, y
    draw_mode_changed = pyqtSignal(DrawMode)  # new draw mode

    def __init__(self, parent=None):
        super().__init__(parent)
        self.draw_mode = DrawMode.SELECT
        self._next_line_id = 0
        self._pulley_id = 0
        self._tendon_id = 0
        self._actuator_id = 0
        self._hole_id = HOLE_ID_OFFSET  # 孔从 -100 开始递减

        # 图元映射
        self._lines: Dict[QtWidgets.QGraphicsLineItem, FoldLine] = {}
        self._pulleys: Dict[QtWidgets.QGraphicsEllipseItem, Pulley] = {}
        self._holes: Dict[QtWidgets.QGraphicsEllipseItem, Hole] = {}
        self._actuator_graphics: Dict[QtWidgets.QGraphicsEllipseItem, Tuple[int, QPointF]] = {}  # item -> (type, pos)
        self._tendons: Dict[int, Tendon] = {}
        self._tendon_graphics: Dict[int, List[QtWidgets.QGraphicsLineItem]] = {}  # tendon_id -> 永久线列表

        # 阻尼器
        self._dampers: Dict[QtWidgets.QGraphicsEllipseItem, Damper] = {}

        # 自由绘图（鼠标点击）临时状态
        self._start_point = None
        self._temp_line = None

        # 命令行/混合模式状态
        self._cmd_active = False         # 是否处于命令模式（按L/M/V/P/T/A后）
        self._cmd_draw_mode = None       # 当前命令对应的 DrawMode
        self._cmd_line_type = None       # 如果是画线，保存 FoldType
        self._cmd_start_point = None
        self._cmd_temp_line = None
        self._cmd_input = None
        self._cmd_proxy = None

        # 腱绳绘制临时
        self._tendon_temp_path = []      # QGraphicsEllipseItem 列表（滑轮或驱动器）
        self._tendon_temp_lines = []

        # 吸附距离
        self.snap_distance = 10.0
        self.setSceneRect(-2000, -2000, 4000, 4000)

    # ========== 公共方法 ==========
    def set_draw_mode(self, mode: DrawMode):
        self._exit_command_mode()
        self._clear_free_drawing()
        self._cancel_tendon()
        self.draw_mode = mode
        self.draw_mode_changed.emit(mode)

    def get_all_fold_lines(self) -> List[FoldLine]:
        return list(self._lines.values())

    def clear_lines(self):
        for item in list(self._lines.keys()):
            self.removeItem(item)
        self._lines.clear()
        self._next_line_id = 0

    def clear_all(self):
        """清空场景中的所有图元（折痕线、滑轮、孔、阻尼器、驱动器、腱绳），
        确保从 QGraphicsScene 中移除每个 item，避免打开新文件时旧图元残留在场景中。"""
        # 清除折痕线
        self.clear_lines()
        # 清除滑轮（先 removeItem 再 clear dict）
        for item in list(self._pulleys.keys()):
            self.removeItem(item)
        self._pulleys.clear()
        # 清除孔
        for item in list(self._holes.keys()):
            self.removeItem(item)
        self._holes.clear()
        # 清除阻尼器
        for item in list(self._dampers.keys()):
            self.removeItem(item)
        self._dampers.clear()
        # 清除驱动器
        for item in list(self._actuator_graphics.keys()):
            self.removeItem(item)
        self._actuator_graphics.clear()
        # 清除腱绳永久线段
        for lines in self._tendon_graphics.values():
            for line in lines:
                self.removeItem(line)
        self._tendon_graphics.clear()
        self._tendons.clear()
        # 清除腱绳临时线段
        for line in self._tendon_temp_lines:
            self.removeItem(line)
        self._tendon_temp_lines.clear()
        self._tendon_temp_path.clear()
        # 清除自由绘图临时线
        self._clear_free_drawing()
        # 退出命令模式
        self._exit_command_mode()
        # 重置 ID 计数器
        self._next_line_id = 0
        self._pulley_id = 0
        self._hole_id = HOLE_ID_OFFSET


    def load_lines(self, fold_lines: List[FoldLine]):
        self.clear_lines()
        for fl in fold_lines:
            self.add_line_item(fl)

    def load_pulleys(self, pulleys: Dict[int, Pulley]):
        for pid, pulley in pulleys.items():
            # 创建图元并添加到 _pulleys
            rad = 5
            pos = pulley.position
            ellipse = QtWidgets.QGraphicsEllipseItem(pos.x - rad, pos.y - rad, rad*2, rad*2)
            ellipse.setBrush(QColor(128,128,128))
            ellipse.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.addItem(ellipse)
            self._pulleys[ellipse] = pulley
            self._pulley_id = max(self._pulley_id, pid + 1)

    def load_holes(self, holes: Dict[int, Hole]):
        """从数据加载孔图元"""
        for hid, hole in holes.items():
            rad = 3
            pos = hole.position
            ellipse = QtWidgets.QGraphicsEllipseItem(pos.x - rad, pos.y - rad, rad*2, rad*2)
            ellipse.setBrush(QColor(255, 140, 0))  # 橙色
            ellipse.setPen(QPen(Qt.darkYellow, 1))
            ellipse.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.addItem(ellipse)
            self._holes[ellipse] = hole
            self._hole_id = min(self._hole_id, hid - 1)

    def load_tendons(self, tendons: Dict[int, Tendon]):
        """从数据重建腱绳图元（需在 load_pulleys 和 load_actuators 之后调用）"""
        # 清除旧的腱绳图元
        for lines in self._tendon_graphics.values():
            for line in lines:
                self.removeItem(line)
        self._tendon_graphics.clear()
        self._tendons = tendons
        self._tendon_id = max(tendons.keys(), default=-1) + 1

        # 重建图元：遍历每个腱绳，根据 pulley_sequence 中的 ID 找到对应图元，画虚线连接
        for tid, tendon in self._tendons.items():
            permanent_lines = []
            prev_item = None
            for i, seq_id in enumerate(tendon.pulley_sequence):
                # 找对应的图元 item
                cur_item = self._find_graphics_item_by_id(seq_id)
                if cur_item is None and seq_id < 0 and not is_hole_id(seq_id) and not is_damper_id(seq_id):
                    # 负数且不是孔/阻尼器：驱动器，-1 -> A (type 0), -2 -> B (type 1)
                    # 有多个相同类型驱动器时，选择距离最近邻节点最近的驱动器
                    target_type = -(seq_id + 1)  # -1->0(A), -2->1(B)

                    # 找出参考位置（上一个节点，或下一个节点）
                    ref_pos = None
                    if prev_item is not None:
                        ref_pos = prev_item.sceneBoundingRect().center()
                    else:
                        # 如果是第一个节点，找路径中下一个节点的位置
                        for j in range(i + 1, len(tendon.pulley_sequence)):
                            next_id = tendon.pulley_sequence[j]
                            next_item = self._find_graphics_item_by_id(next_id)
                            if next_item is not None:
                                ref_pos = next_item.sceneBoundingRect().center()
                                break

                    # 选择距离参考位置最近的同类型驱动器
                    best_dist = float('inf')
                    best_item = None
                    for item, (atype, pos) in self._actuator_graphics.items():
                        if atype == target_type:
                            if ref_pos is not None:
                                dist = QLineF(ref_pos, item.sceneBoundingRect().center()).length()
                                if dist < best_dist:
                                    best_dist = dist
                                    best_item = item
                            else:
                                # 没有参考位置，选第一个
                                best_item = item
                                break
                    cur_item = best_item

                if cur_item is None:
                    continue
                if prev_item is not None:
                    line = QtWidgets.QGraphicsLineItem(QLineF(
                        prev_item.sceneBoundingRect().center(),
                        cur_item.sceneBoundingRect().center()))
                    line.setPen(QPen(Qt.darkGray, 1, Qt.DashLine))
                    line.setFlag(QGraphicsItem.ItemIsSelectable, True)
                    self.addItem(line)
                    permanent_lines.append(line)
                prev_item = cur_item
            if permanent_lines:
                self._tendon_graphics[tid] = permanent_lines
        
        # ===== 后处理：基于腱绳路径更新孔和阻尼器的关联折痕 =====
        # 这对从 .ohd 文件加载的场景很重要，可以修复旧文件中使用最近原则设置的孔关联折痕
        for tid, tendon in self._tendons.items():
            self._post_process_tendon_path(tendon)


    def load_actuators(self, actuator_positions: list):
        """从保存的位置列表加载驱动器图元"""
        for act in actuator_positions:
            self._place_actuator(QPointF(act['x'], act['y']), act['type'])

    # ========== 鼠标事件 ==========
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        pos = event.scenePos()
        # 如果处于命令模式，且鼠标点击在空白处，执行命令模式下的鼠标输入
        if self._cmd_active:
            self._cmd_mouse_press(pos)
            super().mousePressEvent(event)
            return

        # 正常模式处理
        if self.draw_mode == DrawMode.SELECT:
            self._select_click(pos)
            return  # 选择模式自己处理，不让super()重复处理
        elif self.draw_mode in (DrawMode.DRAW_OUTLINE, DrawMode.DRAW_MOUNTAIN, DrawMode.DRAW_VALLEY):
            self._free_draw_click(pos)
        elif self.draw_mode == DrawMode.PLACE_PULLEY:
            self._place_pulley(pos)
        elif self.draw_mode == DrawMode.DRAW_TENDON:
            self._add_tendon_point(pos)
        elif self.draw_mode == DrawMode.PLACE_ACTUATOR:
            self._place_actuator(pos, 0)  # 默认A
        elif self.draw_mode == DrawMode.PLACE_HOLE:
            self._place_hole(pos)
        elif self.draw_mode == DrawMode.PLACE_DAMPER:
            self._place_damper(pos)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.scenePos()
        # 发射鼠标坐标信号（用于状态栏显示）
        self.mouse_moved.emit(pos.x(), pos.y())
        # 移动命令模式输入框
        if self._cmd_proxy:
            self._cmd_proxy.setPos(pos)
        # 更新临时预览线
        preview_line = self._cmd_temp_line if self._cmd_active else self._temp_line
        start = self._cmd_start_point if self._cmd_active else self._start_point
        if preview_line and start is not None:
            preview_line.setLine(QLineF(start, pos))
        super().mouseMoveEvent(event)

    def contextMenuEvent(self, event):
        if self.draw_mode == DrawMode.DRAW_TENDON and self._tendon_temp_path:
            self._finish_tendon()
            return
        super().contextMenuEvent(event)

    # ========== 键盘事件（指令入口）==========
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        # 全局快捷键
        if event.key() == Qt.Key_Delete:
            self._delete_selected_lines()
            return
        if event.key() == Qt.Key_Escape:
            if self._cmd_active:
                self._exit_command_mode()
            else:
                self.set_draw_mode(DrawMode.SELECT)
                self._select_none()
            return

        # 命令模式激活：L/M/V/P/A/H/D（需要坐标输入或鼠标点击）
        cmd_keys = (Qt.Key_L, Qt.Key_M, Qt.Key_V, KEY_TO_PULLEY_MODE, KEY_TO_ACTUATOR_MODE, Qt.Key_H, KEY_TO_DAMPER_MODE)
        if not self._cmd_active and event.key() in cmd_keys:
            self._start_command_mode(event.key())
            return

        # 腱绳模式 T：无需输入框，直接切换模式用鼠标点击滑轮/驱动器
        if not self._cmd_active and event.key() == KEY_TO_TENDON_MODE:
            self.set_draw_mode(DrawMode.DRAW_TENDON)
            return

        # 如果命令模式下按回车并输入框有焦点，由输入框自己处理；否则不做额外动作
        super().keyPressEvent(event)

    # ========== 命令模式 ==========
    def _start_command_mode(self, key: Qt.Key):
        self._cmd_active = True
        self._cmd_start_point = None
        if key in KEY_TO_LINE_TYPE:
            self._cmd_draw_mode = DrawMode.DRAW_OUTLINE  # 占位，具体类型存下面
            self._cmd_line_type = KEY_TO_LINE_TYPE[key]
        elif key == KEY_TO_PULLEY_MODE:
            self._cmd_draw_mode = DrawMode.PLACE_PULLEY
            self._cmd_line_type = None
        elif key == KEY_TO_TENDON_MODE:
            self._cmd_draw_mode = DrawMode.DRAW_TENDON
            self._cmd_line_type = None
        elif key == KEY_TO_ACTUATOR_MODE:
            self._cmd_draw_mode = DrawMode.PLACE_ACTUATOR
            self._cmd_line_type = None
        elif key == Qt.Key_H:
            self._cmd_draw_mode = DrawMode.PLACE_HOLE
            self._cmd_line_type = None
        elif key == KEY_TO_DAMPER_MODE:
            self._cmd_draw_mode = DrawMode.PLACE_DAMPER
            self._cmd_line_type = None
        else:
            return

        # 创建浮动输入框
        self._cmd_input = QLineEdit()
        prefix = {Qt.Key_L: 'L', Qt.Key_M: 'M', Qt.Key_V: 'V',
                  Qt.Key_P: 'P', Qt.Key_T: 'T', Qt.Key_A: 'A', Qt.Key_H: 'H',
                  KEY_TO_DAMPER_MODE: 'D'}.get(key, '?')
        self._cmd_input.setPlaceholderText(f"{prefix}: x,y")
        self._cmd_input.returnPressed.connect(self._on_cmd_return)
        self._cmd_input.setMaximumWidth(150)
        self._cmd_proxy = self.addWidget(self._cmd_input)
        # 输入框保持固定显示大小，不受缩放影响
        self._cmd_proxy.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self._cmd_proxy.setPos(0, 0)
        self._cmd_input.setFocus()
        if self.views():
            self.views()[0].setCursor(Qt.CrossCursor)

    def _exit_command_mode(self):
        # 清理命令模式资源
        if self._cmd_temp_line:
            self.removeItem(self._cmd_temp_line)
            self._cmd_temp_line = None
        if self._cmd_proxy:
            self.removeItem(self._cmd_proxy)
            self._cmd_proxy = None
        self._cmd_active = False
        self._cmd_draw_mode = None
        self._cmd_line_type = None
        self._cmd_start_point = None
        if self.views():
            self.views()[0].setCursor(Qt.ArrowCursor)

    def _on_cmd_return(self):
        """命令模式下输入框回车：解析坐标并执行动作"""
        text = self._cmd_input.text().strip()
        if not text:
            return
        try:
            x, y = map(float, text.split(','))
        except ValueError:
            self._cmd_input.clear()
            return

        pt = QPointF(x, y)
        # 根据当前命令模式执行相应操作
        if self._cmd_draw_mode == DrawMode.PLACE_ACTUATOR:
            self._place_actuator(pt, atype=0)  # 默认A
        elif self._cmd_draw_mode in (DrawMode.DRAW_OUTLINE, DrawMode.DRAW_MOUNTAIN, DrawMode.DRAW_VALLEY):
            self._cmd_line_click(pt)
        elif self._cmd_draw_mode == DrawMode.PLACE_PULLEY:
            self._place_pulley(pt)
        elif self._cmd_draw_mode == DrawMode.PLACE_HOLE:
            self._place_hole(pt)
        elif self._cmd_draw_mode == DrawMode.PLACE_DAMPER:
            self._place_damper(pt)
        # 腱绳模式不支持坐标输入，忽略
        self._cmd_input.clear()

    def _cmd_mouse_press(self, pos: QPointF):
        """命令模式下的鼠标点击，执行与回车相同的动作（除了腱绳）"""
        if self._cmd_draw_mode == DrawMode.PLACE_ACTUATOR:
            self._place_actuator(pos, atype=0)  # 默认A
        elif self._cmd_draw_mode in (DrawMode.DRAW_OUTLINE, DrawMode.DRAW_MOUNTAIN, DrawMode.DRAW_VALLEY):
            self._cmd_line_click(pos)
        elif self._cmd_draw_mode == DrawMode.PLACE_PULLEY:
            self._place_pulley(pos)
        elif self._cmd_draw_mode == DrawMode.PLACE_HOLE:
            self._place_hole(pos)
        elif self._cmd_draw_mode == DrawMode.PLACE_DAMPER:
            self._place_damper(pos)
        # 腱绳模式鼠标点击由场景正常处理（因为腱绳需要选择已有图元），这里不做，交给 _add_tendon_point

    def _cmd_line_click(self, pt: QPointF):
        """命令模式：处理线段绘制（坐标输入或鼠标点击）"""
        if self._cmd_start_point is None:
            self._cmd_start_point = pt
            self._cmd_temp_line = QtWidgets.QGraphicsLineItem(QLineF(pt, pt))
            self._cmd_temp_line.setPen(self._pen_for_line_type(self._cmd_line_type))
            self.addItem(self._cmd_temp_line)
        else:
            if self._cmd_temp_line:
                self.removeItem(self._cmd_temp_line)
                self._cmd_temp_line = None
            fl = FoldLine(
                id=self._next_line_id,
                start=Point2D(self._cmd_start_point.x(), self._cmd_start_point.y()),
                end=Point2D(pt.x(), pt.y()),
                fold_type=self._cmd_line_type)
            self.add_line_item(fl)
            self.line_added.emit(fl)
            self._next_line_id += 1
            # 连续模式：终点变起点
            self._cmd_start_point = pt
            self._cmd_temp_line = QtWidgets.QGraphicsLineItem(QLineF(pt, pt))
            self._cmd_temp_line.setPen(self._pen_for_line_type(self._cmd_line_type))
            self.addItem(self._cmd_temp_line)
        self.update()

    # ========== 正常模式鼠标绘图 ==========
    def _free_draw_click(self, pos):
        snap = self._snap_to_endpoint(pos)
        pt = snap if snap else pos
        if self._start_point is None:
            self._start_point = pt
            self._temp_line = QtWidgets.QGraphicsLineItem(QLineF(pt, pt))
            self._temp_line.setPen(self._pen_for_mode())
            self.addItem(self._temp_line)
        else:
            if self._temp_line:
                self.removeItem(self._temp_line)
                self._temp_line = None
            if (self._start_point - pt).manhattanLength() > 0.5:
                fl = FoldLine(
                    id=self._next_line_id,
                    start=Point2D(self._start_point.x(), self._start_point.y()),
                    end=Point2D(pt.x(), pt.y()),
                    fold_type=self._fold_type_for_mode())
                self.add_line_item(fl)
                self.line_added.emit(fl)
                self._next_line_id += 1
            self._start_point = None

    def _clear_free_drawing(self):
        if self._temp_line:
            self.removeItem(self._temp_line)
            self._temp_line = None
        self._start_point = None

    # ========== 滑轮 ==========
    def _place_pulley(self, pos: QPointF):
        snap_pt, fold_id = self._snap_to_fold_line(pos)
        final_pos = snap_pt if snap_pt else pos
        rad = 5
        ellipse = QtWidgets.QGraphicsEllipseItem(final_pos.x() - rad, final_pos.y() - rad, rad*2, rad*2)
        ellipse.setBrush(QColor(128,128,128))
        ellipse.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.addItem(ellipse)
        pulley = Pulley(
            id=self._pulley_id,
            position=Point2D(final_pos.x(), final_pos.y()),
            attached_fold_line_id=fold_id)
        self._pulleys[ellipse] = pulley
        self._pulley_id += 1

    def set_pulley_properties(self, pulley_id: int, friction: Optional[float] = None, radius: Optional[float] = None):
        """更改滑轮属性"""
        for item, p in self._pulleys.items():
            if p.id == pulley_id:
                if friction is not None:
                    p.friction_coefficient = friction
                if radius is not None:
                    p.radius = radius
                break

    # ========== 腱绳 ==========
    def _add_tendon_point(self, pos: QPointF):
        # 检查是否点到滑轮、孔、阻尼器或驱动器图元
        item = self.itemAt(pos, self.views()[0].transform())
        if isinstance(item, QtWidgets.QGraphicsEllipseItem):
            if item in self._pulleys or item in self._actuator_graphics or item in self._holes or item in self._dampers:
                # 避免重复点击同一个点连续两次
                if self._tendon_temp_path and self._tendon_temp_path[-1] == item:
                    return
                self._tendon_temp_path.append(item)
                if len(self._tendon_temp_path) >= 2:
                    prev = self._tendon_temp_path[-2]
                    line = QtWidgets.QGraphicsLineItem(QLineF(
                        prev.sceneBoundingRect().center(),
                        item.sceneBoundingRect().center()))
                    line.setPen(QPen(Qt.darkGray, 1, Qt.DashLine))
                    line.setFlag(QGraphicsItem.ItemIsSelectable, True)
                    self.addItem(line)
                    self._tendon_temp_lines.append(line)

    def _finish_tendon(self):
        """完成腱绳绘制：将临时虚线变为永久虚线，保存数据"""
        if len(self._tendon_temp_path) < 2:
            self._cancel_tendon()
            return

        seq = []
        for item in self._tendon_temp_path:
            if item in self._pulleys:
                seq.append(self._pulleys[item].id)
            elif item in self._actuator_graphics:
                # 用负数表示驱动器：-1 为 A, -2 为 B
                atype = self._actuator_graphics[item][0]
                seq.append(-(atype + 1))
            elif item in self._holes:
                seq.append(self._holes[item].id)
            elif item in self._dampers:
                seq.append(self._dampers[item].id)
            else:
                return   # 非法节点，丢弃

        tendon = Tendon(id=self._tendon_id, pulley_sequence=seq)
        self._tendons[self._tendon_id] = tendon

        # 把临时虚线变成永久虚线，并存储
        permanent_lines = []
        for line in self._tendon_temp_lines:
            line.setPen(QPen(Qt.darkGray, 1, Qt.DashLine))
            line.setFlag(QGraphicsItem.ItemIsSelectable, True)
            permanent_lines.append(line)
        self._tendon_graphics[self._tendon_id] = permanent_lines

        # ===== 后处理：基于腱绳路径更新孔和阻尼器的关联折痕 =====
        self._post_process_tendon_path(tendon)

        self._tendon_id += 1
        # 只清理列表，不移除图元
        self._tendon_temp_lines.clear()
        self._tendon_temp_path.clear()

    def _find_fold_line_between(self, pos_a: QPointF, pos_b: QPointF) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        找到线段 pos_a-pos_b 穿越的折痕线。
        
        Returns:
            (fold_line_id, side_a, side_b)
            side_a, side_b: +1/-1 表示线段两端在折痕的法线正/负侧
            None 表示找不到穿越的折痕
        """
        a = np.array([pos_a.x(), pos_a.y()])
        b = np.array([pos_b.x(), pos_b.y()])
        ab_dir = b - a
        ab_norm = np.linalg.norm(ab_dir)
        if ab_norm < 1e-10:
            return None, None, None
        ab_unit = ab_dir / ab_norm
        
        best = float('inf')
        best_id = None
        best_side_a = None
        best_side_b = None
        
        for item, fl in self._lines.items():
            if not fl.is_fold:
                continue  # 只检查折痕线（mount/valley）
            
            p1 = np.array([fl.start.x, fl.start.y])
            p2 = np.array([fl.end.x, fl.end.y])
            
            # 检查线段 AB 和折痕线是否相交
            # 使用叉积法
            d1 = p2 - p1
            d2 = b - a
            cross = np.cross(d1, d2)
            if abs(cross) < 1e-10:
                continue  # 平行或共线
            
            t = np.cross(a - p1, d2) / cross
            u = np.cross(a - p1, d1) / cross
            
            # 判断交点是否在两条线段内部（给一点容差）
            eps = 1e-6
            if t >= -eps and t <= 1 + eps and u >= -eps and u <= 1 + eps:
                # 线段 AB 确实穿过了这条折痕
                # 计算离线段中点的距离（优先选靠近中心的折痕）
                mid = (a + b) / 2
                intersect_pt = p1 + t * d1
                dist_to_mid = np.linalg.norm(intersect_pt - mid)
                
                if dist_to_mid < best:
                    best = dist_to_mid
                    best_id = fl.id
                    
                    # 计算两侧
                    normal = np.array([-d1[1], d1[0]])
                    side_a = 1 if np.dot(a - p1, normal) > 0 else -1
                    side_b = 1 if np.dot(b - p1, normal) > 0 else -1
                    best_side_a = side_a
                    best_side_b = side_b
        
        return best_id, best_side_a, best_side_b

    def _post_process_tendon_path(self, tendon: Tendon):
        """
        根据腱绳路径更新孔和阻尼器的关联折痕。
        
        原理：
        - 步骤1: 对路径中相邻的孔对，找出它们穿越的折痕，
          并将该折痕设为两个孔的 attached_fold_line_id。
          注意：孔对不一定是紧邻的（中间可能有滑轮/阻尼器/驱动器），
          算法会扫描所有连续段，找到所有孔孔段。
        - 步骤2: 对路径中的阻尼器，收集该腱绳中所有孔对识别的折痕，
          将这些折痕关联到阻尼器。因为阻尼器在物理上与它所在的腱绳段
          所控制的所有折痕都相关。
        """
        seq = tendon.pulley_sequence
        
        # 1) 处理所有相邻的孔孔段：找出穿越的折痕
        hole_fold_map = {}  # hole_id -> fold_line_id
        for i in range(len(seq) - 1):
            eid_a = seq[i]
            eid_b = seq[i + 1]
            
            if not (is_hole_id(eid_a) and is_hole_id(eid_b)):
                continue
            
            # 查找两个孔的图元
            item_a = None
            item_b = None
            for item, h in self._holes.items():
                if h.id == eid_a:
                    item_a = item
                if h.id == eid_b:
                    item_b = item
            
            if item_a is None or item_b is None:
                continue
            
            # 找出两个孔之间的折痕
            pos_a = item_a.sceneBoundingRect().center()
            pos_b = item_b.sceneBoundingRect().center()
            fold_id, side_a, side_b = self._find_fold_line_between(pos_a, pos_b)
            
            if fold_id is not None:
                # 更新两个孔的 attached_fold_line_id 和 face_id
                self._holes[item_a].attached_fold_line_id = fold_id
                self._holes[item_a].face_id = side_a
                self._holes[item_b].attached_fold_line_id = fold_id
                self._holes[item_b].face_id = side_b
                hole_fold_map[eid_a] = fold_id
                hole_fold_map[eid_b] = fold_id
        
        # 2) 处理阻尼器：收集该腱绳路径中所有孔关联的折痕，
        #    将这些折痕全都关联到阻尼器。
        #    阻尼器在物理上与腱绳段控制的所有折痕都相关。
        for i in range(len(seq)):
            eid = seq[i]
            if not is_damper_id(eid):
                continue
            
            # 找到阻尼器图元
            damper_item = None
            for item, d in self._dampers.items():
                if d.id == eid:
                    damper_item = item
                    break
            if damper_item is None:
                continue
            
            damper = self._dampers[damper_item]
            
            # 收集该腱绳路径中所有孔关联的唯一折痕
            all_folds = set()
            all_folds.update(hole_fold_map.values())
            
            # 也检查路径中非孔元素与孔的连接段
            for j in range(len(seq) - 1):
                eid_a = seq[j]
                eid_b = seq[j + 1]
                
                # 处理 孔-非孔 和 非孔-孔 段（例如阻尼器-孔）
                if is_hole_id(eid_a) and not is_hole_id(eid_b):
                    hole_id = eid_a
                    if hole_id in hole_fold_map:
                        all_folds.add(hole_fold_map[hole_id])
                elif not is_hole_id(eid_a) and is_hole_id(eid_b):
                    hole_id = eid_b
                    if hole_id in hole_fold_map:
                        all_folds.add(hole_fold_map[hole_id])
            
            # 更新阻尼器的 attached_fold_line_ids
            for fold_id in all_folds:
                if fold_id not in damper.attached_fold_line_ids:
                    damper.attached_fold_line_ids.append(fold_id)
                    damper.transmission_ratios.append(1.0)

    def _find_graphics_item_by_id(self, eid: int) -> Optional[QtWidgets.QGraphicsEllipseItem]:
        """根据 ID 找到对应的图元（滑轮、孔、阻尼器）。
        
        注意：不处理驱动器 ID。驱动器的查找由 load_tendons() 中的
        距离最近原则处理，因为同类型可能有多个驱动器副本（视觉辅助），
        需要在路径中找到距离最近的那个。
        """
        if eid >= 0:
            # 滑轮
            for item, p in self._pulleys.items():
                if p.id == eid:
                    return item
        elif is_hole_id(eid):
            for item, h in self._holes.items():
                if h.id == eid:
                    return item
        elif is_damper_id(eid):
            for item, d in self._dampers.items():
                if d.id == eid:
                    return item
        # 驱动器 ID（-1, -2 等）不在此处理，由 load_tendons 中的距离选择逻辑处理
        return None

    def _cancel_tendon(self):
        for line in self._tendon_temp_lines:
            self.removeItem(line)
        self._tendon_temp_lines.clear()
        self._tendon_temp_path.clear()

    # ========== 孔 ==========
    def _place_hole(self, pos: QPointF):
        """放置孔（到折痕的距离保持在用户点击的位置）"""
        # 注意：折痕吸附仅用于识别最近的折痕（记录到 attached_fold_line_id），
        # 但孔的实际位置保留原始点击位置，不吸附到折痕线上。
        # 因为孔的原始位置决定了 d（孔到折痕的水平距离）和 h（板厚偏移），
        # 这些几何参数会在 find_hole_pairs() 中根据孔的实际坐标精确计算。
        fold_id, side_sign = self._find_nearest_fold_line(pos)
        
        rad = 3
        ellipse = QtWidgets.QGraphicsEllipseItem(pos.x() - rad, pos.y() - rad, rad*2, rad*2)
        ellipse.setBrush(QColor(255, 140, 0))  # 橙色
        ellipse.setPen(QPen(Qt.darkYellow, 1))
        ellipse.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.addItem(ellipse)
        hole = Hole(
            id=self._hole_id,
            position=Point2D(pos.x(), pos.y()),
            attached_fold_line_id=fold_id,
            face_id=side_sign)  # side_sign: +1 或 -1 表示折痕两侧
        self._holes[ellipse] = hole
        self._hole_id -= 1  # 递减确保唯一负 ID
    
    # ========== 阻尼器 ==========
    def _place_damper(self, pos: QPointF):
        """放置阻尼器（浅绿色圆表示）"""
        # 找到最近的折痕线，获取关联的 fold_line_id
        fold_id, side_sign = self._find_nearest_fold_line(pos)
        fold_ids = [fold_id] if fold_id is not None else []
        
        rad = 6
        ellipse = QtWidgets.QGraphicsEllipseItem(pos.x() - rad, pos.y() - rad, rad*2, rad*2)
        ellipse.setBrush(QColor(144, 238, 144))  # 浅绿色
        ellipse.setPen(QPen(Qt.darkGreen, 1.5))
        ellipse.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.addItem(ellipse)
        
        # 使用负 ID（在 DAMPER_ID_OFFSET 以下）
        damper_id = DAMPER_ID_OFFSET - len(self._dampers)
        damper = Damper(
            id=damper_id,
            position=Point2D(pos.x(), pos.y()),
            attached_fold_line_ids=fold_ids,
            transmission_ratios=[1.0] * len(fold_ids) if fold_ids else [],
            damping_coefficient=1.0,
            name=f"Damper_{damper_id}"
        )
        self._dampers[ellipse] = damper

    def load_dampers(self, dampers: Dict[int, Damper]):
        """从数据加载阻尼器图元"""
        # 清除旧的阻尼器图元
        for item in list(self._dampers.keys()):
            self.removeItem(item)
        self._dampers.clear()
        
        for did, damper in dampers.items():
            pos = damper.position
            rad = 6
            ellipse = QtWidgets.QGraphicsEllipseItem(pos.x - rad, pos.y - rad, rad*2, rad*2)
            ellipse.setBrush(QColor(144, 238, 144))  # 浅绿色
            ellipse.setPen(QPen(Qt.darkGreen, 1.5))
            ellipse.setFlag(QGraphicsItem.ItemIsSelectable, True)
            self.addItem(ellipse)
            self._dampers[ellipse] = damper

    # ========== 驱动器 ==========
    def _place_actuator(self, pos: QPointF, atype: int = 0):
        """放置驱动器点 0=A (蓝), 1=B (红)"""
        r = 6
        color = QColor(Qt.blue) if atype == 0 else QColor(Qt.red)
        ellipse = QtWidgets.QGraphicsEllipseItem(pos.x()-r, pos.y()-r, r*2, r*2)
        ellipse.setBrush(color)
        ellipse.setPen(QPen(Qt.black, 1))
        ellipse.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.addItem(ellipse)
        self._actuator_graphics[ellipse] = (atype, pos)
        self._actuator_id += 1

    def set_actuator_type(self, item: QtWidgets.QGraphicsEllipseItem, atype: int):
        """更改驱动器类型（用于属性面板编辑）"""
        if item in self._actuator_graphics:
            pos = self._actuator_graphics[item][1]
            color = QColor(Qt.blue) if atype == 0 else QColor(Qt.red)
            item.setBrush(color)
            self._actuator_graphics[item] = (atype, pos)

    # ========== 线段管理 ==========
    def add_line_item(self, fl: FoldLine):
        # 使用 ClickableLineItem 便于鼠标选中
        item = ClickableLineItem(fl.start.x, fl.start.y, fl.end.x, fl.end.y)
        item.setPen(self._pen_for_line_type(fl.fold_type))
        item.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.addItem(item)
        self._lines[item] = fl

    def _select_click(self, pos):
        items = self.items(pos)
        for item in items:
            if isinstance(item, QtWidgets.QGraphicsLineItem):
                if item in self._lines:
                    self._select_line(item)
                    return
                # 检查是否为腱绳图元
                tendon_id = self._find_tendon_id_by_line(item)
                if tendon_id is not None:
                    self._select_tendon(item)
                    return
            if isinstance(item, QtWidgets.QGraphicsEllipseItem):
                if item in self._pulleys:
                    self._select_pulley(item)
                    return
                elif item in self._actuator_graphics:
                    self._select_actuator(item)
                    return
                elif item in self._holes:
                    self._select_hole(item)
                    return
                elif item in self._dampers:
                    self._select_damper(item)
                    return
        self._select_none()

    def _find_tendon_id_by_line(self, line_item):
        """查找腱绳图元所属的 tendon_id"""
        for tid, lines in self._tendon_graphics.items():
            if line_item in lines:
                return tid
        return None

    def _select_line(self, item):
        for it in self._lines:
            it.setSelected(False)
        for it in self._pulleys:
            it.setSelected(False)
        for it in self._actuator_graphics:
            it.setSelected(False)
        if item is not None:
            item.setSelected(True)
            self.line_selected.emit(self._lines[item].id)
        else:
            self.line_selected.emit(-1)

    def _select_pulley(self, item):
        """选中滑轮"""
        for it in self._lines: it.setSelected(False)
        for it in self._pulleys: it.setSelected(False)
        for it in self._actuator_graphics: it.setSelected(False)
        for it in self._holes: it.setSelected(False)
        item.setSelected(True)
        pid = self._pulleys[item].id
        self.pulley_selected.emit(pid)

    def _select_actuator(self, item):
        """选中驱动器"""
        for it in self._lines: it.setSelected(False)
        for it in self._pulleys: it.setSelected(False)
        for it in self._actuator_graphics: it.setSelected(False)
        for it in self._holes: it.setSelected(False)
        item.setSelected(True)
        atype, _ = self._actuator_graphics[item]
        self.actuator_selected.emit(atype)

    def _select_hole(self, item):
        """选中孔"""
        for it in self._lines: it.setSelected(False)
        for it in self._pulleys: it.setSelected(False)
        for it in self._actuator_graphics: it.setSelected(False)
        for it in self._holes: it.setSelected(False)
        for it in self._dampers: it.setSelected(False)
        item.setSelected(True)
        hid = self._holes[item].id
        self.hole_selected.emit(hid)

    def _select_damper(self, item):
        """选中阻尼器"""
        for it in self._lines: it.setSelected(False)
        for it in self._pulleys: it.setSelected(False)
        for it in self._actuator_graphics: it.setSelected(False)
        for it in self._holes: it.setSelected(False)
        for it in self._dampers: it.setSelected(False)
        item.setSelected(True)
        did = self._dampers[item].id
        self.damper_selected.emit(did)

    def _select_none(self):
        self._select_line(None)
        # 同时取消所有选中
        for it in self._pulleys: it.setSelected(False)
        for it in self._actuator_graphics: it.setSelected(False)
        for it in self._holes: it.setSelected(False)
        for it in self._dampers: it.setSelected(False)

    def _select_tendon(self, item):
        """选中腱绳（高亮其所属整条腱绳的所有线段）"""
        tendon_id = self._find_tendon_id_by_line(item)
        if tendon_id is None:
            return
        # 取消所有其他选中
        for it in self._lines: it.setSelected(False)
        for it in self._pulleys: it.setSelected(False)
        for it in self._actuator_graphics: it.setSelected(False)
        # 高亮整条腱绳
        for line in self._tendon_graphics[tendon_id]:
            line.setSelected(True)
        self.line_selected.emit(-1)  # 复用 line_selected，传递-1表示腱绳

    def _delete_selected_lines(self):
        for item in list(self.selectedItems()):
            if isinstance(item, QtWidgets.QGraphicsLineItem) and item in self._lines:
                self._remove_line_item(item)
            elif isinstance(item, QtWidgets.QGraphicsEllipseItem) and item in self._actuator_graphics:
                del self._actuator_graphics[item]
                self.removeItem(item)
            elif isinstance(item, QtWidgets.QGraphicsEllipseItem) and item in self._pulleys:
                del self._pulleys[item]
                self.removeItem(item)
            elif isinstance(item, QtWidgets.QGraphicsEllipseItem) and item in self._holes:
                del self._holes[item]
                self.removeItem(item)
            elif isinstance(item, QtWidgets.QGraphicsEllipseItem) and item in self._dampers:
                del self._dampers[item]
                self.removeItem(item)
            elif isinstance(item, QtWidgets.QGraphicsLineItem):
                # 检查是否为腱绳图元
                tendon_id = self._find_tendon_id_by_line(item)
                if tendon_id is not None:
                    # 删除整条腱绳的所有图元和数据
                    for line in self._tendon_graphics[tendon_id]:
                        self.removeItem(line)
                    del self._tendon_graphics[tendon_id]
                    del self._tendons[tendon_id]

    def _remove_line_item(self, item):
        if item in self._lines:
            del self._lines[item]
            self.removeItem(item)

    # ========== 辅助 ==========
    def _fold_type_for_mode(self):
        return {DrawMode.DRAW_OUTLINE: FoldType.OUTLINE,
                DrawMode.DRAW_MOUNTAIN: FoldType.MOUNTAIN,
                DrawMode.DRAW_VALLEY: FoldType.VALLEY}.get(self.draw_mode, FoldType.OUTLINE)

    def _pen_for_mode(self):
        return self._pen_for_line_type(self._fold_type_for_mode())

    def _pen_for_line_type(self, ft):
        c = {FoldType.OUTLINE: Qt.black, FoldType.MOUNTAIN: Qt.red, FoldType.VALLEY: Qt.blue}.get(ft, Qt.black)
        return QPen(c, 1)

    def _snap_to_endpoint(self, pos):
        for item in self._lines:
            p1, p2 = item.line().p1(), item.line().p2()
            if QLineF(p1, pos).length() < self.snap_distance:
                return p1
            if QLineF(p2, pos).length() < self.snap_distance:
                return p2
        return None

    def _snap_to_fold_line(self, pos) -> Tuple[Optional[QPointF], Optional[int]]:
        best = self.snap_distance
        best_pt, best_id = None, None
        for item, fl in self._lines.items():
            p1, p2 = item.line().p1(), item.line().p2()
            v = p2 - p1
            denom = v.x()**2 + v.y()**2
            if denom < 1e-9:
                closest = p1
            else:
                t = QPointF.dotProduct(pos - p1, v) / denom
                t = max(0.0, min(1.0, t))
                closest = p1 + t * v
            dist = QLineF(closest, pos).length()
            if dist < best:
                best = dist
                best_pt = closest
                best_id = fl.id
        return best_pt, best_id

    def _find_nearest_fold_line(self, pos) -> Tuple[Optional[int], Optional[int]]:
        """找到最近的折痕线，返回 (fold_line_id, side_sign)。
        
        side_sign: +1 = 折痕normal正方向一侧, -1 = 负方向一侧, None = 没有折痕
        注意：孔不会吸附到折痕线上，孔的实际位置被保留用于后续精确几何计算。
        """
        best = self.snap_distance * 3  # 允许更大范围
        best_id = None
        best_side = None
        
        for item, fl in self._lines.items():
            if not fl.is_fold:
                continue  # 只吸附到折痕线（mount/valley）
            p1 = np.array([fl.start.x, fl.start.y])
            p2 = np.array([fl.end.x, fl.end.y])
            pos_arr = np.array([pos.x(), pos.y()])
            
            # 计算点到直线的最短距离
            dir_vec = p2 - p1
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < 1e-10:
                continue
            dir_unit = dir_vec / dir_norm
            
            # 垂线距离
            v = pos_arr - p1
            proj_len = np.dot(v, dir_unit)
            closest = p1 + proj_len * dir_unit
            
            # 判断投影点是否在线段范围内
            proj_t = proj_len / dir_norm
            if proj_t < -0.3 or proj_t > 1.3:
                continue  # 离线段太远
            
            dist = np.linalg.norm(pos_arr - closest)
            if dist < best:
                best = dist
                best_id = fl.id
                # 计算侧边符号
                normal = np.array([-dir_unit[1], dir_unit[0]])
                side = 1 if np.dot(v, normal) > 0 else -1
                best_side = side
        
        return best_id, best_side

    def get_line_by_id(self, line_id):
        for fl in self._lines.values():
            if fl.id == line_id:
                return fl
        return None
