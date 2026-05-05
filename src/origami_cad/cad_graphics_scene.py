# src/origami_cad/cad_graphics_scene.py
"""
重制版 QGraphicsScene：支持命令行/鼠标混合绘制，互斥工具栏，驱动器，腱绳端点保存。
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QLineF, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QPen, QPainter, QColor, QBrush, QPainterPath, QPainterPathStroker
from PyQt5.QtWidgets import (QGraphicsScene, QGraphicsLineItem,
                             QGraphicsEllipseItem, QGraphicsProxyWidget, QLineEdit,
                             QGraphicsItem, QGraphicsRectItem)
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple
from src.models.origami_design import FoldLine, Point2D, FoldType, Pulley, Tendon

class DrawMode(Enum):
    SELECT = auto()
    DRAW_OUTLINE = auto()
    DRAW_MOUNTAIN = auto()
    DRAW_VALLEY = auto()
    PLACE_PULLEY = auto()
    DRAW_TENDON = auto()
    PLACE_ACTUATOR = auto()

# 键盘映射到绘线类型
KEY_TO_LINE_TYPE = {
    Qt.Key_L: FoldType.OUTLINE,
    Qt.Key_M: FoldType.MOUNTAIN,
    Qt.Key_V: FoldType.VALLEY,
}
KEY_TO_PULLEY_MODE = Qt.Key_P
KEY_TO_TENDON_MODE = Qt.Key_T
KEY_TO_ACTUATOR_MODE = Qt.Key_A

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
    actuator_selected = pyqtSignal(int)  # 0 or 1 (type)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.draw_mode = DrawMode.SELECT
        self._next_line_id = 0
        self._pulley_id = 0
        self._tendon_id = 0
        self._actuator_id = 0

        # 图元映射
        self._lines: Dict[QtWidgets.QGraphicsLineItem, FoldLine] = {}
        self._pulleys: Dict[QtWidgets.QGraphicsEllipseItem, Pulley] = {}
        self._actuator_graphics: Dict[QtWidgets.QGraphicsEllipseItem, Tuple[int, QPointF]] = {}  # item -> (type, pos)
        self._tendons: Dict[int, Tendon] = {}
        self._tendon_graphics: Dict[int, List[QtWidgets.QGraphicsLineItem]] = {}  # tendon_id -> 永久线列表

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

    def get_all_fold_lines(self) -> List[FoldLine]:
        return list(self._lines.values())

    def clear_lines(self):
        for item in list(self._lines.keys()):
            self.removeItem(item)
        self._lines.clear()
        self._next_line_id = 0

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
            for seq_id in tendon.pulley_sequence:
                # 找对应的图元 item
                cur_item = None
                if seq_id >= 0:
                    # 正数：滑轮
                    for item, p in self._pulleys.items():
                        if p.id == seq_id:
                            cur_item = item
                            break
                else:
                    # 负数：驱动器，-1 -> A (type 0), -2 -> B (type 1)
                    for item, (atype, pos) in self._actuator_graphics.items():
                        if -(atype + 1) == seq_id:
                            cur_item = item
                            break
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
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.scenePos()
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

        # 命令模式激活：L/M/V/P/A（需要坐标输入或鼠标点击）
        cmd_keys = (Qt.Key_L, Qt.Key_M, Qt.Key_V, KEY_TO_PULLEY_MODE, KEY_TO_ACTUATOR_MODE)
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
        else:
            return

        # 创建浮动输入框
        self._cmd_input = QLineEdit()
        if hasattr(Qt.Key, 'name') and key != Qt.Key_A:
            prefix = Qt.Key(key).name.decode()
        else:
            prefix = {Qt.Key_L: 'L', Qt.Key_M: 'M', Qt.Key_V: 'V',
                      Qt.Key_P: 'P', Qt.Key_T: 'T', Qt.Key_A: 'A'}.get(key, '?')
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
        # 检查是否点到滑轮或驱动器图元
        item = self.itemAt(pos, self.views()[0].transform())
        if isinstance(item, QtWidgets.QGraphicsEllipseItem):
            if item in self._pulleys or item in self._actuator_graphics:
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

        self._tendon_id += 1
        # 只清理列表，不移除图元
        self._tendon_temp_lines.clear()
        self._tendon_temp_path.clear()

    def _cancel_tendon(self):
        for line in self._tendon_temp_lines:
            self.removeItem(line)
        self._tendon_temp_lines.clear()
        self._tendon_temp_path.clear()

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
        item.setSelected(True)
        pid = self._pulleys[item].id
        self.pulley_selected.emit(pid)

    def _select_actuator(self, item):
        """选中驱动器"""
        for it in self._lines: it.setSelected(False)
        for it in self._pulleys: it.setSelected(False)
        for it in self._actuator_graphics: it.setSelected(False)
        item.setSelected(True)
        atype, _ = self._actuator_graphics[item]
        self.actuator_selected.emit(atype)

    def _select_none(self):
        self._select_line(None)
        # 同时取消所有选中
        for it in self._pulleys: it.setSelected(False)
        for it in self._actuator_graphics: it.setSelected(False)

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

    def get_line_by_id(self, line_id):
        for fl in self._lines.values():
            if fl.id == line_id:
                return fl
        return None
