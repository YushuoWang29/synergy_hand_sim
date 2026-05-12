from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QTransform
from PyQt5.QtCore import Qt, QLineF, QRectF


class CadGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        # 默认 NoDrag，让场景完整处理鼠标点击
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.SmartViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Y轴翻转（向上为正） + 200% 初始缩放
        self.setTransform(QTransform.fromScale(2.0, -2.0))
        self._panning = False
        self._zoom_level = 2.0  # 初始缩放

    def wheelEvent(self, event: QtGui.QWheelEvent):
        zoom = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(zoom, zoom)
        self._zoom_level *= zoom
        self._zoom_level = max(0.01, min(100.0, self._zoom_level))  # 限制缩放范围
        self.viewport().update()  # 触发重绘，更新 HUD

    def drawForeground(self, painter: QPainter, rect: QRectF):
        """在视图前景绘制 HUD 信息：缩放百分比和当前绘图模式"""
        # 保存原来 painter 状态
        painter.save()

        # 获取当前绘图模式
        from .cad_graphics_scene import DrawMode
        scene = self.scene()
        if hasattr(scene, 'draw_mode'):
            mode_names = {
                DrawMode.SELECT: "选择",
                DrawMode.DRAW_OUTLINE: "轮廓线",
                DrawMode.DRAW_MOUNTAIN: "峰折线",
                DrawMode.DRAW_VALLEY: "谷折线",
                DrawMode.PLACE_PULLEY: "滑轮",
                DrawMode.PLACE_HOLE: "孔",
                DrawMode.PLACE_DAMPER: "阻尼器",
                DrawMode.DRAW_TENDON: "腱绳",
                DrawMode.PLACE_ACTUATOR: "驱动器",
            }
            mode_name = mode_names.get(scene.draw_mode, "未知")
        else:
            mode_name = "?"

        # 构造 HUD 文本
        zoom_text = f"缩放: {self._zoom_level * 100:.0f}%"
        mode_text = f"模式: {mode_name}"
        hud_text = f"{zoom_text}  |  {mode_text}"

        # 在视口左上角绘制 HUD
        viewport_rect = self.viewport().rect()
        # 将视口坐标转换为场景坐标（保持 HUD 在视口左上角固定位置）
        top_left = self.mapToScene(QtCore.QPoint(12, 8))
        painter.resetTransform()

        # 设置字体和画笔
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 1))
        painter.setBrush(QBrush(QColor(0, 0, 0, 140)))

        # 背景矩形
        text_rect = painter.boundingRect(QtCore.QRectF(0, 0, 500, 50), Qt.AlignLeft | Qt.AlignTop, hud_text)
        text_rect.adjust(-6, -4, 6, 4)
        painter.drawRoundedRect(text_rect, 4, 4)

        # 绘制文本
        painter.setPen(QPen(QColor(255, 255, 255, 230)))
        painter.drawText(text_rect.adjusted(4, 0, -4, 0), Qt.AlignLeft | Qt.AlignVCenter, hud_text)

        painter.restore()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        # 转发给场景
        self.scene().keyPressEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.MiddleButton:
            # 中键按下：切换为 ScrollHandDrag 进行平移
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._panning = True
            # 模拟左键按下启动拖拽
            fake_event = QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonPress,
                event.localPos(), event.windowPos(), event.screenPos(),
                Qt.LeftButton, Qt.LeftButton, event.modifiers())
            super().mousePressEvent(fake_event)
            return
        elif event.button() == Qt.LeftButton:
            # 左键：NoDrag 模式，场景完整处理
            super().mousePressEvent(event)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if self._panning and event.button() == Qt.MiddleButton:
            self._panning = False
            # 结束拖拽
            fake_event = QtGui.QMouseEvent(
                QtCore.QEvent.MouseButtonRelease,
                event.localPos(), event.windowPos(), event.screenPos(),
                Qt.LeftButton, Qt.LeftButton, event.modifiers())
            super().mouseReleaseEvent(fake_event)
            # 恢复 NoDrag
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            return
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        super().mouseMoveEvent(event)
