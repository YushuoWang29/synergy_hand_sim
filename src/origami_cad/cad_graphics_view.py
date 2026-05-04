from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPainter, QColor, QPen, QTransform
from PyQt5.QtCore import Qt, QLineF

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

    def wheelEvent(self, event: QtGui.QWheelEvent):
        zoom = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(zoom, zoom)

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
