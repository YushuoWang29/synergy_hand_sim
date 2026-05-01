# src/interactive/cad_viewer.py

import numpy as np
import matplotlib
# 强制使用qt5后端
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from typing import Dict, Tuple, Optional
from src.models.origami_design import (
    OrigamiHandDesign, FoldLine, FoldType, JointConnection
)


class CADViewer:
    """2D CAD预览窗口"""
    
    def __init__(self, design: OrigamiHandDesign):
        self.design = design
        self.line_artists: Dict[int, plt.Line2D] = {}
        self.face_patches: Dict[int, Polygon] = {}
        self.selected_line_id: Optional[int] = None
        self._on_select_callback = None
        
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._init_plot()
        
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
    
    def _init_plot(self):
        """初始化2D绘图"""
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_title("CAD Preview\nClick to select fold, ↑↓ to change angle", 
                          fontsize=10)
        self.ax.grid(True, alpha=0.3)
        
        # 计算边界
        all_x, all_y = [], []
        for line in self.design.fold_lines.values():
            all_x.extend([line.start.x, line.end.x])
            all_y.extend([line.start.y, line.end.y])
        
        if all_x:
            x_margin = (max(all_x) - min(all_x)) * 0.1 + 10
            y_margin = (max(all_y) - min(all_y)) * 0.1 + 10
            self.ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
            self.ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        # 绘制面片
        for face_id, face in self.design.faces.items():
            vertices = [(v.x, v.y) for v in face.vertices]
            poly = Polygon(vertices, closed=True, facecolor='lightgray',
                         edgecolor='none', alpha=0.3, zorder=1)
            self.ax.add_patch(poly)
            self.face_patches[face_id] = poly
            
            centroid = face.centroid
            self.ax.annotate(f"F{face_id}", (centroid.x, centroid.y),
                           fontsize=8, ha='center', va='center',
                           color='gray', zorder=5)
        
        # 绘制折痕线
        for line_id, line in self.design.fold_lines.items():
            if line.fold_type == FoldType.MOUNTAIN:
                color, lw, ls = 'red', 2.5, '-'
            elif line.fold_type == FoldType.VALLEY:
                color, lw, ls = 'blue', 2.5, '-'
            else:
                color, lw, ls = 'black', 1.5, '-'
            
            self.line_artists[line_id], = self.ax.plot(
                [line.start.x, line.end.x],
                [line.start.y, line.end.y],
                color=color, linewidth=lw, linestyle=ls,
                picker=5, zorder=3
            )
            
            if line.is_fold:
                mid = line.midpoint
                self.ax.annotate(
                    f"{line_id}", (mid.x, mid.y),
                    fontsize=7, ha='center', va='center',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.2',
                            facecolor=color, alpha=0.8),
                    zorder=4
                )
        
        # 图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Mountain'),
            Line2D([0], [0], color='blue', lw=2, label='Valley'),
            Line2D([0], [0], color='black', lw=1.5, label='Outline'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', 
                      fontsize=7, framealpha=0.9)
        self.fig.tight_layout()
        self.fig.canvas.draw()
    
    def set_select_callback(self, callback):
        """设置选择回调 callback(line_id, fold_type)"""
        self._on_select_callback = callback
    
    def _on_click(self, event):
        """鼠标点击事件"""
        if event.inaxes != self.ax:
            return
        
        # 查找最近的折痕线
        min_dist = float('inf')
        closest_id = None
        
        for line_id, artist in self.line_artists.items():
            line = self.design.fold_lines[line_id]
            if not line.is_fold:
                continue
            
            dist = self._point_to_line_distance(
                event.xdata, event.ydata,
                line.start.x, line.start.y,
                line.end.x, line.end.y
            )
            
            if dist < min_dist:
                min_dist = dist
                closest_id = line_id
        
        if closest_id is not None and min_dist < 20:
            self._select_line(closest_id)
    
    def _point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """计算点到线段的屏幕距离"""
        disp = self.ax.transData.transform
        p = disp((px, py))
        s = disp((x1, y1))
        e = disp((x2, y2))
        d = e - s
        t = max(0, min(1, np.dot(p - s, d) / np.dot(d, d)))
        proj = s + t * d
        return np.linalg.norm(p - proj)
    
    def _select_line(self, line_id):
        """选中折痕并高亮"""
        if self.selected_line_id is not None:
            old = self.line_artists[self.selected_line_id]
            fl = self.design.fold_lines[self.selected_line_id]
            if fl.fold_type == FoldType.MOUNTAIN:
                old.set_color('red')
            else:
                old.set_color('blue')
            old.set_linewidth(2.5)
        
        self.selected_line_id = line_id
        artist = self.line_artists[line_id]
        artist.set_color('lime')
        artist.set_linewidth(4)
        self.fig.canvas.draw()
        
        if self._on_select_callback:
            line = self.design.fold_lines[line_id]
            self._on_select_callback(line_id, line.fold_type)
    
    def get_figure(self):
        return self.fig