#!/usr/bin/env python3
"""
生成 Origami CAD 编辑器主界面截图（基于 ohd_6 渲染）
使用 matplotlib 重绘 CAD 编辑器的二维视图，模拟 PyQt5 界面风格。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from src.models.origami_design import (
    OrigamiHandDesign, FoldLine, Point2D, FoldType,
    Pulley, Hole, Tendon, is_hole_id, is_actuator_id, is_damper_id, is_pulley_id
)

OHD_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'ohd test', 'ohd_6.ohd')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'cad_ohd6_screenshot.png')

def load_design(path):
    """加载 .ohd 文件"""
    design = OrigamiHandDesign.load(path)
    print(f"  Loaded: {design.name}")
    print(f"  Fold lines: {len(design.fold_lines)}")
    print(f"  Faces: {len(design.faces)}")
    print(f"  Joints: {len(design.joints)}")
    print(f"  Pulleys: {len(design.pulleys)}")
    print(f"  Holes: {len(design.holes)}")
    print(f"  Tendons: {len(design.tendons)}")
    print(f"  Actuators: {len(design.actuators)}")
    print(f"  Actuator positions: {len(design.actuator_positions)}")
    return design

def draw_cad_view(design):
    """
    使用 matplotlib 绘制类似于 CAD 编辑器主界面的二维视图。
    模拟 PyQt5 界面风格：灰色背景、彩色折痕线、滑轮/孔/驱动器/腱绳。
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 计算图纸范围（留边距）
    all_x, all_y = [], []
    for fl in design.fold_lines.values():
        all_x.extend([fl.start.x, fl.end.x])
        all_y.extend([fl.start.y, fl.end.y])
    for p in design.pulleys.values():
        all_x.append(p.position.x)
        all_y.append(p.position.y)
    for h in design.holes.values():
        all_x.append(h.position.x)
        all_y.append(h.position.y)
    for ap in design.actuator_positions:
        all_x.append(ap['x'])
        all_y.append(ap['y'])
    
    if not all_x:
        print("  WARNING: No geometry data found!")
        return None

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    margin = max((x_max - x_min), (y_max - y_min)) * 0.15 + 20
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin

    # 创建图形：模拟 CAD 窗口（16:9 比例）
    fig_width = 10
    fig_height = fig_width * (y_max - y_min) / (x_max - x_min)
    if fig_height > 7:
        fig_height = 7
        fig_width = fig_height * (x_max - x_min) / (y_max - y_min)

    fig, ax = plt.subplots(figsize=(fig_width + 3, fig_height + 1))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_facecolor('#F5F5F5')  # 浅灰背景

    # --- 绘制面片（填充） ---
    face_colors = ['#E8F5E9', '#FFF3E0', '#E3F2FD', '#FCE4EC', '#F3E5F5',
                   '#E0F2F1', '#FFF8E1', '#FBE9E7', '#EFEBE9', '#E8EAF6',
                   '#F1F8E9', '#F9FBE7', '#E0F7FA', '#FFF9C4', '#F1F8E9']
    for i, (fid, face) in enumerate(design.faces.items()):
        verts = [(v.x, v.y) for v in face.vertices]
        color = face_colors[i % len(face_colors)]
        poly = plt.Polygon(verts, facecolor=color, edgecolor='none', alpha=0.6, zorder=0)
        ax.add_patch(poly)
        # 面片中心标签
        cx = np.mean([v.x for v in face.vertices])
        cy = np.mean([v.y for v in face.vertices])
        ax.text(cx, cy, f'F{fid}', fontsize=5, ha='center', va='center',
                color='#999999', zorder=0)

    # --- 绘制折痕线 ---
    fold_colors = {
        FoldType.MOUNTAIN: '#E74C3C',  # 红——峰折
        FoldType.VALLEY: '#3498DB',    # 蓝——谷折
        FoldType.OUTLINE: '#555555',   # 灰——轮廓
    }
    fold_styles = {
        FoldType.MOUNTAIN: 'solid',
        FoldType.VALLEY: 'solid',
        FoldType.OUTLINE: 'dashed',
    }
    fold_widths = {
        FoldType.OUTLINE: 1.5,
    }

    for fl in design.fold_lines.values():
        color = fold_colors.get(fl.fold_type, '#333333')
        ls = fold_styles.get(fl.fold_type, 'solid')
        if fl.is_fold:
            lw = 3.0
        else:
            lw = fold_widths.get(fl.fold_type, 2.0)
        ax.plot([fl.start.x, fl.end.x], [fl.start.y, fl.end.y],
                color=color, linewidth=lw, linestyle=ls, zorder=2)
        # 折痕类型标签（M/V）
        mx, my = (fl.start.x + fl.end.x) / 2, (fl.start.y + fl.end.y) / 2
        label = 'M' if fl.fold_type == FoldType.MOUNTAIN else \
                'V' if fl.fold_type == FoldType.VALLEY else ''
        if label:
            ax.text(mx, my - 5, label, fontsize=7, fontweight='bold',
                    color=color, ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                              edgecolor='none', alpha=0.7), zorder=5)

    # --- 绘制滑轮 ---
    for pid, pulley in design.pulleys.items():
        circle = plt.Circle((pulley.position.x, pulley.position.y),
                            pulley.radius if pulley.radius > 0 else 3.0,
                            facecolor='#2ECC71', edgecolor='#27AE60',
                            linewidth=2, zorder=4)
        ax.add_patch(circle)
        # 滑轮 ID
        ax.text(pulley.position.x, pulley.position.y + pulley.radius + 3,
                f'P{pid}', fontsize=5, ha='center', va='bottom',
                color='#27AE60', fontweight='bold', zorder=4)

    # --- 绘制孔 ---
    for hid, hole in design.holes.items():
        circle = plt.Circle((hole.position.x, hole.position.y),
                            hole.radius if hole.radius > 0 else 2.0,
                            facecolor='#E67E22', edgecolor='#D35400',
                            linewidth=2, zorder=4)
        ax.add_patch(circle)
        ax.text(hole.position.x, hole.position.y + hole.radius + 3,
                f'H{hid}', fontsize=5, ha='center', va='bottom',
                color='#D35400', fontweight='bold', zorder=4)

    # --- 绘制驱动器 ---
    actuator_colors = {0: '#E74C3C', 1: '#2980B9'}
    for i, ap in enumerate(design.actuator_positions):
        color = actuator_colors.get(i, '#9B59B6')
        rect = plt.Rectangle((ap['x'] - 6, ap['y'] - 6), 12, 12,
                              facecolor=color, edgecolor='#2C3E50',
                              linewidth=2, zorder=4)
        ax.add_patch(rect)
        label = 'A' if i == 0 else 'B'
        ax.text(ap['x'], ap['y'] + 10, f'Motor {label}', fontsize=6,
                ha='center', va='bottom', color=color,
                fontweight='bold', zorder=4)

    # --- 绘制腱绳路径 ---
    tendon_colors = ['#8E44AD', '#1ABC9C', '#F39C12', '#E74C3C']
    for ti, (tid, tendon) in enumerate(design.tendons.items()):
        seq = tendon.pulley_sequence
        path_pts = []
        for eid in seq:
            if is_actuator_id(eid):
                # 从 actuator_positions 查找
                for ap in design.actuator_positions:
                    path_pts.append((ap['x'], ap['y']))
            elif is_pulley_id(eid) and eid in design.pulleys:
                p = design.pulleys[eid]
                path_pts.append((p.position.x, p.position.y))
            elif is_hole_id(eid) and eid in design.holes:
                h = design.holes[eid]
                path_pts.append((h.position.x, h.position.y))
        if path_pts:
            xs, ys = zip(*path_pts)
            color = tendon_colors[ti % len(tendon_colors)]
            ax.plot(xs, ys, color=color, linewidth=2.0, linestyle='-',
                    marker='o', markersize=3, markerfacecolor=color,
                    alpha=0.8, zorder=3)

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 添加标题（模拟 CAD 窗口标题栏）
    ax.text(0.5, 0.98, 'Origami Hand CAD — ohd_6 设计视图',
            transform=fig.transFigure, ha='center', va='top',
            fontsize=13, fontweight='bold', color='#2C3E50')

    # --- 图例 ---
    legend_elements = [
        Line2D([0], [0], color='#E74C3C', linewidth=3, label='峰折 (Mountain)'),
        Line2D([0], [0], color='#3498DB', linewidth=3, label='谷折 (Valley)'),
        Line2D([0], [0], color='#555555', linewidth=2, linestyle='dashed', label='轮廓 (Outline)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71',
               markersize=8, label='滑轮 (Pulley)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E67E22',
               markersize=8, label='孔 (Hole)'),
        plt.Rectangle((0,0), 1, 1, facecolor='#E74C3C', edgecolor='#2C3E50',
                      label='驱动器 A (Motor A)'),
        plt.Rectangle((0,0), 1, 1, facecolor='#2980B9', edgecolor='#2C3E50',
                      label='驱动器 B (Motor B)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              fontsize=7, framealpha=0.9,
              bbox_to_anchor=(-0.02, 1.02))

    plt.tight_layout()
    return fig

def main():
    print("生成 CAD 编辑器主界面截图...")
    print(f"  加载: {OHD_PATH}")
    design = load_design(OHD_PATH)
    fig = draw_cad_view(design)
    if fig is not None:
        fig.savefig(OUT_PATH, dpi=250, bbox_inches='tight', pad_inches=0.5)
        print(f"  [OK] {OUT_PATH}")
    print("完成！")

if __name__ == '__main__':
    main()
