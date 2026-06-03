#!/usr/bin/env python3
"""
生成第3.2节 图3-1: SDAS 系统架构图
使用 matplotlib 绘制模块化框图。
用法: python docs/figures/generate_architecture_diagram.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 字体配置
try:
    matplotlib.font_manager.findfont('SimSun')
    plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'DejaVu Sans']
except Exception:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def draw_arrow(ax, start, end, color='#555555', lw=1.5):
    """在坐标轴间绘制箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle='arc3,rad=0'),
                zorder=5)


def draw_box(ax, center, width, height, text, subtext='',
             facecolor='#D6EAF8', edgecolor='#2C3E50',
             text_color='black', fontsize=10, sub_fontsize=8):
    """绘制带标题和副标题的方框"""
    x, y = center
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.1",
        facecolor=facecolor, edgecolor=edgecolor, lw=2, zorder=3)
    ax.add_patch(rect)

    # 主标题
    ax.text(x, y + height * 0.15, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)
    # 副标题
    if subtext:
        ax.text(x, y - height * 0.20, subtext, ha='center', va='center',
                fontsize=sub_fontsize, color='#444444', zorder=4)


def draw_io_box(ax, center, width, height, text,
                facecolor='#F9E79F', edgecolor='#9A7D0A'):
    """绘制数据/文件节点（棱形感）"""
    x, y = center
    rect = mpatches.FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.1",
        facecolor=facecolor, edgecolor=edgecolor, lw=2, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=9, fontweight='bold', color='#7D6608', zorder=4)


def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-1, 13)
    ax.set_ylim(-1, 9)
    ax.axis('off')

    # ================== 定义模块位置 ==================
    # 行1: CAD 编辑器
    cad_center = (1.0, 7.0)
    ohd_center = (3.5, 7.0)

    # 行2: 数据模型层
    design_center = (2.25, 5.0)

    # 行3: 传动计算层
    trans_center = (2.25, 3.0)

    # 行4: SDAS 协同层
    sdas_center = (2.25, 1.0)

    # 右半部分：仿真层
    urdf_center = (7.0, 6.0)
    mapping_center = (7.0, 4.5)

    mujoco_center = (9.5, 3.0)
    dynamics_center = (9.5, 1.0)

    viz_center = (11.5, 2.0)

    # ================== 绘制模块 ==================
    # --- 第一层：设计与数据入口 ---
    draw_box(ax, cad_center, 2.6, 1.2, 'Origami CAD 编辑器',
             '用户交互界面\n绘制折痕/滑轮/孔/腱绳/驱动器',
             facecolor='#AED6F1', fontsize=11, sub_fontsize=8)

    draw_io_box(ax, ohd_center, 1.8, 0.7, '.ohd 文件\n(JSON格式)')

    # --- 第二层：数据模型 ---
    draw_box(ax, design_center, 2.8, 1.2, 'OrigamiHandDesign',
             '数据模型容器\n折痕/面片/滑轮/孔/腱绳/驱动器',
             facecolor='#A9DFBF', fontsize=11, sub_fontsize=8)

    # --- 第三层：传动计算 ---
    draw_box(ax, trans_center, 3.0, 1.2, 'TransmissionBuilder',
             'compute_R_one_sided()\nR_A 与 R_B  (1×n) 传动向量',
             facecolor='#FADBD8', fontsize=11, sub_fontsize=8)

    # --- 第四层：SDAS 协同求解 ---
    draw_box(ax, sdas_center, 3.0, 1.2, 'SDASModel',
             'solve_motors() / solve_synergies()\nSchur补 + 单向公式',
             facecolor='#D7BDE2', fontsize=11, sub_fontsize=8)

    # --- 右半部分：URDF 导出与映射 ---
    draw_box(ax, urdf_center, 2.4, 1.0, 'URDF 导出',
             'export_urdf()\nURDF + STL 文件',
             facecolor='#F9E79F', fontsize=10, sub_fontsize=8)

    draw_box(ax, mapping_center, 2.4, 1.0, '关节映射构建',
             'URDF关节名称 ↔ synergy索引\n最近邻空间匹配',
             facecolor='#F9E79F', fontsize=10, sub_fontsize=8)

    # --- 仿真层 ---
    draw_box(ax, mujoco_center, 2.8, 1.2, 'MuJoCoSimulator',
             '运动学交互仿真\n滑块控制 → synergy_callback → qpos → mj_forward',
             facecolor='#85C1E9', fontsize=11, sub_fontsize=8)

    draw_box(ax, dynamics_center, 2.8, 1.2, 'HandSimulator\n(动力学数值仿真)',
             'Phase 1/2/3  ODE积分\n含惯性/阻尼/接触',
             facecolor='#85C1E9', fontsize=11, sub_fontsize=8)

    # --- 可视化 ---
    draw_box(ax, viz_center, 2.0, 1.0, 'Visualization',
             'MeshCat 3D / matplotlib',
             facecolor='#F0F3F4', fontsize=10, sub_fontsize=8)

    # ================== 绘制箭头（数据流） ==================
    # CAD → .ohd
    draw_arrow(ax, (cad_center[0] + 1.3, cad_center[1]),
               (ohd_center[0] - 0.9, ohd_center[1]))

    # .ohd → OrigamiHandDesign
    draw_arrow(ax, (ohd_center[0], ohd_center[1] - 0.35),
               (design_center[0], design_center[1] + 0.6))

    # OrigamiHandDesign → URDF 导出
    draw_arrow(ax, (design_center[0] + 1.4, design_center[1] + 0.3),
               (urdf_center[0] - 1.2, urdf_center[1] + 0.3))

    # OrigamiHandDesign → TransmissionBuilder
    draw_arrow(ax, (design_center[0], design_center[1] - 0.6),
               (trans_center[0], trans_center[1] + 0.6))

    # TransmissionBuilder → SDASModel
    draw_arrow(ax, (trans_center[0], trans_center[1] - 0.6),
               (sdas_center[0], sdas_center[1] + 0.6))

    # URDF → 关节映射
    draw_arrow(ax, (urdf_center[0], urdf_center[1] - 0.5),
               (mapping_center[0], mapping_center[1] + 0.5))

    # SDAS → MuJoCoSimulator
    draw_arrow(ax, (sdas_center[0] + 1.5, sdas_center[1] + 0.3),
               (mujoco_center[0] - 1.4, mujoco_center[1] + 0.3))

    # 关节映射 → MuJoCo
    draw_arrow(ax, (mapping_center[0], mapping_center[1] - 0.5),
               (mujoco_center[0], mujoco_center[1] + 0.6),
               color='#888888', lw=1.0)

    # SDAS → HandSimulator
    draw_arrow(ax, (sdas_center[0] + 1.5, sdas_center[1] - 0.3),
               (dynamics_center[0] - 1.4, dynamics_center[1] - 0.2))

    # MuJoCo/HandSimulator → Visualization
    draw_arrow(ax, (mujoco_center[0] + 1.4, mujoco_center[1] + 0.2),
               (viz_center[0] - 1.0, viz_center[1] + 0.3))
    draw_arrow(ax, (dynamics_center[0] + 1.4, dynamics_center[1] + 0.1),
               (viz_center[0] - 1.0, viz_center[1] - 0.1))

    # ================== 添加标注 ==================
    # 添加左侧模块分层标签
    ax.text(-0.3, 7.0, '设计层', fontsize=10, color='#2C3E50',
            fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.text(-0.3, 5.0, '数据层', fontsize=10, color='#2C3E50',
            fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.text(-0.3, 3.0, '传动层', fontsize=10, color='#2C3E50',
            fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.text(-0.3, 1.0, '协同层', fontsize=10, color='#2C3E50',
            fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.text(11.5, 7.8, '仿真层', fontsize=10, color='#2C3E50',
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # 标题
    ax.set_title('SDAS 模型软件系统架构与数据流', fontsize=14,
                 fontweight='bold', pad=15)

    # 保存
    path = os.path.join(OUTPUT_DIR, 'architecture_diagram.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.3)
    print(f"  [OK] {path}")
    plt.close(fig)


if __name__ == '__main__':
    print("生成图 3-1: SDAS 系统架构图...")
    generate_architecture_diagram()
    print("完成！")
