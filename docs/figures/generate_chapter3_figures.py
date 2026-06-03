#!/usr/bin/env python3
"""
为第三章第 3.2 节生成学术风格示意图。
需要本地安装的字体：SimSun（宋体）、SimHei（黑体）
如果字体缺失，会自动回退到系统默认字体。

用法：
    cd docs/figures
    python generate_chapter3_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# ---- 输出目录 ----
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- 字体配置 ----
try:
    matplotlib.font_manager.findfont('SimSun')
    plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'DejaVu Sans']
except Exception:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def savefig(fig, name):
    """保存图片到输出目录"""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  [OK] {path}")
    plt.close(fig)


# ============================================================
# 图 3-1: 软件模块数据流图（框图文）
# ============================================================
def fig_sdas_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 颜色定义
    c_build = '#D6EAF8'  # 浅蓝 - 构建阶段
    c_run  = '#D5F5E3'  # 浅绿 - 运行阶段
    c_data = '#FADBD8'  # 浅红 - 数据
    edge_c = '#2C3E50'

    def draw_box(x, y, w, h, text, color=c_build, fontsize=9):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor=edge_c, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold')

    def draw_arrow(x1, y1, x2, y2, style='->', lw=1.5, color='#2C3E50', ls='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                    linestyle=ls))

    # 布局参数
    x_center = 5
    y_positions = [7.1, 5.8, 4.5, 3.2, 1.9]
    h = 0.85

    # 五个模块（自顶向下）
    modules = [
        (x_center - 2.8, y_positions[0], 5.6, h,
         "交互仿真层\n(MuJoCoSimulator)", c_run),
        (x_center - 3.0, y_positions[1], 6.0, h,
         "URDF 导出与映射层\n(origami_to_urdf.py)", c_build),
        (x_center - 2.8, y_positions[2], 5.6, h,
         "协同求解层\n(SDASModel)", c_run),
        (x_center - 3.0, y_positions[3], 6.0, h,
         "传动矩阵构建层\n(transmission_builder.py)", c_build),
        (x_center - 2.8, y_positions[4], 5.6, h,
         "数据模型层\n(OrigamiHandDesign)", c_data),
    ]

    for args in modules:
        draw_box(*args)

    # 模块之间的实线箭头（数据依赖）
    for i in range(4):
        y1 = y_positions[i] - h/2
        y2 = y_positions[i+1] + h/2
        draw_arrow(x_center, y1, x_center, y2, lw=1.8)

    # 右侧标注：构建 vs 运行
    ax.text(8.8, 6.2, '构建阶段\n(一次性)', ha='center', va='center',
            fontsize=8, color='#21618C', bbox=dict(facecolor=c_build, alpha=0.7,
                                                     edgecolor='none', boxstyle='round'))
    ax.text(8.8, 4.0, '运行阶段\n(每帧调用)', ha='center', va='center',
            fontsize=8, color='#1E8449', bbox=dict(facecolor=c_run, alpha=0.7,
                                                    edgecolor='none', boxstyle='round'))

    # 左侧：输入/输出标注
    ax.text(0.5, 7.1, '.ohd', fontsize=10, fontweight='bold', color='#A93226')
    ax.text(0.5, 1.5, 'URDF\n+ STL', fontsize=9, fontweight='bold', color='#6C3483')

    ax.set_title('图 3-1  SDAS 模型的软件架构与模块调用关系', fontsize=11,
                 fontweight='bold', pad=10)
    return fig


# ============================================================
# 图 3-2: Capstan 指数衰减导致 R_A 与 R_B 分布
# ============================================================
def fig_RA_RB_distribution():
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    n_joints = 10
    beta = 0.09
    # 假设路径元素 N = n_joints + 5 (两侧各有过渡)
    N = n_joints + 4
    # 虚构一个分布，模拟 ohd_2 设计
    np.random.seed(42)
    radii = 1.0 + 0.1 * np.random.randn(n_joints)
    radii = np.clip(radii, 0.5, 1.5)

    R_A = np.zeros(n_joints)
    R_B = np.zeros(n_joints)

    # 模拟从关节 0 到 9 依次布置滑轮
    for i in range(n_joints):
        d_A = i + 2   # 到 A 的路径步数
        d_B = N - 1 - (i + 2)
        R_A[i] = radii[i] * np.exp(-beta * d_A)
        R_B[i] = radii[i] * np.exp(-beta * d_B)

    # 归一化
    R_A = R_A / R_A.sum() * n_joints
    R_B = R_B / R_B.sum() * n_joints

    joints = np.arange(n_joints)
    ax.bar(joints - 0.18, R_A, width=0.36, label=r'$R_A$（从 Motor A 出发）',
           color='#3498DB', alpha=0.85, edgecolor='white')
    ax.bar(joints + 0.18, R_B, width=0.36, label=r'$R_B$（从 Motor B 出发）',
           color='#E74C3C', alpha=0.85, edgecolor='white')

    ax.set_xlabel('关节序号', fontsize=11)
    ax.set_ylabel('归一化传动比', fontsize=11)
    ax.set_xticks(joints)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('图 3-2  Capstan 指数衰减导致的 $R_A$ 与 $R_B$ 分布', fontsize=11,
                 fontweight='bold')
    fig.tight_layout()
    return fig


# ============================================================
# 图 3-3: SDAS 求解流程图
# ============================================================
def fig_sdas_flowchart():
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    c_start = '#AED6F1'
    c_dec = '#A9DFBF'
    c_if = '#F9E79F'
    c_calc = '#FADBD8'
    c_end = '#D5D8DC'

    def draw_rounded(x, y, w, h, text, color, fontsize=8):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor='#2C3E50', linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold')

    def draw_diamond(cx, cy, size, text, fontsize=8):
        from matplotlib.patches import Polygon
        pts = np.array([[cx, cy+size], [cx+size*1.3, cy],
                        [cx, cy-size], [cx-size*1.3, cy]])
        poly = Polygon(pts, closed=True, facecolor='#F9E79F',
                       edgecolor='#2C3E50', linewidth=1.2)
        ax.add_patch(poly)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold')

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color='#2C3E50'))

    # 流程框
    draw_rounded(3.5, 8.7, 3, 0.7, '输入 θA, θB', c_start)
    draw_rounded(3.5, 7.3, 3, 0.7, '松弛钳位\nθ = max(0, θ)', c_dec)
    draw_diamond(5, 6.0, 0.5, 'θA > ε\n&&\nθB > ε?')
    draw_diamond(3.2, 4.7, 0.5, 'θA > ε?')
    draw_diamond(6.8, 4.7, 0.5, 'θB > ε?')
    draw_rounded(1.5, 3.2, 3.2, 0.7, 'Schur 补公式\nq = SA_schur·θA + SB_schur·θB', c_calc)
    draw_rounded(3.5, 1.8, 3, 0.7, '单向公式\nq = SA_paper·θA', c_calc)
    draw_rounded(6.5, 1.8, 3, 0.7, '单向公式\nq = SB_paper·θB', c_calc)
    draw_rounded(8.5, 3.2, 2.5, 0.7, 'q = 0\n(松弛)', c_dec)
    draw_rounded(4, 0.5, 2, 0.7, '限位\n输出 q', c_end)

    # 箭头
    arrow(5, 8.7, 5, 8.0)
    arrow(5, 7.3, 5, 6.6)
    arrow(5, 5.5, 5, 4.8)
    ax.annotate('是', xy=(5.6, 5.4), fontsize=8)
    arrow(3.2, 5.5, 3.2, 4.8)
    ax.annotate('否', xy=(3.8, 5.7), fontsize=8)
    arrow(6.8, 5.5, 6.8, 4.8)
    ax.annotate('否', xy=(6.2, 5.7), fontsize=8)

    arrow(5, 4.2, 5, 3.6)  # 向下到 Schur 补
    arrow(3.2, 4.2, 3.2, 3.55)  # 向左
    ax.annotate('是', xy=(2.5, 4.5), fontsize=8)
    # 到 SA_paper
    arrow(3.2, 3.9, 3.2, 2.9)
    # SB 到 SB_paper
    arrow(6.8, 3.9, 6.8, 2.9)
    ax.annotate('是', xy=(7.5, 4.5), fontsize=8)
    # 双否 → q=0
    ax.plot([3.2, 2.5, 2.5, 8.5, 8.5, 9.75], [2.5, 2.5, 3.5, 3.5, 3.55, 3.55],
            color='#2C3E50', lw=1.2)
    ax.annotate('均否', xy=(5.8, 3.4), fontsize=8)

    # 汇总到限位
    arrow(3.1, 1.8, 4, 1.2)
    arrow(5, 1.8, 5, 1.2)
    ax.plot([6.5, 8, 8, 9.2], [2.5, 2.5, 2.9, 2.9],
            color='#2C3E50', lw=1.2)
    arrow(9.7, 3.2, 9.7, 0.5)

    ax.set_title('图 3-3  SDAS 求解器的完整工作流程', fontsize=11,
                 fontweight='bold', pad=10)
    return fig


# ============================================================
# 图 3-4: 单向 vs Schur 补的关节角度对比
# ============================================================
def fig_single_vs_dual():
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))

    n_joints = 10
    joints = np.arange(n_joints)

    # 模拟 R_A 衰减
    beta = 0.09
    N = n_joints + 4
    R_A_vals = np.array([np.exp(-beta * (i + 2)) for i in range(n_joints)])
    R_B_vals = np.array([np.exp(-beta * (N - 1 - (i + 2))) for i in range(n_joints)])
    # 归一化模拟
    R_A_vals = R_A_vals / R_A_vals.sum()
    R_B_vals = R_B_vals / R_B_vals.sum()

    theta_A = 3.0
    theta_B = 3.0

    # 单向 A
    q_A = R_A_vals * theta_A
    # 单向 B
    q_B = R_B_vals * theta_B
    # Schur 补（模拟交叉耦合）
    a = np.dot(R_A_vals, R_A_vals)
    b = np.dot(R_A_vals, R_B_vals)
    c = np.dot(R_B_vals, R_B_vals)
    Delta = a * c - b**2
    coeff_A = (c * R_A_vals - b * R_B_vals) / Delta
    coeff_B = (-b * R_A_vals + a * R_B_vals) / Delta
    q_dual = coeff_A * theta_A + coeff_B * theta_B
    # 简单平均
    q_avg = (q_A + q_B) / 2.0

    width = 0.22
    ax.bar(joints - width*1.5, q_A, width, label='仅 Motor A', color='#3498DB', alpha=0.85)
    ax.bar(joints - width*0.5, q_B, width, label='仅 Motor B', color='#E67E22', alpha=0.85)
    ax.bar(joints + width*0.5, q_dual, width, label='双马达 (Schur 补)', color='#2ECC71', alpha=0.85)
    ax.bar(joints + width*1.5, q_avg, width, label='单向平均', color='#9B59B6', alpha=0.5, hatch='//')

    ax.set_xlabel('关节序号', fontsize=11)
    ax.set_ylabel('关节角度 (rad)', fontsize=11)
    ax.set_xticks(joints)
    ax.legend(fontsize=8.5, ncol=2, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('图 3-4  单/双马达驱动下各关节角度对比', fontsize=11,
                 fontweight='bold')
    fig.tight_layout()
    return fig


# ============================================================
# 图 3-5: 三段 σ_f 输入下的关节角度时间历程
# ============================================================
def fig_three_phase_simulation():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.5),
                                    sharex=True,
                                    gridspec_kw={'height_ratios': [1, 2.5]})

    dt = 1e-4
    t_per = 0.3
    t_total = 0.9
    t = np.arange(0, t_total, dt)

    # σ_f 时间函数
    sigma_f = np.zeros_like(t)
    sigma_f[t < 0.3] = 0.0
    sigma_f[(t >= 0.3) & (t < 0.6)] = -2.0
    sigma_f[t >= 0.6] = 2.0

    sigma = 5.0 * np.ones_like(t)

    # 模拟带有平滑过渡的关节响应
    n_joints = 10
    beta = 0.09
    N = n_joints + 4
    R_A_vals = np.array([np.exp(-beta * (i + 2)) for i in range(n_joints)])
    R_B_vals = np.array([np.exp(-beta * (N - 1 - (i + 2))) for i in range(n_joints)])
    a = np.dot(R_A_vals, R_A_vals)
    b = np.dot(R_A_vals, R_B_vals)
    c = np.dot(R_B_vals, R_B_vals)
    Delta = a * c - b**2
    S_A = (c * R_A_vals - b * R_B_vals) / Delta
    S_B = (-b * R_A_vals + a * R_B_vals) / Delta

    # 加入低通滤波效果模拟实际动力学
    tau = 0.02  # 时间常数
    q_j0 = np.zeros_like(t)  # 关节 0
    q_j7 = np.zeros_like(t)  # 关节 7

    for i in range(1, len(t)):
        theta_A_t = sigma[i] + sigma_f[i]
        theta_B_t = sigma[i] - sigma_f[i]
        q_desired_0 = S_A[0] * theta_A_t + S_B[0] * theta_B_t
        q_desired_7 = S_A[7] * theta_A_t + S_B[7] * theta_B_t
        # 一阶低通
        q_j0[i] = q_j0[i-1] + (dt/tau) * (q_desired_0 - q_j0[i-1])
        q_j7[i] = q_j7[i-1] + (dt/tau) * (q_desired_7 - q_j7[i-1])

    # 上子图：σ_f 输入
    ax1.plot(t, sigma_f, '#E74C3C', lw=2)
    ax1.set_ylabel(r'$\sigma_f$ (rad)', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.axvline(0.3, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax1.axvline(0.6, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax1.set_title('图 3-5  三段 $\\sigma_f$ 输入下代表性关节的角度时间历程',
                  fontsize=11, fontweight='bold')
    ax1.set_ylim(-2.5, 2.5)

    # 下子图：关节角度
    ax2.plot(t, q_j0, '#3498DB', lw=1.5, label='关节 0（近 Motor A 侧）')
    ax2.plot(t, q_j7, '#E67E22', lw=1.5, label='关节 7（近 Motor B 侧）')
    ax2.axvline(0.3, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax2.axvline(0.6, color='gray', ls='--', lw=0.8, alpha=0.5)
    ax2.set_xlabel('时间 (s)', fontsize=11)
    ax2.set_ylabel('关节角度 (rad)', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ============================================================
# 主函数
# ============================================================
def main():
    print("正在生成第三章 3.2 节所需插图...\n")

    print("[1/5] SDAS 软件架构数据流图...")
    savefig(fig_sdas_architecture(), 'sdas_architecture.png')

    print("[2/5] R_A R_B 分布图...")
    savefig(fig_RA_RB_distribution(), 'RA_RB_distribution.png')

    print("[3/5] SDAS 求解流程图...")
    savefig(fig_sdas_flowchart(), 'sdas_flowchart.png')

    print("[4/5] 单向 vs Schur 补对比图...")
    savefig(fig_single_vs_dual(), 'single_vs_dual.png')

    print("[5/5] 三段仿真时间历程图...")
    savefig(fig_three_phase_simulation(), 'three_phase_simulation.png')

    print("\n全部图片生成完毕！")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
