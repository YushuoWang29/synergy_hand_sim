#!/usr/bin/env python3
"""生成剩余缺失的占位图（mujoco_sdas_gui 和 sdas_vs_augmented）"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    matplotlib.font_manager.findfont('SimSun')
    plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'DejaVu Sans']
except Exception:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def fig_mujoco_gui():
    """MuJoCo 交互仿真器界面占位图"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 主视口
    main = FancyBboxPatch((0.5, 1.5), 6, 5.5, boxstyle="round,pad=0.1",
                          facecolor='#EBF5FB', edgecolor='#2C3E50', linewidth=1.5)
    ax.add_patch(main)
    ax.text(3.5, 5.0, 'MuJoCo 3D 视口\n(折纸手三维渲染)', ha='center', va='center',
            fontsize=13, fontweight='bold', color='#1F618D')

    # 滑块面板
    panel = FancyBboxPatch((7.0, 1.5), 2.8, 5.5, boxstyle="round,pad=0.1",
                           facecolor='#FEF9E7', edgecolor='#2C3E50', linewidth=1.5)
    ax.add_patch(panel)
    ax.text(8.4, 5.8, '控制面板', ha='center', va='center',
            fontsize=10, fontweight='bold')

    sliders = [
        (6.3, 'Motor A'), (5.7, 'Motor B'),
        (5.1, r'$\sigma$'), (4.5, r'$\sigma_f$')
    ]
    for y_pos, label in sliders:
        ax.plot([7.3, 9.3], [y_pos, y_pos], 'k-', lw=5, alpha=0.2)
        ax.text(7.1, y_pos, label, ha='right', va='center', fontsize=9, fontweight='bold')
        knob = np.random.uniform(7.5, 9.0)
        ax.plot(knob, y_pos, 'o', color='#E74C3C', markersize=8)

    ax.set_title('图 3-5  MuJoCo 交互仿真器界面（示意图）', fontsize=11,
                 fontweight='bold', pad=10)
    fig.tight_layout()
    return fig


def fig_sdas_vs_augmented():
    """SDAS vs Augmented 对比图"""
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))

    n_joints = 10
    joints = np.arange(n_joints)
    beta = 0.09
    N = n_joints + 4

    # R_A/R_B 模拟
    R_A_vals = np.array([np.exp(-beta * (i + 2)) for i in range(n_joints)])
    R_B_vals = np.array([np.exp(-beta * (N - 1 - (i + 2))) for i in range(n_joints)])
    R_A_vals = R_A_vals / R_A_vals.sum()
    R_B_vals = R_B_vals / R_B_vals.sum()

    sigma = 5.0
    sigma_f = 2.0

    # SDAS 模型（双马达模式）
    a = np.dot(R_A_vals, R_A_vals)
    b = np.dot(R_A_vals, R_B_vals)
    c = np.dot(R_B_vals, R_B_vals)
    Delta = a * c - b**2
    S_A = (c * R_A_vals - b * R_B_vals) / Delta
    S_B = (-b * R_A_vals + a * R_B_vals) / Delta
    q_sdas = S_A * (sigma + sigma_f) + S_B * (sigma - sigma_f)

    # Augmented 模型：R = (R_A + R_B)/2, Rf 为模拟的摩擦冻结
    R_avg = (R_A_vals + R_B_vals) / 2.0
    Rf_vals = R_A_vals - R_B_vals  # 模拟的差模向量
    Rf_vals = Rf_vals / np.max(np.abs(Rf_vals)) * 0.5 * np.max(R_avg)

    a_aug = np.dot(R_avg, R_avg)
    b_aug = np.dot(R_avg, Rf_vals)
    c_aug = np.dot(Rf_vals, Rf_vals)
    Delta_aug = a_aug * c_aug - b_aug**2
    if abs(Delta_aug) > 1e-15:
        S_sigma = (c_aug * R_avg - b_aug * Rf_vals) / Delta_aug
        S_diff = (-b_aug * R_avg + a_aug * Rf_vals) / Delta_aug
    else:
        S_sigma = R_avg / a_aug
        S_diff = np.zeros_like(Rf_vals)
    q_aug = S_sigma * sigma + S_diff * sigma_f

    ax.bar(joints - 0.2, q_sdas, width=0.4, label='SDAS 模型', color='#3498DB', alpha=0.85)
    ax.bar(joints + 0.2, q_aug, width=0.4, label='Augmented 模型', color='#E74C3C', alpha=0.65, hatch='//')

    ax.set_xlabel('关节序号', fontsize=11)
    ax.set_ylabel('关节角度 (rad)', fontsize=11)
    ax.set_xticks(joints)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('图 3-6  SDAS 模型与 Augmented 模型的关节角度对比\n'
                 r'($\sigma = 5$ rad, $\sigma_f = 2$ rad)',
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig


def main():
    print("生成剩余的占位图...")
    from generate_chapter3_figures import savefig
    savefig(fig_mujoco_gui(), 'mujoco_sdas_gui.png')
    savefig(fig_sdas_vs_augmented(), 'sdas_vs_augmented.png')
    print("完成!")


if __name__ == '__main__':
    main()
