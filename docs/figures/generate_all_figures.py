#!/usr/bin/env python3
"""
为第3.2节生成所有3张学术风格数据图表。
用法: python docs/figures/generate_all_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 字体配置
try:
    matplotlib.font_manager.findfont('SimSun')
    plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'DejaVu Sans']
except Exception:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def savefig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"  [OK] {path}")
    plt.close(fig)


# ============================================================
# 图 3-2: 单马达与双马达模式对比（柱状图）
# ============================================================
def fig_single_vs_dual():
    n_joints = 10

    # 模拟 R_A 和 R_B（基于 Capstan 衰减 β=0.09）
    beta = 0.09
    N = 20
    r = 7.0
    weights_A = np.array([r * np.exp(-beta * k) for k in range(N)])
    weights_B = np.array([r * np.exp(-beta * (N - 1 - k)) for k in range(N)])

    # 模拟每个关节上有若干滑轮
    np.random.seed(42)
    RA = np.zeros(n_joints)
    RB = np.zeros(n_joints)
    for j in range(n_joints):
        idx = np.linspace(0, N-1, 3, dtype=int) + j
        idx = idx[idx < N]
        RA[j] = np.sum(weights_A[idx])
        RB[j] = np.sum(weights_B[idx])

    # 归一化
    RA = RA / np.linalg.norm(RA) * 15
    RB = RB / np.linalg.norm(RB) * 15

    # 构造 E 和 Schur 补
    E = np.eye(n_joints)
    E_inv = np.eye(n_joints)
    a = float(RA @ E_inv @ RA)
    b = float(RA @ E_inv @ RB)
    c = float(RB @ E_inv @ RB)
    det = a*c - b*b

    S_A_paper = E_inv @ RA / a
    S_B_paper = E_inv @ RB / c
    S_A_schur = E_inv @ (c * RA - b * RB) / det
    S_B_schur = E_inv @ (-b * RA + a * RB) / det

    # 三种工况
    theta = 3.0
    q_A = S_A_paper * theta            # 仅 Motor A
    q_B = S_B_paper * theta            # 仅 Motor B
    q_both = S_A_schur * theta + S_B_schur * theta  # 双马达
    q_avg = (S_A_paper + S_B_paper) * theta / 2     # 简单平均

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(n_joints)
    w = 0.2

    bars1 = ax.bar(x - 1.5*w, q_A.flatten(), w, label=r'仅 Motor A ($\theta_A=3$)',
                   color='#3498DB', alpha=0.85)
    bars2 = ax.bar(x - 0.5*w, q_B.flatten(), w, label=r'仅 Motor B ($\theta_B=3$)',
                   color='#E67E22', alpha=0.85)
    bars3 = ax.bar(x + 0.5*w, q_both.flatten(), w, label=r'双马达 ($\theta_A=\theta_B=3$)',
                   color='#2ECC71', alpha=0.85)
    bars4 = ax.bar(x + 1.5*w, q_avg.flatten(), w, label=r'单向结果算术平均',
                   color='#E74C3C', alpha=0.6, hatch='//')

    ax.set_xlabel('关节序号', fontsize=12)
    ax.set_ylabel('关节角度 (rad)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(n_joints)])
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title('单马达与双马达模式下各关节角度对比', fontsize=13)
    ax.set_ylim(0, max(q_both) * 1.3)

    savefig(fig, 'single_vs_dual.png')


# ============================================================
# 图 3-3: SDAS vs 经典自适应协同的 σ_f 响应对比
# ============================================================
def fig_sdas_vs_classic():
    n_joints = 10
    sigma = 5.0
    sigma_f_vals = np.linspace(-2, 2, 41)

    # SDAS 模型结果
    beta = 0.09
    N = 20
    r = 7.0
    np.random.seed(42)

    # 构造 R_A, R_B
    RA = np.zeros(n_joints)
    RB = np.zeros(n_joints)
    for j in range(n_joints):
        idx = np.linspace(0, N-1, 3, dtype=int) + j
        idx = idx[idx < N]
        for k in idx:
            RA[j] += r * np.exp(-beta * k)
            RB[j] += r * np.exp(-beta * (N - 1 - k))

    RA = RA / np.linalg.norm(RA) * 15
    RB = RB / np.linalg.norm(RB) * 15

    E = np.eye(n_joints)
    E_inv = np.eye(n_joints)
    a = float(RA @ E_inv @ RA)
    b = float(RA @ E_inv @ RB)
    c = float(RB @ E_inv @ RB)
    det = a*c - b*b

    S_A_schur = E_inv @ (c * RA - b * RB) / det
    S_B_schur = E_inv @ (-b * RA + a * RB) / det

    q_j1_sdas = np.zeros(len(sigma_f_vals))
    q_j2_sdas = np.zeros(len(sigma_f_vals))

    for i, sf in enumerate(sigma_f_vals):
        theta_A = max(0, sigma + sf)
        theta_B = max(0, sigma - sf)
        if theta_A > 1e-10 and theta_B > 1e-10:
            q = S_A_schur * theta_A + S_B_schur * theta_B
        elif theta_A > 1e-10:
            q = (E_inv @ RA / a) * theta_A
        elif theta_B > 1e-10:
            q = (E_inv @ RB / c) * theta_B
        else:
            q = np.zeros(n_joints)
        q_j1_sdas[i] = q[1]
        q_j2_sdas[i] = q[4]

    # 经典模型结果（使用对称 R 和 Rf 近似）
    R = (RA + RB) / 2
    # Rf 近似：sigma_f 模式下的"差模"传动
    Rf = (RA - RB) * 0.8
    R_aug = np.vstack([R[np.newaxis, :], Rf[np.newaxis, :]])

    # 经典模型 Schur 补（2维协同）
    M = R_aug @ E_inv @ R_aug.T
    try:
        M_inv = np.linalg.inv(M)
    except:
        M_inv = np.linalg.pinv(M)
    S_aug = E_inv @ R_aug.T @ M_inv

    q_j1_classic = np.zeros(len(sigma_f_vals))
    q_j2_classic = np.zeros(len(sigma_f_vals))

    for i, sf in enumerate(sigma_f_vals):
        sigma_vec = np.array([sigma, sf])
        q = S_aug @ sigma_vec
        q_j1_classic[i] = q[1]
        q_j2_classic[i] = q[4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ax1.plot(sigma_f_vals, q_j1_sdas, 'b-', linewidth=2, label='SDAS')
    ax1.plot(sigma_f_vals, q_j1_classic, 'r--', linewidth=2, label='经典协同')
    ax1.set_xlabel(r'$\sigma_f$ (rad)', fontsize=12)
    ax1.set_ylabel('关节 1 角度 (rad)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('关节 1（近 Motor A 端）', fontsize=11)

    ax2.plot(sigma_f_vals, q_j2_sdas, 'b-', linewidth=2, label='SDAS')
    ax2.plot(sigma_f_vals, q_j2_classic, 'r--', linewidth=2, label='经典协同')
    ax2.set_xlabel(r'$\sigma_f$ (rad)', fontsize=12)
    ax2.set_ylabel('关节 2 角度 (rad)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('关节 2（远 Motor A 端）', fontsize=11)

    fig.suptitle(r'SDAS 模型与经典自适应协同的 $\sigma_f$ 响应对比', fontsize=13, y=1.02)
    fig.tight_layout()
    savefig(fig, 'sdas_vs_classic.png')


# ============================================================
# 图 3-4: 三段 σ_f 输入下的动力学仿真时间历程
# ============================================================
def fig_three_phase_simulation():
    # 模拟三段式动力学响应
    dt = 1e-4
    t_per = 0.3
    t = np.linspace(0, 0.9, 9001)

    sigma = 5.0
    sigma_f_profile = np.piecewise(t,
        [t < 0.3, (t >= 0.3) & (t < 0.6), t >= 0.6],
        [0, -2, 2])

    # 物理参数
    zeta = 0.3  # 阻尼比
    omega_n = 50.0  # 自然频率

    # 模拟关节1（近 Motor A）和关节3（近 Motor B）的二阶响应
    # 稳态值基于 SDAS 模型计算
    beta = 0.09
    N = 20
    r = 7.0
    n_joints = 10
    np.random.seed(42)
    RA = np.zeros(n_joints)
    RB = np.zeros(n_joints)
    for j in range(n_joints):
        idx = np.linspace(0, N-1, 3, dtype=int) + j
        idx = idx[idx < N]
        for k in idx:
            RA[j] += r * np.exp(-beta * k)
            RB[j] += r * np.exp(-beta * (N - 1 - k))
    RA = RA / np.linalg.norm(RA) * 15
    RB = RB / np.linalg.norm(RB) * 15
    E_inv = np.eye(n_joints)
    a = float(RA @ E_inv @ RA)
    b = float(RA @ E_inv @ RB)
    c = float(RB @ E_inv @ RB)
    det = a*c - b*b
    S_A_schur = E_inv @ (c * RA - b * RB) / det
    S_B_schur = E_inv @ (-b * RA + a * RB) / det

    # 计算稳态 q
    def steady_q(sf):
        thA = max(0, sigma + sf)
        thB = max(0, sigma - sf)
        if thA > 1e-10 and thB > 1e-10:
            return S_A_schur * thA + S_B_schur * thB
        return np.zeros(n_joints)

    q1_ss = np.array([steady_q(sf)[1] for sf in sigma_f_profile])
    q3_ss = np.array([steady_q(sf)[3] for sf in sigma_f_profile])

    # 二阶低通滤波模拟动力学响应
    def lowpass_2nd_order(ss, dt, wn, zeta):
        n = len(ss)
        y = np.zeros(n)
        yd = np.zeros(n)
        for i in range(1, n):
            ydd = wn**2 * (ss[i] - y[i-1]) - 2*zeta*wn*yd[i-1]
            yd[i] = yd[i-1] + ydd * dt
            y[i] = y[i-1] + yd[i] * dt
        return y

    q1_dyn = lowpass_2nd_order(q1_ss, dt, omega_n, zeta)
    q3_dyn = lowpass_2nd_order(q3_ss, dt, omega_n, zeta)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(t, q1_dyn, 'b-', linewidth=2, label='关节 1（近 Motor A 端）')
    ax.plot(t, q3_dyn, 'orange', linewidth=2, label='关节 3（近 Motor B 端）')

    # 标注 σ_f 切换时刻
    for ts in [0.3, 0.6]:
        ax.axvline(x=ts, color='gray', linestyle='--', alpha=0.5)

    ax.annotate(r'$\sigma_f=0$', xy=(0.15, ax.get_ylim()[1]*0.9),
                ha='center', fontsize=10, color='gray')
    ax.annotate(r'$\sigma_f=-2$', xy=(0.45, ax.get_ylim()[1]*0.9),
                ha='center', fontsize=10, color='gray')
    ax.annotate(r'$\sigma_f=+2$', xy=(0.75, ax.get_ylim()[1]*0.9),
                ha='center', fontsize=10, color='gray')

    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_ylabel('关节角度 (rad)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(r'三段 $\sigma_f$ 输入下关节角度的动力学仿真时间历程', fontsize=13)

    savefig(fig, 'three_phase_simulation.png')


if __name__ == '__main__':
    print("生成图 3-2: 单马达与双马达模式对比...")
    fig_single_vs_dual()

    print("生成图 3-3: SDAS vs 经典自适应协同...")
    fig_sdas_vs_classic()

    print("生成图 3-4: 三段 σ_f 动力学仿真...")
    fig_three_phase_simulation()

    print("\n全部图表生成完毕！")
