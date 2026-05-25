"""
力学数值仿真可视化工具包
=========================
将仿真过程（Phase 1/2）和结果以多面板看板形式可视化。

功能：
    1. 加载 .npz 保存的仿真轨迹或直接传入 SimulationTrajectory
    2. 关节位置/速度/加速度时间历程图
    3. 能量演化曲线（动能、势能、耗散能）
    4. 相平面图（q vs q_dot）
    5. 控制输入 u(t) 示意图
    6. 力矩分量分解（传动/弹性/摩擦/重力）
    7. 准静态平衡柱状图（Phase 1）
    8. 相位对比：合并多个轨迹进行比较
    9. 汇总统计信息面板

使用方式：
    # 从 .npz 文件加载并显示
    python -m src.visualization.simulation_visualizer --load results.npz

    # 直接传入 SimulationTrajectory
    from src.visualization.simulation_visualizer import visualize_simulation
    visualize_simulation(traj, design=design)
"""

import numpy as np
from typing import Optional, Dict, List, Union, Any
from dataclasses import dataclass, field
import os, sys, time, platform
from pathlib import Path


# ============================================================
#  matplotlib 中文字体配置
# ============================================================

def _setup_cjk_font():
    """尝试设置 matplotlib 以支持中文字体显示。"""
    import matplotlib as mpl
    import matplotlib.font_manager as fm

    cjk_candidates = [
        'Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans SC',
        'Source Han Sans SC', 'PingFang SC', 'STHeiti',
        'AR PL UMing CN', 'DejaVu Sans',
    ]

    installed = {f.name for f in fm.fontManager.ttflist}
    chosen = 'DejaVu Sans'
    for cjk in cjk_candidates:
        if cjk in installed:
            chosen = cjk
            break

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans', 'Arial']
    mpl.rcParams['font.monospace'] = [chosen, 'DejaVu Sans Mono', 'Consolas']
    mpl.rcParams['axes.unicode_minus'] = False

    return chosen != 'DejaVu Sans'


_HAS_CJK = _setup_cjk_font()

_LABEL_TRANSLATIONS_EN = {
    '关节角度时间历程': 'Joint Angle Trajectory',
    '关节速度时间历程': 'Joint Velocity Trajectory',
    '能量演化': 'Energy Evolution',
    '控制输入 u(t)': 'Control Inputs u(t)',
    '相平面': 'Phase Portrait',
    '力学仿真 Dashboard': 'Mechanics Simulation Dashboard',
    '关节角度 q(t)': 'Joint Angles q(t)',
    '关节速度 q̇(t)': 'Joint Velocities q̇(t)',
    '准静态平衡角度': 'Quasi-Static Equilibrium',
    '关节角度相关性': 'Joint Angle Correlation',
    '关节': 'Joint',
    '时间': 'Time',
    '角度': 'Angle',
    '速度': 'Velocity',
    '能量': 'Energy',
    '控制输入': 'Control Inputs',
    '动能': 'Kinetic',
    '势能': 'Potential',
    '耗散能': 'Dissipated',
    '总能量': 'Total',
    '传动': 'Transmission',
    '弹性': 'Elastic',
    '摩擦': 'Friction',
    '重力': 'Gravity',
    '起点': 'Start',
    '终点': 'End',
    '输入幅值': 'Input',
    '仿真统计': 'Simulation Statistics',
    '动能对比': 'Kinetic Energy Comparison',
    '关节角度对比': 'Joint Angle Comparison',
    '相位对比': 'Phase Comparison',
    '最终关节角度分布': 'Final Joint Angles',
    '相位对比统计': 'Phase Comparison Stats',
    '采样点数': 'Samples',
    '关节数': 'Joints',
    '时间范围': 'Time Range',
    '总步数': 'Steps',
    '最大关节行程': 'Max Joint Travel',
    '平均关节行程': 'Avg Joint Travel',
    '最大关节速率': 'Max Joint Speed',
    '峰值动能': 'Peak Kinetic Energy',
    '最终动能': 'Final Kinetic Energy',
    '积分方法': 'Integrator',
    '步长': 'dt',
    '求解耗时': 'Solve Time',
    '仿真阶段': 'Phase',
    '准静态力平衡': 'Quasi-Static Equilibrium',
    '力矩分量分解': 'Torque Components',
    '残差演化': 'Residual Evolution',
    'Phase': 'Phase',
    '准静态': 'Quasi-Static',
    '动力学': 'Dynamic',
    '完整动力学 ODE': 'Dynamic ODE',
}


def _tr(text):
    if _HAS_CJK:
        return text
    return _LABEL_TRANSLATIONS_EN.get(text, text)


# ============================================================
#  模拟轨迹加载
# ============================================================

def load_trajectory(path: str):
    from src.simulation.io import SimulationReader
    reader = SimulationReader(path)
    traj = reader.load_trajectory()
    metadata = reader.load_metadata()
    reader.close()
    return traj, metadata


def load_trajectory_simple(path: str):
    import json
    raw = np.load(path, allow_pickle=True)
    traj_keys = {'t', 'q', 'q_dot', 'q_ddot', 'inputs',
                 'energy_kinetic', 'energy_potential', 'energy_dissipated',
                 'residuals'}
    data = {k: raw[k] for k in traj_keys if k in raw}
    metadata = None
    if 'metadata' in raw:
        try:
            metadata = json.loads(str(raw['metadata']))
        except Exception:
            metadata = {'raw': str(raw['metadata'])}
    raw.close()
    return data, metadata


# ============================================================
#  2D 轨迹绘图（基于 matplotlib）
# ============================================================

class TrajectoryPlotter:
    """
    仿真轨迹的 2D 绘图引擎。

    生成多面板看板，包含：
        - Panel 1: 关节角度 q(t)
        - Panel 2: 关节速度 q_dot(t)
        - Panel 3: 能量演化
        - Panel 4: 相平面 (q vs q_dot)
        - Panel 5: 控制输入 u(t)
        - Panel 6: 力矩分量分解
    """

    def __init__(self, data: dict, n_joints: int = None):
        self.data = data
        self.t = np.asarray(data.get('t', []))
        self.q = np.asarray(data.get('q', []))
        self.q_dot = np.asarray(data.get('q_dot', [])) if data.get('q_dot') is not None else None
        self.q_ddot = np.asarray(data.get('q_ddot', [])) if data.get('q_ddot') is not None else None
        self.inputs = np.asarray(data.get('inputs', [])) if data.get('inputs') is not None else None
        self.energy_kinetic = np.asarray(data.get('energy_kinetic', []))
        self.energy_potential = np.asarray(data.get('energy_potential', []))
        self.energy_dissipated = np.asarray(data.get('energy_dissipated', []))
        self.residuals = np.asarray(data.get('residuals', [])) if data.get('residuals') is not None else None

        if n_joints is not None:
            self.n_joints = n_joints
        elif self.q.ndim >= 2:
            self.n_joints = self.q.shape[1]
        else:
            self.n_joints = 0

        self._figs = []

    def has_data(self) -> bool:
        return len(self.t) > 1 and self.q.size > 0

    def plot_joint_positions(self, figsize=(10, 4), joint_names=None,
                             title="关节角度时间历程"):
        import matplotlib.pyplot as plt
        if not self.has_data():
            return None
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        t = self.t
        for i in range(self.n_joints):
            label = joint_names[i] if (joint_names and i < len(joint_names)) else f'Joint {i}'
            ax.plot(t, np.degrees(self.q[:, i]), label=label, lw=1.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Joint angle [deg]')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._figs.append(fig)
        return fig

    def plot_joint_velocities(self, figsize=(10, 4), joint_names=None,
                              title="Joint Velocity Trajectory"):
        import matplotlib.pyplot as plt
        if not self.has_data() or self.q_dot is None:
            return None
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        t = self.t
        for i in range(self.n_joints):
            label = joint_names[i] if (joint_names and i < len(joint_names)) else f'Joint {i}'
            ax.plot(t, np.degrees(self.q_dot[:, i]), label=label, lw=1.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Joint velocity [deg/s]')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._figs.append(fig)
        return fig

    def plot_energy(self, figsize=(10, 4), title="Energy Evolution"):
        import matplotlib.pyplot as plt
        if not self.has_data():
            return None
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        t = self.t
        has_ek = len(self.energy_kinetic) == len(t) and np.any(np.abs(self.energy_kinetic) > 1e-15)
        has_ep = len(self.energy_potential) == len(t) and np.any(np.abs(self.energy_potential) > 1e-15)
        has_ed = len(self.energy_dissipated) == len(t) and np.any(np.abs(self.energy_dissipated) > 1e-15)
        if has_ek:
            ax.plot(t, self.energy_kinetic, label='Kinetic', lw=1.5)
        if has_ep:
            ax.plot(t, self.energy_potential, label='Potential', lw=1.5)
        if has_ed:
            ax.plot(t, self.energy_dissipated, label='Dissipated', lw=1.5)
        total = np.zeros(len(t))
        if has_ek: total += self.energy_kinetic.copy()
        if has_ep: total += self.energy_potential.copy()
        if has_ed: total += self.energy_dissipated.copy()
        if np.any(np.abs(total) > 1e-15):
            ax.plot(t, total, '--k', label='Total', lw=1.5, alpha=0.7)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Energy [J]')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._figs.append(fig)
        return fig

    def plot_phase_portraits(self, joint_indices: List[int] = None,
                             figsize=(10, 8), title_prefix="Phase Portrait"):
        import matplotlib.pyplot as plt
        if not self.has_data() or self.q_dot is None:
            return None
        if joint_indices is None:
            joint_indices = list(range(min(self.n_joints, 6)))
        n_plots = len(joint_indices)
        if n_plots == 0:
            return None
        n_cols = min(n_plots, 3)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()
        for idx, j_idx in enumerate(joint_indices):
            ax = axes[idx]
            q_j = np.degrees(self.q[:, j_idx])
            qd_j = np.degrees(self.q_dot[:, j_idx])
            ax.plot(q_j, qd_j, '-', alpha=0.6, color='steelblue', lw=1.0)
            ax.scatter(q_j[0], qd_j[0], c='green', s=50, marker='o',
                       label='Start', zorder=5, edgecolors='black')
            ax.scatter(q_j[-1], qd_j[-1], c='red', s=50, marker='s',
                       label='End', zorder=5, edgecolors='black')
            ax.set_xlabel('q [deg]')
            ax.set_ylabel('q̇ [deg/s]')
            ax.set_title(f'{title_prefix} - Joint {j_idx}')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        fig.tight_layout()
        self._figs.append(fig)
        return fig

    def plot_inputs(self, figsize=(10, 3.5), title="Control Inputs u(t)"):
        import matplotlib.pyplot as plt
        if self.inputs is None or len(self.inputs) == 0:
            return None
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        t = self.t[:len(self.inputs)]
        n_input_cols = min(self.inputs.shape[1], 3)
        labels = [r'$\tau_M \cdot \sigma$', r'$\sigma$', r'$\sigma_f$']
        for i in range(n_input_cols):
            ax.plot(t, self.inputs[:, i], label=labels[i], lw=1.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Input')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._figs.append(fig)
        return fig

    def plot_residuals(self, figsize=(10, 3), title="Residual Evolution"):
        import matplotlib.pyplot as plt
        if self.residuals is None or len(self.residuals) == 0:
            return None
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        t = self.t[:len(self.residuals)]
        res_norm = np.linalg.norm(self.residuals, axis=1) if self.residuals.ndim > 1 else self.residuals
        ax.semilogy(t, res_norm, 'r-', lw=1.5)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Residual norm')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._figs.append(fig)
        return fig

    def plot_torque_components(self, assembler=None, Q_tauM=None, Q_s=None, Q_sdot=None,
                                figsize=(10, 5), title="Torque Components"):
        import matplotlib.pyplot as plt
        if not self.has_data():
            return None
        n_steps = min(len(self.t), 200)
        step_every = max(1, len(self.t) // n_steps)
        idx = np.arange(0, len(self.t), step_every)[:n_steps]
        t_plot = self.t[idx]
        if assembler is not None:
            tau_trans = np.zeros((len(idx), self.n_joints))
            tau_elas = np.zeros((len(idx), self.n_joints))
            tau_fric = np.zeros((len(idx), self.n_joints))
            tau_grav = np.zeros((len(idx), self.n_joints))
            for k, step in enumerate(idx):
                q_k = self.q[step]
                qd_k = self.q_dot[step] if self.q_dot is not None else np.zeros(self.n_joints)
                u = self.inputs[step] if self.inputs is not None and step < len(self.inputs) else np.zeros(3)
                if Q_tauM is not None:
                    tau_trans[k] = Q_tauM * u[0] + Q_s * u[1] + Q_sdot * u[2]
                tau_elas[k] = assembler.compute_elastic_torque(q_k) if hasattr(assembler, 'compute_elastic_torque') else np.zeros(self.n_joints)
                tau_fric[k] = assembler.compute_friction_torque(q_k, qd_k) if hasattr(assembler, 'compute_friction_torque') else np.zeros(self.n_joints)
            norm_trans = np.linalg.norm(tau_trans, axis=1)
            norm_elas = np.linalg.norm(tau_elas, axis=1)
            norm_fric = np.linalg.norm(tau_fric, axis=1)
            norm_grav = np.linalg.norm(tau_grav, axis=1)
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(t_plot, norm_trans, label='Transmission', lw=1.5)
            ax.plot(t_plot, norm_elas, label='Elastic', lw=1.5)
            ax.plot(t_plot, norm_fric, label='Friction', lw=1.5)
            if np.any(norm_grav > 1e-10):
                ax.plot(t_plot, norm_grav, label='Gravity', lw=1.5)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Torque norm [Nm]')
            ax.set_title(title)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            self._figs.append(fig)
            return fig
        return None

    def plot_dashboard(self, joint_names=None, joint_indices_for_phase=None,
                       figsize=(16, 12), title="Mechanics Simulation Dashboard"):
        import matplotlib.pyplot as plt
        if not self.has_data():
            print("[Visualizer] No trajectory data.")
            return None
        if joint_indices_for_phase is None:
            joint_indices_for_phase = list(range(min(self.n_joints, 3)))
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        ax1 = axes[0, 0]
        t = self.t
        for i in range(self.n_joints):
            label = joint_names[i] if (joint_names and i < len(joint_names)) else f'J{i}'
            ax1.plot(t, np.degrees(self.q[:, i]), label=label, lw=1.2)
        ax1.set_ylabel('Angle [deg]')
        ax1.set_title('Joint Angles q(t)')
        ax1.legend(fontsize=7, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax2 = axes[0, 1]
        if self.q_dot is not None:
            for i in range(self.n_joints):
                label = joint_names[i] if (joint_names and i < len(joint_names)) else f'J{i}'
                ax2.plot(t, np.degrees(self.q_dot[:, i]), label=label, lw=1.2)
        ax2.set_ylabel('Velocity [deg/s]')
        ax2.set_title('Joint Velocities q̇(t)')
        ax2.legend(fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax3 = axes[1, 0]
        has_ek = len(self.energy_kinetic) == len(t) and np.any(np.abs(self.energy_kinetic) > 1e-15)
        has_ep = len(self.energy_potential) == len(t) and np.any(np.abs(self.energy_potential) > 1e-15)
        has_ed = len(self.energy_dissipated) == len(t) and np.any(np.abs(self.energy_dissipated) > 1e-15)
        if has_ek: ax3.plot(t, self.energy_kinetic, label='Kinetic', lw=1.2)
        if has_ep: ax3.plot(t, self.energy_potential, label='Potential', lw=1.2)
        if has_ed: ax3.plot(t, self.energy_dissipated, label='Dissipated', lw=1.2)
        total = np.zeros(len(t))
        if has_ek: total += self.energy_kinetic
        if has_ep: total += self.energy_potential
        if has_ed: total += self.energy_dissipated
        if np.any(np.abs(total) > 1e-15):
            ax3.plot(t, total, '--k', label='Total', lw=1.5, alpha=0.6)
        ax3.set_ylabel('Energy [J]')
        ax3.set_title('Energy Evolution')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax4 = axes[1, 1]
        if self.q_dot is not None:
            colors = plt.cm.viridis(np.linspace(0, 1, len(joint_indices_for_phase)))
            for cidx, j_idx in enumerate(joint_indices_for_phase):
                q_j = np.degrees(self.q[:, j_idx])
                qd_j = np.degrees(self.q_dot[:, j_idx])
                ax4.plot(q_j, qd_j, '-', color=colors[cidx], lw=1.0, alpha=0.7,
                         label=f'J{j_idx}')
                ax4.scatter(q_j[0], qd_j[0], color=colors[cidx], s=30, marker='o', zorder=5)
                ax4.scatter(q_j[-1], qd_j[-1], color=colors[cidx], s=30, marker='s', zorder=5)
        ax4.set_xlabel('q [deg]')
        ax4.set_ylabel('q̇ [deg/s]')
        ax4.set_title('Phase Portrait')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax5 = axes[2, 0]
        if self.inputs is not None and len(self.inputs) > 0:
            t_in = t[:len(self.inputs)]
            n_cols = min(self.inputs.shape[1], 3)
            labels = [r'$\tau_M\sigma$', r'$\sigma$', r'$\sigma_f$']
            for i in range(n_cols):
                ax5.plot(t_in, self.inputs[:, i], label=labels[i], lw=1.5)
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel('Input')
        ax5.set_title('Control Inputs u(t)')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax6 = axes[2, 1]
        ax6.axis('off')
        info_lines = ["Simulation Statistics", "=" * 20,
                      f"Samples: {len(t)}",
                      f"Joints: {self.n_joints}",
                      f"Time range: [{t[0]:.4f}, {t[-1]:.4f}] s",
                      f"Steps: {len(t)}"]
        if self.q.size > 0:
            q_range = np.degrees(np.max(self.q, axis=0) - np.min(self.q, axis=0))
            info_lines.append(f"Max joint travel: {np.max(q_range):.1f} deg")
            info_lines.append(f"Avg joint travel: {np.mean(q_range):.1f} deg")
        if self.q_dot is not None and self.q_dot.size > 0:
            info_lines.append(f"Max joint speed: {np.max(np.abs(np.degrees(self.q_dot))):.1f} deg/s")
        if has_ek:
            info_lines.append(f"Peak kinetic energy: {np.max(self.energy_kinetic):.4f} J")
            info_lines.append(f"Final kinetic energy: {self.energy_kinetic[-1]:.6f} J")
        if 'info' in self.data and isinstance(self.data['info'], dict):
            info = self.data['info']
            info_lines.append(f"---")
            if 'method' in info:
                info_lines.append(f"Integrator: {info['method']}")
            if 'dt' in info:
                info_lines.append(f"dt: {info['dt']:.2e} s")
            if 'elapsed_s' in info:
                info_lines.append(f"Solve time: {info['elapsed_s']*1000:.1f} ms")
            if 'phase' in info:
                info_lines.append(f"Phase: {info['phase']}")
        if 'metadata' in self.data and isinstance(self.data['metadata'], str):
            import json
            try:
                meta = json.loads(self.data['metadata'])
                info_lines.append(f"---")
                for k, v in meta.items():
                    info_lines.append(f"{k}: {v}")
            except Exception:
                pass
        ax6.text(0.05, 0.95, '\n'.join(info_lines),
                 transform=ax6.transAxes, fontsize=9, fontfamily='monospace',
                 verticalalignment='top', horizontalalignment='left')
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        self._figs.append(fig)
        return fig

    def plot_quasistatic_bars(self, q_eq: np.ndarray, q0: np.ndarray = None,
                               joint_names=None, figsize=(10, 5),
                               title="Quasi-Static Equilibrium"):
        import matplotlib.pyplot as plt
        if q_eq is None or len(q_eq) == 0:
            return None
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        n_j = len(q_eq)
        q_deg = np.degrees(q_eq)
        x = np.arange(n_j)
        labels = [joint_names[i] if (joint_names and i < len(joint_names)) else f'Joint {i}' for i in range(n_j)]
        bars = ax.bar(x, q_deg, width=0.6, alpha=0.7, color='steelblue', label='Equilibrium q')
        if q0 is not None:
            q0_deg = np.degrees(q0)
            ax.scatter(x, q0_deg, c='red', s=50, marker='x', label='Initial q0', zorder=5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Angle [deg]')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        for bar, val in zip(bars, q_deg):
            offset = 5 if val >= 0 else -15
            va = 'bottom' if val >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f'{val:.1f}°', ha='center', va=va, fontsize=8)
        fig.tight_layout()
        self._figs.append(fig)
        return fig

    def plot_joint_correlation(self, figsize=(8, 7), title="Joint Angle Correlation"):
        import matplotlib.pyplot as plt
        if not self.has_data() or self.n_joints < 2:
            return None
        corr = np.corrcoef(self.q.T)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        n = self.n_joints
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                        fontsize=8, color='black' if abs(corr[i, j]) < 0.5 else 'white')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xlabel('Joint index')
        ax.set_ylabel('Joint index')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        self._figs.append(fig)
        return fig

    def show(self, block: bool = True):
        import matplotlib.pyplot as plt
        plt.show(block=block)

    def save_figs(self, prefix: str = "sim_viz", dpi: int = 150,
                   formats: List[str] = None):
        if formats is None:
            formats = ['png']
        import matplotlib.pyplot as plt
        for i, fig in enumerate(self._figs):
            for fmt in formats:
                fname = f"{prefix}_{i}.{fmt}"
                fig.savefig(fname, dpi=dpi, bbox_inches='tight')
                print(f"  Saved {fname}")

    def close_all(self):
        import matplotlib.pyplot as plt
        for fig in self._figs:
            plt.close(fig)
        self._figs = []


# ============================================================
#  相位对比绘图
# ============================================================

def plot_phase_comparison(trajs: List[dict], labels: List[str] = None,
                          figsize=(14, 6), title="Phase Comparison"):
    import matplotlib.pyplot as plt
    n = len(trajs)
    if n == 0:
        return None
    if labels is None:
        labels = [f'Traj {i}' for i in range(n)]
    n_joints = max(t['q'].shape[1] if t.get('q') is not None else 0 for t in trajs)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    ax1 = axes[0, 0]
    for i, (t_data, label) in enumerate(zip(trajs, labels)):
        t = t_data['t']
        if t_data.get('q') is not None and t_data['q'].shape[1] > 0:
            ax1.plot(t, np.degrees(t_data['q'][:, 0]), color=colors[i],
                     label=label, lw=1.5)
    ax1.set_ylabel('Joint 0 angle [deg]')
    ax1.set_title('Joint Angle Comparison')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2 = axes[0, 1]
    for i, (t_data, label) in enumerate(zip(trajs, labels)):
        t = t_data['t']
        ek = t_data.get('energy_kinetic')
        if ek is not None and len(ek) > 0:
            ax2.plot(t, ek, color=colors[i], label=label, lw=1.5)
    ax2.set_ylabel('Kinetic energy [J]')
    ax2.set_title('Kinetic Energy Comparison')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax3 = axes[1, 0]
    x = np.arange(n_joints)
    width = 0.8 / max(n, 1)
    for i, (t_data, label) in enumerate(zip(trajs, labels)):
        if t_data.get('q') is not None:
            q_final = np.degrees(t_data['q'][-1])
            ax3.bar(x + i * width - 0.4 + width / 2, q_final, width,
                    color=colors[i], label=label, alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_ylabel('Final angle [deg]')
    ax3.set_title('Final Joint Angles')
    ax3.legend(fontsize=8)
    ax3.grid(True, axis='y', alpha=0.3)
    ax4 = axes[1, 1]
    ax4.axis('off')
    lines = ["Phase Comparison Stats", "=" * 20]
    for i, (t_data, label) in enumerate(zip(trajs, labels)):
        t = t_data['t']
        n_steps = len(t)
        duration = t[-1] - t[0]
        lines.append(f"\n{label}:")
        lines.append(f"  Steps: {n_steps}, Duration: {duration*1000:.1f}ms")
        ek = t_data.get('energy_kinetic')
        if ek is not None and len(ek) > 0:
            lines.append(f"  Peak kinetic: {np.max(ek):.4f} J")
        q = t_data.get('q')
        if q is not None and q.size > 0:
            q_range = np.max(np.degrees(q), axis=0) - np.min(np.degrees(q), axis=0)
            lines.append(f"  Max joint travel: {np.max(q_range):.1f} deg")
    ax4.text(0.05, 0.95, '\n'.join(lines), transform=ax4.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top')
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    return fig


# ============================================================
#  统一可视化入口
# ============================================================

class SimulationVisualizer:
    """
    统一可视化接口。

    支持从 .npz 文件加载轨迹，并生成 2D 绘图。
    """

    def __init__(self, traj_data: dict = None, design=None,
                 metadata: dict = None):
        self.traj_data = traj_data or {}
        self.design = design
        self.metadata = metadata or {}
        self._plotter = None

    @classmethod
    def from_npz(cls, path: str, design=None) -> 'SimulationVisualizer':
        data, metadata = load_trajectory_simple(path)
        return cls(traj_data=data, design=design, metadata=metadata)

    def update_trajectory(self, traj_data: dict):
        self.traj_data = traj_data
        self._plotter = None

    def get_plotter(self) -> TrajectoryPlotter:
        if self._plotter is None:
            self._plotter = TrajectoryPlotter(self.traj_data)
        return self._plotter

    def plot_joint_positions(self, **kwargs):
        return self.get_plotter().plot_joint_positions(**kwargs)

    def plot_joint_velocities(self, **kwargs):
        return self.get_plotter().plot_joint_velocities(**kwargs)

    def plot_energy(self, **kwargs):
        return self.get_plotter().plot_energy(**kwargs)

    def plot_phase_portraits(self, **kwargs):
        return self.get_plotter().plot_phase_portraits(**kwargs)

    def plot_inputs(self, **kwargs):
        return self.get_plotter().plot_inputs(**kwargs)

    def plot_residuals(self, **kwargs):
        return self.get_plotter().plot_residuals(**kwargs)

    def plot_dashboard(self, **kwargs):
        return self.get_plotter().plot_dashboard(**kwargs)

    def plot_quasistatic_bars(self, q_eq, **kwargs):
        return self.get_plotter().plot_quasistatic_bars(q_eq, **kwargs)

    def plot_joint_correlation(self, **kwargs):
        return self.get_plotter().plot_joint_correlation(**kwargs)

    def plot_torque_components(self, **kwargs):
        return self.get_plotter().plot_torque_components(**kwargs)

    def show_all(self, block: bool = True):
        if self._plotter is not None:
            self._plotter.show(block=block)

    def save_all(self, prefix: str = "sim_viz"):
        if self._plotter is not None:
            self._plotter.save_figs(prefix=prefix)

    def close(self):
        if self._plotter is not None:
            self._plotter.close_all()


def visualize_simulation(traj_data: dict = None, design=None,
                          npz_path: str = None,
                          show_dashboard: bool = True,
                          show_energy: bool = True,
                          show_phase: bool = True,
                          show_inputs: bool = True,
                          figsize: tuple = (16, 12),
                          save: bool = False,
                          save_prefix: str = "sim_viz",
                          block: bool = True,
                          **kwargs):
    if npz_path is not None and traj_data is None:
        viz = SimulationVisualizer.from_npz(npz_path, design=design)
    else:
        viz = SimulationVisualizer(traj_data=traj_data, design=design)
    plotter = viz.get_plotter()
    if not plotter.has_data():
        print("[Visualizer] No valid trajectory data.")
        return viz
    if show_dashboard:
        viz.plot_dashboard(figsize=figsize)
    if show_energy:
        viz.plot_energy()
    if show_phase:
        viz.plot_phase_portraits()
    if show_inputs:
        viz.plot_inputs()
    if save:
        viz.save_all(prefix=save_prefix)
    if show_dashboard or show_energy or show_phase or show_inputs:
        viz.show_all(block=block)
    return viz


# ============================================================
#  独立运行入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Simulation Result Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.visualization.simulation_visualizer --load results.npz
    python -m src.visualization.simulation_visualizer --load results.npz --save
        """
    )
    parser.add_argument("--load", type=str, default=None,
                        help="Simulation result .npz file path")
    parser.add_argument("--dashboard", action="store_true", default=True)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--prefix", type=str, default="sim_viz")
    parser.add_argument("--no-block", action="store_true")
    parser.add_argument("--phase-compare", nargs="+", default=None)

    args = parser.parse_args()

    if args.phase_compare is not None and len(args.phase_compare) >= 2:
        trajs = []
        labels = []
        for f in args.phase_compare:
            data, meta = load_trajectory_simple(f)
            trajs.append(data)
            label = os.path.splitext(os.path.basename(f))[0]
            labels.append(label)
        fig = plot_phase_comparison(trajs, labels)
        import matplotlib.pyplot as plt
        plt.show(block=not args.no_block)
        return

    if args.load is None:
        parser.print_help()
        print("\nUse --load to specify a .npz file.")
        return

    if not os.path.exists(args.load):
        print(f"[Error] File not found: {args.load}")
        return

    print(f"  Loading simulation result: {args.load}")
    viz = SimulationVisualizer.from_npz(args.load)
    plotter = viz.get_plotter()
    if not plotter.has_data():
        print("[Error] Invalid trajectory data.")
        return

    if not args.no_dashboard:
        viz.plot_dashboard()
    else:
        viz.plot_joint_positions()
        viz.plot_joint_velocities()
        viz.plot_energy()
        viz.plot_phase_portraits()
        viz.plot_inputs()

    if args.save:
        viz.save_all(prefix=args.prefix)
    viz.show_all(block=not args.no_block)
    print("  Visualization complete.")


if __name__ == "__main__":
    main()
