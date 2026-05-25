# src/simulation/visualization.py
"""
Numerical simulation result visualization.

Provides:
    1. 2D trajectory plots (q-t, q_dot-t, energy-t) via matplotlib
    2. Phase portrait (q vs q_dot)
    3. Comparison: quasi-static equilibrium bar chart
    4. Control input plots

Usage
-----
    from src.simulation.simulation_visualizer import SimulationVisualizer
    viz = SimulationVisualizer(traj, design)
    viz.plot_trajectory()
    viz.plot_energy()
    viz.show()
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass

from .simulator import SimulationTrajectory


class SimulationPlotter:
    """
    2D trajectory plotting with matplotlib.

    Parameters
    ----------
    traj : SimulationTrajectory
        Simulation result to visualize.
    """

    def __init__(self, traj: SimulationTrajectory):
        self.traj = traj
        self._figs = []

    def plot_trajectory(self, title: str = "Joint Trajectory",
                        figsize=(12, 8)) -> List:
        """
        Plot joint positions and velocities vs time.

        Returns list of figure handles; use show() to display all at once.
        """
        import matplotlib.pyplot as plt

        n_j = self.traj.n_joints
        t = self.traj.t

        fig, (ax_q, ax_qd) = plt.subplots(2, 1, figsize=figsize,
                                           sharex=True)

        # --- Joint positions ---
        for i in range(n_j):
            label = f"Joint {i}"
            ax_q.plot(t, np.degrees(self.traj.q[:, i]), label=label)
        ax_q.set_ylabel("Joint angle [deg]")
        ax_q.set_title(title)
        ax_q.legend(loc="best")
        ax_q.grid(True, alpha=0.3)

        # --- Joint velocities ---
        if self.traj.q_dot is not None and len(self.traj.q_dot) > 0:
            for i in range(n_j):
                label = f"Joint {i}"
                ax_qd.plot(t, np.degrees(self.traj.q_dot[:, i]), label=label)
            ax_qd.set_ylabel("Joint velocity [deg/s]")
            ax_qd.set_xlabel("Time [s]")
            ax_qd.legend(loc="best")
            ax_qd.grid(True, alpha=0.3)
        else:
            ax_qd.set_visible(False)

        fig.tight_layout()
        self._figs.append(fig)
        return [fig]

    def plot_energy(self, figsize=(10, 5)) -> List:
        """Plot energy evolution (kinetic, potential, total)."""
        import matplotlib.pyplot as plt

        t = self.traj.t
        has_kinetic = (self.traj.energy_kinetic is not None
                       and len(self.traj.energy_kinetic) > 0)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if has_kinetic and np.any(self.traj.energy_kinetic > 1e-12):
            ek = self.traj.energy_kinetic
            ax.plot(t, ek, label="Kinetic")
            ax.set_title("Energy Evolution")
            ax.set_ylabel("Energy [J]")
            ax.set_xlabel("Time [s]")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No energy data available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=14)
            ax.set_title("Energy")

        fig.tight_layout()
        self._figs.append(fig)
        return [fig]

    def plot_phase_portrait(self, joint_indices: Optional[List[int]] = None,
                            figsize=(10, 8)) -> List:
        """
        Phase portrait (q vs q_dot) for selected joints.
        """
        import matplotlib.pyplot as plt

        if joint_indices is None:
            joint_indices = list(range(min(self.traj.n_joints, 4)))

        n_plots = len(joint_indices)
        if n_plots == 0:
            return []

        n_cols = min(n_plots, 3)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()

        for idx, j_idx in enumerate(joint_indices):
            ax = axes[idx]
            q_j = np.degrees(self.traj.q[:, j_idx])
            qd_j = np.degrees(self.traj.q_dot[:, j_idx]
                              if self.traj.q_dot is not None
                              else np.zeros_like(self.traj.q[:, j_idx]))
            ax.plot(q_j, qd_j, '-', alpha=0.7)
            ax.scatter(q_j[0], qd_j[0], c='green', s=40, marker='o',
                       label='start', zorder=5)
            ax.scatter(q_j[-1], qd_j[-1], c='red', s=40, marker='s',
                       label='end', zorder=5)
            ax.set_xlabel("q [deg]")
            ax.set_ylabel("q̇ [deg/s]")
            ax.set_title(f"Joint {j_idx}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()
        self._figs.append(fig)
        return [fig]

    def plot_quasistatic_bars(self, q_eq: np.ndarray, q0: np.ndarray = None,
                              figsize=(8, 5)) -> List:
        """
        Bar chart showing quasi-static equilibrium angles.
        """
        import matplotlib.pyplot as plt

        n_j = len(q_eq)
        if n_j == 0:
            return []

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        q_deg = np.degrees(q_eq)
        x = np.arange(n_j)

        bars = ax.bar(x, q_deg, width=0.6, alpha=0.7,
                       color='steelblue', label='Equilibrium q')

        if q0 is not None:
            q0_deg = np.degrees(q0)
            ax.scatter(x, q0_deg, c='red', s=40, marker='x',
                       label='Initial q0', zorder=5)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Joint index")
        ax.set_ylabel("Angle [deg]")
        ax.set_title("Quasi-Static Equilibrium")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        # Annotate values
        for i, (bar, val) in enumerate(zip(bars, q_deg)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (5 if val >= 0 else -15),
                    f"{val:.1f}°", ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=8)

        fig.tight_layout()
        self._figs.append(fig)
        return [fig]

    def plot_input(self, figsize=(10, 4)) -> List:
        """Plot input signals over time."""
        import matplotlib.pyplot as plt

        if self.traj.inputs is None or len(self.traj.inputs) == 0:
            return []

        t = self.traj.t
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        labels = [r'$\tau_M \cdot \sigma$', r'$\sigma$', r'$\sigma_f$']
        for i in range(min(3, self.traj.inputs.shape[1])):
            ax.plot(t, self.traj.inputs[:, i], label=labels[i])

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Input")
        ax.set_title("Control Inputs u(t)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self._figs.append(fig)
        return [fig]

    def show(self, block: bool = True):
        """Display all collected figures."""
        import matplotlib.pyplot as plt
        plt.show(block=block)

    def save_figs(self, prefix: str = "sim_viz", dpi: int = 150):
        """Save all figures to files."""
        import matplotlib.pyplot as plt
        for i, fig in enumerate(self._figs):
            fname = f"{prefix}_{i}.png"
            fig.savefig(fname, dpi=dpi, bbox_inches='tight')
            print(f"  Saved {fname}")

    def close_all(self):
        """Close all figure handles."""
        import matplotlib.pyplot as plt
        for fig in self._figs:
            plt.close(fig)
        self._figs = []


@dataclass
class SimulationVisualizer:
    """
    Unified visualizer wrapping 2D plotting.

    Parameters
    ----------
    traj : SimulationTrajectory
    design : OrigamiHandDesign, optional
    """

    traj: SimulationTrajectory = None
    design: object = None

    def __post_init__(self):
        self._plotter = None

    def plot_trajectory(self, **kwargs):
        """2D trajectory plot."""
        if self.traj is None:
            print("[Viz] No trajectory data to plot.")
            return
        self._plotter = SimulationPlotter(self.traj)
        return self._plotter.plot_trajectory(**kwargs)

    def plot_energy(self, **kwargs):
        """2D energy plot."""
        if self.traj is None:
            print("[Viz] No trajectory data to plot.")
            return
        if self._plotter is None:
            self._plotter = SimulationPlotter(self.traj)
        return self._plotter.plot_energy(**kwargs)

    def plot_phase_portrait(self, **kwargs):
        """Phase portrait plot."""
        if self.traj is None:
            print("[Viz] No trajectory data to plot.")
            return
        if self._plotter is None:
            self._plotter = SimulationPlotter(self.traj)
        return self._plotter.plot_phase_portrait(**kwargs)

    def plot_quasistatic(self, q_eq: np.ndarray, q0: np.ndarray = None,
                         **kwargs):
        """Quasi-static equilibrium bar chart."""
        if self._plotter is None:
            self._plotter = SimulationPlotter(self.traj) if self.traj is not None else None
        if self._plotter is not None:
            return self._plotter.plot_quasistatic_bars(q_eq, q0, **kwargs)
        return []

    def show(self, block: bool = True):
        """Show all open figures."""
        if self._plotter is not None:
            self._plotter.show(block=block)

    def save(self, prefix: str = "sim_viz"):
        """Save all figures to PNG files."""
        if self._plotter is not None:
            self._plotter.save_figs(prefix=prefix)

    def close(self):
        """Clean up all resources."""
        if self._plotter is not None:
            self._plotter.close_all()


def visualize_simulation(traj: SimulationTrajectory,
                         design=None,
                         q_eq: np.ndarray = None,
                         q0: np.ndarray = None,
                         plot_type: str = "all",
                         save: bool = False,
                         prefix: str = "sim_viz",
                         block: bool = False,
                         **kwargs):
    """
    Convenience function: visualize a SimulationTrajectory.

    Parameters
    ----------
    traj : SimulationTrajectory
        Trajectory to visualize.
    design : OrigamiHandDesign, optional
    q_eq : ndarray, optional
        Equilibrium angles for quasi-static bar plot.
    q0 : ndarray, optional
        Initial guess for quasi-static plot.
    plot_type : str
        'all', 'trajectory', 'energy', 'phase', 'quasistatic'.
    save : bool
        Save figures to disk.
    prefix : str
        Filename prefix for saved figures.
    block : bool
        Block showing matplotlib windows.

    Returns
    -------
    SimulationVisualizer
    """
    viz = SimulationVisualizer(traj=traj, design=design)

    if plot_type in ("all", "trajectory"):
        viz.plot_trajectory()
    if plot_type in ("all", "energy"):
        viz.plot_energy()
    if plot_type in ("all", "phase") and traj.q_dot is not None:
        viz.plot_phase_portrait()
    if plot_type in ("all", "quasistatic") and q_eq is not None:
        viz.plot_quasistatic(q_eq, q0)

    if save:
        viz.save(prefix=prefix)

    return viz
