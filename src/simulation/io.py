# src/simulation/io.py
"""
Simulation result serialization and logging.

Provides:
    - SimulationWriter: save trajectory to .npz files
    - SimulationReader: load trajectory from .npz files
"""

import numpy as np
import json
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from .simulator import SimulationTrajectory


class SimulationWriter:
    """
    Write simulation results to compressed numpy archive (.npz).

    Usage::

        writer = SimulationWriter("results/sim_001.npz")
        writer.save(trajectory, metadata={"design": "ohd_4"})
    """

    def __init__(self, path: str, overwrite: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite

    def save(self, trajectory: SimulationTrajectory,
             metadata: Optional[dict] = None) -> str:
        """
        Save trajectory to .npz file.

        Parameters
        ----------
        trajectory : SimulationTrajectory
            The trajectory to save.
        metadata : dict, optional
            Additional metadata to include.

        Returns
        -------
        path : str
            Path to saved file.
        """
        data = trajectory.to_dict()

        # Add metadata
        if metadata is not None:
            data['metadata'] = json.dumps(metadata)
        data['saved_at'] = datetime.now().isoformat()

        np.savez_compressed(str(self.path), **data)
        return str(self.path)

    @staticmethod
    def save_csv(trajectory: SimulationTrajectory, path: str):
        """
        Save trajectory as CSV for external analysis.

        Columns: t, q_0, ..., q_n, q_dot_0, ..., q_dot_n, ...
        """
        import pandas as pd

        n_joints = trajectory.n_joints
        columns = ['t']
        columns += [f'q_{i}' for i in range(n_joints)]
        columns += [f'q_dot_{i}' for i in range(n_joints)]
        columns += [f'q_ddot_{i}' for i in range(n_joints)]
        columns += ['E_kinetic', 'E_potential', 'E_dissipated']

        data = {
            't': trajectory.t,
        }
        for i in range(n_joints):
            data[f'q_{i}'] = trajectory.get_angle(i)
            data[f'q_dot_{i}'] = trajectory.q_dot[:, i] if trajectory.q_dot is not None else np.zeros_like(trajectory.t)
            data[f'q_ddot_{i}'] = trajectory.q_ddot[:, i] if trajectory.q_ddot is not None else np.zeros_like(trajectory.t)
        data['E_kinetic'] = trajectory.energy_kinetic
        data['E_potential'] = trajectory.energy_potential
        data['E_dissipated'] = trajectory.energy_dissipated

        df = pd.DataFrame(data)
        df.to_csv(path, index=False)


class SimulationReader:
    """
    Read simulation results from .npz files.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.data = np.load(str(self.path), allow_pickle=True)

    def load_trajectory(self) -> SimulationTrajectory:
        """Load trajectory from file (strips non-trajectory keys)."""
        traj_keys = {'t', 'q', 'q_dot', 'q_ddot', 'inputs', 'contact_forces',
                     'energy_kinetic', 'energy_potential', 'energy_dissipated',
                     'residuals', 'info'}
        filtered = {k: v for k, v in dict(self.data).items() if k in traj_keys}
        return SimulationTrajectory.from_dict(filtered)

    def load_metadata(self) -> Optional[dict]:
        """Load metadata if present."""
        if 'metadata' in self.data:
            return json.loads(str(self.data['metadata']))
        return None

    def list_keys(self) -> list:
        """List all keys in the archive."""
        return list(self.data.keys())

    def close(self):
        """Close the archive."""
        self.data.close()
