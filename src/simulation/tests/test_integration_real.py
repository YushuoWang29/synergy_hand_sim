# src/simulation/tests/test_integration_real.py
"""
Real integration test with an actual .ohd design.
"""
import sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

print("=" * 60)
print("Step 1: Load real .ohd design")
print("=" * 60)
from src.models.origami_design import OrigamiHandDesign

import os
ohd_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'ohd test', 'ohd_1.ohd')
ohd_path = os.path.abspath(ohd_path)
print(f'Loading: {ohd_path}')
design = OrigamiHandDesign.load(ohd_path)
print(f'Design: {len(design.fold_lines)} fold lines, '
      f'{len(design.pulleys)} pulleys, {len(design.tendons)} tendons')

print("\n" + "=" * 60)
print("Step 2: Build Q matrix")
print("=" * 60)
from src.simulation.transmission_force import compute_Q_matrix
Q_tauM, Q_s, Q_sdot = compute_Q_matrix(design)
print(f'Q_tauM shape={Q_tauM.shape}: {Q_tauM}')
print(f'Q_s    shape={Q_s.shape}: {Q_s}')
print(f'Q_sdot shape={Q_sdot.shape}: {Q_sdot}')

print("\n" + "=" * 60)
print("Step 3: Run full dynamics simulation (Phase 2)")
print("=" * 60)
from src.simulation.config import SimulationConfig
from src.simulation.simulator import HandSimulator

cfg = SimulationConfig(dt=1e-4, t_end=0.05, phase=2, verbose=1, record_every=10)
sim = HandSimulator(design, cfg)
traj = sim.run(timeout=30.0)
print(f'Simulation: {traj.n_steps} steps, {traj.n_joints} joints')
print(f'Final q: {traj.q[-1]}')
print(f'Final q_dot: {traj.q_dot[-1]}')
assert traj.n_steps > 0, "Simulation must produce trajectory"

print("\n" + "=" * 60)
print("Step 4: Quasi-static solve (Phase 1)")
print("=" * 60)
result = sim.run_quasistatic(np.zeros(traj.n_joints), u=np.array([5.0, 0, 0]))
print(f'Quasi-static: converged={result.converged}, q={result.q}')
print(f'  residual norm={np.linalg.norm(result.residual):.4e}')

print("\n" + "=" * 60)
print("Step 5: IO round-trip")
print("=" * 60)
import tempfile, os
from src.simulation.io import SimulationWriter, SimulationReader
fname = tempfile.mktemp(suffix='.npz')
writer = SimulationWriter(fname, overwrite=True)
writer.save(traj, metadata={'design': 'ohd_1', 'test': True})
reader = SimulationReader(fname)
loaded = reader.load_trajectory()
reader.close()
print(f'IO round-trip: {len(loaded.t)} steps')
np.testing.assert_allclose(loaded.q, traj.q, atol=1e-15)
print('  q verified: OK')
os.remove(fname)

print("\n" + "=" * 60)
print("Step 6: RigidBodySystem from design")
print("=" * 60)
from src.simulation.rigid_body import RigidBodySystem
rbs = RigidBodySystem.from_design(design)
print(f'RigidBodySystem: {len(rbs.link_inertias)} links, {rbs.n_joints} joints')
print(f'  Total mass: {rbs.get_total_mass()*1000:.1f} g')

print("\n" + "=" * 60)
print("ALL INTEGRATION TESTS PASSED!")
print("=" * 60)
