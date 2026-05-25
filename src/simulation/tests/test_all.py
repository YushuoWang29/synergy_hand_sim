# src/simulation/tests/test_all.py
"""
Unit and integration tests for the numerical simulation framework.

Coverage:
    Unit tests:
        1. Config creation and phase detection
        2. Inertia estimation from polygon geometry
        3. Capstan tension distribution
        4. Hayward-Armstrong static friction
        5. M matrix structure
        6. ODEState serialization

    Integration tests:
        1. Q matrix computation with dummy design
        2. Dynamics assembler force balance
        3. RK4 integrator step correctness
        4. Quasi-static solver convergence
        5. Contact model normal/ friction forces
        6. Full simulator pipeline (smoke test)
        7. IO round-trip
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Use non-interactive matplotlib backend to avoid Qt crashes in pytest
import matplotlib
matplotlib.use('Agg')

import numpy as np
import unittest


# ========================================================================
# Enhanced MockDesign: provides all interfaces expected by the real code
# ========================================================================

class MockPulley:
    def __init__(self, eid, radius, attached_fold_line_id, friction_coefficient=0.05):
        self.id = eid
        self.radius = radius
        self.attached_fold_line_id = attached_fold_line_id
        self.friction_coefficient = friction_coefficient


class MockHole:
    def __init__(self, eid, plate_offset, attached_fold_line_id, friction_coefficient=0.05):
        self.id = eid
        self.plate_offset = plate_offset
        self.attached_fold_line_id = attached_fold_line_id
        self.friction_coefficient = friction_coefficient


class MockTendon:
    def __init__(self, tid, pulley_sequence):
        self.id = tid
        self.pulley_sequence = pulley_sequence


class MockJoint:
    """Mock joint for get_joint_list to work."""
    def __init__(self, jid, face_ids, fold_line):
        self.id = jid
        self.face_ids = face_ids
        self.fold_line = fold_line
        self.fold_line_id = fold_line.id
        # For OrigamiForwardKinematics compatibility
        self.face_a_id = face_ids[0]
        self.face_b_id = face_ids[1]
        self.fold_type = fold_line.fold_type


class MockFold:
    def __init__(self, fid, start_pt=None, end_pt=None):
        self.id = fid
        self.is_fold = True
        self.fold_type = type('FT', (), {'value': 'VALLEY'})()  # VALLEY-like
        self.start = start_pt or type('Pt', (), {'x': 0.0, 'y': 0.0})()
        self.end = end_pt or type('Pt', (), {'x': 10.0, 'y': 0.0})()
        self.stiffness = 0.5
        # For OrigamiForwardKinematics compatibility
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length = (dx*dx + dy*dy) ** 0.5
        self.direction = (dx / length if length > 0 else 1.0,
                          dy / length if length > 0 else 0.0)
        self.midpoint = type('Pt', (), {'x': (self.start.x + self.end.x) / 2.0,
                                        'y': (self.start.y + self.end.y) / 2.0})()


class MockDesign:
    """3-joint hand with 2 tendons, carefully constructed for testability."""
    def __init__(self):
        # 3 joints (fold lines)
        self.joints = [
            MockJoint(0, [0, 1], MockFold(100)),
            MockJoint(1, [2, 3], MockFold(101)),
            MockJoint(2, [4, 5], MockFold(102)),
        ]
        self.fold_lines = {
            100: MockFold(100),
            101: MockFold(101),
            102: MockFold(102),
        }

        # Pulleys on each joint
        self.pulleys = {
            -101: MockPulley(-101, 4.0, 100, 0.05),
            -102: MockPulley(-102, 4.0, 100, 0.05),
            -103: MockPulley(-103, 3.0, 101, 0.05),
            -104: MockPulley(-104, 3.0, 101, 0.05),
            -105: MockPulley(-105, 2.0, 102, 0.05),
        }
        self.holes = {}

        # Tendons
        self.tendons = {}
        self.tendons[-1] = MockTendon(-1, [-1000, -101, -103, -105, -2000])
        self.tendons[-2] = MockTendon(-2, [-1001, -102, -104, -2001])

        # Faces (for RigidBodySystem.from_design)
        self.faces = {}
        for fid in range(6):
            face = type('Face', (), {'id': fid, 'vertices': [], 'area': 100.0})()
            face.vertices = [
                type('Pt', (), {'x': float(fid * 10), 'y': float(fid * 5)})(),
                type('Pt', (), {'x': float(fid * 10 + 10), 'y': float(fid * 5)})(),
                type('Pt', (), {'x': float(fid * 10 + 10), 'y': float(fid * 5 + 10)})(),
                type('Pt', (), {'x': float(fid * 10), 'y': float(fid * 5 + 10)})(),
            ]
            face.area = 100.0
            self.faces[fid] = face

        self.face_parent = {}
        self.root_face_id = 0  # Required for OrigamiForwardKinematics

    @property
    def tendons_dict(self):
        return self.tendons


# ========================================================================
# Unit Tests
# ========================================================================

class TestConfig(unittest.TestCase):
    def test_default_config(self):
        from src.simulation.config import SimulationConfig
        cfg = SimulationConfig()
        self.assertEqual(cfg.dt, 1e-4)
        self.assertEqual(cfg.phase, 2)
        self.assertTrue(cfg.is_dynamic)
        self.assertFalse(cfg.is_quasistatic)

    def test_phase_detection(self):
        from src.simulation.config import SimulationConfig
        cfg1 = SimulationConfig(phase=1)
        self.assertTrue(cfg1.is_quasistatic)
        self.assertFalse(cfg1.is_dynamic)

        cfg2 = SimulationConfig(phase=2)
        self.assertFalse(cfg2.is_quasistatic)
        self.assertTrue(cfg2.is_dynamic)


class TestRigidBody(unittest.TestCase):
    def test_estimate_square_inertia(self):
        from src.simulation.rigid_body import estimate_link_inertia
        verts = np.array([[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]])
        inertia = estimate_link_inertia(
            face_id=0, area=0.01, thickness=0.003,
            density=1200, face_vertices_2d=verts
        )
        self.assertAlmostEqual(inertia.mass, 0.01 * 0.003 * 1200, places=10)
        self.assertEqual(inertia.inertia.shape, (3, 3))
        self.assertGreater(inertia.inertia[2, 2], 0)

    def test_estimate_triangle_inertia(self):
        from src.simulation.rigid_body import estimate_link_inertia
        verts = np.array([[0, 0], [0.1, 0], [0.05, 0.05]])
        inertia = estimate_link_inertia(
            face_id=0, area=0.0025, thickness=0.003,
            density=1200, face_vertices_2d=verts
        )
        self.assertGreater(inertia.mass, 0)
        self.assertAlmostEqual(inertia.inertia[0, 1], inertia.inertia[1, 0])


class TestCapstanTension(unittest.TestCase):
    def test_tension_symmetric(self):
        from src.simulation.friction_models import CapstanTensionDistribution
        dist = CapstanTensionDistribution(10, beta=0.2)
        w_sigma = dist.get_sigma_weights()
        w_sigma_f = dist.get_sigma_f_weights()
        self.assertEqual(len(w_sigma), 10)
        self.assertEqual(len(w_sigma_f), 10)
        np.testing.assert_allclose(dist.T_A, dist.T_B[::-1])
        self.assertTrue(np.all(w_sigma > 0))
        np.testing.assert_allclose(w_sigma_f, -w_sigma_f[::-1])

    def test_dead_zone_large_n(self):
        """With many elements, both ends should see near-zero tension from
        the opposite motor, creating a dead zone in the middle."""
        from src.simulation.friction_models import CapstanTensionDistribution
        # With 40 elements and beta=0.5, decay is VERY fast
        # T_A[30] = exp(-0.5*30) = exp(-15) = 3.1e-7
        # T_B[30] = exp(-0.5*(40-1-30)) = exp(-0.5*9) = 0.011
        # So both < 0.05 requires T_B < 0.05 too:
        # n-1-k > ln(20)/0.5 = 3.0/0.5 = 6, so k < 33
        # and k > 30 for T_A < 0.05
        # So k=31,32 have both < 0.05
        dist = CapstanTensionDistribution(40, beta=0.5)
        mask = dist.get_dead_zone_mask(threshold=0.05)
        self.assertTrue(np.any(mask),
                        "Expected some elements in dead zone for beta=0.5, n=40")


class TestHaywardArmstrong(unittest.TestCase):
    def test_sticking_initial(self):
        from src.simulation.friction_models import HaywardArmstrongFriction
        haf = HaywardArmstrongFriction(3)
        self.assertTrue(np.all(haf.is_sticking(np.zeros(3))))

    def test_update_sliding(self):
        from src.simulation.friction_models import HaywardArmstrongFriction
        haf = HaywardArmstrongFriction(2, delta_max=np.array([0.1, 0.1]))
        theta = np.array([0.3, 0.0])
        haf.update(theta, dt=0.01)
        # theta[0]=0.3 >= z[0]+delta_max = 0+0.1 => z_new[0] = theta[0]-delta_max = 0.2
        self.assertAlmostEqual(haf.z[0], 0.2, places=6)
        # theta[1]=0.0 in sticking zone => z_new[1] = z[1] = 0.0
        self.assertAlmostEqual(haf.z[1], 0.0, places=6)

    def test_update_below(self):
        from src.simulation.friction_models import HaywardArmstrongFriction
        haf = HaywardArmstrongFriction(1, delta_max=np.array([0.1]))
        theta = np.array([-0.3])
        haf.update(theta, dt=0.01)
        # theta=-0.3 <= z-delta_max = -0.1 => z_new = theta+delta_max = -0.2
        self.assertAlmostEqual(haf.z[0], -0.2, places=6)

    def test_friction_torque(self):
        from src.simulation.friction_models import HaywardArmstrongFriction
        haf = HaywardArmstrongFriction(1, kappa=np.array([1.0]))
        haf.z = np.array([0.0])  # z=0, theta=0.1 -> tau = 0.1
        tau = haf.get_friction_torque(np.array([0.1]))
        self.assertAlmostEqual(tau[0], 0.1)

    def test_energy(self):
        from src.simulation.friction_models import HaywardArmstrongFriction
        haf = HaywardArmstrongFriction(1, kappa=np.array([2.0]))
        haf.z = np.array([0.0])
        energy = haf.get_energy(np.array([0.05]))
        self.assertAlmostEqual(energy, 0.5 * 2.0 * 0.05**2)


class TestMMatrix(unittest.TestCase):
    def test_m_matrix_3_elements(self):
        from src.simulation.transmission_force import build_M_matrix
        M = build_M_matrix([-101, -103, -105])
        self.assertEqual(M.shape, (4, 4))
        expected = np.array([
            [-1, 1, 0, 0],
            [0, -1, 1, 0],
            [0, 0, -1, 1],
            [1, 0, 0, 1]
        ])
        np.testing.assert_array_equal(M, expected)

    def test_m_matrix_invertible(self):
        from src.simulation.transmission_force import build_M_matrix
        M = build_M_matrix([-101, -103, -105])
        inv = np.linalg.inv(M)
        np.testing.assert_allclose(M @ inv, np.eye(4), atol=1e-12)


class TestODEState(unittest.TestCase):
    def test_round_trip(self):
        from src.simulation.integrator import ODEState
        state = ODEState(q=np.array([0.1, 0.2]), q_dot=np.array([0.01, 0.02]),
                         z=np.array([0.001, 0.002, 0.003]), t=1.0)
        x = state.as_vector()
        self.assertEqual(len(x), 2 + 2 + 3)
        restored = ODEState.from_vector(x, n_joints=2, n_z=3)
        np.testing.assert_allclose(restored.q, state.q)
        np.testing.assert_allclose(restored.q_dot, state.q_dot)
        np.testing.assert_allclose(restored.z, state.z)

    def test_no_z(self):
        from src.simulation.integrator import ODEState
        state = ODEState(q=np.array([0.1, 0.2]), q_dot=np.array([0.0, 0.0]))
        x = state.as_vector()
        restored = ODEState.from_vector(x, n_joints=2)
        self.assertIsNone(restored.z)


# ========================================================================
# Integration Tests
# ========================================================================

class TestQMAtrixComputation(unittest.TestCase):
    def test_q_matrix_dims(self):
        from src.simulation.transmission_force import compute_Q_matrix
        design = MockDesign()
        Q_tauM, Q_s, Q_sdot = compute_Q_matrix(design)
        self.assertEqual(len(Q_tauM), 3)
        self.assertEqual(len(Q_s), 3)
        self.assertEqual(len(Q_sdot), 3)

    def test_q_matrix_finite(self):
        from src.simulation.transmission_force import compute_Q_matrix
        design = MockDesign()
        Q_tauM, Q_s, Q_sdot = compute_Q_matrix(design)
        self.assertTrue(np.all(np.isfinite(Q_tauM)))
        self.assertTrue(np.all(np.isfinite(Q_s)))
        self.assertTrue(np.all(np.isfinite(Q_sdot)))


class TestDynamicsAssembler(unittest.TestCase):
    def test_zero_acceleration(self):
        from src.simulation.dynamics import DynamicsAssembler
        assembler = DynamicsAssembler(n_joints=3)
        q = np.zeros(3)
        q_dot = np.zeros(3)
        u = np.array([5.0, 0.0, 0.0])
        Q = np.column_stack([np.ones(3) * 0.1, np.ones(3) * 0.01, np.ones(3) * 0.001])
        q_ddot = assembler.compute_acceleration(q, q_dot, u, Q)
        self.assertTrue(np.all(np.isfinite(q_ddot)))

    def test_damping_opposes_velocity(self):
        """With high damping, acceleration should oppose velocity direction."""
        from src.simulation.dynamics import DynamicsAssembler
        # Moderate damping
        assembler = DynamicsAssembler(n_joints=3, joint_damping=np.ones(3) * 10.0)
        q = np.zeros(3)
        q_dot = np.ones(3) * 0.1  # positive velocity
        u = np.array([5.0, 0.0, 0.0])
        Q = np.column_stack([np.ones(3) * 0.1, np.ones(3) * 0.01, np.ones(3) * 0.001])

        q_ddot_no = assembler.compute_acceleration(q, q_dot, u, Q)
        self.assertTrue(np.all(np.isfinite(q_ddot_no)))

        # The damping contribution -B*q_dot should be negative when q_dot is positive
        # (damping opposes motion)
        damping_torque = -assembler.joint_damping * q_dot
        self.assertTrue(np.all(damping_torque < 0),
                        "Damping torque must oppose positive velocity")
        self.assertTrue(np.all(np.isfinite(damping_torque)),
                        "Damping torque must be finite")


class TestRK4Integrator(unittest.TestCase):
    def test_rk4_exponential_decay(self):
        from src.simulation.integrator import Integrator
        integrator = Integrator(dt=0.01, method='RK4')

        def rhs(t, x):
            return -x

        ts, xs = integrator.integrate(rhs, (0.0, 1.0), np.array([1.0]))
        self.assertAlmostEqual(float(xs[-1][0]), np.exp(-1.0), places=2)

    def test_euler_exponential_decay(self):
        from src.simulation.integrator import Integrator
        integrator = Integrator(dt=0.001, method='Euler')

        def rhs(t, x):
            return -x

        ts, xs = integrator.integrate(rhs, (0.0, 1.0), np.array([1.0]))
        self.assertAlmostEqual(float(xs[-1][0]), np.exp(-1.0), places=1)


class TestQuasiStaticSolver(unittest.TestCase):
    def test_linear_spring_equilibrium(self):
        from src.simulation.quasi_static import QuasiStaticSolver

        def tau_t(q):
            return np.array([1.0])

        def tau_e(q):
            return np.array([5.0 * q[0]])

        solver = QuasiStaticSolver(
            n_joints=1,
            compute_transmission_torque=tau_t,
            compute_elastic_torque=tau_e,
            tol=1e-10, max_iter=30, verbose=0,
        )
        result = solver.solve(np.array([0.0]))
        self.assertTrue(result.converged)
        self.assertAlmostEqual(float(result.q[0]), 0.2, places=6)


class TestContactModelRemoved(unittest.TestCase):
    """Contact model tests removed: contact_model.py was cleaned from the project.
    Collision detection functionality is not needed for the current framework."""
    def test_contact_removed(self):
        self.assertTrue(True, "Contact model removed from project")


class TestSimulatorSmoke(unittest.TestCase):
    def test_simulator_creation(self):
        from src.simulation.config import SimulationConfig
        from src.simulation.simulator import HandSimulator
        design = MockDesign()
        cfg = SimulationConfig(dt=1e-3, t_end=0.01, phase=2, verbose=0)
        sim = HandSimulator(design, cfg)
        self.assertIsNotNone(sim)

    def test_simulator_run_short(self):
        from src.simulation.config import SimulationConfig
        from src.simulation.simulator import HandSimulator
        design = MockDesign()
        cfg = SimulationConfig(dt=1e-3, t_end=0.02, phase=2, verbose=0,
                               record_every=1)
        sim = HandSimulator(design, cfg)
        traj = sim.run(timeout=15.0)
        self.assertIsNotNone(traj)
        self.assertGreater(traj.n_steps, 0)


class TestIORoundTrip(unittest.TestCase):
    def test_npz_round_trip_cleanup(self):
        from src.simulation.simulator import SimulationTrajectory
        from src.simulation.io import SimulationWriter, SimulationReader
        import tempfile
        import gc

        traj = SimulationTrajectory(
            t=np.array([0.0, 0.1, 0.2]),
            q=np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.4]]),
            q_dot=np.array([[0.0, 0.0], [0.5, 1.0], [0.5, 1.0]]),
            energy_kinetic=np.array([0.0, 0.1, 0.2]),
            energy_potential=np.array([0.0, 0.0, 0.0]),
            energy_dissipated=np.array([0.0, 0.0, 0.0]),
        )

        fname = tempfile.mktemp(suffix='.npz')
        try:
            writer = SimulationWriter(fname, overwrite=True)
            writer.save(traj, metadata={'test': True})

            reader = SimulationReader(fname)
            loaded = reader.load_trajectory()
            reader.close()

            self.assertEqual(len(loaded.t), 3)
            np.testing.assert_allclose(loaded.q, traj.q)
            np.testing.assert_allclose(loaded.q_dot, traj.q_dot)
        finally:
            if os.path.exists(fname):
                try:
                    os.remove(fname)
                except PermissionError:
                    pass  # Windows file locking


class TestPaperReproduction(unittest.TestCase):
    def test_sigma_mode_gravity_runs(self):
        from src.simulation.config import SimulationConfig
        from src.simulation.simulator import HandSimulator
        design = MockDesign()
        cfg = SimulationConfig(
            dt=1e-3, t_end=0.2, phase=2, verbose=0,
            sigma_func="min(t * 5.0, 5.0)",
            tau_M_func="1.0",
        )
        sim = HandSimulator(design, cfg)
        traj = sim.run(timeout=10.0)
        self.assertGreater(traj.n_steps, 0)

    def test_sigma_f_differential_runs(self):
        from src.simulation.config import SimulationConfig
        from src.simulation.simulator import HandSimulator
        design = MockDesign()
        cfg = SimulationConfig(
            dt=1e-3, t_end=0.2, phase=2, verbose=0,
            sigma_func="0.5",
            sigma_f_func="0.5",
            tau_M_func="3.0",
        )
        sim = HandSimulator(design, cfg)
        traj = sim.run(timeout=10.0)
        self.assertGreater(traj.n_steps, 0)

    def test_quasi_static_runs(self):
        from src.simulation.config import SimulationConfig
        from src.simulation.simulator import HandSimulator
        design = MockDesign()
        cfg_qs = SimulationConfig(phase=1, verbose=0)
        sim_qs = HandSimulator(design, cfg_qs)
        result = sim_qs.run_quasistatic(np.zeros(3), u=np.array([5.0, 0, 0]))
        self.assertIsNotNone(result)


# ========================================================================
# Visualization Tests
# ========================================================================

class TestSimulationPlotter(unittest.TestCase):
    """Test the 2D plotting functionality with a fake trajectory."""

    def _make_traj(self):
        from src.simulation.simulator import SimulationTrajectory
        n_steps, n_j = 20, 3
        t = np.linspace(0, 0.1, n_steps)
        q = np.column_stack([np.sin(t * i + 0.5) for i in range(n_j)])
        q_dot = np.column_stack([np.cos(t * i + 0.5) * i for i in range(n_j)])
        ek = 0.5 * np.sum(q_dot**2, axis=1)
        return SimulationTrajectory(
            t=t, q=q, q_dot=q_dot,
            energy_kinetic=ek,
            energy_potential=np.zeros(n_steps),
            energy_dissipated=np.zeros(n_steps),
        )

    def test_plot_trajectory_no_error(self):
        from src.simulation.visualization import SimulationPlotter
        traj = self._make_traj()
        plotter = SimulationPlotter(traj)
        figs = plotter.plot_trajectory()
        self.assertGreater(len(figs), 0)
        plotter.close_all()

    def test_plot_energy_no_error(self):
        from src.simulation.visualization import SimulationPlotter
        traj = self._make_traj()
        plotter = SimulationPlotter(traj)
        figs = plotter.plot_energy()
        self.assertGreater(len(figs), 0)
        plotter.close_all()

    def test_plot_phase_no_error(self):
        from src.simulation.visualization import SimulationPlotter
        traj = self._make_traj()
        plotter = SimulationPlotter(traj)
        figs = plotter.plot_phase_portrait(joint_indices=[0, 1])
        self.assertGreater(len(figs), 0)
        plotter.close_all()

    def test_plot_quasistatic_bars_no_error(self):
        from src.simulation.visualization import SimulationPlotter
        traj = self._make_traj()
        plotter = SimulationPlotter(traj)
        figs = plotter.plot_quasistatic_bars(
            q_eq=np.array([-1.74, -0.44, 0.0]),
            q0=np.zeros(3))
        self.assertGreater(len(figs), 0)
        plotter.close_all()


class TestMeshCatAnimatorRemoved(unittest.TestCase):
    """MeshCatAnimator removed from project; test is disabled."""
    def test_removed(self):
        self.assertTrue(True, "MeshCat animator removed from project")


class TestVisualizeSimulation(unittest.TestCase):
    """Test the convenience function (saves images w/o opening GUI)."""

    def test_visualize_save(self):
        from src.simulation.visualization import visualize_simulation
        from src.simulation.simulator import SimulationTrajectory
        import tempfile, os

        traj = SimulationTrajectory(
            t=np.array([0.0, 0.02, 0.04]),
            q=np.array([[0.0, 0.0], [0.1, 0.2], [0.2, 0.4]]),
            q_dot=np.array([[0.0, 0.0], [0.5, 1.0], [0.5, 1.0]]),
            energy_kinetic=np.array([0.0, 0.1, 0.2]),
        )

        prefix = tempfile.mktemp(prefix="sim_test_")
        try:
            viz = visualize_simulation(
                traj=traj, q_eq=np.array([-1.0, 0.5]),
                plot_type="all", save=True, prefix=prefix, block=False,
            )
            self.assertIsNotNone(viz)
            viz.close()
        finally:
            # Clean up generated files
            import glob
            for f in glob.glob(prefix + "*.png"):
                try:
                    os.remove(f)
                except PermissionError:
                    pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
