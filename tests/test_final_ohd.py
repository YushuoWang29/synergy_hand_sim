"""Final comprehensive test with time-limited MuJoCo."""
import sys
sys.path.insert(0, '.')
import numpy as np
import os
import time

from src.models.origami_design import OrigamiHandDesign
from src.models.transmission_builder import build_synergy_model
from src.interactive.mujoco_simulator import MuJoCoSimulator

def test_model(name, ohd_path):
    print(f"\n{'='*60}")
    print(f'Testing: {name}')
    print(f"{'='*60}")
    
    design = OrigamiHandDesign.load(ohd_path)
    model, _ = build_synergy_model(design)
    
    print(f"\nk={model.k}, m={model.m}, n={model.n_joints}")
    
    # Test sigma only (both motors same direction)
    q1 = model.solve(np.ones(model.k), np.zeros(model.m))
    print(f"sigma=1, sigma_f=0 -> q = {np.array2string(q1, precision=4)}")
    
    # Test sigma_f only (motors opposite)
    q2 = model.solve(np.zeros(model.k), np.ones(model.m))
    print(f"sigma=0, sigma_f=1 -> q = {np.array2string(q2, precision=4)}")
    
    # Motor A only
    qA = model.solve(np.ones(model.k)*0.5, np.ones(model.m)*0.5)
    max_joint = np.argmax(np.abs(qA))
    print(f"Motor A only -> max effect at joint {max_joint} = {qA[max_joint]:.6f}")
    
    # Motor B only
    qB = model.solve(np.ones(model.k)*0.5, -np.ones(model.m)*0.5)
    print(f"Motor B only -> max effect at joint {max_joint} = {qB[max_joint]:.6f}")
    
    # Both should produce positive motion (finger curling)
    assert np.any(qA > 1e-10), f"Motor A should produce positive motion in some joint, got all zero"
    assert np.any(qB > 1e-10), f"Motor B should produce positive motion in some joint, got all zero"
    assert np.all(qA >= -1e-10), f"Motor A should not produce negative motion: {qA}"
    assert np.all(qB >= -1e-10), f"Motor B should not produce negative motion: {qB}"
    print("  ✓ All assertion checks passed")
    return model, design

# Test all three models
m1, d1 = test_model("ohd_4 (4 fingers)", "models/ohd test/ohd_4.ohd")
m2, d2 = test_model("ohd_4_temp (1 tendon)", "models/ohd test/ohd_4_temp.ohd")
m3, d3 = test_model("ohd_5 (5 fingers)", "models/ohd test/ohd_5.ohd")

# Full MuJoCo pipeline test with ohd_4
print(f"\n{'='*60}")
print("Full MuJoCo pipeline: ohd_4")
print(f"{'='*60}")
hand_name = "ohd_4"
urdf_path = f"models/{hand_name}/{hand_name}.urdf"
mesh_dir = f"models/{hand_name}/meshes"
sim = MuJoCoSimulator(urdf_path, mesh_dir)
time.sleep(1)

# Test sigma only
q1 = m1.solve(np.ones(m1.k), np.zeros(m1.m))
angle_dict = {f'joint_{i}': np.degrees(q1[i]) for i in range(q1.shape[0])}
sim.set_joint_angles(angle_dict)
print(f"sigma=1, sigma_f=0: q_deg={[f'{v:.2f}' for v in angle_dict.values()]}")
time.sleep(1)

# Reset
sim.reset()
sim.close()
print("\n✓ All tests passed!")
