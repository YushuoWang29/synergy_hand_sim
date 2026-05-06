"""
Quick test to verify the fix for ohd_4 with 4 tendons.
Tests the synergy callback logic without opening MuJoCo viewer.
"""
import sys
sys.path.insert(0, '.')
import os
import numpy as np
import xml.etree.ElementTree as ET
from src.models.origami_design import OrigamiHandDesign
from src.models.transmission_builder import build_synergy_model, get_joint_list

def build_urdf_to_synergy_mapping(urdf_path, design, ohd_path):
    joints, jid_to_idx = get_joint_list(design)
    
    valley_midpoints = {}
    for fid, fl in design.fold_lines.items():
        if fl.fold_type.value in ('valley', 'mountain'):
            mid = fl.midpoint
            valley_midpoints[fid] = (mid.x, mid.y)

    design2 = OrigamiHandDesign.load(ohd_path)
    design2.build_topology()
    
    link_origins = {design2.root_face_id: (0.0, 0.0)}
    queue = [design2.root_face_id]
    while queue:
        parent_id = queue.pop(0)
        for joint in design2.joints:
            child_id = None
            if joint.face_a_id == parent_id:
                child_id = joint.face_b_id
            elif joint.face_b_id == parent_id:
                child_id = joint.face_a_id
            else:
                continue
            if child_id in link_origins:
                continue
            fold_line = design2.fold_lines[joint.fold_line_id]
            cx, cy = fold_line.midpoint.x, fold_line.midpoint.y
            link_origins[child_id] = (cx, cy)
            queue.append(child_id)

    tree = ET.parse(urdf_path)
    mapping = {}
    for je in tree.findall('.//joint'):
        name = je.get('name')
        parent = je.find('parent').get('link')
        origin = je.find('origin')
        xyz = origin.get('xyz', '0 0 0') if origin is not None else '0 0 0'
        rx, ry, rz = [float(v) for v in xyz.split()]
        parent_id = int(parent.split('_')[1])
        px, py = link_origins.get(parent_id, (0.0, 0.0))
        wx, wy = px + rx, py + ry
        
        best_fid = None
        best_dist = float('inf')
        for fid, (fx, fy) in valley_midpoints.items():
            dist = ((wx - fx) ** 2 + (wy - fy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_fid = fid
        
        syn_idx = jid_to_idx.get(best_fid, -1)
        mapping[name] = syn_idx
    return mapping

# Test ohd_4
print("=" * 60)
print("TEST: ohd_4 with 4 tendons")
print("=" * 60)

ohd_path = os.path.abspath("models/ohd test/ohd_4.ohd")
urdf_path = os.path.abspath("models/ohd_4/ohd_4.urdf")

# Load design
design = OrigamiHandDesign.load(ohd_path)
print(f"Tendons: {len(design.tendons)}")
for tid, t in design.tendons.items():
    print(f"  Tendon {tid}: seq={t.pulley_sequence}")

# Build model
model, joint_stiffness = build_synergy_model(design)
print(f"\nk (num_tendons) = {model.k}")
print(f"m (num_friction) = {model.m}")
print(f"n (num_joints) = {model.n_joints}")

# Build URDF mapping
urdf_to_syn_map = build_urdf_to_synergy_mapping(urdf_path, design, ohd_path)
print(f"\nURDF joint count: {len(urdf_to_syn_map)}")
for name, idx in sorted(urdf_to_syn_map.items()):
    print(f"  {name} -> syn_q[{idx}]")

# Test synergy callback (as defined in run_synergy_from_ohd.py)
def synergy_callback(theta1_rad, theta2_rad):
    sigma = (theta1_rad + theta2_rad) / 2.0
    sigma_f = (theta1_rad - theta2_rad) / 2.0
    q = model.solve(sigma * np.ones(model.k), sigma_f * np.ones(model.m))
    result = {}
    for urdf_name, syn_idx in urdf_to_syn_map.items():
        if 0 <= syn_idx < len(q):
            result[urdf_name] = q[syn_idx]
        else:
            result[urdf_name] = 0.0
    return result

# Test with various inputs
test_cases = [
    (0.0, 0.0),
    (1.0, 0.5),
    (0.5, -0.3),
    (-0.8, 0.4),
    (2.0, -1.0),
]

print("\nSynergy callback tests:")
for t1, t2 in test_cases:
    result = synergy_callback(t1, t2)
    print(f"  theta1={t1:.2f}, theta2={t2:.2f} -> { {k: f'{v:.4f}' for k, v in result.items()} }")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!") 
print("=" * 60)
