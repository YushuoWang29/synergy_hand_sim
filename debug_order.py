import sys
sys.path.insert(0, '.')
from src.models.origami_design import OrigamiHandDesign
from src.models.transmission_builder import get_joint_list

# Case 1: WITHOUT build_topology (what run_synergy_from_ohd.py does for synergy model)
d1 = OrigamiHandDesign.load('models/ohd test/three_finger_gripper_a.ohd')
joints1, jid_to_idx1 = get_joint_list(d1)
print('=== WITHOUT build_topology (used by synergy model) ===')
print('  Sorted fold_line IDs:', list(jid_to_idx1.keys()))
for fid, idx in sorted(jid_to_idx1.items()):
    fl = d1.fold_lines[fid]
    print(f'  syn_q[{idx}] = fold_line {fid}')

# Case 2: WITH build_topology (used by URDF export + mapping)
d2 = OrigamiHandDesign.load('models/ohd test/three_finger_gripper_a.ohd')
d2.build_topology()
joints2, jid_to_idx2 = get_joint_list(d2)
print()
print('=== WITH build_topology (used by URDF) ===')
joint_to_fid = {}
for j in joints2:
    joint_to_fid[j.id] = j.fold_line_id
    print(f'  joint_{j.id} = fold_line {j.fold_line_id}')

print()
print('=== MAPPING: URDF joint_i -> syn_q[row] (if no build_topology) ===')
for jid in sorted(joint_to_fid.keys()):
    fid = joint_to_fid[jid]
