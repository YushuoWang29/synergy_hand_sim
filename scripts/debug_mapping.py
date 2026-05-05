"""
调试脚本：检查 URDF 关节顺序与协同模型关节顺序的映射关系。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.origami_design import OrigamiHandDesign
from src.models.transmission_builder import get_joint_list
import xml.etree.ElementTree as ET


def build_urdf_to_synergy_mapping(urdf_path, design, ohd_path=None):
    joints, jid_to_idx = get_joint_list(design)

    # 1) Get fold line midpoints
    valley_midpoints = {}
    for fid, fl in design.fold_lines.items():
        if fl.fold_type.value in ('valley', 'mountain'):
            mid = fl.midpoint
            valley_midpoints[fid] = (mid.x, mid.y)

    # 2) Build URDF link origins by recreating design with build_topology
    if ohd_path is None:
        # Derive ohd_path from urdf_path
        if 'ohd_2' in urdf_path:
            ohd_path = 'models/ohd test/ohd_2.ohd'
        elif 'ohd_3' in urdf_path:
            ohd_path = 'models/ohd test/ohd_3.ohd'
        else:
            ohd_path = 'models/ohd test/' + os.path.basename(urdf_path).replace('.urdf', '.ohd')

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

    # 3) Parse URDF joints, match to nearest fold line
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
        print(f'    {name}: world=({wx:.0f},{wy:.0f}) -> fold_{best_fid} -> syn_q[{syn_idx}]')

    return mapping


print('=== ohd_2 映射 ===')
design = OrigamiHandDesign.load('models/ohd test/ohd_2.ohd')
build_urdf_to_synergy_mapping('models/ohd_2/ohd_2.urdf', design, 'models/ohd test/ohd_2.ohd')

print()
print('=== ohd_3 映射 ===')
design3 = OrigamiHandDesign.load('models/ohd test/ohd_3.ohd')
build_urdf_to_synergy_mapping('models/ohd_3/ohd_3.urdf', design3, 'models/ohd test/ohd_3.ohd')
