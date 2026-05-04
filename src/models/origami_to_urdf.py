# src/models/origami_to_urdf.py

import numpy as np
import os
from .origami_design import OrigamiHandDesign, Point2D


def _write_face_stl(face, filepath, offset=(0.0, 0.0), thickness=3.0, top_z=0.0):
    """
    为一个面片生成STL文件
    offset: (ox, oy) 面片link原点在2D绝对坐标系中的位置
    top_z: 上表面在 link 坐标系中的 z 坐标
    """
    ox, oy = offset
    vertices = [(v.x - ox, v.y - oy) for v in face.vertices]
    
    n = len(vertices)
    if n < 3:
        return
    
        # 计算有向面积（2D 叉积和）
    area = 0.0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i+1) % n]
        area += x1 * y2 - x2 * y1
    area *= 0.5
    
    # 如果面积为负（顺时针），反转顶点顺序
    if area < 0:
        vertices.reverse()
        print(f"  Reversed vertex order for face (area={area:.2f})")
        
    top = np.array([[x, y, top_z] for x, y in vertices])
    bottom = np.array([[x, y, top_z - thickness] for x, y in vertices])
    
    triangles = []
    
    for i in range(1, n - 1):
        triangles.append([top[0], top[i], top[i+1]])
    
    for i in range(1, n - 1):
        triangles.append([bottom[0], bottom[i+1], bottom[i]])
    
    for i in range(n):
        j = (i + 1) % n
        triangles.append([bottom[i], bottom[j], top[j]])
        triangles.append([bottom[i], top[j], top[i]])
    
    _write_binary_stl(filepath, triangles)


def _write_binary_stl(filepath, triangles):
    with open(filepath, 'wb') as f:
        f.write(b'\x00' * 80)
        n = len(triangles)
        f.write(n.to_bytes(4, 'little'))
        
        for tri in triangles:
            p1, p2, p3 = tri
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal = normal / norm
            
            for val in normal:
                f.write(np.float32(val).tobytes())
            for p in [p1, p2, p3]:
                for val in p:
                    f.write(np.float32(val).tobytes())
            f.write(b'\x00\x00')


def export_urdf(design: OrigamiHandDesign, output_path: str, thickness: float = 3.0):
    """
    将折纸手设计导出为URDF和STL
    
    定位策略（折痕对齐法 + 峰/谷分层 + 方向统一）：
    - 所有面片上表面在展开状态时平齐于世界 z=0
    - 谷折关节轴在父面片上表面 (z=0)，子link原点也在 z=0
    - 峰折关节轴在父面片下表面 (z=-thickness)，子link原点也在 z=-thickness
    - 峰折子面片的STL上表面在link坐标系中位于 z=+thickness
    - axis 方向统一：谷折正转 = 向上，峰折正转 = 向下
    """
    
    urdf_dir = os.path.dirname(os.path.abspath(output_path))
    stl_dir = os.path.join(urdf_dir, "meshes")
    os.makedirs(stl_dir, exist_ok=True)
    
    hand_name = os.path.splitext(os.path.basename(output_path))[0]
    
    if design.root_face_id is None:
        raise ValueError("Root face not set")
    
    # ========== 第一步：确定每个 link 的参数 ==========
    link_origins = {}
    link_origin_z = {}
    link_top_z = {}
    
    root = design.root_face_id
    link_origins[root] = (0.0, 0.0)
    link_origin_z[root] = 0.0
    link_top_z[root] = 0.0
    
    queue = [root]
    
    while queue:
        parent_id = queue.pop(0)
        
        for joint in design.joints:
            if joint.face_a_id == parent_id:
                child_id = joint.face_b_id
            elif joint.face_b_id == parent_id:
                child_id = joint.face_a_id
            else:
                continue
            
            if child_id in link_origins:
                continue
            
            fold_line = design.fold_lines[joint.fold_line_id]
            mid = fold_line.midpoint
            
            cx, cy = mid.x, mid.y
            
            if joint.fold_type.value == 'mountain':
                cz = -thickness
            else:
                cz = 0.0
            
            top_z = -cz
            
            link_origins[child_id] = (cx, cy)
            link_origin_z[child_id] = cz
            link_top_z[child_id] = top_z
            
            queue.append(child_id)
    
    print("  Link parameters (world):")
    for fid in sorted(link_origins.keys()):
        x, y = link_origins[fid]
        z = link_origin_z[fid]
        top = link_top_z[fid]
        print(f"    face_{fid}: origin_xy=({x:.1f}, {y:.1f}), origin_z={z:.1f}, top_z_in_link={top:.1f}")
    
    # ========== 第二步：生成 STL 文件 ==========
    for face_id, face in design.faces.items():
        ox, oy = link_origins[face_id]
        top_z = link_top_z[face_id]
        stl_path = os.path.join(stl_dir, f"face_{face_id}.stl")
        _write_face_stl(face, stl_path, offset=(ox, oy), thickness=thickness, top_z=top_z)
        print(f"  Generated face_{face_id}.stl (offset_xy=({ox:.1f},{oy:.1f}), top_z={top_z:.1f})")
    
    # ========== 第三步：生成 URDF ==========
    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append(f'<robot name="{hand_name}">')
    lines.append('  <material name="face_material">')
    lines.append('    <color rgba="0.8 0.8 0.8 1.0"/>')
    lines.append('  </material>')
    
    for face_id in design.faces:
        lines.append(f'  <link name="face_{face_id}">')
        lines.append('    <visual>')
        lines.append('      <geometry>')
        lines.append(f'        <mesh filename="meshes/face_{face_id}.stl" scale="1.0 1.0 1.0"/>')
        lines.append('      </geometry>')
        lines.append('      <material name="face_material"/>')
        lines.append('    </visual>')
        lines.append('  </link>')
    
    for joint in design.joints:
        # 确定父子
        parent, child = None, None
        
        if (joint.face_a_id in design.face_parent and 
            design.face_parent[joint.face_a_id] == joint.face_b_id):
            parent = joint.face_b_id
            child = joint.face_a_id
        elif (joint.face_b_id in design.face_parent and 
              design.face_parent[joint.face_b_id] == joint.face_a_id):
            parent = joint.face_a_id
            child = joint.face_b_id
        else:
            parent = joint.face_a_id
            child = joint.face_b_id
        
        fold_line = design.fold_lines[joint.fold_line_id]
        d = fold_line.direction               # 单位方向向量 (dx, dy)
        n = np.array([-d[1], d[0]])           # 法向量（d逆时针转90°）
        
        # 计算子面片中心相对折痕中点的偏移
        child_face = design.faces[child]
        child_center = child_face.centroid
        mid = fold_line.midpoint
        v = np.array([child_center.x - mid.x, child_center.y - mid.y])
        
        s = np.dot(v, n)   # 投影符号
        
        # 确定 axis 方向
        # 谷折(正转=向上): 子面片在上表面侧 → axis = sign(s) * d
        # 峰折(正转=向下): 子面片在下表面侧 → axis = -sign(s) * d
        sign = 1.0 if s > 0 else -1.0
        type_mult = 1.0 if joint.fold_type.value == 'valley' else -1.0
        axis_x = type_mult * sign * d[0]
        axis_y = type_mult * sign * d[1]
        
        px, py = link_origins[parent]
        pz = link_origin_z[parent]
        cx, cy = link_origins[child]
        cz = link_origin_z[child]
        
        rel_x = cx - px
        rel_y = cy - py
        rel_z = cz - pz
        
        lines.append(f'  <joint name="joint_{joint.id}" type="revolute">')
        lines.append(f'    <parent link="face_{parent}"/>')
        lines.append(f'    <child link="face_{child}"/>')
        lines.append(f'    <origin xyz="{rel_x:.6f} {rel_y:.6f} {rel_z:.6f}" rpy="0 0 0"/>')
        lines.append(f'    <axis xyz="{axis_x:.6f} {axis_y:.6f} 0.000000"/>')
        lines.append(f'    <limit effort="10" velocity="3.14" lower="-1.57" upper="1.57"/>')
        lines.append('  </joint>')
    
    lines.append('</robot>')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"URDF saved to {output_path}")