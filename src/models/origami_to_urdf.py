# src/models/origami_to_urdf.py

import numpy as np
import os
from .origami_design import OrigamiHandDesign


def _write_face_stl(face, filepath, thickness=0.001):
    """
    为一个面片生成STL文件
    
    将2D多边形拉伸成薄片（上下两个面+侧面）
    """
    vertices_2d = [(v.x, v.y) for v in face.vertices]
    n = len(vertices_2d)
    
    if n < 3:
        return
    
    # 下表面顶点 (z = -thickness/2)
    bottom = np.array([[x, y, -thickness/2] for x, y in vertices_2d])
    # 上表面顶点 (z = +thickness/2)
    top = np.array([[x, y, +thickness/2] for x, y in vertices_2d])
    
    triangles = []
    
    # 扇三角化上表面
    for i in range(1, n - 1):
        triangles.append([top[0], top[i], top[i+1]])
    
    # 扇三角化下表面（法线朝下，顺序相反）
    for i in range(1, n - 1):
        triangles.append([bottom[0], bottom[i+1], bottom[i]])
    
    # 侧面
    for i in range(n):
        j = (i + 1) % n
        triangles.append([bottom[i], bottom[j], top[j]])
        triangles.append([bottom[i], top[j], top[i]])
    
    _write_binary_stl(filepath, triangles)


def _write_binary_stl(filepath, triangles):
    """写入二进制STL文件"""
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
            
            f.write(np.float32(normal[0]).tobytes())
            f.write(np.float32(normal[1]).tobytes())
            f.write(np.float32(normal[2]).tobytes())
            
            for p in [p1, p2, p3]:
                f.write(np.float32(p[0]).tobytes())
                f.write(np.float32(p[1]).tobytes())
                f.write(np.float32(p[2]).tobytes())
            
            f.write(b'\x00\x00')


def export_urdf(design: OrigamiHandDesign, output_path: str):
    """
    将折纸手设计导出为URDF格式，同时生成STL文件
    
    STL文件生成在 output_path 同级目录的 meshes/ 下
    URDF中引用路径为 "meshes/face_X.stl"
    
    Args:
        design: 折纸手设计
        output_path: URDF文件的完整输出路径
    """
    
    urdf_dir = os.path.dirname(os.path.abspath(output_path))
    stl_dir = os.path.join(urdf_dir, "meshes")
    os.makedirs(stl_dir, exist_ok=True)
    
    # 获取手部名称（从URDF文件名）
    hand_name = os.path.splitext(os.path.basename(output_path))[0]
    
    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append(f'<robot name="{hand_name}">')
    
    # 材质
    lines.append('  <material name="face_material">')
    lines.append('    <color rgba="0.8 0.8 0.8 1.0"/>')
    lines.append('  </material>')
    
    # 为每个面片创建link，并生成STL
    for face_id, face in design.faces.items():
        stl_filename = f"face_{face_id}.stl"
        stl_path = os.path.join(stl_dir, stl_filename)
        
        _write_face_stl(face, stl_path)
        print(f"  Generated {stl_filename}")
        
        stl_relative = f"meshes/{stl_filename}"
        
        lines.append(f'  <link name="face_{face_id}">')
        lines.append('    <visual>')
        lines.append('      <geometry>')
        lines.append(f'        <mesh filename="{stl_relative}" scale="1.0 1.0 1.0"/>')
        lines.append('      </geometry>')
        lines.append('      <material name="face_material"/>')
        lines.append('    </visual>')
        lines.append('  </link>')
    
    # 为每个关节创建joint
    for joint in design.joints:
        # 确定父子关系
        child, parent = None, None
        
        if (joint.face_a_id in design.face_parent and 
            design.face_parent.get(joint.face_a_id) == joint.face_b_id):
            child = joint.face_a_id
            parent = joint.face_b_id
        elif (joint.face_b_id in design.face_parent and 
              design.face_parent.get(joint.face_b_id) == joint.face_a_id):
            child = joint.face_b_id
            parent = joint.face_a_id
        else:
            parent = joint.face_a_id
            child = joint.face_b_id
        
        # 计算关节轴在展开状态下的位姿
        fold_line = design.fold_lines[joint.fold_line_id]
        
        # 关节轴上一点（折痕中点）
        mid = fold_line.midpoint
        thickness = design.material_thickness
        
        if joint.fold_type.value == 'mountain':
            origin_z = -thickness / 2
        else:
            origin_z = +thickness / 2
        
        # 关节轴方向（在xy平面内）
        d = fold_line.direction
        axis_x = float(d[0])
        axis_y = float(d[1])
        axis_z = 0.0
        
        # 计算相对于父link frame的origin
        parent_face = design.faces[parent]
        parent_centroid = parent_face.centroid
        
        rel_x = mid.x - parent_centroid.x
        rel_y = mid.y - parent_centroid.y
        rel_z = origin_z
        
        lines.append(f'  <joint name="joint_{joint.id}" type="revolute">')
        lines.append(f'    <parent link="face_{parent}"/>')
        lines.append(f'    <child link="face_{child}"/>')
        lines.append(f'    <origin xyz="{rel_x:.6f} {rel_y:.6f} {rel_z:.6f}" rpy="0 0 0"/>')
        lines.append(f'    <axis xyz="{axis_x:.6f} {axis_y:.6f} {axis_z:.6f}"/>')
        lines.append(f'    <limit effort="10" velocity="3.14" lower="-1.57" upper="1.57"/>')
        lines.append('  </joint>')
    
    lines.append('</robot>')
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"URDF saved to {output_path}")
    print(f"STL files saved to {stl_dir}/")