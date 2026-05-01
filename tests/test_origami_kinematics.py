# tests/test_origami_kinematics.py

import numpy as np
from src.models.origami_design import *
from src.models.origami_kinematics import OrigamiForwardKinematics
from src.models.origami_to_urdf import export_urdf


def create_test_design():
    """和之前一样的简单对折设计"""
    design = OrigamiHandDesign("simple_fold", material_thickness=0.002)
    
    p0 = Point2D(0, 0)
    p1 = Point2D(5, 0)
    p2 = Point2D(5, 3)
    p3 = Point2D(0, 3)
    mid_top = Point2D(2.5, 3)
    mid_bottom = Point2D(2.5, 0)
    
    lines = [
        FoldLine(0, p0, mid_bottom, FoldType.OUTLINE),
        FoldLine(1, mid_bottom, p1, FoldType.OUTLINE),
        FoldLine(2, p1, p2, FoldType.OUTLINE),
        FoldLine(3, p2, mid_top, FoldType.OUTLINE),
        FoldLine(4, mid_top, p3, FoldType.OUTLINE),
        FoldLine(5, p3, p0, FoldType.OUTLINE),
        FoldLine(6, mid_bottom, mid_top, FoldType.VALLEY),
    ]
    
    for line in lines:
        design.add_fold_line(line)
    
    face_left = OrigamiFace(0, [p0, mid_bottom, mid_top, p3], [0, 6, 4, 5])
    face_right = OrigamiFace(1, [mid_bottom, p1, p2, mid_top], [1, 2, 3, 6])
    
    design.add_face(face_left)
    design.add_face(face_right)
    
    design.add_joint(JointConnection(0, 6, 0, 1, FoldType.VALLEY))
    design.set_root_face(0)
    design.build_face_tree()
    
    return design


if __name__ == "__main__":
    design = create_test_design()
    
    # 测试运动学
    fk = OrigamiForwardKinematics(design)
    
    print("=" * 50)
    print("展开状态 (所有关节角=0)")
    print("=" * 50)
    
    angles = {}
    transforms = fk.forward_kinematics(angles)
    for fid, T in transforms.items():
        print(f"Face {fid}:")
        print(f"  translation: {T[:3, 3]}")
    
    vertices_world = fk.get_face_vertices_world(angles)
    for fid, verts in vertices_world.items():
        print(f"Face {fid} vertices (world):")
        for i, v in enumerate(verts):
            print(f"  v{i}: ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})")
    
    print()
    print("=" * 50)
    print("折叠状态 (关节0=90度)")
    print("=" * 50)
    
    angles_90 = {0: np.pi / 2}
    transforms_90 = fk.forward_kinematics(angles_90)
    
    for fid, T in transforms_90.items():
        print(f"Face {fid}:")
        print(f"  translation: {T[:3, 3]}")
        # 提取欧拉角看旋转
        R = T[:3, :3]
        print(f"  rotation det: {np.linalg.det(R):.3f}")
    
    vertices_world_90 = fk.get_face_vertices_world(angles_90)
    for fid, verts in vertices_world_90.items():
        print(f"Face {fid} vertices (world):")
        for i, v in enumerate(verts):
            print(f"  v{i}: ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})")
    
    # 验证：face_1应该绕x=2.5的轴旋转90度
    # face_0应该完全不动
    print()
    print("=" * 50)
    print("验证:")
    print("=" * 50)
    
    face0_verts = vertices_world_90[0]
    face0_orig = vertices_world[0]
    print(f"Face 0 displaced: {np.max(np.abs(face0_verts - face0_orig)):.6f}")
    print("  (should be 0)")
    
    face1_verts = vertices_world_90[1]
    face1_orig = vertices_world[1]
    # 旋转后z坐标应该变化
    print(f"Face 1 z-range (flat): {face1_orig[:, 2].min():.3f} to {face1_orig[:, 2].max():.3f}")
    print(f"Face 1 z-range (folded): {face1_verts[:, 2].min():.3f} to {face1_verts[:, 2].max():.3f}")
    print("  (should change from all-0 to including non-zero)")
    
    # 导出URDF
    print()
    export_urdf(design, "tests/simple_fold.urdf")