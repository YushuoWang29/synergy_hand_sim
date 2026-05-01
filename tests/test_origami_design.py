# tests/test_origami_design.py
"""
用一个简单例子测试数据结构：
一个矩形板，中间有一条谷折，对折后形成两段
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.origami_design import *

def create_test_design():
    """创建最简单的折纸：2个面，1条谷折"""
    design = OrigamiHandDesign("simple_fold", material_thickness=0.002)
    
    # 4个角点
    p0 = Point2D(0, 0)
    p1 = Point2D(5, 0)
    p2 = Point2D(5, 3)
    p3 = Point2D(0, 3)
    mid_top = Point2D(2.5, 3)
    mid_bottom = Point2D(2.5, 0)
    
    # 定义线段
    lines = [
        FoldLine(0, p0, mid_bottom, FoldType.OUTLINE),      # 下轮廓左
        FoldLine(1, mid_bottom, p1, FoldType.OUTLINE),      # 下轮廓右
        FoldLine(2, p1, p2, FoldType.OUTLINE),              # 右轮廓
        FoldLine(3, p2, mid_top, FoldType.OUTLINE),         # 上轮廓右
        FoldLine(4, mid_top, p3, FoldType.OUTLINE),         # 上轮廓左
        FoldLine(5, p3, p0, FoldType.OUTLINE),              # 左轮廓
        FoldLine(6, mid_bottom, mid_top, FoldType.VALLEY),  # 中间谷折！
    ]
    
    for line in lines:
        design.add_fold_line(line)
    
    # 两个面片
    face_left = OrigamiFace(
        id=0,
        vertices=[p0, mid_bottom, mid_top, p3],
        edge_ids=[0, 6, 4, 5]
    )
    
    face_right = OrigamiFace(
        id=1,
        vertices=[mid_bottom, p1, p2, mid_top],
        edge_ids=[1, 2, 3, 6]
    )
    
    design.add_face(face_left)
    design.add_face(face_right)
    
    # 关节
    design.add_joint(JointConnection(
        id=0,
        fold_line_id=6,
        face_a_id=0,
        face_b_id=1,
        fold_type=FoldType.VALLEY
    ))
    
    # 设左面为根
    design.set_root_face(0)
    design.build_face_tree()
    
    return design


if __name__ == "__main__":
    design = create_test_design()
    
    # 打印摘要
    design.summary()
    
    # 验证
    print("\nValidating...")
    design.validate()
    
    # 检查一些属性
    fold = design.fold_lines[6]
    print(f"\nFold line 6:")
    print(f"  Type: {fold.fold_type.value}")
    print(f"  Length: {fold.length:.3f}")
    print(f"  Direction: {fold.direction}")
    
    face = design.faces[0]
    print(f"\nFace 0:")
    print(f"  Area: {face.area:.3f}")
    print(f"  Centroid: {face.centroid}")
    print(f"  Contains (1,1)? {face.contains_point(Point2D(1, 1))}")
    
    print(f"\nFace tree:")
    for fid, pid in design.face_parent.items():
        print(f"  Face {fid} -> parent {pid}")