# tests/test_visualize_origami.py

import sys
sys.path.insert(0, '.')

from src.models.origami_design import *
from src.models.origami_kinematics import OrigamiForwardKinematics
from src.visualization.origami_visualizer import OrigamiVisualizer
import numpy as np


def create_test_design():
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
    print("Creating design...")
    design = create_test_design()
    
    print("Creating forward kinematics...")
    fk = OrigamiForwardKinematics(design)
    
    print("Starting visualizer...")
    viz = OrigamiVisualizer()
    
    # 手动打开浏览器（解决自动打开问题）
    viz.open_browser()
    
    # 给MeshCat一点时间初始化
    import time
    time.sleep(1.0)
    
    # 显示展开状态
    print("Displaying flat state...")
    viz.display_hand(fk, {})
    
    input("Press Enter to fold to 90°...")
    viz.animate_folding(fk, 0, 0.0, np.pi/2, steps=30)
    
    input("Press Enter to exit...")
    
    print("Done!")