# tests/test_visualize_origami.py
# NOTE: MeshCat-based OrigamiVisualizer has been removed from the project.
# This test is disabled. Use MuJoCo-based visualization instead.

import sys
sys.path.insert(0, '.')

from src.models.origami_design import *
from src.models.origami_kinematics import OrigamiForwardKinematics

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
    
    # MeshCat-based OrigamiVisualizer removed; use MuJoCo for 3D visualization.
    print("\n  [MeshCat removed] Use MuJoCo simulator for 3D visualization.")
    print("Done!")
