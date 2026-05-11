"""Quick test for fold angle clamping"""
import sys; sys.path.insert(0, '.')
import numpy as np
from src.models.origami_design import *
from src.models.origami_kinematics import OrigamiForwardKinematics, clamp_fold_angle
from src.models.origami_design import FoldType

# Test 1: clamp_fold_angle function
print("=== Test 1: clamp_fold_angle() ===")
# VALLEY: should be [0, pi]
assert clamp_fold_angle(-1.0, FoldType.VALLEY) == 0.0, "VALLEY negative should clamp to 0"
assert clamp_fold_angle(4.0, FoldType.VALLEY) == np.pi, "VALLEY > pi should clamp to pi"
assert clamp_fold_angle(1.0, FoldType.VALLEY) == 1.0, "VALLEY normal should pass through"
# MOUNTAIN: should be [-pi, 0]
assert clamp_fold_angle(1.0, FoldType.MOUNTAIN) == 0.0, "MOUNTAIN positive should clamp to 0"
assert clamp_fold_angle(-4.0, FoldType.MOUNTAIN) == -np.pi, "MOUNTAIN < -pi should clamp to -pi"
assert clamp_fold_angle(-1.0, FoldType.MOUNTAIN) == -1.0, "MOUNTAIN normal should pass through"
print("  clamp_fold_angle: ALL PASSED")

# Test 2: Forward kinematics with VALLEY fold
print("\n=== Test 2: FK with VALLEY fold clamping ===")
design = OrigamiHandDesign('test', material_thickness=0.002)
p0, p1 = Point2D(0,0), Point2D(5,0)
p2, p3 = Point2D(5,3), Point2D(0,3)
mid_top, mid_bottom = Point2D(2.5,3), Point2D(2.5,0)
lines = [
    FoldLine(0, p0, mid_bottom, FoldType.OUTLINE),
    FoldLine(1, mid_bottom, p1, FoldType.OUTLINE),
    FoldLine(2, p1, p2, FoldType.OUTLINE),
    FoldLine(3, p2, mid_top, FoldType.OUTLINE),
    FoldLine(4, mid_top, p3, FoldType.OUTLINE),
    FoldLine(5, p3, p0, FoldType.OUTLINE),
    FoldLine(6, mid_bottom, mid_top, FoldType.VALLEY),
]
for l in lines: design.add_fold_line(l)
design.add_face(OrigamiFace(0, [p0,mid_bottom,mid_top,p3], [0,6,4,5]))
design.add_face(OrigamiFace(1, [mid_bottom,p1,p2,mid_top], [1,2,3,6]))
design.add_joint(JointConnection(0, 6, 0, 1, FoldType.VALLEY))
design.set_root_face(0)
design.build_face_tree()

fk = OrigamiForwardKinematics(design, thickness=0.002)

# Negative angle -> should clamp to 0 (flat)
angles_neg = {0: -1.0}
transforms = fk.forward_kinematics(angles_neg)
diff_neg = np.max(np.abs(transforms[1] - transforms[0]))
assert diff_neg < 1e-10, f"Negative angle should be flat, diff={diff_neg}"
print("  VALLEY negative angle -> flat: PASSED")

# Over 180 deg -> should clamp to pi (fully folded)
angles_over = {0: np.radians(200)}
transforms_over = fk.forward_kinematics(angles_over)
# At pi fold, the two faces should be maximally folded (not flat)
assert transforms_over[1][2,3] != 0, "Pi fold should not be flat"
print("  VALLEY > 180 deg -> pi clamp: PASSED")

# Normal 90 deg
angles_90 = {0: np.pi/2}
transforms_90 = fk.forward_kinematics(angles_90)
assert transforms_90[1][2,3] != 0, "90 deg should move face 1"
print("  VALLEY 90 deg normal: PASSED")

print("\n=== ALL TESTS PASSED ===")
