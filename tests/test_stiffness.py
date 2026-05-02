# tests/test_stiffness.py
import sys
sys.path.insert(0, '.')
from src.models.origami_parser import OrigamiParser

# 用 test_2.dxf 测试
parser = OrigamiParser(point_tolerance=2.0)
design = parser.parse("tests/test_2.dxf")
stiff_list = design.get_joint_stiffness_list()
for j, k in zip(design.joints, stiff_list):
    print(f"Joint {j.id} (fold {j.fold_line_id}) stiffness = {k:.2f} N·m/rad")

# 手动修改某条折痕的刚度
line_id = design.joints[0].fold_line_id
design.fold_lines[line_id].stiffness = 5.0
print("\nAfter setting joint 0 stiffness to 5.0:")
stiff_list2 = design.get_joint_stiffness_list()
print(stiff_list2)