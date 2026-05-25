#!/usr/bin/env python3
"""
Debug: 检查 ohd_7 的传动矩阵 Q 为什么为零。
===============================================

关键怀疑：
  build_topology() 重建折痕线并重新分配 ID（从 0 开始），
  但孔元素的 attached_fold_line_id 仍然指向旧 ID（如 25,26,...），
  导致传动矩阵计算为零。
"""

import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.origami_design import OrigamiHandDesign, is_hole_id, is_pulley_id
from src.models.transmission_builder import get_joint_list
from src.simulation.transmission_force import (
    compute_Q_matrix, build_R_bar_matrix, build_M_matrix,
    build_viscous_damping_matrix, build_static_friction_matrix, build_N_matrix
)
from src.simulation.config import SimulationConfig
from src.simulation.simulator import HandSimulator


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


ohd_path = os.path.abspath("models/ohd test/ohd_7.ohd")
print(f"加载设计: {ohd_path}")

# ================================================================
# 1. 加载设计并检查原始状态
# ================================================================
design = OrigamiHandDesign.load(ohd_path)

print_section("1. 原始设计状态")
print(f"折痕线总数: {len(design.fold_lines)}")
print(f"面片数: {len(design.faces)}")
print(f"关节数: {len(design.joints)}")
print(f"腱绳数: {len(design.tendons)}")
print(f"孔数: {len(design.holes)}")
print(f"滑轮数: {len(design.pulleys)}")

print("\n原始折痕线 ID:")
for fid, fl in sorted(design.fold_lines.items()):
    print(f"  fold_line[{fid}]: {fl.fold_type.value} " +
          f"({fl.start.x:.1f},{fl.start.y:.1f})->({fl.end.x:.1f},{fl.end.y:.1f})")

print("\n孔的 attached_fold_line_id:")
for hid, hole in sorted(design.holes.items()):
    fl_id = hole.attached_fold_line_id
    fl_exists = fl_id in design.fold_lines if fl_id is not None else False
    pos = (hole.position.x, hole.position.y)
    print(f"  hole[{hid}]: pos={pos}, attached_fl={fl_id}, exists_in_fold_lines={fl_exists}")

# ================================================================
# 2. build_topology() 后检查
# ================================================================
print_section("2. build_topology() 后")
print("重建面片拓扑...")
design2 = OrigamiHandDesign.load(ohd_path)  # 重新加载
design2.build_topology()

print(f"\n新折痕线总数: {len(design2.fold_lines)}")
print(f"新面片数: {len(design2.faces)}")
print(f"新关节数: {len(design2.joints)}")

print("\n新折痕线 ID:")
for fid, fl in sorted(design2.fold_lines.items()):
    print(f"  fold_line[{fid}]: {fl.fold_type.value} " +
          f"({fl.start.x:.1f},{fl.start.y:.1f})->({fl.end.x:.1f},{fl.end.y:.1f})")

joints, jid_to_idx = get_joint_list(design2)
print(f"\n关节映射 (jid_to_idx):")
for fl_id, j_idx in sorted(jid_to_idx.items()):
    fl = design2.fold_lines.get(fl_id)
    fl_type = fl.fold_type.value if fl else "N/A"
    print(f"  fold_line[{fl_id}] ({fl_type}) → joint[{j_idx}]")

print("\n孔的 attached_fold_line_id (build_topology 后):")
for hid, hole in sorted(design2.holes.items()):
    fl_id = hole.attached_fold_line_id
    exists_in_new = fl_id in design2.fold_lines if fl_id is not None else False
    in_jid_to_idx = fl_id in jid_to_idx if fl_id is not None else False
    pos = (hole.position.x, hole.position.y)
    print(f"  hole[{hid}]: pos={pos}, attached_fl={fl_id}, " +
          f"exists_in_new={exists_in_new}, in_jid_to_idx={in_jid_to_idx}")

# ================================================================
# 3. 检查传动矩阵 Q
# ================================================================
print_section("3. 传动矩阵 Q (build_topology 后)")

# 直接调用 compute_Q_matrix
Q_tauM, Q_s, Q_sdot = compute_Q_matrix(design2)

print(f"\nQ_tauM: {Q_tauM}")
print(f"Q_s:    {Q_s}")
print(f"Q_sdot: {Q_sdot}")

# 如果有关节，打印每个关节的 Q 值
if design2.joints:
    for i in range(len(joints)):
        print(f"  joint[{i}]: Q={Q_tauM[i]:+.6e}, effective_R={abs(Q_tauM[i]):.6f}")

# ================================================================
# 4. 构建 HandSimulator 并检查
# ================================================================
print_section("4. HandSimulator 内部 Q 矩阵")
cfg = SimulationConfig(verbose=1)
sim = HandSimulator(design2, cfg)
print(f"\nsim.Q_tauM: {sim.Q_tauM}")
print(f"sim.Q_s:    {sim.Q_s}")
print(f"sim.Q_sdot: {sim.Q_sdot}")

# ================================================================
# 5. 手动修复孔关联后检查
# ================================================================
print_section("5. 手动修复孔关联后")

design3 = OrigamiHandDesign.load(ohd_path)
design3.build_topology()

# 修复孔：按几何位置重新挂载到最近的谷折/峰折折痕
repaired_holes = 0
for hid, hole in list(design3.holes.items()):
    px, py = hole.position.x, hole.position.y
    best_id = None
    best_dist_sq = float('inf')
    for fl_id, fl in design3.fold_lines.items():
        if not fl.is_fold:
            continue
        mx = (fl.start.x + fl.end.x) / 2.0
        my = (fl.start.y + fl.end.y) / 2.0
        d2 = (px - mx)**2 + (py - my)**2
        if d2 < best_dist_sq:
            best_dist_sq = d2
            best_id = fl_id
    best_dist = np.sqrt(best_dist_sq)
    if best_id is not None and best_dist < 30.0:
        old_id = hole.attached_fold_line_id
        hole.attached_fold_line_id = best_id
        if old_id != best_id:
            repaired_holes += 1
            print(f"  hole[{hid}]: {old_id} → {best_id} " +
                  f"(dist={best_dist:.2f}mm, new_fl_type={design3.fold_lines[best_id].fold_type.value})")

print(f"修复了 {repaired_holes} 个孔")

# 重新计算 Q
Q_tauM2, Q_s2, Q_sdot2 = compute_Q_matrix(design3)
print(f"\nQ_tauM (fix): {Q_tauM2}")
print(f"Q_s (fix):    {Q_s2}")
print(f"Q_sdot (fix): {Q_sdot2}")

if np.any(np.abs(Q_tauM2) > 1e-10):
    print(f"\n✅ 修复后 Q_tauM 非零！最大分量: {np.max(np.abs(Q_tauM2)):.6f}")
else:
    print(f"\n❌ Q_tauM 仍然为零。继续排查...")

# 打印 R_bar 矩阵
print_section("6. R_bar 矩阵细节 (修复后)")

for tendon in design3.tendons.values():
    elements = [eid for eid in tendon.pulley_sequence
                if eid >= 0 or (eid <= -100 and eid > -200)]
    print(f"\n腱绳 {tendon.id}: {len(elements)} 个元素")
    print(f"  元素序列: {elements}")
    
    R_bar = build_R_bar_matrix(elements, design3, jid_to_idx)
    print(f"  R_bar shape: {R_bar.shape}")
    print(f"  R_bar 非零项:")
    for seg_idx in range(R_bar.shape[0]):
        for j_idx in range(R_bar.shape[1]):
            if abs(R_bar[seg_idx, j_idx]) > 1e-10:
                print(f"    R_bar[{seg_idx}, {j_idx}] = {R_bar[seg_idx, j_idx]:.4f}")


print("\n" + "="*60)
print("  诊断完成")
print("="*60)
