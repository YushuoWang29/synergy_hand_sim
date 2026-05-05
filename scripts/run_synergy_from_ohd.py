#!/usr/bin/env python3
"""
从 .ohd 文件加载设计，计算传动矩阵，启动 MuJoCo 交互式仿真。

右侧面板显示四个滑块：
  Motor A, Motor B  —— 独立电机控制
  σ (同动量)  —— (A+B)/2
  σ_f (差动量) —— (A-B)/2

【关键修复 v2】：
  1. design.source_path 不存在 → 改为显式传递 ohd_path
  2. URDF 关节顺序 vs 协同模型顺序不一致问题已通过在 synergy_callback
     中使用 URDF→syn_q 映射解决
  3. 确认 R_f 计算已遵循论文公式 (20)/(25)/(32) 且自然编码路径信息
"""

import sys
sys.path.insert(0, '.')

import os, argparse
import xml.etree.ElementTree as ET
import numpy as np

from src.models.origami_design import OrigamiHandDesign
from src.models.transmission_builder import build_synergy_model, get_joint_list
from src.interactive.mujoco_simulator import MuJoCoSimulator


def build_urdf_to_synergy_mapping(urdf_path: str,
                                   design: OrigamiHandDesign,
                                   ohd_path: str) -> dict:
    """
    构建 URDF joint_name → synergy_model q_index 的映射关系。
    通过最近邻匹配：URDF joint 世界坐标 ↔ fold_line 中点 ↔ synergy index
    """
    joints, jid_to_idx = get_joint_list(design)

    # 1) 获取协同模型中各 fold_line_id 对应的中点
    valley_midpoints = {}
    for fid, fl in design.fold_lines.items():
        if fl.fold_type.value in ('valley', 'mountain'):
            mid = fl.midpoint
            valley_midpoints[fid] = (mid.x, mid.y)

    # 2) 通过 build_topology 获得 face 之间的父子关系和原点坐标
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

    # 3) 解析 URDF 关节世界坐标，匹配最近的 fold line
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

    return mapping


def main():
    parser = argparse.ArgumentParser(description="从 .ohd 文件运行增强协同仿真")
    parser.add_argument("ohd_file", help=".ohd 设计文件路径")
    parser.add_argument("--urdf", default=None,
                        help="URDF 文件路径（可选，默认为 models/<name>/<name>.urdf）")
    args = parser.parse_args()

    # 1. 解析路径
    ohd_file = os.path.abspath(args.ohd_file)
    ohd_path = os.path.normpath(args.ohd_file)  # 保留相对路径也可
    hand_name = os.path.splitext(os.path.basename(ohd_file))[0]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.urdf:
        urdf_path = os.path.normpath(args.urdf)
    else:
        urdf_path = os.path.join(project_root, "models", hand_name,
                                  f"{hand_name}.urdf")
    mesh_dir = os.path.join(os.path.dirname(urdf_path), "meshes")

    if not os.path.exists(urdf_path):
        print(f"错误：URDF 不存在 {urdf_path}。请先运行 export_urdf.py。")
        return

    # 2. 加载设计
    design = OrigamiHandDesign.load(ohd_path)
    if len(design.tendons) == 0:
        print("错误：设计中没有任何肌腱。")
        return

    # 3. 构建 URDF → Synergy 关节映射
    print("正在建立 URDF → 协同模型关节映射...")
    urdf_to_syn_map = build_urdf_to_synergy_mapping(urdf_path, design,
                                                      ohd_path)
    print("  URDF → Synergy 映射:")
    for urdf_name in sorted(urdf_to_syn_map.keys()):
        syn_idx = urdf_to_syn_map[urdf_name]
        print(f"    {urdf_name} -> syn_q[{syn_idx}]")

    # 4. 构建 Synergy 模型
    model, joint_stiffness = build_synergy_model(design)
    print(f"\n  R (shape={model.R_aug[:model.k].shape}):\n", model.R_aug[:model.k])
    print(f"  Rf (shape={model.R_aug[model.k:].shape}):\n", model.R_aug[model.k:])
    print(f"\n  S_aug (synergy directions, shape={model.S_aug.shape}):")
    for i in range(model.S_aug.shape[1]):
        vec = model.S_aug[:, i]
        print(f"    Dir {i}: {vec}")

    # 5. 定义协同回调函数
    #    接收 θ1 (Motor A rad), θ2 (Motor B rad)，
    #    返回 {urdf_joint_name: radians} 字典
    def synergy_callback(theta1_rad, theta2_rad):
        sigma = (theta1_rad + theta2_rad) / 2.0
        sigma_f = (theta1_rad - theta2_rad) / 2.0
        q = model.solve(np.array([sigma]), np.array([sigma_f]))
        result = {}
        for urdf_name, syn_idx in urdf_to_syn_map.items():
            if 0 <= syn_idx < len(q):
                result[urdf_name] = q[syn_idx]
            else:
                result[urdf_name] = 0.0
        return result

    # 6. 启动 MuJoCo（四滑块协同模式）
    print("\n启动 MuJoCo 查看器…")
    sim = MuJoCoSimulator(
        urdf_path, mesh_dir,
        synergy_callback=synergy_callback,
        synergy_motor_names=["Motor A", "Motor B"],
        ctrl_range_deg=720.0,
        synergy_with_sigma_sliders=True,
    )

    print("\n仿真已启动。")
    print("右侧面板有四个滑块：Motor A, Motor B, σ, σ_f")
    print("拖动任一滑块即可通过协同模型驱动手指运动。")
    sim.run()


if __name__ == "__main__":
    main()
