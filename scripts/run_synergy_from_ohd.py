#!/usr/bin/env python3
"""
从 .ohd 文件加载设计，计算传动矩阵，启动 MuJoCo 交互式仿真。

右侧面板显示五个滑块（动态模式）：
  Motor A, Motor B  —— 独立电机控制
  σ (同动量)  —— (A+B)/2
  σ_f (差动量) —— (A-B)/2
  Speed (rad/s) —— 电机转速设定（0~10 rad/s）

动态协同模式 (--dynamic):
  根据 Speed (rad/s) 滑块值在 fast synergy (S_f) 和 slow synergy (S_s) 之间切换：
  - Speed ≈ 0 rad/s  → 手指按 S_s 方向运动（慢速/准静态，如捏取）
  - Speed ≥ 10 rad/s → 手指按 S_f 方向运动（快速/动态，如握拳）
  - 中间值 → 平滑插值
  σ 和 σ_f 均受动态影响。

【变更记录 v3】：
  1. 新增 Speed (rad/s) 滑块替代帧差追踪速度
  2. sigma_f 默认受动态影响（移除 --dynamic-on-f 参数）
  3. 精简动态模式终端输出信息
"""

import sys
sys.path.insert(0, '.')

import os, argparse
import xml.etree.ElementTree as ET
import numpy as np

from src.models.origami_design import OrigamiHandDesign
from src.models.transmission_builder import (
    build_synergy_model, build_dynamic_synergy_model, get_joint_list
)
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
    parser = argparse.ArgumentParser(
        description="从 .ohd 文件运行协同仿真（支持动态协同模式）")
    parser.add_argument("ohd_file", help=".ohd 设计文件路径")
    parser.add_argument("--urdf", default=None,
                        help="URDF 文件路径（可选，默认为 models/<name>/<name>.urdf）")
    parser.add_argument("--dynamic", action="store_true",
                        help="启用动态协同模式（根据 Speed 滑块选择 fast/slow synergy）")
    args = parser.parse_args()

    # 1. 解析路径
    ohd_file = os.path.abspath(args.ohd_file)
    ohd_path = os.path.normpath(args.ohd_file)
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
    if args.dynamic:
        print("\n=== 动态协同模式 ===")
        model, info = build_dynamic_synergy_model(design,
                                                   use_augmented=True)
        n_joints = info['n_joints']
        n_dampers = info['n_dampers']

        # 输出简明的协同方向信息
        print(f"  关节数 = {n_joints}, 阻尼器数 = {n_dampers}")
        print(f"  慢速方向 0 (S_s[0]): {model.S_s[:, 0]}")
        if n_dampers > 0:
            print(f"  快速方向 0 (S_f[0]): {model.S_f[:, 0]}")
        if model.S_s.shape[1] > 1:
            print(f"  慢速方向 1 (S_s[1]): {model.S_s[:, 1]}")
        if model.S_f.shape[1] > 1 and n_dampers > 0:
            print(f"  快速方向 1 (S_f[1]): {model.S_f[:, 1]}")
        print(f"  Speed 滑块范围: 0 ~ 10 rad/s (滑块值为 0~10 对应 speed_factor=0~1)")

        def synergy_callback(theta1_rad, theta2_rad, speed_rad_s=0.0):
            sigma = (theta1_rad + theta2_rad) / 2.0
            sigma_f = (theta1_rad - theta2_rad) / 2.0

            # Speed 滑块值直接映射为 speed_factor (0~10 rad/s → 0~1)
            speed_factor = np.clip(speed_rad_s / 10.0, 0.0, 1.0)

            # sigma 和 sigma_f 都受动态影响
            q = model.solve_combined(
                np.array([sigma]),
                np.array([sigma_f]),
                speed_factor=speed_factor,
                use_dynamic_on_f=True  # 默认启用
            )

            result = {}
            for urdf_name, syn_idx in urdf_to_syn_map.items():
                if 0 <= syn_idx < len(q):
                    result[urdf_name] = q[syn_idx]
                else:
                    result[urdf_name] = 0.0
            return result

        print("\n动态协同模式已启用。")
        print("  → Speed = 0~1 rad/s: 慢速协同 (S_s, 准静态，如捏取)")
        print("  → Speed ≈ 10 rad/s:  快速协同 (S_f, 动态，如握拳)")
        print("  → Speed 中间值: 平滑插值")

    else:
        # 标准准静态模式（原始行为不变）
        model, joint_stiffness = build_synergy_model(design)
        print(f"\n  R (shape={model.R_aug[:model.k].shape}):\n", model.R_aug[:model.k])
        print(f"  Rf (shape={model.R_aug[model.k:].shape}):\n", model.R_aug[model.k:])
        print(f"\n  S_aug (synergy directions, shape={model.S_aug.shape}):")
        for i in range(model.S_aug.shape[1]):
            vec = model.S_aug[:, i]
            print(f"    Dir {i}: {vec}")

        def synergy_callback(theta1_rad, theta2_rad, speed_rad_s=0.0):
            sigma = (theta1_rad + theta2_rad) / 2.0
            sigma_f = (theta1_rad - theta2_rad) / 2.0
            q = model.solve(sigma * np.ones(model.k), sigma_f * np.ones(model.m))
            result = {}
            for urdf_name, syn_idx in urdf_to_syn_map.items():
                if 0 <= syn_idx < len(q):
                    result[urdf_name] = q[syn_idx]
                else:
                    result[urdf_name] = 0.0
            return result

    # 5. 启动 MuJoCo
    print("\n启动 MuJoCo 查看器…")
    sim = MuJoCoSimulator(
        urdf_path, mesh_dir,
        synergy_callback=synergy_callback,
        synergy_motor_names=["Motor A", "Motor B"],
        ctrl_range_deg=3600.0,
        synergy_with_sigma_sliders=True,
        synergy_with_speed_slider=args.dynamic,  # 动态模式下显示 Speed 滑块
    )

    if args.dynamic:
        print("\n仿真已启动。")
        print("右侧面板有五个滑块：Motor A, Motor B, σ, σ_f, Speed (rad/s)")
        print("调节 Speed 滑块切换慢速/快速协同模式。")
    else:
        print("\n仿真已启动。")
        print("右侧面板有四个滑块：Motor A, Motor B, σ, σ_f")
    print("拖动任一滑块即可通过协同模型驱动手指运动。")
    print("\n操作提示：关闭 MuJoCo 查看器窗口退出。")
    sim.run()


if __name__ == "__main__":
    main()
