# scripts/synergy_mujoco_demo.py
"""
演示基础自适应协同驱动 MuJoCo 查看器。

用法：
    python scripts/synergy_mujoco_demo.py tests/test_2.dxf
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import os
import time
import argparse

from src.models.origami_parser import OrigamiParser
from src.synergy.base_adaptive import AdaptiveSynergyModel
from src.interactive.mujoco_simulator import MuJoCoSimulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dxf_file", help="DXF 手模型文件路径")
    parser.add_argument("--thickness", type=float, default=3.0,
                        help="折纸板材厚度 (mm)")
    parser.add_argument("--urdf-dir", default=None,
                        help="URDF/STL输出目录（默认根据dxf路径推断）")
    args = parser.parse_args()

    # 1. 加载手设计
    print(f"加载手模型: {args.dxf_file}")
    dxf_parser = OrigamiParser(point_tolerance=2.0)
    design = dxf_parser.parse(args.dxf_file)
    design.summary()

    # 2. 确保 URDF 和 STL 已存在
    hand_name = os.path.splitext(os.path.basename(args.dxf_file))[0]
    if args.urdf_dir:
        urdf_dir = args.urdf_dir
    else:
        urdf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "models", hand_name)
    urdf_path = os.path.join(urdf_dir, f"{hand_name}.urdf")
    mesh_dir = os.path.join(urdf_dir, "meshes")

    if not os.path.exists(urdf_path):
        print(f"URDF 不存在: {urdf_path}")
        print("请先运行 'python scripts/export_urdf.py {}' 生成 URDF。")
        sys.exit(1)

    # 3. 定义传动矩阵 R 和 刚度
    n_joints = len(design.joints)
    E_vec = np.array(design.get_joint_stiffness_list())

    # 使用两个基础协同：根关节 (0,1,2) 和 尖关节 (3,4,5) 分开控制
    k = 2
    R = np.zeros((k, n_joints))
    # 根据当前模型，根关节通常是前3个，尖关节后3个（取决于解析顺序）
    # 更稳健的方法：通过关节在面片树中的深度自动确定，但此处手动指定
    # 此处假设 joints 列表中前3个是根关节，后3个是尖关节
    if n_joints == 6:
        for i in range(3):
            R[0, i] = 1.0   # 根关节
        for i in range(3, 6):
            R[1, i] = 1.0   # 尖关节
    else:
        print(f"警告: 关节数 {n_joints} 不是6，将使用默认全驱动。")
        R = np.eye(n_joints)[:2]  # 随意取前两个关节

    # 也可做归一化使 R 行正交（简化情况）
    # R = R / np.linalg.norm(R, axis=1, keepdims=True)

    # 4. 创建求解器
    model = AdaptiveSynergyModel(n_joints, R, E_vec)
    print(f"协同矩阵 S:\n{model.S}")

    # 5. 建立关节名称映射: joint ID → MuJoCo 关节名
    # URDF 中关节名为 "joint_{id}"，id 对应 design.joints 中的 id
    joint_id_to_mj_name = {}
    for joint in design.joints:
        joint_id_to_mj_name[f"joint_{joint.id}"] = joint.id

    mj_joint_names = [f"joint_{j.id}" for j in design.joints]

    # 6. 启动 MuJoCo 查看器
    print(f"启动 MuJoCo: {urdf_path}")
    sim = MuJoCoSimulator(urdf_path, mesh_dir,
                          show_left_ui=True, show_right_ui=True)
    time.sleep(0.5)  # 给查看器时间初始化

    # 7. 协同扫描动画：逐渐增加 sigma_0，然后 sigma_1
    import math
    print("动画演示：协同0 (手指根关节弯曲) ...")
    for phase in np.linspace(0, math.pi/3, 30):
        sigma = np.array([phase, 0.0])
        q = model.solve(sigma)
        angles_deg = np.degrees(q)
        # 设置 MuJoCo 关节角度
        angle_dict = {}
        for j, name in enumerate(mj_joint_names):
            angle_dict[name] = angles_deg[j]
        sim.set_joint_angles(angle_dict)
        time.sleep(0.03)

    print("动画演示：协同1 (手指尖关节弯曲) ...")
    for phase in np.linspace(0, math.pi/4, 30):
        sigma = np.array([math.pi/3, phase])  # 保持根关节弯曲
        q = model.solve(sigma)
        angles_deg = np.degrees(q)
        angle_dict = {}
        for j, name in enumerate(mj_joint_names):
            angle_dict[name] = angles_deg[j]
        sim.set_joint_angles(angle_dict)
        time.sleep(0.03)

    input("按回车键重置并退出...")
    sim.reset()
    sim.close()

if __name__ == "__main__":
    main()