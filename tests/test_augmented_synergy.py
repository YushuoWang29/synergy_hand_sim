# scripts/test_augmented_synergy.py
import sys
sys.path.insert(0, '.')

import numpy as np
import os
import time

from src.models.origami_parser import OrigamiParser
from src.synergy.augmented_adaptive import AugmentedAdaptiveSynergyModel
from src.interactive.mujoco_simulator import MuJoCoSimulator

def main():
    # 1. 加载手
    dxf_path = "tests/test_2.dxf"
    parser = OrigamiParser(point_tolerance=2.0)
    design = parser.parse(dxf_path)
    n_joints = len(design.joints)
    E_vec = np.array(design.get_joint_stiffness_list())
    
    # 2. 定义传动矩阵 R (基础协同)
    k = 2
    R = np.zeros((k, n_joints))
    # 假设前3个是根关节，后3个是尖关节
    R[0, 0:3] = 1.0   # σ0 弯曲根关节
    R[1, 3:6] = 1.0   # σ1 弯曲尖关节
    
    # 3. 定义摩擦滑动传动矩阵 R_f (1 个额外 DoA)
    # 示例：让滑动使拇指(关节0)与其他手指反方向运动
    m = 1
    R_f = np.zeros((m, n_joints))
    R_f[0, 0] = 1.0    # 拇指
    R_f[0, 1] = -0.5   # 食指相反方向，力度减半
    R_f[0, 3] = 0.5    # 中指指根也参与
    
    # 4. 创建求解器
    model = AugmentedAdaptiveSynergyModel(n_joints, R, R_f, E_vec)
    print("S_aug:\n", model.S_aug)
    
    # 5. 查找 URDF
    hand_name = "test_2"
    urdf_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "models", hand_name)
    urdf_path = os.path.join(urdf_dir, f"{hand_name}.urdf")
    mesh_dir = os.path.join(urdf_dir, "meshes")
    
    if not os.path.exists(urdf_path):
        print("URDF missing. Run export_urdf first.")
        return
    
    # 6. 启动 MuJoCo
    sim = MuJoCoSimulator(urdf_path, mesh_dir)
    time.sleep(0.5)
    
    # 7. 测试：只使用基础协同弯曲根关节
    sigma = np.array([0.5, 0.0])
    sigma_f = np.array([0.0])
    q = model.solve(sigma, sigma_f)
    mj_joint_names = [f"joint_{j.id}" for j in design.joints]
    angle_dict = {name: np.degrees(q[i]) for i, name in enumerate(mj_joint_names)}
    sim.set_joint_angles(angle_dict)
    print("基础协同 (sigma_f=0): 角度 =", np.degrees(q))
    time.sleep(2)
    
    # 8. 添加滑动协同
    sigma_f = np.array([0.3])
    q = model.solve(sigma, sigma_f)
    for i, name in enumerate(mj_joint_names):
        angle_dict[name] = np.degrees(q[i])
    sim.set_joint_angles(angle_dict)
    print("添加滑动 (sigma_f=0.3): 角度 =", np.degrees(q))
    time.sleep(2)
    
    sim.reset()
    sim.close()

if __name__ == "__main__":
    main()