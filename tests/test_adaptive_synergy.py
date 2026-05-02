# scripts/test_adaptive_synergy.py

import sys
sys.path.insert(0, '.')

import numpy as np
import os

from src.models.origami_parser import OrigamiParser
from src.synergy.base_adaptive import AdaptiveSynergyModel

def main():
    # 1. 加载手设计
    dxf_path = "tests/test_2.dxf"
    print(f"Loading hand from {dxf_path}")
    parser = OrigamiParser(point_tolerance=2.0)
    design = parser.parse(dxf_path)
    design.summary()
    
    # 2. 获取关节数和刚度
    n_joints = len(design.joints)
    E_vec = np.array(design.get_joint_stiffness_list())
    print(f"\nNumber of joints: {n_joints}")
    print(f"Joint stiffnesses: {E_vec}")
    
    # 3. 定义传动矩阵 R (或协同矩阵 S)
    # 为了方便，直接定义一个简单的 R：每个协同分别控制一部分关节
    # 例如 k=2: 第一个协同弯曲所有手指根关节，第二个协同弯曲所有手指尖关节
    k = 2
    R = np.zeros((k, n_joints))
    
    # 根据 test_2 的关节分布：
    # joint_0, joint_1, joint_2 是谷折（手指根） → 第0协同
    # joint_3, joint_4, joint_5 是峰折（手指尖） → 第1协同
    # 注意：设计中的关节顺序与 URDF 导出顺序一致，即 joints 列表顺序
    # 我们在此假设 joints[0..2] 是根关节，joints[3..5] 是尖关节
    for i in range(3):
        R[0, i] = 1.0   # 根关节受 sigma[0] 控制
    for i in range(3, 6):
        R[1, i] = 1.0   # 尖关节受 sigma[1] 控制
    
    # 4. 创建求解器
    model = AdaptiveSynergyModel(n_joints, R, E_vec)
    print("\nSynergy matrix S:")
    print(model.S)
    
    # 5. 自由运动测试：只激活根关节协同
    sigma = np.array([0.5, 0.0])  # 0.5 rad ≈ 28.6°
    q = model.solve(sigma)
    print(f"\nInput synergy: {sigma}")
    print(f"Output joint angles (rad): {q}")
    print(f"Output joint angles (deg): {np.degrees(q)}")
    
    # 6. 验证：根关节应非零，尖关节应接近零
    assert abs(q[0]) > 0.1, "Root joints should be non-zero"
    assert abs(q[3]) < 1e-10, "Tip joints should be zero"
    print("Basic test passed!")
    
    # 7. 后续可视化（可选）：将 q 输出为文件，供 MuJoCo 查看器读取
    # 或者在下一轮对话中集成

if __name__ == "__main__":
    main()