# src/models/transmission_builder.py
"""
从 OrigamiHandDesign 中提取传动矩阵 R 和 R_f。
数学模型基于 Della Santina 等人 2018 年的论文。

核心修复：
1. 直接使用 .ohd 文件中的原始 fold line ID 创建关节映射，
   避免 build_topology() 破坏 ID。
2. R_f 按论文公式 (20)/(25)/(32) 正确计算，无需路由置换矩阵 P，
   因为 pulley_sequence 的顺序已经自然编码了路径信息。
3. MuJoCo 协同模式用 dummy joint + kp=0 防止 actuator 冲突。
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from .origami_design import OrigamiHandDesign, FoldLine, FoldType, JointConnection, Point2D


def get_joint_list(design: OrigamiHandDesign) -> Tuple[List[JointConnection], Dict[int, int]]:
    """
    从设计中提取所有关节 JointConnection 及其索引映射。

    如果设计尚未重建拓扑（joints 为空），直接根据 fold_type 为
    VALLEY/MOUNTAIN 的 fold lines 创建关节映射，以保留原始 fold line ID，
    从而保持 pulley → fold line → joint 的映射关系不断裂。
    """
    if design.joints:
        # 已有拓扑重建结果
        joints = sorted(design.joints, key=lambda j: j.id)
        jid_to_idx = {j.fold_line_id: idx for idx, j in enumerate(joints)}
        return joints, jid_to_idx

    # 无拓扑信息时，直接从 fold-type fold lines 创建 JointConnection
    # 这保留了 .ohd 文件中的原始 fold line ID
    fold_lines_sorted = sorted(
        [fl for fl in design.fold_lines.values()
         if fl.fold_type in (FoldType.VALLEY, FoldType.MOUNTAIN)],
        key=lambda fl: fl.id
    )

    joints = []
    jid_to_idx = {}
    for i, fl in enumerate(fold_lines_sorted):
        joint = JointConnection(
            id=i,
            fold_line_id=fl.id,
            face_a_id=-1,  # 占位符，不影响计算
            face_b_id=-1,
            fold_type=fl.fold_type
        )
        joints.append(joint)
        jid_to_idx[fl.id] = i

    # 诊断：检查 pulley → joint 映射
    n_mapped = sum(
        1 for p in design.pulleys.values()
        if p.attached_fold_line_id is not None
        and p.attached_fold_line_id in jid_to_idx
    )
    n_total = sum(
        1 for p in design.pulleys.values()
        if p.attached_fold_line_id is not None
    )
    joint_fids = sorted(jid_to_idx.keys())
    print(f"  Joint-fold_line IDs: {joint_fids}")
    print(f"  Pulley-to-joint mapping: {n_mapped}/{n_total} pulleys mapped")

    if n_mapped < n_total:
        for pid, p in design.pulleys.items():
            if p.attached_fold_line_id is not None and p.attached_fold_line_id not in jid_to_idx:
                fl = design.fold_lines.get(p.attached_fold_line_id)
                if fl:
                    print(f"    WARNING: pulley {pid} at ({p.position.x:.1f},{p.position.y:.1f}) "
                          f"-> fold_line {p.attached_fold_line_id} ({fl.fold_type.value}) "
                          f"is not in joint map (only OUTLINE edges are excluded)")

    return joints, jid_to_idx


def compute_R(design: OrigamiHandDesign) -> np.ndarray:
    """
    计算基础传动矩阵 R (num_tendons x num_joints)。

    论文公式 (31): R^T_i = sum(r_j) over all pulleys on joint i
    对于每个关节，求和所有连接到该关节的滑轮半径。
    """
    joints, jid_to_idx = get_joint_list(design)
    n = len(joints)
    if n == 0:
        return np.empty((0, 0))

    R_list = []
    for tendon in design.tendons.values():
        row = np.zeros(n)
        for pid in tendon.pulley_sequence:
            if pid < 0:
                continue  # 跳过执行器（负 ID）
            pulley = design.pulleys.get(pid)
            if pulley is None or pulley.attached_fold_line_id is None:
                continue
            j_idx = jid_to_idx.get(pulley.attached_fold_line_id)
            if j_idx is not None:
                row[j_idx] += pulley.radius
        R_list.append(row)

    result = np.array(R_list) if R_list else np.empty((0, n))
    print(f"  R = {result}")
    return result


def compute_Rf(design: OrigamiHandDesign) -> np.ndarray:
    """
    计算摩擦滑动传动矩阵 R_f (num_tendons x num_joints)。

    论文公式 (20)/(25)/(32):
        R_f^T = -AR^T . M^{-1} . V_max . e_v

    其中：
        AR in R^{(m+1)x n}:   传动半径矩阵
        M in R^{(m+1)x(m+1)}: 速度耦合矩阵 (27)
        V_max in R^{(m+1)x(m+1)}: 摩擦系数对角阵 (30)
        e_v = [1,...,1]^T in R^{m+1}
        m = pulley 数量

    注意：pulley_sequence 已经包含路径顺序信息，因此无需额外的 P 矩阵。
    两种路由（ohd_2 vs ohd_3）具有不同的 pulley_sequence，因此
    AR 矩阵中的元素分布不同，导致 R_f 也不同。
    """
    joints, jid_to_idx = get_joint_list(design)
    n = len(joints)
    if n == 0:
        return np.empty((0, 0))

    Rf_list = []
    for tendon in design.tendons.values():
        pulley_seq = [pid for pid in tendon.pulley_sequence if pid >= 0]
        m = len(pulley_seq)  # pulley 数量
        if m == 0:
            Rf_list.append(np.zeros(n))
            continue

        # ---- 1. 构建 AR in R^{(m+1)x n} ----
        # AR[seg_idx, j_idx] = pulley radius if pulley at seg_idx is on joint j_idx
        # 注意：pulley_seq[seg_idx] 对应的滑轮影响段 seg_idx（公式 8）
        AR = np.zeros((m + 1, n))
        for seg_idx, pid in enumerate(pulley_seq):
            pulley = design.pulleys.get(pid)
            if pulley is None or pulley.attached_fold_line_id is None:
                continue
            j_idx = jid_to_idx.get(pulley.attached_fold_line_id)
            if j_idx is not None:
                AR[seg_idx, j_idx] = pulley.radius
        # 最后一个段（第 m+1 段）没有对应的滑轮，保持零行

        # ---- 2. 构建 M 矩阵 (m+1)x(m+1) - 公式 (27) ----
        M = np.zeros((m + 1, m + 1))
        for i in range(m):
            M[i, i] = -1.0
            M[i, i + 1] = 1.0
        M[m, 0] = 1.0
        M[m, m] = 1.0

        # ---- 3. 构建 V_max 对角阵 (m+1)x(m+1) - 公式 (30) ----
        V_max = np.zeros((m + 1, m + 1))
        for seg_idx, pid in enumerate(pulley_seq):
            pulley = design.pulleys.get(pid)
            if pulley is not None:
                V_max[seg_idx, seg_idx] = pulley.friction_coefficient
        # V_max[m, m] = 0（最后一个段无滑轮 -> 无摩擦损失）

        # ---- 4. e_v = [1,...,1]^T ----
        e_v = np.ones(m + 1)

        # ---- 5. 计算 R_f^T = -AR^T . M^{-1} . V_max . e_v ----
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

        # R_f^T in R^n
        Rf_vec = -AR.T @ M_inv @ V_max @ e_v
        Rf_list.append(Rf_vec)

    result = np.array(Rf_list) if Rf_list else np.empty((0, n))
    print(f"  Rf = {result}")
    return result


def build_synergy_model(design: OrigamiHandDesign):
    """
    从设计中构建完整的增强自适应协同模型。
    返回 (model, joint_stiffness)，其中 model 为 AugmentedAdaptiveSynergyModel 实例，
    joint_stiffness 为 {joint_name: stiffness} 便于后续使用。
    """
    joints, _ = get_joint_list(design)
    n = len(joints)
    if n == 0:
        raise ValueError("设计中没有任何关节。")

    R = compute_R(design)
    Rf = compute_Rf(design)
    E_vec = np.array([
        design.fold_lines[j.fold_line_id].stiffness
        for j in joints
    ])

    if R.size == 0:
        raise ValueError("设计中未找到任何肌腱，无法构建驱动模型。")

    # CRITICAL: All tendons share the same A/B motors, so they all receive the same
    # sigma = (θ_A + θ_B)/2 and sigma_f = (θ_A - θ_B)/2 inputs.
    # Instead of treating each tendon as an independent actuator input (k=num_tendons, m=num_tendons),
    # average the tendon rows into a single effective R and Rf (k=1, m=1).
    # This means: R_avg * q = sigma, Rf_avg * q = sigma_f — with only 1 sigma and 1 sigma_f input.
    num_tendons = R.shape[0]
    if num_tendons > 1:
        R = R.mean(axis=0, keepdims=True)    # shape (1, n)
        Rf = Rf.mean(axis=0, keepdims=True)  # shape (1, n)

    from src.synergy.augmented_adaptive import AugmentedAdaptiveSynergyModel
    model = AugmentedAdaptiveSynergyModel(n, R, Rf, E_vec)

    # 显示协同方向
    print(f"\n  Synergy directions (from S_aug):")
    for i in range(model.S_aug.shape[1]):
        print(f"    Direction {i}: {model.S_aug[:, i]}")

    joint_names = [f"joint_{j.id}" for j in joints]
    return model, dict(zip(joint_names, E_vec))
