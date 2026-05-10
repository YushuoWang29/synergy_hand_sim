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
from .origami_design import (OrigamiHandDesign, FoldLine, FoldType, JointConnection, Point2D,
                              is_hole_id, is_actuator_id, is_pulley_id)
from .hole_transmission import (find_hole_pairs, compute_hole_equivalent_R_row,
                                 compute_hole_lever_arm)


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


def _has_holes(design: OrigamiHandDesign) -> bool:
    """检查设计中是否有腱绳包含孔元素"""
    return any(t.has_holes for t in design.tendons.values())


def compute_R(design: OrigamiHandDesign) -> np.ndarray:
    """
    计算基础传动矩阵 R (num_tendons x num_joints)。

    论文公式 (31): R^T_i = sum(r_j) over all pulleys on joint i

    扩展：当腱绳路径中包含孔时，孔对折痕的等效传动比
    使用 q=0 处的线性化值: R_hole[joint] = h (plate_offset)

    返回的 R 矩阵综合了滑轮和孔两部分贡献。
    """
    joints, jid_to_idx = get_joint_list(design)
    n = len(joints)
    if n == 0:
        return np.empty((0, 0))

    R_list = []
    for tendon in design.tendons.values():
        row = np.zeros(n)

        if tendon.has_holes:
            print(f"  Tendon {tendon.id}: mixing pulleys and holes")

            # ---- Part 1: 滑轮贡献 ----
            for pid in tendon.pulley_sequence:
                if pid < 0:
                    continue  # 跳过执行器（负 ID）
                if is_hole_id(pid):
                    continue  # 跳过孔元素
                pulley = design.pulleys.get(pid)
                if pulley is None or pulley.attached_fold_line_id is None:
                    continue
                j_idx = jid_to_idx.get(pulley.attached_fold_line_id)
                if j_idx is not None:
                    row[j_idx] += pulley.radius

            # ---- Part 2: 孔贡献 (q=0 线性化) ----
            hole_row = compute_hole_equivalent_R_row(design, tendon.id, jid_to_idx)
            row += hole_row
        else:
            # 纯滑轮路径，保持原有逻辑
            for pid in tendon.pulley_sequence:
                if pid < 0:
                    continue
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

    扩展 v2：同时支持滑轮和孔元素。
    - 孔被视为路径中的一个摩擦节点，plate_offset 作为等效"半径"（q=0 线性化），
      friction_coefficient 作为摩擦系数。
    - 路径中的所有元素（滑轮+孔）按顺序构成完整的摩擦链。
    """
    joints, jid_to_idx = get_joint_list(design)
    n = len(joints)
    if n == 0:
        return np.empty((0, 0))

    Rf_list = []
    for tendon in design.tendons.values():
        # ---- 1. 提取所有传动元素（滑轮和孔），排除驱动器 ----
        elements = []  # [(type_name, element_id), ...]
        for eid in tendon.pulley_sequence:
            if is_pulley_id(eid) or is_hole_id(eid):
                elements.append((eid,))
        m = len(elements)
        if m == 0:
            Rf_list.append(np.zeros(n))
            continue

        # ---- 2. 构建 AR matrix (m+1) x n ----
        AR = np.zeros((m + 1, n))
        for seg_idx, (eid,) in enumerate(elements):
            if is_pulley_id(eid):
                pulley = design.pulleys.get(eid)
                if pulley and pulley.attached_fold_line_id is not None:
                    j_idx = jid_to_idx.get(pulley.attached_fold_line_id)
                    if j_idx is not None:
                        AR[seg_idx, j_idx] = pulley.radius
            elif is_hole_id(eid):
                hole = design.holes.get(eid)
                if hole and hole.attached_fold_line_id is not None:
                    j_idx = jid_to_idx.get(hole.attached_fold_line_id)
                    if j_idx is not None:
                        # 孔在 q=0 处等效传动比 = plate_offset (h)
                        AR[seg_idx, j_idx] = hole.plate_offset

        # ---- 3. 构建 M 矩阵 (m+1) x (m+1) ----
        M = np.zeros((m + 1, m + 1))
        for i in range(m):
            M[i, i] = -1.0
            M[i, i + 1] = 1.0
        M[m, 0] = 1.0
        M[m, m] = 1.0

        # ---- 4. 构建 V_max 对角阵 (m+1) x (m+1) ----
        V_max = np.zeros((m + 1, m + 1))
        for seg_idx, (eid,) in enumerate(elements):
            if is_pulley_id(eid):
                pulley = design.pulleys.get(eid)
                if pulley:
                    V_max[seg_idx, seg_idx] = pulley.friction_coefficient
            elif is_hole_id(eid):
                hole = design.holes.get(eid)
                if hole:
                    V_max[seg_idx, seg_idx] = hole.friction_coefficient

        # ---- 5. e_v ----
        e_v = np.ones(m + 1)

        # ---- 6. 计算 R_f^T = -AR^T @ M^{-1} @ V_max @ e_v ----
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M)

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

    # 识别"驱动腱绳"——只有包含驱动器电机(-1/-2)的腱绳才是真正的驱动腱绳。
    # 纯阻尼器腱绳（如序列 [-200, -171, ..., -176]）不连接电机，
    # 不应参与 R/Rf 的平均，否则会稀释传动矩阵并改变协同方向。
    drive_indices = [i for (i, t) in enumerate(design.tendons.values())
                     if -1 in t.pulley_sequence or -2 in t.pulley_sequence]
    if len(drive_indices) == 0:
        raise ValueError("设计中没有包含驱动器(-1/-2)的驱动腱绳。")

    if len(drive_indices) < R.shape[0]:
        n_excluded = R.shape[0] - len(drive_indices)
        print(f"  Excluded {n_excluded} non-drive tendon(s) from R/Rf averaging.")
        R = R[drive_indices]
        Rf = Rf[drive_indices]

    # All drive tendons share the same A/B motors, so they all receive the same
    # sigma = (θ_A + θ_B)/2 and sigma_f = (θ_A - θ_B)/2 inputs.
    # Average tendon rows into a single effective R and Rf (k=1, m=1).
    # This models the physical reality: there is only 1 sigma and 1 sigma_f
    # driving all tendons simultaneously.
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


def compute_damper_T(design: OrigamiHandDesign) -> np.ndarray:
    """
    计算阻尼器传动矩阵 T (n_dampers x n_joints)。

    论文公式 (2): T q = x
    其中 x 为阻尼器位移，T 将关节角 q 映射到阻尼器位移。

    每个 Damper 的 attached_fold_line_ids + transmission_ratios
    定义了 T 矩阵的对应行。
    """
    joints, jid_to_idx = get_joint_list(design)
    n = len(joints)
    n_d = len(design.dampers)
    if n == 0 or n_d == 0:
        return np.empty((0, n))

    T = np.zeros((n_d, n))
    for idx, dm in enumerate(design.dampers.values()):
        for fold_id, ratio in zip(dm.attached_fold_line_ids, dm.transmission_ratios):
            j_idx = jid_to_idx.get(fold_id)
            if j_idx is not None:
                T[idx, j_idx] = ratio
            else:
                print(f"  WARNING: Damper {dm.id}: fold_line {fold_id} "
                      f"not found in joint map")

    print(f"  Damper T = {T}")
    return T


def build_dynamic_synergy_model(
    design: OrigamiHandDesign,
    use_augmented: bool = True,
    speed_factor: float = 0.0
):
    """
    从设计中构建完整的 dynamic synergy 模型。
    可单独使用或与 augmented synergy 结合。

    Parameters
    ----------
    design : OrigamiHandDesign
        折纸手设计（含 pulleys, holes, dampers）。
    use_augmented : bool
        如果为 True，将 R_f 与 R 合并为 R_aug，构建
        DynamicSynergyModel 以兼容 augmented synergy。
        如果为 False，仅使用基础 R 构建。
    speed_factor : float
        默认速度因子（0=慢速，1=快速）。

    Returns
    -------
    model : DynamicSynergyModel
    info : dict
        包含 'joint_names', 'E_vec', 'n_joints', 'n_dampers',
        'has_augmented', 'T_diag' 等辅助信息。
    """
    joints, _ = get_joint_list(design)
    n = len(joints)
    if n == 0:
        raise ValueError("设计中没有任何关节。")

    R = compute_R(design)          # shape (k, n)
    Rf = compute_Rf(design)        # shape (m, n)
    T_mat = compute_damper_T(design)  # shape (n_d, n)
    E_vec = np.array([
        design.fold_lines[j.fold_line_id].stiffness
        for j in joints
    ])

    n_tendons = R.shape[0]
    n_dampers = T_mat.shape[0]

    if R.size == 0:
        raise ValueError("设计中未找到任何肌腱。")

    # 识别"驱动腱绳"——只对包含驱动器(-1/-2)的腱绳做平均
    drive_indices = [i for (i, t) in enumerate(design.tendons.values())
                     if -1 in t.pulley_sequence or -2 in t.pulley_sequence]
    if len(drive_indices) == 0:
        raise ValueError("设计中没有包含驱动器(-1/-2)的驱动腱绳。")

    if len(drive_indices) < n_tendons:
        n_excluded = n_tendons - len(drive_indices)
        print(f"  Dynamic: Excluded {n_excluded} non-drive tendon(s) from R/Rf averaging.")
        R = R[drive_indices]
        Rf = Rf[drive_indices]
        n_tendons = R.shape[0]

    # 平均多腱绳为单输入
    if n_tendons > 1:
        R = R.mean(axis=0, keepdims=True)
        Rf = Rf.mean(axis=0, keepdims=True)

    # 阻尼系数向量
    C_diag = np.array([
        dm.damping_coefficient
        for dm in design.dampers.values()
    ])

    if n_dampers == 0:
        print("  No dampers found. Using standard synergy (no dynamic effect).")
        # 仍然返回模型，但 T 为空矩阵 → 无阻尼，动态效果不起作用
        T_mat = np.empty((0, n))
        C_diag = np.array([])

    # 合并 R 和 R_f 构建 R_aug
    if use_augmented and Rf.size > 0:
        m = Rf.shape[0]
        # 平均 Rf 为单行
        if m > 1:
            Rf = Rf.mean(axis=0, keepdims=True)  # (1, n)
        R_aug = np.vstack([R, Rf])  # (2, n)
    else:
        R_aug = R.copy()  # (k, n)

    from src.synergy.dynamic_synergy import DynamicSynergyModel

    model = DynamicSynergyModel(
        n_joints=n,
        R=R_aug,
        E_vec=E_vec,
        T=T_mat,
        C_diag=C_diag
    )

    joint_names = [f"joint_{j.id}" for j in joints]
    info = {
        'joint_names': joint_names,
        'E_vec': E_vec,
        'n_joints': n,
        'n_dampers': n_dampers,
        'has_augmented': use_augmented and Rf.size > 0,
        'T_diag': C_diag,
        'speed_factor': speed_factor,
    }
    return model, info


