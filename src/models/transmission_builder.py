# src/models/transmission_builder.py
"""
从 OrigamiHandDesign 中提取传动矩阵 R 和 R_f。
数学模型基于 Della Santina 等人 2018 年的论文。

核心修复 v2（2025-05-11）：
1. σ mode: R 使用 Capstan 指数衰减加权（非均匀几何求和），
   物理上 Captures 中指驱动最小、拇指/小指最大的 U-shape 现象。
2. σ_f mode: Rf 使用 "Capstan 衰减 + slack 侧钳制" 模型，
   克服原论文线性模型在对称路径上 Rf=0 的问题。
   物理原理：张力从张紧的驱动器向远端指数衰减，衰减至低于
   静摩擦阈值的区段被"冻结"（不滑动），关节必须旋转来吸收
   缆绳长度变化。
3. 腱绳单向传力特性：张力不会为负（腱绳无法受推），
   放松侧自然产生 slack，进一步放大了非对称性。

参考文献：
  Della Santina et al. TRO 2018
  Capstan friction: T_out = T_in · exp(-μ·θ)
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from .origami_design import (OrigamiHandDesign, FoldLine, FoldType, JointConnection, Point2D,
                              is_hole_id, is_actuator_id, is_pulley_id)
from .hole_transmission import (find_hole_pairs, compute_hole_equivalent_R_row,
                                 compute_hole_lever_arm)


# ============================================================
# Capstan 摩擦衰减参数（全局）
# ============================================================
# β = μ · θ_eff  有效摩擦系数，包含 pulley wrap angle 和摩擦系数
# β=0.09 → 经过 42 个元素后 T≈exp(-3.78)≈0.023
# β=0.3  → 经过 10 个元素后 T≈exp(-3.0)≈0.05
DEFAULT_BETA = 0.09

# 静摩擦阈值：当张力衰减到初始值的多少倍时，该段被"冻结"
# 物理意义：静摩擦力可以阻止张力极小的区段滑动
DEFAULT_STICTION_THRESHOLD = 0.05  # 初始张力的 5%


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


def _get_element_radius(eid: int, design: OrigamiHandDesign) -> float:
    """获取任意类型元素的半径/等效半径。"""
    if is_pulley_id(eid) and eid in design.pulleys:
        return design.pulleys[eid].radius
    elif is_hole_id(eid) and eid in design.holes:
        return design.holes[eid].plate_offset
    return 0.0


def _get_element_friction(eid: int, design: OrigamiHandDesign) -> float:
    """获取任意类型元素的摩擦系数。"""
    if is_pulley_id(eid) and eid in design.pulleys:
        return design.pulleys[eid].friction_coefficient
    elif is_hole_id(eid) and eid in design.holes:
        return design.holes[eid].friction_coefficient
    return 0.3


def _get_element_joint_idx(eid: int, design: OrigamiHandDesign,
                            jid_to_idx: Dict[int, int]) -> Optional[int]:
    """获取元素关联的关节索引。"""
    if is_pulley_id(eid) and eid in design.pulleys:
        p = design.pulleys[eid]
        if p.attached_fold_line_id is not None:
            return jid_to_idx.get(p.attached_fold_line_id)
    elif is_hole_id(eid) and eid in design.holes:
        h = design.holes[eid]
        if h.attached_fold_line_id is not None:
            return jid_to_idx.get(h.attached_fold_line_id)
    return None


def compute_R_capstan(design: OrigamiHandDesign,
                      beta: float = DEFAULT_BETA) -> np.ndarray:
    """
    Capstan 指数衰减权重下的 R 矩阵。

    物理模型：
      当两个电机同时拉紧（σ 模式），张力从两端向中间指数衰减。
      每个关节上的传动比不是简单的半径求和，而是经 Capstan 衰减后的加权和：
        R[j] = Σ r_k · (exp(-β·d_Ak) + exp(-β·d_Bk)) / Z
      其中 d_Ak / d_Bk 是元素 k 到 Motor A / Motor B 的路径距离，
      Z = 2 · min(exp(-β·0) + exp(-β·(N-1)) = 2 为归一化因子。

      结果：中间关节的有效传动比小于两端关节 → U-shape → 中指驱动最小

    参数：
      design : OrigamiHandDesign
      beta : Capstan 摩擦系数（默认 0.09）

    返回：
      R : (num_tendons, n_joints) 数组
    """
    joints, jid_to_idx = get_joint_list(design)
    n = len(joints)
    if n == 0:
        return np.empty((0, 0))

    R_list = []
    for tendon in design.tendons.values():
        # 提取路径元素（排除驱动器）
        elements = [eid for eid in tendon.pulley_sequence
                    if eid >= 0 or is_hole_id(eid)]
        N = len(elements)
        if N == 0:
            R_list.append(np.zeros(n))
            continue

        row = np.zeros(n)
        for k, eid in enumerate(elements):
            r = _get_element_radius(eid, design)
            if r <= 0:
                continue
            j_idx = _get_element_joint_idx(eid, design, jid_to_idx)
            if j_idx is None:
                continue

            # 从 A 端传来的张力衰减
            d_A = float(k)            # A 端到元素 k 的距离（元素个数）
            w_A = np.exp(-beta * d_A)

            # 从 B 端传来的张力衰减
            d_B = float(N - 1 - k)    # B 端到元素 k 的距离
            w_B = np.exp(-beta * d_B)

            row[j_idx] += r * (w_A + w_B)

        # 归一化：使得平均有效传动比与几何求和一致
        # 几何求和 R_geo[j] ≈ r̄ · count_j
        # Capstan 加权 R_cap[j] ≈ r̄ · Σ(w_A + w_B)
        # 归一化目标：max(R_cap) ≈ max(R_geo)
        norm_factor = 1.0
        if np.max(np.abs(row)) > 0:
            # 归一化到 [0, 2*N*r̄] 量级
            # 实际上 w_A + w_B 在两端最大 ≈ 1+exp(-β·(N-1)) ≈ 1
            norm_factor = 2.0  # 使得两端权重 ~1

        row = row / norm_factor
        R_list.append(row)

    result = np.array(R_list) if R_list else np.empty((0, n))
    print(f"  R_capstan = {result}")
    return result


def compute_R(design: OrigamiHandDesign) -> np.ndarray:
    """
    计算基础传动矩阵 R (num_tendons x num_joints) [几何版本，兼容旧代码]。

    论文公式 (31): R^T_i = sum(r_j) over all pulleys on joint i

    现在内部委托给 compute_R_capstan() 以获得 U-shape 物理行为。
    """
    return compute_R_capstan(design, beta=DEFAULT_BETA)


def compute_Rf(design: OrigamiHandDesign,
               beta: float = DEFAULT_BETA,
               stiction_threshold: float = DEFAULT_STICTION_THRESHOLD) -> np.ndarray:
    """
    计算摩擦滑动传动矩阵 R_f (num_tendons x num_joints)。

    ============================================================
    物理模型（Capstan 衰减 + slack 侧钳制，v2）
    ============================================================

    原论文模型 (公式 20/25/32)：
      R_f^T = -AR^T · M^{-1} · V_max · e_v
    该模型假设整条腱绳在滑动（ṡ ≠ 0），张力对称分布。
    对于对称路径（如单手指 ohd_1），计算得 Rf=0，
    但这与物理实验不符。

    物理事实（基于真实实验）：
      当 σ_f > 0（Motor A 拉紧，Motor B 放松）时：
        1. Motor A 施加张力 T₀，Motor B 端张力 ≈ 0
        2. T₀ 沿路径向远端 Capstan 指数衰减：T_k = T₀ · exp(-β · k)
        3. 当 T_k < stiction_threshold · T₀ 时，该段摩擦力足以阻止滑动，
           该段被"冻结"（dead zone）
        4. 从 Motor A 到冻结点之间的关节，必须旋转来吸收缆绳长度变化
        5. 冻结点之后的关节不受影响（或受极小影响）
        6. 关键：这产生了**不对称**的关节角分布

    新模型：
      Rf[j] = Σ r_k_on_joint_j · w_k · sign_k

      其中：
        w_k = max(0, exp(-β·d_Ak) - exp(-β·d_Bk))
        sign_k = +1（A 侧元素）或 -1（B 侧元素）
        d_Ak / d_Bk：元素 k 到 Motor A / Motor B 的路径步数

      物理含义：
        - 靠近 A 端的元素：w_k > 0 → 当 σ_f > 0 时主动驱动
        - 靠近 B 端的元素：w_k < 0 → 当 σ_f < 0 时主动驱动
        - 中间元素（在冻结点之后）：w_k = 0 → dead zone，不贡献

    参数：
      design : OrigamiHandDesign
      beta : Capstan 衰减系数（默认 0.09）
      stiction_threshold : 静摩擦阈值（默认 0.05 = 初始张力的 5%）

    返回：
      Rf : (num_tendons, n_joints) 数组
    """
    joints, jid_to_idx = get_joint_list(design)
    n = len(joints)
    if n == 0:
        return np.empty((0, 0))

    Rf_list = []
    for tendon in design.tendons.values():
        # 提取路径元素（排除驱动器）
        elements = [eid for eid in tendon.pulley_sequence
                    if eid >= 0 or is_hole_id(eid)]
        N = len(elements)
        if N == 0:
            Rf_list.append(np.zeros(n))
            continue

        row = np.zeros(n)

        # ---- 计算每个元素的 Capstan 权重 ----
        for k, eid in enumerate(elements):
            r = _get_element_radius(eid, design)
            if r <= 0:
                continue
            j_idx = _get_element_joint_idx(eid, design, jid_to_idx)
            if j_idx is None:
                continue

            # 到 A 端距离
            d_A = float(k)

            # 到 B 端距离
            d_B = float(N - 1 - k)

            # 从 A 端传来的张力
            T_from_A = np.exp(-beta * d_A)

            # 从 B 端传来的张力
            T_from_B = np.exp(-beta * d_B)

            # σ_f 模式下的净张力：A 紧 B 松
            # 如果 T_from_A < stiction_threshold，该段冻结
            if T_from_A < stiction_threshold and T_from_B < stiction_threshold:
                # 两端都冻结 → dead
                w = 0.0
            elif T_from_A > T_from_B + 1e-10:
                # A 端占主导 → 正驱动权重
                w = T_from_A - np.clip(T_from_B, 0, stiction_threshold * 2)
            elif T_from_B > T_from_A + 1e-10:
                # B 端占主导 → 负驱动权重
                w = -(T_from_B - np.clip(T_from_A, 0, stiction_threshold * 2))
            else:
                # 几乎相等 → dead
                w = 0.0

            # 更简洁的实现：
            # 净张力 = T_from_A - T_from_B，但钳制到非负（腱绳不能有负张力）
            # slack 侧张力 = 0（不能是负的）
            # 所以有效净张力 = max(0, T_from_A - T_from_B) 或 max(0, T_from_B - T_from_A) 带符号
            if T_from_A > T_from_B:
                net = max(0.0, T_from_A - max(T_from_B, stiction_threshold * T_from_A))
                if net > 0:
                    row[j_idx] += r * net
            elif T_from_B > T_from_A:
                net = max(0.0, T_from_B - max(T_from_A, stiction_threshold * T_from_B))
                if net > 0:
                    row[j_idx] -= r * net
            # else equal: no contribution

        Rf_list.append(row)

    result = np.array(Rf_list) if Rf_list else np.empty((0, n))
    print(f"  Rf (Capstan+slack) = {result}")
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

    # 识别驱动腱绳
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
        T_mat = np.empty((0, n))
        C_diag = np.array([])

    # 合并 R 和 R_f 构建 R_aug
    if use_augmented and Rf.size > 0:
        m = Rf.shape[0]
        if m > 1:
            Rf = Rf.mean(axis=0, keepdims=True)
        R_aug = np.vstack([R, Rf])
    else:
        R_aug = R.copy()

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
