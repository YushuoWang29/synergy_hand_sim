"""
孔类腱绳传动计算模块。
================================

数学推导
--------
考虑 2D 截面（正视折痕轴线 x）：
  - 折痕 O 在 (0,0)
  - 初始 q=0: 孔 A(-d, h), 孔 B(d, h)
    d = 两孔水平半间距, h = plate_offset (板b上表面到折痕平面的高度)
  - 折叠时: B 固定, A绕O旋转角度 q (谷折)
    
旋转后 A(q):
  A_y = -d*cos(q) + h*sin(q)
  A_z = d*sin(q) + h*cos(q)

腱绳（AB线）到折痕O的垂直距离（驱动力臂）：
  arm(q) = |det(A, B)| / |B - A|
  
其中 det(A, B) = A_y * B_z - A_z * B_y

代入 B = (d, h)：
  det = (-d*cos(q) + h*sin(q))*h - (d*sin(q) + h*cos(q))*d
      = -2*d*h*cos(q) + (h² - d²)*sin(q)
  
  |AB| = sqrt((d + d*cos(q) - h*sin(q))² + (h - d*sin(q) - h*cos(q))²)

验证：
  q=0:   arm = h   (水平线, 高度h)
  q=π/2: arm = (d+h)/√2
  q=π:   arm = 0   (AB线经过原点)

关节驱动力矩: τ(q) = T * arm(q)     (T = 腱绳张力)

与滑轮模型的对比：
  滑轮: τ = r * T        (常数传动比)
  孔:   τ = arm(q) * T   (q依赖传动比)
  
对于协同框架的线性化近似：
  R_hole[i] = arm(0) = h_i  (在q=0处等效传动比)

用法
----
```python
from .hole_transmission import compute_hole_lever_arm

# 对一对孔计算力臂
arm = compute_hole_lever_arm(d=10.0, h=3.0, q=np.pi/4)
```

参考文献
--------
Della Santina et al. "Toward Dexterous Manipulation With Augmented
Adaptive Synergies: The Pisa/IIT SoftHand 2" IEEE TRO 2018.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from .origami_design import OrigamiHandDesign, Hole, FoldLine, Point2D, is_hole_id, is_actuator_id


@dataclass
class HolePairParams:
    """
    描述跨过某折痕的一对孔的几何参数。
    
    Attributes:
        hole_a_id: 折痕一侧A的孔ID
        hole_b_id: 折痕另一侧B的孔ID
        d: 两孔到折痕垂直投影线的水平半间距
        h: plate_offset（板b上表面到折痕平面的高度）
        fold_line_id: 关联折痕ID
    """
    hole_a_id: int
    hole_b_id: int
    d: float
    h: float
    fold_line_id: int


def compute_hole_lever_arm(d: float, h: float, q: float) -> float:
    """
    计算孔类腱绳的驱动力臂（精确解，无小量近似）。
    
    Parameters
    ----------
    d : float
        两孔到折痕的水平半间距（初始 q=0 时 A(-d,h), B(d,h)）
    h : float
        plate_offset（板b上表面到折痕平面的高度）
    q : float
        折叠角度 (radians)。正值为谷折。
    
    Returns
    -------
    arm : float
        腱绳 AB 到折痕 O 的垂直距离（驱动力臂）
    
    Raises
    ------
    ValueError
        当 d <= 0 或 h <= 0 时
    """
    if d <= 0:
        raise ValueError(f"d must be > 0, got {d}")
    if h <= 0:
        raise ValueError(f"h must be > 0, got {h}")
    
    # 安全处理大的角度（避免溢出）
    if abs(q) > np.pi:
        q = np.sign(q) * np.pi  # clamp to ±π
    
    cos_q = np.cos(q)
    sin_q = np.sin(q)
    
    # A 点坐标（旋转后）
    A_y = -d * cos_q + h * sin_q
    A_z = d * sin_q + h * cos_q
    
    # B 点坐标（固定）
    B_y = d
    B_z = h
    
    # 行列式 det(A, B) = A_y * B_z - A_z * B_y
    det = A_y * B_z - A_z * B_y
    
    # 向量 AB
    AB_y = B_y - A_y
    AB_z = B_z - A_z
    AB_norm = np.sqrt(AB_y**2 + AB_z**2)
    
    if AB_norm < 1e-12:
        return 0.0  # A 和 B 重合
    
    # 力臂 = |det(A,B)| / |AB|
    arm = abs(det) / AB_norm
    
    return arm


def compute_hole_transmission_ratio(d: float, h: float, q: float = 0.0) -> float:
    """
    计算孔类腱绳在给定角度 q 下的等效传动比。
    
    在 q=0 时等效传动比即为 plate_offset h。
    对于协同框架的设计使用 q=0 (线性化近似)。
    
    参数对照:
        pulley: τ = r * T,    R = r (常数)
        hole:   τ = arm(q) * T, R_eq = h (q=0近似)
    
    Parameters
    ----------
    d : float
        两孔水平半间距
    h : float
        plate_offset
    q : float
        折叠角度，默认 0
    
    Returns
    -------
    ratio : float
        等效传动比 τ/T
    """
    if q == 0.0:
        return h  # arm(0) = h
    return compute_hole_lever_arm(d, h, q)


def find_hole_pairs(design: OrigamiHandDesign, tendon_id: int) -> List[HolePairParams]:
    """
    从设计和腱绳路径中提取跨折痕的孔对。
    
    算法：遍历腱绳路径的 pulley_sequence 列表，对每对连续的孔元素，
    检查它们是否位于同一个折痕的两侧（用 face_id 或几何计算判断）。
    如果是，计算 d（两孔到折痕的半间距）和 h（板厚偏移）。
    **支持同一条折痕上有多对孔**（类似于滑轮模型中一条折痕上
    可以有多个滑轮）。
    
    Parameters
    ----------
    design : OrigamiHandDesign
    tendon_id : int
    
    Returns
    -------
    pairs : List[HolePairParams]
        跨折痕孔对列表（每个元素对应一条腱绳段驱动一个折痕）
    """
    if tendon_id not in design.tendons:
        raise ValueError(f"Tendon {tendon_id} not found")
    
    tendon = design.tendons[tendon_id]
    seq = tendon.pulley_sequence
    
    pairs: List[HolePairParams] = []
    
    # 找到所有孔在序列中的索引和 ID
    hole_positions = []  # [(seq_index, hole_id), ...]
    for i, eid in enumerate(seq):
        if is_hole_id(eid):
            hole_positions.append((i, eid))
    
    # 遍历相邻孔对：路径中每两个相邻孔构成一对
    for j in range(len(hole_positions) - 1):
        idx_a, hid_a = hole_positions[j]
        idx_b, hid_b = hole_positions[j + 1]
        
        hole_a = design.holes.get(hid_a)
        hole_b = design.holes.get(hid_b)
        
        if hole_a is None or hole_b is None:
            continue
        
        # 步骤1：检查是否关联到同一个折痕
        fl_a = hole_a.attached_fold_line_id
        fl_b = hole_b.attached_fold_line_id
        
        if fl_a is not None and fl_b is not None and fl_a == fl_b:
            # 两个孔都关联到同一个折痕，正常流程
            fold_line = design.fold_lines.get(fl_a)
        else:
            # ===== 几何回退：通过腱绳跨过折痕来识别 =====
            # 如果 attached_fold_line_id 未设置或不匹配（例如旧文件使用最近原则），
            # 尝试通过两孔连线穿越哪条折痕来识别
            fold_line = None
            pos_a = np.array([hole_a.position.x, hole_a.position.y])
            pos_b = np.array([hole_b.position.x, hole_b.position.y])
            
            best_dist = float('inf')
            best_fold_id = None
            best_normal = None
            
            for fid, fl in design.fold_lines.items():
                if not fl.is_fold:
                    continue
                start = np.array([fl.start.x, fl.start.y])
                end = np.array([fl.end.x, fl.end.y])
                d1 = end - start
                d2 = pos_b - pos_a
                cross = np.cross(d1, d2)
                if abs(cross) < 1e-10:
                    continue
                t = np.cross(pos_a - start, d2) / cross
                u = np.cross(pos_a - start, d1) / cross
                eps = 1e-6
                if t >= -eps and t <= 1 + eps and u >= -eps and u <= 1 + eps:
                    # 两孔连线穿越了这条折痕
                    mid = (pos_a + pos_b) / 2
                    intersect_pt = start + t * d1
                    dist_to_mid = np.linalg.norm(intersect_pt - mid)
                    if dist_to_mid < best_dist:
                        best_dist = dist_to_mid
                        best_fold_id = fid
                        d1_norm = d1 / np.linalg.norm(d1)
                        best_normal = np.array([-d1_norm[1], d1_norm[0]])
            
            if best_fold_id is not None:
                # 使用几何识别的折痕
                fold_line = design.fold_lines.get(best_fold_id)
                # 更新孔的 attached_fold_line_id 和 face_id（如果尚未设置）
                if fl_a is None:
                    hole_a.attached_fold_line_id = best_fold_id
                    if best_normal is not None:
                        v_a = pos_a - np.array([fold_line.start.x, fold_line.start.y])
                        hole_a.face_id = 1 if np.dot(v_a, best_normal) > 0 else -1
                if fl_b is None:
                    hole_b.attached_fold_line_id = best_fold_id
                    if best_normal is not None:
                        v_b = pos_b - np.array([fold_line.start.x, fold_line.start.y])
                        hole_b.face_id = 1 if np.dot(v_b, best_normal) > 0 else -1
            elif fl_a is not None:
                # 仍然使用原有的关联（可能是未匹配或一侧无关联）
                fold_line = design.fold_lines.get(fl_a)
            elif fl_b is not None:
                fold_line = design.fold_lines.get(fl_b)
            else:
                continue  # 完全无法识别
        
        if fold_line is None:
            continue
        
        # 步骤2：用几何坐标精确计算两孔到折痕的有符号距离
        start = np.array([fold_line.start.x, fold_line.start.y])
        end = np.array([fold_line.end.x, fold_line.end.y])
        direction = end - start
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            continue
        direction = direction / dir_norm
        
        pos_a = np.array([hole_a.position.x, hole_a.position.y])
        pos_b = np.array([hole_b.position.x, hole_b.position.y])
        
        # 有符号垂直距离（2D叉积 = direction_x * v_y - direction_y * v_x）
        v_a = pos_a - start
        v_b = pos_b - start
        cross_a = direction[0] * v_a[1] - direction[1] * v_a[0]
        cross_b = direction[0] * v_b[1] - direction[1] * v_b[0]
        
        if cross_a * cross_b >= 0:
            # 同侧，无法形成跨折痕对
            continue
        
        # === 关键约束：两个孔必须沿折痕方向对齐 ===
        # 物理意义：一对孔应垂直跨过折痕（在折痕的同一位置附近）
        # 计算两孔在折痕线上的投影位置 t（参数化位置）
        t_a = np.dot(v_a, direction)  # 孔A在折痕线上的投影位置
        t_b = np.dot(v_b, direction)  # 孔B在折痕线上的投影位置
        alignment_tol = dir_norm * 0.3  # 允许偏移量（折痕长度的30%）
        if abs(t_a - t_b) > alignment_tol:
            # 两个孔在折痕方向上相距太远，不是同一对
            continue
        
        # 半间距 d = (距离A + 距离B) / 2
        dist_a = abs(cross_a)
        dist_b = abs(cross_b)
        d = (dist_a + dist_b) / 2.0
        
        # h = plate_offset（两个孔的平均）
        h = (hole_a.plate_offset + hole_b.plate_offset) / 2.0
        
        pairs.append(HolePairParams(
            hole_a_id=hid_a,
            hole_b_id=hid_b,
            d=d,
            h=h,
            fold_line_id=fold_line.id  # 使用步骤1确定的折痕ID（支持几何回退）
        ))
    
    return pairs


def compute_hole_equivalent_R_row(design: OrigamiHandDesign, tendon_id: int,
                                   joint_fold_map: Dict[int, int]) -> np.ndarray:
    """
    计算腱绳中孔类元素对传动矩阵 R 的贡献行向量。
    
    在协同框架中，孔的等效传动比用 q=0 处线性化值:
    R_hole[joint_idx] = arm(0) = h （plate_offset）
    
    注意：滑轮部分仍由 transmission_builder 中的 compute_R 处理。
    本函数仅处理孔部分。
    
    Parameters
    ----------
    design : OrigamiHandDesign
    tendon_id : int
    joint_fold_map : Dict[int, int]
        {fold_line_id: joint_index}
    
    Returns
    -------
    hole_row : np.ndarray
        形状 (num_joints,)，孔对 R 的贡献
    """
    n_joints = max(joint_fold_map.values()) + 1 if joint_fold_map else 0
    hole_row = np.zeros(n_joints)
    
    pairs = find_hole_pairs(design, tendon_id)
    
    for pair_params in pairs:
        j_idx = joint_fold_map.get(pair_params.fold_line_id)
        if j_idx is not None:
            # 等效传动比: arm(0) = h
            hole_row[j_idx] += pair_params.h
    
    return hole_row


def compute_hole_torque(d: float, h: float, q: float, tension: float) -> float:
    """
    计算孔类腱绳在给定角度和张力下的驱动力矩。
    
    适用于动态仿真中替代滑轮模型。
    
    τ(q) = tension * arm(q)
    
    Parameters
    ----------
    d : float
        两孔水平半间距
    h : float
        plate_offset
    q : float
        当前折叠角度
    tension : float
        腱绳张力
    
    Returns
    -------
    torque : float
        作用在折痕上的驱动力矩
    """
    arm = compute_hole_lever_arm(d, h, q)
    return tension * arm


def plot_arm_vs_q(d=10.0, h=3.0, q_max=np.pi/2):
    """
    快速可视化力臂随折叠角度的变化。
    用于调试和理解孔传动特性。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    qs = np.linspace(0, q_max, 100)
    arms = [compute_hole_lever_arm(d, h, q) for q in qs]
    
    plt.figure(figsize=(8, 5))
    plt.plot(qs * 180 / np.pi, arms, 'b-', linewidth=2)
    plt.axhline(y=h, color='r', linestyle='--', alpha=0.5, label=f'arm(0) = h = {h}')
    plt.xlabel('Fold angle q (degrees)')
    plt.ylabel('Lever arm')
    plt.title(f'Hole tendon lever arm vs fold angle (d={d}, h={h})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
