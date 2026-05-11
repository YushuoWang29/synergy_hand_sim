# src/synergy/augmented_adaptive.py
"""
Augmented Adaptive Synergy 模型 (Della Santina et al. TRO 2018)

在基础 adaptive synergy (σ) 之上，添加肌腱滑动摩擦产生的第二个
协同输入 σ_f (或 u_2)。

物理核心：
  R:   H 模式（双电机同向）的传动矩阵，基于 Capstan 指数衰减加权
       → 中间关节有效传动比小于两端，产生 U-shape
  Rf:  H_f 模式（双电机异向）的传动矩阵，基于张力差加权 + slack 侧钳制
       → 对称路径 Rf=0（需静摩擦模型才能解耦）
       → 非对称路径产生手指重配模式

参考文献:
  Della Santina et al. TRO 2018, Section III-V
"""
import numpy as np
from .base_adaptive import AdaptiveSynergyModel


class AugmentedAdaptiveSynergyModel:
    """
    Augmented Adaptive Synergy 模型 (SoftHand 2 论文, 2018)

    在基础 adaptive synergy (σ) 之上，添加肌腱滑动摩擦产生的第二个
    协同输入 σ_f (或 u_2)。本质是将 R 和 R_f 堆叠成 R_aug，然后复用
    与基础模型相同的数学结构。

    参数:
        n_joints : 关节数
        R        : 基础传动矩阵 (k, n)
        R_f      : 摩擦滑动传动矩阵 (m, n)，通常 m=1
        E_vec    : 关节刚度向量 (n,)
    """

    def __init__(self, n_joints: int, R: np.ndarray, R_f: np.ndarray,
                 E_vec: np.ndarray):
        self.n_joints = n_joints
        self.k = R.shape[0]
        self.m = R_f.shape[0]

        # 堆叠成扩展传动矩阵
        self.R_aug = np.vstack([R, R_f])          # shape (k+m, n)
        self.E_vec = E_vec

        # 检测 R_aug 是否满秩：R 和 R_f 可能共线（所有滑轮半径/摩擦系数相同）
        # 或 R_f 为零（对称路径）
        aug_rank = np.linalg.matrix_rank(self.R_aug, tol=1e-10)
        if aug_rank < self.k + self.m:
            # R 和 R_f 共线或 R_f 为零 → sigma_f 方向退化
            # 物理理解：
            # - 对称路径（如单手指）：R_f=0，σ_f 无独立影响 → 安全降级
            # - 对称双电机布置：R ∝ R_f（比例系数相同）→ 需重新设计参数
            self._base = AdaptiveSynergyModel(n_joints, R, E_vec)
            S_base = self._base.S                  # shape (n, k)
            zero_pad = np.zeros((n_joints, self.m)) # shape (n, m)
            self.S_aug = np.hstack([S_base, zero_pad])
            self.C_aug = self._base.C
            self._rank_deficient = True
            print(f"  [INFO] Rf rank-deficient (rank={aug_rank}, "
                  f"need {self.k+self.m}). σ_f direction disabled.")
            print(f"    This is expected for symmetric paths (e.g., single finger).")
        else:
            self._base = AdaptiveSynergyModel(n_joints, self.R_aug, E_vec)
            self.S_aug = self._base.S              # shape (n, k+m)
            self.C_aug = self._base.C              # shape (n, n)
            self._rank_deficient = False

    def solve(self, sigma: np.ndarray, sigma_f: np.ndarray,
              J: np.ndarray = None, f_ext: np.ndarray = None) -> np.ndarray:
        """
        计算关节角 q

        参数:
            sigma   : 基础协同输入 (k,)
            sigma_f : 摩擦滑动协同输入 (m,)
            J, f_ext: 可选，外力相关
        返回:
            q : 关节角 (n,)
        """
        # 拼接协同输入向量
        aug_input = np.concatenate([sigma, sigma_f])
        q = self.S_aug @ aug_input
        if J is not None and f_ext is not None:
            q += self.C_aug @ J.T @ f_ext
        return q

    @property
    def R(self) -> np.ndarray:
        return self.R_aug[:self.k]

    @property
    def R_f(self) -> np.ndarray:
        return self.R_aug[self.k:]
