# src/synergy/sdas_model.py
"""
State-Dependent Adaptive Synergy (SDAS) 模型

基于 docs/SDAS.md 第 2.3 节的理论推导和 Schur 补正确求解。

核心思想：
  双端驱动的闭环腱绳系统中，两端的张力经 Capstan 摩擦沿路径衰减，
  分别对应两个不同的传动向量 R_A 和 R_B。两个驱动器独立控制各自的
  协同模式，关节构型为两者的线性叠加。

约束类型（关键！）：
  绳子绷紧条件：总绳长缩短量 θ_A + θ_B > 0（或等价地 σ_c > 0）。
  只要绳子绷紧，两端约束 R_A·q = θ_A 和 R_B·q = θ_B 同时生效，
  使用 Schur 补公式（含 R_A-R_B 交叉耦合）。

  当 σ_c ≤ 0（无绳长缩短）时，绳子松弛，无主动约束，q = 0。

  注意：单马达拉动（θ_A > 0, θ_B = 0）时总绳长仍缩短，
  两端依然绷紧，因此同样使用 Schur 补公式。不存在"单端松弛"情形。

参考文献：
  docs/SDAS.md 第 2.3 节，公式 (2.22)-(2.49)
"""

import numpy as np
from typing import Optional, Tuple


class SDASModel:
    """
    状态依赖自适应协同 (SDAS) 求解器。

    使用精确约束选择策略：
      - 总绳长缩短 σ_c > 0 → 两端绷紧 → Schur 补公式
      - 总绳长 σ_c ≤ 0 → 松弛 → 无主动约束
      - 严格满足 R_A·q = θ_A 和 R_B·q = θ_B（双约束同时生效）
      - 无静摩擦冻结参数、无 Rf 概念
      - 无单向公式（纯拉力驱动下不存在"单端松弛"）

    参数：
        n_joints : int
            关节数量
        R_A : np.ndarray, shape (1, n)
            从 Motor A 出发的传动向量（Capstan 衰减加权）
        R_B : np.ndarray, shape (1, n)
            从 Motor B 出发的传动向量（Capstan 衰减加权）
        E_vec : np.ndarray, shape (n,)
            关节刚度对角元

    用法：
        model = SDASModel(n_joints, R_A, R_B, E_vec)
        q = model.solve_motors(theta_A, theta_B)      # 直接电机输入
        q = model.solve_synergies(sigma_c, sigma_d)    # σ_c/σ_d 输入
    """

    def __init__(self, n_joints: int,
                 R_A: np.ndarray, R_B: np.ndarray,
                 E_vec: np.ndarray):
        self.n_joints = n_joints
        assert R_A.shape == (1, n_joints), f"R_A expected (1,{n_joints}), got {R_A.shape}"
        assert R_B.shape == (1, n_joints), f"R_B expected (1,{n_joints}), got {R_B.shape}"

        self.R_A = R_A.copy()
        self.R_B = R_B.copy()
        self.E_vec = E_vec.copy()
        self.E = np.diag(E_vec)
        self.E_inv = np.diag(1.0 / E_vec)

        # Schur 补的标量系数
        #   系统 (2.38): 见下详细推导
        self.a = float(self.R_A @ self.E_inv @ self.R_A.T)
        self.b = float(self.R_A @ self.E_inv @ self.R_B.T)
        self.c = float(self.R_B @ self.E_inv @ self.R_B.T)
        self.det = self.a * self.c - self.b * self.b

        if abs(self.det) < 1e-15:
            raise ValueError(
                f"Schur 补行列式为零 (a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f})。\n"
                f"R_A 和 R_B 线性相关（对称路径且 E 均匀），系统无法唯一求解。\n"
                f"考虑使用较小的 β 值增加 R_A, R_B 的非对称性。"
            )

        # ============================================================
        # S_A_schur, S_B_schur (Schur 补公式)：同时满足两个约束
        #
        # 系统 (2.38):
        #   [ -E    R_A^T  R_B^T ] [ q   ]   [ J^T f_c ]
        #   [ R_A   0      0     ] [ F_A ] = [ θ_A     ]
        #   [ R_B   0      0     ] [ F_B ]   [ θ_B     ]
        #
        # 从第 1 行解出 q = E^{-1}(R_A^T F_A + R_B^T F_B - J^T f_c)
        # 代入第 2、3 行得到 2×2 系统：
        #   [ a  b ] [ F_A ] = [ θ_A + R_A E^{-1} J^T f_c ]
        #   [ b  c ] [ F_B ]   [ θ_B + R_B E^{-1} J^T f_c ]
        #
        # Schur 补逆公式：
        #   [a b]^{-1} = 1/det [ c  -b ]
        #   [b c]               [-b   a ]
        #
        # 代入 q 的表达式，提取 θ_A, θ_B 系数得到：
        #   S_A_schur = E^{-1} · (R_A^T · c - R_B^T · b) / det
        #   S_B_schur = E^{-1} · (-R_A^T · b + R_B^T · a) / det
        #
        # 性质：
        #   R_A·S_A_schur = 1,  R_B·S_A_schur = 0
        #   R_A·S_B_schur = 0,  R_B·S_B_schur = 1
        #
        # 适用：所有绷紧情形（两电机任意组合，总绳长缩短）
        # ============================================================
        self.S_A_schur = (self.E_inv @ (self.R_A.T * self.c - self.R_B.T * self.b)
                          / self.det)   # (n, 1)
        self.S_B_schur = (self.E_inv @ (-self.R_A.T * self.b + self.R_B.T * self.a)
                          / self.det)   # (n, 1)

        # ============================================================
        # σ_c / σ_d 模式对应的协同方向（Schur 公式）
        #   θ_A = σ_c + σ_d,  θ_B = σ_c - σ_d
        #   q = S_A_schur·θ_A + S_B_schur·θ_B
        #     = (S_A_schur + S_B_schur)·σ_c + (S_A_schur - S_B_schur)·σ_d
        # ============================================================
        self.S_sigma_schur = self.S_A_schur + self.S_B_schur  # (n, 1)
        self.S_diff_schur = self.S_A_schur - self.S_B_schur   # (n, 1)

    def _solve(self, theta_A: float, theta_B: float,
               J=None, f_ext=None) -> np.ndarray:
        """Schur 补公式求解（双约束同时满足）"""
        q = (self.S_A_schur * theta_A + self.S_B_schur * theta_B).flatten()
        if J is not None and f_ext is not None:
            q += self._compute_external_compliance(J, f_ext)
        return q

    def _compute_external_compliance(self, J, f_ext) -> np.ndarray:
        """
        计算外力项 C·J^T·f_c。

        Schur 被动柔顺矩阵 (Eq. 2.41)
        C = E^{-1} - S_A_schur·R_A·E^{-1} - S_B_schur·R_B·E^{-1}
        """
        C = (self.E_inv
             - self.S_A_schur @ self.R_A @ self.E_inv
             - self.S_B_schur @ self.R_B @ self.E_inv)
        return C @ J.T @ f_ext

    def solve_motors(self,
                     theta_A: float, theta_B: float,
                     J: Optional[np.ndarray] = None,
                     f_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """
        给定两个电机的位移输入，计算关节角 q。

        约束选择策略：
          - θ_A + θ_B > ε（总绳长缩短 > 0）→ 两端绷紧 → Schur 补公式
          - θ_A + θ_B ≤ ε（总绳长不缩短）→ 松弛 → 无主动约束

        注意：不对 θ_A 或 θ_B 单独钳位。只要总绳长缩短，
        绳子就是紧的，两端约束同时生效。Schur 补公式本身
        能处理 θ_A, θ_B 正负任意组合（例如 A 拉 B 放的情况）。

        参数：
            theta_A : Motor A 位移输入（弧度）
            theta_B : Motor B 位移输入（弧度）
            J       : 雅可比矩阵 [可选]
            f_ext   : 接触力 [可选]

        返回：
            q : (n,) 关节角度向量
        """
        EPS = 1e-10

        # 总绳长缩短 > 0 → 绷紧，两端约束同时生效
        if theta_A + theta_B > EPS:
            return self._solve(theta_A, theta_B, J, f_ext)
        else:
            q = np.zeros(self.n_joints)
            if J is not None and f_ext is not None:
                q += self._compute_external_compliance(J, f_ext)
            return q

    def solve_synergies(self,
                        sigma_c: float, sigma_d: float,
                        J: Optional[np.ndarray] = None,
                        f_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """
        给定 σ_c / σ_d 输入，计算关节角 q。

        θ_A = σ_c + σ_d
        θ_B = σ_c - σ_d

        约束选择策略：
          σ_c > 0 → 总绳长缩短 2σ_c > 0 → 两端绷紧 → Schur 补公式
          σ_c ≤ 0 → 总绳长不缩短 → 松弛 → 无主动运动

        注意：不对 θ_A 或 θ_B 单独钳位。σ_c > 0 时绳子一定绷紧，
        两端约束 R_A·q = θ_A 和 R_B·q = θ_B 同时生效。
        即使某个 θ 为负（该端释放绳长），另一端拉得更多，
        Schur 补公式天然处理正负任意组合。

        参数：
            sigma_c : 共模输入（两马达同向成分，决定总绳长）
            sigma_d : 差模输入（两马达差动成分）
            J       : 雅可比矩阵 [可选]
            f_ext   : 接触力 [可选]

        返回：
            q : (n,) 关节角度向量
        """
        sigma_c = max(0.0, float(sigma_c))  # σ_c ≥ 0

        if sigma_c <= 1e-10:
            # σ_c = 0 → 绳子最长，无主动约束
            q = np.zeros(self.n_joints)
            if J is not None and f_ext is not None:
                q += self._compute_external_compliance(J, f_ext)
            return q

        theta_A = sigma_c + sigma_d
        theta_B = sigma_c - sigma_d

        return self._solve(theta_A, theta_B, J, f_ext)

    def get_synergy_directions(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 σ_c 和 σ_d 对应的协同方向向量"""
        return self.S_sigma_schur.flatten(), self.S_diff_schur.flatten()
