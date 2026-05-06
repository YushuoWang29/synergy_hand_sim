# src/synergy/base_adaptive.py

import numpy as np
from typing import Optional, Tuple

class AdaptiveSynergyModel:
    """
    基础自适应协同求解器（Grioli et al. 2012）
    
    输入：
        n_joints : 关节数
        R        : 传动矩阵 (k x n)，实现 R q = sigma
        E_vec    : 对角关节刚度 (n,) 数组
    
    用法：
        model = AdaptiveSynergyModel(n_joints, R, E_vec)
        q = model.solve(sigma)                     # 无外力
        q = model.solve(sigma, J, f_ext)           # 有外力
    """
    
    def __init__(self, n_joints: int, R: np.ndarray, E_vec: np.ndarray):
        self.n_joints = n_joints
        self.k = R.shape[0]
        self.R = R
        self.E = np.diag(E_vec)  # n x n 对角阵
        
        # 预计算常用矩阵
        self._precompute()
    
    def _precompute(self):
        """预计算协同矩阵 S 和柔顺矩阵 C
        
        使用伪逆 (pinv) 而非普通逆 (inv) 以处理 R 行之间可能共线的情况。
        例如当 R 和 R_f 完全成比例时 (所有滑轮半径/摩擦相同)，
        RE = R E^{-1} R^T 可能秩亏缺，用 inv 会产生数值垃圾。
        pinv 自动将零空间投影为零，保持非零空间正确。
        """
        E_inv = np.diag(1.0 / np.diag(self.E))
        RE = self.R @ E_inv @ self.R.T
        RE_inv = np.linalg.pinv(RE)
        
        # 主动协同矩阵 S^(k) = E^{-1} R^T (R E^{-1} R^T)^{-1}
        self.S = E_inv @ self.R.T @ RE_inv
        
        # 被动柔顺矩阵 C = E^{-1} - S R E^{-1}
        self.C = E_inv - self.S @ self.R @ E_inv
    
    def solve(self, 
              sigma: np.ndarray,
              J: Optional[np.ndarray] = None,
              f_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """
        求解关节角 q
        
        参数:
            sigma : 协同输入向量 (k,)
            J     : 抓取雅可比矩阵 (m x n)，接触点对关节的映射
            f_ext : 外力旋量 (m,)
        
        返回:
            q : 关节角 (n,)
        """
        # 主动部分：沿协同方向运动
        q = self.S @ sigma
        
        # 被动部分：外力通过柔顺矩阵产生偏移
        if J is not None and f_ext is not None:
            q += self.C @ J.T @ f_ext
        
        return q
    
    @staticmethod
    def compute_R_from_synergy(
            S: np.ndarray, 
            E_vec: np.ndarray, 
            tol: float = 1e-10) -> np.ndarray:
        """
        从期望的协同基矩阵 S (n x k) 反向计算传动矩阵 R (k x n)
        
        基于公式 (13): S = E^{-1} R^T (R E^{-1} R^T)^{-1}
        当 E 对角且假设 R 行正交时，有 R = S^T E
        更通用的方法使用伪逆。
        """
        E = np.diag(E_vec)
        E_inv = np.diag(1.0 / E_vec)
        
        # 计算 M = S^T E S，希望 M 可逆
        M = S.T @ E @ S
        if np.linalg.cond(M) > 1/tol:
            print("Warning: near-singular M, result may be inaccurate")
        
        # R = (M^{-1} S^T E)  满足 R q = sigma ? 需要验证
        # 但更简单：期望 R = S^T E (当 R 行正交且 E=αI 时成立)
        # 在一般情况下，我们通过解 S = E^{-1} R^T (R E^{-1} R^T)^{-1} 得到 R
        # 这需要求解代数 Riccati 方程，这里采用简单的近似：
        R = S.T @ E  # 一种常见近似，适用于 R 近乎正交的情况
        
        # 验证误差
        S_test = AdaptiveSynergyModel(len(E_vec), R, E_vec).S
        err = np.linalg.norm(S - S_test)
        if err > tol:
            print(f"Warning: synergy reconstruction error = {err:.2e}")
        
        return R