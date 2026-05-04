# src/synergy/augmented_adaptive.py
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
        
        # 使用基础模型的核心来预计算 S_aug 和 C_aug
        self._base = AdaptiveSynergyModel(n_joints, self.R_aug, E_vec)
        self.S_aug = self._base.S                 # shape (n, k+m)
        self.C_aug = self._base.C                 # shape (n, n)
        
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