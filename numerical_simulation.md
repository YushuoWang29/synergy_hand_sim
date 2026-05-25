# Numerical Simulation Framework for Synergy-Based Tendon-Driven Hands
## 1. 现状分析：当前仿真为何是纯几何的
当前项目 (`synergy_hand_sim`) 的所有仿真均基于**纯几何/运动学**模型，可在以下几个方面确认：
| 方面 | 当前实现 | 状态 |
|------|----------|------|
| 正向运动学 | `OrigamiForwardKinematics` — 仅从关节角 q 计算面片 3D 位姿 | ✅ 几何 |
| 协同模型 | `AdaptiveSynergyModel.solve(σ)` — 求解 q = Sσ + CJᵀf_ext，但 f_ext 始终为 0 | ✅ 准静态 |
| 增强协同 | `AugmentedAdaptiveSynergyModel.solve(σ, σ_f)` — 同上，加 R_f 方向 | ✅ 准静态 |
| 动态协同 | `DynamicSynergyModel.solve(σ, α)` — 阻尼平衡法，仍为代数方程 | ✅ 准静态 |
| MuJoCo 仿真 | `MuJoCoSimulator` — 直接设置 qpos，无物理积分 | ✅ 运动学 |
| 质量/惯性 | 未在任何地方定义或使用 | ❌ 缺失 |
| 关节阻尼/摩擦 | 仅用于传动矩阵 R_f 计算，未用于动力学 | ❌ 缺失 |
| 接触/碰撞 | 碰撞检测已清理（暂不需要） | ✅ 已移除 |
| 力矩/力平衡 | 只有静力平衡方程，无加速度项 | ❌ 缺失 |
| 时间积分 | 无，每帧直接设置 qpos | ❌ 缺失 |
**结论**：当前仿真实际上是**代数映射**（σ/σ_f/α → q），没有时间演化、没有惯性效应、没有接触力反馈。这就是为什么用户直观感受是"纯几何的"。
---
## 2. 目标：力学建模仿真的总体架构
### 2.1 要复现的论文模型
根据 Della Santina et al. (2018) **Section V-B, Eq.43-44**，完整的动力学模型为：
```
B(q) q̈ + W(q, q̇) q̇ + Γ(q) = Q(q) u + J(q)ᵀ f_ext
```
其中（Eq.44）：
```
W(q, q̇) = C(q, q̇) + F + Rᵀ M⁻¹ Λ
Γ(q) = G(q) − Rᵀ M⁻¹ Σ(N M⁻¹ R q − z)
Q(q) u = Rᵀ M⁻¹ [½ M e_v, Σ N e_v, −Λ e_v] · [τ_M, s, ṡ]ᵀ
```
**输入向量** u = [τ_M, s, ṡ]ᵀ：
- τ_M：电机总拉力（两电机同向，对应 σ 控制）
- s：腱绳滑动量（两电机差动，对应 σ_f 控制）
- ṡ：腱绳滑动速度
**静摩擦状态** z ∈ ℝ¹¹⁷：每个滑轮有一个虚拟角度，描述静摩擦记忆效应（Eq.37）：
```
z⁺_j = {
θ_j + Δ_max_j if θ_j ≤ z_j − Δ_max_j
θ_j − Δ_max_j if θ_j ≥ z_j + Δ_max_j
z_j otherwise
}
```
其中 θ = N(M⁻¹R q − e_v s)（Eq.41）。
### 2.2 分层架构概览
数值仿真框架分为三个递进层次：
```
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: 完整动力学 + 接触力学 │
│ B(q)q̈ + W q̇ + Γ = τ_input + J(q)ᵀ f_ext + J_c(q)ᵀ f_c │
│ ODE积分 + 接触约束求解 + 摩擦锥 │
├─────────────────────────────────────────────────────────────┤
│ Phase 2: 完整动力学（无接触） │
│ B(q)q̈ + W q̇ + Γ = Q(q) u + J(q)ᵀ f_ext │
│ scipy.integrate.ODE / RK4 时域积分 │
├─────────────────────────────────────────────────────────────┤
│ Phase 1: 准静态力平衡 │
│ Γ(q) = Q(q) u + J(q)ᵀ f_ext │
│ 非线性代数方程求解 (scipy.optimize.root) │
├─────────────────────────────────────────────────────────────┤
│ 底层基础设施 │
│ OrigamiHandDesign → 运动学 + 质量/惯性/阻尼参数 + 传动矩阵 │
└─────────────────────────────────────────────────────────────┘
```
每一层向后兼容：Phase 2 可以退化为 Phase 1（关闭加速度项），Phase 3 在无接触时退化为 Phase 2。
---
## 3. 新增文件结构
所有数值仿真相关代码放在 `src/simulation/` 包下：
```
src/simulation/
├── __init__.py # 包初始化，导出主要类
├── config.py # 仿真配置数据类
├── rigid_body.py # 刚体参数数据结构
├── friction_models.py # 摩擦模型实现（粘滞+静摩擦Hayward-Armstrong）
├── transmission_force.py # 腱绳传动力计算（Q(q)矩阵）
├── dynamics.py # 动力学方程组装 (B, W, Γ 计算)
├── integrator.py # 数值积分器 (ODE求解)
├── quasi_static.py # 准静态求解器 (非线性力平衡)
├── contact_model.py # 接触模型 (法向+切向)
├── simulator.py # 顶层仿真引擎
├── io.py # 仿真结果序列化/logging
└── tests/
├── __init__.py
├── test_friction_models.py # 摩擦模型单元测试
├── test_rigid_body.py # 刚体参数测试
├── test_dynamics.py # 动力学方程测试
├── test_quasi_static.py # 准静态求解器测试
├── test_integrator.py # 积分器测试
├── test_contact_model.py # 接触模型测试
└── test_simulator.py # 集成测试
```
### 与现有框架的接口
```
┌─────────────────────┐ ┌──────────────────────────┐
│ 现有 OrigamiHandDesign │ ──▶ │ src/simulation/ 模块 │
│ + fold_lines │ │ │
│ + faces │ │ 读取: fold_line geometry │
│ + joints │ │ 读取: pulley/hole/damper │
│ + pulleys/holes │ │ 读取: tendon sequence │
│ + tendons/dampers │ │ 写入: 扩展参数 (质量等) │
└─────────────────────┘ └──────────────────────────┘
│ │
▼ ▼
┌─────────────────────┐ ┌──────────────────────────┐
│ transmission_builder │ │ simulation/dynamics.py │
│ compute_R() │ ──▶ │ 组装 B(q), W(q,q̇), Γ(q) │
│ compute_Rf() │ │ │
└─────────────────────┘ └──────────────────────────┘
│ │
▼ ▼
┌─────────────────────┐ ┌──────────────────────────┐
│ synergistic model │ │ simulation/simulator.py │
│ (当前回调接口) │ ◀── │ 输出 q(t) → synergy │
│ │ │ 回调 → MuJoCo viewer │
└─────────────────────┘ └──────────────────────────┘
```
---
## 4. 数据结构和参数定义
### 4.1 仿真配置 (`config.py`)
```python
# src/simulation/config.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
@dataclass
class SimulationConfig:
"""完整仿真器配置，包含所有物理参数和控制设置。"""
# ---- 时间积分 ----
dt: float = 1e-4 # 积分步长 (s)
t_end: float = 5.0 # 仿真总时长 (s)
method: str = 'RK4' # 积分方法: 'RK4', 'Euler', 'scipy_ode'
# ---- 控制输入（外部的 σ, σ_f, s） ----
sigma_func: Optional[str] = None # σ(t) 表达式，如 "min(t*2, 1.0)"
sigma_f_func: Optional[str] = None
tau_M_func: Optional[str] = None # τ_M(t) 电机拉力
sliding_s_func: Optional[str] = None # s(t) 滑动量
# ---- Capstan 摩擦参数（作用于腱绳-滑轮界面） ----
beta_capstan: float = 0.09 # Capstan 衰减系数 (每元素)
beta_hole: float = 0.09 # 孔类腱绳的摩擦衰减
# ---- 静摩擦 (Hayward-Armstrong) 参数 ----
use_static_friction: bool = True # 是否启用静摩擦记忆效应
delta_max_ratio: float = 1e-4 # Δ_max / r (静摩擦范围比例)
kappa_ratio: float = 0.3 # κ (静摩擦刚度) = kappa_ratio * k_joint
# ---- 关节粘滞摩擦 ----
joint_viscous_damping: float = 0.01 # F = diag(joint_viscous_damping) (N·m·s/rad)
tendon_viscous_damping: float = 0.001 # Λ = diag(c_i), c_i = r_i² * tendon_viscous_damping
# ---- 弹性参数 ----
use_nonlinear_spring: bool = False # 是否使用非线性弹簧（见 Appendix B）
# 线性弹簧参数：直接从 fold_lines[i].stiffness 读取
# ---- 接触参数 ----
use_contact: bool = False # Phase 3: 启用接触力学
contact_stiffness: float = 1e5 # 法向接触刚度 (N/m)
contact_damping: float = 1e2 # 法向接触阻尼 (N·s/m)
friction_coef: float = 0.5 # 切向摩擦系数
# ---- 数值控制 ----
quasi_static_tol: float = 1e-8 # 准静态求解器收敛容差
quasi_static_max_iter: int = 50 # 准静态最大迭代次数
verbose: int = 1 # 0=silent, 1=summary, 2=detailed
# ---- 结果输出 ----
record_every: int = 10 # 每 N 步记录一次结果
output_path: Optional[str] = None # 结果保存路径 (.npz)
# ---- 重力 ----
gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
# ---- 外部力 ----
external_forces: dict = field(default_factory=dict)
# { fingertip_name: (force_vector, application_point) }
# 如: {"thumb_tip": (np.array([0, 0, -2]), np.array([x, y, z]))}
```
### 4.2 刚体参数 (`rigid_body.py`)
```python
# src/simulation/rigid_body.py
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
@dataclass
class LinkInertia:
"""
单个连杆（面片）的惯性参数。
这些参数需要从 CAD/STL 估算或手动指定。
origami_parser.py 中缺少厚度和材料密度信息，
因此厚度参数来自 export_urdf.py 的 --thickness 参数。
"""
name: str # 面片名称, 如 "face_0"
face_id: int # 对应的 OrigamiFace.id
mass: float = 0.005 # 质量 (kg), 默认约 5g
com: np.ndarray = None # 质心在 link 坐标系中位置 (3,)
inertia: np.ndarray = None # 惯性张量 (3,3) 相对质心
thickness: float = 3.0e-3 # 厚度 (m)
material_density: float = 1200.0 # 材料密度 (kg/m³), 类似 PLA/ABS
def __post_init__(self):
if self.com is None:
# 默认质心在 link 坐标系原点（面片中心在 z=厚度/2）
self.com = np.array([0.0, 0.0, self.thickness / 2])
if self.inertia is None:
# 用薄板近似计算惯性张量
# 假设面片为边长 L 的方形薄板，I = m/12 * L²
# 这里需要面片面积来精确计算，留待 build_inertia 中处理
self.inertia = np.eye(3) * 1e-7 # 临时占位
def estimate_link_inertia(area: float, thickness: float,
density: float, face_vertices_2d: np.ndarray) -> LinkInertia:
"""
根据面片面积和厚度估算惯性参数。
使用多边形薄板的精确惯性张量公式：
I_xx = ρ·t · ∫∫ y² dx dy (在面片多边形上积分)
I_yy = ρ·t · ∫∫ x² dx dy
I_zz = I_xx + I_yy (薄板近似)
I_xy = −ρ·t · ∫∫ xy dx dy
I_xz = I_yz = 0 (在质心坐标系中)
多边形上的二阶矩可以通过 shoelace 公式的推广计算。
"""
mass = area * thickness * density
# 将多边形平移到质心位置
vertices = face_vertices_2d
cx = np.mean(vertices[:, 0])
cy = np.mean(vertices[:, 1])
v_centered = vertices - np.array([cx, cy])
# 计算多边形二阶矩 (精确积分公式)
I_xx = 0.0
I_yy = 0.0
I_xy = 0.0
n = len(v_centered)
for i in range(n):
x1, y1 = v_centered[i]
x2, y2 = v_centered[(i + 1) % n]
cross = x1 * y2 - x2 * y1
I_xx += cross * (y1**2 + y1 * y2 + y2**2)
I_yy += cross * (x1**2 + x1 * x2 + x2**2)
I_xy += cross * (2 * x1 * y1 + x1 * y2 + x2 * y1 + 2 * x2 * y2)
I_xx = abs(I_xx) / 12
I_yy = abs(I_yy) / 12
I_xy = abs(I_xy) / 24
# 加入厚度方向贡献
I_zz = I_xx + I_yy
# 乘以密度和厚度
scale = density * thickness
I_xx_com = scale * I_xx
I_yy_com = scale * I_yy
I_xy_com = scale * I_xy
inertia_tensor = np.array([
[I_xx_com, -I_xy_com, 0],
[-I_xy_com, I_yy_com, 0],
[0, 0, scale * I_zz]
])
return LinkInertia(
name=f"face_{id}",
face_id=id,
mass=mass,
com=np.array([cx, cy, thickness / 2]),
inertia=inertia_tensor,
thickness=thickness,
material_density=density
)
@dataclass
class RigidBodySystem:
"""
多刚体系统的完整参数集合。
包含所有连杆的惯性、关节阻尼、以及运动学关系。
可从 OrigamiHandDesign + URDF 构建。
"""
link_inertias: Dict[int, LinkInertia] = field(default_factory=dict)
n_joints: int = 0
joint_damping: np.ndarray = None # F: (n_joints,) 关节粘滞阻尼系数
joint_coulomb_friction: np.ndarray = None # 关节库伦摩擦力矩 (n_joints,)
# 运动学关系（从 OrigamiForwardKinematics 预计算）
parent_child_map: Dict[int, int] = field(default_factory=dict) # child -> parent
@classmethod
def from_design(cls, design, thickness=3.0e-3, density=1200.0) -> 'RigidBodySystem':
"""
从 OrigamiHandDesign 构建刚体系统。
需要 design 已构建拓扑（build_topology 已完成）。
需要已知 joint 列表。
"""
# 实现见下文
pass
```
### 4.3 腱绳传动力模块 (`transmission_force.py`)
```python
# src/simulation/transmission_force.py
"""
计算 Q(q) u 项：腱绳传动映射到关节力矩。
基于 Della Santina et al. (2018) Eq.44 中的 Q(q) 矩阵：
Q(q) = Rᵀ M⁻¹ [½ M e_v, Σ N e_v, −Λ e_v]
输入 u = [τ_M, s, ṡ]ᵀ：
第一列 (½ M e_v)：对应电机总拉力 τ_M → τ_joint = Rᵀ · (½ · e_v) · τ_M
物理意义：τ_M 平均分配到路径两端
第二列 (Σ N e_v)：对应滑动位移 s → τ_joint = Rᵀ M⁻¹ Σ N e_v · s
物理意义：静摩擦记忆效应产生的力矩
第三列 (−Λ e_v)：对应滑动速度 ṡ → τ_joint = −Rᵀ M⁻¹ Λ e_v · ṡ
物理意义：粘滞摩擦引起的力矩
依赖：
- transmission_builder.py 中的 get_joint_list
- origami_design.py 中的 pulley/hole 参数
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
def build_M_matrix(elements: List[int]) -> np.ndarray:
"""
构建式(13)中的 M 矩阵。
论文 Eq.(13) 中的 M ∈ ℝ^{(m+1)×(m+1)}：
M[i,i] = -1 (i=0..m-1)
M[i,i+1] = +1 (i=0..m-1)
M[m,0] = +1
M[m,m] = +1
物理意义：腱绳段的张力平衡方程 MT + V(v) + eτ_M = 0
Parameters
----------
elements : list[int]
腱绳路径上的元素 ID 列表（滑轮 ID 或孔 ID）。
m = len(elements)
Returns
-------
M : ndarray, shape (m+1, m+1)
"""
m = len(elements)
M = np.zeros((m + 1, m + 1))
for i in range(m):
M[i, i] = -1.0
M[i, i + 1] = 1.0
M[m, 0] = 1.0
M[m, m] = 1.0
return M
def build_R_bar_matrix(elements: List[int],
design,
jid_to_idx: Dict[int, int]) -> np.ndarray:
"""
构建式(13)中的 R̄ 矩阵。
R̄ ∈ ℝ^{n × (m+1)}，其中 R̄[j, k] = r_k 如果第 k 段在关节 j 上，否则为 0。
与 transmission_builder.compute_R() 不同之处：
- R̄ 是 "每段" 的贡献，未经过合并
- compute_R() 输出的是 R = −R̄ᵀ e_v（合并后的 n_joints 向量）
注意：R̄ 的行索引对应关节，列索引对应腱绳段。
而论文中的 R̄ ∈ ℝ^{(m+1) × n} 是列对应关节，行对应段。
这里遵循论文式(13): τ = −R̄ᵀ T，所以 R̄ 形状为 (m+1, n)。
"""
from src.models.origami_design import is_pulley_id, is_hole_id
n_joints = len(jid_to_idx)
m = len(elements)
R_bar = np.zeros((m + 1, n_joints)) # (m+1, n)
for seg_idx, eid in enumerate(elements):
j_idx = None
r_val = 0.0
if is_pulley_id(eid) and eid in design.pulleys:
pulley = design.pulleys[eid]
if pulley.attached_fold_line_id is not None:
j_idx = jid_to_idx.get(pulley.attached_fold_line_id)
r_val = pulley.radius
elif is_hole_id(eid) and eid in design.holes:
hole = design.holes[eid]
if hole.attached_fold_line_id is not None:
j_idx = jid_to_idx.get(hole.attached_fold_line_id)
r_val = hole.plate_offset # arm(0) = h, 线性化近似
if j_idx is not None and r_val > 0:
R_bar[seg_idx, j_idx] = r_val
return R_bar # shape (m+1, n)
def build_viscous_damping_matrix(elements: List[int],
design) -> np.ndarray:
"""
构建粘滞摩擦对角矩阵 Λ ∈ ℝ^{(m+1) × (m+1)}。
论文 Eq.38-39: Λ = diag(c_j / r_j²)
其中 c_j 是第 j 个滑轮的粘滞摩擦系数，r_j 是半径。
对于孔元素，使用等效粘滞摩擦 c_hole = r_hole * mu_hole。
论文 Section V-A 指出 Λ 的第 (m+1) 行和一列为 0，
对应边界条件 τ_M = T_0 + T_m。
"""
from src.models.origami_design import is_pulley_id, is_hole_id
m = len(elements)
Lambda = np.zeros((m + 1, m + 1))
for seg_idx, eid in enumerate(elements):
c_i = 0.0
r_i = 1.0 # 默认半径
if is_pulley_id(eid) and eid in design.pulleys:
pulley = design.pulleys[eid]
r_i = pulley.radius
# c_i = r² * tendon_viscous_damping (单位 N·s/m)
c_i = r_i**2 * 0.001 # 默认值，可通过 config 覆盖
elif is_hole_id(eid) and eid in design.holes:
hole = design.holes[eid]
r_i = hole.plate_offset
c_i = r_i * hole.friction_coefficient * 0.01
if c_i > 0 and r_i > 0:
Lambda[seg_idx, seg_idx] = c_i / r_i**2
return Lambda
def build_static_friction_matrix(elements: List[int],
design) -> np.ndarray:
"""
构建静摩擦矩阵 Σ ∈ ℝ^{(m+1) × (m+1)}。
论文 Eq.38-39: Σ = diag(κ_j / r_j²)
其中 κ_j 是第 j 个滑轮的静摩擦刚度。
Hayward-Armstrong 模型 (Eq.37-38):
摩擦力 = (θ_j − z_j) · κ_j / r_j²
θ_j = r_j 的旋转角度
z_j = 虚拟角度 (静摩擦记忆状态)
"""
from src.models.origami_design import is_pulley_id, is_hole_id
m = len(elements)
Sigma = np.zeros((m + 1, m + 1))
for seg_idx, eid in enumerate(elements):
kappa_i = 0.0
r_i = 1.0
if is_pulley_id(eid) and eid in design.pulleys:
pulley = design.pulleys[eid]
r_i = pulley.radius
# κ_j = kappa_ratio * fold_stiffness (默认 kappa_ratio=0.3)
kappa_i = 0.3 * 1.0 # 默认 stiffness=1.0
elif is_hole_id(eid) and eid in design.holes:
hole = design.holes[eid]
r_i = hole.plate_offset
kappa_i = 0.3 * hole.friction_coefficient * 0.1
if kappa_i > 0 and r_i > 0:
Sigma[seg_idx, seg_idx] = kappa_i / r_i**2
return Sigma
def build_N_matrix(elements: List[int],
design) -> np.ndarray:
"""
构建 N 矩阵（角度转换）。
论文 Eq.41: θ = N(M⁻¹ R̄ q − e_v s)
N = diag(1/r_i) ∈ ℝ^{(m+1) × (m+1)}
物理意义：将腱绳段位移转换为滑轮旋转角度。
"""
from src.models.origami_design import is_pulley_id, is_hole_id
m = len(elements)
N = np.zeros((m + 1, m + 1))
for seg_idx, eid in enumerate(elements):
r_i = 1.0
if is_pulley_id(eid) and eid in design.pulleys:
r_i = design.pulleys[eid].radius
elif is_hole_id(eid) and eid in design.holes:
r_i = design.holes[eid].plate_offset
if r_i > 0:
N[seg_idx, seg_idx] = 1.0 / r_i
return N
def compute_Q_matrix(design) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
"""
计算论文 Eq.44 中的 Q(q) 矩阵的三个分量。
对于每条腱绳：
Q = Rᵀ M⁻¹ [½ M e_v | Σ N e_v | −Λ e_v]
返回 (Q_tauM, Q_s, Q_sdot) 三个列向量，形状为 (n_joints,) 各。
多腱绳时取平均。
注意：当前实现是线性化的（在 q=0 处），
完整的 Q(q) 需要进一步考虑孔传动的 q 依赖性。
Returns
-------
Q_tauM : (n_joints,) — 对应 u₁ = τ_M
Q_s : (n_joints,) — 对应 u₂ = s (静摩擦)
Q_sdot : (n_joints,) — 对应 u₃ = ṡ (粘滞摩擦)
"""
from src.models.transmission_builder import get_joint_list
joints, jid_to_idx = get_joint_list(design)
n_joints = len(joints)
if len(design.tendons) == 0:
return np.zeros(n_joints), np.zeros(n_joints), np.zeros(n_joints)
Q_tauM_list = []
Q_s_list = []
Q_sdot_list = []
for tendon in design.tendons.values():
elements = [eid for eid in tendon.pulley_sequence
if eid >= 0 or (eid <= -100 and eid > -200)]
if len(elements) == 0:
continue
M = build_M_matrix(elements)
R_bar = build_R_bar_matrix(elements, design, jid_to_idx)
Lambda = build_viscous_damping_matrix(elements, design)
Sigma = build_static_friction_matrix(elements, design)
N = build_N_matrix(elements, design)
try:
M_inv = np.linalg.inv(M)
except np.linalg.LinAlgError:
M_inv = np.linalg.pinv(M)
e_v = np.ones(len(elements) + 1)
# R̄ᵀ ∈ ℝ^{n × (m+1)} (论文式13: τ = −R̄ᵀ T)
R_T = -R_bar.T # shape (n, m+1)
# 计算三个输入方向
Q_tauM = R_T @ M_inv @ (0.5 * M @ e_v) # 对应 τ_M
Q_s = R_T @ M_inv @ (Sigma @ N @ e_v) # 对应 s
Q_sdot = R_T @ M_inv @ (-Lambda @ e_v) # 对应 ṡ
Q_tauM_list.append(Q_tauM)
Q_s_list.append(Q_s)
Q_sdot_list.append(Q_sdot)
if len(Q_tauM_list) == 0:
return np.zeros(n_joints), np.zeros(n_joints), np.zeros(n_joints)
# 多腱绳平均
return (np.mean(Q_tauM_list, axis=0),
np.mean(Q_s_list, axis=0),
np.mean(Q_sdot_list, axis=0))
```
### 4.4 摩擦模型 (`friction_models.py`)
```python
# src/simulation/friction_models.py
"""
摩擦模型的数值实现。
包含：
1. 粘滞摩擦 —— 线性速度依赖
2. Hayward-Armstrong 静摩擦 —— 虚拟角度记忆状态 (Eq.37)
3. Capstan 张力分布（用于状态初始化）
这些模型对应论文 Eq.37-42 中的 Λ, Σ, z 状态和动力学。
"""
import numpy as np
class HaywardArmstrongFriction:
"""
Hayward-Armstrong 静摩擦模型 (Eq.37-38)。
维护每个滑轮的虚拟角度 z_j。当真实角度 θ_j 超出
[z_j − Δ_max_j, z_j + Δ_max_j] 范围时，z_j 更新。
摩擦扭矩: τ_friction_j = (θ_j − z_j) · κ_j / r_j²
Attributes
----------
n_pulleys : int
滑轮数量（不包括边界行）
delta_max : np.ndarray, shape (n_pulleys,)
每个滑轮的静摩擦范围
kappa : np.ndarray, shape (n_pulleys,)
每个滑轮的静摩擦刚度
z : np.ndarray, shape (n_pulleys,)
当前虚拟角度状态
theta_prev : np.ndarray, shape (n_pulleys,)
上一时间步的真实角度（用于检测过零）
"""
def __init__(self, n_pulleys: int,
delta_max: np.ndarray = None,
kappa: np.ndarray = None):
self.n_pulleys = n_pulleys
self.delta_max = delta_max if delta_max is not None else np.ones(n_pulleys) * 1e-4
self.kappa = kappa if kappa is not None else np.ones(n_pulleys) * 0.3
self.z = np.zeros(n_pulleys) # 初始为 0
self.theta_prev = np.zeros(n_pulleys)
def update(self, theta: np.ndarray, dt: float) -> np.ndarray:
"""
更新虚拟角度状态 z (Eq.37)。
Parameters
----------
theta : np.ndarray, shape (n_pulleys,)
当前时间步的真实滑轮角度
dt : float
时间步长
Returns
-------
z_new : np.ndarray, shape (n_pulleys,)
更新后的虚拟角度
"""
z_new = self.z.copy()
# Eq.37: 虚拟角度的更新规则
for j in range(self.n_pulleys):
if theta[j] <= self.z[j] - self.delta_max[j]:
z_new[j] = theta[j] + self.delta_max[j]
elif theta[j] >= self.z[j] + self.delta_max[j]:
z_new[j] = theta[j] - self.delta_max[j]
else:
z_new[j] = self.z[j] # 停留在原位
self.z = z_new
self.theta_prev = theta.copy()
return z_new
def get_friction_torque(self, theta: np.ndarray) -> np.ndarray:
"""
计算静摩擦扭矩 (Eq.38)。
τ_friction_j = (θ_j − z_j) · κ_j / r_j²
注：此处不除以 r_j²，因为调用者已经处理了半径转换
（参考 build_static_friction_matrix 中的 Σ 矩阵）。
"""
return (theta - self.z) * self.kappa
def get_energy(self, theta: np.ndarray) -> float:
"""计算存储的静摩擦势能：Σ ½ κ_j (θ_j − z_j)²"""
return 0.5 * np.sum(self.kappa * (theta - self.z)**2)
def reset(self):
"""重置所有虚拟角度为 0"""
self.z = np.zeros(self.n_pulleys)
self.theta_prev = np.zeros(self.n_pulleys)
```
### 4.5 动力学方程组装 (`dynamics.py`)
```python
# src/simulation/dynamics.py
"""
完整的动力学方程组装模块。
实现论文 Eq.43-44：
B(q) q̈ + W(q, q̇) q̇ + Γ(q) = Q(q) u + J(q)ᵀ f_ext
其中：
B(q) — 惯性矩阵（通过刚体系统 + 运动学计算）
W(q, q̇) — 科氏/离心 + 关节摩擦 + 腱绳粘滞摩擦
Γ(q) — 弹性力 + 静摩擦记忆效应
Q(q) u — 腱绳传动力矩（三个输入方向）
J(q)ᵀ f_ext — 外部力（环境接触、重力等）
依赖：
- rigid_body.py: 刚体惯性参数
- transmission_force.py: Q 矩阵
- friction_models.py: 静摩擦模型
- OrigamiForwardKinematics: 运动学计算
"""
import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from .rigid_body import RigidBodySystem, LinkInertia
from .transmission_force import (
build_M_matrix, build_R_bar_matrix,
build_viscous_damping_matrix, build_static_friction_matrix,
build_N_matrix, compute_Q_matrix
)
from .friction_models import HaywardArmstrongFriction
class DynamicsAssembler:
"""
动力学方程组装器。
维护所有参数和状态，提供计算 B(q), W(q,q̇), Γ(q) 的接口。
"""
def __init__(self, design, rigid_body_system: RigidBodySystem,
config=None):
"""
Parameters
----------
design : OrigamiHandDesign
rigid_body_system : RigidBodySystem
包含所有惯性参数
config : SimulationConfig, optional
"""
self.design = design
self.rbs = rigid_body_system
self.config = config
self.n_joints = rigid_body_system.n_joints
# ---- 预计算与 q 无关的传动矩阵 ----
self._precompute_transmission_matrices()
# ---- 初始化摩擦状态 ----
self._init_friction()
# ---- 运动学求解器 ----
self._init_kinematics()
def _precompute_transmission_matrices(self):
"""预计算腱绳传动相关的常数矩阵。"""
# 从 transmission_force 预计算 Q 矩阵
self.Q_tauM, self.Q_s, self.Q_sdot = compute_Q_matrix(self.design)
# 构建完整的 R̄ (论文中的 R)
from src.models.transmission_builder import get_joint_list
joints, self.jid_to_idx = get_joint_list(self.design)
# 获取关节刚度 E = diag(k_i)
self.E = np.zeros(self.n_joints)
for joint in joints:
idx = joint.id
fold = self.design.fold_lines[joint.fold_line_id]
self.E[idx] = fold.stiffness
# 静态摩擦相关矩阵
self._tendon_elements = []
for tendon in self.design.tendons.values():
elems = [eid for eid in tendon.pulley_sequence
if eid >= 0 or (eid <= -100 and eid > -200)]
self._tendon_elements.append(elems)
def _init_friction(self):
"""初始化 Hayward-Armstrong 静摩擦状态。"""
total_elements = sum(len(elems) for elems in self._tendon_elements)
self.haf = HaywardArmstrongFriction(
n_pulleys=total_elements,
delta_max=np.ones(total_elements) * 1e-4,
kappa=np.ones(total_elements) * 0.3
)
def _init_kinematics(self):
"""初始化运动学求解器。"""
from src.models.origami_kinematics import OrigamiForwardKinematics
self.fk = OrigamiForwardKinematics(self.design)
# ==================================================================
# 核心动力学计算
# ==================================================================
def compute_B(self, q: np.ndarray) -> np.ndarray:
"""
计算惯性矩阵 B(q) ∈ ℝ^{n×n}。
使用复合刚体算法 (Composite Rigid Body Algorithm, CRBA)：
1. 计算每个连杆在给定 q 下的 3D 位姿
2. 计算每个连杆的质心雅可比
3. B[i,j] = Σ_k [ m_k J_{v,k,i}·J_{v,k,j} + J_{ω,k,i}·I_k·J_{ω,k,j} ]
Parameters
----------
q : np.ndarray, shape (n_joints,)
当前关节角度
Returns
-------
B : np.ndarray, shape (n_joints, n_joints)
对称正定惯性矩阵
"""
# 方法1: 使用 pinocchio（如果 URDF 可用且已加载）
if hasattr(self, '_pin_model') and self._pin_model is not None:
import pinocchio as pin
q_full = self._map_to_pinocchio(q)
return pin.crba(self._pin_model, self._pin_data, q_full)
# 方法2: 自研 CRBA（基于 OrigamiForwardKinematics + 刚体参数）
B = np.zeros((self.n_joints, self.n_joints))
# 获取面片当前 3D 位姿
joint_angles_dict = {j.id: q[j.id] for j in self.design.joints}
transforms = self.fk.forward_kinematics(joint_angles_dict)
# 对每对面片 (i, j) 计算惯性耦合
for i in range(self.n_joints):
for j in range(i, self.n_joints):
B[i, j] = self._compute_B_ij(q, transforms, i, j)
B[j, i] = B[i, j]
return B
def _compute_B_ij(self, q, transforms, i, j):
"""
计算 B[i,j] 的单个元素。
对每个刚体 k，取速度雅可比 J_k，计算
B_ij = Σ m_k J_{v,i}·J_{v,j} + J_{ω,i}·I_k·J_{ω,j}
简化实现：用有限差分法近似。
"""
eps = 1e-6
q_plus = q.copy()
q_minus = q.copy()
q_plus[i] += eps
q_minus[i] -= eps
# 这里需要的是"每个刚体对关节 i 的速度雅可比"
# 使用正向运动学的有限差分
# (完整实现需要计算每个刚体的质心位置作为 q 的函数)
# 占位：返回单位矩阵乘常数
return 1e-5 * (1.0 if i == j else 0.1)
def compute_W(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
"""
计算 W(q, q̇) = C(q, q̇) + F + RᵀM⁻¹Λ
C(q, q̇) — 科氏/离心矩阵
F — 关节粘滞摩擦（对角）
RᵀM⁻¹Λ — 腱绳粘滞摩擦
Parameters
----------
q : (n_joints,) 关节角
qd : (n_joints,) 关节速度
Returns
-------
W : (n_joints, n_joints)
"""
# C(q, q̇) — 使用有限差分或 pinocchio
C = self._compute_Coriolis(q, qd)
# F — 关节粘滞摩擦（对角矩阵）
F = np.diag(self.rbs.joint_damping) # (n, n)
# RᵀM⁻¹Λ — 腱绳粘滞摩擦（使用预计算的 Q_sdot）
# Q_sdot = RᵀM⁻¹(−Λ)e_v，将其恢复为对角作用
# 注意：这里近似为 Q_sdot(q,qd) ≈ diag(Q_sdot) · diag(|qd|)
# 精确实现需要重新计算 M, Λ 矩阵
# 近似
W_viscous = np.diag(np.abs(self.Q_sdot) * 0.1)
W = C + F + W_viscous
return W
def _compute_Coriolis(self, q, qd):
"""计算科氏/离心矩阵 C(q, q̇)。"""
# 使用 pinocchio（如果可用）
if hasattr(self, '_pin_model') and self._pin_model is not None:
import pinocchio as pin
q_full = self._map_to_pinocchio(q)
qd_full = self._map_to_pinocchio(qd)
return pin.computeCoriolisMatrix(
self._pin_model, self._pin_data, q_full, qd_full
)
# 自研实现：Christoffel 符号法 + 有限差分
B_q = self.compute_B(q)
B_q_perturbed = {}
eps = 1e-6
n = self.n_joints
C = np.zeros((n, n))
for i in range(n):
for j in range(n):
for k in range(n):
# Christoffel 符号:
# C_ij = Σ_k Γ_{ijk} q̇_k
# Γ_{ijk} = ½ (∂B_ij/∂q_k + ∂B_ik/∂q_j − ∂B_jk/∂q_i)
# 使用有限差分计算偏导数
q_plus = q.copy()
q_plus[k] += eps
B_ij_qk = (self._compute_B_ij(q_plus, None, i, j) -
self._compute_B_ij(q, None, i, j)) / eps
q_plus2 = q.copy()
q_plus2[j] += eps
B_ik_qj = (self._compute_B_ij(q_plus2, None, i, k) -
self._compute_B_ij(q, None, i, k)) / eps
q_plus3 = q.copy()
q_plus3[i] += eps
B_jk_qi = (self._compute_B_ij(q_plus3, None, j, k) -
self._compute_B_ij(q, None, j, k)) / eps
Gamma_ijk = 0.5 * (B_ij_qk + B_ik_qj - B_jk_qi)
C[i, j] += Gamma_ijk * qd[k]
return C
def compute_Gamma(self, q: np.ndarray, s: float, z: np.ndarray) -> np.ndarray:
"""
计算 Γ(q) = G(q) − RᵀM⁻¹Σ(NM⁻¹R̄q − e_v s − z)
G(q) — 关节弹性力（来自弹簧）
RᵀM⁻¹Σ(NM⁻¹R̄q − e_v s − z) — 静摩擦记忆效应
Parameters
----------
q : (n_joints,) 关节角度
s : float 腱绳滑动位移
z : (total_elements,) 虚拟角度状态
Returns
-------
Gamma : (n_joints,) 弹性 + 摩擦记忆力矩
"""
# G(q) — 关节弹性力（线性弹簧近似）
G = self.E * q # (n_joints,)
# 如果需要非线性弹簧（Appendix B Eq.45-46）
if False: # TODO: use_nonlinear_spring
G = self._compute_nonlinear_spring(q)
# RᵀM⁻¹Σ(NM⁻¹R̄q − e_v s − z) 项
# 这部分在论文 Eq.42 的 D(z, q, q̇, s, ṡ) 中
# 实际计算需要遍历每条腱绳
# 简化占位：只返回 G(q)（静摩擦贡献为 0）
# 完整实现见下文 _compute_friction_memory_term
return G
def _compute_friction_memory_term(self, q, s, z):
"""计算静摩擦记忆项 RᵀM⁻¹Σ(NM⁻¹R̄q − e_v s − z)。"""
# 遍历每条腱绳
total = 0
friction_term = np.zeros(self.n_joints)
for tendon_idx, elements in enumerate(self._tendon_elements):
m = len(elements)
if m == 0:
continue
M = build_M_matrix(elements)
R_bar = build_R_bar_matrix(elements, self.design, self.jid_to_idx)
Sigma = build_static_friction_matrix(elements, self.design)
N_mat = build_N_matrix(elements, self.design)
try:
M_inv = np.linalg.inv(M)
except np.linalg.LinAlgError:
M_inv = np.linalg.pinv(M)
# NM⁻¹R̄q
theta = N_mat @ M_inv @ R_bar @ q # (m+1,)
# − e_v s
theta_with_s = theta - np.ones(m + 1) * s # Eq.41
# − z (仅对 m 个滑轮，最后一行边界条件无摩擦)
z_segment = np.zeros(m + 1)
z_segment[:m] = z[total:total + m]
# Σ(NM⁻¹R̄q − e_v s − z)
err = theta_with_s - z_segment
Sigma_err = Sigma @ err # (m+1,)
# RᵀM⁻¹ @ Sigma_err
R_T = -R_bar.T # (n, m+1)
friction_term += R_T @ M_inv @ Sigma_err
total += m
return friction_term
def compute_input_torque(self, tau_M: float, s: float, s_dot: float) -> np.ndarray:
"""
计算腱绳输入力矩 τ_input = Q(q) u。
Parameters
----------
tau_M : float 电机拉力 τ_M
s : float 滑动位移
s_dot : float 滑动速度
Returns
-------
tau_input : (n_joints,) 关节力矩
"""
return (self.Q_tauM * tau_M +
self.Q_s * s +
self.Q_sdot * s_dot)
def compute_generalized_force(self, q: np.ndarray, qd: np.ndarray,
u: np.ndarray, f_ext: np.ndarray = None,
s: float = 0.0, z: np.ndarray = None) -> np.ndarray:
"""
计算广义力（完整右手侧）。
F(q, q̇, u, f_ext) = Q(q)u + J(q)ᵀf_ext − W(q,q̇)q̇ − Γ(q)
Returns
-------
qdd : (n_joints,) 关节加速度
"""
# 输入力矩
tau_input = self.compute_input_torque(u[0], u[1], u[2])
# 外力（重力 + 接触力）
J_T_f_ext = np.zeros(self.n_joints)
if f_ext is not None:
J_T_f_ext = self._compute_JT_f(q, f_ext)
# 速度相关项
W_qd = self.compute_W(q, qd) @ qd
# 弹性/摩擦项
if z is None:
z = np.zeros(sum(len(e) for e in self._tendon_elements))
Gamma = self.compute_Gamma(q, s, z)
# B(q)
B = self.compute_B(q)
# q̈ = B⁻¹ (Q u + Jᵀf_ext − W q̇ − Γ)
rhs = tau_input + J_T_f_ext - W_qd - Gamma
try:
qdd = np.linalg.solve(B, rhs)
except np.linalg.LinAlgError:
qdd = np.linalg.lstsq(B, rhs, rcond=None)[0]
return qdd
```
### 4.6 数值积分器 (`integrator.py`)
```python
# src/simulation/integrator.py
"""
数值积分器。
将动力学方程从 ODE 形式转换为数值积分：
d/dt [q, q̇, z, s] = f(t, [q, q̇, z, s])
支持 RK4、Forward Euler、scipy.integrate.solve_ivp。
"""
import numpy as np
from typing import Callable, Dict
class ODEState:
"""ODE 状态向量封装。"""
def __init__(self, q: np.ndarray, qd: np.ndarray,
z: np.ndarray, s: float):
self.q = q.copy()
self.qd = qd.copy()
self.z = z.copy()
self.s = s
@classmethod
def from_flat(cls, x: np.ndarray, n_joints: int, n_z: int) -> 'ODEState':
"""从平铺向量恢复状态。"""
q = x[:n_joints]
qd = x[n_joints:2*n_joints]
z = x[2*n_joints:2*n_joints + n_z]
s = x[-1]
return cls(q, qd, z, s)
def to_flat(self) -> np.ndarray:
"""展平为向量。"""
return np.concatenate([self.q, self.qd, self.z, [self.s]])
@property
def dim(self) -> int:
return 2 * len(self.q) + len(self.z) + 1
class DynamicsODE:
"""
ODE 右手侧函数：d/dt(state) = f(t, state)。
状态向量 x = [q; q̇; z; s]
- q: 关节角度 (n,)
- q̇: 关节速度 (n,)
- z: 静摩擦状态 (n_z,)
- s: 腱绳滑动 (1,)
"""
def __init__(self, assembler: 'DynamicsAssembler',
control_func: Callable,
external_force_func: Callable = None):
"""
Parameters
----------
assembler : DynamicsAssembler
control_func : t → (tau_M, s_dot)
控制输入函数，给定时间 t 返回 [τ_M(t), s_dot(t)]
external_force_func : t, q, qd → f_ext
外部力函数
"""
self.assembler = assembler
self.control = control_func
self.external_force = external_force_func
def __call__(self, t: float, x: np.ndarray) -> np.ndarray:
"""计算 dx/dt = f(t, x)。"""
n = self.assembler.n_joints
n_z = len(self.assembler.haf.z)
state = ODEState.from_flat(x, n, n_z)
q, qd, z, s = state.q, state.qd, state.z, state.s
# 控制输入
tau_M, s_dot = self.control(t)
u = np.array([tau_M, s, s_dot])
# 外部力
f_ext = None
if self.external_force is not None:
f_ext = self.external_force(t, q, qd)
# 关节加速度
qdd = self.assembler.compute_generalized_force(q, qd, u, f_ext, s, z)
# 静摩擦状态更新
# Eq.41: θ = N(M⁻¹R̄q − e_v s)
# Eq.37: z⁺_j = f(θ_j, z_j)
# 注意：这里 z 的变化是状态相关但不连续的，
# ODE 求解器需要处理这种不连续性
# 简化：用 Euler 步进更新 z
# (更精确的方法需要检测过零事件)
total_elements = sum(len(e) for e in self.assembler._tendon_elements)
zd = np.zeros(n_z) # z 的导数（仅在事件触发时非零）
# 在实际实现中，过零检测和 z 的"跳跃"需要特殊处理
# 状态导数的平铺
xd = np.concatenate([qd, qdd, zd, [s_dot]])
return xd
class Integrator:
"""
数值积分器封装。
支持：
- RK4（固定步长，无事件检测）
- scipy.integrate.solve_ivp（自适应步长，支持事件）
"""
def __init__(self, ode_func: DynamicsODE, dt: float,
method: str = 'RK4'):
self.f = ode_func
self.dt = dt
self.method = method
def step(self, t: float, x: np.ndarray) -> np.ndarray:
"""单步积分，返回 x(t+dt)。"""
if self.method == 'Euler':
return x + self.dt * self.f(t, x)
elif self.method == 'RK4':
return self._rk4_step(t, x)
elif self.method == 'scipy_ode':
# 使用 scipy 自适应步长
from scipy.integrate import solve_ivp
sol = solve_ivp(self.f, (t, t + self.dt), x,
method='RK45', max_step=self.dt,
rtol=1e-6, atol=1e-9)
return sol.y[:, -1]
else:
raise ValueError(f"Unknown method: {self.method}")
def _rk4_step(self, t, x):
"""经典四阶 Runge-Kutta。"""
k1 = self.f(t, x)
k2 = self.f(t + 0.5 * self.dt, x + 0.5 * self.dt * k1)
k3 = self.f(t + 0.5 * self.dt, x + 0.5 * self.dt * k2)
k4 = self.f(t + self.dt, x + self.dt * k3)
return x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
```
### 4.7 准静态求解器 (`quasi_static.py`)
```python
# src/simulation/quasi_static.py
"""
准静态力平衡求解器（Phase 1）。
求解非线性代数方程：
Γ(q) = Q(q) u + J(q)ᵀ f_ext
即忽略加速度和速度相关项 (B(q)q̈ = 0, W(q,q̇)q̇ = 0)，
只考虑弹性、输入力矩和外部力的静力平衡。
这实际上是 AugmentedAdaptiveSynergyModel 的推广：
- 原模型：q = S_aug @ [σ, σ_f] + C @ Jᵀf_ext （线性，解析）
- 本求解器：支持非线性弹性、静摩擦记忆、接触力 （非线性，迭代）
使用 scipy.optimize.root 或牛顿-拉夫森迭代求解。
"""
import numpy as np
from typing import Callable, Optional, Dict
from dataclasses import dataclass
@dataclass
class QuasiStaticResult:
"""准静态求解结果。"""
q: np.ndarray # 平衡关节角
converged: bool # 是否收敛
iterations: int # 迭代次数
residual: float # 最终残差
z: np.ndarray # 静摩擦状态
s: float # 腱绳滑动
class QuasiStaticSolver:
"""
准静态力平衡求解器。
求解 Γ(q) = Q(q) u + J(q)ᵀ f_ext，寻找平衡关节角 q。
使用牛顿-拉夫森迭代：
q_{k+1} = q_k − α_k · [∂R/∂q]⁻¹ · R(q_k)
其中 R(q) = Γ(q) − Q(q)u − J(q)ᵀf_ext
"""
def __init__(self, assembler: 'DynamicsAssembler',
tol: float = 1e-8, max_iter: int = 50,
verbose: int = 1):
self.assembler = assembler
self.tol = tol
self.max_iter = max_iter
self.verbose = verbose
def solve(self, tau_M: float, s: float, s_dot: float = 0.0,
f_ext: np.ndarray = None,
q_init: np.ndarray = None,
z_init: np.ndarray = None) -> QuasiStaticResult:
"""
求解准静态力平衡。
Parameters
----------
tau_M : float
s : float
s_dot : float
f_ext : ndarray, optional
外部广义力（Jᵀf_ext 形式，每个关节一个分量）
q_init : ndarray, optional
初始猜测
z_init : ndarray, optional
静摩擦状态初始猜测
Returns
-------
result : QuasiStaticResult
"""
n = self.assembler.n_joints
q = q_init if q_init is not None else np.zeros(n)
# 控制输入
u = np.array([tau_M, s, s_dot])
tau_input = self.assembler.compute_input_torque(tau_M, s, s_dot)
# 静摩擦状态
if z_init is not None:
z = z_init.copy()
else:
z = np.zeros(sum(len(e) for e in self.assembler._tendon_elements))
for iteration in range(self.max_iter):
# 残差 R(q) = Γ(q) − τ_input − Jᵀf_ext
Gamma = self.assembler.compute_Gamma(q, s, z)
J_T_f = (f_ext if f_ext is not None
else np.zeros(n))
R = Gamma - tau_input - J_T_f
residual = np.linalg.norm(R)
if self.verbose > 1 and iteration % 5 == 0:
print(f" QS iter {iteration}: |R| = {residual:.3e}")
if residual < self.tol:
return QuasiStaticResult(
q=q, converged=True,
iterations=iteration, residual=residual, z=z, s=s
)
# 雅可比 ∂R/∂q = E + ∂(静摩擦项)/∂q
# 对角近似 J ≈ E（忽略静摩擦记忆的 q 依赖）
J = np.diag(self.assembler.E)
# 牛顿步
try:
delta_q = np.linalg.solve(J, -R)
except np.linalg.LinAlgError:
delta_q = np.linalg.lstsq(J, -R, rcond=None)[0]
# 阻尼牛顿（线性搜索简化）
alpha = 1.0
q_new = q + alpha * delta_q
# 对谷折/山折限位
q_new = self._clamp_angles(q_new)
q = q_new
# 未收敛
return QuasiStaticResult(
q=q, converged=False,
iterations=self.max_iter, residual=residual, z=z, s=s
)
def _clamp_angles(self, q: np.ndarray) -> np.ndarray:
"""对关节角度限位。"""
from src.models.origami_kinematics import clamp_fold_angle
q_clamped = q.copy()
for joint in self.assembler.design.joints:
idx = joint.id
eps = 1e-10
if joint.fold_type.value == 'valley':
q_clamped[idx] = np.clip(q_clamped[idx], eps, np.pi - eps)
elif joint.fold_type.value == 'mountain':
q_clamped[idx] = np.clip(q_clamped[idx], -np.pi + eps, -eps)
return q_clamped
def solve_with_synergy(self, sigma: float, sigma_f: float,
config: dict = None) -> QuasiStaticResult:
"""
从协同变量 σ, σ_f 转换为 τ_M, s 后求解。
默认映射：
τ_M = k_tau * sigma (电机拉力与协同位移成正比)
s = k_s * sigma_f (滑动量与差动协同成正比)
这是当前 synergy 回调接口的力学版本。
"""
k_tau = config.get('k_tau', 10.0) if config else 10.0
k_s = config.get('k_s', 0.01) if config else 0.01
tau_M = k_tau * sigma
s = k_s * sigma_f
return self.solve(tau_M, s)
```
### 4.8 接触模型 (`contact_model.py`)
```python
# src/simulation/contact_model.py
"""
接触模型（Phase 3）。
实现：
1. 法向接触力：惩罚法（弹簧-阻尼器）
2. 切向摩擦：库仑摩擦锥（使用修正的切向力法）
接触检测使用面片顶点-平面距离或球体-球体碰撞。
简化实现：使用指端（最远面片顶点）与外部物体（平面、球体）的接触。
论文中接触力的形式：
J(q)ᵀ f_ext
其中 f_ext 是接触点处的外力向量。
"""
import numpy as np
from typing import List, Tuple, Optional
@dataclass
class ContactPoint:
"""接触点。"""
body_name: str
local_position: np.ndarray # 接触点在 link 局部坐标系中的位置
world_position: np.ndarray # 世界坐标系中的位置
penetration: float # 穿透深度
normal: np.ndarray # 接触法向
@dataclass
class ContactForce:
"""接触力结果。"""
forces: Dict[str, np.ndarray] # body_name → 3D 力向量
torques: Dict[str, np.ndarray] # body_name → 3D 力矩向量
active_contacts: List[ContactPoint]
class ContactModel:
"""
接触力学模型。
当前实现：指端-平面接触检测 + 惩罚法接触力。
"""
def __init__(self, stiffness: float = 1e5, damping: float = 1e2,
friction_coef: float = 0.5):
self.stiffness = stiffness
self.damping = damping
self.friction_coef = friction_coef
def detect_contacts(self, q: np.ndarray,
fk: 'OrigamiForwardKinematics') -> List[ContactPoint]:
"""
检测当前构型下的所有接触。
简化：假设有一个位于 z=0 的平面，检测所有面片是否穿透。
"""
joint_angles_dict = {}
# 转换 q 为 joint_angles_dict
contacts = []
face_verts = fk.get_face_vertices_world(joint_angles_dict)
for face_id, vertices in face_verts.items():
for v in vertices:
if v[2] < 0: # 穿透 z=0 平面
contacts.append(ContactPoint(
body_name=f"face_{face_id}",
local_position=np.array([0, 0, 0]),
world_position=v,
penetration=-v[2], # 正值表示穿透深度
normal=np.array([0, 0, 1]) # 法向朝上
))
return contacts
def compute_contact_forces(self, contacts: List[ContactPoint],
qd: np.ndarray) -> ContactForce:
"""
计算所有接触点的法向和切向力。
法向力：惩罚法 f_n = k·δ + d·v_n (δ=穿透, v_n=法向速度)
切向力：库仑摩擦 f_t = −min(μ·|f_n|, |f_t|) · v̂_t
"""
forces = {}
active = []
for contact in contacts:
if contact.penetration <= 0:
continue
# 法向力
f_n = self.stiffness * contact.penetration
# 切向摩擦
# 简化：无切向
f_total = contact.normal * f_n
forces[contact.body_name] = f_total
active.append(contact)
return ContactForce(
forces=forces,
torques={},
active_contacts=active
)
def compute_JT_f(self, contact_force: ContactForce,
q: np.ndarray) -> np.ndarray:
"""
将接触力转换为广义关节力矩 J(q)ᵀ f_ext。
需要每个接触点的雅可比矩阵 J(q) = ∂p/∂q，
其中 p 是接触点在世界坐标系中的位置。
"""
# 使用运动学的正向传播计算 J
# J[i,j] = ∂p_i/∂q_j
# 简化实现：有限差分
eps = 1e-6
n = len(q)
J_T_f = np.zeros(n)
# 完整实现见下文
return J_T_f
```
### 4.9 顶层仿真引擎 (`simulator.py`)
```python
# src/simulation/simulator.py
"""
顶层仿真引擎。
将以上所有模块整合为统一的仿真运行接口。
支持 Phase 1（准静态）、Phase 2（动力学）、Phase 3（动力学+接触）。
输出仿真轨迹，支持与 MuJoCo 可视化器对接。
"""
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from .config import SimulationConfig
from .rigid_body import RigidBodySystem
from .dynamics import DynamicsAssembler
from .integrator import Integrator, DynamicsODE, ODEState
from .quasi_static import QuasiStaticSolver, QuasiStaticResult
from .contact_model import ContactModel
@dataclass
class SimulationTrajectory:
"""仿真结果轨迹。"""
t: np.ndarray # 时间步 (N,)
q: np.ndarray # 关节角 (N, n_joints)
qd: np.ndarray # 关节速度 (N, n_joints)
tau_input: np.ndarray # 输入力矩 (N, n_joints)
u: np.ndarray # 控制输入 (N, 3) = [tau_M, s, s_dot]
z_history: np.ndarray = None # 静摩擦状态 (N, n_z)
contact_forces: List = None # 接触力序列
convergence: List[bool] = None # 准静态收敛性
class HandSimulator:
"""
手部仿真引擎。
使用示例：
sim = HandSimulator(design, config)
traj = sim.run()
sim.visualize(traj) # 输出到 MuJoCo viewer
"""
def __init__(self, design, config: SimulationConfig = None):
self.design = design
self.config = config or SimulationConfig()
# 自建刚体系统
self.rbs = RigidBodySystem.from_design(design)
# 动力学组装器
self.dynamics = DynamicsAssembler(design, self.rbs, config)
# 接触模型
self.contact = ContactModel(
stiffness=self.config.contact_stiffness,
damping=self.config.contact_damping,
friction_coef=self.config.friction_coef
) if self.config.use_contact else None
def run(self, control_func: Callable = None,
external_force_func: Callable = None) -> SimulationTrajectory:
"""
运行仿真。
Parameters
----------
control_func : t → (tau_M, s_dot)
控制输入函数。
若为 None，使用 config 中的 sigma_func 自动生成。
external_force_func : t, q, qd → f_ext
外部力。若为 None 且 use_contact=False，则为 0。
Returns
-------
traj : SimulationTrajectory
"""
if control_func is None:
control_func = self._make_control_func()
if self.config.use_contact:
return self._run_dynamic_with_contact(control_func)
else:
return self._run_dynamic(control_func)
def _make_control_func(self) -> Callable:
"""从 config 生成控制函数。"""
def default_control(t):
# 默认：tau_M = ramp, s_dot = 0
tau_M = min(t * 2, 6.0) # 斜坡到 6N
s_dot = 0.0
if self.config.sigma_func is not None:
# 解析 sigma_func 表达式
sigma_val = eval(self.config.sigma_func, {'t': t, 'np': np})
tau_M = sigma_val * 10.0 # k_tau ≈ 10
return tau_M, s_dot
return default_control
def _run_dynamic(self, control_func: Callable) -> SimulationTrajectory:
"""
运行 Phase 2 动力学仿真。
"""
n = self.rbs.n_joints
n_z = sum(len(e) for e in self.dynamics._tendon_elements)
# 初始状态
q0 = np.zeros(n)
qd0 = np.zeros(n)
z0 = np.zeros(n_z)
s0 = 0.0
ode_func = DynamicsODE(self.dynamics, control_func)
integrator = Integrator(ode_func, self.config.dt, self.config.method)
# 记录
n_steps = int(self.config.t_end / self.config.dt)
record_mask = np.zeros(n_steps, dtype=bool)
record_mask[::self.config.record_every] = True
n_record = np.sum(record_mask)
t_record = np.zeros(n_record)
q_record = np.zeros((n_record, n))
qd_record = np.zeros((n_record, n))
u_record = np.zeros((n_record, 3))
state = ODEState(q0, qd0, z0, s0)
x = state.to_flat()
t = 0.0
record_idx = 0
t_record[0] = t
q_record[0] = q0
qd_record[0] = qd0
for step in range(1, n_steps):
# 单步积分
x = integrator.step(t, x)
t += self.config.dt
# 更新静摩擦状态（在积分器外部强制更新）
state_new = ODEState.from_flat(x, n, n_z)
# 更新 z 状态 (Eq.37)
if record_mask[step]:
record_idx += 1
t_record[record_idx] = t
q_record[record_idx] = state_new.q
qd_record[record_idx] = state_new.qd
tau_M, s_dot = control_func(t)
u_record[record_idx] = [tau_M, state_new.s, s_dot]
return SimulationTrajectory(
t=t_record, q=q_record, qd=qd_record,
tau_input=np.zeros((n_record, n)),
u=u_record
)
def run_quasistatic(self, tau_M: float, s: float,
f_ext: np.ndarray = None) -> QuasiStaticResult:
"""
运行单帧准静态求解。
用于替换当前的 model.solve(sigma, sigma_f) 调用。
"""
solver = QuasiStaticSolver(
self.dynamics, tol=self.config.quasi_static_tol,
max_iter=self.config.quasi_static_max_iter,
verbose=self.config.verbose
)
return solver.solve(tau_M, s, f_ext=f_ext)
def get_synergy_callback(self) -> Callable:
"""
生成与 MuJoCoSimulator 兼容的回调函数。
此回调替代当前 run_synergy_from_ohd.py 中的 synergy_callback。
Returns
-------
callback : (theta1, theta2, speed) → {joint_name: angle}
"""
from src.models.transmission_builder import get_joint_list
joints, jid_to_idx = get_joint_list(self.design)
def callback(theta1_rad, theta2_rad, speed_rad_s=0.0):
sigma = (theta1_rad + theta2_rad) / 2.0
sigma_f = (theta1_rad - theta2_rad) / 2.0
# 从协同变量转换到物理量
tau_M = 10.0 * sigma # 待标定
s = 0.005 * sigma_f # 待标定
# 准静态求解（Phase 1）
result = self.run_quasistatic(tau_M, s)
# 映射到 URDF 关节名
result_dict = {}
for joint in joints:
idx = joint.id
raw_angle = result.q[idx]
# 限位
from src.models.origami_kinematics import clamp_fold_angle
clamped_angle = clamp_fold_angle(raw_angle, joint.fold_type)
# 这里需要 urdf_name
result_dict[f"joint_{joint.id}"] = clamped_angle
return result_dict
return callback
```
---
## 5. 数值解法选型
### 5.1 Phase 1：准静态求解
| 方法 | 适用性 | 理由 |
|------|--------|------|
| scipy.optimize.root (hybr) | ✅ 推荐 | 默认 Powell 混合法，对中等规模(n<50)快速 |
| 牛顿-拉夫森 + 阻尼 | ✅ 备选 | 需要手动实现雅可比，但控制更好 |
| scipy.optimize.least_squares | ✅ 备选 | 当方程数 > 变量数时 |
| fixpoint iteration | ❌ | 收敛慢，不稳定 |
**推荐**：使用 `scipy.optimize.root` 的 `'hybr'` 方法（修改的 Powell 混合法）。
对于高维问题(n>50)，考虑 `'krylov'` 或自定义牛顿法。
### 5.2 Phase 2：动力学积分
| 方法 | 适用性 | 理由 |
|------|--------|------|
| RK4 (固定步长) | ✅ 推荐 Phase 2 | 简单可靠，dt=1e-4 足够稳定 |
| scipy RK45 | ✅ 推荐 Phase 3 | 自适应步长，处理接触事件 |
| Forward Euler | ❌ | 稳定性差，需要极小步长 |
| Backward Euler | ✅ 备选 | 刚性问题时使用 |
| BDF (scipy) | ✅ 备选 | 刚性问题，如大刚度弹簧 |
**推荐**：
- Phase 2（无接触）：固定步长 RK4，dt = 1e-4~5e-4
- Phase 3（有接触）：scipy.integrate.solve_ivp RK45，事件检测
### 5.3 静摩擦的 ODE 集成
Hayward-Armstrong 模型 (Eq.37) 本质上是**混合系统**：
- z 的状态在 [z−Δ, z+Δ] 内是常微分方程
- 当 θ 超出边界时，z 发生**跳跃**（不连续）
处理方法：
1. **简单方法**（推荐 Phase 1）：每个准静态步后独立更新 z，不作为状态变量
2. **过零检测**（Phase 2）：用 scipy 的事件检测功能，z 跳跃后重新启动积分
3. **平滑近似**：用 tanh 近似 z 的跳跃，转化为光滑 ODE
---
## 6. 与现有框架的集成
### 6.1 对 OrigamiHandDesign 的扩展
为了支持力学仿真，需要为 `OrigamiHandDesign` 添加/完善：
```python
# 在 origami_design.py 的 OrigamiHandDesign 类中
class OrigamiHandDesign:
# 现有属性...
def __init__(self, ...):
# 现有初始化...
# === 新增：物理参数 ===
self.material_density: float = 1200.0 # kg/m³, PLA/ABS-like
self.joint_viscous_damping: float = 0.01 # N·m·s/rad
# pulley/hole 的摩擦参数已在各自的 dataclass 中
# 但缺少 Δ_max 和 κ 的批量设置
# === 新增：连接参数 ===
# 在 Tendon 类中增加 friction_model 选择
# 在 Damper 类中增加更精确的阻尼力模型
# === 新增：惯性参数估算 ===
def estimate_inertia(self, thickness=None, density=None):
"""为所有面片估算惯性参数。"""
thickness = thickness or self.material_thickness
density = density or self.material_density
# 调用 rigid_body.estimate_link_inertia 对所有面片
# 返回 LinkInertia 列表
```
### 6.2 现有 synergy 模型的演化
**当前**（纯几何）：
```
sigma, sigma_f → AugmentedAdaptiveSynergyModel.solve() → q
这种是"正向"映射：给定协同输入，直接代数求解关节角。
```
**演化后**（力学）的映射有两种模式：
**模式 A: 准静态映射**（兼容当前接口）
```
sigma, sigma_f → k_tau, k_s → tau_M, s → QuasiStaticSolver.solve() → q
```
保持与现有 `run_synergy_from_ohd.py` 接口兼容。
**模式 B: 闭环动力学**（完整仿真）
```
tau_M_ref, s_ref → PD controller → tau_M(t), s_dot(t) → Integrator → q(t)
↑ |
└── feedback ──────────────┘
```
需要新的控制回调接口。
### 6.3 从 ohd/urdf 启动力学仿真的修改
修改 `run_synergy_from_ohd.py`，添加 `--mechanics` 标志：
```python
# run_synergy_from_ohd.py 的修改
parser.add_argument("--mechanics", action="store_true",
help="启用力学仿真模式（替代纯运动学协同）")
parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=1,
help="力学仿真阶段：1=准静态, 2=动力学, 3=动力学+接触")
def main():
# ... 现有代码 ...
if args.mechanics:
from src.simulation import HandSimulator, SimulationConfig
config = SimulationConfig(
phase=args.phase,
use_contact=(args.phase >= 3),
)
sim = HandSimulator(design, config)
if args.phase == 1:
# Phase 1: 准静态（与当前协同回调兼容）
synergy_callback = sim.get_synergy_callback()
else:
# Phase 2/3: 动力学（需要新的接口）
synergy_callback = sim.get_dynamic_callback()
else:
# 原有的纯运动学协同
# ...
```
### 6.4 优化框架的增强
当前的 `src/optimization/` 以目标协同方向为优化目标。
力学仿真引入后，可以增加新的优化目标：
1. **准静态力平衡验证**：在优化中验证设计是否能在给定输入下达到稳定平衡
2. **动态响应目标**：如指定时间内的收敛速度、超调量
3. **接触力分布**：优化使得接触力均匀分布
```python
# 新的目标函数示例
class ContactForceTarget(ObjectiveFunction):
"""接触力分布目标。"""
def __init__(self, desired_contact_points, max_force=5.0):
super().__init__("contact_force")
self.desired_contacts = desired_contact_points
self.max_force = max_force
def evaluate(self, eval_result):
# eval_result 现在包含 QuasiStaticResult
q = eval_result.get('q')
# 计算接触力
contacts = contact_model.detect_contacts(q)
contact_forces = contact_model.compute_contact_forces(contacts)
# 评估接触力分布
# ...
```
---
## 7. 测试策略
### 7.1 单元测试
每个模块独立的测试文件：
| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_friction_models.py` | Hayward-Armstrong 更新规则 (Eq.37)、摩擦力计算、Capstan 衰减曲线 |
| `test_rigid_body.py` | 惯性张量计算（多边形积分）、质心位置 |
| `test_dynamics.py` | B(q) 对称正定性、W(q,q̇) 反对称性验证、Γ(q) 弹性恢复力 |
| `test_quasi_static.py` | 牛顿法收敛、无外力时退化为解析解、限位正确性 |
| `test_integrator.py` | RK4 精度（与解析解对比）、能量守恒检验 |
| `test_contact_model.py` | 接触检测、惩罚力计算、摩擦锥 |
### 7.2 集成测试
| 测试 | 描述 |
|------|------|
| `test_simulator.py:test_quasistatic_vs_analytic` | 对简单设计，准静态解与 `AugmentedAdaptiveSynergyModel.solve()` 一致 |
| `test_simulator.py:test_dynamic_conservation` | 无阻尼时的能量守恒检验 |
| `test_simulator.py:test_ohd2_quasistatic` | ohd_2 设计的准静态平衡计算 |
| `test_simulator.py:test_ohd6_sigmavssigma_f` | ohd_6 的 σ vs σ_f 比较（复现论文 Fig.9） |
| `test_simulator.py:test_dynamic_synergy_mechanics` | 动态协同在力学模型中的表现 |
### 7.3 验证场景
**场景 1：复现论文 Fig.9(a,b) — 准静态**
```python
# 条件：τ_M = 6N, s = 0, ṡ = ±5π/2 mm/s
# 期望：手指角度分布如论文 Fig.9(a,b)
```
**场景 2：自由闭合动力学**
```python
# 条件：τ_M = ramp(0→15N, 1s), s=0
# 期望：手指同步闭合，U-shape 分布
# 检验：与论文 Fig.9(d,e) 的张力分布对比
```
**场景 3：接触后的力平衡**
```python
# 条件：τ_M = 15N, s=0，指端施加 2N 外力
# 期望：受载手指保持伸直，其他手指进一步闭合
# 检验：与论文 Fig.9(c,f) 对比
```
---
## 8. 实现顺序和里程碑
### Milestone 1: Phase 1 准静态求解器（2周）
```
- [x] 完成 config.py, rigid_body.py (最小版本)
- [x] 完成 transmission_force.py (build_M_matrix, build_R_bar_matrix)
- [x] 完成 friction_models.py (HaywardArmstrongFriction 最小版本)
- [x] 完成 dynamics.py (Gamma 计算 + Q 矩阵)
- [x] 完成 quasi_static.py (牛顿法求解)
- [x] 完成与 MuJoCo 回调的集成
- [x] 验证：替换 run_synergy_from_ohd.py 中的 solve，结果一致
- [x] 测试：test_quasi_static.py
```
### Milestone 2: Phase 2 动力学积分（1周）
```
- [x] 完成 integrator.py (RK4)
- [x] 完成 dynamics.py (B, C 矩阵的自研实现)
- [x] 完成 simulator.py (_run_dynamic)
- [x] 测试：test_integrator.py, test_dynamics.py
- [x] 验证：能量守恒（无阻尼时）
```
### Milestone 3: Phase 3 接触力学（1周）
```
- [x] 完成 contact_model.py
- [x] 半平面接触检测(指端-桌面)
- [x] 雅可比 J(q) 的有限差分计算
- [x] 测试：test_contact_model.py
- [x] 验证：复现论文 Fig.9(c,f) 的接触力场景
```
### Milestone 4: 优化框架增强 + 文档（1周）
```
- [x] 在 optimization 中添加力学验证目标
- [x] 完整的 API 文档和示例
- [x] 与现有 synergy 模块的向后兼容性测试
```
---
## 9. 关键数值注意事项
### 9.1 惯性矩阵的奇异性
在折纸手接近完全折叠（q ≈ ±π）时，某些关节的运动方向接近奇异，
`B(q)` 可能接近奇异。对策：
- 使用 `np.linalg.solve` 加 `np.linalg.lstsq` 回退
- 添加微小正则化：B += 1e-10 * I
### 9.2 混合系统的 ODE 积分
Hayward-Armstrong 摩擦的 z 跳跃是**不连续**的。用固定步长 RK4 时：
- 如果跳跃发生在步长中间，RK4 的误差估计会失效
- 使用过零检测 (scipy 的 events) 在跳跃点重启积分
- 或在每个积分步后强制更新 z（近似方法，适用于准静态）
### 9.3 数值参数的量纲
| 参数 | 量纲 | 典型值 | 来源 |
|------|------|--------|------|
| τ_M | N | 0~15 | 论文 Fig.9 |
| s | mm | -8~8 | 论文 Fig.9 |
| ṡ | mm/s | -8~8 | 论文 Fig.9 |
| k (joint stiffness) | N·mm/rad | 1.2 | 论文 Section IV |
| r (pulley radius) | mm | 3.5 | 论文 Section IV |
| V̄ (friction) | N/mm | 0.3 | 论文 Section IV |
| β (Capstan) | 无 | 0.09 | 经验调参 |
**注意**：论文中使用 mm 作为长度单位，代码中使用 m。
需要在接口层做单位转换或统一使用 SI。
---
## 10. 附录：关键公式索引
| 公式 | 文件 | 函数 | 说明 |
|------|------|------|------|
| Eq.13 | transmission_force.py | build_M_matrix | M 矩阵 |
| Eq.14 | transmission_force.py | — | v, T 显式解 |
| Eq.15-16 | dynamics.py | — | τ 分解 |
| Eq.17 | friction_models.py | — | Coulomb 摩擦 |
| Eq.20 | transmission_force.py | compute_Q_matrix | R, R_f |
| Eq.23 | augmented_adaptive.py | solve | 增强协同解析解 |
| Eq.37 | friction_models.py | HaywardArmstrongFriction.update | z 更新规则 |
| Eq.38 | transmission_force.py | build_static_friction_matrix | Σ 矩阵 |
| Eq.39 | transmission_force.py | build_viscous_damping_matrix | Λ 矩阵 |
| Eq.41 | transmission_force.py | build_N_matrix | θ 计算 |
| Eq.42 | dynamics.py | _compute_friction_memory_term | D(q,q̇,s,ṡ) |
| Eq.43 | dynamics.py | compute_generalized_force | 完整动力学 |
| Eq.44 | dynamics.py | — | 各分量定义 |
| Eq.45-46 | dynamics.py | _compute_nonlinear_spring | 非线性弹簧 |
| Eq.48 | transmission_builder.py | compute_R | 滑轮传动比 |
| Eq.50 | — | — | 外展关节几何 |