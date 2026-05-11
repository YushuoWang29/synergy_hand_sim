# synergy_hand_sim

**折纸手设计仿真工具包** — 从CAD图纸(DXF)到3D交互式折纸仿真的完整流程，支持 URDF 导出、Pinocchio 机器人学库集成、**增强自适应协同 (Augmented Adaptive Synergy)** 和 **动态协同 (Dynamic Synergy)** 交互式仿真。

---

## 目录

- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [项目文件结构](#项目文件结构)
- [各文件详细说明](#各文件详细说明)
- [文件间调用关系](#文件间调用关系)
- [动态协同 (Dynamic Synergy)](#动态协同-dynamic-synergy)
- [增强自适应协同 (Augmented Adaptive Synergy)](#增强自适应协同-augmented-adaptive-synergy)
- [孔类腱绳传动 (Hole Transmission)](#孔类腱绳传动-hole-transmission)
- [摩擦模型分析 (Friction Analysis)](#摩擦模型分析-friction-analysis)
- [优化框架 (Optimization)](#优化框架-optimization)
- [Origami CAD 编辑器](#origami-cad-编辑器-srcorigami_cad)
- [测试文件](#测试文件)
- [用法示例](#用法示例)
- [DXF 绘图规范](#dxf-绘图规范)

---

## 环境配置

使用 Conda 环境管理依赖：

```bash
conda env create -f environment.yml
conda activate synergy_hand_sim
```

### 核心依赖

| 包 | 用途 |
|---|---|
| `numpy` | 数值计算与线性代数 |
| `ezdxf` | DXF文件解析 |
| `matplotlib` | 2D CAD视图绘制 (Qt5Agg后端) |
| `PyQt5` | GUI框架 |
| `meshcat-python` | 3D浏览器端可视化 |
| `pinocchio` | 机器人学库 (URDF加载与运动学) |
| `PySide6` | Qt6绑定 (与PyQt5可选) |
| `mujoco` | 物理引擎 (增强协同交互式仿真) |

---

## 快速开始

### 1. 交互式折纸仿真器 (自研运动学)

```bash
python scripts/run_origami_simulator.py tests/test_hand.dxf
```

### 2. URDF 导出

```bash
python scripts/export_urdf.py "models/ohd test/ohd_2.dxf" --thickness 3.0
```

### 3. Pinocchio URDF 仿真器

```bash
python scripts/run_pinocchio_simulator.py models/test_2/test_2.urdf
```

### 4. 增强自适应协同仿真 (从 .ohd 文件)

```bash
# 先导出 URDF
python scripts/export_urdf.py "models/ohd test/ohd_2.dxf" --thickness 3.0

# 再启动增强协同仿真
python scripts/run_synergy_from_ohd.py "models/ohd test/ohd_2.ohd"
```

### 5. MuJoCo 直接仿真

```bash
python scripts/run_mujoco_simulator.py models/ohd_2/ohd_2.urdf
```

### 6. 动态协同模型测试

```bash
conda run -n synergy_hand_sim python -m pytest tests/test_dynamic_synergy_extended.py -v

# 或直接运行
conda run -n synergy_hand_sim python tests/test_dynamic_synergy_extended.py
```

### 7. 优化设计

```bash
python scripts/run_optimization.py
```

---

## 项目文件结构

```
synergy_hand_sim/
│
├── environment.yml               # Conda 环境定义（完整依赖）
├── requirements.txt              # pip 依赖（ezdxf等）
├── README.md                     # 本文档
│
├── scripts/                      # 入口脚本
│   ├── run_origami_simulator.py      # 主入口：交互式折纸仿真器启动脚本
│   ├── run_pinocchio_simulator.py    # Pinocchio URDF 交互式仿真器启动脚本
│   ├── run_mujoco_simulator.py       # MuJoCo 直接 URDF 仿真器
│   ├── export_urdf.py                # URDF导出脚本：DXF → URDF + STL
│   ├── run_synergy_from_ohd.py       # 增强协同+动态协同仿真：从 .ohd 加载 → MuJoCo 四滑块交互
│   ├── synergy_mujoco_demo.py        # 协同模型测试脚本
│   ├── run_optimization.py           # 优化设计入口脚本（参数扫描 + 最优路由搜索）
│   ├── run_cad.py                    # CAD 图形化编辑器启动脚本
│   ├── debug_mapping.py              # URDF-joint到协同-关节的映射调试
│   └── diagnose_dxf.py               # 诊断工具：分析DXF文件的面片识别情况
│
├── src/                          # 核心源码
│   ├── __init__.py
│   │
│   ├── models/                   # 数据模型与算法
│   │   ├── origami_design.py         # 数据模型：折纸手设计的核心数据结构（含滑、腱绳、驱动器、孔、阻尼器）
│   │   ├── origami_kinematics.py     # 运动学：3D正向运动学计算（支持环状结构）
│   │   ├── origami_parser.py         # 解析器：从DXF文件生成OrigamiHandDesign
│   │   ├── origami_to_urdf.py        # URDF导出：将设计导出为URDF+STL格式
│   │   ├── transmission_builder.py   # 传动矩阵构建：R (Capstan模型), R_f (Capstan+Slack), T (阻尼器), 工厂函数
│   │   └── hole_transmission.py      # 孔类腱绳传动：力臂计算、孔对识别、等效传动比
│   │
│   ├── synergy/                  # 协同控制模块
│   │   ├── base_adaptive.py          # 基础自适应协同模型 (AdaptiveSynergyModel)
│   │   ├── augmented_adaptive.py     # 增强自适应协同模型 (AugmentedAdaptiveSynergyModel)
│   │   ├── dynamic_synergy.py        # 动态协同模型 (DynamicSynergyModel) — 阻尼平衡方法
│   │   └── friction_analysis.py      # 摩擦模型分析（Capstan衰减、死区、归一化）
│   │
│   ├── optimization/             # 计算设计优化框架
│   │   ├── design_space.py           # 可编辑设计变量与约束定义
│   │   ├── design_evaluator.py       # 参数 → OrigamiHandDesign → 协同矩阵
│   │   ├── objective_functions.py    # 设计目标（方向匹配、速度依赖）
│   │   ├── optimization_engine.py    # 搜索算法（DE、SHGO、Grid、Random）
│   │   └── tendon_routing.py         # 组合腱绳路径优化
│   │
│   ├── visualization/            # 3D可视化
│   │   └── origami_visualizer.py     # 3D可视化：MeshCat浏览器端渲染
│   │
│   ├── interactive/              # 交互式GUI
│   │   ├── cad_viewer.py             # 2D视图：matplotlib CAD图纸预览与交互
│   │   ├── mujoco_simulator.py       # MuJoCo 仿真器：支持协同回调模式与四滑块互联动
│   │   ├── origami_simulator.py      # 自研运动学仿真器：Qt GUI，整合2D+3D交互
│   │   └── pinocchio_simulator.py    # Pinocchio仿真器：基于URDF的交互式仿真
│   │
│   └── origami_cad/              # CAD 图形化编辑器
│       ├── cad_graphics_scene.py     # QGraphicsScene 核心
│       ├── cad_graphics_view.py      # QGraphicsView 交互
│       ├── main_window.py            # 主窗口
│       └── property_panel.py         # 属性面板
│
├── models/                       # 导出的URDF模型
│   ├── ohd_1...ohd_8/               # ohd 设计的 URDF+STL
│   ├── test_1...test_4/
│   └── ohd test/                     # .ohd 设计源文件
│
├── tests/                        # 测试文件
│   ├── test_adaptive_synergy.py          # 基础协同模型测试
│   ├── test_augmented_synergy.py         # 增强协同模型测试
│   ├── test_dynamic_synergy.py           # 动态协同模型测试
│   ├── test_dynamic_synergy_extended.py  # 动态协同扩展测试 (17项)
│   ├── test_final_ohd.py                 # 最终OHD设计验证
│   ├── test_ohd4_fix.py                  # OHD4修复测试
│   ├── test_full_ohd4.py                 # 完整OHD4测试
│   ├── test_friction_models.py           # 摩擦模型对比测试
│   ├── test_optimization_framework.py    # 优化框架测试
│   ├── test_pinocchio_urdf.py            # Pinocchio URDF加载验证
│   ├── test_origami_kinematics.py        # 运动学测试
│   ├── test_origami_design.py            # 设计数据结构测试
│   ├── test_cad_import.py                # CAD导入测试
│   ├── test_visualize_origami.py         # 可视化测试
│   ├── test_fold_clamping.py             # 折叠夹具测试
│   └── test_stiffness.py                 # 刚度测试
│
└── notebooks/                    # Jupyter notebooks
```

---

## 各文件详细说明

### 1. `src/models/origami_design.py` — 数据模型层

**功能**：定义折纸手设计的完整数据结构，是整个项目的数据基础。

**包含的类**：

| 类 | 说明 |
|---|---|
| `Point2D` | 二维点，支持距离计算、向量运算、哈希、等值比较 |
| `FoldType` | 枚举：`MOUNTAIN`（峰折）、`VALLEY`（谷折）、`OUTLINE`（轮廓） |
| `FoldLine` | 一条折痕线：记录起点、终点、类型、长度、方向向量、法向量、中点、刚度 |
| `OrigamiFace` | 一个面片：由顶点序列和边ID序列定义的封闭多边形，含面积计算、点包含判断 |
| `JointConnection` | 关节连接：记录两个面片通过某条折痕形成旋转关节的关系，含关节偏移方向 |
| `Pulley` | 滑轮：id、位置、半径、摩擦系数、关联折痕ID |
| `Hole` | 孔：用于孔类腱绳传动的路径节点（ID ≤ -100），含 plate_offset、摩擦系数、关联折痕ID |
| `Damper` | 阻尼器：用于 dynamic synergy，可连接多个关节（attached_fold_line_ids + transmission_ratios），含 damping_coefficient |
| `Tendon` | 腱绳：id、滑轮/驱动器/孔/阻尼器 ID 序列（正数=滑轮，负数=驱动器，≤-100=孔，≤-200=阻尼器）；提供 `has_holes`, `has_dampers`, `get_element_info()` 辅助方法 |
| `Actuator` | 驱动器控制器：id、名称、位移 |
| `OrigamiHandDesign` | 顶层容器：管理所有折痕线、面片、关节、滑轮、孔、阻尼器、腱绳、驱动器；支持面片树构建（BFS）、`build_topology()` 自动拓扑重建、验证、摘要输出、JSON序列化 |

**ID 约定**：
- 滑轮：`>= 0`
- 驱动器 A：`-1`，驱动器 B：`-2`
- 孔：`<= -100` 且 `> -200`
- 阻尼器：`<= -200`

**无外部依赖**（仅依赖 `numpy` 和 Python 标准库）。

---

### 2. `src/models/origami_kinematics.py` — 运动学层

**功能**：将2D折纸设计转换为3D空间中的运动学模型，计算面片在任意关节角度下的3D位姿。

**依赖**：`origami_design.py`

**包含的函数/类**：

| 函数/类 | 说明 |
|---|---|
| `compute_joint_frame_in_parent(design, joint)` | 计算关节轴在父面片局部坐标系中的3D位姿（轴上一点 + 方向），根据峰/谷折决定z偏移 |
| `rotation_around_axis(axis, angle)` | Rodrigues旋转公式：绕任意3D轴旋转 |
| `OrigamiForwardKinematics` | 正向运动学求解器（支持环状结构）：<br>• `__init__`：预计算所有关节的局部轴位姿和面片3D顶点，构建生成树，检测环边<br>• `forward_kinematics(joint_angles)`：沿生成树传播计算 + 迭代松弛法处理环约束<br>• `_build_spanning_tree`：BFS构建以根面片为根的生成树<br>• `_find_cycle_edges`：识别环状约束边<br>• `_resolve_cycles`：迭代约束松弛，将环闭合误差分配到路径上的关节<br>• `get_face_vertices_world(joint_angles)`：获取所有面片顶点在世界坐标中的位置 |

---

### 3. `src/models/origami_parser.py` — DXF解析层

**功能**：从CAD图纸（DXF格式）自动提取几何信息，生成 `OrigamiHandDesign`。

**依赖**：`origami_design.py`、`ezdxf`

**包含的类**：

| 类 | 说明 |
|---|---|
| `OrigamiParser` | DXF解析器，核心流程：<br>① `_extract_segments`：提取LINE和LWPOLYLINE实体，按颜色分类<br>② `_split_at_intersections`：在所有线段交点处强制分割，确保图为严格平面图<br>③ `_build_graph`：建立无向图（节点=唯一点，边=原子线段）<br>④ `_find_minimal_cycles`：用"最左转"算法找到所有最小环路（面片）<br>⑤ `_remove_outer_face`：去掉面积最大的外轮廓面<br>⑥ `_create_design`：组装 `OrigamiHandDesign`，过滤碎片面，创建关节连接，构建面片树 |

**颜色映射**（`_dxf_color_to_fold_type`）：
- 红/品红/橙 → 峰折（关节在板下表面）
- 蓝/青 → 谷折（关节在板上表面）
- 黑/深色 → 轮廓

---

### 4. `src/models/origami_to_urdf.py` — URDF导出层

**功能**：将折纸手设计导出为标准URDF+STL格式，可供Pinocchio、MuJoCo等物理引擎加载。

**依赖**：`origami_design.py`

**包含的函数**：

| 函数 | 说明 |
|---|---|
| `_write_face_stl(face, filepath, offset, thickness)` | 为一个面片生成二进制STL文件（上表面、下表面、侧面三角化） |
| `_write_binary_stl(filepath, triangles)` | 写入二进制STL格式 |
| `export_urdf(design, output_path, thickness)` | 主导出函数：<br>• 为每个面片创建 `<link>`（引用对应STL）<br>• 为每个关节创建 `<joint type="revolute">`<br>• 折痕对齐定位策略：子link原点放在折痕中点<br>• 峰折子面片整体下移以便关节轴对齐下表面<br>• 谷折子面片保持上表面齐平 |

---

### 5. `src/models/transmission_builder.py` — 传动矩阵构建

**功能**：从 `OrigamiHandDesign` 中自动提取传动矩阵 R、R_f、T（阻尼器），实现 Della Santina et al. (2018) 和 Piazza et al. (2016) 论文中的协同模型。

**依赖**：`origami_design.py`、`hole_transmission.py`

**包含的函数**：

| 函数 | 说明 |
|---|---|
| `get_joint_list(design)` | 从设计中提取所有关节，保留原始 fold line ID，确保 pulley→fold_line→joint 映射完整；当 joints 列表为空时自动从 fold lines 创建 |
| `compute_R(design)` → `compute_R_capstan(d, beta)` | **Capstan 指数衰减模型**：腱绳张力从两端向中间指数衰减，中间关节有效传动比小于两端 → U-shape 分布，物理上正确反映中指驱动最小、拇指/小指最大 |
| `compute_Rf(design)` | **Capstan + slack 侧钳制模型**：σ_f 模式下，Motor A 紧、Motor B 松，张力从 A 端指数衰减，低于静摩擦阈值的区段被"冻结"，产生不对称关节角分布。克服了原论文 Coulomb 模型在对称路径上 Rf=0 的问题 |
| `compute_damper_T(design)` | 阻尼器传动矩阵 T (n_dampers × n_joints)：将阻尼器到各关节的传动比映射到正确关节索引 |
| `build_synergy_model(design)` | 从设计构建完整的 `AugmentedAdaptiveSynergyModel` 实例 |
| `build_dynamic_synergy_model(design, use_augmented, speed_factor)` | 从设计构建完整的 `DynamicSynergyModel` 实例（含 R、R_f、E、T、C 阻尼系数） |

**核心参数**：
- `DEFAULT_BETA = 0.09`：Capstan 摩擦衰减系数（每元素）
- `DEFAULT_STICTION_THRESHOLD = 0.05`：静摩擦阈值（初始张力的 5%）

**关键特性**：
- 支持**孔类腱绳**: `compute_R`/`compute_Rf` 内部调用 `hole_transmission` 处理孔元素
- 多腱绳平均: 当有多条驱动腱绳时自动平均为单输入
- 物理正确的 U-shape 分布（Capstan 模型）

---

### 6. `src/models/hole_transmission.py` — 孔类腱绳传动

**功能**：实现孔类腱绳传动的精确数学建模，计算驱动力臂和等效传动比。

**依赖**：`origami_design.py`

**包含的函数**：

| 函数 | 说明 |
|---|---|
| `HolePairParams` | 数据类：描述跨过某折痕的一对孔的几何参数（d=半间距、h=plate_offset） |
| `compute_hole_lever_arm(d, h, q)` | 精确计算孔类腱绳的驱动力臂（无小量近似），适用于任意折叠角度 |
| `compute_hole_transmission_ratio(d, h, q=0)` | 在 q=0 处线性化的等效传动比 (= h)，用于协同框架 |
| `find_hole_pairs(design, tendon_id)` | 从腱绳路径中提取跨折痕的孔对，支持几何回退识别（当 attached_fold_line_id 未设置时自动通过路径-折痕交点判断） |
| `compute_hole_equivalent_R_row(design, tendon_id, joint_fold_map)` | 腱绳中孔元素对传动矩阵 R 的贡献行向量 |
| `compute_hole_torque(d, h, q, tension)` | 计算给定角度和张力的驱动力矩 |
| `plot_arm_vs_q(d, h, q_max)` | 力臂-角度关系可视化 |

**物理模型**: 滑轮 τ = r·T (常数传动比) vs 孔 τ = arm(q)·T (q依赖传动比)。在协同框架中使用线性化近似 arm(0) = h。

---

### 7. `src/synergy/augmented_adaptive.py` — 增强自适应协同模型

**功能**：实现 Augmented Adaptive Synergy 模型，在基础 adaptive synergy ($\sigma$) 之上添加肌腱滑动摩擦产生的第二个协同输入 $\sigma_f$。

**依赖**：`base_adaptive.py`

**类 `AugmentedAdaptiveSynergyModel`**：

| 方法 | 说明 |
|---|---|
| `__init__(n_joints, R, R_f, E_vec)` | 堆叠 R 和 R_f 为 R_aug，预计算 S_aug（主动协同矩阵）和 C_aug（被动柔顺矩阵） |
| `solve(sigma, sigma_f, J, f_ext)` | 计算关节角 q = S_aug @ [sigma, sigma_f]^T + C_aug @ J^T @ f_ext |

**核心公式**（论文 Eq. 23）：
$$ q = \begin{bmatrix} R \\ R_f \end{bmatrix}^+_E \begin{bmatrix} \sigma \\ \sigma_f \end{bmatrix} + P^\perp_{R,R_f} E^{-1} J^T f_{ext} $$

---

### 8. `src/synergy/base_adaptive.py` — 基础自适应协同模型

**功能**：实现 Grioli et al. (2012) 的基础自适应协同求解器。

**类 `AdaptiveSynergyModel`**：

| 方法 | 说明 |
|---|---|
| `__init__(n_joints, R, E_vec)` | 预计算协同矩阵 S 和柔顺矩阵 C |
| `solve(sigma, J, f_ext)` | 求解 q = S @ sigma + C @ J^T @ f_ext |

---

### 9. `src/synergy/dynamic_synergy.py` — 动态协同模型

**功能**：实现 Piazza et al. (2016) *"SoftHand Pro-D"* 中的 **Dynamic Synergy** 框架，通过阻尼器的被动阻尼实现速度依赖的协同方向切换。

**依赖**：`base_adaptive.py`、`numpy`

**类 `DynamicSynergyModel`**：

| 方法/属性 | 说明 |
|---|---|
| `__init__(n_joints, R, E_vec, T, C_diag)` | 构造函数。`T`: 阻尼器传动矩阵 (n_d×n)，`C_diag`: 阻尼系数向量。预计算慢速协同 S_s 和阻尼刚度项 |
| `_compute_damped_synergy(speed_factor)` | **核心方法**：计算阻尼平衡协同矩阵 S_eff(α)，使用**对角化** T^T C T（独立阻尼 per joint，避免耦合导致的符号翻转伪影） |
| `solve(sigma, speed_factor, J, f_ext)` | 主求解接口：q = S_eff(α) @ sigma + 外力柔顺项 |
| `solve_combined(sigma_dyn, sigma_f, speed_factor, use_dynamic_on_f)` | 与 augmented synergy 联合求解 |
| `slow_synergy` (property) | S_s：慢速协同矩阵（α=0 的阻尼平衡） |
| `fast_synergy` (property) | S_f：阻尼平衡快速协同矩阵（α=1），非描述子系统 S_f |
| `synergy_diff` (property) | S_f - S_s：动态调制方向 |
| `compute_R_for_fast_synergy(...)` | 静态方法：逆设计工具，从期望 S_f 反求 R |

**核心模型**（Piazza et al. Eq. 3 的阻尼平衡解释）：

$$ T^T C T \dot{q} + E q = R^T u $$

采用**阻尼平衡方法**而非论文原始描述子系统分解：
$$ E_{\text{eff}}(\alpha) = E + \alpha \cdot \text{diag}(T^T C T) $$
$$ S_{\text{eff}}(\alpha) = E_{\text{eff}}^{-1} R^T (R E_{\text{eff}}^{-1} R^T)^{-1} $$

**对角化 T^T C T 的物理原因**：当单个阻尼器连接多个关节（如 T=[1,1,1]），耦合的 T^T C T 会导致符号翻转伪影。对角化后每个关节独立获得附加刚度 ΔE_j = Σ_i C_i · T_ij²，物理上对应阻尼器腱绳的串联弹性。

**关键特性**：
- **speed_factor** α ∈ [0,1]：从慢速（准静态）到快速（阻尼主导）的平滑过渡
- 保持所有关节同向弯曲（无符号翻转）
- 约束条件 R·q = σ 始终满足
- 与 Augmented Adaptive Synergy 完全兼容（R_aug = vstack([R, R_f])）

---

### 10. `src/synergy/friction_analysis.py` — 摩擦模型分析

**功能**：提供三种物理摩擦模型的统一计算和分析接口。

**依赖**：`origami_design.py`、`transmission_builder.py`

包含的函数：
- `compute_capstan_R_Rf(design, beta, deadzone_frac)`: Capstan 衰减模型，同时计算 R 和 R_f
- `apply_dead_zone(Rf, threshold, deadzone_frac)`: 施加死区滤波
- `normalize_Rf_to_R(Rf, R)`: 归一化 R_f 量级到与 R 一致（防止 σ_f 模式主导 SVD）
- `compute_coulomb_Rf(design, normalize, deadzone_frac)`: 原论文 Coulomb 摩擦模型（含归一化修复）
- `analyze_R_Rf(design, use_capstan, beta, normalize, deadzone_frac)`: 综合分析，返回 R、R_f、协同方向及物理合理性检查

---

### 11. `src/optimization/` — 计算设计优化框架

**功能**：通过优化设计参数（如滑轮半径、阻尼器位置等），使设计达到期望的协同方向。

| 模块 | 说明 |
|---|---|
| `design_space.py` | 定义可编辑设计变量（DesignVariable, VariableType）和约束 DesignSpace |
| `design_evaluator.py` | 将参数化设计映射为 OrigamiHandDesign → 协同矩阵评估 |
| `objective_functions.py` | 设计目标：DirectionTarget（方向匹配）、SpeedDependentTarget（速度依赖）、CompositeObjective（组合） |
| `optimization_engine.py` | 搜索算法：DE（差分进化）、SHGO（单纯形全局）、Grid（网格）、Random（随机），统一 OptimizationResult 接口 |
| `tendon_routing.py` | 组合腱绳路径优化：枚举/搜索最优路径排列 |

**使用**：`python scripts/run_optimization.py`

---

### 12. `src/interactive/mujoco_simulator.py` — MuJoCo 仿真器

**功能**：基于 MuJoCo 的交互式 URDF 仿真器，支持**协同回调模式**和**四滑块互联动**。

**依赖**：`mujoco`

**类 `MuJoCoSimulator`**：

| 方法/特性 | 说明 |
|---|---|
| `__init__(urdf_path, mesh_dir, synergy_callback, synergy_motor_names, ctrl_range_deg, synergy_with_sigma_sliders)` | 将 URDF 转换为 MJCF XML，创建 dummy slider joints + kp=0 actuators 承载滑块控制 |
| `run()` | 主循环：读取滑块 ctrl → 互联动计算 → 调用 synergy_callback 求关节角 → 设置 qpos → `mj_forward` → `viewer.sync()` |
| 四滑块模式 | `synergy_with_sigma_sliders=True` 时创建 4 个滑块：Motor A, Motor B, σ, σ_f |
| 滑块互联动 | 检测用户拖动的滑块对（A/B 或 σ/σ_f），自动计算另一对的值，保持同步 |
| 直接模式 | `synergy_callback=None` 时直接复制 ctrl → qpos |

---

## 文件间调用关系

### 自研运动学仿真器

```
run_origami_simulator.py
    │
    ├──▶ OrigamiParser.parse(dxf_path)
    │       └──▶ OrigamiHandDesign
    │
    └──▶ OrigamiSimulator(design)
            │
            ├──▶ OrigamiForwardKinematics(design)
            │       ├── 构建生成树（_build_spanning_tree）
            │       ├── 检测环边（_find_cycle_edges）
            │       └── 迭代松弛（_resolve_cycles）
            │
            ├──▶ CADViewer(design)              [左侧2D面板]
            │
            ├──▶ OrigamiVisualizer()            [3D浏览器]
            │       └── display_hand(fk, joint_angles)
            │
            └──▶ 定时器驱动 _update_3d()
```

### Pinocchio URDF 仿真器

```
run_pinocchio_simulator.py
    │
    ├──▶ OrigamiParser.parse(dxf_path)     [仅用于2D视图]
    │
    └──▶ PinocchioSimulator(design, urdf_path, mesh_dir)
            ├──▶ pin.buildModelFromUrdf(urdf)
            ├──▶ CADViewer(design)
            └──▶ MeshcatVisualizer.display(q)
```

### 增强自适应协同仿真（从 .ohd 文件）

```
run_synergy_from_ohd.py
    │
    ├──▶ OrigamiHandDesign.load(ohd_path)
    │       └── 加载滑轮/孔/阻尼器/腱绳/驱动器信息
    │
    ├──▶ build_urdf_to_synergy_mapping(urdf_path, design, ohd_path)
    │       ├── 最近邻空间匹配：URDF关节世界坐标 ↔ fold_line中点 ↔ synergy索引
    │       └── 返回 {urdf_joint_name: synergy_index} 映射字典
    │
    ├──▶ build_dynamic_synergy_model(design)
    │       ├──▶ compute_R(design)              # Capstan R
    │       ├──▶ compute_Rf(design)             # Capstan+slack R_f
    │       ├──▶ compute_damper_T(design)       # 阻尼器 T
    │       └──▶ DynamicSynergyModel(n, R_aug, E, T, C)
    │
    └──▶ MuJoCoSimulator(urdf_path, mesh_dir,
                          synergy_callback, synergy_motor_names,
                          ctrl_range_deg, synergy_with_sigma_sliders)
            └── run()
                └── 每帧：ctrl滑块 → 互联动 → synergy_callback → qpos → mj_forward → sync
```

### 数据流向

```
DXF文件 ──▶ OrigamiParser ──▶ OrigamiHandDesign ──▶ build_dynamic_synergy_model()
(CAD图纸)       │                  │                     │
                 │                  │                     ├── compute_R_capstan() → R
                 │                  │                     ├── compute_Rf() → Rf
                 │                  │                     ├── compute_damper_T() → T
                 │                  │                     └── DynamicSynergyModel(R_aug, E, T, C)
                 │                  │
                 │                  ├──▶ export_urdf() → URDF + STL
                 │                  │          │
                 │                  │          ├──▶ PinocchioSimulator
                 │                  │          └──▶ MuJoCoSimulator
                 │                  │
                 │                  └──▶ AugmentedAdaptiveSynergyModel
                 │                            └── solve(sigma, sigma_f) → q
                 │
                 └──▶ OrigamiSimulator (自研运动学)

.ohd文件 ──▶ OrigamiHandDesign.load()
(JSON)         │
                ├──▶ build_urdf_to_synergy_mapping()
                ├──▶ build_dynamic_synergy_model()
                └──▶ MuJoCoSimulator(synergy_callback)
                        └── run() → 实时交互仿真

优化设计:
run_optimization.py
    │
    └──▶ DesignSpace → DesignEvaluator
            │
            ├──▶ compute_R(), compute_Rf(), compute_damper_T()
            └──▶ ObjectiveFunction(s) → OptimizationEngine
                    └── DE/SHGO/Grid/Random search
```

---

## 动态协同 (Dynamic Synergy)

### 论文背景

实现 Piazza et al., *"SoftHand Pro-D: Matching Dynamic Content of Natural User Commands with Hand Embodiment for Enhanced Prosthesis Control"*, ICRA 2016。

### 核心思想

在欠驱动软体手中，利用**被动阻尼元件**实现多模式运动。在**慢速**驱动下，阻尼力可忽略，手沿**慢速协同方向** S_s 运动（准静态平衡）。在**快速**驱动下，阻尼力产生等效附加刚度，手沿**阻尼平衡协同方向** S_f 运动。当手接触物体后，逐渐收敛到慢速协同平衡点，从而实现 power grasp 和 pinch grasp 的自然切换。

### 数学模型（阻尼平衡方法）

力平衡方程（论文 Eq. 3）：
$$T^T C T \dot{q} + E q = R^T u$$

**阻尼平衡解释** — 将阻尼力 T^T C T q̇ 建模为速度依赖的附加刚度：
$$E_{\text{eff}}(\alpha) = E + \alpha \cdot \text{diag}(T^T C T)$$
$$S_{\text{eff}}(\alpha) = E_{\text{eff}}^{-1} R^T (R E_{\text{eff}}^{-1} R^T)^{-1}$$

**关键区别（与论文原始描述子系统分解）**：
- 原始论文：S_f = T_⊥^T R^+_{E_y}，强制 T·S_f = 0（速率约束），多关节共享阻尼器时可能产生负角度
- 本实现：对角化 T^T C T，每个阻尼关节独立获得附加刚度，保持所有关节同向弯曲

### 物理实现

| 输入 | 物理意义 | 效果 |
|---|---|---|
| $\sigma$ (慢速驱动, α=0) | 准静态闭合 | 慢速协同：所有手指同步闭合（power grasp） |
| $\sigma$ (快速驱动, α=1) | 阻尼主导闭合 | 阻尼平衡：阻尼关节受限（pinch grasp） |
| $\alpha = \text{speed\_factor}$ | 插值系数 | 慢速→快速平滑过渡 |

### 使用示例

```python
from src.synergy.dynamic_synergy import DynamicSynergyModel
import numpy as np

# 构建模型
model = DynamicSynergyModel(
    n_joints=4,
    R=np.array([[7, 7, 7, 7]]),         # 传动矩阵 (1×4)
    E_vec=np.array([1, 1, 1, 1]),       # 关节刚度 (4,)
    T=np.array([[1, 0, 0, 0]]),         # 阻尼器传动矩阵 (1×4)
    C_diag=np.array([10.0])             # 阻尼系数 (1,)
)

# 慢速闭合 → power grasp
q_slow = model.solve(np.array([1.0]), speed_factor=0.0)

# 快速闭合 → 阻尼效果
q_fast = model.solve(np.array([1.0]), speed_factor=1.0)

# 平滑过渡 (α=0.5)
q_mid = model.solve(np.array([1.0]), speed_factor=0.5)

# 查看协同方向
print('Slow synergy:', model.slow_synergy)
print('Fast synergy:', model.fast_synergy)
print('Difference:', model.synergy_diff)

# 与 augmented synergy 联合（R_aug = vstack([R, R_f])）
R_aug = np.vstack([R, Rf])  # shape (2, n)
model_aug = DynamicSynergyModel(n, R_aug, E_vec, T, C_diag)
q = model_aug.solve_combined(
    sigma_dyn=np.array([1.0]),
    sigma_f=np.array([0.3]),
    speed_factor=0.7,
    use_dynamic_on_f=True
)
```

### .ohd 文件中定义阻尼器

```json
{
  "dampers": [
    {
      "id": -200,
      "position": {"x": 0, "y": 0},
      "attached_fold_line_ids": [25, 26, 27],
      "transmission_ratios": [1.0, 1.0, 1.0],
      "damping_coefficient": 10.0,
      "name": "thumb_damper"
    }
  ]
}
```

---

## 增强自适应协同 (Augmented Adaptive Synergy)

### 论文背景

实现 Della Santina et al., *"Toward Dexterous Manipulation With Augmented Adaptive Synergies: The Pisa/IIT SoftHand 2"*, IEEE TRO 2018。

### 核心思想

在腱绳驱动系统中，利用**库仑摩擦**作为第二驱动通道。当两台电机同向运动时，腱绳缩短，所有手指同时闭合（第一协同方向 $\sigma$）。当电机反向运动时，腱绳在滑轮间滑动，摩擦力引起张力阶梯分布，驱动部分手指打开、部分闭合（第二协同方向 $\sigma_f$）。

### 两个协同方向

| 输入 | 物理意义 | 效果 |
|---|---|---|
| $\sigma = (\theta_1 + \theta_2)/2$ | 同动量 | 所有手指同步闭合/打开 |
| $\sigma_f = (\theta_1 - \theta_2)/2$ | 差动量 | 手指间的相对运动 |

### 改变腱绳路径（Routing）的影响

- 基础传动矩阵 **R 不变**：仅取决于滑轮半径分布（实际使用 Capstan 模型）
- 摩擦传动矩阵 **R_f 变化**：取决于腱绳穿过的滑轮顺序（Capstan 衰减 + slack 钳制）
- 因此，通过改变路径可独立设计第二协同方向，无需修改机械结构

### 改进的 Rf 模型（与论文的不同）

本实现使用 **Capstan + slack 侧钳制**模型代替原论文的 Coulomb 摩擦模型：
- 物理原理：Motor A 拉紧时张力沿路径指数衰减，低于静摩擦阈值区段被"冻结"
- 解决了原 Coulomb 模型在对称路径上 Rf=0 的问题
- 产生物理正确的**不对称关节角分布**

### 验证案例

`models/ohd test/` 目录包含具有不同路由的 3 指 6 关节设计：

| 文件 | 路由 | 预期行为 |
|---|---|---|
| `ohd_2.ohd` | 路由 (a) | 第二协同：左手（拇指+中指指根）主导打开 |
| `ohd_3.ohd` | 路由 (b) | 第二协同：中间手指主导打开 |
| `ohd_8.ohd` | 五指串联 | 第二协同 + 阻尼器仅作用于拇指三个关节（25,26,27） |

---

## 孔类腱绳传动 (Hole Transmission)

### 物理模型

与滑轮（常数传动比 τ = r·T）不同，孔类腱绳的驱动力臂随折叠角度变化：
$$\tau(q) = T \cdot \text{arm}(q)$$

其中 arm(q) 是腱绳连线到折痕轴线的垂直距离，在协同框架中近似为 arm(0) = h（plate_offset）。

### 孔对识别算法

`find_hole_pairs` 函数通过以下步骤识别跨折痕的孔对：
1. 遍历腱绳路径中连续相邻的孔元素
2. 检查是否关联到同一折痕（attached_fold_line_id）
3. 如果未关联或未匹配，通过几何回退判断两孔连线穿越了哪条折痕
4. 验证两孔在折痕两侧（有符号距离异号）
5. 验证两孔在折痕方向上的投影对齐

---

## 摩擦模型分析 (Friction Analysis)

`src/synergy/friction_analysis.py` 提供三种摩擦模型的统一接口：

| 模型 | 描述 | 适用场景 |
|---|---|---|
| **Coulomb** | 原论文式：R_f^T = -AR^T M^{-1} V_max e_v | 原论文复现、对比 |
| **Capstan** | 指数衰减：T_A[k]=exp(-β·k), T_B[k]=exp(-β·(N-1-k)) | 物理正确、推荐 |
| **归一化组合** | Capstan Rf + 死区 + 量级匹配 | 实际模型、避免 SVD 主导 |

**关键诊断**：`analyze_R_Rf()` 返回物理合理性检查：
- R 的 U-shape 比率（中指 vs 两端）
- R_f 的极化方向（左正右负）
- 死区元素数量

---

## 优化框架 (Optimization)

`src/optimization/` 包提供计算设计优化：

### 设计空间 (DesignSpace)

可编辑变量包括：
- 滑轮半径
- 弹簧刚度
- 阻尼系数
- 阻尼器连接拓扑

### 设计评估器 (DesignEvaluator)

`DesignEvaluator`：参数 → OrigamiHandDesign → 协同矩阵 → 目标函数

### 目标函数 (ObjectiveFunction)

| 目标 | 说明 |
|---|---|
| `DirectionTarget` | 匹配期望协同方向（余弦相似度） |
| `SpeedDependentTarget` | 速度依赖协同切换 |
| `CompositeObjective` | 加权组合多个目标 |

### 优化引擎 (OptimizationEngine)

支持算法：DE（差分进化）、SHGO（单纯形全局）、Grid（网格搜索）、Random（随机搜索）

### 腱绳路径优化 (tendon_routing)

组合优化：枚举/搜索最优腱绳路径排列，最大化 σ_f 方向差异度。

---

## 测试数据

| 文件 | 描述 |
|---|---|
| `tests/test_hand.dxf` | **主测试文件**：手掌(50×30) + 近端手指 + 远端手指，含谷折关节 |
| `tests/test_1.dxf` ... `test_4.dxf` | 简单/多关节折纸设计 |
| `models/ohd test/ohd_1.ohd` | 单指设计 |
| `models/ohd test/ohd_2.ohd` | 三指设计 - 路由 (a) |
| `models/ohd test/ohd_3.ohd` | 三指设计 - 路由 (b) |
| `models/ohd test/ohd_4.ohd` .. `ohd_7.ohd` | 多指设计变体 |
| `models/ohd test/ohd_8.ohd` | 五指设计，含阻尼器（关联拇指折痕25,26,27） |
| `models/ohd_2/ohd_2.urdf` | ohd_2 的导出 URDF+STL |
| `models/ohd_3/ohd_3.urdf` | ohd_3 的导出 URDF+STL |
| `models/ohd_8/ohd_8.urdf` | ohd_8 的导出 URDF+STL（五指+阻尼器验证） |

---

## 测试文件

| 测试文件 | 说明 |
|---|---|
| `test_dynamic_synergy.py` | 基础动态协同测试 |
| `test_dynamic_synergy_extended.py` | **动态协同扩展测试（17项）**：数学验证、边界情况、ohd_8阻尼、蒙特卡罗 |
| `test_augmented_synergy.py` | 增强协同测试 |
| `test_adaptive_synergy.py` | 基础协同测试 |
| `test_final_ohd.py` | 最终 OHD 设计验证 |
| `test_ohd4_fix.py` | OHD4 修复测试 |
| `test_full_ohd4.py` | 完整 OHD4 系统测试 |
| `test_friction_models.py` | 摩擦模型对比测试（Coulomb vs Capstan） |
| `test_optimization_framework.py` | 优化框架测试 |
| `test_pinocchio_urdf.py` | URDF 加载验证 |
| `test_cad_import.py` | CAD 导入测试 |
| `test_origami_kinematics.py` | 运动学测试 |
| `test_origami_design.py` | 设计数据结构测试 |

---

## Origami CAD 编辑器 (`src/origami_cad/`)

**功能**：基于 PyQt5 的图形化 CAD 编辑器，用于可视化的折纸手设计。支持折线绘制、滑轮/腱绳/驱动器/孔/阻尼器放置与编辑，保存为 `.ohd` 格式。

### 启动

```bash
python scripts/run_cad.py
```

### 绘图模式

| 快捷键 | 工具栏按钮 | 模式 | 操作方式 |
|---|---|---|---|
| `L` | 轮廓线 | 绘制黑色轮廓线 | 输入 `x,y` 坐标或鼠标点击 → 再输入/点击终点 |
| `M` | 峰折 | 绘制红色峰折线 | 同上 |
| `V` | 谷折 | 绘制蓝色谷折线 | 同上 |
| `P` | 滑轮 | 放置灰色滑轮(半径5px) | 输入 `x,y` 坐标或鼠标点击，自动吸附到最近折痕线 |
| `T` | 腱绳 | 连接滑轮/驱动器的虚线路径 | 依次点击滑轮或驱动器 → 连续创建灰色虚线连接段 → **右键完成并保存** |
| `A` | 驱动器 | 放置驱动器点 | 输入 `x,y` 坐标或鼠标点击，默认蓝色(A型) |
| `Esc` | - | 退出当前模式 / 取消选择 | - |
| `Del` | 删除选中 | 删除选中的折线/滑轮/驱动器 | - |

### 属性编辑

选中图元后，右侧属性面板显示对应编辑控件：

- **折线**：显示 ID、类型（轮廓/峰折/谷折），可编辑**刚度**(N·m/rad)；轮廓线刚度控件禁用
- **滑轮**：显示 ID、关联折痕(若有)，可编辑**摩擦系数**(0~1.0) 和**半径**(2.0~10.0)
- **驱动器**：显示类型（A 蓝色 / B 红色），可通过下拉菜单切换类型，颜色实时更新
- **孔**：显示 ID、关联折痕、plate_offset、摩擦系数
- **阻尼器**：显示 ID、关联折痕列表、传动比列表、阻尼系数

**关键关联**：滑轮/孔/阻尼器必须关联到折痕线，建立 pulley/hole/damper → fold_line → joint 的传动链。

---

## 用法示例

### 完整工作流：CAD 设计 → 协同仿真

```bash
# 1. 打开 CAD 编辑器，设计折纸手
python scripts/run_cad.py
#    - 绘制轮廓线和折痕线
#    - 放置滑轮（自动吸附到折痕线）
#    - 穿腱绳（按路径顺序点击滑轮，右键完成）
#    - 放置驱动器（A/B 型）
#    - 可选：放置孔和阻尼器
#    - 保存为 .ohd 文件

# 2. 导出 URDF + STL
python scripts/export_urdf.py "models/ohd test/ohd_2.dxf" --thickness 3.0

# 3. 启动动态/增强协同交互仿真
python scripts/run_synergy_from_ohd.py "models/ohd test/ohd_2.ohd"
#    - 右侧面板四个滑块：Motor A, Motor B, σ, σ_f
#    - 拖动滑块驱动手指运动
#    - σ 控制所有手指同步闭合/打开
#    - σ_f 控制手指间的相对运动
#    - 可选 dynamic synergy 模式（提供 speed_factor 滑块）

# 4. 动态协同验证 (ohd_8: 五指+阻尼器)
python scripts/run_synergy_from_ohd.py "models/ohd test/ohd_8.ohd"

# 5. 优化设计
python scripts/run_optimization.py
```

---

## DXF 绘图规范

在CAD软件中绘制折纸手设计时需遵循以下规则：

1. **轮廓线**：使用黑色（颜色索引7）绘制所有面片的外边界
2. **折痕线**：
   - **峰折**（红色，颜色索引1）：关节在板的下表面，子面片折叠时向上抬起
   - **谷折**（蓝色，颜色索引5）：关节在板的上表面，子面片折叠时向下弯折
3. **共享边**：相邻面片的共用边只绘制一次，应为折痕色（红/蓝）
4. **线段类型**：支持 `LINE` 和 `LWPOLYLINE` 实体
5. **滑轮、孔、阻尼器、腱绳、驱动器**：使用 CAD 编辑器 (`scripts/run_cad.py`) 在 `.ohd` 文件中添加，不在 DXF 中绘制
