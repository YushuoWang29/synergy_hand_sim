# synergy_hand_sim

**折纸手设计仿真工具包** — 从CAD图纸(DXF)到3D交互式折纸仿真的完整流程，支持 URDF 导出、Pinocchio 机器人学库集成，以及**增强自适应协同 (Augmented Adaptive Synergy)** 交互式仿真。

---

## 目录

- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [项目文件结构](#项目文件结构)
- [各文件详细说明](#各文件详细说明)
- [文件间调用关系](#文件间调用关系)
- [动态协同 (Dynamic Synergy)](#动态协同-dynamic-synergy)
- [增强自适应协同 (Augmented Adaptive Synergy)](#增强自适应协同-augmented-adaptive-synergy)
- [Origami CAD 编辑器](#origami-cad-编辑器-srcorigami_cad)
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
python scripts/export_urdf.py tests/test_hand.dxf --thickness 3.0
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
# 基础测试（7 项：基本功能、分离方向、增强组合、平滑过渡、OHD集成、工厂函数、MuJoCo）
python tests/test_dynamic_synergy.py

# 扩展测试（16 项：数学验证、边界情况、数值稳定性、蒙特卡罗随机测试）
python tests/test_dynamic_synergy_extended.py
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
│   ├── run_synergy_from_ohd.py       # 增强协同仿真：从 .ohd 加载 → MuJoCo 四滑块交互
│   ├── synergy_mujoco_demo.py        # 协同模型测试脚本
│   ├── diagnose_dxf.py               # 诊断工具：分析DXF文件的面片识别情况
│   └── run_cad.py                    # CAD 图形化编辑器启动脚本
│
├── src/                          # 核心源码
│   ├── __init__.py
│   │
│   ├── models/                   # 数据模型与算法
│   │   ├── origami_design.py         # 数据模型：折纸手设计的核心数据结构（含滑轮、腱绳、驱动器）
│   │   ├── origami_kinematics.py     # 运动学：3D正向运动学计算（支持环状结构）
│   │   ├── origami_parser.py         # 解析器：从DXF文件生成OrigamiHandDesign
│   │   ├── origami_to_urdf.py        # URDF导出：将设计导出为URDF+STL格式
│   │   └── transmission_builder.py   # 传动矩阵构建：R 和 R_f 矩阵，论文公式实现
│   │
│   ├── synergy/                  # 协同控制模块
│   │   ├── base_adaptive.py          # 基础自适应协同模型 (AdaptiveSynergyModel)
│   │   ├── augmented_adaptive.py     # 增强自适应协同模型 (AugmentedAdaptiveSynergyModel)
│   │   └── dynamic_synergy.py        # 动态协同模型 (DynamicSynergyModel) [新增]
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
│   ├── ohd_2/                        # ohd_2 设计（路由a）的URDF+STL
│   ├── ohd_3/                        # ohd_3 设计（路由b）的URDF+STL
│   ├── ohd_1/
│   ├── test_1/
│   ├── test_2/
│   ├── test_3/
│   ├── test_4/
│   └── ohd test/                     # .ohd 设计源文件
│
├── tests/                        # 测试文件
│   ├── test_adaptive_synergy.py      # 基础协同模型测试
│   ├── test_augmented_synergy.py     # 增强协同模型测试
│   ├── test_dynamic_synergy.py       # 动态协同模型测试 (7 variants)
│   └── test_dynamic_synergy_extended.py  # 动态协同扩展测试 (16 variants, 数学验证+数值稳定性)
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
| `Tendon` | 腱绳：id、滑轮/驱动器 ID 序列（正数=滑轮，负数=驱动器(A=-1,B=-2)） |
| `Actuator` | 驱动器控制器：id、名称、位移 |
| `OrigamiHandDesign` | 顶层容器：管理所有折痕线、面片、关节、滑轮、腱绳、驱动器，支持面片树构建（BFS）、完整性验证、摘要输出、JSON序列化 |

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

**功能**：从 `OrigamiHandDesign` 中自动提取传动矩阵 R 和 R_f，实现 Della Santina et al. (2018) 论文中的增强自适应协同模型。

**依赖**：`origami_design.py`

**包含的函数**：

| 函数 | 说明 |
|---|---|
| `get_joint_list(design)` | 从设计中提取所有关节，保留原始 fold line ID，确保 pulley→fold_line→joint 映射完整 |
| `compute_R(design)` | 计算基础传动矩阵 R ($k \times n$)，按论文公式(31)对每个关节求和所有关联滑轮半径 |
| `compute_Rf(design)` | 计算摩擦滑动传动矩阵 R_f ($m \times n$)，按论文公式(20)/(25)/(32)：<br>`R_f^T = -AR^T M^{-1} V_{max} e_v`<br>自动适应任意滑轮数量和路径顺序 |
| `build_synergy_model(design)` | 从设计构建完整的 `AugmentedAdaptiveSynergyModel` 实例 |

**关键特性**：
- R_f 计算通过动态构建 $(m+1) \times (m+1)$ 的速度耦合矩阵 M 和摩擦对角阵 V_max，**无需置换矩阵 P**，因为 `pulley_sequence` 已经自然编码了路径顺序信息
- **完全通用**：适应任意数量的肌腱、滑轮、关节，不限于三指案例

---

### 6. `src/synergy/augmented_adaptive.py` — 增强自适应协同模型

**功能**：实现 Augmented Adaptive Synergy 模型，在基础 adaptive synergy ($\sigma$) 之上添加肌腱滑动摩擦产生的第二个协同输入 $\sigma_f$。

**依赖**：`base_adaptive.py`

**类 `AugmentedAdaptiveSynergyModel`**：

| 方法 | 说明 |
|---|---|
| `__init__(n_joints, R, R_f, E_vec)` | 堆叠 R 和 R_f 为 R_aug，预计算 S_aug（主动协同矩阵）和 C_aug（被动柔顺矩阵） |
| `solve(sigma, sigma_f, J, f_ext)` | 计算关节角 q = S_aug @ [sigma, sigma_f]^T + C_aug @ J^T @ f_ext |

**核心公式**（论文 Eq. 23）：
$$
q = \begin{bmatrix} R \\ R_f \end{bmatrix}^+_E \begin{bmatrix} \sigma \\ \sigma_f \end{bmatrix} + P^\perp_{R,R_f} E^{-1} J^T f_{ext}
$$

---

### 7. `src/synergy/base_adaptive.py` — 基础自适应协同模型

**功能**：实现 Grioli et al. (2012) 的基础自适应协同求解器。

**类 `AdaptiveSynergyModel`**：

| 方法 | 说明 |
|---|---|
| `__init__(n_joints, R, E_vec)` | 预计算协同矩阵 S 和柔顺矩阵 C |
| `solve(sigma, J, f_ext)` | 求解 q = S @ sigma + C @ J^T @ f_ext |

---

### 8. `src/synergy/dynamic_synergy.py` — 动态协同模型 [新增]

**功能**：实现 Piazza et al. (2016) *"SoftHand Pro-D: Matching dynamic content of natural user commands with hand embodiment for enhanced prosthesis control"* 中的 **Dynamic Synergy** 框架，基于 descriptor linear system 理论，将手部运动分解为慢速协同和快速协同两个方向。

**依赖**：`base_adaptive.py`、`numpy`

**类 `DynamicSynergyModel`**：

| 方法 | 说明 |
|---|---|
| `__init__(n_joints, R, E_vec, T_damper)` | 计算慢速协同 S_s (Eq.6) 和快速协同 S_f (Eq.14) |
| `_compute_slow_synergy(R, E_inv)` | S_s = E^{-1} R^T (R E^{-1} R^T)^{-1} — 准静态平衡（阻尼力可忽略） |
| `_compute_fast_synergy(R, E, T)` | S_f = T_⊥^T · R^+_{E_y} — 阻尼主导的快速闭合方向 |
| `solve_slow(sigma)` | q_s = S_s @ sigma — 慢速协同（对应 power grasp） |
| `solve_fast(sigma)` | q_f = S_f @ sigma — 快速协同（对应 pinch grasp） |
| `solve(sigma, speed_factor)` | q = (1-α)·S_s·σ + α·S_f·σ — 平滑插值 |
| `solve_combined(sigma_dyn, sigma_aug, speed_factor, use_dynamic_on_f)` | 与 augmented synergy 联合求解，支持两种模式 |

**核心公式**（论文 Eq. 3-14）：
$$T^T C T \dot{q} + E q = R^T u$$
$$q_{slow} = S_s \sigma, \quad q_{fast} = S_f \sigma \quad (\text{瞬态}), \quad q(\alpha) = (1-\alpha) q_{slow} + \alpha q_{fast}$$

**关键特性**：
- **speed_factor** $\alpha \in [0,1]$：从慢速（准静态，power grasp）到快速（阻尼主导，pinch grasp）的平滑过渡
- **与 Augmented Adaptive Synergy 兼容**：统一 `solve_combined()` 接口，`use_dynamic_on_f` 控制是否对摩擦滑动也应用动态效应
- **工厂函数** `build_dynamic_synergy_model(design)`：从 `OrigamiHandDesign` 自动提取 R、E、T 并构建模型

---

### 9. `src/interactive/mujoco_simulator.py` — MuJoCo 仿真器

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

### 增强自适应协同仿真（核心新功能）

```
run_synergy_from_ohd.py
    │
    ├──▶ OrigamiHandDesign.load(ohd_path)
    │       └── 加载滑轮、腱绳、驱动器信息
    │
    ├──▶ build_urdf_to_synergy_mapping(urdf_path, design, ohd_path)
    │       ├── 最近邻空间匹配：URDF关节世界坐标 ↔ fold_line中点 ↔ synergy索引
    │       └── 返回 {urdf_joint_name: synergy_index} 映射字典
    │
    ├──▶ build_synergy_model(design)
    │       ├──▶ compute_R(design)           # 基础传动矩阵
    │       ├──▶ compute_Rf(design)          # 摩擦滑动传动矩阵
    │       └──▶ AugmentedAdaptiveSynergyModel(n, R, Rf, E)
    │
    └──▶ MuJoCoSimulator(urdf_path, mesh_dir,
                          synergy_callback, synergy_motor_names,
                          ctrl_range_deg, synergy_with_sigma_sliders)
            └── run()
                └── 每帧：ctrl滑块 → 互联动 → synergy_callback → qpos → mj_forward → sync
```

### 数据流向

```
DXF文件 ──▶ OrigamiParser ──▶ OrigamiHandDesign
(CAD图纸)       │                          │
                 │                          ├──▶ export_urdf() → URDF + STL
                 │                          │               │
                 │                          │               ├──▶ PinocchioSimulator
                 │                          │               └──▶ MuJoCoSimulator
                 │                          │
                 │                          ├──▶ transmission_builder.py
                 │                          │       ├── compute_R()    → R 矩阵
                 │                          │       └── compute_Rf()   → Rf 矩阵
                 │                          │
                 │                          └──▶ AugmentedAdaptiveSynergyModel
                 │                                    └── solve(sigma, sigma_f) → q
                 │
                 └──▶ OrigamiSimulator (自研运动学)

.ohd文件 ──▶ OrigamiHandDesign.load()
(JSON)         │
                ├──▶ build_urdf_to_synergy_mapping()
                ├──▶ build_synergy_model()
                └──▶ MuJoCoSimulator(synergy_callback)
                        └── run() → 实时交互仿真
```

---

## 动态协同 (Dynamic Synergy)

### 论文背景

实现 Piazza et al., *"SoftHand Pro-D: Matching Dynamic Content of Natural User Commands with Hand Embodiment for Enhanced Prosthesis Control"*, ICRA 2016。

### 核心思想

在欠驱动软体手中，利用**被动阻尼元件**实现多模式运动。在**慢速**驱动下，阻尼力可忽略，手沿**慢速协同方向** S_s 运动（准静态平衡）。在**快速**驱动下，阻尼力产生张力差，手沿**快速协同方向** S_f 运动（阻尼主导的瞬态响应）。当手接触物体后，从快速协同收敛到慢速协同，从而实现 power grasp 和 pinch grasp 的自然切换。

### 数学模型 (Descriptor Linear System)

力平衡方程（论文 Eq. 3）：
$$T^T C T \dot{q} + E q = R^T u$$

- **慢速闭合**（准静态，$\dot{q} \approx 0$）：阻尼力可忽略 → $Eq = R^T u$ → $S_s = E^{-1}R^T(RE^{-1}R^T)^{-1}$（Eq. 6）
- **快速闭合**（瞬态，$T^T C T \dot{q}$ 主导）：描述子系统分解，分为快/慢两个子系统（Eq. 9-10）
- **快速协同方向**：$S_f = T_\perp^T R^+_{E_y}$（Eq. 14），其中 $E_y = T_\perp E T_\perp^T$

### 物理实现

| 输入 | 物理意义 | 效果 |
|---|---|---|
| $\sigma$ (慢速驱动) | 准静态闭合 | 慢速协同：所有手指同步闭合（power grasp） |
| $\sigma$ (快速驱动) | 阻尼主导闭合 | 快速协同：拇指对指（pinch grasp） |
| $\alpha = \text{speed\_factor}$ | 插值系数 | 慢速→快速平滑过渡 |

### 使用示例

```python
from src.synergy.dynamic_synergy import DynamicSynergyModel
import numpy as np

# 构建模型
model = DynamicSynergyModel(
    n_joints=4,
    R=np.array([[7, 7, 7, 7]]),         # 传动矩阵
    E_vec=np.array([1, 1, 1, 1]),       # 关节刚度
    T_damper=np.array([[1, 0, 0, 0]])   # 阻尼器传动到 thumb 关节
)

# 慢速闭合 → power grasp (power grasp)
q_slow = model.solve_slow(np.array([1.0]))

# 快速闭合 → pinch grasp
q_fast = model.solve_fast(np.array([1.0]))

# 平滑过渡 (α=0.5)
q_mid = model.solve(np.array([1.0]), speed_factor=0.5)

# 与 augmented synergy 联合
from src.synergy.augmented_adaptive import AugmentedAdaptiveSynergyModel
aug_model = AugmentedAdaptiveSynergyModel(4, R, Rf, E_vec)
q = model.solve_combined(
    sigma_dyn=np.array([1.0]),
    sigma_aug=np.array([0.3]),
    speed_factor=0.7,
    use_dynamic_on_f=True
)
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

- 基础传动矩阵 **R 不变**：仅取决于滑轮半径分布
- 摩擦传动矩阵 **R_f 变化**：取决于腱绳穿过的滑轮顺序（路径 permutation 矩阵 P）
- 因此，通过改变路径可独立设计第二协同方向，无需修改机械结构

### 验证案例

`models/ohd test/` 目录包含具有不同路由的 3 指 6 关节设计：

| 文件 | 路由 | 预期行为 |
|---|---|---|
| `ohd_2.ohd` | 路由 (a) | 第二协同：左手（拇指+中指指根）主导打开 |
| `ohd_3.ohd` | 路由 (b) | 第二协同：中间手指主导打开 |

两种路由的 **R 相同**（均为 `[7,7,7,7,7,7]`），但 **R_f 不同**（值在关节间重排），验证了论文公式(31)-(36)的理论预测。

---

## 测试数据

| 文件 | 描述 |
|---|---|
| `tests/test_hand.dxf` | **主测试文件**：手掌(50×30) + 近端手指 + 远端手指，含谷折关节 |
| `tests/test_1.dxf` | 简单折纸设计 |
| `tests/test_2.dxf` | 多关节折纸设计（含DWG格式源文件） |
| `models/ohd test/ohd_1.ohd` | 单指设计 (测试用) |
| `models/ohd test/ohd_2.ohd` | 三指设计 - 路由 (a)，含滑轮、腱绳、驱动器 |
| `models/ohd test/ohd_3.ohd` | 三指设计 - 路由 (b)，路径顺序与 ohd_2 不同 |
| `models/ohd_2/ohd_2.urdf` | ohd_2 的导出 URDF+STL |
| `models/ohd_3/ohd_3.urdf` | ohd_3 的导出 URDF+STL |

---

## Origami CAD 编辑器 (`src/origami_cad/`)

**功能**：基于 PyQt5 的图形化 CAD 编辑器，用于可视化的折纸手设计。支持折线绘制、滑轮/腱绳/驱动器放置与编辑，保存为 `.ohd` 格式。

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

**关键关联**：滑轮必须放置在折痕线上（自动吸附），`attached_fold_line_id` 记录关联的折痕 ID，从而建立 pulley → fold_line → joint 的传动链。

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
#    - 保存为 .ohd 文件

# 2. 导出 URDF + STL
python scripts/export_urdf.py "models/ohd test/ohd_2.dxf" --thickness 3.0

# 3. 启动增强协同交互仿真
python scripts/run_synergy_from_ohd.py "models/ohd test/ohd_2.ohd"
#    - 右侧面板四个滑块：Motor A, Motor B, σ, σ_f
#    - 拖动滑块驱动手指运动
#    - σ 控制所有手指同步闭合/打开
#    - σ_f 控制手指间的相对运动
```

### 对比不同路由

```bash
# 路由 (a)
python scripts/run_synergy_from_ohd.py "models/ohd test/ohd_2.ohd"

# 路由 (b)
python scripts/run_synergy_from_ohd.py "models/ohd test/ohd_3.ohd"
```

两种路由的 σ（同向运动）行为相同，但 σ_f（差向运动）的手指分配不同，可直观验证论文理论。

---

## DXF 绘图规范

在CAD软件中绘制折纸手设计时需遵循以下规则：

1. **轮廓线**：使用黑色（颜色索引7）绘制所有面片的外边界
2. **折痕线**：
   - **峰折**（红色，颜色索引1）：关节在板的下表面，子面片折叠时向上抬起
   - **谷折**（蓝色，颜色索引5）：关节在板的上表面，子面片折叠时向下弯折
3. **共享边**：相邻面片的共用边只绘制一次，应为折痕色（红/蓝）
4. **线段类型**：支持 `LINE` 和 `LWPOLYLINE` 实体
5. **滑轮、腱绳、驱动器**：使用 CAD 编辑器 (`scripts/run_cad.py`) 在 `.ohd` 文件中添加，不在 DXF 中绘制
