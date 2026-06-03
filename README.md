# synergy_hand_sim

**折纸手设计仿真工具包** — 从CAD图纸(DXF)到3D交互式折纸仿真的完整流程，支持 URDF 导出、Pinocchio 机器人学库集成、**状态依赖自适应协同 (SDAS, State-Dependent Adaptive Synergy)**、**增强自适应协同 (Augmented Adaptive Synergy)**、**动态协同 (Dynamic Synergy)** 以及 **三层递进力学数值仿真 (Phase 1-3)**。

---

## 目录

- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [项目文件结构](#项目文件结构)
- [各文件详细说明](#各文件详细说明)
- [文件间调用关系](#文件间调用关系)
- [状态依赖自适应协同 (SDAS)](#状态依赖自适应协同-sdas)
- [动态协同 (Dynamic Synergy)](#动态协同-dynamic-synergy)
- [增强自适应协同 (Augmented Adaptive Synergy)](#增强自适应协同-augmented-adaptive-synergy)
- [孔类腱绳传动 (Hole Transmission)](#孔类腱绳传动-hole-transmission)
- [摩擦模型分析 (Friction Analysis)](#摩擦模型分析-friction-analysis)
- [数值仿真框架 (Numerical Simulation)](#数值仿真框架-numerical-simulation)
- [优化框架 (Optimization)](#优化框架-optimization)
- [仿真可视化 (Simulation Visualization)](#仿真可视化-simulation-visualization)
- [Origami CAD 编辑器](#origami-cad-编辑器-srcorigami_cad)
- [测试文件](#测试文件)
- [用法示例](#用法示例)
- [DXF 绘图规范](#dxf-绘图规范)
- [参考文献](#参考文献)


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
| `matplotlib` | 2D CAD视图绘制 / 仿真结果可视化 |
| `PyQt5` | CAD编辑器GUI框架 |
| `meshcat-python` | 3D浏览器端可视化 |
| `pinocchio` | 机器人学库 (URDF加载与运动学) |
| `mujoco` | 物理引擎 (协同交互式仿真 + 轨迹回放) |
| `scipy` | ODE积分器 / 准静态非线性求解 |

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

### 4. SDAS 状态依赖自适应协同仿真 (默认模式)

```bash
# 先导出 URDF
python scripts/export_urdf.py "models/ohd test/ohd_2.dxf" --thickness 3.0

# 启动 SDAS 交互式仿真（四个滑块: Motor A, Motor B, σ, σ_f）
python scripts/run_synergy_from_ohd.py "models/ohd test/ohd_2.ohd"
```

### 5. 动态协同仿真 (--dynamic)

```bash
# 五指+阻尼器设计，速度滑块控制 fast/slow synergy 切换
python scripts/run_synergy_from_ohd.py "models/ohd test/ohd_8.ohd" --dynamic
```

### 6. 力学数值仿真 (三层递进架构)

```bash
# Phase 1 - 准静态力平衡
python scripts/run_numerical_simulation.py "models/ohd test/ohd_1.ohd" --phase 1

# Phase 2 - 完整动力学 ODE 积分
python scripts/run_numerical_simulation.py "models/ohd test/ohd_1.ohd" --phase 2

# 三段连续仿真 (σ=5, σ_f=0→-2→+2) 并可视化
python scripts/run_numerical_simulation.py "models/ohd test/ohd_2.ohd" --phase 2 --sigma 5 5 5 --sigma-f 0 -2 2 --t-per 0.3 --viz

# 三段仿真 + MuJoCo 轨迹回放
python scripts/run_numerical_simulation.py "models/ohd test/ohd_2.ohd" --phase 2 --sigma 5 5 5 --sigma-f 0 -2 2 --t-per 0.3 --mujoco

# 交互式 step-by-step 输入 sigma / sigma_f
python scripts/run_numerical_simulation.py "models/ohd test/ohd_1.ohd" --interactive --phase 1

# 三步分段动力学仿真专用脚本
python scripts/run_three_phase_demo.py --viz
python scripts/run_three_phase_demo.py --t-per 0.5 --save result.npz
```

### 7. 优化设计

```bash
python scripts/run_optimization.py
```

### 8. MuJoCo 直接 URDF 仿真

```bash
python scripts/run_mujoco_simulator.py models/ohd_2/ohd_2.urdf
```

---

## 项目文件结构

```
synergy_hand_sim/
│
├── environment.yml               # Conda 环境定义（完整依赖）
├── requirements.txt              # pip 依赖
├── README.md                     # 本文档
│
├── scripts/                      # 入口脚本
│   ├── run_origami_simulator.py      # 主入口：交互式折纸仿真器启动脚本
│   ├── run_pinocchio_simulator.py    # Pinocchio URDF 交互式仿真器启动脚本
│   ├── run_mujoco_simulator.py       # MuJoCo 直接 URDF 仿真器
│   ├── export_urdf.py                # URDF导出脚本：DXF → URDF + STL
│   ├── run_synergy_from_ohd.py       # 协同交互仿真（SDAS/动态协同）：从 .ohd 加载 → MuJoCo 交互
│   ├── run_numerical_simulation.py   # 力学数值仿真入口：三层递进架构 (Phase 1-3)
│   ├── run_three_phase_demo.py       # 三步分段动力学仿真示例
│   ├── synergy_mujoco_demo.py        # 协同模型测试脚本
│   ├── run_optimization.py           # 优化设计入口脚本（参数扫描 + 最优路由搜索）
│   ├── run_cad.py                    # CAD 图形化编辑器启动脚本
│   ├── debug_mapping.py              # URDF-joint到协同-关节的映射调试
│   ├── debug_simulation_issues.py    # 仿真问题诊断工具
│   ├── debug_ohd7_qmatrix.py         # OHD7 Q矩阵调试
│   └── diagnose_dxf.py               # 诊断工具：分析DXF文件的面片识别情况
│
├── src/                          # 核心源码
│   ├── __init__.py
│   │
│   ├── models/                   # 数据模型与算法
│   │   ├── origami_design.py         # 数据模型：折纸手设计的核心数据结构（含面、腱绳、驱动器、孔、阻尼器）
│   │   ├── origami_kinematics.py     # 运动学：3D正向运动学计算（支持环状结构）
│   │   ├── origami_parser.py         # 解析器：从DXF文件生成OrigamiHandDesign
│   │   ├── origami_to_urdf.py        # URDF导出：将设计导出为URDF+STL格式
│   │   ├── transmission_builder.py   # 传动矩阵构建：R (几何/Capstan), R_A/R_B (单侧), R_f, T (阻尼器), SDAS模型工厂
│   │   └── hole_transmission.py      # 孔类腱绳传动：力臂计算、孔对识别、等效传动比
│   │
│   ├── synergy/                  # 协同控制模块
│   │   ├── base_adaptive.py          # 基础自适应协同模型 (AdaptiveSynergyModel)
│   │   ├── augmented_adaptive.py     # 增强自适应协同模型 (AugmentedAdaptiveSynergyModel)
│   │   ├── sdas_model.py             # **状态依赖自适应协同 (SDASModel)** — 最新模型，用R_A/R_B替换R/R_f
│   │   ├── dynamic_synergy.py        # 动态协同模型 (DynamicSynergyModel) — 阻尼平衡方法
│   │   └── friction_analysis.py      # 摩擦模型分析（Capstan衰减、死区、归一化）
│   │
│   ├── simulation/               # **力学数值仿真框架 (三层递进架构)**
│   │   ├── __init__.py              # 包入口，统一导出
│   │   ├── config.py                # SimulationConfig：仿真参数配置
│   │   ├── rigid_body.py            # LinkInertia, RigidBodySystem：刚体惯量参数估计
│   │   ├── transmission_force.py    # 腱绳传动力矩阵：M矩阵、R̄、Q矩阵、粘滞阻尼、静摩擦
│   │   ├── friction_models.py       # HaywardArmstrongFriction, CapstanTensionDistribution
│   │   ├── dynamics.py              # DynamicsAssembler：动力学方程装配
│   │   ├── integrator.py            # ODEState, DynamicsODE, Integrator（RK4/Euler/scipy）
│   │   ├── quasi_static.py          # QuasiStaticSolver：准静态力平衡求解器（Phase 1）
│   │   ├── simulator.py             # HandSimulator：顶层编排引擎，SimulationTrajectory
│   │   ├── io.py                    # SimulationWriter/Reader：结果序列化（.npz）
│   │   ├── mujoco_physics.py        # MuJoCo 物理引擎集成接口
│   │   ├── visualization.py         # 仿真过程实时可视化
│   │   └── tests/                   # 测试文件
│   │       ├── test_all.py              # 30项综合测试
│   │       └── test_integration_real.py # 真实 .ohd 集成验证
│   │
│   ├── optimization/             # 计算设计优化框架
│   │   ├── design_space.py           # 可编辑设计变量与约束定义
│   │   ├── design_evaluator.py       # 参数 → OrigamiHandDesign → 协同矩阵
│   │   ├── objective_functions.py    # 设计目标（方向匹配、速度依赖）
│   │   ├── optimization_engine.py    # 搜索算法（DE、SHGO、Grid、Random）
│   │   └── tendon_routing.py         # 组合腱绳路径优化
│   │
│   ├── visualization/            # 3D 可视化 & 仿真结果可视化
│   │   ├── origami_visualizer.py     # 3D可视化：MeshCat浏览器端渲染
│   │   ├── simulation_visualizer.py  # **力学数值仿真可视化**：多面板看板 (q(t), 能量, 相平面, 力矩分量)
│   │   └── mujoco_animator.py        # MuJoCo 动画工具
│   │
│   ├── interactive/              # 交互式GUI
│   │   ├── cad_viewer.py             # 2D视图：matplotlib CAD图纸预览与交互
│   │   ├── mujoco_simulator.py       # MuJoCo 仿真器：支持协同回调模式与滑块互联动
│   │   ├── origami_simulator.py      # 自研运动学仿真器：Qt GUI，整合2D+3D交互
│   │   └── pinocchio_simulator.py    # Pinocchio仿真器：基于URDF的交互式仿真
│   │
│   └── origami_cad/              # CAD 图形化编辑器
│       ├── cad_graphics_scene.py     # QGraphicsScene 核心
│       ├── cad_graphics_view.py      # QGraphicsView 交互
│       ├── main_window.py            # 主窗口
│       └── property_panel.py         # 属性面板
│
├── models/                       # 导出的URDF模型和.ohd设计源文件
│   ├── ohd_1...ohd_8/               # ohd 设计的 URDF+STL
│   ├── test_1...test_4/
│   ├── ohd test/                     # .ohd 设计源文件
│   └── index.md
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
├── docs/                         # 理论文档
│   ├── algorithm.md                  # run_synergy_from_ohd.py 完整计算链路文档
│   ├── SDAS.md                       # 状态依赖自适应协同模型理论
│   ├── polarized_adaptive_synergy.md  # 极化自适应协同模型理论
│   ├── friction_model_analysis.md    # 摩擦模型分析
│   └── software_architecture.pdf     # 软件架构文档
│
└── reference/                     # 参考文献
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
| `clamp_fold_angle(angle, fold_type)` | 关节角度限位：谷折 [0, π]，山折 [-π, 0] |
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

**功能**：从 `OrigamiHandDesign` 中自动提取传动矩阵 R、R_A/R_B、R_f、T（阻尼器），实现经典自适应协同、增强自适应协同、SDAS 等协同模型。

**依赖**：`origami_design.py`、`hole_transmission.py`

**包含的函数**：

| 函数 | 说明 |
|---|---|
| `get_joint_list(design)` | 从设计中提取所有关节，保留原始 fold line ID，确保 pulley→fold_line→joint 映射完整 |
| `compute_R(design)` | 基础传动矩阵 (委托给 `compute_R_capstan`) |
| `compute_R_capstan(design, beta)` | **Capstan 指数衰减模型**：腱绳张力从两端向中间指数衰减，中间关节有效传动比小于两端 → U-shape 分布 |
| `compute_R_one_sided(design, beta)` | **单侧 Capstan 衰减模型**：分别计算从 Motor A 出发的 R_A 和从 Motor B 出发的 R_B，用于 SDAS 模型 |
| `compute_Rf(design, beta, stiction_threshold)` | **Capstan + slack 侧钳制模型**：σ_f 模式下 Motor A 紧、Motor B 松，张力从 A 端指数衰减，低于静摩擦阈值的区段被"冻结" |
| `compute_damper_T(design)` | 阻尼器传动矩阵 T (n_dampers × n_joints)：将阻尼器到各关节的传动比映射到正确关节索引 |
| `build_synergy_model(design)` | 从设计构建完整的 `AugmentedAdaptiveSynergyModel` 实例 |
| `build_sdas_model(design, beta)` | **从设计构建 SDAS 模型**：基于 R_A/R_B 和 Schur 补公式，取消 Rf/静摩擦冻结概念 |
| `build_one_sided_synergy_model(design, beta)` | **[已废弃]** 请使用 `build_sdas_model()` 替代 |
| `solve_one_sided(theta_A, theta_B, model_info)` | **[已废弃]** 请直接使用 `SDASModel.solve_motors()` 替代 |
| `build_dynamic_synergy_model(design, use_augmented, speed_factor)` | 从设计构建完整的 `DynamicSynergyModel` 实例（含 R、R_f、E、T、C 阻尼系数） |

**核心参数**：
- `DEFAULT_BETA = 0.09`：Capstan 摩擦衰减系数（每元素）
- `DEFAULT_STICTION_THRESHOLD = 0.05`：静摩擦阈值（初始张力的 5%）

**关键特性**：
- 支持**孔类腱绳**: `compute_R`/`compute_Rf` 内部调用 `hole_transmission` 处理孔元素
- 多腱绳平均: 当有多条驱动腱绳时自动平均为单输入
- 物理正确的 U-shape 分布（Capstan 模型）
- **SDAS 模型**: 用 R_A/R_B 替换旧的 R/R_f，Schur 补双约束求解

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
| `find_hole_pairs(design, tendon_id)` | 从腱绳路径中提取跨折痕的孔对，支持几何回退识别 |
| `compute_hole_equivalent_R_row(design, tendon_id, joint_fold_map)` | 腱绳中孔元素对传动矩阵 R 的贡献行向量 |
| `compute_hole_torque(d, h, q, tension)` | 计算给定角度和张力的驱动力矩 |
| `plot_arm_vs_q(d, h, q_max)` | 力臂-角度关系可视化 |

**物理模型**: 滑轮 τ = r·T (常数传动比) vs 孔 τ = arm(q)·T (q依赖传动比)。在协同框架中使用线性化近似 arm(0) = h。

---

### 7. `src/synergy/sdas_model.py` — 状态依赖自适应协同模型 (SDAS)

**功能**：实现**状态依赖自适应协同 (SDAS)** 模型，使用精确约束选择策略代替旧的 Rf/静摩擦冻结概念。

**依赖**：`numpy`

**类 `SDASModel`**：

| 方法 | 说明 |
|---|---|
| `__init__(n_joints, R_A, R_B, E_vec)` | 预计算单向公式 (S_A_paper, S_B_paper) 和 Schur 补公式 (S_A_schur, S_B_schur) |
| `solve_motors(theta_A, theta_B, J, f_ext)` | **主求解接口**：根据电机位移自动选择约束公式：双马达同时拉动→Schur补；单马达→单向公式(S_A_paper或S_B_paper)；松弛→零输出 |
| `solve_synergies(sigma, sigma_f, J, f_ext)` | 将σ/σ_f转化为θ_A/θ_B后委托给`solve_motors` |

**核心公式**（docs/SDAS.md Eq. 2.46-2.49）：
- **单向公式**（单马达）：q = S_A_paper·θ_A 或 q = S_B_paper·θ_B（仅满足单侧约束）
- **Schur 补公式**（双马达）：q = S_A_schur·θ_A + S_B_schur·θ_B + C_schur·J^T·f_ext（双约束精确满足，交叉耦合解耦）

**核心修复**：
- 取消了旧的 Rf / 静摩擦冻结概念
- 用 **R_A / R_B 的自然非对称性**替代 R_f
- 单马达/双马达使用不同的物理公式（状态依赖的约束选择）
- Slack 钳位：负电机位移→0（腱绳不能受推）
- σ/σ_f 模式与 θ_A/θ_B 模式完全等价（内部自动转换）


---

### 8. `src/synergy/base_adaptive.py` — 基础自适应协同模型

**功能**：实现 Grioli et al. (2012) 的基础自适应协同求解器。

**类 `AdaptiveSynergyModel`**：

| 方法 | 说明 |
|---|---|
| `__init__(n_joints, R, E_vec)` | 预计算协同矩阵 S 和柔顺矩阵 C |
| `solve(sigma, J, f_ext)` | 求解 q = S @ sigma + C @ J^T @ f_ext |

---

### 9. `src/synergy/augmented_adaptive.py` — 增强自适应协同模型

**功能**：实现 Augmented Adaptive Synergy 模型，在基础 adaptive synergy (σ) 之上添加肌腱滑动摩擦产生的第二个协同输入 σ_f。

**依赖**：`base_adaptive.py`

**类 `AugmentedAdaptiveSynergyModel`**：

| 方法 | 说明 |
|---|---|
| `__init__(n_joints, R, R_f, E_vec)` | 堆叠 R 和 R_f 为 R_aug，预计算 S_aug（主动协同矩阵）和 C_aug（被动柔顺矩阵） |
| `solve(sigma, sigma_f, J, f_ext)` | 计算关节角 q = S_aug @ [sigma, sigma_f]^T + C_aug @ J^T @ f_ext |

**核心公式**（论文 Eq. 23）：
$$ q = \begin{bmatrix} R \\ R_f \end{bmatrix}^+_E \begin{bmatrix} \sigma \\ \sigma_f \end{bmatrix} + P^\perp_{R,R_f} E^{-1} J^T f_{ext} $$

---

### 10. `src/synergy/dynamic_synergy.py` — 动态协同模型

**功能**：实现 Piazza et al. (2016) *"SoftHand Pro-D"* 中的 **Dynamic Synergy** 框架，通过阻尼器的被动阻尼实现速度依赖的协同方向切换。

**依赖**：`base_adaptive.py`、`numpy`

**类 `DynamicSynergyModel`**：

| 方法/属性 | 说明 |
|---|---|
| `__init__(n_joints, R, E_vec, T, C_diag)` | 构造函数。`T`: 阻尼器传动矩阵 (n_d×n)，`C_diag`: 阻尼系数向量。预计算慢速协同 S_s 和阻尼刚度项 |
| `_compute_damped_synergy(speed_factor)` | **核心方法**：计算阻尼平衡协同矩阵 S_eff(α)，使用**对角化** T^T C T |
| `solve(sigma, speed_factor, J, f_ext)` | 主求解接口：q = S_eff(α) @ sigma + 外力柔顺项 |
| `solve_combined(sigma_dyn, sigma_f, speed_factor, use_dynamic_on_f)` | 与 augmented synergy 联合求解 |
| `slow_synergy` (property) | S_s：慢速协同矩阵（α=0 的阻尼平衡） |
| `fast_synergy` (property) | S_f：阻尼平衡快速协同矩阵（α=1） |
| `synergy_diff` (property) | S_f - S_s：动态调制方向 |
| `compute_R_for_fast_synergy(...)` | 静态方法：逆设计工具，从期望 S_f 反求 R |

**核心模型**：

$$ T^T C T \dot{q} + E q = R^T u $$

采用**阻尼平衡方法**：
$$ E_{\text{eff}}(\alpha) = E + \alpha \cdot \text{diag}(T^T C T) $$
$$ S_{\text{eff}}(\alpha) = E_{\text{eff}}^{-1} R^T (R E_{\text{eff}}^{-1} R^T)^{-1} $$

**关键特性**：
- **speed_factor** α ∈ [0,1]：从慢速（准静态）到快速（阻尼主导）的平滑过渡
- 保持所有关节同向弯曲（无符号翻转）
- 约束条件 R·q = σ 始终满足
- 与 Augmented Adaptive Synergy 完全兼容（R_aug = vstack([R, R_f])）

---

### 11. `src/synergy/friction_analysis.py` — 摩擦模型分析

**功能**：提供三种物理摩擦模型的统一计算和分析接口。

**依赖**：`origami_design.py`、`transmission_builder.py`

包含的函数：
- `compute_capstan_R_Rf(design, beta, deadzone_frac)`: Capstan 衰减模型，同时计算 R 和 R_f
- `apply_dead_zone(Rf, threshold, deadzone_frac)`: 施加死区滤波
- `normalize_Rf_to_R(Rf, R)`: 归一化 R_f 量级到与 R 一致
- `compute_coulomb_Rf(design, normalize, deadzone_frac)`: 原论文 Coulomb 摩擦模型
- `analyze_R_Rf(design, use_capstan, beta, normalize, deadzone_frac)`: 综合分析

---

### 12. `src/simulation/` 力学数值仿真框架

实现从纯几何仿真到完整力学仿真的三层递进架构，基于 Della Santina et al. (2018) TRO Eq.43-44 的动力学方程。

#### 三层递进架构

| 阶段 | 描述 | 核心方程 | 数值方法 |
|---|---|---|---|
| **Phase 1** 准静态力平衡 | f(q) = 0，无时间演化 | Kq = τ_input | `scipy.optimize.root` (hybr) |
| **Phase 2** 完整动力学 | ODE 积分，含惯性与阻尼 | M(q)q̈ + B q̇ + Kq = τ_input | RK4 / scipy RK45 |
| **Phase 3** 动力学+接触 | + Hencky 接触模型 | M q̈ = τ_ext - B q̇ - Kq + J^T f_c | RK4 + 接触力 |

#### 模块总览

| 模块 | 类/函数 | 说明 |
|---|---|---|
| `config.py` | `SimulationConfig` | 仿真参数：dt、t_end、phase、method、σ(t)/σ_f(t)/τ_M(t) 表达式控制输入 |
| `rigid_body.py` | `LinkInertia`, `RigidBodySystem`, `estimate_link_inertia()` | 刚体惯性参数，从面片几何估计质量/惯量 |
| `transmission_force.py` | `build_M_matrix()`, `build_R_bar_matrix()`, `build_viscous_damping_matrix()`, `build_static_friction_matrix()`, `build_N_matrix()`, `compute_Q_matrix()` | 腱绳传动矩阵构建与 Q(q) 三列向量 (Q_τM, Q_s, Q_sdot) |
| `friction_models.py` | `HaywardArmstrongFriction`, `CapstanTensionDistribution` | Hayward-Armstrong 连续静摩擦模型，Capstan 指数衰减 |
| `dynamics.py` | `DynamicsAssembler` | 动力学方程装配器：compute_B(q), compute_W(q,q̇), compute_Γ(q), compute_input_torque() |
| `integrator.py` | `ODEState`, `DynamicsODE`, `Integrator` | 数值积分器（RK4, Euler, scipy RK45 包装），状态向量 = [q, q̇, z, s] |
| `quasi_static.py` | `QuasiStaticSolver`, `QuasiStaticResult` | 准静态力平衡非线性求解器 |
| `simulator.py` | `HandSimulator`, `SimulationTrajectory` | 顶层仿真引擎：编排 Phase 1-3，支持分段控制输入 (σ/σ_f/τ_M 时间函数) |
| `io.py` | `SimulationWriter`, `SimulationReader` | 仿真结果序列化（.npz）+ 后处理加载 |
| `mujoco_physics.py` | MuJoCo 物理引擎接口 | 在 MuJoCo 中回放仿真轨迹 |
| `visualization.py` | 仿真实时可视化 | 使用 matplotlib 实时绘制 q(t) |

#### 动力学方程（Della Santina et al. 2018, Eq.43-44）

$$B(q) \ddot{q} + (W + C(q, \dot{q})) \dot{q} + K q = \underbrace{Q(q) u}_{\text{肌肉驱动力}} + \underbrace{J(q)^T f_{\text{ext}}}_{\text{接触力}} + \underbrace{\Gamma(q, \dot{q}, u)}_{\text{摩擦}}$$

#### 控制输入 u(t) = [τ_M · σ, σ, σ_f]

仿真支持**时间相关的表达式控制输入**：
- `cfg.sigma_func`：σ(t) 表达式，如 `"min(t*2, 1.0)"`（斜坡激活）
- `cfg.sigma_f_func`：σ_f(t) 表达式，如分段常数 `"(t<0.3)*0 + (t>=0.3 and t<0.6)*(-2) + (t>=0.6)*(2)"`
- `cfg.tau_M_func`：τ_M(t) 电机拉力（默认 5.0 常量）

#### 使用示例

```python
from src.simulation.config import SimulationConfig
from src.simulation.simulator import HandSimulator
from src.models.origami_design import OrigamiHandDesign

# 加载设计
design = OrigamiHandDesign.load('models/ohd test/ohd_1.ohd')

# 三段连续仿真 (σ=5, σ_f=0→-2→+2)
cfg = SimulationConfig(dt=1e-4, t_end=0.9, phase=2)
cfg.sigma_func = "5.0"
cfg.sigma_f_func = "(t<0.3)*0 + (t>=0.3 and t<0.6)*(-2) + (t>=0.6)*(2)"
cfg.tau_M_func = "5.0"

sim = HandSimulator(design, cfg)
traj = sim.run(timeout=120.0)
```

#### 测试

运行全套 30 项测试：

```bash
conda run -n synergy_hand_sim python -m pytest src/simulation/tests/test_all.py -v
```

---

### 13. `src/optimization/` — 计算设计优化框架

**功能**：通过优化设计参数（如滑轮半径、阻尼器位置等），使设计达到期望的协同方向。

| 模块 | 说明 |
|---|---|
| `design_space.py` | 定义可编辑设计变量（DesignVariable, VariableType）和约束 DesignSpace |
| `design_evaluator.py` | 将参数化设计映射为 OrigamiHandDesign → 协同矩阵评估 |
| `objective_functions.py` | 设计目标：DirectionTarget（方向匹配）、SpeedDependentTarget（速度依赖）、CompositeObjective（组合） |
| `optimization_engine.py` | 搜索算法：DE（差分进化）、SHGO（单纯形全局）、Grid（网格）、Random（随机） |
| `tendon_routing.py` | 组合腱绳路径优化：枚举/搜索最优路径排列 |

**使用**：`python scripts/run_optimization.py`

---

### 14. `src/interactive/mujoco_simulator.py` — MuJoCo 交互式仿真器

**功能**：基于 MuJoCo 的交互式 URDF 仿真器，支持**协同回调模式**和**滑块互联动**。

**依赖**：`mujoco`

**类 `MuJoCoSimulator`**：

| 方法/特性 | 说明 |
|---|---|
| `__init__(urdf_path, mesh_dir, synergy_callback, synergy_motor_names, ctrl_range_deg, synergy_with_sigma_sliders, synergy_with_speed_slider)` | 将 URDF 转换为 MJCF XML，创建滑块 actuators |
| `run()` | 主循环：读取滑块 ctrl → 调用 synergy_callback 求关节角 → 设置 qpos → `mj_forward` → `viewer.sync()` |
| 五滑块模式 | SDAS 默认：Motor A, Motor B, σ, σ_f；动态模式：+ Speed (rad/s) |
| 滑块互联动 | 检测用户拖动的滑块对（A/B 或 σ/σ_f），自动计算另一对的值 |
| 直接模式 | `synergy_callback=None` 时直接复制 ctrl → qpos |

---

### 15. `src/visualization/simulation_visualizer.py` — 仿真结果可视化

**功能**：力学数值仿真结果的多面板看板可视化工具。

**类 `SimulationVisualizer`** / `TrajectoryPlotter`：

| 方法 | 说明 |
|---|---|
| `plot_joint_positions()` | 关节角度 q(t) 时间历程 |
| `plot_joint_velocities()` | 关节速度 q̇(t) 时间历程 |
| `plot_energy()` | 能量演化曲线（动能、势能、耗散能、总能） |
| `plot_phase_portraits()` | 相平面图（q vs q̇） |
| `plot_inputs()` | 控制输入 u(t) 示意图 |
| `plot_residuals()` | 残差演化 |
| `plot_torque_components()` | 力矩分量分解（传动/弹性/摩擦/重力） |
| `plot_dashboard()` | **综合看板**：多面板显示所有关键信息 |
| `plot_quasistatic_bars()` | 准静态平衡柱状图（Phase 1） |
| `plot_joint_correlation()` | 关节角度相关性热力图 |

**独立运行**：
```bash
python -m src.visualization.simulation_visualizer --load results.npz
python -m src.visualization.simulation_visualizer --phase-compare phase1.npz phase2.npz --no-block
```

---

## 文件间调用关系

### SDAS 协同仿真（最新默认模式）

```
run_synergy_from_ohd.py
    │
    ├──▶ OrigamiHandDesign.load(ohd_path)
    │
    ├──▶ build_sdas_model(design)              ← SDAS 模型
    │       ├──▶ compute_R_one_sided(design)   # 计算 R_A, R_B
    │       └──▶ SDASModel(n, R_A, R_B, E_vec) # Schur 补 + 单向公式
    │
    ├──▶ build_urdf_to_synergy_mapping(urdf_path, design, ohd_path)
    │       └── 最近邻空间匹配：URDF关节世界坐标 ↔ fold_line中点 ↔ synergy索引
    │
    └──▶ MuJoCoSimulator(urdf_path, mesh_dir,
                          synergy_callback, synergy_motor_names,
                          ctrl_range_deg, synergy_with_sigma_sliders)
            └── run()
                └── 每帧：ctrl滑块 → slack钳位 → SDASModel.solve_motors() → 限位 → qpos → sync
```

### 力学数值仿真

```
run_numerical_simulation.py
    │
    ├──▶ OrigamiHandDesign.load(ohd_path)
    │
    ├──▶ SimulationConfig (dt, t_end, phase, 控制输入表达式)
    │
    ├──▶ HandSimulator(design, cfg)
    │       ├──▶ RigidBodySystem.from_design(design)    # 惯性参数
    │       ├──▶ DynamicsAssembler(n_joints, ...)        # 动力学方程
    │       ├──▶ compute_Q_matrix(design)                # Q矩阵 (Q_τM, Q_s, Q_sdot)
    │       ├──▶ HaywardArmstrongFriction(...)           # 静摩擦状态
    │       └──▶ Integrator(dt, method)                  # 数值积分器
    │
    ├──▶ Phase 1: sim.run_quasistatic(q0, u) → QuasiStaticResult
    ├──▶ Phase 2: sim.run(timeout) → SimulationTrajectory
    └──▶ 可视化: SimulationVisualizer / MuJoCo 轨迹回放
```

### 数据流向总览

```
DXF文件 ──▶ OrigamiParser ──▶ OrigamiHandDesign
(CAD图纸)                          │
                                    ├──▶ export_urdf() → URDF + STL
                                    │          ├──▶ PinocchioSimulator
                                    │          └──▶ MuJoCoSimulator
                                    │
                                    ├──▶ build_sdas_model() → SDASModel
                                    │       └── run_synergy_from_ohd.py (交互仿真)
                                    │
                                    ├──▶ build_dynamic_synergy_model() → DynamicSynergyModel
                                    │       └── run_synergy_from_ohd.py --dynamic
                                    │
                                    ├──▶ HandSimulator (力学数值仿真)
                                    │       └── run_numerical_simulation.py
                                    │
                                    └──▶ OrigamiSimulator (自研运动学)

.ohd文件 ──▶ OrigamiHandDesign.load()
(JSON)         │
                ├──▶ build_sdas_model() → SDASModel
                ├──▶ build_urdf_to_synergy_mapping()
                └──▶ MuJoCoSimulator(synergy_callback)
                        └── run() → 实时交互仿真

优化设计:
run_optimization.py
    └──▶ DesignSpace → DesignEvaluator
            ├──▶ compute_R(), compute_Rf(), compute_damper_T()
            └──▶ ObjectiveFunction(s) → OptimizationEngine
                    └── DE/SHGO/Grid/Random search
```

---

## 状态依赖自适应协同 (SDAS)

### 最新模型

SDAS 模型是当前 `run_synergy_from_ohd.py` **默认**使用的模型。理论详细推导见 `docs/SDAS.md`，完整计算链路说明见 `docs/algorithm.md`。

### 核心思想

经典自适应协同模型假设传动矩阵 R 是固定的，但真实绳驱系统中，由于 Capstan 摩擦和腱绳单向传力特性，实际的传动关系依赖于系统状态。

SDAS 模型将传动矩阵分解为两个方向分量：
- **R_A**：从 Motor A 出发的 Capstan 衰减加权传动向量
- **R_B**：从 Motor B 出发的 Capstan 衰减加权传动向量

### 约束选择策略

| 状态 | θ_A | θ_B | 公式 |
|------|-----|-----|------|
| 双马达同向 | >0 | >0 | **Schur 补公式**：q = S_A·θ_A + S_B·θ_B |
| 仅 Motor A | >0 | =0 | **单向公式**：q = S_A·θ_A |
| 仅 Motor B | =0 | >0 | **单向公式**：q = S_B·θ_B |
| 双松弛 | =0 | =0 | q = 0 |

### 与旧模型的区别

| 方面 | 旧模型 (Augmented Adaptive) | SDAS 模型 |
|------|---------------------------|-----------|
| 传动矩阵 | R (对称平均) + R_f (摩擦滑动) | R_A + R_B (单侧 Capstan 衰减) |
| 第二协同来源 | R_f (静摩擦冻结) | R_B - R_A (自然非对称) |
| 双马达模式 | σ/σ_f 线性叠加 + 增强协同 | Schur 补公式，含交叉耦合 |
| 单马达模式 | 仅一种传动矩阵 | 精确使用对应 R_A 或 R_B |
| 物理假设 | 静摩擦冻结阈值 | Capstan 衰减方向非对称性 |

### 关键改进

1. **取消了 Rf 概念**：用 R_A、R_B 的自然非对称性替代了旧的"摩擦冻结"假设
2. **Schur 补交叉耦合**：双马达同时拉动时，R_A 和 R_B 的交叉耦合在 Schur 补中精确处理
3. **状态依赖的约束选择**：单马达/双马达使用不同的物理公式（Schur 补 vs 单向公式），自动判断约束激活状态
4. **Slack 钳位**：负电机位移 → 0（腱绳不能受推），自动切换状态


---

## 动态协同 (Dynamic Synergy)

### 论文背景

实现 Piazza et al., *"SoftHand Pro-D: Matching Dynamic Content of Natural User Commands with Hand Embodiment for Enhanced Prosthesis Control"*, ICRA 2016。

### 核心思想

在欠驱动软体手中，利用**被动阻尼元件**实现多模式运动。在**慢速**驱动下，阻尼力可忽略，手沿**慢速协同方向** S_s 运动（准静态平衡）。在**快速**驱动下，阻尼力产生等效附加刚度，手沿**阻尼平衡协同方向** S_f 运动。

### 数学模型（阻尼平衡方法）

力平衡方程：
$$T^T C T \dot{q} + E q = R^T u$$

**阻尼平衡解释** — 将阻尼力 T^T C T q̇ 建模为速度依赖的附加刚度：
$$E_{\text{eff}}(\alpha) = E + \alpha \cdot \text{diag}(T^T C T)$$
$$S_{\text{eff}}(\alpha) = E_{\text{eff}}^{-1} R^T (R E_{\text{eff}}^{-1} R^T)^{-1}$$

### 物理实现

| 输入 | 物理意义 | 效果 |
|---|---|---|
| σ (慢速驱动, α=0) | 准静态闭合 | 慢速协同：所有手指同步闭合（power grasp） |
| σ (快速驱动, α=1) | 阻尼主导闭合 | 阻尼平衡：阻尼关节受限（pinch grasp） |
| α = speed_factor | 插值系数 | 慢速→快速平滑过渡 |

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

# 与 augmented synergy 联合
R_aug = np.vstack([R, Rf])
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

在腱绳驱动系统中，利用**库仑摩擦**作为第二驱动通道。当两台电机同向运动时，腱绳缩短，所有手指同时闭合（第一协同方向 σ）。当电机反向运动时，腱绳在滑轮间滑动，摩擦力引起张力阶梯分布，驱动部分手指打开、部分闭合（第二协同方向 σ_f）。

### 两个协同方向

| 输入 | 物理意义 | 效果 |
|---|---|---|
| σ = (θ₁ + θ₂)/2 | 同动量 | 所有手指同步闭合/打开 |
| σ_f = (θ₁ - θ₂)/2 | 差动量 | 手指间的相对运动 |

### 改进的 Rf 模型

本实现使用 **Capstan + slack 侧钳制**模型代替原论文的 Coulomb 摩擦模型：
- 物理原理：Motor A 拉紧时张力沿路径指数衰减，低于静摩擦阈值区段被"冻结"
- 解决了原 Coulomb 模型在对称路径上 Rf=0 的问题
- 产生物理正确的**不对称关节角分布**

### 验证案例

`models/ohd test/` 目录包含具有不同路由的设计：

| 文件 | 路由 | 预期行为 |
|---|---|---|
| `ohd_2.ohd` | 路由 (a) | 第二协同：左手（拇指+中指指根）主导打开 |
| `ohd_3.ohd` | 路由 (b) | 第二协同：中间手指主导打开 |
| `ohd_8.ohd` | 五指串联 | 第二协同 + 阻尼器仅作用于拇指三个关节 (25,26,27) |

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

---

## 数值仿真框架 (Numerical Simulation)

`src/simulation/` 包实现从纯几何仿真到完整力学仿真的升级，基于 Della Santina et al. (2018) TRO Eq.43-44 的动力学方程。

### 三层递进架构

| 阶段 | 描述 | 核心方程 | 数值方法 |
|---|---|---|---|
| **Phase 1** 准静态力平衡 | f(q) = 0，无时间演化 | Kq = τ_input(u) | `scipy.optimize.root` (hybr) |
| **Phase 2** 完整动力学 | ODE 积分，含惯性与阻尼 | M(q)q̈ + B q̇ + Kq = τ_input | RK4 / scipy RK45 |
| **Phase 3** 动力学+接触 | 惩罚法接触 + 库仑摩擦 | Mq̈ = τ_ext - Bq̇ - Kq + J^T f_c | RK4 + 罚函数 |

### 控制输入

仿真支持**时间函数表达式**作为控制输入，通过 `SimulationConfig` 设置：
- `sigma_func`：σ(t) 表达式，如 `"min(t*2, 1.0)"`、`"5.0"`
- `sigma_f_func`：σ_f(t) 表达式，如分段常数 `"(t<0.3)*0 + (t>=0.3 and t<0.6)*(-2) + (t>=0.6)*(2)"`
- `tau_M_func`：τ_M(t) 电机拉力

### 能量追踪

`SimulationTrajectory` 记录：
- **动能**：$E_k = \frac{1}{2} \dot{q}^T M(q) \dot{q}$
- **势能**：$E_p = \frac{1}{2} q^T K q$
- **耗散能**：$E_d = \int_0^t \dot{q}^T B \dot{q} \, dt$

### 使用示例

```python
from src.simulation.config import SimulationConfig
from src.simulation.simulator import HandSimulator
from src.models.origami_design import OrigamiHandDesign

# 加载设计
design = OrigamiHandDesign.load('models/ohd test/ohd_1.ohd')

# Phase 1: 准静态力平衡
cfg = SimulationConfig(phase=1, tol=1e-8, max_iter=50)
sim = HandSimulator(design, cfg)
result = sim.run_quasistatic(q0=np.zeros(sim.n_joints), u=np.array([5.0]))

# Phase 2: 完整动力学（时间函数控制输入）
cfg = SimulationConfig(dt=1e-4, t_end=0.1, phase=2)
cfg.sigma_func = "5.0"
cfg.tau_M_func = "5.0"
sim = HandSimulator(design, cfg)
traj = sim.run(timeout=30.0)
```

### 分段连续仿真

```bash
# 三段: σ=5 常量, σ_f=0→-2→+2, 每段 0.3s
python scripts/run_numerical_simulation.py "models/ohd test/ohd_2.ohd" \
  --phase 2 --sigma 5 5 5 --sigma-f 0 -2 2 --t-per 0.3 --viz
```

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

## 仿真可视化 (Simulation Visualization)

`src/visualization/simulation_visualizer.py` 提供全面的仿真结果可视化。

### 独立运行

```bash
# 显示单个仿真结果
python -m src.visualization.simulation_visualizer --load results.npz

# 显示多阶段对比
python -m src.visualization.simulation_visualizer --phase-compare phase1.npz phase2.npz phase3.npz

# 保存图片
python -m src.visualization.simulation_visualizer --load results.npz --save --prefix my_results
```

### 在脚本中使用

```python
from src.visualization.simulation_visualizer import visualize_simulation

# Phase 2 结果可视化
visualize_simulation(traj_data={
    't': traj.t,
    'q': traj.q,
    'q_dot': traj.q_dot,
    'energy_kinetic': traj.energy_kinetic,
    'info': {'phase': 2, 'method': 'RK4', 'dt': 1e-4},
})
```

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
| `test_sdas_model.py` (通过 `test_adaptive_synergy.py`) | SDAS 模型测试 |
| `test_dynamic_synergy.py` | 基础动态协同测试 |
| `test_dynamic_synergy_extended.py` | **动态协同扩展测试（17项）**：数学验证、边界情况、ohd_8阻尼、蒙特卡罗 |
| `test_augmented_synergy.py` | 增强协同测试 |
| `test_adaptive_synergy.py` | 基础协同测试（含 SDAS 模型验证） |
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
| `L` | 轮廓线 | 绘制黑色轮廓线 | 输入 x,y 坐标或鼠标点击 → 再输入/点击终点 |
| `M` | 峰折 | 绘制红色峰折线 | 同上 |
| `V` | 谷折 | 绘制蓝色谷折线 | 同上 |
| `P` | 滑轮 | 放置灰色滑轮(半径5px) | 输入 x,y 坐标或鼠标点击，自动吸附到最近折痕线 |
| `T` | 腱绳 | 连接滑轮/驱动器的虚线路径 | 依次点击滑轮或驱动器 → 连续创建灰色虚线连接段 → **右键完成并保存** |
| `A` | 驱动器 | 放置驱动器点 | 输入 x,y 坐标或鼠标点击，默认蓝色(A型) |
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

### 完整工作流：CAD 设计 → SDAS 协同仿真

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

# 3. 启动 SDAS 交互仿真（默认模式）
python scripts/run_synergy_from_ohd.py "models/ohd test/ohd_2.ohd"
#    - 右侧面板四个滑块：Motor A, Motor B, σ, σ_f
#    - 拖动滑块驱动手指运动
#    - SDAS 模型自动处理单/双马达状态切换

# 4. 动态协同模式（五指+阻尼器）
python scripts/run_synergy_from_ohd.py "models/ohd test/ohd_8.ohd" --dynamic
#    - 五个滑块：Motor A, Motor B, σ, σ_f, Speed (rad/s)
#    - Speed 控制快速/慢速协同切换

# 5. 力学数值仿真
python scripts/run_numerical_simulation.py "models/ohd test/ohd_2.ohd" --phase 2 --sigma 5 5 5 --sigma-f 0 -2 2 --t-per 0.3 --viz --mujoco

# 6. 优化设计
python scripts/run_optimization.py
```

### 三步分段动力学仿真

```bash
# 基本用法（无需安装额外依赖）
python scripts/run_three_phase_demo.py

# 带可视化
python scripts/run_three_phase_demo.py --viz

# 保存结果
python scripts/run_three_phase_demo.py --t-per 0.5 --save result.npz

# 加载并可视化保存的结果
python -m src.visualization.simulation_visualizer --load result.npz
```

### 多段连续力学仿真

```bash
# 三段: σ=5, σ_f=0→-2→+2, 每段 0.3s, 可视化+MuJoCo回放
python scripts/run_numerical_simulation.py "models/ohd test/ohd_2.ohd" \
  --phase 2 --sigma 5 5 5 --sigma-f 0 -2 2 --t-per 0.3 --viz --mujoco
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

---

## 参考文献

1. Santello, M., et al. "Postural Hand Synergies for Tool Use." *Journal of Neuroscience*, 1998.
2. Bicchi, A., et al. "Soft synergies: A new variable stiffness paradigm for robotic hands." *IEEE ICRA*, 2011.
3. Grioli, G., et al. "Adaptive synergies for a humanoid robot hand." *IEEE-RAS International Conference on Humanoid Robots*, 2012.
4. Della Santina, C., et al. "Toward dexterous manipulation with augmented adaptive synergies: The Pisa/IIT SoftHand 2." *IEEE Transactions on Robotics*, 34(5), 2018.
5. Piazza, C., et al. "SoftHand Pro-D: Matching dynamic content of natural user commands with hand embodiment for enhanced prosthesis control." *IEEE ICRA*, 2016.
6. Feix, T., et al. "The GRASP Taxonomy of Human Grasp Types." *IEEE Transactions on Human-Machine Systems*, 2016.
