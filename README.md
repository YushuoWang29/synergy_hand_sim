# synergy_hand_sim

**折纸手设计仿真工具包** — 从CAD图纸(DXF)到3D交互式折纸仿真的完整流程，支持 URDF 导出与 Pinocchio 机器人学库集成。

---

## 目录

- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [项目文件结构](#项目文件结构)
- [各文件详细说明](#各文件详细说明)
- [文件间调用关系](#文件间调用关系)
- [测试数据](#测试数据)
- [用法示例](#用法示例)

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

### 4. DXF 诊断

```bash
python scripts/diagnose_dxf.py tests/test_hand.dxf
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
│   ├── export_urdf.py                # URDF导出脚本：DXF → URDF + STL
│   └── diagnose_dxf.py               # 诊断工具：分析DXF文件的面片识别情况
│
├── src/                          # 核心源码
│   ├── __init__.py
│   │
│   ├── models/                   # 数据模型与算法
│   │   ├── origami_design.py         # 数据模型：折纸手设计的核心数据结构
│   │   ├── origami_kinematics.py     # 运动学：3D正向运动学计算（支持环状结构）
│   │   ├── origami_parser.py         # 解析器：从DXF文件生成OrigamiHandDesign
│   │   └── origami_to_urdf.py        # URDF导出：将设计导出为URDF+STL格式
│   │
│   ├── visualization/            # 3D可视化
│   │   └── origami_visualizer.py     # 3D可视化：MeshCat浏览器端渲染
│   │
│   └── interactive/              # 交互式GUI
│       ├── cad_viewer.py             # 2D视图：matplotlib CAD图纸预览与交互
│       ├── origami_simulator.py      # 自研运动学仿真器：Qt GUI，整合2D+3D交互
│       └── pinocchio_simulator.py    # Pinocchio仿真器：基于URDF的交互式仿真
│
├── models/                       # 导出的URDF模型
│   ├── test_1/
│   │   ├── test_1.urdf
│   │   └── meshes/                   # STL几何文件
│   └── test_2/
│       ├── test_2.urdf
│       └── meshes/
│
├── tests/                        # 测试文件
│   ├── __init__.py
│   ├── test_hand.dxf                 # 测试用DXF（手掌+两节手指）
│   ├── test_1.dxf                    # 简单折纸DXF
│   ├── test_2.dxf / test_2.dwg      # 多关节折纸设计
│   ├── simple_fold.urdf              # 简单对折URDF（测试导出）
│   ├── test_origami_design.py        # 数据结构单元测试
│   ├── test_origami_kinematics.py    # 运动学单元测试
│   ├── test_visualize_origami.py     # 可视化测试
│   ├── test_cad_import.py            # DXF导入端到端测试
│   └── test_pinocchio_urdf.py        # Pinocchio URDF加载测试
│
└── test_2_dxf.txt / test_2_urdf.txt  # 诊断输出记录
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
| `FoldLine` | 一条折痕线：记录起点、终点、类型、长度、方向向量、法向量、中点 |
| `OrigamiFace` | 一个面片：由顶点序列和边ID序列定义的封闭多边形，含面积计算、点包含判断 |
| `JointConnection` | 关节连接：记录两个面片通过某条折痕形成旋转关节的关系，含关节偏移方向 |
| `OrigamiHandDesign` | 顶层容器：管理所有折痕线、面片、关节，支持面片树构建（BFS）、完整性验证、摘要输出 |

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

**功能**：将折纸手设计导出为标准URDF+STL格式，可供Pinocchio等机器人学库加载。

**依赖**：`origami_design.py`

**包含的函数**：

| 函数 | 说明 |
|---|---|
| `_write_face_stl(face, filepath, offset, thickness)` | 为一个面片生成二进制STL文件（上表面、下表面、侧面三角化） |
| `_write_binary_stl(filepath, triangles)` | 写入二进制STL格式 |
| `export_urdf(design, output_path, thickness)` | 主导出函数：<br>• 为每个面片创建 `<link>`（引用对应STL）<br>• 为每个关节创建 `<joint type="revolute">`<br>• 折痕对齐定位策略：子link原点放在折痕中点<br>• 峰折子面片整体下移以便关节轴对齐下表面<br>• 谷折子面片保持上表面齐平 |

---

### 5. `src/visualization/origami_visualizer.py` — 3D可视化层

**功能**：在浏览器中实时渲染折纸手的3D折叠状态。

**依赖**：`origami_kinematics.py`、`meshcat`

**包含的类**：

| 类 | 说明 |
|---|---|
| `OrigamiVisualizer` | MeshCat可视化器：<br>• `display_hand(fk, joint_angles)`：显示给定关节角度下的手部姿态（三角化面片+颜色区分）<br>• `animate_folding(fk, joint_id, start, end, steps)`：动画展示单个关节的折叠过程<br>• `open_browser()`：在默认浏览器中打开可视化页面<br>• `set_camera_position(position, lookat)`：设置相机位置<br>• `auto_set_camera(fk, margin)`：根据几何自动调整相机 |

---

### 6. `src/interactive/cad_viewer.py` — 2D交互视图层

**功能**：提供CAD图纸的2D预览，支持鼠标点击选折痕。

**依赖**：`origami_design.py`、`matplotlib`（Qt5Agg后端）

**包含的类**：

| 类 | 说明 |
|---|---|
| `CADViewer` | matplotlib 2D视图：<br>• `_init_plot`：绘制面片半透明填充 + 折痕线（红=峰、蓝=谷、黑=轮廓）+ 面片编号 + 图例<br>• `_on_click`：计算点击点到各折痕的屏幕距离，选中最近的<br>• `_select_line(line_id)`：高亮选中的折痕（变绿加粗），触发回调<br>• `set_select_callback(callback)`：注册选中回调 |

---

### 7. `src/interactive/origami_simulator.py` — 自研运动学仿真器

**功能**：整合2D视图和3D可视化的Qt GUI窗口，使用自研正向运动学引擎。

**依赖**：`origami_design.py`、`origami_kinematics.py`、`origami_visualizer.py`、`cad_viewer.py`、`PyQt5`、`matplotlib`

**包含的类**：

| 类 | 说明 |
|---|---|
| `OrigamiSimulator(QtWidgets.QMainWindow)` | 主窗口：<br>• **左侧面板**：嵌入`CADViewer`的2D视图 + matplotlib工具栏<br>• **右侧面板**：选中信息显示、角度滑块（-90°~+90°）、精确输入框、所有关节状态列表、重置按钮、快捷键提示<br>• **定时器**：每50ms检查角度变化，自动刷新3D视图<br>• **键盘交互**：↑↓±5°、←→±1°、Enter应用、R重置、Esc取消选中<br>• **内部状态**：维护`joint_angles`字典，连接折痕ID与关节ID的映射 |

---

### 8. `src/interactive/pinocchio_simulator.py` — Pinocchio URDF仿真器

**功能**：基于Pinocchio的交互式URDF仿真器，加载预导出的URDF+STL模型，使用Pinocchio进行运动学计算和3D渲染。

**依赖**：`origami_design.py`、`cad_viewer.py`、`pinocchio`、`PyQt5`、`matplotlib`

**包含的类**：

| 类 | 说明 |
|---|---|
| `PinocchioSimulator(QtWidgets.QMainWindow)` | 主窗口：<br>• **左侧面板**：`CADViewer` 2D视图，点击折痕选中对应Pinocchio关节<br>• **右侧面板**：角度滑块（-180°~+180°）、精确输入框、关节状态列表、重置按钮<br>• **3D视图**：MeshCat浏览器端渲染，Pinocchio实时更新<br>• `_load_model`：加载URDF，处理相对mesh路径为绝对路径<br>• `_init_3d_view`：初始化MeshCatVisualizer<br>• **折痕→Pinocchio关节映射**：通过URDF joint name `joint_{id}` 建立关联<br>• **定时器**：每50ms检查配置向量变化，自动刷新3D视图 |

---

### 9. `scripts/export_urdf.py` — URDF导出脚本

**功能**：将DXF文件解析后导出为URDF+STL格式，输出到 `models/<hand_name>/` 目录。

**用法**：
```bash
python scripts/export_urdf.py tests/test_hand.dxf --tolerance 2.0 --thickness 3.0
```

**参数**：
- `dxf_file`：输入DXF路径
- `--tolerance`：点合并容差（默认2.0mm）
- `--thickness, -t`：材料厚度（默认3.0mm）

---

### 10. `scripts/run_pinocchio_simulator.py` — Pinocchio仿真器启动脚本

**功能**：启动基于Pinocchio的交互式URDF仿真器。

**用法**：
```bash
python scripts/run_pinocchio_simulator.py models/test_2/test_2.urdf
python scripts/run_pinocchio_simulator.py models/test_2/test_2.urdf --dxf tests/test_2.dxf
```

**参数**：
- `urdf_file`：URDF文件路径
- `--dxf`：DXF文件路径（可选，不指定则自动推断）
- `--tolerance`：DXF解析容差（默认2.0mm）

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
            │       └── 使用 design.fold_lines, design.faces
            │
            ├──▶ OrigamiVisualizer()            [3D浏览器]
            │       └── display_hand(fk, joint_angles)
            │               └──▶ fk.get_face_vertices_world(joint_angles)
            │
            └──▶ 定时器驱动 _update_3d()
                    └──▶ viz.display_hand(fk, joint_angles)
```

### Pinocchio URDF 仿真器

```
run_pinocchio_simulator.py
    │
    ├──▶ OrigamiParser.parse(dxf_path)     [仅用于2D视图]
    │       └──▶ OrigamiHandDesign
    │
    └──▶ PinocchioSimulator(design, urdf_path, mesh_dir)
            │
            ├──▶ pin.buildModelFromUrdf(urdf)
            ├──▶ pin.buildGeomFromUrdf(...)   [加载STL几何]
            │
            ├──▶ CADViewer(design)            [左侧2D面板]
            │       └── 折痕点击 → 映射到Pinocchio joint index
            │
            ├──▶ MeshcatVisualizer            [3D浏览器]
            │       └── viz.display(q)        [Pinocchio实时计算]
            │
            └──▶ 定时器驱动 _update_3d()
                    └──▶ viz.display(q)
```

### URDF 导出流程

```
export_urdf.py
    │
    ├──▶ OrigamiParser.parse(dxf_path)
    │       └──▶ OrigamiHandDesign
    │
    └──▶ export_urdf(design, output_path, thickness)
            ├── 确定每个link的原点（折痕对齐法）
            ├── 生成每个面片的STL二进制文件
            └── 生成URDF XML（link + revolute joint）
```

### 数据流向

```
DXF文件 ──▶ OrigamiParser ──▶ OrigamiHandDesign ──▶ OrigamiForwardKinematics
(CAD图纸)                                                   │
                                                      OrigamiVisualizer (3D)
                                                            │
                                                      CADViewer (2D)
                                                            │
                                          ┌─────────────────┴─────────────────┐
                                          ▼                                   ▼
                                   OrigamiSimulator                export_urdf()
                                  (自研运动学GUI)                        │
                                                                        ▼
                                                               URDF + STL 文件
                                                                        │
                                                                        ▼
                                                          PinocchioSimulator
                                                          (Pinocchio URDF GUI)
```

---

## 测试数据

| 文件 | 描述 |
|---|---|
| `tests/test_hand.dxf` | **主测试文件**：手掌(50×30) + 近端手指 + 远端手指，含谷折关节 |
| `tests/test_1.dxf` | 简单折纸设计 |
| `tests/test_2.dxf` | 多关节折纸设计（含DWG格式源文件） |
| `models/test_1/` | test_1的导出URDF+STL |
| `models/test_2/` | test_2的导出URDF+STL（含6个关节） |

### 运行测试

```bash
# 数据结构测试
python tests/test_origami_design.py

# 运动学测试
python tests/test_origami_kinematics.py

# DXF导入端到端测试（生成DXF → 解析 → 可视化）
python tests/test_cad_import.py

# 可视化测试
python tests/test_visualize_origami.py

# Pinocchio URDF加载测试
python tests/test_pinocchio_urdf.py
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

---

## DXF 诊断工具

当解析结果不符合预期时，可使用诊断工具逐步骤检查：

```bash
python scripts/diagnose_dxf.py tests/test_hand.dxf
```

输出包括：
- 原始线段提取情况
- 交点分割后的线段数
- 图结构（节点数、边数、边类型）
- 找到的面片及其顶点和边
- 边→面片映射关系
