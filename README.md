## 项目文件结构总结

```
synergy_hand_sim/
│
├── scripts/
│   ├── run_origami_simulator.py      # 主入口：交互式仿真器启动脚本
│   └── diagnose_dxf.py               # 诊断工具：分析DXF文件的面片识别情况
│
└── src/
    ├── models/
    │   ├── origami_design.py          # 数据模型：折纸手设计的核心数据结构
    │   ├── origami_kinematics.py      # 运动学：3D正向运动学计算
    │   ├── origami_parser.py          # 解析器：从DXF文件生成OrigamiHandDesign
    │   └── origami_to_urdf.py         # URDF导出：将设计导出为URDF格式
    │
    ├── visualization/
    │   └── origami_visualizer.py      # 3D可视化：MeshCat浏览器端渲染
    │
    └── interactive/
        ├── cad_viewer.py              # 2D视图：matplotlib CAD图纸预览与交互
        └── origami_simulator.py       # 主控制器：Qt GUI窗口，整合2D+3D交互
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
| `OrigamiForwardKinematics` | 正向运动学求解器：<br>• `__init__`：预计算所有关节的局部轴位姿和面片3D顶点<br>• `forward_kinematics(joint_angles)`：BFS递归计算所有面片的世界变换矩阵<br>• `get_face_vertices_world(joint_angles)`：获取所有面片顶点在世界坐标中的位置 |

---

### 3. `src/models/origami_parser.py` — DXF解析层

**功能**：从CAD图纸（DXF格式）自动提取几何信息，生成 `OrigamiHandDesign`。

**依赖**：`origami_design.py`、`ezdxf`

**包含的类**：

| 类 | 说明 |
|---|---|
| `OrigamiParser` | DXF解析器，核心流程：<br>① `_extract_segments`：提取LINE和LWPOLYLINE实体，按颜色分类<br>② `_split_at_intersections`：在所有线段交点处强制分割，确保图为严格平面图<br>③ `_build_graph`：建立无向图（节点=唯一点，边=原子线段）<br>④ `_find_minimal_cycles`：用"最左转"算法找到所有最小环路（面片）<br>⑤ `_remove_outer_face`：去掉面积最大的外轮廓面<br>⑥ `_create_design`：组装 `OrigamiHandDesign`，创建关节连接，构建面片树 |

**颜色映射**（`_dxf_color_to_fold_type`）：
- 红/品红/橙 → 峰折
- 蓝/青 → 谷折
- 黑/深色 → 轮廓

---

### 4. `src/models/origami_to_urdf.py` — URDF导出层

**功能**：将折纸手设计导出为标准URDF格式，可供Pinocchio等机器人学库加载。

**依赖**：`origami_design.py`、`origami_kinematics.py`

**包含的函数**：

| 函数 | 说明 |
|---|---|
| `export_urdf(design, output_path)` | 为每个面片创建`<link>`，为每个关节创建`<joint type="revolute">`，计算关节在父link坐标系中的原点位置和旋转轴方向 |

---

### 5. `src/visualization/origami_visualizer.py` — 3D可视化层

**功能**：在浏览器中实时渲染折纸手的3D折叠状态。

**依赖**：`origami_kinematics.py`、`meshcat`

**包含的类**：

| 类 | 说明 |
|---|---|
| `OrigamiVisualizer` | MeshCat可视化器：<br>• `display_hand(fk, joint_angles)`：显示给定关节角度下的手部姿态（三角化面片+颜色区分）<br>• `animate_folding(fk, joint_id, ...)`：动画展示单个关节的折叠过程<br>• `open_browser()`：在默认浏览器中打开可视化页面 |

---

### 6. `src/interactive/cad_viewer.py` — 2D交互视图层

**功能**：提供CAD图纸的2D预览，支持鼠标点击选折痕。

**依赖**：`origami_design.py`、`matplotlib`（Qt5Agg后端）

**包含的类**：

| 类 | 说明 |
|---|---|
| `CADViewer` | matplotlib 2D视图：<br>• `_init_plot`：绘制面片半透明填充 + 折痕线（红=峰、蓝=谷、黑=轮廓）+ 图例<br>• `_on_click`：计算点击点到各折痕的屏幕距离，选中最近的<br>• `_select_line(line_id)`：高亮选中的折痕（变绿加粗），触发回调 |

---

### 7. `src/interactive/origami_simulator.py` — 主控制器层

**功能**：整合2D视图和3D可视化的Qt GUI窗口，提供完整的交互式仿真体验。

**依赖**：以上所有模块 + `PyQt5` + `matplotlib`（Qt5Agg后端）

**包含的类**：

| 类 | 说明 |
|---|---|
| `OrigamiSimulator(QtWidgets.QMainWindow)` | 主窗口：<br>• **左侧面板**：嵌入`CADViewer`的2D视图 + matplotlib工具栏<br>• **右侧面板**：选中信息显示、角度滑块、精确输入框、所有关节状态列表、重置按钮、快捷键提示<br>• **定时器**：每50ms检查角度变化，自动刷新3D视图<br>• **键盘交互**：↑↓±5°、←→±1°、Enter应用、R重置、Esc取消选中<br>• **内部状态**：维护`joint_angles`字典，连接折痕ID与关节ID的映射 |

---

## 文件间调用关系

```
run_origami_simulator.py
    │
    ├──▶ OrigamiParser.parse(dxf_path)
    │       └──▶ OrigamiHandDesign
    │
    └──▶ OrigamiSimulator(design)
            │
            ├──▶ OrigamiForwardKinematics(design)
            │       └── 使用 design 中的面和关节信息
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

**数据流向**：
```
DXF文件 → OrigamiParser → OrigamiHandDesign → OrigamiForwardKinematics
                                                    ↓
                                              OrigamiVisualizer (3D)
                                                    ↓
                                              CADViewer (2D)
                                                    ↓
                                           OrigamiSimulator (GUI整合)
```