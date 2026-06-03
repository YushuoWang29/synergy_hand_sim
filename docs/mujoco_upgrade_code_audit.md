# MuJoCo numerical upgrade code audit

本文档记录 `synergy_hand_sim` 从几何仿真升级到 MuJoCo 数值仿真的代码审计结果。目标版本不引入阻尼器、粘滞阻尼或速度阻尼项。

## 1. 现有输入与数据结构

当前 `.ohd` 文件主要是结构设计文件，而不是完整仿真任务文件。已有字段包括：

- `fold_lines`：折痕和轮廓线，折痕含 `stiffness`。
- `faces` / `joints` / `root_face_id` / `face_parent`：拓扑信息；部分样例中这些字段为空，需要由 `build_topology()` 重建。
- `pulleys` / `holes` / `tendons`：腱绳路径和与折痕的关联关系。
- `actuators` / `actuator_positions`：默认包含 `Motor_A` 和 `Motor_B`。
- `dampers`：旧的 dynamic synergy 相关字段，本次 MuJoCo 升级中不能使用。

建议新增一个向后兼容的顶层字段 `simulation`，用于放置仿真总时长、时间步长、驱动输入序列、分布类型和输出设置。旧 `.ohd` 没有该字段时，命令行工具应能生成默认示例输入。

## 2. 可直接复用的模块

### `.ohd` 解析

`src/models/origami_design.py` 中的 `OrigamiHandDesign.load()` 可以继续作为设计入口。需要注意：当前 `load()` 不保存 `source_path`，后续仿真入口应在加载后显式补充，以便导出 URDF 或定位已有模型目录。

### URDF/STL 导出

`src/models/origami_to_urdf.py` 可以继续复用。它把折纸面片导出为 STL，并生成 revolute joint。后续 MuJoCo 数值仿真可先复用该 URDF，再转换为 MJCF。转换时需要补充质量、惯量、关节刚度、执行器和接触参数。

### 传动矩阵与协同映射

`src/models/transmission_builder.py` 中以下函数可以复用：

- `get_joint_list()`：将折痕转换为协同模型的关节列表，且能处理 `faces/joints` 为空的 `.ohd` 样例。
- `compute_R_one_sided()`：生成从 Motor A / Motor B 两端出发的单侧传动向量 `R_A`、`R_B`。
- `build_sdas_model()`：虽然命名仍为 SDAS，但数学结构可作为双端腱路映射求解器使用。

`src/synergy/sdas_model.py` 中的 Schur 补求解没有速度项或阻尼项，可作为自由空间参考构型生成器：

```math
q = S_A \theta_A + S_B \theta_B
```

或

```math
q = (S_A + S_B)\sigma_c + (S_A - S_B)\sigma_d
```

其中 `theta_A = sigma_c + sigma_d`，`theta_B = sigma_c - sigma_d`。

## 3. 不能直接复用的模块

### `src/simulation/mujoco_physics.py`

该文件虽然调用了 `mj_step()`，但与本次约束冲突：

- MJCF 中 hinge joint 写入了 `damping="0.1"`。
- `_compute_joint_torques()` 中加入了 `-joint_damping_ratio * 0.01 * q_dot`。
- 依赖 `compute_Q_matrix()`，其中包含 `Q_sdot` 等速度相关摩擦项。

因此不能作为升级版数值仿真的实现基础，只能参考其 URDF-to-MJCF 的结构遍历方式。

### `src/simulation/config.py` 与自定义积分器

旧的自定义数值仿真配置包含：

- `joint_viscous_damping`
- `tendon_viscous_damping`
- `sliding_s_func`
- 静摩擦记忆和速度相关项

这些内容与“只使用 MuJoCo、无阻尼器、无速度阻尼”的目标不一致。新接口应另建配置数据结构，避免误用旧字段。

### `src/synergy/dynamic_synergy.py`

该模块面向 dynamic synergy 和被动阻尼器，不进入本次升级。

## 4. 现有 MuJoCo 几何展示的定位

`src/interactive/mujoco_simulator.py` 会把 URDF 转为 MJCF，并启动 MuJoCo viewer。但在协同模式下，它每帧直接把协同模型求得的角度写入 `data.qpos`，然后调用 `mj_forward()`。这适合做交互式几何展示，但不是数值仿真，因为没有通过 `mj_step()` 积分关节运动和接触响应。

升级版应新增独立执行层：

1. 读取 `.ohd` 和 `simulation` 配置。
2. 生成或定位 URDF/STL，再构建无阻尼 MJCF。
3. 按时间序列计算驱动输入。
4. 将输入映射为关节参考构型或关节力矩。
5. 使用 `mj_step()` 推进仿真。
6. 保存日志、截图和可选视频。

## 5. 建议的新接口

建议新增脚本：

```text
scripts/run_mujoco_numerical_from_ohd.py
```

建议新增模块：

```text
src/simulation/mujoco_numerical.py
```

建议 `.ohd` 顶层增加：

```json
{
  "simulation": {
    "model_path": "models/five_finger_1/five_finger_1.urdf",
    "duration": 2.0,
    "dt": 0.002,
    "input_mode": "position",
    "distribution": {
      "type": "transmission"
    },
    "inputs": [
      {
        "name": "sigma_c",
        "times": [0.0, 0.5, 2.0],
        "values": [0.0, 0.8, 0.8]
      },
      {
        "name": "sigma_d",
        "times": [0.0, 2.0],
        "values": [0.0, 0.0]
      }
    ],
    "output": {
      "directory": "outputs/mujoco_numerical/demo",
      "screenshots": [0.5, 1.5],
      "save_npz": true
    }
  }
}
```

其中 `distribution.type` 可支持：

- `transmission`：由 `.ohd` 腱路和折痕刚度生成 `R_A/R_B`，再用 Schur 补得到协同映射。
- `joint_space`：驱动输入直接对应关节空间向量。
- `custom`：用户给定矩阵 `B`，计算 `q_ref = B u`。
- `endpoint`：作为后续扩展接口；当前最小实现可要求用户提供显式矩阵或关节权重。

## 6. 实现边界

本次最小可运行版本应采用以下边界：

- MuJoCo 模型中的 `damping` 全部为 0 或缺省。
- 不创建 `<damper>`、不使用 velocity actuator。
- 不在 Python 中加入 `-D q_dot` 或类似速度阻尼项。
- 允许使用关节 `stiffness` 表示折痕弹性，因为它是保守弹性项，不是阻尼。
- 允许使用关节 `armature` 改善小惯量系统的数值稳定性，因为它增加的是转动惯量，不是耗散项。
- 允许保留 MuJoCo 接触摩擦；它属于接触求解的一部分，不是腱绳内部速度阻尼。

## 7. 第 3 步实施建议

优先实现“位置输入模式”：

```math
q_{ref}(t) = B_{\sigma} u(t)
```

并用无速度项的比例力矩控制：

```math
\tau(t) = K_p \left(q_{ref}(t) - q(t)\right)
```

随后可支持“力输入模式”：

```math
\tau(t) = B_{\tau} u(t)
```

其中 `B_tau` 可以由 `R_A^T`、`R_B^T` 或用户给定矩阵生成。两个模式都通过 MuJoCo `motor` actuator 输入关节力矩并使用 `mj_step()` 推进。
