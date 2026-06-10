# Physical Simulation Poster Content Plan

本文用于制作课程物理仿真项目汇报海报。只保留 `template.pptx` 的画布尺寸和基础容器，不参考同文件夹或 `docs` 中任何已有 poster 草稿、海报文本或 Markdown 尝试稿。

海报整体重点：介绍 `synergy_hand_sim` 项目的仿真部分。叙事主线应说明项目如何把几何/运动学仿真、FMAS 索传动建模、MuJoCo 数值积分和接触响应分析组织成一个完整仿真体系。几何仿真负责生成和解释参考运动 `q_ref(t)`，数值仿真在此基础上求解时间响应 `q(t)`、接触、约束和力学日志。

建议总标题：

```text
Modeling Adaptive Grasping in a Cable-Driven Origami Dexterous Hand
```

建议副标题：

```text
Geometry, FMAS transmission, and MuJoCo contact simulation from .ohd designs
```

作者行：

```text
Yushuo Wang | Physical Simulation Final Project
```

---

## 1. Problem Statement

### 中文排版说明

放在海报左上区域。该部分要先让评审理解整个项目的物理仿真问题，而不是直接进入代码实现。建议使用两栏结构：左侧用一张小流程图表现从 `.ohd` 到几何参考、再到数值接触仿真的问题链路；右侧用简短文字说明本项目要回答的物理问题。

版面元素建议：

- 一个醒目的小标题：`Problem Statement`
- 一句核心问题：同一个折纸手设计既需要解释“腱路期望产生什么运动”，也需要求解“接触和时间积分下实际发生什么运动”。
- 一个三行表格：Simulation inputs / Simulation layers / Simulation outputs。
- 两个公式卡片：`q_ref(t)` 和 `e_q(t)`。
- 图片放置：几何参考和 MuJoCo 轨迹对比图，突出虚线和实线不完全重合。

这部分不要写成“我要做一个软件”。要写成：本项目的仿真任务包含从设计几何、协同参考、数值动力学到接触响应的完整物理建模链条。

### Poster English Text

The simulation problem in `synergy_hand_sim` has two connected levels. The geometric and kinematic level explains what motion the origami hand design and tendon routing are intended to produce. It generates a reference joint configuration `q_ref(t)` from low-dimensional synergy commands.

The physical simulation level then asks how this reference motion is realized when the hand is represented as an articulated model and advanced through numerical integration and contact constraints. MuJoCo is used to solve the time response `q(t)`, contact count, contact force, and object interaction.

The input is an `.ohd` hand and simulation definition containing geometry, tendon routing, driver time series, distribution type, and optional grasp objects. The output is a reference trajectory, time-integrated joint trajectory, contact log, force summary, rendered screenshots, process GIF, and a generated MJCF model that can be audited.

The central simulation question is: how can a cable-driven origami hand design be simulated across geometry, synergy transmission, numerical dynamics, and contact response in one consistent pipeline?

### Key Formula Text

```text
Geometry reference:
q_ref(t) = B_sigma sigma(t)
```

```text
Numerical simulation error:
e_q(t) = q(t) - q_ref(t)
```

```text
Simulation goal:
Use geometric synergy to define q_ref(t), then use MuJoCo to solve q(t), contact, and object response.
```

### 图片内容说明

图片 1：几何参考姿态与 MuJoCo 数值状态的轨迹对比图。图中虚线代表 `q_ref(t)`，实线代表 MuJoCo 积分后的 `q(t)`。这张图用于说明项目同时包含参考运动生成和数值物理响应分析，两者是同一仿真链条的不同输出。

建议使用本项目输出图：

```text
docs/Physical Simulation/poster/generated_figures/poster_ready/fig1_free_space_reference_vs_mujoco.png
```

图片 2：一张简单流程图，内容为 `.ohd design -> geometric simulation -> FMAS transmission -> MuJoCo mj_step -> q(t), contact logs`。该图可以人工重画，不要使用现成海报图。

---

## 2. Literature Review

### 中文排版说明

放在 Problem Statement 下方或左中区域。该部分要服务于物理仿真主题：为什么协同、柔顺性、欠驱动和接触响应是相关的。

建议做成“模型谱系表”，不要写成长篇综述。表格左列是理论或工作，中列是物理含义，右列是对本项目的作用。重点是从 `postural synergy` 逐步过渡到 `soft/adaptive synergy`，再引出本项目的 FMAS。

必须注意：FMAS 不是直接来自 SoftHand 文献，而是本项目针对双端闭环索驱动折纸手提出的模型。文献综述应写成动机和对照。

### Poster English Text

Postural synergy studies show that high-dimensional hand postures can often be represented by a few low-dimensional coordinates. This motivates synergy-driven hands: a small number of commands can organize many joints into coordinated grasping motions.

Soft synergy models add an important physical interpretation. A synergy command should not be treated as a rigid final posture, but as a compliant reference. When the hand contacts an object, the realized posture can deviate from the reference due to external forces and structural compliance.

Adaptive synergy and the Pisa/IIT SoftHand demonstrate how this idea can be implemented mechanically. A tendon transmission maps actuator displacement into multiple joint coordinates, while passive compliance allows the hand to adapt during contact. SoftHand 2 further shows that tendon routing and friction-related transmission effects can create additional low-dimensional motion directions.

The proposed FMAS model extends this idea to the cable-driven origami hand in this project. A two-ended closed-loop tendon has direction-dependent transmission: pulling from side A and side B creates different tension distributions. FMAS derives two one-sided transmissions from the routed tendon path and connects them to MuJoCo numerical contact simulation.

### Literature Comparison Table Text

```text
Postural synergy:
Low-dimensional coordinates approximate high-dimensional hand posture.
Role in this project: motivates sigma-space control.

Soft synergy:
The synergy command is a reference, and contact can change the realized posture.
Role in this project: explains why q_ref(t) and q(t) should be compared.

Adaptive synergy:
Mechanical transmission maps actuator displacement into joint motion.
Role in this project: provides the R, E -> B_sigma mapping structure.

SoftHand 2:
Tendon routing and transmission effects can enrich the control basis.
Role in this project: motivates multiple physically meaningful synergy directions.

FMAS, this project:
Two one-sided transmissions R_A and R_B model a two-ended closed-loop tendon.
Role in this project: converts origami tendon routing into MuJoCo simulation inputs.
```

### Key Formula Text

```text
Postural synergy:
q = S sigma
```

```text
Soft synergy with contact:
q = S sigma - C J^T f_ext
```

```text
Adaptive transmission:
R q = x
tau = R^T tau_M
```

### 图片内容说明

图片 1：文献脉络图。画成从 `Postural Synergy` 到 `Soft Synergy`，再到 `Adaptive Synergy / SoftHand`，最后到 `FMAS for Origami Tendon Hand` 的水平箭头。每个节点下方只写一句物理含义。

图片 2：一个小示意图，对比固定传动矩阵 `R` 和双端传动 `R_A / R_B`。左边画一个普通单向 tendon transmission，右边画双端闭环 tendon，从 A 和 B 两侧分别拉动。重点表现“路径方向不同，所以传动不同”。

---

## 3. Methodology

### 中文排版说明

放在海报中间核心区域，占据最显眼位置。该部分是评分中方法分最高的区域，应当最清楚、最有逻辑。

建议做成三层垂直流程：

1. Geometry and model generation
2. FMAS transmission mapping
3. MuJoCo numerical execution

每层左侧放简短标题，右侧放关键公式或步骤。中间用箭头连接，强调 `.ohd` 文件如何进入物理仿真。

这部分必须突出物理仿真边界：几何/运动学仿真用于生成参考运动和解释传动效果；MuJoCo 数值层使用 `mj_step` 推进；控制律只含位置误差比例力和可选协同力输入；在本次数值实验中不使用 damper、joint damping、velocity actuator 或手写 `-D qdot`。

### Poster English Text

The simulation method has three layers. First, the `.ohd` design file defines the origami hand geometry: faces, creases, joints, holes, actuators, and routed tendons. The software reconstructs the hand topology, supports geometric/kinematic simulation, and exports URDF/STL geometry for numerical simulation.

Second, the FMAS layer converts the two-ended tendon routing into a distribution matrix. Pulling from side A and side B gives two one-sided transmissions, `R_A` and `R_B`. With the joint stiffness matrix `E`, these transmissions generate two physically interpretable synergy bases. Their sum gives a closing mode, and their difference gives a redistribution mode.

Third, the MuJoCo layer converts the FMAS reference into generalized joint forces. Driver samples are interpolated over time, the distribution matrix generates `q_ref(t)`, and MuJoCo advances the articulated model with `mj_step`. Contact count, contact force, joint state, target state, torque, screenshots, and GIF frames are recorded.

For the numerical dynamics experiments reported here, the MJCF path intentionally excludes damper elements, velocity actuators, viscous joint damping, tendon velocity friction, and any manually written `-D q_dot` term. The goal is to isolate how FMAS-derived inputs drive MuJoCo time integration and contact response.

### Method Flow Text

```text
.ohd design and simulation definition
-> reconstruct geometry, topology, and tendon routing
-> run geometric/kinematic reference simulation
-> export or load URDF/STL
-> build FMAS distribution B_sigma
-> compute q_ref(t) = B_sigma sigma(t)
-> apply generalized torque tau(t)
-> MuJoCo mj_step for numerical dynamics/contact
-> save q(t), q_ref(t), ncon, contact force, screenshots, GIF
```

### Key Formula Text

```text
One-sided tendon transmissions:
R_A, R_B
```

```text
FMAS modes:
S_close = S_A + S_B
S_diff  = S_A - S_B
```

```text
MuJoCo reference:
q_ref(t) = B_sigma sigma(t)
```

```text
Applied generalized force:
tau(t) = Kp (q_ref(t) - q(t)) + force_scale B_sigma f_sigma(t)
```

```text
MuJoCo dynamics, solved by the engine:
M(q) qddot + h(q, qdot) = tau(t) + J_c(q)^T lambda
```

### 图片内容说明

图片 1：主方法流程图。内容为 `.ohd` 文件进入 parser，生成 topology、几何/运动学参考、URDF/STL 和 tendon routing；随后进入 FMAS 层，得到 `R_A, R_B` 和 `B_sigma`；最后进入 MuJoCo，输出 `q(t)`、contact logs、screenshots 和 GIF。该图应是整张海报最重要的图之一。

建议使用：

```text
docs/Physical Simulation/poster/fmas_method_pipeline.svg
```

图片 2：FMAS 物理示意图。画一根闭环腱绳穿过多个折痕或孔洞，左端标为 A，右端标为 B。用两种颜色表示 A 端拉动和 B 端拉动时的张力传播方向，旁边标注 `R_A` 和 `R_B`。

图片 3：控制律示意图。左侧是 driver time series `sigma(t)`，中间是 `B_sigma`，右侧是 `q_ref(t)`，再通过比例广义力进入 MuJoCo。重点说明这是物理仿真输入，而不是直接设定最终姿态。

建议使用：

```text
docs/Physical Simulation/poster/fmas_control_mapping.svg
```

---

## 4. Experimental Results

### 中文排版说明

放在海报右侧或右下，占据较大面积。该部分要突出“定性结果 + 定量结果 + 可复现输出”。课程是物理仿真，因此不能只放好看的截图，要同时放误差曲线、接触统计和日志输出说明。

建议分成三组结果：

1. Geometry-to-dynamics free-space comparison
2. Contact grasping simulation demos
3. Reproducibility and audit outputs

截图选择要克制。优先使用 step demo 对比图、cylinder 接触交互截图、scanned mug 接触交互截图。sphere 结果接触力偏大，不建议作为主图。

### Poster English Text

The simulation pipeline was tested in free space using a three-finger origami gripper with 9 hinge joints and 2 FMAS drivers. The run contains both the geometry-derived reference trajectory and the MuJoCo time-integrated trajectory. The simulation ran for 1.2 s with 600 MuJoCo steps at `dt = 0.002 s`. The MuJoCo state follows the reference trend while showing the expected difference between reference motion and numerical dynamics.

For the free-space step demo, the RMS target-state difference is `0.1848 rad`, and the maximum absolute difference is `0.5661 rad`.

The contact-interaction demos evaluate the same FMAS-driven hand with external objects. With a cylinder object, the simulation records 158 contact steps, up to 5 simultaneous contacts, and a maximum total contact force of 4.415. With a scanned mug object, the poster-ready run records 114 contact steps, up to 5 simultaneous contacts, and a maximum total contact force of 0.999.

Each run automatically writes a CSV log, compressed NPZ trajectory, summary JSON, generated MJCF file, rendered screenshots, and optional process GIF. For the reported MuJoCo numerical experiments, the generated MJCF is checked to contain no damper, damping, or velocity actuator terms.

### Result Table Text

```text
Free-space step demo:
Duration: 1.2 s
Steps: 600
dt: 0.002 s
Joints: 9
Drivers: 2
Distribution: FMAS
RMS error: 0.1848 rad
Max error: 0.5661 rad
```

```text
Contact interaction demos:
Cylinder:
Contact steps: 158
Max contacts: 5
Max total contact force: 4.415

Scanned mug:
Contact steps: 114
Max contacts: 5
Max total contact force: 0.999
```

```text
Generated outputs:
CSV trajectory log
NPZ numerical arrays
summary JSON
MJCF audit file
MuJoCo screenshots
optional process GIF
```

### 图片内容说明

图片 1：自由空间 step demo 的曲线对比图。重点展示 `q_ref(t)` 和 `q(t)` 的关系。该图对应项目中“几何参考运动”和“数值动力学响应”的连接。

建议使用：

```text
docs/Physical Simulation/poster/generated_figures/poster_ready/fig1_free_space_reference_vs_mujoco.png
```

图片 2：自由空间两个姿态截图。一个是早期 common-mode closing 阶段，一个是后期 differential FMAS posture 阶段。用于展示低维 driver 如何产生不同手型。

建议使用：

```text
docs/Physical Simulation/poster/generated_figures/poster_ready/fig2_free_space_two_postures.png
```

图片 3：cylinder 接触交互截图。重点展示手和圆柱进入接触阶段后，姿态由 FMAS 驱动输入和 MuJoCo 接触约束共同决定。不要把这张图说成稳定最终抓取；更准确的图注应是 contact interaction 或 contact-constrained response。

建议使用：

```text
docs/Physical Simulation/poster/generated_figures/poster_ready/fig3_cylinder_contact_grasp.png
```

图片 4：scanned mug 接触交互截图或 GIF 关键帧。重点展示非简单几何体也能进入 MuJoCo 接触仿真，说明物理仿真接口具有扩展性。

建议使用：

```text
docs/Physical Simulation/poster/generated_figures/poster_ready/fig4_scanned_mug_contact_grasp.png
docs/Physical Simulation/poster/generated_figures/poster_ready/fig4_scanned_mug_process.gif
```

---

## 5. Conclusion

### 中文排版说明

放在海报底部或右下角窄条区域。Conclusion 不要再堆公式，也不要重复方法细节。用三到四句话收束贡献，最后给一个简短 future work。

重点要回到物理仿真课程：项目贡献不是单纯做了 GUI 或截图，而是形成了包含几何/运动学仿真、FMAS 传动建模、MuJoCo 数值积分和接触响应分析的完整仿真部分。

### Poster English Text

This project presents a physics-aware simulation framework for a cable-driven origami dexterous hand. The geometric layer converts `.ohd` designs into hand topology, URDF/STL geometry, tendon routing, and reference synergy motion.

The proposed FMAS model derives physically interpretable closing and redistribution modes from the two-ended closed-loop tendon transmission, instead of prescribing a fixed synergy basis.

The MuJoCo layer converts these model-derived inputs into numerical time integration and contact simulation, producing joint trajectories, contact statistics, screenshots, GIFs, and auditable MJCF files.

Future work will refine contact geometry, calibrate material and contact-force parameters, and extend FMAS distributions toward task-space grasp objectives.

### One-Line Takeaway

```text
The contribution is an integrated simulation pipeline from origami hand geometry and FMAS tendon modeling to MuJoCo contact-aware dynamics.
```

### 图片内容说明

图片 1：结论区可以放一个小型 pipeline 图标式总结，不需要复杂。内容为 `Design -> Geometry/Kinematics -> FMAS -> MuJoCo -> Contact-aware results`。

图片 2：如果版面空间很小，只放一个代表性接触交互截图即可，建议用 scanned mug 或 cylinder，而不是自由空间截图。它更能体现物理仿真和接触求解。
