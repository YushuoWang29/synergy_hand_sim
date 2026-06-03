# SoftHand 2 数值方法提炼：面向无阻尼 MuJoCo 升级

本文档提炼 Della Santina 等人在 *Toward Dexterous Manipulation With Augmented Adaptive Synergies: The Pisa/IIT SoftHand 2* 中与数值仿真升级相关的数学结构。目标是为 `synergy_hand_sim` 的 MuJoCo 数值仿真提供理论接口，而不是完整复现 SoftHand 2 的具体机械结构。

## 1. 可采用的核心思想

SoftHand 2 的关键贡献是将腱绳传动中的摩擦从扰动转化为可设计的第二驱动方向。对于低维协同驱动手，第一输入主要产生全手协调闭合，第二输入通过腱绳相对滑动改变张力分布，从而产生手指间的相对重构。

这与本项目的折纸灵巧手相容：折纸手不需要复制 SoftHand 2 的滑轮机构，但可以沿用“驱动输入 - 传动分布 - 关节响应”的矩阵表达，并将腱路顺序、折痕刚度和等效摩擦分布作为用户可配置的机械智能参数。

## 2. 基础协同与软协同模型

人手姿态协同通常写为

```math
q = S\sigma
```

其中 `q` 为关节构型向量，`S` 为协同矩阵，`\sigma` 为低维协同坐标。Soft synergy 模型进一步考虑外力与柔顺性：

```math
q = S\sigma - C J^T f_{ext}
```

其中 `C` 为手部等效柔顺矩阵，`J^T f_{ext}` 为接触外力映射到关节空间后的广义力。

对 MuJoCo 升级而言，这一式子不需要手工求接触平衡。更合理的实现方式是：

- 用 `S\sigma` 或由传动矩阵生成的 `q_ref` 作为自由空间参考构型；
- 由 MuJoCo 负责接触、重力和约束求解；
- 日志中输出实际 `q` 与参考 `q_ref` 的差异，作为几何仿真与数值仿真的对比。

## 3. Adaptive synergy 的传动矩阵形式

SoftHand 系列用传动矩阵 `R` 描述腱绳位移和关节变量之间的关系：

```math
R q = x
```

其中 `x` 为腱绳/驱动器位移。由运动静力对偶关系可得关节力矩：

```math
\tau = R^T \tau_M
```

考虑关节弹性和外力，平衡关系为

```math
J^T f_{ext} = R^T \tau_M - E q
```

无外力自由闭合时，协同输入到关节构型的映射可写为

```math
q = E^{-1} R^T (R E^{-1} R^T)^{-1} x
```

因此可定义

```math
S_R = E^{-1} R^T (R E^{-1} R^T)^{-1}
```

使得

```math
q = S_R x
```

在本项目中，`R` 可由 `.ohd` 文件中的腱绳路径、折痕顺序、折痕力臂或等效传动比生成；`E` 可由折痕刚度配置生成。该形式是后续 `.ohd -> distribution -> MuJoCo target/control` 的主要数学接口。

## 4. Augmented adaptive synergy 的第二输入

SoftHand 2 将一根腱绳分为多个张力段，并用 `s` 表示腱绳两端之间的相对滑动。其基本思想是：普通拉力输入 `u_1` 产生第一方向，滑动或差动输入 `u_2` 通过摩擦改变张力分布，产生第二方向。

在平衡、简化摩擦模型下，论文将关节空间力平衡写为

```math
J^T f_{ext} = R^T u_1 + R_f^T u_2 - E q
```

其中

```math
u_1 = \tau_M
```

```math
u_2 = \tanh(2\dot{s})
```

并有

```math
R^T = -\bar{R}^T e_v
```

```math
R_f^T = -\bar{R}^T M^{-1} V_{max} e_v
```

这里 `R` 对应普通协同闭合方向，`R_f` 对应由摩擦和腱路顺序产生的第二协同方向。

对本项目的无阻尼 MuJoCo 版本，应避免直接使用速度相关的 `\dot{s}` 作为摩擦动力学变量。更稳妥的工程化简化是：

```math
q_{ref} = B_\sigma \sigma
```

其中

```math
B_\sigma =
\begin{bmatrix}
b_1 & b_2 & \cdots & b_k
\end{bmatrix}
```

`b_1` 可由普通传动矩阵 `R` 和刚度矩阵 `E` 生成，`b_2` 可由腱路顺序、等效摩擦权重或用户指定分布生成。这样保留了 SoftHand 2 的“摩擦增强第二协同方向”思想，但不引入速度阻尼项。

## 5. 腱路顺序作为设计变量

SoftHand 2 用置换矩阵 `P` 描述腱绳经过各滑轮/关节的顺序变化。论文指出，更改腱路顺序会保持第一方向基本不变，但会改变摩擦增强方向 `R_f`。抽象为本项目的语言：

```math
R_f = f(R, P, \mu)
```

其中 `P` 是腱路顺序，`\mu` 是等效摩擦/传动损失参数。对折纸灵巧手而言，这正好对应“可更换腱路编程掌板”：掌板改变腱绳经过各手指/折痕的顺序，从而改变第二协同方向。

## 6. 必须排除的内容

论文 Section V 的完整数值模型包含以下内容，和本项目当前约束冲突，不应进入实现：

- 粘性摩擦项：速度线性相关的 tendon friction；
- 关节速度摩擦项：例如 `F \dot{q}`；
- 阻尼器或动态协同中的速度阻尼；
- 用于实体电机闭环的 PD/PI 控制器；
- 依赖 `\dot{s}` 的动态摩擦积分模型。

对应地，MuJoCo 模型中应显式避免：

- joint/body/tendon 的 `damping` 非零配置；
- velocity actuator 作为默认控制方式；
- damper 元件；
- 任何由本项目代码手动加入的 `-D \dot{q}` 或类似阻尼力。

接触摩擦属于物体接触求解的一部分，可以保留；但腱绳内部的速度相关摩擦暂不建模。

## 7. 面向代码实现的推荐抽象

建议将数值仿真接口拆成三层：

1. **Distribution layer**  
   从 `.ohd` 或用户配置中得到分布矩阵 `B_\sigma`。分布类型可以包括：
   - `joint_space`：直接指定每个协同输入到各关节的权重；
   - `transmission`：由腱路和刚度生成 `E^{-1}R^T(RE^{-1}R^T)^{-1}`；
   - `custom`：用户直接提供矩阵；
   - `endpoint`：以后可扩展为末端轨迹/任务空间分布。

2. **Synergy command layer**  
   根据时间序列输入生成协同坐标：

```math
\sigma(t) =
\begin{bmatrix}
\sigma_1(t) & \sigma_2(t) & \cdots & \sigma_k(t)
\end{bmatrix}^T
```

并计算自由空间参考：

```math
q_{ref}(t) = B_\sigma \sigma(t)
```

3. **MuJoCo execution layer**  
   将 `q_ref(t)` 或等效广义力输入到 MuJoCo。最小可运行版本优先采用位置型 joint actuator 跟踪 `q_ref`，并将所有阻尼设为零。更高物理保真版本可改为 tendon actuator，但仍不加入阻尼器。

## 8. 对本项目的结论

SoftHand 2 对本项目最有价值的不是完整动力学方程，而是两点：

1. 协同输入可以通过 `R`、`E` 和传动结构映射为关节空间运动；
2. 腱路顺序和等效摩擦分布可以被设计成第二协同方向的来源。

因此，后续 MuJoCo 升级应采用“通用分布矩阵 + 可选 SoftHand2-inspired 生成器”的结构：先保证 `.ohd` 能驱动 MuJoCo 数值仿真，再逐步增强腱路、接触和抓取评估。
