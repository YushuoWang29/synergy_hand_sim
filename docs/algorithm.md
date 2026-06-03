# 从舵机位移到关节角度：`run_synergy_from_ohd.py` 完整计算链路

> 本文档以**叙事**方式完整讲解：当你在 MuJoCo 仿真器中拖动 Motor A 的滑块、给了一个位移值之后，程序内部到底经过哪些步骤，最终算出手指每个关节的角度。

> ⚠️ **诚实声明**：当前模型**不是第一性原理物理模型**，而是一个**一阶现象学近似**。R 矩阵中的 Capstan 指数衰减 $e^{-\beta k}$ **不是**从"力→力矩→弹簧平衡"推导出来的结果，而是用指数函数作为拟合基函数去**复现实验观察到的 U-shape 位移分布**。它"看起来有物理基础"只是因为：在线性弹簧+线性张力梯度的准静态假设下，平衡解恰好呈现指数形式。$\beta$ 不是物理测量的摩擦系数，而是一个**纯拟合参数**。严格物理建模应该显式求解力-位移耦合方程组，当前代码跳过了这个步骤。

---

## 目录

1. [前置问题：滑块值到底代表什么？](#1-前置问题滑块值到底代表什么)
2. [第一层：从 .ohd 文件中提取了什么？](#2-第一层从-ohd-文件中提取了什么)
3. [第二层：R_A / R_B 是怎么算出来的？](#3-第二层-ra--rb-是怎么算出来的)
4. [第三层：怎么从 R 到 S（协同矩阵）？](#4-第三层怎么从-r-到-s协同矩阵)
5. [第四层：运行时的 Slack 钳位](#5-第四层运行时的-slack-钳位)
6. [第五层：状态判断——现在哪个马达在工作？](#6-第五层状态判断现在哪个马达在工作)
7. [第六层-情况A：只拉 Motor A（仅一个马达激活）](#7-第六层-情况a只拉-motor-a仅一个马达激活)
8. [第六层-情况B：两个马达都在拉（双马达同向）](#8-第六层-情况b两个马达都在拉双马达同向)
9. [第六层-情况C：两个马达都松弛](#9-第六层-情况c两个马达都松弛)
10. [第七层：关节角度限位](#10-第七层关节角度限位)
11. [第八层：把结果送给 MuJoCo](#11-第八层把结果送给-mujoco)
12. [整个流程的金字塔总结](#12-整个流程的金字塔总结)
13. [动态协同模式有什么不同？](#13-动态协同模式有什么不同)
14. [附录：关键参数表](#14-附录关键参数表)

---

## 1. 前置问题：滑块值到底代表什么？

MuJoCo 右侧面板上 Motor A 的滑块，单位是 **弧度**。拖动滑块改变的是**电机旋转了多少角度**。

```
滑块值 1.0 → Motor A 转了 1 弧度（≈57.3°）
滑块值 3.14 → Motor A 转了 π 弧度（半圈）
滑块值 6.28 → Motor A 转了一圈
```

**这个值是位移，不是力。**

为什么？因为：
- 代码里变量名叫 `theta1_rad`，`theta` 就是角度
- 控制范围 `ctrl_range_deg=3600°`（10 整圈）——这是位置范围，力不会有这种范围
- 后面的 slack 钳位做了 `max(0, θ)`——负角度意味着"电机反转放松腱绳"，这对应位移的直觉

> **物理对应关系**：电机转角 → 腱绳被拉动的长度 = 电机半径 × 电机转角。滑块值实际上是"电机的角位移"。

---

## 2. 第一层：从 .ohd 文件中提取了什么？

在拖动滑块之前，程序启动时先做了准备工作。以下所有东西都是从 `.ohd` 文件解析出来的。

### 2.1 关节系统

```python
joints, jid_to_idx = get_joint_list(design)
```

遍历设计中所有折痕，找到每条 VALLEY 和 MOUNTAIN 折痕，给每条分配一个索引（0, 1, 2, ...）。

```
折痕 ID 102（谷折）→ 关节索引 0
折痕 ID 104（谷折）→ 关节索引 1
折痕 ID 106（谷折）→ 关节索引 2
...
```

建立映射：
```
jid_to_idx = {102: 0, 104: 1, 106: 2, ...}
```

这个映射的意义：**知道一个滑轮装在哪个折痕上，就知道它影响哪个关节**。

### 2.2 腱绳路径

每条腱绳定义了一个 `pulley_sequence`，即一串元素 ID：

```
[-1, 0, 1, 2, 3, 4, -2]
  ↑                        ↑
 Motor A                  Motor B
（起点）                  （终点）
```

中间的非负整数是滑轮 ID，负数 (<0) 是特殊元素：
- `-1` = Motor A（舵机 A）
- `-2` = Motor B（舵机 B）
- `<= -100` = 孔（Hole，替代滑轮）
- `<= -200` = 阻尼器（Damper，动态协同用）

**一条腱绳就是一根绳子，从 Motor A 出发，绕过若干个滑轮/孔，最终连到 Motor B。**

### 2.3 每个滑轮/孔的属性

遍历路径中的每个元素时，查询：

```python
def _get_element_radius(eid, design):
    if eid >= 0:                    # 滑轮
        return design.pulleys[eid].radius       # 默认 3.5
    elif eid <= -100:               # 孔
        return design.holes[eid].plate_offset   # 默认 3.0
    return 0.0
```

```python
def _get_element_joint_idx(eid, design, jid_to_idx):
    # 滑轮 → 找到 attached_fold_line_id → 查 jid_to_idx → 得到关节索引
    # 孔   → 找到 attached_fold_line_id → 查 jid_to_idx → 得到关节索引
    return j_idx  # 或者 None（如果没有关联关关节）
```

### 2.4 关节刚度

每个关节（折痕）有一个 stiffness 参数，默认值是 1.0。所有刚度构成对角矩阵 $E = \text{diag}(e_0, e_1, ..., e_{n-1})$。

---

## 3. 第二层：R_A / R_B 是怎么算出来的？

这是**最核心的计算**。所有后续的一切都基于这个结果。

### 3.1 Capstan 效应的来源——指数衰减的定性动机

绳子绕过滑轮时，因为有摩擦力，出绳端的张力比入绳端小：

$$T_{\text{out}} = T_{\text{in}} \cdot e^{-\beta}$$

这是 Capstan 方程。$\beta$ 是"有效摩擦系数"，默认 0.09。

如果我们假设系统的稳态位移分布和张力分布的衰减模式相同（在准静态+线性弹簧下这个假设近似成立），那么关节位移从近到远也呈现类似的指数衰减。这个"巧合"提供了指数函数的合理性——**但不是从物理推导出来的，而是事后用指数去拟合实验观察到的形状**。

### 3.2 正式定义

对腱绳路径上的第 $k$ 个元素：

- 到 Motor A 的距离（按元素个数计）：$d_A = k$
- 到 Motor B 的距离（按元素个数计）：$d_B = N - 1 - k$

**仅 Motor A 拉紧时**的权重：

$$w_A(k) = e^{-\beta \cdot k}$$

关节 $j$ 上的有效权重：

$$R_A^{[j]} = \sum_{k \in \text{joint }j} r_k \cdot e^{-\beta \cdot k}$$

**仅 Motor B 拉紧时**：

$$R_B^{[j]} = \sum_{k \in \text{joint }j} r_k \cdot e^{-\beta \cdot (N-1-k)}$$

**两个马达同时拉紧时**，两端叠加再平均：

$$R_{\text{avg}}^{[j]} = \frac{R_A^{[j]} + R_B^{[j]}}{2}$$

### 3.3 数值例子

5 个关节，$\beta = 0.09$，所有 $r_k = 1$，$N = 6$：

```
元素位置 k:   0      1      2      3      4      5
            MotorA  pulley0 pulley1 pulley2 pulley3 MotorB
                     ↓       ↓       ↓       ↓
                   关节0    关节1    关节2    关节3
```

$w_A(k) = e^{-0.09k}$：
```
k=0: 1.000
k=1: 0.914
k=2: 0.835
k=3: 0.764
k=4: 0.699
k=5: 0.639 (Motor B 端)
```

$R_A$（仅 Motor A 拉紧）：
```
关节0 (k=1): r_1 · w_A(1) = 1 × 0.914 = 0.914
关节1 (k=2): r_2 · w_A(2) = 1 × 0.835 = 0.835
关节2 (k=3): r_3 · w_A(3) = 1 × 0.764 = 0.764
关节3 (k=4): r_4 · w_A(4) = 1 × 0.699 = 0.699
```

**关键观察**：从 Motor A 向远端，值越来越小——这就是"单调衰减"。

$R_B$（仅 Motor B 拉紧）：
```
关节3 (k=4, 即从B端的 d_B=1): w_B = e^{-0.09×1} = 0.914
关节2 (k=3, d_B=2):           w_B = e^{-0.09×2} = 0.835
关节1 (k=2, d_B=3):           w_B = e^{-0.09×3} = 0.764
关节0 (k=1, d_B=4):           w_B = e^{-0.09×4} = 0.699
```

$R_{\text{avg}} = (R_A + R_B) / 2$：
```
关节0: (0.914 + 0.699) / 2 = 0.807
关节1: (0.835 + 0.764) / 2 = 0.800
关节2: (0.764 + 0.835) / 2 = 0.800
关节3: (0.699 + 0.914) / 2 = 0.807
```

**中间关节权重最小（0.800），两端关节最大（0.807）**——这就是 U 形分布。

### 3.4 多条腱绳怎么办？

如果设计中有两条腱绳（比如手指两侧各一条），它们共享相同的 Motor A/B，所以：

```python
R_A = R_A[drive_indices].mean(axis=0, keepdims=True)  # (1, n)
R_B = R_B[drive_indices].mean(axis=0, keepdims=True)  # (1, n)
```

将多条腱绳的平均值作为最终 R_A 和 R_B。

### 3.5 再强调一次：这是现象学拟合，不是物理推导

> ⚠️ 上述 $w_A(k) = e^{-\beta k}$ 的写法**看起来**像是在做 Capstan 力衰减，但实际上我们直接把它当成了"位移分配权重"。严格物理链路应该是：
>
> $$
> \theta_A \xrightarrow{\text{腱绳刚度}} T_0 \xrightarrow{\text{Capstan}} T_k \xrightarrow{\text{力矩}} \tau_j \xrightarrow{\text{弹簧}} q_j
> $$
>
> 当前代码跳过了中间的 $T_0, T_k, \tau_j$，直接假设 $q_j \propto e^{-\beta k}$。这等价于用指数函数拟合实验数据。$\beta$ 是拟合参数，不是物理摩擦系数。

---

## 4. 第三层：怎么从 R 到 S（协同矩阵）？

现在我们有 $R$ 矩阵（例如 $R_A$ 是 $1 \times n$ 的行向量），但我们还需要关节刚度 $E$。

### 4.1 基础方程

传动约束：腱绳位移 $\sigma$ 等于各关节滑轮位移之和：

$$R q = \sigma$$

静力平衡：腱绳张力 $f$ 产生的关节力矩 = 关节弹簧恢复力矩：

$$R^\top f = E q$$

其中 $E = \text{diag}(e_0, ..., e_{n-1})$。

### 4.2 联立求解

把两个方程写成矩阵形式：

$$
\begin{bmatrix}
-E & R^\top \\
R & 0
\end{bmatrix}
\begin{bmatrix}
q \\ f
\end{bmatrix}
=
\begin{bmatrix}
0 \\ \sigma
\end{bmatrix}
$$

从 $R q = \sigma$ 和 $E q = R^\top f$：

$$R (E^{-1} R^\top f) = \sigma$$

$$(R E^{-1} R^\top) f = \sigma$$

$$f = (R E^{-1} R^\top)^{-1} \sigma$$

回代到 $q = E^{-1} R^\top f$：

$$q = \underbrace{E^{-1} R^\top (R E^{-1} R^\top)^{-1}}_{S} \sigma$$

### 4.3 代码中的实际执行

```python
E_inv = np.diag(1.0 / np.diag(self.E))   # 刚度矩阵的逆
RE = self.R @ E_inv @ self.R.T            # (k, n) × (n, n) × (n, k) = (k, k)
RE_inv = np.linalg.pinv(RE)               # 伪逆，处理可能的秩亏缺
self.S = E_inv @ self.R.T @ RE_inv        # (n, n) × (n, k) × (k, k) = (n, k)
```

**为什么用 pinv 而不是 inv？**

当 $R$ 只有 1 行时（$k=1$），$R E^{-1} R^\top$ 是一个 $1 \times 1$ 的标量，直接取倒数（inv）和 pinv 结果相同。但当有多行且存在共线性时，inv 报错而 pinv 自动处理。

### 4.4 所以 $S_A$ 是什么？

$S_A$ 是一个 $n \times 1$ 的列向量：

$$S_A = E^{-1} R_A^\top (R_A E^{-1} R_A^\top)^{-1}$$

同样：
- $S_B$ = $E^{-1} R_B^\top (R_B E^{-1} R_B^\top)^{-1}$
- $S_{\text{avg}}$ = $E^{-1} R_{\text{avg}}^\top (R_{\text{avg}} E^{-1} R_{\text{avg}}^\top)^{-1}$

### 4.5 数值例子（续前）

设 $E = I$（所有刚度=1），$R_A = [0.914,\ 0.835,\ 0.764,\ 0.699]$：

$$S_A = I^{-1} R_A^\top (R_A I^{-1} R_A^\top)^{-1}$$

先算分母：$R_A R_A^\top = 0.914^2 + 0.835^2 + 0.764^2 + 0.699^2 = 2.614$

$$S_A = \frac{R_A^\top}{2.614} = [0.350,\ 0.319,\ 0.292,\ 0.267]^\top$$

**物理意义**：Motor A 转了 1 弧度时，关节 0 弯 0.350 弧度，关节 1 弯 0.319 弧度，... 从近到远依次减小。

---

## 5. 第四层：运行时的 Slack 钳位

现在准备完毕。用户开始在 MuJoCo 中拖动滑块。

### 5.1 进入回调函数

每次滑块变化，调用：

```python
synergy_callback(theta1_rad, theta2_rad, speed_rad_s=0.0)
```

`theta1_rad` = Motor A 的滑块值（弧度）
`theta2_rad` = Motor B 的滑块值（弧度）

### 5.2 第一步：负值截断

如果用户在 ui 中把滑块拖到负值区（比如 Motor A = -2.0），代码做：

```python
theta_A = max(0.0, float(theta1_rad))   # -2.0 → 0.0
theta_B = max(0.0, float(theta2_rad))
```

**为什么？**

舵机只能**拉**腱绳，不能**推**。正的转角 = 腱绳被拉紧。负的转角在物理上对应"舵机反转把腱绳松开"。但松开后的腱绳已经松弛了，不会产生任何驱动力。所以负值等价于 0（不工作）。

这个操作叫做 **slack 钳位**——负值被钳到 0。

> **重要理解**：slack 钳位把模型的输入从可以正负双向的线性输入空间，变成只有非负的单向输入空间。这是当前模型和 Della Santina 2018 论文的**最根本区别之一**——论文假设腱绳可以受推（双向），实际物理中腱绳只能受拉（单向）。

---

## 6. 第五层：状态判断——现在哪个马达在工作？

钳位后，程序判断当前处于哪种状态：

```python
epsilon = 1e-10

both_active = (theta_A > epsilon and theta_B > epsilon)
only_A = (theta_A > epsilon and theta_B <= epsilon)
only_B = (theta_B > epsilon and theta_A <= epsilon)
# 两者都 ≤ epsilon → 双松弛
```

| 状态 | $\theta_A$ | $\theta_B$ | 含义 |
|------|-----------|-----------|------|
| 双马达同向(both_active) | > 1e-10 | > 1e-10 | 两个舵机都在拉 |
| 仅 Motor A(only_A) | > 1e-10 | = 0 | 只有 A 在工作，B 松弛 |
| 仅 Motor B(only_B) | = 0 | > 1e-10 | 只有 B 在工作，A 松弛 |
| 双松弛 | = 0 | = 0 | 都不工作，手不动 |

每种状态使用**不同的传动矩阵**来计算关节角度。

---

## 7. 第六层-情况A：只拉 Motor A（仅一个马达激活）

这是最简单的情况。

### 7.1 计算

```python
q = S_A @ np.array([theta_A])  # 一个 1×1 "矩阵" × 标量 = 标量 × 向量
```

展开：

$$q = S_A \cdot \theta_A$$

$S_A = [s_0, s_1, ..., s_{n-1}]^\top$，$\theta_A$ 是标量，所以：

$$q_j = s_j \cdot \theta_A$$

### 7.2 物理含义

- $S_A$ 在构建阶段已固定（取决于滑轮布局、刚度分布、$\beta$）
- $\theta_A$ 是唯一可变参数（滑块值）
- 所有关节的比例关系是固定的——$q_0 : q_1 : q_2 : ... = s_0 : s_1 : s_2 : ...$
- 比例关系从近到远**单调递减**（因为 $R_A$ 的指数衰减权重保证了这一点）

### 7.3 数值例子

沿用前面的 $S_A = [0.350, 0.319, 0.292, 0.267]^\top$：

| $\theta_A$ (滑块) | $q_0$ (关节0) | $q_1$ | $q_2$ | $q_3$ |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0 | 0 | 0 | 0 |
| 1.0 | 0.350 | 0.319 | 0.292 | 0.267 |
| 2.0 | 0.700 | 0.638 | 0.584 | 0.534 |
| 5.0 | 1.750 | 1.595 | 1.460 | 1.335 |

近端关节（关节 0）弯曲最多，远端关节（关节 3）弯曲最少——物理上合理。

---

## 8. 第六层-情况B：两个马达都在拉（双马达同向）

这是更复杂的情况。SDAS 默认模式（非 `--dynamic`）使用 **Schur 补公式**（docs/SDAS.md §2.3）处理双马达同时拉动。

### 8.1 SDAS 模型的双马达求解

当两个马达同时拉动（$\theta_A > 0, \theta_B > 0$），两端腱绳都绷紧，两个约束 $R_A \cdot q = \theta_A$ 和 $R_B \cdot q = \theta_B$ **同时生效**。代码调用 `SDASModel.solve_motors()`，内部自动选择 `_solve_both_active()`：

```python
# solve_motors() 检测到 both_active → 调用 _solve_both_active()
q = S_A_schur · theta_A + S_B_schur · theta_B
```

同时在 σ/σ_f 语言下等价为：

```python
sigma = (theta_A + theta_B) / 2.0      # 同模（共模）
sigma_f = (theta_A - theta_B) / 2.0    # 差模

q = (S_A_schur + S_B_schur) · σ + (S_A_schur - S_B_schur) · σ_f
```

- **$\sigma$（同模/共模）**：两个马达的平均拉力，代表"手整体弯曲多少"
- **$\sigma_f$（差模）**：两个马达的拉力差，代表"形状偏移"

如果 $\theta_A = \theta_B$，则 $\sigma_f = 0$，完全对称弯曲。
如果 $\theta_A > \theta_B$，则 $\sigma_f > 0$，偏向 Motor A 侧。

### 8.2 Schur 补的数学推导（对应 docs/SDAS.md Eq. 2.38-2.41）

写出包含两个约束的扩展系统：

$$
\begin{bmatrix}
-\mathbf{E} & \mathbf{R}_A^\mathsf{T} & \mathbf{R}_B^\mathsf{T} \\
\mathbf{R}_A & 0 & 0 \\
\mathbf{R}_B & 0 & 0
\end{bmatrix}
\begin{bmatrix} \mathbf{q} \\ F_A \\ F_B \end{bmatrix}
=
\begin{bmatrix} \mathbf{J}^\mathsf{T} \mathbf{f}_c \\ \theta_A \\ \theta_B \end{bmatrix}.
$$

从第 1 行解出 $\mathbf{q} = \mathbf{E}^{-1}(\mathbf{R}_A^\mathsf{T} F_A + \mathbf{R}_B^\mathsf{T} F_B - \mathbf{J}^\mathsf{T} \mathbf{f}_c)$，代入第 2、3 行得到 $2\times2$ 系统：

$$
\begin{bmatrix} a & b \\ b & c \end{bmatrix}
\begin{bmatrix} F_A \\ F_B \end{bmatrix}
=
\begin{bmatrix}
\theta_A + \mathbf{R}_A \mathbf{E}^{-1} \mathbf{J}^\mathsf{T} \mathbf{f}_c \\
\theta_B + \mathbf{R}_B \mathbf{E}^{-1} \mathbf{J}^\mathsf{T} \mathbf{f}_c
\end{bmatrix},
$$

其中 $a = \mathbf{R}_A \mathbf{E}^{-1} \mathbf{R}_A^\mathsf{T}$, $b = \mathbf{R}_A \mathbf{E}^{-1} \mathbf{R}_B^\mathsf{T}$, $c = \mathbf{R}_B \mathbf{E}^{-1} \mathbf{R}_B^\mathsf{T}$。

用 Schur 补逆公式求解：

$$
\begin{bmatrix} a & b \\ b & c \end{bmatrix}^{-1}
= \frac{1}{\det}
\begin{bmatrix} c & -b \\ -b & a \end{bmatrix},
\quad \det = ac - b^2.
$$

回代得：

$$
\mathbf{q} = \mathbf{S}_A^{\text{schur}} \theta_A + \mathbf{S}_B^{\text{schur}} \theta_B + \mathbf{C}^{\text{schur}} \mathbf{J}^\mathsf{T} \mathbf{f}_c,
$$

其中：

$$
\mathbf{S}_A^{\text{schur}} = \mathbf{E}^{-1}(\mathbf{R}_A^\mathsf{T} c - \mathbf{R}_B^\mathsf{T} b) / \det,
\quad
\mathbf{S}_B^{\text{schur}} = \mathbf{E}^{-1}(-\mathbf{R}_A^\mathsf{T} b + \mathbf{R}_B^\mathsf{T} a) / \det,
$$

$$
\mathbf{C}^{\text{schur}} = \mathbf{E}^{-1} - \mathbf{S}_A^{\text{schur}} \mathbf{R}_A \mathbf{E}^{-1} - \mathbf{S}_B^{\text{schur}} \mathbf{R}_B \mathbf{E}^{-1}.
$$

**Schur 补的关键性质**：

| 性质 | $\mathbf{S}_A^{\text{schur}}$ | $\mathbf{S}_B^{\text{schur}}$ |
|:---:|:---:|:---:|
| $\mathbf{R}_A \cdot \mathbf{S} = 1$ | ✓ | 0 |
| $\mathbf{R}_B \cdot \mathbf{S} = 1$ | 0 | ✓ |

这意味着 $\mathbf{S}_A^{\text{schur}}$ 精确满足 Motor A 的约束且不干扰 Motor B 的约束——两个协同方向完全解耦。

**注意**：这里的 $\mathbf{S}_A^{\text{schur}}$ 和 §7 中的单向 $S_A$ 完全不同。$S_A$（单向公式）仅满足 $R_A \cdot S_A = 1$ 但不保证 $R_B \cdot S_A = 0$；而 $S_A^{\text{schur}}$ 同时满足 $R_A \cdot S_A^{\text{schur}} = 1$ 且 $R_B \cdot S_A^{\text{schur}} = 0$。这是单马达 vs 双马达物理情境的根本差异。

### 8.3 $R_A$ 和 $R_B$ 是如何计算的？

SDAS 模型的 $R_A$ 和 $R_B$ 直接来自 Capstan 指数衰减的**方向非对称性**（docs/SDAS.md Eq. 2.22-2.30），**不需要**引入额外的静摩擦冻结或 Rf 概念：

```python
# compute_R_one_sided() 的实现
for k, eid in enumerate(elements):
    r = get_radius(eid)
    d_A = float(k)                     # 距 Motor A 的步数
    w_A = exp(-beta * d_A)             # Capstan 衰减
    R_A[j_idx] += r * w_A

    d_B = float(N - 1 - k)             # 距 Motor B 的步数
    w_B = exp(-beta * d_B)             # Capstan 衰减
    R_B[j_idx] += r * w_B
```

当路径对称时，$R_A \neq R_B$（除非 $\beta = 0$），这种天然非对称性替代了旧的 Rf/静摩擦冻结模型。理想无摩擦（$\beta \to 0$）时，$R_A \to R_B$，退化到经典自适应协同模型。

### 8.4 区别总结：SDAS vs 旧的 Augmented Adaptive

| 方面 | 旧模型（Augmented Adaptive） | SDAS 模型（当前默认） |
|------|---------------------------|-------------------|
| 传动矩阵 | R（对称平均）+ Rf（冻结模型） | R_A + R_B（非对称 Capstan） |
| 第二协同来源 | Rf（静摩擦阈值冻结） | R_A - R_B（自然非对称性） |
| 双马达公式 | AugmentedAdaptiveSynergyModel | Schur 补公式（交叉耦合） |
| 物理假设 | 存在静摩擦冻结区 | Capstan 方向衰减 + slack 物理 |


---

## 9. 第六层-情况C：两个马达都松弛

```python
# 两者都 ≤ epsilon
q = np.zeros(n_joints)
```

什么都不做，所有关节角度为 0。

---

## 10. 第七层：关节角度限位

无论哪种情况算出的 $q$，都还要做最后一步处理。

### 10.1 折痕的物理限位

| 折痕类型 | 允许范围 | 物理含义 |
|---------|---------|---------|
| 谷折 (VALLEY) | $[0, \pi]$ | 0=展平，$\pi$=合拢 |
| 山折 (MOUNTAIN) | $[-\pi, 0]$ | 0=展平，$-\pi$=合拢 |

### 10.2 执行限位

```python
for urdf_name, syn_idx in urdf_to_syn_map.items():
    raw_angle = q[syn_idx]
    fold_type = syn_idx_to_fold_type.get(syn_idx)  # 查这个关节是谷折还是山折
    if fold_type is not None:
        clamped_angle = clamp_fold_angle(raw_angle, fold_type)
    else:
        clamped_angle = raw_angle
    result[urdf_name] = clamped_angle
```

`clamp_fold_angle` 的实现：

```python
def clamp_fold_angle(angle, fold_type):
    if fold_type == FoldType.VALLEY:
        return np.clip(angle, 0.0, np.pi)        # 谷折：夹到 [0, π]
    elif fold_type == FoldType.MOUNTAIN:
        return np.clip(angle, -np.pi, 0.0)       # 山折：夹到 [-π, 0]
    return angle
```

### 10.3 这意味着什么？

如果协同模型算出一个谷折关节的 $q = 4.0$（超过 $\pi$），会被钳到 $\pi$。所以即使滑块推到很大，关节也不会超物理范围。

---

## 11. 第八层：把结果送给 MuJoCo

### 11.1 URDF→Synergy 索引映射

MuJoCo 认识的是关节名称（`joint_0`, `joint_1`, ...），协同模型认识的是索引（`q[0]`, `q[1]`, ...）。

在启动时通过 `build_urdf_to_synergy_mapping()` 建立了映射：

```python
{
    "joint_0": 0,   # URDF的 joint_0 → q[0]
    "joint_1": 1,   # URDF的 joint_1 → q[1]
    "joint_2": 2,
    "joint_3": 3,
}
```

这个映射怎么建立的？

1. 遍历设计中所有 VALLEY/MOUNTAIN 折痕，计算其中点坐标（2D 折痕的中点）
2. 遍历 URDF 文件中的每个关节，读取其 parent link 和 origin 偏移，计算世界坐标
3. 对每个 URDF 关节的世界坐标，找最近的折痕中点，就是对应关系

### 11.2 回调函数返回

```python
result = {
    "joint_0": 0.350,    # q[0] = 0.350 弧度，钳位后
    "joint_1": 0.319,    # q[1] = 0.319 弧度
    "joint_2": 0.292,
    "joint_3": 0.267,
}
```

MuJoCo 收到这个字典，把 `joint_0` 的角度设为 0.350 rad，`joint_1` 设为 0.319 rad，等等。

---

## 12. 整个流程的金字塔总结

### 构建阶段（启动时执行一次）

```
.ohd 文件（JSON）
    │
    ├── 提取关节列表 (get_joint_list)
    │   └── 建立 fold_line_id → joint_index 映射
    │
    ├── 计算传动矩阵 (compute_R_one_sided)
    │   ├── 遍历每条腱绳的 pully_sequence
    │   │   ├── 元素 k 获取半径 r_k、关联关节 j_idx
    │   │   ├── w_A = exp(-β · k)              ← 经验权重（指数拟合）
    │   │   ├── w_B = exp(-β · (N-1-k))        ← 经验权重（指数拟合）
    │   │   ├── R_A[j_idx] += r_k · w_A
    │   │   └── R_B[j_idx] += r_k · w_B
    │   └── 平均多腱绳 → 最终 R_A, R_B
    │
    ├── 构建 SDAS 模型 (SDASModel)
    │   ├── S_A_paper = E⁻¹R_Aᵀ(R_A E⁻¹R_Aᵀ)⁻¹          ← 仅 Motor A 激活时使用
    │   ├── S_B_paper = E⁻¹R_Bᵀ(R_B E⁻¹R_Bᵀ)⁻¹          ← 仅 Motor B 激活时使用
    │   ├── S_A_schur = E⁻¹(R_Aᵀc - R_Bᵀb)/det          ← 双马达时 Motor A 分量
    │   ├── S_B_schur = E⁻¹(-R_Aᵀb + R_Bᵀa)/det         ← 双马达时 Motor B 分量
    │   ├── S_sigma_schur = S_A_schur + S_B_schur        ← 共模协同方向
    │   ├── S_diff_schur  = S_A_schur - S_B_schur        ← 差模协同方向
    │   ├── a = R_A E⁻¹ R_Aᵀ, b = R_A E⁻¹ R_Bᵀ, c = R_B E⁻¹ R_Bᵀ
    │   └── det = ac - b²
    │
    │   约束策略：
    │     - 单马达 → 用 S_A_paper 或 S_B_paper（单向公式，另一侧松弛约束自动释放）
    │     - 双马达 → 用 S_A_schur / S_B_schur（Schur 补公式，双约束同时精确满足）
    │     - 外力 → 自动匹配对应 C 矩阵

    │
    ├── 构建 URDF→Synergy 映射
    │   └── URDF 关节世界坐标 ↔ 折痕中点坐标最近邻匹配
    │
    └── 提取折痕类型映射
        └── synergy_index → fold_type (valley/mountain)
```

### 运行阶段（每帧/每次滑块变化）

```
用户拖动 Motor A 滑块 → theta_A = 2.5 (rad)
用户拖动 Motor B 滑块 → theta_B = 1.5 (rad)
                  │
                  ▼
synergy_callback(2.5, 1.5, 0.0)
                  │
                  ▼
[Slack 钳位]
    theta_A = max(0, 2.5) = 2.5
    theta_B = max(0, 1.5) = 1.5
                  │
                  ▼
[状态判断]
    theta_A > 1e-10? 是 ✓
    theta_B > 1e-10? 是 ✓
    → both_active
                  │
                  ▼
[计算协同变量]
    σ = (2.5 + 1.5) / 2 = 2.0      ← 同模（整体闭合力）
    σ_f = (2.5 - 1.5) / 2 = 0.5    ← 差模（形状偏移）
                  │
                  ▼
[SDAS 求解（Schur 补公式）]
    double_active → _solve_both_active()
    q = S_A_schur · 2.5 + S_B_schur · 1.5
       = (S_A_schur + S_B_schur)·2.0 + (S_A_schur - S_B_schur)·0.5
    → q = [0.85, 0.72, 0.68, 0.63]  (假设数值)

                  │
                  ▼
[关节角度限位]
    joint_0: q=0.85, fold=valley → clip(0.85, 0, π) = 0.85
    joint_1: q=0.72, fold=valley → clip(0.72, 0, π) = 0.72
    joint_2: q=0.68, fold=valley → clip(0.68, 0, π) = 0.68
    joint_3: q=0.63, fold=valley → clip(0.63, 0, π) = 0.63
                  │
                  ▼
[输出到 MuJoCo]
    {"joint_0": 0.85, "joint_1": 0.72,
     "joint_2": 0.68, "joint_3": 0.63}
```

---

## 13. 动态协同模式有什么不同？

动态模式（`--dynamic`）把上面的**增强协同模型**替换为**动态协同模型**。

### 13.1 物理差异

准静态模型中，阻尼力忽略不计。但实际快速运动时，阻尼器会阻碍关节运动。

系统方程变成：

$$T^\top C T \dot{q} + E q = R^\top u$$

### 13.2 实现方式

不直接求解微分方程，而是把阻尼效应等效为**速度依赖的附加刚度**：

$$E_{\text{eff}}(\alpha) = E + \alpha \cdot \operatorname{diag}(T^\top C T)$$

其中 $\alpha \in [0, 1]$ 由速度滑块控制。

然后用 $E_{\text{eff}}$ 代替 $E$ 计算协同矩阵：

$$S_{\text{eff}}(\alpha) = E_{\text{eff}}(\alpha)^{-1} R^\top (R E_{\text{eff}}(\alpha)^{-1} R^\top)^{-1}$$

### 13.3 运行流程

```
用户拖动 Motor A = 3.0, Motor B = 1.0
用户拖动 Speed 滑块 = 8.0 (rad/s)
                  │
                  ▼
θ_A = max(0, 3.0) = 3.0
θ_B = max(0, 1.0) = 1.0
σ = 2.0, σ_f = 1.0
α = clip(8.0 / 10.0, 0, 1) = 0.8
                  │
                  ▼
E_eff = E + 0.8 · diag(Tᵀ C T)
S_eff = E_eff⁻¹ Rᵀ (R E_eff⁻¹ Rᵀ)⁻¹
q = S_eff @ [σ, σ_f]ᵀ
                  │
                  ▼
[限位 → 输出] （同上）
```

**效果**：速度高时（Speed=10），阻尼关节上的有效刚度增大 → 这些关节的弯曲量相对减小 → 手指姿态偏向"快速握拳"模式。速度低时（Speed=0），阻尼贡献可忽略 → 标准准静态协同。

### 13.4 关于对角化的说明

代码中使用 `diag(T^\top C T)` 而非全矩阵 $T^\top C T$。原因是：

当一个阻尼器连接多个关节时（如阻尼器线同时绕过 3 个关节），全矩阵形式会引入交叉耦合项，导致部分关节在数学解中出现**反向弯曲**（符号翻转）。这在物理上不合理——拉手应该所有关节一起朝一个方向弯曲。

对角化后，每个关节独立感受到阻尼，没有交叉耦合：

$$\Delta E_j = \sum_i C_i \cdot T_{ij}^2$$

这保证了所有关节朝同一方向弯曲。

---

## 14. 附录：关键参数表

| 参数 | 默认值 | 作用位置 | 真实含义 |
|------|--------|---------|---------|
| $\beta$ (DEFAULT_BETA) | 0.09 | `compute_R_capstan`, `compute_R_one_sided` | **拟合参数**，用来调控指数衰减的快慢。不是物理摩擦系数。 |
| 滑轮默认半径 | 3.5 | `_get_element_radius` | 传动比基础参数 |
| 孔默认 `plate_offset` | 3.0 | `_get_element_radius` | 传动比基础参数 |
| 关节默认刚度 | 1.0 | `FoldLine.stiffness` | 抵抗折痕弯曲的弹簧刚度 |
| Slack 阈值 | 1e-10 | `synergy_callback` | 判断马达激活/松弛的边界 |
| 速度因子映射 | speed/10.0 | `synergy_callback`（动态模式） | 速度滑块到 $\alpha$ 的线性映射 |

> **移除的参数**：旧模型的 $\epsilon_{\text{stiction}}$（静摩擦冻结阈值，原用于 `compute_Rf`）已被 SDAS 模型的 R_A/R_B 自然非对称性取代，不再需要。


---

> **文档版本**: v2.2（SDAS 模型完全替代 Augmented Adaptive：§8 更新为 Schur 补公式，§12 更新为 SDAS 构建流程，§14 移除 $\epsilon_{\text{stiction}}$ 参数）
> **最后更新**: 2026-05-27
> **对应代码**: `scripts/run_synergy_from_ohd.py`, `src/models/transmission_builder.py`, `src/synergy/sdas_model.py`, `src/synergy/base_adaptive.py`


