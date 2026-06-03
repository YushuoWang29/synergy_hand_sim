# 极化自适应协同（Polarized Adaptive Synergy）

## 1 自适应协同模型回顾

在运动神经控制的研究中，协同（synergy）的核心内涵是：多个自由度可以在一个比可用维度数更低的维度空间（即协同空间）中被控制[1, 2]。姿势协同（postural synergy）是指手部在进行抓握、使用工具等动作时，中枢神经系统并非独立控制每个关节的角度，而是通过少量固定的协同模式来同时协调多个关节的运动[3]。

一个包含 $n$ 个关节角度的灵巧手，其运动构型可由向量 $q \in \mathbb{R}^n$ 唯一确定。协同基是关节运动空间中的一组特殊正交基，记为 $S \in \mathbb{R}^{n \times n}$，其每一列代表一种协同模式。人手的运动构型可以表示为：

$$
q = S\sigma \tag{1}
$$

其中 $\sigma \in \mathbb{R}^n$ 为协同空间中的向量。研究表明，前几个协同基已足以解释人手运动构型绝大部分的变异[3]，因此可通过仅保留 $S$ 的前 $k$ 列来近似：

$$
q \approx S^{(k)} \sigma^{(k)} \tag{2}
$$

这样就实现了在 $k$ 维空间中控制 $n$ 个自由度。

Bicchi 等人提出的软协同（soft synergy）模型[4]引入了关节刚度矩阵 $E$ 和接触力 $f_c$，手的实际构型 $q$ 在弹性恢复力矩与接触力矩之间达到平衡：

$$
J^\top f_c = E(q_r - q) \tag{3}
$$

其中 $q_r = S^{(k)} \sigma_r^{(k)}$ 为协同空间中主动驱动的虚拟参考构型。求解得：

$$
q = S^{(k)} \sigma_r^{(k)} + E^{-1} J^\top f_c \tag{4}
$$

为了在机械层面上高效实现软协同的行为，Grioli 等人提出了自适应协同（adaptive synergy）模型[5]。该模型引入差动机构（如滑轮组或腱绳）来同时驱动多个关节。设系统中有 $k$ 个驱动器，$n$ 个关节。传动矩阵 $R \in \mathbb{R}^{k \times n}$ 将驱动器位移 $x \in \mathbb{R}^k$ 映射到关节位移：

$$
R q = x \tag{5}
$$

根据力学系统的对偶性，$R$ 也将驱动器施加的力 $f \in \mathbb{R}^k$ 映射到各关节的驱动力矩 $\tau_a \in \mathbb{R}^n$：

$$
\tau_a = R^\top f \tag{6}
$$

关节扭矩的平衡引入关节的刚度矩阵 $E = \operatorname{diag}(e_1, e_2, \dots, e_n)$（弹性元件与驱动系统并联），以及接触力 $f_{\text{ext}}$ 引起的反作用力矩 $J^\top f_{\text{ext}}$。系统的静力平衡方程为：

$$
J^\top f_{\text{ext}} = R^\top f - E q \tag{7}
$$

将式 (5) 和式 (7) 写成矩阵形式：

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
J^\top f_{\text{ext}} \\ x
\end{bmatrix} \tag{8}
$$

利用 Schur 补公式求解 $q$，并令 $x = \sigma$（驱动器位移直接等于协同变量），可得：

$$
q = E^{-1} R^\top \left( R E^{-1} R^\top \right)^{-1} \sigma + \left( -E^{-1} + E^{-1} R^\top \left( R E^{-1} R^\top \right)^{-1} R E^{-1} \right) J^\top f_{\text{ext}} \tag{9}
$$

记协同矩阵 $S$ 和柔顺矩阵 $C$ 分别为：

$$
S = E^{-1} R^\top \left( R E^{-1} R^\top \right)^{-1} \tag{10}
$$

$$
C = E^{-1} - S R E^{-1} \tag{11}
$$

则式 (9) 可写为：

$$
q = S \sigma + C J^\top f_{\text{ext}} \tag{12}
$$

至此，自适应协同模型通过 $R$ 和 $E$ 两个可由机械设计自由决定的参数矩阵，实现了软协同的机械嵌入。

---

## 2 自适应协同模型的局限性

自适应协同模型的一个基本假设是：传动矩阵 $R$ 在所有驱动方向上具有相同的传递特性。这一假设建立在腱绳驱动的理想化模型之上——不考虑摩擦导致的张力分布变化，也不考虑腱绳单向传力特性。

然而，在真实的绳驱系统中，以下两个物理效应使得这一假设不再成立。

**效应一：Capstan 摩擦导致的张力非对称衰减**

在绳驱系统中，腱绳绕过滑轮时，由于 Capstan 效应，张力沿路径指数衰减：

$$
T_{\text{out}} = T_{\text{in}} \, e^{-\mu\theta} \tag{13}
$$

其中 $\mu$ 为摩擦系数，$\theta$ 为腱绳在滑轮上的包角。考虑一根腱绳由两个同轴电机（Motor A 和 Motor B）从两端驱动。腱绳路径包含 $N$ 个传动元素（滑轮），各元素沿腱绳路径从 Motor A 端到 Motor B 端依次编号为 $1, 2, \dots, N$。第 $k$ 个元素到 Motor A 和 Motor B 的路径步数分别为 $d_k^{(A)} = k$ 和 $d_k^{(B)} = N - k + 1$。

当两个电机以相同方向转动时，两端同时施加张力，第 $k$ 个元素上的有效张力为：

$$
T_k^{(\sigma)} = T_0 \left( e^{-\beta d_k^{(A)}} + e^{-\beta d_k^{(B)}} \right) \tag{14}
$$

其中 $\beta = \mu\theta_{\text{eff}}$ 为有效摩擦系数。此时两端张力在中间区域叠加，形成中间小、两端大的 U 形分布。

而当两个电机以相反方向转动时，一端拉紧、另一端放松，第 $k$ 个元素上的张力仅来自拉紧端：

$$
T_k^{(\sigma_f)} = T_0 \, e^{-\beta d_k^{(A)}} \quad (\text{仅 Motor A 拉紧}) \tag{15}
$$

两式的张力分布模式完全不同。这意味着，从 Motor A 端看去的"等效传动比"和从 Motor B 端看去的"等效传动比"是两个不同的向量，无法用一个单一的 $R$ 矩阵同时描述。

**效应二：腱绳的单向传力与松弛约束**

腱绳在物理本质上是一种单向传力元件——它可以承受拉伸力，但无法承受压缩力（即不能"推"）。当一端电机向放松方向转动时，该段腱绳完全丧失传力能力。

设 $\theta_A, \theta_B \in \mathbb{R}$ 为两个电机的角位移（正方向定义为拉动腱绳）。物理上有效的电机位移应满足：

$$
\tilde{\theta}_A = \max(0, \theta_A), \quad \tilde{\theta}_B = \max(0, \theta_B) \tag{16}
$$

任何负位移意味着腱绳松弛，不应在关节空间产生驱动力矩。

在经典自适应协同模型中，协同输入通常定义为 $\sigma = (\theta_A + \theta_B)/2$，这一表达式隐含地允许 $\theta_A$ 或 $\theta_B$ 为负值。当 $\theta_B < 0$ 时，经典模型将其视为对关节产生"反向驱动力"，但真实物理中 $\theta_B < 0$ 仅对应腱绳进一步松弛，并不产生任何驱动力。这一松弛约束是绳驱系统区别于刚性连杆系统的核心非线性来源之一。

---

## 3 摩擦模型的数学化描述

为将上述物理效应纳入统一的数学框架，首先需要建立包含摩擦的腱绳传动模型。考虑由 $m$ 个滑轮分隔出的 $m+1$ 段腱绳段，设第 $j$ 段的张力为 $T_j$，速度为 $v_j$。

在滑轮 $j$ 处的速度平衡给出：

$$
v_j = v_{j-1} + \sum_{i=1}^n r_{j,i} \dot{q}_i \tag{17}
$$

其中 $r_{j,i}$ 为第 $j$ 个滑轮在第 $i$ 个关节上的半径（若不在该关节上则为 0）。将式 (17) 及边界条件写为矩阵形式：

$$
M v - \bar{R} \dot{q} = -2 e \dot{s} \tag{18}
$$

其中 $M \in \mathbb{R}^{(m+1)\times(m+1)}$ 为速度耦合矩阵，$\bar{R} \in \mathbb{R}^{n \times (m+1)}$ 为元素级的传动矩阵，$e = [0,0,\dots,0,1]^\top \in \mathbb{R}^{m+1}$，$\dot{s}$ 为肌腱的整体滑动速度（额外的自由度，独立于关节运动）。

滑轮 $j$ 处的张力平衡给出：

$$
T_j = T_{j-1} - V_j(v_j) \tag{19}
$$

其中 $V_j(v_j)$ 为第 $j$ 个滑轮上的摩擦力。将式 (19) 及 $\tau_M = T_0 + T_m$（电机总拉力）写为矩阵形式：

$$
M T + V(v) + e \tau_M = 0 \tag{20}
$$

关节上的驱动力矩为：

$$
\tau_i = -\sum_{j=1}^{m+1} r_{j,i} T_j \quad \Rightarrow \quad \tau = -\bar{R}^\top T \tag{21}
$$

联立式 (18)、(20)、(21)，可以解得速度分布和张力分布：

$$
v = M^{-1} \bar{R} \dot{q} - 2 M^{-1} e \dot{s} \tag{22}
$$

$$
T = -M^{-1} V(v) - M^{-1} e \tau_M \tag{23}
$$

将式 (23) 代入式 (21)，得驱动力矩的完整表达式：

$$
\tau = \bar{R}^\top M^{-1} V(v) + \bar{R}^\top M^{-1} e \tau_M \tag{24}
$$

注意到 $e_v = -M^{-1} e = [1,1,\dots,1]^\top \in \mathbb{R}^{m+1}$，因此：

$$
\tau = R^\top \tau_M + D(\dot{q}, \dot{s}) \tag{25}
$$

其中

$$
R^\top = -\bar{R}^\top e_v, \qquad D(\dot{q}, \dot{s}) = -\bar{R}^\top \left( -M^{-1} V(M^{-1} \bar{R} \dot{q} - 2 e_v \dot{s}) \right) \tag{26}
$$

式 (25) 清晰地展示了摩擦引入的第二个驱动通道：肌腱滑动速度 $\dot{s}$ 通过非线性摩擦场 $D(\dot{q}, \dot{s})$ 产生额外的驱动力矩，在传统驱动器 $\tau_M$ 之外增加了第二个控制自由度。

---

## 4 Capstan 摩擦模型下的传动极化

为得到可解析计算的形式，采用 Capstan 摩擦模型。设每个滑轮上的摩擦力与张力成正比：

$$
V_j(v_j) = V_j^{\max} \operatorname{sgn}(v_j) \tag{27}
$$

其中 $V_j^{\max}$ 为第 $j$ 个滑轮的最大静摩擦力。在准静态条件下（$\dot{q} \equiv 0$），张力从一段到另一端的衰减由 Capstan 公式描述。

对于由两个电机（Motor A、Motor B）从两端驱动的单根腱绳，设路径顺序从 Motor A 端到 Motor B 端共有 $N$ 个传动元素，依次编号为 $1, 2, \dots, N$。定义：

- $d_k^{(A)} = k$：元素 $k$ 到 Motor A 的路径步数
- $d_k^{(B)} = N - k + 1$：元素 $k$ 到 Motor B 的路径步数
- $r_k$：第 $k$ 个元素的等效半径
- $\beta$：有效 Capstan 摩擦系数

定义以 Motor A 为参考的**前向传动向量** $R_A \in \mathbb{R}^n$ 和以 Motor B 为参考的**后向传动向量** $R_B \in \mathbb{R}^n$：

$$
R_A^{[j]} = \sum_{k: \text{元素 }k\text{ 在关节 }j\text{ 上}} r_k \, e^{-\beta d_k^{(A)}}, \qquad
R_B^{[j]} = \sum_{k: \text{元素 }k\text{ 在关节 }j\text{ 上}} r_k \, e^{-\beta d_k^{(B)}} \tag{28}
$$

这两个向量分别刻画了当仅 Motor A 或仅 Motor B 拉动时，各关节接收到的有效传动比。由于 $e^{-\beta d_k^{(A)}} \neq e^{-\beta d_k^{(B)}}$（当 $\beta > 0$ 且 $k \neq (N+1)/2$），$R_A$ 和 $R_B$ 具有不同的分布模式：$R_A$ 从靠近 Motor A 的元素到靠近 Motor B 的元素单调递减，$R_B$ 则单调递增。

当两个电机同向驱动时（$\sigma$ 模式），张力从两端同时衰减，等效传动向量为两者的平均：

$$
R_{\text{avg}} = \frac{R_A + R_B}{2} \tag{29}
$$

$R_{\text{avg}}$ 关于路径中点对称，且中间关节的传动比最小，形成 U 形分布。

当两个电机反向驱动时（$\sigma_f$ 模式），一端拉紧、另一端放松，张力差沿路径分布。考虑 Motor A 拉紧、Motor B 放松的情况，由 Capstan 衰减可得元素 $k$ 上的有效张力为 $e^{-\beta d_k^{(A)}}$（来自 A 端），而 B 端由于松弛不产生张力。净张力为 A 端贡献减去 B 端贡献。此外，当张力衰减至低于静摩擦阈值 $\epsilon_{\text{stiction}}$ 时，该段被"冻结"（不滑动），不再参与传动。

由此定义摩擦滑动传动向量 $R_f \in \mathbb{R}^n$：

$$
R_f^{[j]} = \sum_{k: \text{元素 }k\text{ 在关节 }j\text{ 上}} r_k \cdot \max\left(0,\ e^{-\beta d_k^{(A)}} - e^{-\beta d_k^{(B)}}\right) \cdot \mathbf{1}\!\left(e^{-\beta d_k^{(A)}} > \epsilon_{\text{stiction}}\right) \tag{30}
$$

其中 $\mathbf{1}(\cdot)$ 为指示函数。$R_f$ 捕获了非对称张力梯度产生的净驱动效应：靠近拉紧端的关节获得较大的正驱动，靠近放松端的关节被静摩擦"冻结"。

至此，我们得到了两个独立的传动方向：$R_{\text{avg}}$（描述双电机同向驱动）和 $R_f$（描述单侧驱动或反向驱动时的摩擦滑动效应）。将二者堆叠为扩展传动矩阵：

$$
R_{\text{aug}} = 
\begin{bmatrix}
R_{\text{avg}} \\
R_f
\end{bmatrix}
\in \mathbb{R}^{2 \times n} \tag{31}
$$

---

## 5 松弛约束与方向解耦

两个电机在实际物理中的行为受到腱绳单向传力特性的约束。设 $\theta_A, \theta_B \in \mathbb{R}$ 为两个电机的实际角位移，物理有效位移为：

$$
\tilde{\theta}_A = \max(0, \theta_A), \quad \tilde{\theta}_B = \max(0, \theta_B) \tag{32}
$$

系统存在三种基本驱动状态：

**状态 I（双电机同向驱动）：** $\tilde{\theta}_A > 0,\ \tilde{\theta}_B > 0$

此时两电机均张紧，传动矩阵取 $R_{\text{avg}}$，协同输入为：

$$
\sigma = \frac{\tilde{\theta}_A + \tilde{\theta}_B}{2} \tag{33}
$$

**状态 II（单侧驱动）：** $\tilde{\theta}_A > 0,\ \tilde{\theta}_B = 0$ 或 $\tilde{\theta}_A = 0,\ \tilde{\theta}_B > 0$

此时仅一个电机产生驱动力。当 Motor A 单独驱动时，传动矩阵应使用 $R_A$（而非 $R_{\text{avg}}$），因为张力仅从 A 端向远端指数衰减，不存在来自 B 端的对称分量。协同输入为：

$$
\sigma_A = \tilde{\theta}_A \quad \text{或} \quad \sigma_B = \tilde{\theta}_B \tag{34}
$$

**状态 III（松弛）：** $\tilde{\theta}_A = \tilde{\theta}_B = 0$

两电机均松弛，$q = 0$。

---

## 6 极化自适应协同的完整模型

将上述分析纳入统一的求解框架。协同模型的输入为两个电机的物理有效位移 $\tilde{\theta}_A, \tilde{\theta}_B$，系统参数包括极化传动矩阵 $R_A, R_B$、摩擦滑动矩阵 $R_f$、关节刚度矩阵 $E$，以及（可选）阻尼参数 $T, C$。

**第一步：状态判定与传动矩阵选择**

$$
(R_{\text{active}}, \sigma_{\text{active}}) =
\begin{cases}
(R_{\text{avg}},\ (\tilde{\theta}_A + \tilde{\theta}_B)/2), & \tilde{\theta}_A > 0 \land \tilde{\theta}_B > 0 \\
(R_A,\ \tilde{\theta}_A), & \tilde{\theta}_A > 0 \land \tilde{\theta}_B = 0 \\
(R_B,\ \tilde{\theta}_B), & \tilde{\theta}_A = 0 \land \tilde{\theta}_B > 0 \\
(\mathbf{0},\ 0), & \tilde{\theta}_A = 0 \land \tilde{\theta}_B = 0
\end{cases} \tag{35}
$$

**第二步：阻尼-刚度耦合调制（可选）**

当系统中含有阻尼器时，设阻尼器传动矩阵 $T \in \mathbb{R}^{n_d \times n}$ 将关节速度映射为阻尼器位移速度，阻尼系数对角阵 $C = \operatorname{diag}(c_1, \dots, c_{n_d})$。取速度因子 $\alpha \in [0, 1]$，有效刚度为：

$$
E_{\text{eff}}(\alpha) = E + \alpha \cdot \operatorname{diag}\left(T^\top C T\right) \tag{36}
$$

$\alpha = 0$ 对应准静态（慢速），$\alpha = 1$ 对应阻尼主导（快速）。

**第三步：协同求解**

$$
q = E_{\text{eff}}(\alpha)^{-1} R_{\text{active}}^\top \left(R_{\text{active}} E_{\text{eff}}(\alpha)^{-1} R_{\text{active}}^\top\right)^{-1} \sigma_{\text{active}} \tag{37}
$$

**第四步：外载荷补偿（可选）**

当存在外载荷 $f_{\text{ext}}$ 及抓取雅可比 $J$ 时：

$$
q \leftarrow q + C J^\top f_{\text{ext}}, \qquad C = E^{-1} - S_{\text{avg}} R_{\text{avg}} E^{-1} \tag{38}
$$

其中 $S_{\text{avg}} = E^{-1}R_{\text{avg}}^\top(R_{\text{avg}} E^{-1}R_{\text{avg}}^\top)^{-1}$。

**第五步：关节角度限位**

$$
q_j = \operatorname{clamp}(q_j,\ q_{j,\min},\ q_{j,\max}) \tag{39}
$$

谷折痕（VALLEY）限位 $[0, \pi]$，山折痕（MOUNTAIN）限位 $[-\pi, 0]$。

---

回到式 (12) 的形式，极化自适应协同模型对应的完整协同矩阵可以写为：

当使用扩展传动矩阵 $R_{\text{aug}} = [R_{\text{avg}}^\top, R_f^\top]^\top$ 时（适用于双电机同向+摩擦滑动并存的情形）：

$$
S_{\text{aug}} = E^{-1} R_{\text{aug}}^\top \left(R_{\text{aug}} E^{-1} R_{\text{aug}}^\top\right)^{-1} \tag{40}
$$

其对应的协同空间由两个方向张成：

$$
\operatorname{Span}\{S_{\text{aug}}\} = \operatorname{Span}\{S_{\text{avg}}, S_f\} \tag{41}
$$

其中 $S_{\text{avg}} = E^{-1}R_{\text{avg}}^\top(R_{\text{avg}} E^{-1}R_{\text{avg}}^\top)^{-1}$ 为双电机同向驱动的协同方向，$S_f$ 为摩擦滑动产生的第二个协同方向。当采用单侧驱动模式时，协同方向退化为 $S_A$ 或 $S_B$，分别对应从单一电机出发的极化传动方向。

---

## 7 物理意义与对比分析

极化自适应协同模型相对于经典自适应协同模型，在物理建模层面的提升可以归纳为以下几点。

**关于传动方向极化：**

经典模型用单一 $R$ 矩阵描述所有驱动方向。极化模型将传动矩阵分解为 $R_A$ 和 $R_B$ 两个方向分量，物理上对应于 Capstan 摩擦导致的张力非对称衰减。$R_A$ 从路径起点到终点单调递减，$R_B$ 单调递增，$R_{\text{avg}} = (R_A + R_B)/2$ 为二者的对称平均，用于双电机同向驱动模式。

以五关节系统为例，取 $\beta = 0.09$，$r_k = 1$：

$$
\begin{aligned}
R_A &= [0.913, 0.835, 0.764, 0.699, 0.639] \\
R_B &= [0.639, 0.699, 0.764, 0.835, 0.913] \\
R_{\text{avg}} &= [0.776, 0.767, 0.764, 0.767, 0.776]
\end{aligned}
$$

$R_{\text{avg}}$ 中间关节传动比最小，形成 U 形分布。经典模型的几何求和 $R_{\text{geo}} = [1,1,1,1,1]$ 完全无法捕捉这一物理分布模式。

**关于摩擦滑动模式：**

摩擦滑动传动向量 $R_f$ 由式 (30) 定义，其物理来源是：当两个电机反向驱动时，张力在拉紧端和放松端之间的梯度产生净驱动力矩，而张力衰减至低于静摩擦阈值的区段被"冻结"。$R_f$ 为系统提供了第二个独立的传动方向，且其数值可由 $\beta$（摩擦系数）和 $\epsilon_{\text{stiction}}$（静摩擦阈值）独立于 $R_{\text{avg}}$ 而设计。

**关于松弛约束：**

松弛约束式 (32) 确保模型在物理上不产生负张力驱动。当 $\theta_B < 0$ 时，经典模型将其错误地视为"反向驱动"，而极化模型检测到 $\tilde{\theta}_B = 0$，自动切换为单侧驱动模式并使用 $R_A$ 求解。这消除了经典模型中物理不合理的负张力假设。

**关于阻尼耦合的速度调制：**

阻尼器传动矩阵 $T$ 通过式 (36) 对关节有效刚度进行速度依赖的调制，使协同行为从准静态（$\alpha=0$）连续过渡到阻尼主导（$\alpha=1$）状态。这一效应在经典模型中未被涉及。

---

## 参考资料

[1] Bizzi, E., et al. "Modular organization of motor behavior." *Nature Reviews Neuroscience*, 2000.

[2] d'Avella, A., et al. "Combinations of muscle synergies in the construction of a natural motor behavior." *Nature Neuroscience*, 2003.

[3] Santello, M., et al. "Postural Hand Synergies for Tool Use." *Journal of Neuroscience*, 1998.

[4] Bicchi, A., et al. "Soft synergies: A new variable stiffness paradigm for robotic hands." *IEEE ICRA*, 2011.

[5] Grioli, G., et al. "Adaptive synergies for a humanoid robot hand." *IEEE-RAS International Conference on Humanoid Robots*, 2012.

[6] Della Santina, C., et al. "Toward dexterous manipulation with augmented adaptive synergies: The Pisa/IIT SoftHand 2." *IEEE Transactions on Robotics*, 34(5), 2018.
