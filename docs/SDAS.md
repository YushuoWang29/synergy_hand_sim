# 理论建模

## 2.1 灵巧手的姿态协同理论

在关于运动神经控制的研究中，协同（synergy）根据所研究的感觉运动系统的层次和尺度的不同，有多种不同的定义方式，比如运动单元协同、肌肉协同、关节协同等[1,2]。但无论在哪个层次上被定义，协同的主要内涵是：多个自由度可以在一个比可用维度数更低的维度空间（即协同空间）中被控制。

姿势协同（postural synergy）是指手部在进行抓握、使用工具等动作时，中枢神经系统并非独立控制每个关节的角度，而是通过少量固定的协同模式来同时协调多个关节的运动[3]。

人手的运动构型可以由一个包含$n$个关节角度的向量唯一确定，可表示为$q \in \mathbb{R}^{n}$。协同基是关节运动空间中一组特殊的正交基，可表示为$S \in \mathbb{R}^{n \times n}$，$S$被称为协同矩阵，其每一列代表一种协同模式。协同基可以通过对人手的运动构型数据做主成分分析得到。因此，人手的运动构型可以用协同矩阵$S$和协同空间中的向量$\sigma \in \mathbb{R}^{n}$表示为：

$$\begin{array}{r}
q = S\sigma.\ \#(2.1)
\end{array}$$

Santello等人的研究表明，通过主成分分析得到的前几个协同基已足以解释人手运动构型的绝大部分的变异[3]。因此可以通过仅保留$S$的前$k$个列向量来近似表示人手运动构型，即：

$$\begin{array}{r}
q \approx S^{(k)}\sigma^{(k)},\ \#(2.2)
\end{array}$$

其中$S^{(k)} \in \mathbb{R}^{n \times k}$代表$S$的前$k$列组成的矩阵， $\sigma^{(k)}$是一个$k$维向量，每个分量表示前$k$个协同模式在合成最终的手运动构型时的权重。这样就做到了在$k$维空间中控制人手的$n$个自由度，实现降维。

然而，由于在Santello的研究中，手部运动构型数据是在被试者模拟做出抓握日常物体、未与物体发生实际接触的情况下采集的。因此这一协同模型可以很好地描述手在预成形阶段的动作，但在实际抓取中，会遇到手与物体之间只能形成极少数的接触点的问题。

## 2.2 软协同和适应性协同模型

为了解决上述问题，Bicchi等人提出了软协同（soft synergy）模型[4]，其核心思想是引入关节刚度矩阵$K$和接触力$f_{c}$，以及在协同空间中主动驱动的虚拟参考构型$q_{r} \in R^{n}$，表示为：

$$\begin{array}{r}
q_{r} = S^{(k)}\sigma_{r}^{(k)}.\ \#(2.3)
\end{array}$$

当手的实际构型由于受到物体阻挡，而与虚拟参考构型产生偏差时，各个关节处产生弹性恢复力矩：

$$\begin{array}{r}
\tau_{e} = K\left( q_{r} - q \right),\#(2.4)
\end{array}$$

尝试将实际构型拉到参考构型。而静平衡状态下，这一弹性恢复力矩与接触力施加在各个关节的力矩相互平衡，可写出力矩平衡方程：

$$\begin{array}{r}
J^{T}f_{c} = \tau_{e}.\ \#(2.5)
\end{array}$$

其中$J$为手的雅可比矩阵。手的实际构型$q$可通过力矩平衡方程解出：

$$\begin{array}{r}
q = S^{(k)}\sigma_{r}^{(k)} + K^{- 1}J^{T}f_{c}.\#(2.6)
\end{array}$$

对比(2.2)和(2.5)式可知，软协同模型使手在保持低维驱动的同时，保留了手全部的运动自由度，使其能够被动地顺应物体形状，从而在抓握中形成多个接触点。然而，软协同的机械实现通常仍需要全驱动及复杂的阻抗控制，难以直接转化为简洁的机械结构。

为了在机械层面上高效实现软协同的行为，Grioli 等提出了自适应协同（adaptive synergy）模型。自适应协同模型的核心特点是引入了欠驱动模式。每个驱动器通过一组差动机构（如滑轮组或腱绳）来同时驱动多个关节，驱动器的主动运动规定了手的运动构型在协同空间中的各个分量，但运动构型在协同空间的补空间中则是被动顺应的。具体而言，差动机构将驱动器位移传导到各个关节的位移，满足：

$$\begin{array}{r}
rq = s,\#(2.7)
\end{array}$$

其中行向量$r \in \mathbb{R}^{1 \times n}$表示从驱动器到各关节的传动比，$s$代表驱动器的输入位移。

受此启发，自适应协同模型提出，一个具有自适应协同能力的机械手，其期望行为是：沿前$k$个协同方向主动驱动，而在其正交补方向上完全被动。可以描述为：

$$\begin{array}{r}
q = S^{(k)}\sigma^{(k)} + N^{(k)}\lambda,\#(2.8)
\end{array}$$

其中$N^{(k)}$为$S^{(k)}$的互补矩阵，满足$N^{(k)}\mathcal{\in N(}S^{(k)T})$，即$N^{(k)}$的列向量张成$S^{(k)}$的零空间；$\lambda$为描述手在该补空间内运动幅度的向量。

当系统内有多个差动机构时，可以用传动矩阵$R$描述各个驱动器经差动机构到各个关节的传动比，$R = \left\lbrack r_{1}^{T},\cdots,r_{k}^{T} \right\rbrack^{T} \in \mathbb{R}^{k \times n}$，则系统满足：

$$\begin{array}{r}
Rq = x,\#(2.9)
\end{array}$$

其中$x \in \mathbb{R}^{k}$为表示全部$k$个驱动器的运动的列向量。当$R$的各个行向量线性无关时，$x$的$k$个输入彼此独立。可以设计合适的$R$，使得同时满足$RS^{(k)} = I_{k}$和$RN^{(k)} = 0$,代入式(2.8)，即：

$$\begin{array}{r}
Rq = \sigma^{(k)},\#(2.10)
\end{array}$$

因此有：

$$\begin{array}{r}
x = \sigma^{(k)},\#\left. （2.11 \right.）
\end{array}$$

即驱动器的输入向量直接等于协同变量。这意味着每个驱动器独立控制一个协同运动，而手在补空间$N^{(k)}$中的被动自适应运动$\lambda$不会反馈到驱动系统中，从而实现了协同驱动与被动适应性的解耦。

根据力学系统的对偶性，$R$将驱动器施加的力$f \in \mathbb{R}^{k}$映射到各个关节的驱动力矩$\tau \in \mathbb{R}^{n}$：

$$\begin{array}{r}
\tau = R^{T}f.\#(2.12)
\end{array}$$

关节扭矩的平衡需要引入关节的刚度矩阵$E \in \mathbb{R}^{n \times n}$（弹性元件与驱动系统并联），和接触力$f_{c}$引起的反作用力矩$J^{T}f_{c}$，写出静力平衡方程

$$\begin{array}{r}
J^{T}f_{c} = R^{T}f - Eq,\#(2.13)
\end{array}$$

其中$J$为手的雅可比矩阵。将式(2.10)和式(2.13)写成矩阵形式，有：

$$\begin{array}{r}
\begin{bmatrix}
 - E & R^{T} \\
R & 0
\end{bmatrix}\left\lbrack \begin{array}{r}
q \\
f
\end{array} \right\rbrack = \left\lbrack \begin{array}{r}
J^{T}f_{c} \\
\sigma^{(k)}
\end{array} \right\rbrack.\#(2.14)
\end{array}$$

利用Schur补公式求解，可得：

$$\begin{aligned}
q = & \left( - E^{- 1} + E^{- 1}R^{T}\left( RE^{- 1}R^{T} \right)^{- 1}RE^{- 1} \right)J^{T}f_{c}\# \\
 & + E^{- 1}R^{T}\left( RE^{- 1}R^{T} \right)^{- 1}\sigma^{(k)}\ \#(2.15)
\end{aligned}$$

$$\begin{array}{r}
f = \left( RE^{- 1}R^{T} \right)^{- 1}RE^{- 1}J^{T}f_{c} + \left( RE^{- 1}R^{T} \right)^{- 1}\sigma^{(k)}.\#(2.16)
\end{array}$$

将式(2.15)与式(2.6)对照，发现为实现一个期望的协同矩阵$S^{(k)}$，只需选取合适的$R$与$E$使其满足：

$$\begin{array}{r}
S^{(k)} = E^{- 1}R^{T}\left( RE^{- 1}R^{T} \right)^{- 1},\#(2.17)
\end{array}$$

便有：

$$\begin{array}{r}
q = S^{(k)}\sigma_{r}^{(k)} + {(S^{(k)}R - I)E}^{- 1}J^{T}f_{c}.\#(2.18)
\end{array}$$

在$E = \alpha I$（即所有关节的刚度相等）且$RR^{T} = I_{k}$的特殊情况下，式(2.17)可以简化为：

$$\begin{array}{r}
S^{(k)} = R^{T},\#(2.19)
\end{array}$$

此时。协同矩阵直接对应传动矩阵的转置，可见差动机构的引入为协同模型的机械实现提供了简洁易行的方式。

差模共模的引入是为了更好的可解释性。绳长的变与不变。

## 2.3 状态依赖自适应协同模型

经典自适应协同模型默认驱动器与关节之间的传动关系由固定传动矩阵 **R** 描述，即驱动输入与关节运动之间满足线性映射关系。然而，对于真实的绳驱欠驱动系统，这一假设通常仅在理想无摩擦条件下成立。由于腱绳与导向元件（如滑轮、导管、导向孔等）之间普遍存在摩擦，同时腱绳只能传递拉力而无法传递压力，因此系统的有效传动关系往往会随运动状态发生改变。特别地，当腱绳整体滑动方向发生变化时，张力沿路径的传播规律也会随之改变，从而导致系统在不同运动状态下呈现不同的有效协同结构。

特别地，在双端驱动的闭环腱绳系统中，两端驱动器会同时向系统注入张力。由于摩擦导致的张力衰减具有路径相关性，因此来自不同驱动端的张力在系统中的传播分布通常并不相同。为了描述这种由内部传动状态引起的协同行为变化，本文提出**状态依赖自适应协同（state-dependent adaptive synergy, SDAS）模型**。与经典自适应协同模型中固定传动矩阵的假设不同，SDAS模型认为系统的有效传动关系依赖于内部传动状态 **S**，即：

$$
\mathbf{R} = \mathbf{R}(\mathbf{S}). \tag{2.20}
$$

这里的状态变量 **S** 包括腱绳路径、张紧状态以及局部摩擦状态等因素。

考虑一根依次经过 *m* 个导向元件的腱绳，其两端分别连接驱动器 A 和 B，共驱动 *n* 个关节，关节构型向量为 $\mathbf{q} \in \mathbb{R}^n$。定义矩阵 $\bar{\mathbf{R}} \in \mathbb{R}^{m \times n}$，其中元素 $\bar{R}_{j,i}$ 表示第 *i* 个关节运动对第 *j* 个导向元件处腱绳长度变化的贡献。当关节发生微小位移 $\delta \mathbf{q}$ 时，各滑轮对应的腱绳长度变化满足：

$$
\delta \mathbf{l} = \bar{\mathbf{R}} \, \delta \mathbf{q}. \tag{2.21}
$$

设第 *j* 段腱绳张力为 $T_j$，定义系统张力向量 $\mathbf{T} = [T_1, \cdots, T_{m+1}]^\mathsf{T} \in \mathbb{R}^{m+1}$。驱动器 A 与 B 的输入张力分别为 $F_A \ge 0$, $F_B \ge 0$。由于腱绳只能受拉不能受压，因此非负约束天然满足。

我们独立地考虑两个驱动器输入的张力的衰减。采用 Capstan 摩擦模型，对于第 *k* 个导向元件，设其摩擦系数与包角分别为 $\mu_k$ 与 $\theta_k$，定义 $\beta_k = \mu_k \theta_k$。对驱动器 A 输入的张力，传播至第 *j* 段腱绳时，可写为：

$$
T_j^{(A)} = \alpha_j^{(A)} F_A, \tag{2.22}
$$

其中

$$
\alpha_j^{(A)} = \exp\!\left( -\sum_{k \in \mathcal{P}_A(j)} \beta_k \right), \tag{2.23}
$$

$\mathcal{P}_A(j)$ 表示从驱动器 A 到第 *j* 段腱绳之间经过的导向元件集合。类似地，对驱动器 B 输入的张力，传播至第 *j* 段腱绳时，可写为：

$$
T_j^{(B)} = \alpha_j^{(B)} F_B, \tag{2.24}
$$

其中

$$
\alpha_j^{(B)} = \exp\!\left( -\sum_{k \in \mathcal{P}_B(j)} \beta_k \right). \tag{2.25}
$$

准静态平衡下，第 *j* 段腱绳的总张力可表示为两端张力的叠加：

$$
T_j = T_j^{(A)} + T_j^{(B)}. \tag{2.26}
$$

写成向量形式，则有：

$$
\mathbf{T} = \begin{bmatrix} \alpha_1^{(A)} \\ \vdots \\ \alpha_{m+1}^{(A)} \end{bmatrix} F_A + \begin{bmatrix} \alpha_1^{(B)} \\ \vdots \\ \alpha_{m+1}^{(B)} \end{bmatrix} F_B \; := \mathbf{A}_A F_A + \mathbf{A}_B F_B, \tag{2.27}
$$

腱绳张力在关节处产生驱动力矩 $\boldsymbol{\tau} \in \mathbb{R}^n$，根据虚功原理，有：

$$
\boldsymbol{\tau} = \bar{\mathbf{R}}^\mathsf{T} \mathbf{T}. \tag{2.28}
$$

将式 (2.27) 带入式 (2.28)，得到：

$$
\boldsymbol{\tau} = \bar{\mathbf{R}}^\mathsf{T} \mathbf{A}_A F_A + \bar{\mathbf{R}}^\mathsf{T} \mathbf{A}_B F_B, \tag{2.29}
$$

由此定义两个驱动器分别对应的传动向量：

$$
\mathbf{R}_A = \mathbf{A}_A^\mathsf{T} \bar{\mathbf{R}} \in \mathbb{R}^{1 \times n}, \qquad
\mathbf{R}_B = \mathbf{A}_B^\mathsf{T} \bar{\mathbf{R}} \in \mathbb{R}^{1 \times n}. \tag{2.30}
$$

则驱动力矩重新写为：

$$
\boldsymbol{\tau} = \mathbf{R}_A^\mathsf{T} F_A + \mathbf{R}_B^\mathsf{T} F_B. \tag{2.31}
$$

式 (2.31) 表明，双端驱动闭环腱绳系统对应两个不同的张力传动向量。二者通常并不相同，其差异来源于摩擦导致的路径相关的张力衰减。特别地，考虑理想无摩擦条件，则有：

$$
\mathbf{A}_A = \mathbf{A}_B, \quad \mathbf{R}_A = \mathbf{R}_B. \tag{2.32}
$$

此时系统退化为经典自适应协同模型中的固定传动矩阵。因此，经典自适应协同实际上对应于 SDAS 模型在无摩擦条件下的特殊情况。

根据虚功原理，驱动器对系统输入的虚功和关节处驱动力矩的虚功应该满足：

$$
F_A \delta x_A + F_B \delta x_B = \boldsymbol{\tau}^\mathsf{T} \delta \mathbf{q}, \tag{2.33}
$$

代入式 (2.31)，得到：

$$
F_A \delta x_A + F_B \delta x_B = F_A \mathbf{R}_A \delta \mathbf{q} + F_B \mathbf{R}_B \delta \mathbf{q}, \tag{2.34}
$$

因此有：

$$
\delta x_A = \mathbf{R}_A \delta \mathbf{q}, \quad \delta x_B = \mathbf{R}_B \delta \mathbf{q}. \tag{2.35}
$$

积分并写成矩阵形式，有

$$
\begin{bmatrix} x_A \\ x_B \end{bmatrix} = \begin{bmatrix} \mathbf{R}_A \\ \mathbf{R}_B \end{bmatrix} \mathbf{q}. \tag{2.36}
$$

引入关节的刚度矩阵 $\mathbf{E} \in \mathbb{R}^{n \times n}$（弹性元件与驱动系统并联），和接触力 $\mathbf{f}_c$ 引起的反作用力矩 $\mathbf{J}^\mathsf{T} \mathbf{f}_c$，写出静力平衡方程

$$
\mathbf{J}^\mathsf{T} \mathbf{f}_c = \boldsymbol{\tau} - \mathbf{E} \mathbf{q} = \mathbf{R}_A^\mathsf{T} F_A + \mathbf{R}_B^\mathsf{T} F_B - \mathbf{E} \mathbf{q}, \tag{2.37}
$$

其中 $\mathbf{J}$ 为手的雅可比矩阵。将式 (2.36) 和式 (2.37) 写成矩阵形式，有：

$$
\begin{bmatrix}
-\mathbf{E} & \mathbf{R}_A^\mathsf{T} & \mathbf{R}_B^\mathsf{T} \\
\mathbf{R}_A & 0 & 0 \\
\mathbf{R}_B & 0 & 0
\end{bmatrix}
\begin{bmatrix} \mathbf{q} \\ F_A \\ F_B \end{bmatrix}
=
\begin{bmatrix} \mathbf{J}^\mathsf{T} \mathbf{f}_c \\ x_A \\ x_B \end{bmatrix}. \tag{2.38}
$$

利用 Schur 补公式求解，可得：

$$
\mathbf{q} = \mathbf{S}_A x_A + \mathbf{S}_B x_B + \mathbf{C} \mathbf{J}^\mathsf{T} \mathbf{f}_c, \tag{2.39}
$$

其中：

$$
\begin{bmatrix} \mathbf{S}_A & \mathbf{S}_B \end{bmatrix} = \mathbf{E}^{-1} \begin{bmatrix} \mathbf{R}_A^\mathsf{T} & \mathbf{R}_B^\mathsf{T} \end{bmatrix}
\left( \begin{bmatrix}
\mathbf{R}_A \mathbf{E}^{-1} \mathbf{R}_A^\mathsf{T} & \mathbf{R}_A \mathbf{E}^{-1} \mathbf{R}_B^\mathsf{T} \\
\mathbf{R}_B \mathbf{E}^{-1} \mathbf{R}_A^\mathsf{T} & \mathbf{R}_B \mathbf{E}^{-1} \mathbf{R}_B^\mathsf{T}
\end{bmatrix} \right)^{-1}, \tag{2.40}
$$

以及：

$$
\mathbf{C} = \mathbf{E}^{-1} - \mathbf{S}_A \mathbf{R}_A \mathbf{E}^{-1} - \mathbf{S}_B \mathbf{R}_B \mathbf{E}^{-1}. \tag{2.41}
$$

当 $\mathbf{R}_A$ 与 $\mathbf{R}_B$ 正交时，式 (2.40) 可解得：

$$
\mathbf{S}_A = \mathbf{E}^{-1} \mathbf{R}_A^\mathsf{T} (\mathbf{R}_A \mathbf{E}^{-1} \mathbf{R}_A^\mathsf{T})^{-1}, \qquad
\mathbf{S}_B = \mathbf{E}^{-1} \mathbf{R}_B^\mathsf{T} (\mathbf{R}_B \mathbf{E}^{-1} \mathbf{R}_B^\mathsf{T})^{-1}. \tag{2.42}
$$

尽管根据 $\mathbf{R}_A$ 与 $\mathbf{R}_B$ 的定义，现实中二者不可能正交，但可以通过设计合适的腱绳路径和摩擦特性，使其主导控制的关节不重合以达到近似正交的效果，从而能够独立地构造 $\mathbf{S}_A$ 和 $\mathbf{S}_B$ 使其与想要的协同模式对应，此时两个驱动器 A 和 B 分别独立控制一个协同模式。

综上所述，SDAS 模型利用了腱绳路径上的摩擦带来的非对称传动特性，实现了在双端驱动的闭环腱绳系统中，使用两个驱动器独立控制两个协同模式，为二维协同空间控制提供了简单易行的机械实现方式。