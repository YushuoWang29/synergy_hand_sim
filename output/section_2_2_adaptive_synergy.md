## 2.2 自适应协同与增强自适应协同模型

### 2.2.1 自适应协同的理论框架

绳驱折纸灵巧手作为一种高度欠驱动系统，其核心挑战在于如何用有限的驱动器实现对多自由度关节的有效控制。人体手部运动学研究表明，尽管人手拥有超过20个自由度，但大多数日常抓取动作可以由少数几个"协同模式"（synergy patterns）有效表征[101-102]。基于这一生物学原理，自适应协同（Adaptive Synergy）模型将高维关节空间投影到低维驱动空间，通过弹性元件的被动顺应性实现欠驱动下的自适应抓取。

设系统具有 $n$ 个关节，$k$ 个驱动器（电机），则自适应协同模型的基本方程为：

$$\boldsymbol{q} = \boldsymbol{S} \boldsymbol{\sigma} + \boldsymbol{C} \boldsymbol{J}^T \boldsymbol{f}_{\text{ext}} \tag{2.1}$$

其中 $\boldsymbol{q} \in \mathbb{R}^n$ 为关节角度向量，$\boldsymbol{\sigma} \in \mathbb{R}^k$ 为协同输入向量（由驱动器位移经传动映射得到），$\boldsymbol{S} \in \mathbb{R}^{n \times k}$ 为协同矩阵（synergy matrix），$\boldsymbol{C} \in \mathbb{R}^{n \times n}$ 为关节柔度矩阵，$\boldsymbol{J} \in \mathbb{R}^{6 \times n}$ 为抓取雅可比矩阵，$\boldsymbol{f}_{\text{ext}} \in \mathbb{R}^6$ 为外力旋量。

协同矩阵 $\boldsymbol{S}$ 的构造基于系统的传动结构。设传动矩阵 $\boldsymbol{R} \in \mathbb{R}^{k \times n}$ 描述了驱动器位移与关节角度之间的运动学映射关系，即 $\boldsymbol{\sigma} = \boldsymbol{R} \boldsymbol{q}$。将该关系代入静力平衡方程，可得：

$$\boldsymbol{\tau} = \boldsymbol{R}^T \boldsymbol{\sigma}_t \tag{2.2}$$

其中 $\boldsymbol{\tau} \in \mathbb{R}^n$ 为关节驱动力矩，$\boldsymbol{\sigma}_t \in \mathbb{R}^k$ 为驱动器张力。当系统包含弹性元件时，关节力矩与关节角度通过刚度矩阵 $\boldsymbol{E} = \text{diag}(e_1, e_2, \ldots, e_n)$ 关联：

$$\boldsymbol{\tau} = \boldsymbol{E} \boldsymbol{q} \tag{2.3}$$

综合以上关系可得：

$$\boldsymbol{E} \boldsymbol{q} = \boldsymbol{R}^T \boldsymbol{\sigma}_t \tag{2.4}$$

由此求解协同矩阵：

$$\boldsymbol{S} = \boldsymbol{E}^{-1} \boldsymbol{R}^T (\boldsymbol{R} \boldsymbol{E}^{-1} \boldsymbol{R}^T)^{-1} \tag{2.5}$$

该式本质上是将驱动空间到关节空间的映射通过弹性柔度加权伪逆实现。在无外载荷的情况下，$\boldsymbol{q} = \boldsymbol{S} \boldsymbol{\sigma}$，即关节运动完全由协同方向 $\boldsymbol{S}$ 的列向量张成的低维子空间决定；当存在外载荷时，被动柔度项 $\boldsymbol{C} \boldsymbol{J}^T \boldsymbol{f}_{\text{ext}}$ 提供了子空间之外的顺应性调节，这是自适应协同实现"形状自适应"抓取的关键机制。

### 2.2.2 增强自适应协同模型

在实际绳驱系统中，简单的线性传动模型（式 2.2）无法完整描述缆绳传动的物理行为。缆绳与滑轮/孔道之间的摩擦会产生显著的力矩传递非对称性：当一侧驱动器拉紧而另一侧放松时，摩擦效应使得张力沿路径呈指数衰减（Capstan效应），且缆绳单向传力特性（只能受拉不能受压）导致放松侧自然产生松弛（slack），进一步加剧了这种非对称性。为了捕捉这一物理现象，我们在传动模型中引入摩擦滑动传动矩阵 $\boldsymbol{R}_f \in \mathbb{R}^{m \times n}$，从而得到增强自适应协同（Augmented Adaptive Synergy）模型[103]。

增强模型将基本协同与摩擦诱导的差动协同相结合。设 $\boldsymbol{\sigma} \in \mathbb{R}^k$ 为同向协同输入（两电机同向转动，对应 $\sigma = (\theta_A + \theta_B)/2$），$\boldsymbol{\sigma}_f \in \mathbb{R}^m$ 为差动协同输入（两电机反向转动，对应 $\sigma_f = (\theta_A - \theta_B)/2$），则传动矩阵扩展为增强形式：

$$\boldsymbol{R}_{\text{aug}} = \begin{bmatrix} \boldsymbol{R} \\ \boldsymbol{R}_f \end{bmatrix} \in \mathbb{R}^{(k+m) \times n} \tag{2.6}$$

对应的协同输入向量为 $\boldsymbol{\sigma}_{\text{aug}} = [\boldsymbol{\sigma}^T, \boldsymbol{\sigma}_f^T]^T$。增强协同矩阵由下式给出：

$$\boldsymbol{S}_{\text{aug}} = \boldsymbol{E}^{-1} \boldsymbol{R}_{\text{aug}}^T (\boldsymbol{R}_{\text{aug}} \boldsymbol{E}^{-1} \boldsymbol{R}_{\text{aug}}^T)^{-1} \tag{2.7}$$

值得指出的是，$\boldsymbol{R}_f$ 矩阵的物理含义与 $\boldsymbol{R}$ 有本质区别。$\boldsymbol{R}$ 描述了缆绳在张紧状态下对所有关节的均匀驱动能力，而 $\boldsymbol{R}_f$ 则刻画了因摩擦引发的差动驱动特性——当 $\sigma_f > 0$（即 Motor A 拉紧、Motor B 放松）时，靠近 Motor A 的关节获得优先驱动，远离 Motor A 的关节由于张力衰减和松弛效应而驱动不足或不被驱动。这种非对称性在物理上表现为：同一根腱绳路径上的不同关节，在差动输入下呈现出不同的有效传动比，从而在单一的缆绳回路内实现了类似"差动机构"的运动分配功能。

### 2.2.3 传动矩阵的物理建模

**基本传动矩阵 $\boldsymbol{R}$。** 对于绳驱折纸结构，每个关节的驱动力矩由跨越该关节折痕的缆绳张力产生。$\boldsymbol{R}$ 矩阵的每一个元素 $R_{ij}$ 表示第 $i$ 根腱绳对第 $j$ 个关节的传动比。在均匀张紧的理想情况下，$R_{ij}$ 等于路径中与关节 $j$ 相关联的所有滑轮/孔的半径之和。然而，物理实验中观察到一种重要的"U形"现象：在对称驱动的缆绳回路中，位于路径中间的关节（对应中指）的驱动力矩显著小于两端关节（对应拇指和小指）。这一现象源于 Capstan 摩擦效应：张力从两端驱动器向中间呈指数衰减，导致中间关节获得的有效驱动力最小。我们采用指数衰减权重模型描述这一效应：

$$R_{ij} = \sum_{k \in \mathcal{P}_i \cap \mathcal{J}_j} r_k \cdot \frac{\exp(-\beta d_{A,k}) + \exp(-\beta d_{B,k})}{2} \tag{2.8}$$

其中 $r_k$ 为第 $k$ 个传动元素的半径，$\beta$ 为有效 Capstan 摩擦系数，$d_{A,k}$ 和 $d_{B,k}$ 分别为元素 $k$ 到驱动器 A 和驱动器 B 的路径距离（以元素个数计）。

**摩擦滑动传动矩阵 $\boldsymbol{R}_f$。** 该矩阵描述了差动输入下的非对称驱动特性。其建模基础是 Capstan 指数衰减模型与松弛钳制效应的结合。当 $\sigma_f > 0$ 时，Motor A 端施加张力 $T_0$，而 Motor B 端因松弛张力接近零。张力沿路径的分布为：

$$T_k = T_0 \cdot \exp(-\beta \cdot k) \tag{2.9}$$

式中 $k$ 为从 Motor A 出发沿路径经过的元素个数。当 $T_k$ 衰减至低于静摩擦阈值 $\gamma T_0$ 时，该段缆绳被摩擦力"冻结"（stiction），不再发生相对滑动。因此，只有从 Motor A 到"冻结点"之间的关节参与差动运动。$\boldsymbol{R}_f$ 矩阵的元素可表示为：

$$R_{f,ij} = \sum_{k \in \mathcal{P}_i \cap \mathcal{J}_j} r_k \cdot \text{sgn}(k) \cdot \max\left(0, \exp(-\beta d_{A,k}) - \exp(-\beta d_{B,k})\right) \tag{2.10}$$

其中符号函数 $\text{sgn}(k)$ 根据元素 $k$ 相对于路径中点的位置确定驱动方向。这一模型的关键特征在于：对称路径中部的关节 $\boldsymbol{R}_f$ 贡献趋近于零（即"冻结区"），而靠近两端关节贡献较大且符号相反，从而在单一缆绳回路中实现了非对称的差动驱动。

【图 2.1 约占半页：增强自适应协同模型的传动矩阵示意图。左图为基本传动矩阵 $\boldsymbol{R}$ 的指数衰减权重示意图，展示从两端驱动器往中间衰减的张力分布。右图为摩擦滑动传动矩阵 $\boldsymbol{R}_f$ 的净张力分布，展示非对称差动驱动的空间分区（主动区-冻结区-被动区）。】

【图 2.2 约占半页：三种协同模型对比。从上到下依次为：基本自适应协同（仅有 $\boldsymbol{R}$ 和 $\boldsymbol{\sigma}$）、含 $\boldsymbol{R}_f$ 的增强自适应协同、以及后续2.3节将介绍的含摩擦与阻尼的耗散型协同驱动模型。以单手指（ohd_1）为例展示各模型预测的关节角分布。】
