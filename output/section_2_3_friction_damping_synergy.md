## 2.3 含摩擦与阻尼的耗散型协同驱动模型

### 2.3.1 从增强自适应协同到耗散驱动的动机

第 2.2 节介绍的增强自适应协同模型虽然有效刻画了缆绳摩擦引起的传动非对称性，但仍然是在准静态框架下描述系统的驱动行为。该模型隐含了两个关键假设：(1) 系统处于或接近准静态平衡态，忽略速度相关的动力学效应；(2) 摩擦仅以拟静力方式影响张力分布，不引入速度依赖的阻尼特性。

然而，在实际的绳驱折纸灵巧手操作中，这两种假设均面临挑战。首先，手指在进行快速抓取（如动态握拳）与慢速操作（如精密捏取）时，表现出截然不同的运动模式——快速运动时阻尼效应显著，关节响应被抑制，运动模式更加"集中"于少数主导自由度；而慢速运动时关节响应更加充分，可以更均匀地分布到所有自由度。这种速度依赖的行为无法由纯弹性协同模型解释。其次，绳驱系统中的摩擦不仅存在拟静力效应（如 Capstan 张力衰减），还包含速度相关的动摩擦和黏性阻尼成分，这些因素在高速度梯度下不可忽略。

受人体手部运动中肌肉协同的"阻尼调谐"机制启发——人体在快速运动中通过共收缩（co-contraction）和肌肉阻尼调节关节刚度与阻尼——我们提出**耗散型协同驱动模型**（Dissipative Synergy Drive Model），将阻尼效应显式地纳入协同框架，使系统的驱动行为能够根据运动速度动态调整协同方向。该模型不是对增强自适应协同的简单扩展，而是从物理层面重新思考了协同驱动的本质：在弹性协同（存储能量，产生运动）的基础上，引入耗散协同（消耗能量，抑制运动），二者共同决定了系统在不同时间尺度上的运动模式。

### 2.3.2 阻尼器传动与耗散协同矩阵

本模型中，阻尼器（damper）作为核心耗散元件，其物理作用通过传动矩阵 $\boldsymbol{T} \in \mathbb{R}^{n_d \times n}$ 描述：

$$\boldsymbol{x} = \boldsymbol{T} \boldsymbol{q} \tag{2.11}$$

其中 $\boldsymbol{x} \in \mathbb{R}^{n_d}$ 为 $n_d$ 个阻尼器的位移向量，$\boldsymbol{T}$ 的元素 $T_{ij}$ 表示第 $j$ 个关节的单位角度变化引起的第 $i$ 个阻尼器的位移量。阻尼器产生的耗散力为：

$$\boldsymbol{f}_d = -\boldsymbol{C}_d \dot{\boldsymbol{x}} = -\boldsymbol{C}_d \boldsymbol{T} \dot{\boldsymbol{q}} \tag{2.12}$$

其中 $\boldsymbol{C}_d = \text{diag}(c_1, c_2, \ldots, c_{n_d})$ 为阻尼系数矩阵。阻尼力通过传动映射转换为关节空间的广义阻尼力矩：

$$\boldsymbol{\tau}_d = \boldsymbol{T}^T \boldsymbol{f}_d = -\boldsymbol{T}^T \boldsymbol{C}_d \boldsymbol{T} \dot{\boldsymbol{q}} \triangleq -\boldsymbol{D} \dot{\boldsymbol{q}} \tag{2.13}\]

其中 $\boldsymbol{D} = \boldsymbol{T}^T \boldsymbol{C}_d \boldsymbol{T} \in \mathbb{R}^{n \times n}$ 为关节空间的阻尼矩阵。注意到 $\boldsymbol{D}$ 的秩不超过 $n_d$（通常 $n_d \ll n$），因此阻尼效应仅作用于由 $\boldsymbol{T}$ 的行向量张成的子空间。

引入阻尼后的系统运动方程为：

$$\boldsymbol{E} \boldsymbol{q} + \boldsymbol{D} \dot{\boldsymbol{q}} = \boldsymbol{R}_{\text{aug}}^T \boldsymbol{\sigma}_{t,\text{aug}} + \boldsymbol{J}^T \boldsymbol{f}_{\text{ext}} \tag{2.14}$$

式(2.14)体现了"弹性+耗散"的混合驱动特性。在准静态极限（$\dot{\boldsymbol{q}} \approx \boldsymbol{0}$）下，系统退化为2.2节的增强自适应协同模型；而在动态工况下，阻尼项 $\boldsymbol{D} \dot{\boldsymbol{q}}$ 提供了速度依赖的额外约束。

### 2.3.3 基于速度调制的协同切换机制

式(2.14)的完整解涉及一阶微分方程组的求解。为了在保证计算效率的同时捕捉速度依赖的协同切换行为，我们采用模态分解方法，定义两组极限工况下的协同模式：

**慢速协同（Slow Synergy）**——当运动速度趋近于零时，系统处于准静态，阻尼效应可忽略。此极限对应于经典的增强自适应协同：

$$\boldsymbol{S}_s = \boldsymbol{E}^{-1} \boldsymbol{R}_{\text{aug}}^T (\boldsymbol{R}_{\text{aug}} \boldsymbol{E}^{-1} \boldsymbol{R}_{\text{aug}}^T)^{-1} \tag{2.15}$$

**快速协同（Fast Synergy）**——当运动速度足够高时，阻尼效应占主导地位，系统等效于在增广刚度矩阵 $(\boldsymbol{E} + \alpha \boldsymbol{D})$ 下（其中 $\alpha$ 为特征速度因子）求解协同方向：

$$\boldsymbol{S}_f = (\boldsymbol{E} + \alpha \boldsymbol{D})^{-1} \boldsymbol{R}_{\text{aug}}^T (\boldsymbol{R}_{\text{aug}} (\boldsymbol{E} + \alpha \boldsymbol{D})^{-1} \boldsymbol{R}_{\text{aug}}^T)^{-1} \tag{2.16}$$

实际运动过程中，系统在两种极限模式之间通过速度因子平滑切换：

$$\boldsymbol{S}(\dot{\boldsymbol{q}}) = (1 - \eta(\dot{\boldsymbol{q}})) \boldsymbol{S}_s + \eta(\dot{\boldsymbol{q}}) \boldsymbol{S}_f \tag{2.17}$$

其中 $\eta(\dot{\boldsymbol{q}}) \in [0, 1]$ 为速度加权函数，定义为：

$$\eta(\dot{\boldsymbol{q}}) = \frac{\|\dot{\boldsymbol{q}}\|_v}{\|\dot{\boldsymbol{q}}\|_v + v_0} \tag{2.18}$$

$v_0$ 为特征速度（约 1-2 rad/s），$\|\dot{\boldsymbol{q}}\|_v$ 为关节速度的加权范数。

### 2.3.4 摩擦耗散的微观机制与参数辨识

本节模型中的"摩擦"包含两个层次的含义。第一层次是已在2.2节中讨论的**拟静力摩擦效应**——缆绳与滑轮/孔道间的 Capstan 摩擦导致张力沿路径非均匀分布，该效应通过 $\boldsymbol{R}_f$ 矩阵建模并固化在增强传动矩阵 $\boldsymbol{R}_{\text{aug}}$ 中。第二层次是本节关注的**动态摩擦与阻尼效应**——包括滑轮轴承的黏性阻尼、缆绳与孔壁之间的动摩擦、以及折痕材料本身的黏弹性耗散。这些微观耗散机制共同贡献于 $\boldsymbol{D}$ 矩阵。

参数 $\boldsymbol{C}_d$ 和特征速度 $v_0$ 可通过系统辨识实验标定。具体方法为：对灵巧手施加已知幅值与频率的正弦驱动信号，同时记录各关节的角度响应。在低频驱动下（$\omega \ll v_0$），系统行为接近慢速协同模式，响应幅度较大且相位滞后较小；在高频驱动下（$\omega \gg v_0$），系统行为趋向快速协同模式，响应幅度被阻尼抑制且相位滞后增大。通过拟合不同频率下的幅频响应曲线，可辨识出阻尼系数矩阵 $\boldsymbol{C}_d$ 和特征速度 $v_0$。

此外，我们引入电缆松弛检测机制完善模型的物理一致性：当差动输入 $\sigma_f$ 使某一侧驱动器位移为负值时，该侧缆绳进入松弛状态，不产生驱动力。这一松弛钳位（slack clamping）处理确保了模型忠实地反映缆绳单向传力的物理约束。

### 2.3.5 模型特点与讨论

本文提出的耗散型协同驱动模型具有以下特点：

（1）**物理一致性**。模型从弹性-耗散耦合的基本物理原理出发，避免了纯数据驱动的黑箱建模，所有参数均具有明确的物理含义。

（2）**速度自适应**。通过速度调制的协同切换机制，模型可以统一描述从慢速捏取到快速握拳的连续行为谱，无需在不同速度区域切换不同模型。

（3）**与增强自适应协同的兼容性**。在准静态极限下，模型自然退化为2.2节的增强自适应协同模型，保证了理论体系的整体连贯性。

（4）**计算高效性**。模态分解与切换机制避免了在线求解微分代数方程组，仅需预先计算两种极限协同模式并实时插值，计算开销极低，适合部署于实时控制系统。

【图 2.3 约占半页：耗散型协同驱动的物理示意图。上排：阻尼器安装在折纸关节上的布置方式，标注阻尼系数。下排：慢速协同（左）与快速协同（右）模式下同一输入引起的不同关节角分布，展示速度对驱动模式的调制作用。】

【图 2.4 约占半页：速度-响应幅值曲线。展示不同阻尼器配置下，各关节的响应幅值随驱动角频率的变化关系（幅频响应），标注特征速度 $v_0$ 位置。】
