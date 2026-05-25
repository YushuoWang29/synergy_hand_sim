# 参考文献说明（编号 101 起）

以下为本文第三章 3.2 节所引用文献的编号与对应关系。编号从 101 开始，以区别于原文档中已有的文献编号。

## 已存入项目 reference/ 文件夹的文献

| 编号 | 对应文件 | 说明 |
|------|----------|------|
| [101] | `reference/Postural Hand Synergies for Tool Use.pdf` | Santello et al. 关于人手抓取姿态协同模式的开创性研究，为自适应协同模型提供了生物学基础。 |
| [102] | `reference/PisaIIT_SoftHand_0.pdf` | Pisa IIT SoftHand 的原始论文，提出了基于协同驱动的欠驱动软体手设计方法论。 |
| [103] | `reference/Toward_Dexterous_Manipulation_With_Augmented_Adaptive_Synergies_The_Pisa_IIT_SoftHand_2.pdf` | Della Santina et al. 关于增强自适应协同（Augmented Adaptive Synergy）的论文，提出了包含摩擦滑动传动矩阵 $R_f$ 的增强模型。 |

## 引用的公开文献

| 编号 | 完整引用 | 说明 |
|------|----------|------|
| [104] | Piazza, C., et al. "A Centripetal Force Based Synergy For The Pisa/IIT SoftHand." IEEE-RAS International Conference on Humanoid Robots, 2016. | 提出了基于离心力（即阻尼器）的动态协同模型，本文耗散型协同驱动模型的阻尼器传动理论直接受此启发。 |
| [105] | Della Santina, C., et al. "Towards Dexterous Manipulation with Augmented Adaptive Synergies: The Pisa/IIT SoftHand 2." IEEE Transactions on Robotics, 2018. | Pisa/IIT SoftHand 2 论文，详细阐述了传动矩阵构建方法（指代本文第三章3.2.6节中传动矩阵的构建算法的基础文献）。 |
| [106] | Della Santina, C., et al. "Soft Robots that Mimic Human Hand Synergies." IEEE Robotics & Automation Magazine, 2018. | 综述了从基本自适应协同到增强自适应协同再到动态协同的发展脉络。 |
| [107] | Todorov, E., Erez, T., & Tassa, Y. "MuJoCo: A Physics Engine for Model-Based Control." IEEE/RSJ IROS, 2012. | MuJoCo 物理引擎的原始论文，本文仿真平台的交互式仿真模块基于此引擎。 |
| [108] | Featherstone, R. "Rigid Body Dynamics Algorithms." Springer, 2008. | 多体动力学基础参考，本文 URDF 生成和关节坐标变换理论依据。 |
| [109] | Quigley, M., et al. "ROS: an open-source Robot Operating System." ICRA Workshop, 2009. | URDF 格式规范的基础参考文献。 |
| [110] | Carpin, S., et al. "The USARSim project." IEEE Robotics & Automation Magazine, 2007. | 高保真仿真中的环境建模参考。 |

## 在本文中的引用分布

- **2.2节**：引用 [101]（手部协同的生物学基础）、[102]（Pisa/IIT SoftHand 原始协同模型）、[103]（增强自适应协同）、[105]（传动矩阵构建）、[106]（协同模型综述）
- **2.3节**：引用 [104]（阻尼器动态协同理论）、[106]（阻尼器在软体手中的应用）
- **3.2节**：引用 [107]（MuJoCo 物理引擎）、[108]（多体动力学）、[109]（URDF 格式规范）

## 引用原文档已有文献的说明

本文在写作中引用了原文档中已列出的文献[1]-[n]，编号保持不变。
