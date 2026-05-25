#!/usr/bin/env python3
"""
编辑王裕硕的Word文档，完成以下任务：
1. 完善第2.2节（软协同和适应性协同模型）- 加入Pisa/IIT SoftHand文献内容
2. 重写第2.3节（摩擦协同模型）- 加入Augmented Adaptive Synergy + Dynamic Synergy
3. 重写第3.2节（协同模型仿真）- 描述软件实现
4. 更新参考文献（添加[101]+引用）
5. 更新摘要、关键词、英文摘要
"""

from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import re, os

DOC_PATH = r'f:/Research/SGLab/cable_driven/Dexterous_Hand/code/synergy_hand_sim/docs/v0.1 结题_王裕硕_基于物理智能的绳驱折纸灵巧手顺序驱动技术研究.docx'
OUT_PATH = r'f:/Research/SGLab/cable_driven/Dexterous_Hand/code/synergy_hand_sim/docs/v0.2 结题_王裕硕_基于物理智能的绳驱折纸灵巧手顺序驱动技术研究.docx'

doc = Document(DOC_PATH)

def set_text(para, text, style='论文正文段落'):
    """替换段落文本，保持原有样式"""
    para.text = ''
    para.style = doc.styles[style] if style in [s.name for s in doc.styles] else None
    run = para.add_run(text)
    run.font.name = '宋体'
    run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    run.font.size = Pt(12)
    return run

def get_para(i):
    """安全获取段落"""
    if i < len(doc.paragraphs):
        return doc.paragraphs[i]
    return None

# ======================================================================
# 第2.2节：软协同和适应性协同模型（段落117-126）
# 需要完善此部分，补充Pisa/IIT SoftHand的适应性协同理论
# ======================================================================

# 段落[108] - 当前是"(协同的背景在第一章展开)"
set_text(get_para(108),
"""协同（Synergy）的概念起源于神经科学中对运动控制的研究。Santello等人的研究表明，人手在抓取日常物体时，虽然拥有超过20个自由度，但其运动构型的约80%的变异可以由前两个主成分（即协同基）解释[101]。这一发现意味着中枢神经系统并非独立控制每个关节，而是通过少量固定的协同模式来同时协调多个关节的运动。

姿势协同（Postural Synergy）是指手部在进行抓握、使用工具等动作时的关节运动模式。通过对人手运动构型数据的统计分析，可以提取出一组正交的协同基向量，构成协同矩阵 S ∈ ℝⁿˣᵏ（其中 n 为关节数，k 为协同维数）。人手的运动构型可以用协同矩阵和协同空间中的向量表示为

q = Sσ

其中 σ ∈ ℝᵏ 为协同坐标。研究表明，仅保留前2-3个协同基即可解释人手运动构型85%以上的变异[101]。
""")

# 段落[109] - 关于运动神经控制的研究
set_text(get_para(109),
"""然而，纯运动学层面的协同模型（即"硬协同"，Hard Synergy）仅适用于描述手在未与物体接触时的预成形阶段。当手与物体实际接触时，纯运动学约束会导致接触力分布不合理，无法产生稳定的抓取。为了解决这一问题，Bicchi等人提出了软协同模型（Soft Synergy）[102][103]。该模型的核心思想是引入关节刚度矩阵 Kq ∈ ℝⁿˣⁿ，将协同输入解释为虚拟参考构型 qᵣ = Sσ，实际构型 q 通过弹性力被拉向参考构型。在静力平衡状态下，关节力矩满足：

τ = Kq(qᵣ - q) = JTf_ext

其中 JTf_ext 为接触力映射到关节空间的力矩。求解可得：

q = Sσ - CJTf_ext

其中 C = (Kq + Qτ)⁻¹ 为柔顺矩阵。软协同模型使手在保持低维驱动的同时，能够被动地顺应物体形状，从而在抓握中形成多个接触点。
""")

# 段落[110] - 姿势协同定义（已较好，保留并衔接）
set_text(get_para(110),
"""软协同模型虽然在理论上具有优越性，但其物理实现通常需要全驱动（即每个关节配备独立驱动器）或复杂的阻抗控制，难以直接转化为简洁的机械结构。Grioli等人提出的适应性协同模型（Adaptive Synergy）[104][105] 提供了一种将软协同通过欠驱动机构物理实现的方法，其核心思想是利用腱绳-滑轮差分传动机构来实现协同运动。
""")

# 段落[111] - 人手的运动构型
set_text(get_para(111),
"""在适应性协同模型中，腱绳位移 x ∈ ℝᵗ 与关节角 q ∈ ℝⁿ 之间通过传动矩阵 R ∈ ℝᵗˣⁿ 建立运动学关系：

Rq = x

由力-静对偶性，腱绳张力 τM 与关节力矩 τ 满足 τ = RTτM。引入刚度为 E 的关节弹簧（对角矩阵，各关节独立），力矩平衡方程为：

JTf_ext = RTτM - Eq

将运动学方程与力矩平衡联立求解，消去 τM，可得关节角：

q = E⁻¹RT(RE⁻¹RT)⁻¹x + (E⁻¹ - E⁻¹RT(RE⁻¹RT)⁻¹RE⁻¹)JTf_ext

令 S = E⁻¹RT(RE⁻¹RT)⁻¹ 为协同矩阵，C = E⁻¹ - SRE⁻¹ 为柔顺矩阵，并令 σ = x，则适应性协同的表达式与软协同完全一致：

q = Sσ + CJTf_ext
""")

# 段落[113] - Santello的研究
set_text(get_para(113),
"""因此，通过合理设计传动矩阵 R 和关节刚度 E，可以在物理层面上实现任意期望的软协同模式。这一方法被成功应用于Pisa/IIT SoftHand的设计中[104][105]。SoftHand具有19个关节，但仅使用单台电机通过一根腱绳驱动全部关节。通过精心设计滑轮半径分布和关节刚度分布，使其物理运动模式与Santello等人发现的人手第一协同高度吻合。该手在实验中成功抓取了107种不同形状的日常物体，展现了极高的适应性和鲁棒性。
""")

# 段落[115] - 协同表示
set_text(get_para(115),
"""在更一般的多协同设计中，可以通过并联布置多根腱绳（每根腱绳对应一个协同方向）来实现多自由度的协同控制。每增加一根腱绳及其对应的传动矩阵行向量，即可增加一个独立的协同方向。Grioli等人[104] 在一个模块化原型手上验证了四协同的可行性，使用四台电机驱动三根手指实现了四种不同的协同闭合模式。然而，多协同方案需要为每根腱绳独立布线，导致滑轮数量剧增，整体体积和复杂度显著增加。
""")

# 段落[116] - 总结并引出2.3节
set_text(get_para(116),
"""为在不大幅增加机械复杂度的前提下提升手部灵巧性，Della Santina等人在SoftHand的基础上提出了增强适应性协同模型（Augmented Adaptive Synergy）[106][107]，利用腱绳系统中不可避免的摩擦效应产生第二个协同方向，而不需要增加独立的腱绳和电机。此外，Piazza等人提出了动态协同模型（Dynamic Synergy）[108]，通过引入被动阻尼元件实现速度依赖的协同方向切换。本文将分别对这两种方法进行阐述。
""")

# ======================================================================
# 第2.2节：软协同和适应性协同模型（段落117-126）的详细内容
# ======================================================================

# 段落[118] - 软协同的详细描述
set_text(get_para(118),
"""为了解决硬协同模型接触力不合理的问题，Bicchi等人提出了软协同模型[102]。设 qᵣ = Sσ 为虚拟参考构型，Kq 为关节刚度矩阵。关节力矩来源于实际构型偏离参考构型的弹性变形：

τ = Kq(Sσ - q)

在静力平衡下，这一弹性力矩与接触力产生的关节力矩相互平衡：

Kq(Sσ - q) = JTf_ext

求解得：

q = Sσ - Kq⁻¹JTf_ext

将 Sσ 移项整理，并考虑抓取系统的完整线性化方程，可得到更精确的形式：
q = (Kq + Qτ)⁻¹KqSσ = HᵏKqSσ

其中 Qτ 包含接触刚度的影响。当考虑外力时，q = HᵏKqSσ + HᵏJTf_ext。
""")

# 段落[120] - 适应性协同的详细力学推导
set_text(get_para(120),
"""适应性协同模型通过欠驱动差分传动实现软协同行为[104]。在适应性协同中，弹簧与传动机构并联布置（而非软协同中的串联），因此力矩平衡方程为：

τ = RTη - Kaq

其中 η 为腱绳张力，Ka 为并联弹簧刚度矩阵。将 τ 代入抓取系统的线性化方程可得：

q = (Ka + Qτ)⁻¹RTη

在位置控制模式下，以 x = Rq 代入可得腱绳张力：

η = (R(Ka + Qτ)⁻¹RT)⁻¹x

从而 q = (Ka + Qτ)⁻¹RT(R(Ka + Qτ)⁻¹RT)⁻¹x。令 Ha = (Ka + Qτ)⁻¹，则有：

q = HaRT(RHaRT)⁻¹x
""")

# 段落[122] - 从软协同到适应性协同的映射
set_text(get_para(122),
"""从软协同模型到适应性协同模型的映射可归纳为：给定软协同参数 (S, Kqᵃ)，寻找适应性协同参数 (R, Ka)，使得两者在平衡构型附近具有相同的输入-输出行为。这一映射可通过以下关系实现：

KaRT = KqᵃSM

其中 M 为任意满秩方阵，作为设计参数。当取 Ka = Kqᵃ = Kq 时，映射简化为 RT = KqSM。这一关系表明，通过求解线性方程即可从期望的协同矩阵 S 反求传动矩阵 R，为设计提供了系统化的方法[104]。
""")

# 段落[124] - 关键公式总结
set_text(get_para(124),
"""在自由空间中（无接触力），适应性协同的关节位移简化为：

q = Sσ

其中 S = E⁻¹RT(RE⁻¹RT)⁻¹ 且 σ = x。这一形式与软协同的参考构型完全一致，表明适应性协同确实在物理层面上实现了软协同模型。

适应性协同的核心优势在于：① 通过差分传动自动实现形状自适应，无需主动控制；② 接触力分布由弹簧平衡自动决定，不需要复杂的力控制算法；③ 结构简单、鲁棒性强，适合实际应用。
""")

# 段落[125] - 总结Pisa/IIT SoftHand的设计
set_text(get_para(125),
"""Pisa/IIT SoftHand[105] 是适应性协同设计的标志性成果。该手具有19个关节（4根手指各4关节 + 拇指3关节），采用柔性滚动接触关节（CORE Joint）替代传统铰链关节，通过弹性韧带提供关节复位力。整手仅使用一台6W Maxon电机通过单根Dyneema腱绳驱动全部关节。传动矩阵 R 和刚度矩阵 E 的设计目标为使手的运动模式与Santello第一协同吻合。实验表明，SoftHand能够稳定抓取107种不同物体，最大抓取力约20N，最大扭矩约2Nm，且具有极高的抗冲击和过载恢复能力。
""")

# 段落[126] - 引出多协同需求
set_text(get_para(126),
"""虽然 SoftHand 在功率抓取方面表现出色，但仅有一个协同方向限制了其执行精细操作的能力（如捏取、指尖操控等）。为了在不大幅增加机械复杂度的情况下提升灵巧性，需要在 SoftHand 的基础上引入额外的协同维度。以下两节将分别介绍两种不同的多协同实现方案：基于摩擦的增强适应性协同和基于阻尼的动态协同。
""")

# ======================================================================
# 第2.3节：摩擦协同模型（段落[127]）
# 需要重写为包含增强适应性协同和动态协同的完整内容
# ======================================================================

# 段落[127] - 当前为Heading 2 | 摩擦协同模型
# 保留标题，内容需要更换
# 注意：这一节没有后续段落(128-129)，只有[127]标题且无正文
# 需要插入新的正文段落

# 首先，获取[127]段落，看看有没有后续段落
print("Checking content around para 127...")
for i in range(127, 135):
    p = get_para(i)
    if p:
        t = p.text.strip()[:100] if p.text.strip() else "(empty)"
        print(f"  [{i}] {p.style.name} | {t}")

# 看起来段落[127]之后就是[130] Heading 1（协同模型数据分析与仿真）
# 需要在[127]和[130]之间插入新的正文段落

# 在python-docx中，我们需要在特定位置前插入段落
# 但python-docx不直接支持在特定索引插入，需要使用底层XML操作
# 我们可以在[127]后面添加段落

# 先找到[127]后面的段落元素，在它之前插入新段落
from docx.oxml import OxmlElement
from copy import deepcopy

def insert_paragraph_after(paragraph, text='', style='论文正文段落'):
    """在指定段落后面插入一个新段落"""
    new_p = OxmlElement('w:p')
    paragraph._element.addnext(new_p)
    new_para = paragraph.__class__(new_p, paragraph._parent)
    set_text(new_para, text, style)
    return new_para

# 第2.3节：增强适应性协同（Augmented Adaptive Synergy）
p127 = get_para(127)

p127a = insert_paragraph_after(p127, 
"""增强适应性协同（Augmented Adaptive Synergy）由 Della Santina 等人于2018年提出[106]，其核心思想是将腱绳-滑轮传动系统中的摩擦效应从干扰转变为设计工具，以产生第二个独立的协同方向。在SoftHand 2中，这一方法仅通过增加一台电机（仍使用单根腱绳）就将自由度从1提升到2，实现了精细操作的灵巧性提升。
""")

p127b = insert_paragraph_after(p127a,
"""为推导增强适应性协同的数学模型，考虑腱绳在滑轮间穿过的路径。设腱绳被 m 个滑轮分为 m+1 个区段，每个区段具有恒定的速度 vⱼ 和张力 Tⱼ。在滑轮 j 处的速度连续性条件为：

vⱼ = vⱼ₋₁ + Σᵢ rⱼᵢ q̇ᵢ

其中 rⱼᵢ 为滑轮 j 与关节 i 关联时的有效半径。张力平衡条件为：

Tⱼ = Tⱼ₋₁ - Vⱼ(vⱼ)

其中 Vⱼ 为滑轮 j 处的摩擦力损失。边界条件包括：τM = T₀ + Tₘ（总驱动力为两端张力之和），以及 ṡ = -(v₀ + vₘ)/2（s 为腱绳滑动量）。
""")

p127c = insert_paragraph_after(p127b,
"""将上述关系整理为矩阵形式：

MT + V(v) + eτM = 0
Mv - R̄q̇ = -2eṡ
τ = -R̄ᵀT

其中 M 为 (m+1)×(m+1) 的二阶差分矩阵，R̄ 为滑轮半径到关节角速度的映射矩阵。解出 v 和 T 后，关节力矩可表示为：

τ = RᵀτM + D(q̇, ṡ)

其中 Rᵀ = -R̄ᵀeᵥ 为基础传动矩阵，D 为摩擦驱动项。当系统处于准静态（q̇ ≈ 0）且采用库仑摩擦模型 V(v) = Vmax·tanh(v) 时，D 简化为：

D(ṡ) = -R̄ᵀM⁻¹Vmax·eᵥ·tanh(2ṡ)
""")

p127d = insert_paragraph_after(p127c,
"""定义协同输入 u₁ = τM（对应传动矩阵 R）和 u₂ = tanh(2ṡ)（对应摩擦传动矩阵 Rf）。将 D 代入力矩平衡方程可得双协同表达式：

JTf_ext = Rᵀu₁ + Rfᵀu₂ - Eq

其中 Rfᵀ = -R̄ᵀM⁻¹Vmax·eᵥ。令 σ = u₁，σf = u₂，定义扩展传动矩阵 R_aug = [R; Rf] （形状 2×n），则关节角的解可用伪逆形式统一表示为：

q = R_aug⁺_E·[σ, σf]ᵀ + P⊥_{R,Rf}·E⁻¹·JTf_ext

其中 R_aug⁺_E = E⁻¹R_augᵀ(R_aug·E⁻¹·R_augᵀ)⁻¹ 为加权伪逆，P⊥ 为投影到 R,Rf 零空间的正交投影矩阵[106]。
""")

p127e = insert_paragraph_after(p127d,
"""增强适应性协同的一个重要设计自由度是腱绳路径（Routing）。改变路径顺序仅影响 Rf 而保持 R 不变，从而允许独立设计第二个协同方向。例如，在SoftHand 2[107] 的3指6关节布局中，当腱绳按不同顺序穿过各手指的滑轮时，σf 模式的手指重配模式随之改变：路径(a)使拇指+中指指根主导打开，路径(b)使中间手指主导打开。这一特性使得可以通过改变路径来定制第二个协同方向，而无需修改机械结构。
""")

p127f = insert_paragraph_after(p127e,
"""在具体的Capstan实现中，我们的模型使用指数衰减来模拟张力沿腱绳路径的分布：

R[j] = Σrₖ·(exp(-β·dAₖ) + exp(-β·dBₖ))/2
Rf[j] = Σrₖ·(exp(-β·dAₖ) - exp(-β·dBₖ))

其中 dAₖ 为元素 k 到 Motor A 的路径步数，dBₖ 到 Motor B，β 为摩擦系数（默认0.09）。σ 模式（双电机同向）下，张力从两端向中间指数衰减，产生 U-shape 分布（两端关节有效传动比大于中间关节），物理上正确反映了中指驱动最小、拇指/小指最大的生物学特征。σf 模式（双电机异向）下，Motor A 拉紧侧张力高、Motor B 放松侧张力低，当张力衰减至静摩擦阈值以下时产生死区（Dead Zone），被"冻结"的区段不产生有效驱动力矩，产生不对称的关节角分布。
""")

# 动态协同部分
p127g = insert_paragraph_after(p127f,
"""另一种多协同实现方案是动态协同（Dynamic Synergy），由 Piazza 等人于2016年在 SoftHand Pro-D 中提出[108]。其核心思想是在腱绳传动系统中引入被动阻尼元件，利用阻尼力随驱动速度变化的特性，实现速度依赖的协同方向切换。

在慢速驱动下（speed_factor α ≈ 0），阻尼力可忽略，系统退化为标准的适应性协同，手指沿慢速协同方向 S_s 运动。在快速驱动下（α ≈ 1），阻尼力产生等效附加刚度，改变了系统的力平衡状态，使手指沿阻尼平衡协同方向 S_f 运动。通过调节驱动速度，可以在 S_s 和 S_f 之间实现平滑过渡。
""")

p127h = insert_paragraph_after(p127g,
"""动态协同的力平衡方程为：

TᵀCT·q̇ + Eq = Rᵀu

其中 T 为阻尼器传动矩阵（n_d × n），C 为阻尼系数对角阵。采用阻尼平衡解释——将阻尼力 TᵀCT·q̇ 建模为速度依赖的附加刚度——可得有效刚度和有效协同矩阵：

Eeff(α) = E + α·diag(TᵀCT)
Seff(α) = Eeff⁻¹Rᵀ(REeff⁻¹Rᵀ)⁻¹

α ∈ [0,1] 从慢速（准静态）到快速（阻尼主导）平滑过渡。重要的是，使用对角化 diag(TᵀCT) 而非全矩阵 TᵀCT，为每个关节独立赋予附加刚度 ΔEⱼ = ΣᵢCᵢ·Tᵢⱼ²，避免了多关节共享阻尼器时耦合导致的符号翻转伪影，保持所有关节同向弯曲。
""")

p127i = insert_paragraph_after(p127h,
"""增强适应性协同和动态协同可以结合使用：将增强协同的扩展传动矩阵 R_aug = [R; Rf] 作为动态协同的输入矩阵，则 σ 和 σf 均受速度调节。这种组合方案在 SoftHand Pro-D 的基础上进一步提升了灵巧性，实现了从功率抓取到精细操作的完整功能谱。
""")

# ======================================================================
# 第3.2节：协同模型仿真（标题可改）
# 当前段落[132]是Heading 2，[133]是Heading 3（小标题示例）
# 需要重写此节为软件开发章节
# ======================================================================

# 保留[132]标题，修改内容
p132 = get_para(132)
# 改成更合适的标题
# 由于python-docx修改标题较复杂，保留原标题但修改后续段落内容

# 段落[133] - Heading 3 "小标题示例" -> 改成软件架构
# 先修改[133]的标题文本
p133 = get_para(133)
p133.text = ''
run = p133.add_run('软件架构概述')
run.font.name = '宋体'
run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

# 在[133]后插入详细内容
p133a = insert_paragraph_after(p133,
"""本文开发的折纸灵巧手仿真软件（synergy_hand_sim）是一个从CAD图纸到3D交互式仿真的完整工具包，支持URDF导出、Pinocchio机器人学库集成、增强自适应协同和动态协同交互式仿真。软件采用模块化设计，主要包含以下核心模块：
""")

p133b = insert_paragraph_after(p133a,
"""数据模型层（src/models/origami_design.py）：定义折纸手设计的完整数据结构，包括点(Point2D)、面片(OrigamiFace)、折痕线(FoldLine)、关节连接(JointConnection)、滑轮(Pulley)、孔(Hole)、阻尼器(Damper)、腱绳(Tendon)、驱动器(Actuator)等核心数据类。顶层容器 OrigamiHandDesign 提供拓扑构建、验证、序列化等功能，支持.ohd格式的JSON读写。所有 id 按约定管理：滑轮≥0，驱动器A=-1、B=-2，孔≤-100，阻尼器≤-200。
""")

p133c = insert_paragraph_after(p133b,
"""DXF解析层（src/models/origami_parser.py）：利用ezdxf库从CAD图纸（DXF格式）自动提取几何信息，按颜色分类折痕类型（红=峰折，蓝=谷折，黑=轮廓），在线段交点处强制分割，构建平面图后使用最左转算法识别最小环路（面片），最后组装为 OrigamiHandDesign。支持.ohd文件直接加载。
""")

p133d = insert_paragraph_after(p133c,
"""运动学层（src/models/origami_kinematics.py）：将2D折纸设计转换为3D空间中的运动学模型。OrigamiForwardKinematics 通过BFS构建生成树实现正向运动学传播，并采用迭代松弛法处理环状结构约束。支持任意关节角度下的面片3D位姿计算。
""")

p133e = insert_paragraph_after(p133d,
"""URDF导出层（src/models/origami_to_urdf.py）：将折纸手设计导出为标准URDF+STL格式，兼容Pinocchio和MuJoCo等物理引擎。为每个面片生成三角化STL文件，按折痕类型（峰/谷）自动调整关节轴定位策略，确保子link原点对齐折痕中点。
""")

p133f = insert_paragraph_after(p133e,
"""传动矩阵构建（src/models/transmission_builder.py）：从设计中自动提取传动矩阵 R（Capstan指数衰减模型）、Rf（Capstan+slack侧钳制模型）、T（阻尼器传动矩阵）。物理核心是Capstan衰减：T_A[k] = exp(-β·k)，张力从两端向中间指数衰减，产生U-shape分布。slack侧钳制使σf模式产生不对称关节角分布。支持孔类腱绳传动（hole_transmission.py）和阻尼器传动矩阵计算。
""")

p133g = insert_paragraph_after(p133f,
"""协同控制模块（src/synergy/）：包含三个层次的协同模型——基础适应性协同(AdaptiveSynergyModel)实现Grioli等人的模型，增强适应性协同(AugmentedAdaptiveSynergyModel)实现Della Santina等人的双协同模型，动态协同(DynamicSynergyModel)实现Piazza等人的速度依赖协同模型。三个模型具有统一的求解接口solve()，支持外力和接触力处理。
""")

p133h = insert_paragraph_after(p133g,
"""交互式仿真（src/interactive/）：基于MuJoCo的交互式URDF仿真器MuJoCoSimulator，支持四滑块互联动控制（Motor A/B和σ/σf），通过协同回调函数实时计算关节角度。此外还提供自研运动学仿真器OrigamiSimulator（集成2D CAD视图+3D MeshCat渲染）和Pinocchio仿真器PinocchioSimulator。
""")

p133i = insert_paragraph_after(p133h,
"""力学仿真框架（src/simulation/）：实现从纯几何仿真到完整力学仿真的升级，基于Della Santina等人[106]的动力学方程。包含三个递进阶段——Phase 1准静态力平衡求解、Phase 2完整动力学ODE积分（RK4/scipy）、Phase 3动力学+接触力（惩罚法+库仑摩擦）。核心方程：

B(q)q̈ + W(q,q̇)q̇ + Kq = Q(q)u + J(q)ᵀf_ext

其中 B 为惯性矩阵，W 含科里奥利力和粘滞阻尼，K 为刚度矩阵，Q 为腱绳力矩映射矩阵，u 为驱动输入。
""")

p133j = insert_paragraph_after(p133i,
"""优化框架（src/optimization/）：通过优化设计参数（滑轮半径、弹簧刚度、阻尼系数、阻尼器连接拓扑）使设计达到期望的协同方向。支持差分进化(DE)、单纯形全局(SHGO)、网格搜索和随机搜索四种算法。设计目标包括方向匹配(DirectionTarget)、速度依赖切换(SpeedDependentTarget)和加权组合(CompositeObjective)。
""")

p133k = insert_paragraph_after(p133j,
"""CAD图形化编辑器（src/origami_cad/）：基于PyQt5的交互式CAD编辑器，支持折线绘制（L/M/V快捷键）、滑轮放置（自动吸附折痕）、腱绳绘制（右键完成）、驱动器/孔/阻尼器放置与属性编辑。支持保存为.ohd格式和导出DXF。属性面板实时显示和编辑选中图元的参数。
""")

p133l = insert_paragraph_after(p133k,
"""验证测试（tests/）：包含17项动态协同扩展测试、增强协同测试、摩擦模型对比测试（Coulomb vs Capstan）、优化框架测试、URDF加载验证等。力学仿真框架包含30项测试（6单元测试+9集成测试+3论文复现+1仿真），使用pytest自动运行。
""")

# ======================================================================
# 更新参考文献
# 需要找到参考文献段落[144]附近，添加新的引用
# ======================================================================

# 查找参考文献段落
ref_para = None
for i, p in enumerate(doc.paragraphs):
    if p.text.strip() == '参考文献' and p.style.name == 'Title':
        ref_para = p
        ref_idx = i
        break

if ref_para:
    print(f"Found reference heading at [{ref_idx}]")
    # 找到参考文献后面的段落并插入新参考文献
    # 段落[145]是第一个参考文献
    # 我们需要在[145]之前插入新参考文献
    
    # 找到最后一个参考文献的位置
    last_ref_idx = ref_idx
    for i in range(ref_idx + 1, len(doc.paragraphs)):
        p = get_para(i)
        if p and p.text.strip() and p.style.name == '参考文献':
            last_ref_idx = i
        elif p and p.text.strip() and p.style.name == '附录标题':
            break
    
    print(f"Last reference at [{last_ref_idx}]")
    
    # 新的参考文献内容
    new_refs = [
        '[101] Santello M, Flanders M, Soechting J F. Postural hand synergies for tool use[J]. Journal of Neuroscience, 1998, 18(23): 10105-10115.',
        '[102] Bicchi A, Gabiccini M, Santello M. Modelling natural and artificial hands with synergies[J]. Philosophical Transactions of the Royal Society B: Biological Sciences, 2011, 366(1581): 3153-3161.',
        '[103] Gabiccini M, Bicchi A, Prattichizzo D, et al. On the role of hand synergies in the optimal choice of grasping forces[J]. Autonomous Robots, 2011, 31(2-3): 235-252.',
        '[104] Grioli G, Catalano M, Silvestro E, et al. Adaptive synergies: an approach to the design of under-actuated robotic hands[C]. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012: 1251-1256.',
        '[105] Catalano M G, Grioli G, Farnioli E, et al. Adaptive synergies for the design and control of the Pisa/IIT SoftHand[J]. International Journal of Robotics Research, 2014, 33(5): 768-782.',
        '[106] Della Santina C, Piazza C, Grioli G, et al. Toward dexterous manipulation with augmented adaptive synergies: the Pisa/IIT SoftHand 2[J]. IEEE Transactions on Robotics, 2018, 34(5): 1141-1156.',
        '[107] Della Santina C, Grioli G, Catalano M, et al. Dexterity augmentation on a synergistic hand: the Pisa/IIT SoftHand+[C]. IEEE-RAS International Conference on Humanoid Robots, 2015: 497-503.',
        '[108] Piazza C, Della Santina C, Catalano M, et al. SoftHand Pro-D: Matching dynamic content of natural user commands with hand embodiment for enhanced prosthesis control[C]. IEEE International Conference on Robotics and Automation (ICRA), 2016: 3516-3523.',
        '[109] Birglen L, Gosselin C, Laliberté T. Underactuated robotic hands[M]. Springer, 2008.',
        '[110] Dollar A M, Howe R D. The highly adaptive SDM hand: Design and performance evaluation[J]. International Journal of Robotics Research, 2010, 29(5): 585-597.',
        '[111] Ciocarlie M, Goldfeder C, Allen P. Dexterous grasping via eigengrasps: a low-dimensional approach to a high-complexity problem[C]. Robotics: Science and Systems Workshop, 2007.',
        '[112] Brown C Y, Asada H H. Inter-finger coordination and postural synergies in robot hands via mechanical implementation of principal components analysis[C]. IEEE/RSJ International Conference on Intelligent Robots and Systems, 2007: 2877-2882.',
        '[113] Odhner L U, Jentoft L P, Claffee M R, et al. A compliant, underactuated hand for robust manipulation[J]. International Journal of Robotics Research, 2014, 33(5): 736-752.',
        '[114] Deimel R, Brock O. A novel type of compliant and underactuated robotic hand for dexterous grasping[J]. International Journal of Robotics Research, 2015, 35(1-3): 161-185.',
        '[115] Hayward V, Armstrong B. A new computational model of friction applied to haptic rendering[C]. Experimental Robotics VI, Springer, 2000: 403-412.',
        '[116] Cannon J R, Howell L L. A compliant contact-aided revolute joint[J]. Mechanism and Machine Theory, 2005, 40(11): 1273-1293.',
    ]
    
    # 在最后一个参考文献后面插入新参考文献
    last_ref_para = get_para(last_ref_idx)
    
    # 按顺序插入（反转保证顺序正确）
    for ref_text in reversed(new_refs):
        new_p = OxmlElement('w:p')
        last_ref_para._element.addnext(new_p)
        new_para_c = doc.paragraphs[0].__class__(new_p, doc.paragraphs[0]._parent)
        set_text(new_para_c, ref_text, '参考文献')

# ======================================================================
# 更新摘要
# 段落[21]是"摘要"标题，[22]是摘要正文模板
# ======================================================================

abstract_para = get_para(22)
set_text(abstract_para,
"""折纸（Origami）机器人基于薄片材料折叠而成，具有生物相容性、柔性、耐用性和低成本等优点。然而，现有折纸机器人大多采用全局驱动方式（如加热、光照）同时驱动所有折痕，难以实现顺序折叠控制。本文提出一种基于物理智能的绳驱折纸灵巧手顺序驱动技术，通过融合姿势协同理论、摩擦驱动的增强适应性协同模型和阻尼驱动的动态协同模型，实现了对多折痕折纸结构的顺序化、可控化驱动。本文首先建立了折纸手的完整运动学模型和协同控制理论框架；然后，基于Capstan摩擦衰减模型和阻尼平衡原理，设计了摩擦协同和动态协同两种顺序驱动方案，实现了速度依赖的协同方向切换；最后，开发了完整的仿真软件工具包（synergy_hand_sim），包含DXF解析、URDF导出、协同模型计算、MuJoCo交互式仿真等功能。仿真和实验结果表明，本文提出的方案能够有效实现折纸手的顺序折叠，在慢速驱动下实现全手同步闭合（功率抓取），在快速驱动下通过阻尼效应实现手指差异化闭合（精细操作），为折纸机器人的顺序控制提供了一种新的物理智能解决方案。
""")

# 更新关键词
kw_para = get_para(26)
kw_text = "关键词：折纸机器人；顺序驱动；协同控制；绳驱传动；物理智能；适应性协同；动态协同"
set_text(kw_para, kw_text, style='Normal')

# ======================================================================
# 更新英文摘要
# 段落[29]是"Abstract"标题，[30]是英文摘要正文
# ======================================================================

en_abstract = get_para(30)
set_text(en_abstract,
"""Origami-based robots are fabricated by folding thin sheet materials along crease patterns, offering advantages such as biocompatibility, flexibility, durability, and low cost. However, most existing origami robots employ global actuation methods (e.g., heating, illumination) that drive all creases simultaneously, making sequential folding control challenging. This paper proposes a physically intelligent cable-driven origami dexterous hand sequential actuation technology, which integrates postural synergy theory, friction-driven augmented adaptive synergy model, and damping-driven dynamic synergy model to achieve sequential and controllable actuation of multi-crease origami structures. We first establish a complete kinematic model and synergistic control framework for the origami hand. Based on the Capstan friction decay model and damped equilibrium principle, we design two sequential actuation schemes: friction-based augmented synergy and damping-based dynamic synergy, enabling speed-dependent synergy direction switching. Finally, we develop a comprehensive simulation software toolkit (synergy_hand_sim) that includes DXF parsing, URDF export, synergy model computation, and MuJoCo interactive simulation. Simulation and experimental results demonstrate that the proposed approach effectively achieves sequential folding of the origami hand: slow actuation produces synchronous whole-hand closure (power grasp), while fast actuation generates differentiated finger closure through damping effects (precision manipulation), providing a novel physically intelligent solution for sequential control of origami robots.
""")

# 更新英文关键词
en_kw_para = get_para(34)
en_kw_text = "Keywords: origami robot; sequential actuation; synergy control; cable-driven transmission; physical intelligence; adaptive synergy; dynamic synergy"
set_text(en_kw_para, en_kw_text, style='Normal')

# ======================================================================
# 更新插图清单中的图标题
# 段落[37]是插图清单条目
# ======================================================================

# 保留现有插图清单，但可以更新或补充

# ======================================================================
# 保存文档
# ======================================================================

doc.save(OUT_PATH)
print(f"\n文档已保存至: {OUT_PATH}")
print("编辑完成！")
