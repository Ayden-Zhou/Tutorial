#!/usr/bin/env python
# coding: utf-8

# # Scaling Laws in Large Languge Models
# 
# [Scaling Laws for Neural Language Models](https://arxiv.org/pdf/2001.08361)
# 
# [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409)
# 
# [Explaining Neural Scaling Laws](https://arxiv.org/pdf/2102.06701)
# 
# [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556)
# 
# [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/pdf/2203.03466)
# 
# 
# 
# 
# ## 1. Scaling Laws 基础介绍与历史背景
# 
# ### 1.1 为什么要严肃对待 Scaling？
# 
# 想象这样一个场景：你手头突然拥有了一万张 H100 显卡的使用权，时长为一个月，目标是构建一个优秀的开源语言模型（LM）。在组建好基础设施团队和准备好预训练数据集之后，你面临的核心问题是：**应该训练哪一个模型？**
# 
# Scaling（规模扩展）并非易事。在模型设计层面，我们需要做出一系列选择：是把模型做宽还是做深？需要多少个注意力头（heads）？选择哪种非线性激活函数？
# 
# 回顾过去几年的模型发展（从 2017 年的原始 Transformer 到 GPT-4、LLAMA 系列），虽然我们可以“盲从”（cargo cult）现有的成功架构，但理解这些配置最初是如何被优化出来的至关重要。
# 
# 我们今天的目标是寻找简单的、具有预测性的“定律”来描述语言模型的行为：
# * **旧且痛苦的方法**：在超大模型上直接调整超参数。
# * **新且（可能过于）乐观的方法**：在小模型上调整，然后外推（extrapolate）到大模型。

# ## 1. Scaling laws, history and background
# 
# 这一部分我们将探讨Scaling Laws（缩放定律）的历史背景。主要关注数据规模（Data Scaling）如何作为经验样本复杂性影响模型性能，以及在这个领域早期的探索工作。
# 
# ### 1.1 样本复杂度与收敛率
# 
# 研究者们思考“Scaling（缩放）”这个问题已经有很长一段时间了。在传统的统计学习理论中，我们通常关注泛化误差的界限。
# 
# 例如，对于有限假设集 $k$，我们有如下的不等式：
# $$\epsilon(\hat{h})\le(min_{h\in H}\epsilon(h))+2\sqrt{\frac{1}{m}log\frac{2k}{\delta}}$$
# 
# 或者对于光滑密度的生成模型，估计器的收敛率可能表现为：
# $$\psi_{n}=n^{-\frac{\beta}{2\beta+1}}$$
# 这意味着均方误差的上界与 $\psi_{n}^{2}$ 成正比。
# 
# **关键点在于：** 这些通常是**上界（Upper Bounds）**，而不是实际观察到的、具体的损失值（Realized Loss Values）。虽然理论提供了误差随样本量 $n$ 衰减的趋势，但它们往往过于宽泛，不能精确预测大模型时代的具体表现。
# 
# <div align="center">
# <img src="../images/scaling-earliest.png" width="50%">
# <br>
# 最早关于 Scaling Law 的论文
# </div>
# 
# #### 1993年的发现：学习曲线的渐近值与收敛率
# 早在1993年，Cortes, Jackel, Vapnik等人（AT&T Bell Labs）就发表了关于学习曲线的早期论文。他们观察到典型的学习曲线具有以下特征：
# * 测试误差（Test Error）总是高于训练误差（Training Error）。
# * 随着训练集规模的增加，两者会渐近地趋向于一个共同的值 $a$。
# 
# 他们将大样本下的误差建模为幂律衰减（Power-law decays）：
# * 测试误差：$\mathcal{E}_{test}=a+\frac{b}{l^{\alpha}}$
# * 训练误差：$\mathcal{E}_{train}=a-\frac{c}{l^{\beta}}$
# 
# 这项工作不仅具有理论基础，还提出了一种实用的预测方法：通过在较小数据集上的表现来预测分类器在完整数据集上的适用性，从而避免在整个数据集上训练效果不佳的分类器，节省计算资源。
# 
# #### 2001年的转折：数据与算法的权衡
# 随后，Banko和Brill在2001年的工作中展示了对数线性（Log-linear）的Scaling关系。他们在处理混淆词消歧（Confusable Disambiguation）任务时发现：
# * 即便是简单的算法（如Perceptron, Naïve Bayes），随着数据量从100万词增加到10亿词，性能仍在持续提升。
# * 在当时常用的训练语料规模下，这些学习器远未达到性能瓶颈。
# 
# 这引发了一个重要的反思：我们是否应该重新权衡在“算法开发”与“语料库开发”上的投入？与其死磕复杂的算法，不如投入更多的时间和金钱去收集更多的数据。
# 

# ### 1.3 机器翻译中的函数形式测试 (2012)
# 
# 到了2012年，Kolachina等人开始测试不同的函数形式来拟合数据与下游性能之间的关系。在机器翻译任务中（使用BLEU分数作为指标），他们对比了多种曲线家族，包括：
# * 指数族（Exponential）
# * 幂律族（Power Law）
# * 对数族（Inverse Log）
# 
# 实验结果表明，数据量与下游性能之间确实存在幂律关系。
# 
# 

# ### 1.4 深度学习Scaling的早期探索：Hestness et al. (2017)
# 
# Hestness等人在2017年的工作是“大规模神经Scaling”领域的先驱，虽然在当时略显超前。他们的研究涵盖了机器翻译、语言模型和语音识别等多个任务，并揭示了深度学习中可预测的Scaling形状。
# 
# #### 三个关键区域
# 他们提出泛化误差随训练数据规模的变化呈现出三个明显的阶段（如下图所示）：
# 1.  **小数据区域（Small Data Region）：** 此时模型学不到什么东西，误差类似于随机猜测（Best Guess Error）。
# 2.  **幂律区域（Power-law Region）：** 随着数据增加，误差在双对数坐标系下呈线性下降。
# 3.  **不可约误差区域（Irreducible Error Region）：** 随着数据无限增加，误差趋于饱和，不再显著下降。
# 
# 
# <div align="center">
# <img src="../images/scaling-three-region.png" width="50%">
# <br>
# 数据Scaling的三个区间
# </div>
# 
# #### 关于“涌现”与小数据测试的挑战
# Hestness指出了一个关键问题：虽然我们希望通过小数据集测试来预测大模型的表现，但必须确保训练数据足够大，使得模型能够进入“幂律区域”。如果优化器参数或初始化不佳，或者数据过少，模型可能会出现“准确率悬崖”，一直停留在随机猜测水平。因此，定义“足够大”的训练集对于小规模测试至关重要。
# 
# #### 计算限制与预测的价值
# * **计算限制：** 在某些情况下，在超大数据集上训练大模型可能需要数月甚至数年的关键路径计算时间，这对实际系统来说是不切实际的。
# * **预测的作用：** 可预测的学习曲线和模型大小曲线提供了一种方法，可以推算出达到特定准确率所需的计算要求。这能指导我们如何扩展计算能力，以解锁那些受限于算力的应用。
# 
# #### 速度与精度的权衡
# 深度学习中的许多软件和硬件技术（如低精度计算/量化、稀疏模型）通常以牺牲一定的模型精度（例如损失20%）为代价来换取计算速度。
# Scaling曲线可以告诉我们，这种牺牲是否值得：如果计算吞吐量的提升允许我们在更大的数据集上训练更大的模型，那么这些因量化带来的精度损失可能很容易被弥补回来。

# ## 2. Neural (LLM) scaling behaviors
# 
# 这一部分我们将深入探讨大语言模型（LLM）的神经缩放行为（Scaling Behaviors）。主要关注三个核心问题：数据与性能的关系、模型规模与数据的权衡，以及超参数如何随规模变化。
# 
# ### 2.1 数据与性能的关系 (Data vs Performance)
# 
# #### 经验观测：对数-对数线性关系
# 对于语言模型而言，最基础的Scaling Law体现在数据量与Loss的关系上。Kaplan等人（2020）的研究表明，Loss与数据集大小（Dataset Size）在双对数坐标系（Log-Log Plot）下呈现出显著的线性关系。这被称为“无标度（Scale-free）”或“幂律（Power law）”行为。
# 公式形式通常为：
# 
# $$L = (D/C)^{- \alpha}$$
# 
# #### 理论基础：为何是幂律？
# 为什么会出现这种Scaling Law？
# * **单调性：** 显然，随着数据增加，泛化误差应该单调下降。
# * **估计误差的自然衰减：** 我们可以用一个简单的“均值估计（Mean Estimation）”例子来理解。假设我们要估计一组服从 $N(\mu, \sigma^2)$ 的数据的均值，估计量的方差为 $\sigma^2/n$。这意味着误差（Error）与 $n^{-1/2}$ 成正比。
# * **多项式速率：** 一般来说，任何 $1/n^{\alpha}$ 形式的多项式速率都可以被视为一种Scaling Law。
# 
# 
# #### 幂律指数的谜题
# 这里存在一个有趣的矛盾。
# * **经典理论预测：** 对于大多数经典模型（如回归），我们预期的Scaling速率通常是 $1/n$。这意味着在对数图上斜率应该是 -1。
# * **实际观测：** 神经Scaling Law中观察到的斜率要平缓得多。
#     * 机器翻译：$\alpha \approx 0.13 - 0.36$
#     * 语言模型：$\alpha \approx 0.095$
# 这表明神经模型的数据效率远低于简单的统计估计器。
# 
# #### 维度依赖的解释
# 对于非参数学习（Nonparametric Learning），其Scaling Law通常与维度 $d$ 有关，形式为 $n^{-1/d}$。
# Bahri (2021) 提出了一种解释：Scaling Law的斜率 $\alpha$ 与数据的**内在维度（Intrinsic Dimensionality）** 密切相关。数据流形越复杂（维度越高），学习效率越低，斜率越平缓。虽然这一理论目前还不是无懈可击，但提供了一个理解不同任务Scaling差异的视角。
# 
# #### Toy Example：任意函数逼近
# 这一部分我们做一个有趣的思维实验，假设我们的任务不是简单的参数估计，而是逼近一个定义在2D单位方块上的任意复杂函数 $f(x)$：
# * **输入：** $x_{1}, ..., x_{n}$ 均匀分布在边长为单位1的2D空间中，观测值 $y_{i}=f(x_{i})+N(0,1)$。
# * **方法：** 我们采用一种非参数的“切分”策略，将2D空间切割成边长为 $h$ 的小方块，并用每个方块内样本的均值作为该区域的预测值。
# 
# #### 误差推导：偏差-方差权衡
# 这里的总误差并非单调下降，而是取决于**近似误差（Bias）**与**估计误差（Variance）**的博弈：
# 1.  **偏差（Bias）：**
#     假设函数 $f(x)$ 是光滑的（满足 Lipschitz 连续性，即函数值的变化受距离限制），那么在边长为 $h$ 的方块内，用均值近似函数值的最大误差与方块大小成正比：
#     $$\text{Bias} \approx \max |f(x) - \text{Avg}| \propto h$$
#     *如果方块太大（$h$ 大），我们就会忽略函数内部的细节变化，导致欠拟合。*
# 
# 2.  **方差（Variance / Estimation Error）：**
#     方块内的样本数量 $m$ 取决于数据密度和方块面积：$m \approx n \cdot h^2$。根据中心极限定理，均值估计的标准误差（Standard Error）与样本数的平方根成反比：
#     $$\text{Est. Error} \propto \frac{1}{\sqrt{m}} = \frac{1}{\sqrt{n \cdot h^2}} = \frac{1}{h\sqrt{n}}$$
#     *如果方块太小（$h$ 小），落入的样本数就会太少，估计极不稳定。*
# 
# 3.  **最优切分（Sweet Spot）：**
#     总误差是两者的叠加。为了最小化总误差，我们需要寻找最优的 $h$：
#     $$\text{Total Error} \approx h + \frac{1}{h\sqrt{n}}$$
#     令两项平衡（$h \approx \frac{1}{h\sqrt{n}}$），解得最优切分粒度为 $h \approx n^{-1/4}$。
# 
# 4.  **最终Scaling：**
#     将最优 $h$ 代回误差公式，我们得到最终的 Scaling Law：
#     $$\text{Var} \propto n^{-1/2}$$
# 
# #### 维度诅咒与Scaling
# 如果我们将这个逻辑推广到 $d$ 维空间，为了维持同样的逼近精度，误差的Scaling Law将变为：
# $$\text{Var} = n^{-1/d}$$
# 这意味着在对数图上，Scaling的斜率是 $-\frac{1}{d}$（$y=-\frac{1}{d}x+C$）。
# 
# **核心结论：** 灵活的“非参数”学习受制于**维度诅咒（Dimension Dependent Scaling）**。维度 $d$ 越高，为了降低同样的误差所需的数据量就呈指数级增加。换句话说，在高维空间中，随着数据增加，误差下降的斜率会变得非常平缓。
# 
# <div align="center">
# <img src="../images/scaling-dimension.png" width="50%">
# <br>
# 不同维度数据的 scaling
# </div>
# 
# 
# 

# ### 2.2 其他数据Scaling规律
# 
# #### 分布偏移 (Distribution Shift)
# 数据组成对性能有何影响？
# Kaplan等人（2021）的研究发现，**数据的组成主要影响Offset（截距），而不影响Slope（斜率）**。这意味着无论是在WebText、Wikipedia还是Common Crawl上训练，Loss随数据增加的下降速度（斜率）是相似的，但不同质量的数据集会有不同的基础Loss水平。这强调了收集多样化、高质量数据的重要性。
# 
# <div align="center">
# <img src="../images/scaling-distribution-shift.png" width="75%">
# <br>
# 数据组成对于 Scaling 的影响
# </div>
# 
# #### 数据重复 (Data Repetition)
# 在实际应用中，数据往往是有限的。Muennighoff等人的研究展示了重复使用数据时的Scaling行为：
# * **收益递减：** 重复数据带来的收益会迅速递减。
# * **阈值：** 大约在重复4个Epoch以内，重复数据的效果几乎等同于新数据。但到了40个Epoch，重复数据就几乎毫无价值了。
# * **有效数据公式：** 可以通过公式计算“有效数据量（Effective Data）”，将重复次数作为惩罚项纳入考量。
# 
# <div align="center">
# <img src="../images/scaling-repeat.png" width="75%">
# <br>
# 重复数据对于 Scaling 的影响
# </div>
# 
# #### 低质量数据池与过滤策略
# 鉴于重复数据的价值较低，我们面临“质量 vs 数量”的权衡（Quality-Quantity Tradeoff）。
# * **自适应过滤：** 数据选择策略应该根据总计算量（Compute）进行调整。
# * **计算量较小时：** 应该采用高度激进的过滤策略，只用最高质量的数据（因为没空看完所有数据）。
# * **计算量较大时：** 过滤策略应更宽松，因为模型需要更多的数据来避免过拟合，哪怕是质量稍差的数据。
# 
# 
# 
# 
# 

# ### 2.3 模型工程的Scaling Laws (Model Engineering)
# 
# 我们的目标是高效地设计巨大的LLM。通过在小规模模型上建立Scaling Law，我们可以预测大模型的最佳设计选择，而无需在全规模上进行昂贵的试错。
# 
# #### 架构选择
# * **Transformer vs LSTM：** Kaplan等人（2021）通过Scaling曲线展示，Transformer在扩展性上显著优于LSTM。LSTM的Loss下降曲线更早趋于平缓。
# * **不同架构的普适性：** Tay等人（2022）对比了多种Transformer变体（如Performer, Switch Transformer等），发现大多数架构在Scaling Plot上都落在相似的曲线上。这表明架构的具体细节往往不如计算量（FLOPs）决定性大。
# 
# <div align="center">
# <img src="../images/scaling-architecture.png" width="75%">
# <br>
# 模型架构对于 Scaling 的影响
# </div>
# 
# #### 优化器选择 (Optimizer)
# * **Adam vs SGD：** 在Hestness等人（2017）的早期实验（RNNs）中，Adam和SGD表现出了不同的Scaling斜率。对于大模型训练，选择具有更好Scaling特性的优化器至关重要。
# 
# <div align="center">
# <img src="../images/scaling-optimizer.png" width="75%">
# <br>
# 优化器对于 Scaling 的影响：左图展示了验证集最小交叉熵损失随训练数据量（字符总数）增加的变化情况，结果显示在对数刻度下，泛化误差遵循清晰的幂律下降规律。尽管 Adam 优化器相比 SGD 能够将损失曲线向下平移约 5%，但两者的幂律指数（即学习曲线斜率）高度接近，分别为 -0.095 和 -0.094，表明优化器的改进仅改变截距而非学习效率。右图刻画了在各数据规模下达到最佳性能所需的模型参数量。实测表明，最佳模型规模随数据量呈亚线性增长，其中 SGD 优化模型的扩展指数为 0.78，而 Adam 优化模型虽然精度更高，但对参数规模的需求增长更为显著（指数为 0.92），在处理大规模数据时通常需要比 SGD 模型多出 8 至 11 倍的参数。
# </div>
# 
# #### 深度与宽度 (Depth/Width)
# * **层数的影响：** 0层、1层和2层之间差别巨大。但一旦超过一定深度（如6层以上），增加层数的边际收益在参数量小于 $10^7$ 时会迅速递减。
# * **参数的平等性：** 并非所有参数都是平等的。特别是Embedding层的参数，其Scaling行为与非Embedding参数（Non-embedding parameters）截然不同。在分析时通常建议排除Embedding参数。
# 
# <div align="center">
# <img src="../images/scaling-layers.png" width="50%">
# <br>
# 模型层数对于 Scaling 的影响
# </div>
# 
# 在设计模型时，我们经常纠结于具体的“形状”：到底是要深一点（Deep）还是要宽一点（Wide）？注意力头数（Heads）选8个还是16个？前馈网络（FFN）的比例是多少？
# 
# 令人惊讶的是，Scaling Law 告诉我们：**只要总非Embedding参数量（$N$）固定，模型的具体形状对性能的影响微乎其微。**
# 
# * **形状不敏感性（Shape Invariance）：** 实验数据展示了一个宽阔的“盆地”——在保持参数量不变的情况下，Loss随模型形状（如宽高比 $d_{model}/n_{layer}$）的变化非常平缓。即使我们将宽高比改变40倍，性能损失也往往控制在3%以内。
# * **算力补偿：** 那些因形状非最优而带来的微小性能差异，通常只需要增加极少量的计算资源（比如增加22%的FLOPs）就能完全弥补1%的Loss增加。
# * **工程自由度：** 这一发现对系统设计至关重要。这意味着我们可以主要根据**硬件效率**（如TPU/GPU的矩阵乘法效率、并行切分策略）来自由选择模型的长宽比，而不用担心这会显著损害模型的最终智能水平。例如，为了推理延迟更低，我们可以放心地把模型做得更宽、更浅。
# 
# <div align="center">
# <img src="../images/scaling-aspect-ratio.png" width="75%">
# <br>
# 模型比例对于 Scaling 的影响
# </div>
# 
# 
# 
# 

# #### Batch Size (Critical Batch Size)
# * **临界Batch Size：** Batch Size存在一个临界点。
#     * 小于临界值：增加Batch Size可以线性加速训练（完美Scaling）。
#     * 大于临界值：收益递减，增加Batch Size不再显著减少训练步数（无效Scaling）。
# * **与Loss的关系：** 目标Loss越低（模型越大/训练越久），临界Batch Size就越大。这意味着大模型训练天然适合使用更大的Batch Size。
# 
# <div align="center">
# <img src="../images/scaling-batch-size.png" width="75%">
# <br>
# Batch size 对于 Scaling 的影响
# </div>
# 
# <div align="center">
# <img src="../images/scaling-batch-size2.png" width="50%">
# <br>
# Batch size 对于 Scaling 的影响
# </div>
# 
# #### 学习率 (Learning Rate)
# * **$\mu P$ (Maximal Update Parametrization)：** 传统的参数化方法在模型宽度增加时，最优学习率会发生漂移，导致无法直接从小模型预测大模型的超参数。
# * **Yang et al (2022)：** 提出$\mu P$，通过特定的初始化和学习率缩放规则，使得最优学习率在不同模型宽度下保持稳定。这使得我们可以直接在小模型上调优LR，然后迁移到大模型上。
# $\mu P$（最大更新参数化）本质上是一套**初始化方差**和**学习率缩放**的规则。
# 
# 它的核心目标是：当模型宽度趋近于无限大时，确保每一层的**激活值（Activations）**和**权重更新量（Weight Updates）**在数值上保持稳定，既不消失也不爆炸，并且保持特征学习（Feature Learning）的能力。
# 
# 为了做到这一点，$\mu P$ 要求我们根据层的类型（Input, Hidden, Output）和宽度，对初始化和LR应用特定的乘数。例如：
# * 某些层的权重需要以 $1/width$ 初始化。
# * 某些层的LR需要随着 width 增加而增加。
# 
# <div align="center">
# <img src="../images/scaling-lr.png" width="75%">
# <br>
# 左图：Learning rate 对于 Scaling 的影响。右图：$\mu P$
# </div>
# 
# 
# #### 谨慎：下游任务的不可预测性
# 虽然预训练Loss（Perplexity）的Scaling是非常可预测的，但**下游任务（如SuperGLUE, 算术题等）的性能往往不可预测**。某些能力可能会突然“涌现（Emergence）”，也可能在特定的参数规模下表现出非单调的波动。

# ### 2.4 联合Scaling Laws：数据与模型 (Joint Scaling Laws)
# 
# 我们不仅需要知道数据或模型单独如何Scaling，更需要知道它们如何共同影响性能，以解决“我是该用更多数据还是更大模型？”的问题。
# 
# #### 联合误差公式
# Rosenfeld和Kaplan分别提出了联合Scaling Law公式：
# * **Rosenfeld:** $Error = n^{-\alpha} + m^{-\beta} + C$
# * **Kaplan** $Error = [m^{-\alpha} + n^{-1}]^{\beta}$
# 这些公式能够很好地拟合不同模型大小和数据量下的Loss曲面。
# 
# <div align="center">
# <img src="../images/scaling-joint.png" width="75%">
# <br>
# 参数量和数据量的联合 scaling law (Kaplan, 2020)
# </div>
# 
# 
# #### 计算量预算权衡 (Performance vs Compute Budget)
# Kaplan等人（2020）提出，在固定的计算预算（Compute Budget $C$）下，存在一个最优的模型大小和数据量的分配。
# * **Kaplan的结论：** 算力应该更多地分配给模型参数。他们认为模型大小的Scaling指数（~0.73）远大于数据Scaling指数（~0.27）。这意味着应该优先做大模型，而不是堆数据。

# ### 2.5 Chinchilla Scaling Laws (修正后的最优Scaling)
# 
# Kaplan的结论后来受到DeepMind团队（Hoffmann et al., 2022）的挑战。他们认为Kaplan高估了模型大小的重要性，因为Kaplan实验中没有充分调整学习率Schedule。
# 
# #### Chinchilla的三种拟合方法
# 1.  **固定模型大小，拟合包络线：** 对不同大小的模型进行训练，提取所有曲线的最小值包络线。
# 2.  **IsoFLOPs（等计算量线）：** 固定计算量，调整模型大小和Token数量，寻找Loss最低点。
# 3.  **参数化拟合：** 直接拟合联合Loss函数（注意：原文中方法3存在计算错误，后来被Besiroglu等人修正，但结论依然支持Chinchilla）。
# 
# #### Chinchilla的核心结论
# * **平分秋色：** 最优Scaling系数对于模型参数 $N$ 和训练数据 $D$ 大约都是 0.5。
#     * $N_{opt} \propto C^{0.5}$
#     * $D_{opt} \propto C^{0.5}$
# * **经验法则：** 计算量每增加10倍，模型大小和数据量应该各增加3.16倍。或者简单来说，对于给定的计算量，**模型参数量与训练Token数的比例应保持约为 1:20**。
# * **对比：** 相比于Kaplan建议的“大模型、少数据”，Chinchilla建议“小一点的模型、多得多的数据”。例如，Gopher（280B）如果是Chinchilla最优的，应该只有60B左右，但训练4倍的数据。

# ### 2.6 训练最优 vs 推理最优 (Train-optimal vs Inference-optimal)
# 
# Scaling Laws通常关注的是“训练计算量最优（Train-optimal）”。但在实际部署中，我们更关心推理成本。
# * **过度训练（Over-training）：** 如果一个模型预计会被大量使用，那么在训练阶段花费额外的算力（使用比Chinchilla最优更多的数据训练一个较小的模型）是值得的。因为小模型推理更便宜。
# * **实例：** LLaMA系列就是典型的“推理最优”模型。
#     * Chinchilla: ~20 tokens/param
#     * LLaMA 65B: ~22 tokens/param (接近Chinchilla)
#     * Llama 3 70B: ~215 tokens/param (远远超过Chinchilla最优，为了极致的推理性能)
# 
# ### 2.7 普适性
# 这种基于计算量的Scaling分析不仅适用于LLM，也已被验证适用于其他模态，如扩散模型（Diffusion Models），证明了这套方法论的强大普适性。
