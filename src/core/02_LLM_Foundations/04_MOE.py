# %% [markdown]
# # 混合专家模型 (Mixture of Experts) 
#
# 近年来，大语言模型（LLM）的发展趋势呈现出参数量急剧增长的态势。从早期的 BERT Large、GPT-2，到后来的 GPT-3 (175B)、MT-NLG (530B)，再到万亿参数级别的 GPT-MoE-1.8T，模型规模不断突破。
#
# 在这个过程中，我们观察到一个明显的**分叉点**：
# * **Dense Models（稠密模型）**：如 PaLM、Chinchilla、BLOOM。
# * **Sparse Models（稀疏模型/MoE）**：如 Switch Transformer、Mistral 8x22B、DeepSeek-V3、Llama 4 系列。
#
# 课件中特别提及了 **Llama 4** 系列（作为2025年的前沿代表）：
# * **Llama 4 Behemoth**：针对最智能的教师模型设计，拥有2000B+ 激活参数（总参数量更高）。
# * **Llama 4 Maverick**：追求速度与效率的平衡，128个专家。
# * **Llama 4 Scout**：更轻量级的多模态模型。
#
# ## 1. 混合专家模型 (Mixture of Experts) 概论
#
# ### 1.1 核心概念：什么是 MoE？
# MoE 的核心思想在于**将稠密模型中的前馈网络（FFN）层替换为多个并行的 FFN（即“专家”），并引入一个“路由（Router）”层**。
#
# * **Dense Model（稠密模型）架构**：
#     * 输入 $x$ -> Self-Attention -> Add+Norm -> **FFN Layer** -> Add+Norm -> 输出 $y$。
#     * **特点**：对于每一个输入 token，网络中**所有**的参数都会被激活参与计算。
#
# * **Sparse Model（稀疏模型/MoE）架构**：
#     * 输入 $x$ -> Self-Attention -> Add+Norm -> **Sparse FFN Layer** -> Add+Norm -> 输出 $y$。
#     * **Sparse FFN Layer 的工作机制**：
#         * 包含多个 FFN（例如 FFN 1, FFN 2, FFN 3, FFN 4...）。
#         * 引入一个 **Selector / Router（选择器/路由器）**。
#         * **路由机制**：对于输入 token（例如 "The" 或 "Dog"），Router 只会选择其中**一部分**专家（例如 Top-k 个）进行处理。
#     * **优势**：你可以在不增加推理时 FLOPs（浮点运算次数）的情况下，大幅增加模型的总参数量（# Experts）。
#
# <div align="center">
#   <img src="../images/moe-architecture.png" width="75%" />
#   <br>
#   MOE架构
# </div>
#
# <div align="center">
#   <img src="../images/moe-experts.png" width="75%" />
#   <br>
#   MOE专家分工
# </div>

# %% [markdown]
# ### 1.2 为什么 MoE 越来越流行？（Efficiency & Scaling）
# MoE 受到青睐的根本原因在于其极高的**训练与推理效率**：**在相同的计算预算（FLOPs）下，拥有更多参数的 MoE 性能更强。**
#
# **规模定律 (Scaling Laws)**
# * 随着专家数量（Experts）的增加（从 1 到 256），测试损失（Test Loss）显著下降。
# * 这意味着在稀疏模型参数量增加时，模型性能持续提升。
#
# **训练速度 (Training Speed)**
# * **更快的收敛**：对比 Switch-Base（MoE）与 T5-Base（Dense），MoE 模型在达到相同负对数困惑度（Neg Log Perplexity）时，速度明显更快。
# * **7倍加速**：数据表明，MoE 模型可以实现高达 7 倍的训练速度提升。
# * **更少的资源消耗**：要达到相同的性能基准（如 HellaSwag），MoE 所需的 FLOPs 或 Token 数量仅为稠密模型的 **1/3** 左右。
# <div align="center">
#   <img src="../images/moe-result.png" width="75%" />
#   <br>
#   MOE架构-训练
# </div>
#
# **MoE vs. Dense 直接对比**
# * 在同等算力条件下（例如都使用 128 张 H100），训练一个 1.3B 的稠密模型和一个总参数量 6.9B（激活参数 1.3B）的 MoE 模型：
#     * MoE 的训练 Loss 下降更快。
#     * MoE 的验证 Loss（Validation Loss）更低。
#     * MoE 在下游任务（如 HellaSwag）上的准确率提升速度快约 **2倍**。
# <div align="center">
#   <img src="../images/moe-result2.png" width="75%" />
#   <br>
#   MOE架构-Dense架构对比
# </div>

# %% [markdown]
# ### 1.3 为什么 MoE 以前没有普及？
# 既然 MoE 效果这么好，为什么直到最近才爆发？主要有两个障碍：
#
# 1.  **基础设施复杂性 (Infrastructure Complexity)**：
#     * **显存需求**：虽然计算量（FLOPs）没变，但总参数量巨大，需要大量的显存来存储所有专家。
#     * **并行策略**：MoE 非常依赖**多机多卡**环境。通常模型训练采用数据并行（Data Parallelism），不同的机器处理不同的数据切片。MoE 利用这些分布式机器来托管海量的模型参数。只有在拥有大量加速器（GPU/TPU）时，稀疏性优势才能体现。
#
# 2.  **训练不稳定性 (Training Instability)**：
#     * MoE 模型在训练过程中，Loss 曲线容易出现剧烈震荡（Spikes）。
#     * 相比于标准的稠密 Transformer，稀疏模型的训练更难收敛，且对超参数敏感。
#
# ### 1.4 MoE 的典型架构样貌
# 总结来说，一个标准的 MoE 层长什么样？
#
# * **主流做法 (Typical)**：将 Transformer Block 中的 MLP (Feed Forward) 层替换为 MoE 层。
#     * 结构：Self-Attention -> Add+Norm -> **[Router + Multiple Experts FFNs]** -> Add+Norm。
# * **非主流做法 (Less Common)**：也有研究尝试将 Attention 头也做成 MoE 形式（如 ModuleFormer, JetMoE），即对 Attention Heads 进行路由选择，但这目前不是主流。

# %% [markdown]
# ## 2. 路由机制详解 (Routing Mechanisms)
#
# MoE 架构中最关键的变量有三个：路由函数（Routing Function）、专家大小（Expert Sizes）和训练目标（Training Objectives）。本章重点拆解**路由函数**，即模型如何决定将输入 Token 分配给哪些专家。
#
# ### 2.1 路由算法概览
# 路由的核心问题是：面对输入 Token，我们该激活哪些专家？
#
# 目前的路由算法主要分为两类视角：
# 1.  **Token Choice (Token 选择专家)**：
#     * **机制**：这是目前最主流的做法（SOTA 模型几乎都用这个）。对于每个 Token，计算它与所有 Expert 的匹配分数，然后选择 Top-K 个分数最高的 Expert。
#     * **效果**：Token 掌握主动权，确保每个 Token 都能被它“最喜欢”的专家处理。
# 2.  **Expert Choice (Expert 选择 Token)**：
#     * **机制**：专家挑选它想处理的 Token，或者通过全局规划（Global Assignment）来分配。
#     * **问题**：虽然理论上能更好地负载均衡，但在实际的大规模语言模型训练中不如 Token Choice 普遍。
#
# <div align="center">
#   <img src="../images/routing-function.png" width="75%" />
#   <br>
#   MOE中的Router方法概览
# </div>

# %% [markdown]
# ### 2.2 常见路由变体 (Common Variants)
#
# #### 2.2.1 Top-K Routing (最主流)
# 这是目前绝大多数 MoE 模型（如 Switch Transformer, Mixtral, DeepSeek, Qwen）采用的标准方案。
# * **原理**：通过一个门控网络（Router，通常是一个线性层 + Softmax），计算 Token $t$ 对每个 Expert $i$ 的分数 $s_{i,t}$。
#     $$s_{i,t} = \text{Softmax}(u_t^T e_i)$$
#     然后取 $s_{i,t}$ 最大的 $K$ 个 Expert 进行激活。
# * **各模型配置**：
#     * **Switch Transformer**: $k=1$ (最极端稀疏)
#     * **Gshard, Grok, Mixtral**: $k=2$
#     * **Qwen, DBRX**: $k=4$
#     * **DeepSeek**: $k=7$ 或 $8$
#
# #### 2.2.2 Hash Routing (哈希路由)
# * **原理**：使用固定的哈希函数将 Token 映射到 Expert。
# * **地位**：通常作为 Baseline（基准线）存在。因为它不需要学习路由参数，如果你的 Learned Router 效果还不如 Hash，那就说明设计有问题。
#
# #### 2.2.3 Reinforcement Learning (强化学习路由)
# * **原理**：将 Expert 选择看作一个决策过程，使用 RL (如 REINFORCE 算法) 来优化路由策略，Loss 为 $-\log P \times \text{Reward}$。
# * **现状**：理论上很美好（因为路由选择是离散的，不可导，RL 正好能解），但由于梯度方差大、实现复杂，**工业界很少使用**。
#
# #### 2.2.4 Linear Assignment (线性分配/全局规划)
# * **原理**：通过求解线性分配问题（Solve Linear Assignment），强制在 Batch 级别实现完美的负载均衡。
# * **现状**：早期工作（如 Bengio 2013）或特定论文（如 Clark '22）中使用，目前不是主流。

# %% [markdown]
# ### 2.3 前沿路由策略：细粒度专家 (Advanced Strategies)
# 在 DeepSeek 推出的MOE架构中，路由机制出现了显著的创新，主要集中在细粒度（Fine-grained）和共享（Shared）机制上。
#
# #### 2.3.1 传统 Top-K 的局限与演进
# * **(a) 传统 Top-2**：标准的 MoE 做法，简单直接。
# * **(b) + Fine-grained Expert Segmentation (细粒度专家分割)**：
#     * **做法**：保持总参数量不变，将原来的“大专家”切分成更多的“小专家”。例如，把 1 个大 FFN 拆成 4 个小 FFN。
#     * **优势**：增加了专家的总数量 $N$，使得知识的切分更精细，组合更灵活。
# * **(c) + Shared Expert Isolation (共享专家隔离)**：
#     * **做法**：将 Expert 分为两类：
#         1.  **Routed Experts (路由专家)**：按需激活，走 Top-K 路由。
#         2.  **Shared Experts (共享专家)**：**总是被激活**，处理所有 Token。
#     * **直觉**：共享专家负责捕获**通用知识**（General Knowledge），路由专家负责捕获**特定知识**。这避免了路由专家在通用知识上的冗余学习。
#     * **代表模型**：**DeepSeek-V2/V3, Qwen 1.5**。
#
#
# <div align="center">
#   <img src="../images/moe-fine-grained.png" width="75%" />
#   <br>
#   DeepSeek MOE
# </div>
#
# <div align="center">
#   <img src="../images/moe-deepseek.png" width="75%" />
#   <br>
#   DeepSeek MOE
# </div>

# %% [markdown]
# #### 2.3.2 消融实验结论 (Ablations)
# * **DeepSeek 的发现**：
#     * 细粒度分割（Fine-grained segmentation）对性能有提升。
#     * 共享专家（Shared expert isolation）也能进一步提升性能。
#     * **结论**：更多的专家数量 + 共享专家机制 = 更强的模型表现。
# * **OlMoE 的发现**：
#     * 确认了细粒度专家带来的增益。
#     * 但在其特定实验设置下，共享专家似乎没有带来明显的额外收益（这表明不同架构或规模下结论可能略有不同）。

# %% [markdown]
# ### 2.4 各主流模型的路由配置大盘点 (Summary Table)
#
# | 模型 (Model) | Routed Experts | Active (k) | Shared Experts | 特点 |
# | :--- | :--- | :--- | :--- | :--- |
# | **Switch Transformer** | 64 | 1 | 0 | 极致稀疏，早期代表 |
# | **Mixtral** | 8 | 2 | 0 | 经典的 Top-2，无共享 |
# | **Grok** | 8 | 2 | 0 | 类似 Mixtral 结构 |
# | **DBRX** | 16 | 4 | 0 | 更多的活跃专家 |
# | **Qwen 1.5** | 60 | 4 | 4 | 引入共享专家，细粒度切分 |
# | **DeepSeek v3** | 256 | 8 | 1 | 极高的专家数量，Top-8 激活 |
# | **Llama 4 (Maverick)** | 128 | 1 | 1 | 2025新模型，极度稀疏 (Top-1) 但带共享 |
#
# **总结趋势**：
# 1.  **专家数量变多**：从早期的 8/16 个增加到 DeepSeek/Llama 4 的 100+ 甚至 200+。
# 2.  **共享专家成为标配**：为了弥补稀疏性带来的通用知识缺失，DeepSeek, Qwen, Llama 4 均引入了 Shared Experts。
# 3.  **细粒度切分**：Fine-grained ratio 越来越高（如 DeepSeek v3 达到 1/14 甚至更高），追求更灵活的参数组合。

# %% [markdown]
# ## 3. MoE 的训练挑战与系统实现 (Training & Systems)
#
# 训练一个高性能的 MoE 模型远比稠密模型复杂。核心矛盾在于：我们需要稀疏性来提高训练效率，但**稀疏的门控决策（Sparse Gating Decision）通常是不可导的**。此外，如何让海量的专家在硬件上高效运行也是系统设计的难点。
#
# ### 3.1 核心难题：路由的可导性
# 在反向传播中，如果路由决策是离散的（非 0 即 1），梯度永远为0。为了解决这个问题，一个最直接的思路就是引入概率。
#
# $$y = \sum_{i} p_i \cdot E_i(x)$$
# 其中 $p_i = \text{Softmax}(s_i)$，$E_i(x)$ 是专家 $i$ 的输出。
#
# 当我们反向传播时，Loss $L$ 对 Router 原始分数 $s_i$ 的导数通过**链式法则**计算如下：
#
# $$
# \frac{\partial L}{\partial s_i} = \underbrace{\frac{\partial L}{\partial y}}_{\text{1. 上游梯度}} \cdot \underbrace{\frac{\partial y}{\partial p_i}}_{\text{2. 专家贡献}} \cdot \underbrace{\frac{\partial p_i}{\partial s_i}}_{\text{3. 概率敏感度}}
# $$
#
# 相应的如何更新routing function也有一些对应的方法：
# 1.  **强化学习 (Reinforcement Learning)**：
#     * **思路**：使用 REINFORCE 算法优化门控策略。
#     * **现状**：虽然理论上是最“正统”的解法，但由于梯度方差大、训练不稳定且实现复杂，目前在工业界**并不常用**。
# 2.  **随机扰动 (Stochastic Perturbations)**：
#     * **思路**：给 Router 的输出加噪声，通过引入随机性来平滑梯度，同时鼓励探索（Exploration），防止模型过早陷入只选某几个专家的局部最优。
#     * **Noisy Top-k Gating (Shazeer et al., 2017)**：
#       在计算 Logits 时加入可学习的高斯噪声：
#       $$s_i = \underbrace{(x \cdot W_g)_i}_{\text{Raw Score}} + \underbrace{\mathcal{N}(0, 1) \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)}_{\text{Stochastic Noise}}$$
#       
#       然后继续应用标准的 Top-k 和 Softmax：
#       $$p_i = \text{Softmax}(\text{KeepTopK}(s_i, k))$$
#       
#       *注：这里使用 `Softplus` 是为了保证噪声的幅度（Standard Deviation）始终为正数。这样设计使得梯度不仅能传导回 $W_g$，还能传导回控制噪声幅度的参数 $W_{\text{noise}}$。*
#
#     * **Input Jitter (Switch Transformer)**：
#       尝试在输入端加入乘性噪声来增加鲁棒性（虽然后来的研究中有些被移除了）。
# 3.  **启发式平衡 Loss (Heuristic 'Balancing' Losses)**：
#     * **现状**：这是目前**最主流**的做法。既然离散选择不可导，那就通过添加辅助 Loss 来强制模型“学会”均匀分配 Token。

# %% [markdown]
# ### 3.2 负载均衡机制 (Load Balancing)
#
# MoE 训练中最大的噩梦是**负载不均（Load Imbalance）**。如果 Router 发现某几个专家特别好用，就会把所有 Token 都发给它们，导致：
# * 少数专家过劳（计算瓶颈）。
# * 大多数专家“摸鱼”（参数得不到更新，造成算力浪费）。
# * 这就是所谓的Collapse（坍塌）现象。
#
# #### 3.2.1 经典的辅助 Loss (Auxiliary Loss)
# 这是 Switch Transformer 确立的标准范式。
# * **目标**：最小化 $\text{Loss} = \alpha \cdot N \cdot \sum f_i \cdot P_i$。
#     * $f_i$：实际分发给专家 $i$ 的 Token 比例。
#     * $P_i$：Router 预测分配给专家 $i$ 的概率均值。
# * **效果**：强制 $f_i$ 和 $P_i$ 接近均匀分布，确保每个专家处理的数据量大致相同。
#
# #### 3.2.2 DeepSeek V2 的改进：专家与设备双重平衡
# DeepSeek V1-V2 沿用了类似的思路，但做得更细致：
# * **Per-expert Balancing**：确保每个专家吃到的 Token 数量均衡。
# * **Per-device Balancing**：确保每张显卡（Device）上的计算负载均衡（因为一张卡可能托管多个专家）。
#
# #### 3.2.3 DeepSeek V3 的创新：去辅助 Loss 化 (Aux-loss-free)
# 在 DeepSeek V3 中，团队发现辅助 Loss 可能会干扰主任务的学习。因此，他们提出了一种**无辅助 Loss 的平衡策略**：
# * **Bias 机制**：再选择专家时，给每个专家的分数 $s_i$ 增加一个偏置项 $b_i$，但是在计算梯度的时候仍然使用 $s_i$，也就是选择的概率和计算的概率分开，增加小概率专家被选择的频率。
# * **动态调整**：如果某个专家过载了，就降低它的 $b_i$（让它难被选中）；如果闲置了，就提高 $b_i$。
# * **优势**：这种方法不需要在 Loss 函数里加额外的项，被称为 "Auxiliary-loss-free balancing"，虽然为了防止极端序列不平衡，他们还是保留了一个序列级别的补充 Loss。
#
# <div align="center">
#   <img src="../images/moe-load-balance.png" width="75%" />
#   <br>
#   Load Balance
# </div>

# %% [markdown]
# ### 3.3 系统实现与并行策略 (Systems & Infrastructure)
#
# MoE 的高效依赖于大规模的并行计算设施，特别是当参数量大到单卡无法放下时。
#
# #### 3.3.1 并行模式
# * **数据并行 (Data Parallelism)**：复制整个模型，切分数据。
# * **模型并行 (Model Parallelism)**：切分模型层。
# * **专家并行 (Expert Parallelism)**：这是 MoE 独有的。将不同的专家放置在不同的 GPU/TPU 上。
#     * 当 Token 被路由到不同设备上的专家时，需要进行 **All-to-All** 的通信（Dispatch 发送 -> Combine 接收）。
#
# <div align="center">
#   <img src="../images/moe-paralllelism.png" width="75%" />
#   <br>
#   Parallelism in MOE：图中每个 4 * 4 的虚线网格代表 16 个核心（Core），阴影方块则表示该核心上承载的数据（包括模型权重或 Token 批次）。途中展示了在不同策略下，模型权重张量和数据张量是如何被拆分的。第一行（权重拆分）：展示了模型权重如何在核心之间进行切分。本行中不同尺寸的形状代表了前馈网络（FFN）层中较大规模的权重矩阵（例如更大的 d_ff 维度）。阴影方块的不同颜色标识了唯一的权重矩阵。虽然每个核心承载的参数量是固定的，但规模更大的权重矩阵会对每个 Token 应用更多的计算量。第二行（数据拆分）：展示了数据批次（Data Batch）如何在核心之间进行切分。每个核心持有相同数量的 Token，以确保在所有策略下都能维持固定的内存占用。不同的切分策略具有不同的特性，允许每个核心拥有相同或不同的 Token，图中的不同颜色即象征了这些不同的 Token。
# </div>
#
# #### 3.3.2 矩阵乘法优化 (Matrix Multiplication)
# 在底层计算中，如何高效处理稀疏数据是关键：
# 1.  **Batched GEMM**：将同一专家的 Token 拼在一起做矩阵乘法。
# 2.  **Block Diagonal GEMM**：将专家计算视为对角块矩阵乘法。
# 3.  **Block Sparse GEMM**：这是现代库（如 MegaBlocks）的做法。它允许变长的专家负载和负载不均，通过稀疏矩阵乘法直接加速，而不必强求完美的负载均衡。
#
# #### 3.3.3 路由的生命周期
# 一个 Token 在 MoE 层中的典型旅程：
# 1.  **Routing**：计算概率，决定去哪个专家。
# 2.  **Permutation (置换)**：根据目的地对 Token 进行重新排序和分组。**注意**：这里通常会发生 **Token Dropping（丢弃）**。如果分配给某专家的 Token 超过了其缓冲区容量（Capacity Factor），多余的 Token 会被直接丢弃（不参与计算）。这意味着别人的 Query 可能会把你的 Token 挤掉！
# 3.  **Computation**：专家进行计算。
# 4.  **Un-permutation**：把算完的 Token 还原回原来的顺序。
#
#
#
# <div align="center">
#   <img src="../images/moe-pipeline.png" width="75%" />
#   <br>
#   MOE中的处理流水线
# </div>

# %% [markdown]
# ### 3.4 稳定性与微调 (Stability & Fine-tuning)
#
# #### 3.4.1 训练不稳定性 (Training Instability)
# MoE 模型的 Loss 曲线经常出现剧烈的震荡（Spikes），比稠密模型更难伺候。
# * **原因**：`Exp` 函数的特性导致 logits 的微小扰动会被放大。特别是在 **bfloat16** 精度下，舍入误差可能导致 Softmax 输出剧变（例如从 0.09 变成 0.14）。
# * **解决方案：Router Z-loss**。
#     * 公式：$L_z(x) = \log^2(\sum e^{x_j})$。
#     * **原理**：惩罚过大的 Logits 值，强制 Router 输出较小的数值，从而保持在数值稳定的范围内。实验证明，加上 Z-loss 后，Loss 曲线平滑了许多，且随着训练进行性能更好。
#
# <div align="center">
#   <img src="../images/moe-loss-spike.png" width="75%" />
#   <br>
#   MOE训练中的不稳定性
# </div>
#
# #### 3.4.2 微调难题 (Fine-tuning Issues)
# MoE 模型在预训练（大数据量）时表现出色，但在微调（小数据量）时容易**过拟合 (Overfit)**。
# * **现象**：在 SuperGLUE 等小样本任务上，MoE 的微调效果有时甚至不如同等规模的稠密模型。
# * **解决方案 1 (Zoph et al.)**：微调时冻结 MoE 参数，只更新非 MoE 层（如 Attention 层）。
# * **解决方案 2 (DeepSeek)**：大力出奇迹。使用海量的 SFT 数据（1.4M 样本），涵盖数学、代码、推理等多个领域，直接进行全量微调。

# %% [markdown]
# ## 4. 进阶训练技术与 DeepSeek 架构演进
#
# 在掌握了 MoE 的基础与路由机制后，本章将探讨如何“从零开始”之外的捷径——Upcycling，并深入剖析当前 MoE 领域的集大成者 DeepSeek V3 的架构演化之路。
#
# ### 4.1 Upcycling：从稠密到稀疏的捷径 (Upcycling)
# **核心理念**：我们是否真的需要从头开始训练一个 MoE？能否利用已有的、训练好的稠密（Dense）模型作为起点？
# 这就是 **Upcycling（升级/回收利用）** 的概念。
#
# * **操作步骤 (The Recipe)**：
#     1.  **复制非 MoE 层**：直接复制原稠密模型的 Attention 层、Layer Norm 层等参数。
#     2.  **克隆 MLP**：将原稠密模型中的 MLP 层复制 $E$ 份，作为 MoE 的 $E$ 个初始专家。
#     3.  **重置 Router**：Router（路由）部分通常需要从头随机初始化。
#     4.  **继续预训练**：在此基础上进行后续的训练。
#
# * **效果对比**：
#     * 相比于从零开始训练（Base），Upcycling 在相同的计算量下能达到更高的验证集准确率（Validation Accuracy）。
#     * 它极大地节省了预训练时间（Extra Pretraining Time）。
#
# * **成功案例**：
#     * **MiniCPM-MoE**：从 MiniCPM-2.4B 稠密模型 Upcycling 而来。仅用了约 520B 的 token 进行训练，性能就显著提升。
#     * **Qwen-MoE**：基于 Qwen-1.8B 初始化，构建了 60 个专家（4 个共享）。这是早期确认 Upcycling 有效的代表性工作之一。
#
# ### 4.2 DeepSeek MoE 的架构演进 (Architecture Evolution)
# DeepSeek 团队在 MoE 架构上进行了快速且高密度的迭代，从 V1 到 V3，每一代都有针对性的系统级优化。
#
# #### 4.2.1 DeepSeek MoE V1
# * **配置**：16B 总参数 / 2.8B 激活参数。
# * **核心贡献**：确立了 **Shared Experts（共享专家） + Fine-grained Experts（细粒度专家）** 的混合结构。
#     * 设置了 2 个共享专家（总是被激活）。
#     * 将路由专家切分得非常细（64个），每次激活 4 个。
#
# #### 4.2.2 DeepSeek MoE V2
# 随着模型规模扩大（236B 总参数），跨 GPU 的通信成为瓶颈。V2 重点解决了这个问题。
# * **Device-Limited Routing (Top-M 路由)**：
#     * **问题**：如果 Top-K 选出的专家分散在太多不同的显卡上，通信开销会很大。
#     * **策略**：对于每个 Token，先选出 $M$ 个目标**设备（Device）**（这些设备包含了得分最高的专家）。然后在这些设备内部进行 Top-K 专家的选择。
#     * **效果**：强制 Token 只去少数几个设备，减少了跨节点通信。实践中 $M \approx 3$ 效果最好。
# * **Communication Balancing Loss**：
#     * 除了专家负载平衡，V2 还引入了**通信平衡 Loss**。
#     * 目标：确保每个设备接收到的 Token 数量大致相等，防止某个设备因处理过多 Token 而成为短板（Straggler）。
#
# #### 4.2.3 DeepSeek MoE V3
# * **配置**：671B 总参数 / 37B 激活参数。258 个路由专家（8 个激活），1 个共享专家。
# * **路由升级 (Sigmoid + Softmax)**：
#     * 计算路由分数时，先经过 `Sigmoid`（控制了score的数值范围），再做归一化选出 Top-K。
#     * 公式：$s_{i,t} = \text{Sigmoid}(u_t^T e_i)$。
# * **负载均衡策略 (Aux-loss-free)**：
#     * 采用 **Bias 调节** 机制（详见第 3 章笔记）来替代传统的辅助 Loss。
#     * *注：为了防止单条序列内的极端不平衡，还是保留了一个序列级（Sequence-wise）的辅助 Loss。*
#
# ### 4.3 DeepSeek V3 的“黑科技”组件 (Bonus Technologies)
# 除了 MoE 架构本身，DeepSeek V3 还集成了两项关键技术来提升效率和性能。
#
# #### 4.3.1 MLA: 多头潜在注意力 (Multi-Head Latent Attention)
# * **痛点**：大模型的 KV Cache（键值缓存）占用显存极大，限制了推理时的最大 Batch Size 和序列长度。
# * **核心思想**：**低秩压缩（Low-Rank Compression）**。
#     * 不再直接存储巨大的 K 和 V 矩阵。
#     * 而是将 Attention 的 Key 和 Value 投影到一个低维的**潜在向量 (Latent Vector, $c_t^{KV}$)** 中。
#     * 在推理时，只需要缓存这个极小的压缩向量 $c_t^{KV}$。
# * **优势**：大幅降低了推理时的显存占用（Memory Savings），使得在有限硬件上能处理更长的上下文。
#
# #### 4.3.2 MTP: 多 Token 预测 (Multi-Token Prediction)
# * **核心思想**：模型不仅预测下一个 Token ($t_{i+1}$)，还顺便预测之后得 $t_{i+2}$ 等 Token。
# * **实现方式**：
#     * 在主模型之外，添加轻量级的 **MTP 模块**（包含 Transformer Block 和 投影层）。
#     * MTP 模块 1 负责根据当前隐层预测“下下个” Token。
#     * *注：虽然架构图展示了多步预测，但 DeepSeek V3 最终配置似乎主要侧重于向后多预测 1 个 Token。*
# * **用途**：
#     1.  **训练信号**：提供更密集的梯度信号，帮助模型学得更好。
#     2.  **投机采样 (Speculative Decoding)**：在推理时，这些额外的预测头可以用于加速生成（类似于 EAGLE 算法），一次生成多个 Token 并验证。
#
# ### 4.4 章节总结
# * **MoE 已成主流**：得益于稀疏性带来的训练与推理效率优势，MoE 正迅速成为高性能大模型的标配。
# * **工程落地可行**：尽管离散路由（Discrete Routing）在理论上难以优化，但通过 Top-K 启发式算法、负载均衡 Loss 以及精心设计的系统架构（如 DeepSeek 的优化），MoE 已经具备了极高的实用性和性价比。
