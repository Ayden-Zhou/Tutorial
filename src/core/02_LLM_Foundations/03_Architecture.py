# %% [markdown]
# # Architecture
#
#
#
# Transfomer架构是现代大模型的基石，自提出起，transformer被不断修改，2017 年原始论文中的设计（Original Transformer）和今天主导 LLM 界的“现代”架构（Modern Variant）之间已经存在显著差异。
#
# ### 1. “标准” Transformer 的演变
#
# 当我们谈论 Transformer 时，现在的“标准配置”和 2017 年那篇《Attention Is All You Need》里的原始版本已经有了很大不同。
#
# #### 原始 Transformer vs. 现代实现
# 我们在原始论文上看到的原始架构通常包含：
# *   **位置编码**：正弦/余弦函数（Sinusoidal）。
# *   **前馈网络（FFN）**：使用 ReLU 激活函数。
# *   **归一化**：Post-Norm（归一化层在残差连接之后） + LayerNorm。
#
# 而如果你现在去扒开开源模型的代码，你会发现一套完全不同的“现代标准”：
# *   **位置编码**：旋转位置编码（RoPE）。
# *   **前馈网络**：SwiGLU 激活函数。
# *   **归一化**：Pre-Norm（归一化层在输入端） + RMSNorm。
# *   **偏置项（Bias）**：全连接层和归一化层通常都不再使用偏置项。
#
# <div align="center">
#   <img src="../images/transformer.jpg" width="50%" />
#   <br>
#   Transformer架构
# </div>

# %% [markdown]
#
# ### 2. 归一化策略：唯一的共识？
#
# 在众多的架构选择中，**Pre-Norm**（前置归一化）可能是目前业界唯一的绝对共识。
#
# *   **Pre-Norm vs. Post-Norm**
#     *   **Post-Norm**（原始设计）：归一化放在残差连接之后。BERT 时代还在用这个。
#     *   **Pre-Norm**（现代设计）：归一化放在子层（Attention/FFN）的输入端，不干扰主要的残差路径。
#     *   **结论**：除了 OPT-350M 这种少数派，2024 年以后的模型几乎清一色使用 Pre-Norm。
#     *   **原因**：直觉上，Pre-Norm 保持了残差流的“纯净”；实践中，它带来了更好的梯度传播稳定性，减少了训练发散的风险。
#
# *   **LayerNorm vs. RMSNorm**
#     *   **LayerNorm**：标准的层归一化。
#     *   **RMSNorm**：去掉了均值中心化（re-centering）的操作，只做缩放。
#     *   **趋势**：RMSNorm 正在成为主流（LLaMA, Gopher, Chinchilla 都在用）。
#     *   **优势**：在实践中效果和 LayerNorm 一样好，但因为少算一个均值，计算效率更高，且参数更少。这也符合现代架构设计的极简趋势——如果一个操作（如 Bias）对性能没有明显贡献但会占用计算资源，那就砍掉它。
#

# %% [markdown]
# #### 2.1 Pre V.S. Post Norm
#
#
#
#
# ##### 实验对比
# <div align="center">
#   <img src="../images/pre_post_norm1.png" width="100%" />
#   <br>
#   Pre V.S. Post Norm
# </div>
# <div align="center">
#   <img src="../images/pre_post_norm2.png" width="100%" />
#   <br>
#   Results: Pre V.S. Post Norm
# </div>
#
# **稳定性分析：** 实验显示 Post-LN 在初始化阶段的梯度期望在深层会迅速放大，而 Pre-Norm 的梯度在各层分布非常均匀。 
#
# **学习率容忍度：** Pre-Norm 模型在没有 Warmup 的情况下依然可以稳定训练，而 Post-LN 必须配合 Warmup 才能避免早期发散。 

# %% [markdown]
# #### 2.2 RMSNorm V.S. LayerNorm
# ##### 数学过程
#
# *   **LayerNorm (层归一化)**
#     *   **公式**: $y = \frac{x - E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta$
#     *   **特点**: 减去均值（中心化），除以标准差（缩放）。它认为特征的“平移”和“放缩”都很重要。
#
# *   **RMSNorm (均方根归一化)**
#     *   **公式**: $y = \frac{x}{RMS[x]} * \gamma$, 其中 $RMS[x] = \sqrt{\frac{1}{n} \sum x_i^2 + \epsilon}$
#     *   **特点**: **不减均值**，直接除以元素的平方根。它认为只要保证特征的“激活强度（量级）”稳定即可。
#
# ##### Example

# %%
import torch
import torch.nn as nn

# 设置随机种子
torch.manual_seed(42)

def compare_norm():
    # 模拟一层隐藏层的输出 (Batch=1, Hidden_dim=4)
    # 输入数据特意选择了一个较大的偏移 (均值为 25)
    x = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    
    print(f"--- 原始输入 x ---")
    print(f"数值: {x}")
    print(f"均值: {x.mean().item():.2f}")
    print(f"RMS: {torch.sqrt(torch.mean(x**2)).item():.2f}\n")

    # 1. PyTorch 官方 LayerNorm
    # 初始化 weight=1, bias=0 以观察原始变换
    ln = nn.LayerNorm(4)
    nn.init.ones_(ln.weight)
    nn.init.zeros_(ln.bias)
    
    ln_out = ln(x)
    print(f"--- LayerNorm 输出 ---")
    print(f"数值: {ln_out}")
    print(f"输出均值: {ln_out.mean().item():.2f} (被拉回了0)")
    print(f"输出方差: {ln_out.var(unbiased=False).item():.2f} (被固定在1)\n")

    # 2. RMSNorm 手写实现 (常用在 LLaMA 等模型)
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            # 核心区别：直接除以均方根，不减均值
            rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight

    rmsnorm = RMSNorm(4)
    rms_out = rmsnorm(x)
    print(f"--- RMSNorm 输出 ---")
    print(f"数值: {rms_out}")
    print(f"输出均值: {rms_out.mean().item():.2f} (保留了原始的偏移趋势)")
    print(f"输出 RMS: {torch.sqrt(torch.mean(rms_out**2)).item():.2f} (缩放到了1)")

if __name__ == "__main__":
    compare_norm()

# %% [markdown]
# ##### 为什么 RMSNorm 更好？
# **RMSNorm** 取代 LayerNorm 并非因为数学更精准，而是为了解决**硬件层面的“数据搬运”瓶颈**。
#
# **机制简化：砍掉“减均值”**
# * **LayerNorm**：既要“平移”（减均值）也要“缩放”（除方差）。
# * **RMSNorm**：**只缩放，不平移**。公式简化为 $y = \frac{x}{\text{RMS}(x)} * \gamma$。
#     * *逻辑*：只要保证数值的“幅度（量级）”稳定，模型就能稳定训练，无需强行中心化。
#
# **算力误区：FLOPs 不是重点**
# * 在 Transformer 中，矩阵乘法占了 99% 以上的计算量，归一化层仅占不到 1%。
# * 因此，仅靠减少数学运算次数（FLOPs），RMSNorm 带来的性能提升微乎其微。
#
# **真正优势：节省内存带宽 (Memory Bandwidth)**
# 这才是 RMSNorm 胜出的关键：
# * **带宽密集型任务**：归一化层的特点是“计算少、数据读写多”（算存比低）。GPU 的瓶颈往往卡在**显存读写速度**，而非计算核心的速度。
# * **减少搬运**：RMSNorm 省去了计算均值的步骤，减少了内存访问和同步的开销。这种对**内存带宽**的节省，直接转化为更快的实际训练速度（Wallclock Time）。
#

# %% [markdown]
# #### 2.3 补充：Double Norm (Non-residual Post-norm)
#
# 在 2024 年后的最新架构中（如 **Grok**, **Gemma 2**, **OLMo 2**），出现了一种对 Pre-norm 结构的进一步改良，被称为 **"Double Norm"**。
#
# 1. 核心逻辑：解决残差路径的“污染”
# * **Pre-norm 的局限性**：虽然 Pre-norm 极大地提升了训练稳定性，但它将 LayerNorm 置于子层（Attention/FFN）的输入端，这意味着归一化后的信号会不断累加进主残差路径（Main Residual Stream），可能导致主路径信号在极深网络中变得不够“纯净”。
# * **Double Norm 的折中**：它在保留 Pre-norm 稳定性的基础上，在子层输出后、回到主残差路径**之前**，额外增加了一个非残差的 Post-norm（层后归一化）。
#
#
#
# 2. 结构对比
# * **Standard Pre-norm**: 
#   $x_{l+1} = x_l + \text{Sublayer}(\text{Norm}(x_l))$
# * **Double Norm**: 
#   $x_{l+1} = x_l + \text{Norm}(\text{Sublayer}(\text{Norm}(x_l)))$
#   * 注意：这里的第二个 Norm 位于残差相加（Addition）操作之外，因此它不会直接修改主干路径的梯度流，但能对支路输出进行二次约束。
#
# 3. 为什么这样做？
# * **初衷**：历史上 Post-norm 的优势在于收敛后的性能，Pre-norm 的优势在于训练稳。Double Norm 试图兼顾两者。
# * **稳定性与性能**：通过在每个 block 内部进行双重归一化，可以进一步抑制数值爆炸或梯度尖峰，尤其在超大规模模型和超大学习率训练时，表现出极强的稳定性。
#
# <div align="center">
#   <img src="../images/double_norm.png" width="75%" />
#   <br>
#   Double norm
# </div>

# %% [markdown]
# ### 3. 激活函数：Gated Variants 的胜利
#
# 前馈神经网络（FFN）中的激活函数也经历了一场从简单到复杂的进化：`ReLU` -> `GeGLU` -> `SwiGLU`。
#
# *   **从 ReLU 到 GLU**
#     *   传统的 FFN 是简单的 `Linear -> ReLU -> Linear`。
#     *   现在的趋势是使用门控线性单元（Gated Linear Units, GLU）的变体。
#     *   简单理解：不仅仅是将输入 $x$ 通过激活函数，而是将输入分成两路，一路做变换，另一路作为“门”来控制信息流：$\text{Activation}(xW) \otimes (xV)$。
#
# *   **SwiGLU 是当下的最常用的选择**
#     *   公式：$\text{SwiGLU}(x) = (\text{Swish}(xW) \otimes xV) W_2$
#     *   **代价与调整**：引入门控机制意味着多了一组参数（$V$）。为了保持参数总量和标准 Transformer 一致，通常会将隐藏层维度（$d_{ff}$）缩小到原来的 $2/3$ 左右。
#     *   **采用者**：PaLM, LLaMA 1/2/3, Mistral, OLMo 等几乎所有 2023 年后的主流模型。

# %% [markdown]
# #### 3.1 对比：ReLU vs. SwiGLU
#
#
# **Standard FFN (ReLU)**:
#     $$ \text{FFN}(x) = \text{ReLU}(xW_1) W_2 $$
# **逻辑**：单路变换。线性投影后直接截断负值。
#
# **SwiGLU FFN**:
#     $$ \text{FFN}(x) = (\text{Swish}(xW) \odot xV) W_2 $$
#  **逻辑**：双路门控。$\odot$ 代表**逐元素乘法** (Element-wise Product)。
# 其中 **Swish** (也称为 SiLU) 激活函数定义为：
#     $$ \text{Swish}(x) = x \cdot Sigmoid(x) = \frac{x}{1 + e^{-x}} $$

# %%
import torch
import torch.nn.functional as F

torch.manual_seed(42)
# 模拟一组中间层数据，包含正数和负数
x = torch.tensor([[-1.2, -0.5, 0.0, 0.8, 2.0]])

# 1. ReLU: "一刀切"
# 凡是小于 0 的直接变成 0，信息彻底丢失
relu_out = F.relu(x)

# 2. SwiGLU 核心逻辑: "软门控"
# Gate 决定保留多少，Value 是原始信息
# 这里简化模拟：假设 W 和 V 都是单位矩阵，只看激活行为
gate = F.silu(x)  # Swish 激活 (x * sigmoid(x))
value = x         # 另一路原始信息
swiglu_out = gate * value

print("--- 输出对比 ---")
print(f"原始输入:   {x.numpy()}")
print(f"ReLU 输出:  {relu_out.numpy()}  <-- 负值全为 0 ")
print(f"SwiGLU输出: {swiglu_out.numpy()} <-- 负值保留了微弱响应，曲线更平滑")

# %% [markdown]
# #### 3.2 SwiGLU 的维度缩放与参数对齐
#
#
# 1. 参数量对比 (Parameter Counting)
# 假设输入维度为 $d$，中间隐藏层维度为 $d_{ff}$：
#
# * **标准 FFN (ReLU):** 包含两个矩阵：$W_{up}$ ($d \to d_{ff}$) 和 $W_{down}$ ($d_{ff} \to d$)。
#     * 总参数 $\approx 2 \times (d \cdot d_{ff})$
#     * 当 $d_{ff} = 4d$ 时，总参数为 **$8d^2$**。
#
#
#
# * **SwiGLU FFN:** 包含三个矩阵：$W$ (Gate), $V$ (Content) 和 $W_2$ (Down)。
#     * 总参数 $\approx 3 \times (d \cdot d_{ff})$
#     * 若强行维持 $d_{ff} = 4d$，总参数将膨胀至 **$12d^2$**（即原来的 1.5 倍）。
#
#
#
# 2. “8/3 规则”的由来
# 为了保持总参数量依然等于 **$8d^2$**，我们需要求解新的 $d_{ff}$：
# $$3 \cdot d \cdot d_{ff} = 8d^2$$
# $$d_{ff} = \frac{8}{3}d \approx 2.66d$$
#
#

# %% [markdown]
# ### 3.3 激活函数的平滑演进：从 ReLU 到高维门控
#
# 除了目前最主流的 SwiGLU，研究者在探索“平滑性”与“非单调性”的过程中还产出了几种关键的激活函数变体。它们的共同直觉是在 $x < 0$ 区域保留一个微小的“下凹” (Dip)，允许极小比例的负信息通过，从而有效缓解了传统 ReLU 导致的神经元永久坏死问题。
#
# * **GELU (Gaussian Error Linear Unit)**
#     * **公式**：$f(x) = x \cdot \Phi(x)$，其中 $\Phi(x)$ 为标准正态分布的累积分布函数 (CDF)。
#     * **设计直觉**：引入了随机正则化的思想，可以看作是输入依据其值的大小，按概率决定是否被“激活”。
#     * **工业地位**：是 BERT、GPT-2、GPT-3 等早期大模型时代的标配。
#
# * **GeGLU (GELU Gated Linear Unit)**
#     * **公式**：$\text{GeGLU}(x) = \text{GELU}(xW) \otimes (xV)$。
#     * **设计直觉**：它是 SwiGLU 的直接前身。在相同的门控框架下，将激活函数从 Swish 换回了 GELU。
#     * **工业地位**：Google 的 PaLM 模型便选用了该方案，通过实验证明门控机制相较于单一激活函数有显著增益。
#
# * **Mish**
#     * **公式**：$f(x) = x \cdot \tanh(\text{softplus}(x))$。
#     * **设计直觉**：追求极致的平滑度和梯度流的连续性。
#     * **应用场景**：虽然计算复杂度略高，但在一些对训练稳定性要求极高的深层模型（尤其是计算机视觉领域）中表现优异。
#
#
#
# <div align="center">
#   <img src="../images/activation-function.png" width="75%" />
#   <br>
#   常用激活函数
# </div>

# %% [markdown]
#
# ### 4. 模块并行化：Parallel Layers
#
# 传统的 Transformer 块是*串行（Serial）的：
# > $x \rightarrow \text{Attention} \rightarrow \text{Add} \rightarrow \text{FFN} \rightarrow \text{Add} \rightarrow \text{Output}$
#
# 但还有一种并行（Parallel）的变体（最初在 GPT-J 中流行，后来被 PaLM 采用）：
# > $y = x + \text{MLP}(\text{LayerNorm}(x)) + \text{Attention}(\text{LayerNorm}(x))$
#
# *   **原理**：Attention 和 MLP 不再有先后依赖，而是同时基于同一个输入 $x$ 进行计算，最后再加在一起。
# *   **优势**：
#     1.  **速度**：在大规模训练下，由于矩阵乘法可以融合（Fuse），训练速度能提升约 15%。
#     2.  **效率**：可以复用 LayerNorm 的计算。
# *   **劣势**：在小模型（如 8B 参数）上可能会有轻微的性能下降，但在超大模型（如 62B+）上，这种性能差异基本消失。
# *   **典型案例**：PaLM, Cohere Command A, Falcon 2。

# %% [markdown]
# ### 5. 位置编码：RoPE 的统治
#
# 在 Transformer 的架构演变中，位置编码经历了从“绝对”到“相对”，再到“旋转”的进化。目前，**RoPE (Rotary Position Embeddings)** 已经成为统治级的主流方案。
#
# ### 5.1 位置编码的家族谱系
#
# Transformer 本质上是排列不变（Permutation Invariant）的，如果不加位置信息，模型就无法区分“我爱你”和“你爱我”。历史上出现过几种主要的解决方案：
#
# * **正弦/余弦编码 (Sine/Cosine Embeddings)** - Original transformer
#     * **公式**：$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$
#     * **直觉解释**：利用不同频率的正弦和余弦波形生成固定的位置向量，叠加到输入上。就像时钟的指针，通过不同周期的组合唯一标识位置，且便于模型推断相对距离。
#
# * **绝对位置编码 (Absolute Embeddings)** - GPT-1/2/3
#     * **公式**：$x_i = x_i + p_i$
#     * **直觉解释**：为序列中的每个绝对位置（如第1个、第2个词）学习一个独立的向量，直接加到词向量上。相当于给每个位置贴上了一个固定的、可学习的标签。
#
# * **相对位置编码 (Relative Embeddings)**
#     * **公式**：$Attention(i, j) = \text{softmax}\left(\frac{q_i k_j^T + b_{i-j}}{\sqrt{d_k}}\right)$
#     * **直觉解释**：不改变输入向量，而是在计算注意力分数时加入一个仅与相对距离 $i-j$ 有关的偏置项。模型关注的是词与词之间的“距离”，而非它们的“坐标”。
#
# * **旋转位置编码 (RoPE)** - LaMA, PaLM, Mistral 以及绝大多数 2024 年后的模型。
#     * **公式**：$f(x, m) = R_m x$ （其中 $R_m$ 是根据位置 $m$ 计算的旋转矩阵）
#     * **直觉解释**：通过将词向量在向量空间中进行旋转来注入位置信息。旋转的角度取决于绝对位置，而向量点积的结果自然地只取决于相对旋转角度（即相对距离），完美融合了绝对和相对位置信息。
#
#
#
# #### 5.2 RoPE：旋转位置编码的核心思想
#
# ##### 动机：我们要什么样的“相对性”？
# 我们要寻找一种编码方式 $f(x, i)$，使得两个 Token 的内积（即 Attention Score）只依赖于它们的相对距离 $i-j$，而不是绝对位置 $i$ 或 $j$。
# 即满足：
# $$\langle f(x, i), f(y, j) \rangle = g(x, y, i-j)$$
#
# 之前的方案为什么不够完美？
# * **Sine**: 包含 $ \langle v_x, v_y \rangle + \langle PE_i, v_y \rangle ...$ 等交叉项，并不是纯粹的相对关系。
# * **Absolute**: 显然是绝对的。
# * **Relative**: 在 Attention 矩阵上硬加一个 Bias，而不是通过 Embedding 的内积自然诱导出来。
#
# ##### 直觉：用旋转代表位置
# RoPE 的灵感来源于**复数乘法**。
#
# 想象一个二维平面：
# * 我们有一个词向量 "we" 和 "know"。
# * 我们将“位置”定义为**旋转的角度**。
# * 如果 "we" 在位置 0，"know" 在位置 1，它们之间的夹角差异是 $\theta$。
# * 如果我们把整个句子平移，"we" 在位置 2，"know" 在位置 3，虽然它们的绝对角度都变了（都转了更多圈），但它们之间的**相对夹角**依然是 $\theta$。
# * **结论**：向量的点积（对应 Attention Score）与两个向量的夹角余弦值有关。只要相对旋转角度不变，点积就不变。这就是 RoPE 实现“相对位置编码”的几何直觉。
#
#
# #### 5.3 RoPE 的数学实现
#
# 在高维空间（$d_{model}$）中，RoPE 将向量维度两两分组（类似于复数），每一组 $(x_1, x_2)$ 在各自的 2D 子空间内进行旋转。
#
# ##### 旋转矩阵
# 对于 Query 或 Key 向量 $x_m$ 在位置 $m$ 处，我们将其乘以一个块对角旋转矩阵 $R_{\Theta, m}^d$：
#
# $$f_{\{q,k\}}(x_m, m) = R_{\Theta, m}^d W_{\{q,k\}} x_m$$
#
# 这个矩阵长这样（稀疏的，本质上是两两旋转）：
#
# $$
# R_{\Theta, m}^d = \begin{pmatrix}
# \cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots \\
# \sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots \\
# 0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots \\
# 0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots \\
# \vdots & \vdots & \vdots & \vdots & \ddots
# \end{pmatrix}
# $$
#
# * **与 Sine Embedding 的区别**：RoPE 是**乘法**性质的（旋转），而 Sine 是**加法**性质的。RoPE 没有引入额外的交叉项，完美保留了相对位置信息。
#

# %%
import torch
import math

# 1. Sinusoidal 位置编码 (加法式)
class SinusoidalEmbedding:
    def __init__(self, d_model=2):
        self.d_model = d_model

    def forward(self, x, pos):
        # 极简还原：pe = [sin(pos), cos(pos)]
        pe = torch.tensor([math.sin(pos), math.cos(pos)])
        return x + pe

# 2. RoPE 旋转位置编码 (乘法式)
class RoPEEmbedding:
    def __init__(self, d_model=2):
        self.d_model = d_model

    def forward(self, x, pos):
        # 极简还原：将向量旋转 pos 弧度
        # x' = [x0*cos - x1*sin, x0*sin + x1*cos]
        angle = float(pos)
        cos, sin = math.cos(angle), math.sin(angle)
        
        # 旋转矩阵逻辑
        x_rotated = torch.zeros_like(x)
        x_rotated[0] = x[0] * cos - x[1] * sin
        x_rotated[1] = x[0] * sin + x[1] * cos
        return x_rotated

def demo_comparison():
    sin_enc = SinusoidalEmbedding()
    rope_enc = RoPEEmbedding()

    # 初始设置：两个相同的词向量 q 和 k
    q = torch.tensor([1.0, 0.0])
    k = torch.tensor([1.0, 0.0])

    print(f"原始向量 q, k: {q.tolist()}, {k.tolist()}\n")

    def test_relative_consistency(name, model):
        print(f"--- 测试 {name} ---")
        # 场景 1: 位置 (1, 3) -> 相对距离 2
        q1 = model.forward(q, 1)
        k3 = model.forward(k, 3)
        dot1 = torch.dot(q1, k3)
        
        # 场景 2: 位置 (11, 13) -> 相对距离 2 (整体平移了 10)
        q11 = model.forward(q, 11)
        k13 = model.forward(k, 13)
        dot2 = torch.dot(q11, k13)
        
        print(f"位置 (1, 3)   的注意分数 (点积): {dot1:.4f}")
        print(f"位置 (11, 13) 的注意分数 (点积): {dot2:.4f}")

    test_relative_consistency("Sinusoidal (加法)", sin_enc)
    test_relative_consistency("RoPE (旋转)", rope_enc)

if __name__ == "__main__":
    demo_comparison()

# %% [markdown]
#
# ### 6. 超参数
#
# 相比于早期的百花齐放，现在的 LLM 在超参数选择上已经趋于保守和统一。大多数模型遵循以下“拇指法则”：
#
# #### 6.1 FFN 维度比率 (FeedForward Ratio)
# 我们要把输入维度 $d_{model}$ 放大多少倍作为 FFN 的中间维度 $d_{ff}$？
#
# * **标准共识**：$d_{ff} = 4 \times d_{model}$。
# * **GLU 变体的调整**：如前所述，为了抵消 GLU 引入的额外参数，通常使用 $d_{ff} = \frac{8}{3} d_{model}$（约为 2.67 倍）。
# * **反面教材**：T5 曾尝试过惊人的 64 倍放大（$d_{ff}=65536$），但这被证明是计算上的浪费，后来的 T5 v1.1 改回了标准的 2.5 倍。
# <div align="center">
#   <img src="../images/d_ff.png" width="75%" />
#   <br>
#   Feef-forward ratio 与 Loss 的关系
# </div>

# %% [markdown]
#
# #### 6.2 注意力头维度 (Head Dimension)
# * **黄金法则**：$Head\_Dim \times Num\_Heads = Model\_Dim$。
# * **主流选择**：大多数模型（如 LLaMA, GPT-3）都遵循这一比例，并没有出现明显的低秩瓶颈（Low rank bottlenecks）。虽然 Google 的一些模型（PaLM, T5）曾尝试过不同的比例，但收益不明显。
#
# <div align="center">
#   <img src="../images/head-dimension.png" width="75%" />
#   <br>
#   注意力头维度
# </div>

# %% [markdown]
# #### 6.3 深宽比 (Aspect Ratio)
# 模型应该是做深（Deep）还是做宽（Wide）？
#
# * **并行性考量**：极深的模型（Layer 数很多）难以并行化，因为每一层都必须等待上一层计算完成，这会增加推理延迟。相反，增加宽度（Width）可以轻松地在更多 GPU 上并行计算。
# * **结论**：在一定的 FLOPs 预算下，不同深宽比的模型性能差异惊人地小（Kaplan et al., 2020）。
# * **现状**：大多数模型的 $d_{model}/n_{layer}$ 比率在 100-200 之间。
#
# <div align="center">
#   <img src="../images/aspect-ratio.png" width="75%" />
#   <br>
#   模型宽度/深度
# </div>

# %% [markdown]
# #### 6.4 词表大小 (Vocabulary Size)
#
# 词表的大小主要取决于模型支持语言的丰富程度：
#
# * **单语言模型**：通常在 30k 到 50k 之间（如 GPT-2/3 的 50,257）。
# * **多语言/生产级系统**：为了覆盖更多语种和符号，词表通常扩大到 100k 到 250k（如 PaLM 的 256,000，GPT-4 的 100,276）。

# %% [markdown]
#
# ### 7. 正则化
#
# #### 7.1 Dropout 还需要吗？
# 在传统的深度学习中，Dropout 是防过拟合的神器。但在 LLM 预训练时代：
#
# * **现状**：大多数现代模型（PaLM, LLaMA, T5 v1.1）在预训练阶段**完全移除了 Dropout**（设为 0）。
# * **原因**：
#     1.  数据量极大（万亿级 token），模型很难死记硬背所有数据，过拟合不是主要矛盾。
#     2.  Dropout 会浪费计算资源。
#
# #### 7.2 权重衰减 (Weight Decay)
# * **现状**：依然保留（通常 $\lambda=0.1$）。
# * **原因**：不是为了防过拟合，而是为了优化动力学（Optimization Dynamics）。LLM在经过RMSNorm会具有“尺度不变性”，即将所有的权重乘以k之后，层输出不变，Loss不变。这意味着可能出现weight很大，但是梯度很小，更新变得微不足道的问题，weight decay能够有效的解决这个问题。研究表明，Weight Decay 会与学习率调度（Cosine Decay）相互作用，有助于模型收敛到更好的解。
#
# <div align="center">
#   <img src="../images/weight-decay.png" width="75%" />
#   <br>
#   权重衰减:右图展示了在恒定学习率（Constant LR）设置下，不同权重衰减系数 $\lambda_{WD}$ 对模型训练损失的影响 。实线轨迹显示，在保持较大的学习率时，增强权重衰减（如从 0.0 增加至 0.3）会导致训练损失上升。这一现象的数学本质在于权重衰减作为一种向原点拉回的约束力，在当前学习率下阻止了参数向损失函数局部极小值点的深度逼近。然而，每隔 10,000 次迭代切换至极低学习率（Tiny LR）的虚线分支显示，具有较强权重衰减的模型表现出更优的潜在收敛深度。这表明权重衰减的主要功能并非正则化以防止过拟合，而是通过约束权重范数来调节优化动态，使得参数分布在后续学习率衰减阶段能够更有效地进入高质量的局部解空间。
# </div>
#

# %% [markdown]
# ### 8. 训练稳定性技巧 (Stability Tricks)
#
# 在训练大规模语言模型（LLM）时，模型的稳定性是一个核心问题。最近的研究（如 OLMo 2）非常关注如何避免训练过程中的梯度爆炸和损失发散。
#
# 以下是目前主流大模型中常用的一些提升训练稳定性的技巧：
#
# #### 8.1 监控梯度的 L2 范数 (Gradient L2 Norm)
# 首先，如何判断训练是否稳定？一个重要的指标是**梯度的 L2 范数**。
# * **现象**：在不稳定的训练中（如 OLMo 0424 7B），梯度的 L2 范数会出现剧烈的震荡和“尖峰”（Spikes）。
# * **目标**：理想的训练曲线（如 OLMo 2 1124 7B）应该是非常平滑的，梯度范数保持在较低且稳定的水平。
# * **建议**：如果你看到梯度范数像心电图一样剧烈跳动，这通常是模型架构或超参数设置有问题的信号。
#
# #### 8.2 问题的根源：Softmax
# 大模型中绝大多数的数值不稳定问题都源于 **Softmax** 操作。
# * Softmax 涉及指数运算 ($e^x$)，这在输入值较大时极易导致数值溢出，或者在分母计算时出现除以零的风险。
# * Transformer 中有两个关键位置使用了 Softmax：
#     1.  **注意力机制 (Attention)** 内部。
#     2.  **输出层 (Output Probabilities)** 计算下一个 token 的概率时。
#
# 针对这两个位置，业界发展出了不同的稳定性技巧：
#
# #### 8.3 输出层稳定性：Z-Loss
# 为了解决输出层 Softmax 的不稳定性，PaLM 模型率先引入了 **Z-loss** 技巧。
#
# * **原理**：
#     回顾 Softmax 的计算公式：
#     $$\log(P(x)) = \log\left(\frac{e^{U_r(x)}}{Z(x)}\right) = U_r(x) - \log(Z(x))$$
#     其中，$Z(x) = \sum e^{U_{r'}(x)}$ 是归一化项（Partition function）。
#     
#     为了增加稳定性，我们希望这个归一化项 $\log(Z(x))$ 保持在 0 附近，不要漂移得太远。
#
# * **实现**：
#     在总损失函数中增加一项辅助损失（Auxiliary Loss）：
#     $$L_{total} = L_{original} + 10^{-4} \cdot \log^2 Z$$
#     
# * **效果**：鼓励 Softmax 的归一化因子 $Z$ 接近 1（即 $\log Z$ 接近 0）。这被证明能显著提高训练稳定性。
# * **采用的模型**：PaLM, Baichuan 2, DCLM, OLMo 2 等。
#
# #### 8.4 注意力层稳定性：QK Norm (Query-Key Normalization)
# 在 Attention 计算中，如果 Query ($Q$) 和 Key ($K$) 的向量模长很大，它们的点积 $QK^T$ 就会非常大，导致 Softmax 进入饱和区，梯度消失或不稳定。
#
# * **思路**：也就是那个著名的梗——"Stack More LayerNorms"。既然 $Q$ 和 $K$ 太大，那就在算点积之前先把它们归一化。
# * **实现**：
#     在 $Q$ 和 $K$ 进入点积操作之前，分别对它们应用 LayerNorm（或 RMSNorm）：
#     $$Attention(Q, K, V) = Softmax\left(\frac{LN(Q)LN(K)^T}{\sqrt{d}}\right)V$$
# * **来源**：这个技巧最初源自视觉 Transformer (ViT) 和多模态模型，现在被广泛用于 LLM。
# * **采用的模型**：DCLM, OLMo 2, Gemma 2。
#
# #### 8.5 Logit Soft-capping (Logit 软截断)
# 这是另一种防止 Logit 值过大的暴力且有效的方法，Gemma 2 等模型使用了此技巧。
#
# * **原理**：通过 `tanh` 函数将 Logit 值强制压缩在一定范围内，然后再缩放回去。这样既保留了梯度，又限制了最大值。
# * **公式**：
#     $$logits \leftarrow soft\_cap * \tanh\left(\frac{logits}{soft\_cap}\right)$$
# * **参数设置**：
#     通常对 Self-attention 层和 Final layer 分别设置不同的截断阈值。例如：
#     * Self-attention layers: `soft_cap = 50.0`
#     * Final layer: `soft_cap = 30.0`
# * **效果**：防止 Logits 爆炸。虽然理论上可能会轻微影响性能，但实验数据显示（如 perplexity 对比），它在保持稳定性的同时，性能与 Baseline 持平甚至略好，且没有明显的副作用。

# %% [markdown]
# ## 9. Attention Heads 变体与推理加速
#
# 大多数大模型在 Attention Heads（注意力头）的设计上并没有太大的改动，基本沿用了标准架构。但在最近，为了解决推理成本和计算效率问题，出现了一些重要的变体。
#
# 主要包括两个方向：
# 1.  **GQA / MQA**：通过减少 Head 的数量来降低推理（Inference）时的显存占用和带宽压力。
# 2.  **Sparse / Sliding Window Attention**：通过限制 Attention 的范围（如 GPT-4 或 Mistral）来减少计算成本。
#
# 本节主要聚焦于 GQA 和 MQA 这类通过减少 Head 来加速推理的技术。
#
# ### 9.1 推理时的算力与内存瓶颈
#
# 要理解为什么要改动 Attention Heads，首先要理解 Attention 计算的瓶颈所在。
#
# **训练阶段（Training / Prefill）：计算密集型**
# 在训练或推理的 Prefill 阶段，我们一次性处理所有 token。
# * 我们需要计算 3 组所有点对点的 Attention分数：$QK^T$。
# * 总的算术运算量是 $O(bnd^2)$，总的内存访问量是 $O(bnd + bhn^2 + d^2)$。
# * 此时的算数强度很高，大约是 $O((1/k + 1/bn)^{-1})$。这意味着计算量相对于内存访问量很大，GPU 的计算单元能跑满，效率很高。
#
# **推理生成阶段（Generation）：内存密集型**
# 但在生成文本时，情况完全不同：
# 1.  **无法并行**：生成过程是逐步（Step-by-step）的，每生成一个 token 依赖于前一个。
# 2.  **KV Cache**：为了不重复计算，我们会把之前所有 token 的 Key 和 Value 缓存起来（即 KV Cache）。
# 3.  **增量更新**：每一步只需要计算当前新 token 与 缓存中所有 KV 的交互。
#
# **问题的核心：算数强度骤降**
# 在增量解码阶段，算数强度变成了：
# $$O\left(\left(\frac{n}{d} + \frac{1}{b}\right)^{-1}\right)$$
# * $n$ 是序列长度，$d$ 是模型维度，$b$ 是 batch size。
# * 由于 $n/d$ 这一项很难减小，导致**算数强度非常低**。
# * 这意味着 GPU 大部分时间都在**等待从显存中读取 KV Cache 数据**，而不是在进行计算。这被称为 "Memory Bound"（内存受限）。
#
# ### 9.2 MQA (Multi-Query Attention)
#
# 为了解决上述内存带宽瓶颈，**MQA** 被提了出来。
#
# * **核心思想**：虽然我们需要多个 Query Heads 来保持模型的表达能力，但真的需要给每个 Query 配对独立的 Key 和 Value Heads 吗？MQA 的答案是：不需要。
# * **做法**：
#     * 保持多头 Query（Q）。
#     * **所有** Query Heads 共享**同一组** Key（K）和 Value（V）。
#     * 即 $K$ 和 $V$ 的维度被压缩到了原来的 $1/h$（$h$ 为头数）。
# * **优势**：
#     * 极大地减少了 KV Cache 的大小。
#     * 也就是减少了需要搬运进出显存的数据量。
#     * 算术强度显著提升，不再被显存带宽卡脖子。
# * **劣势**：根据 Shazeer (2019) 的研究，MQA 会导致 PPL (困惑度) 出现轻微的上升，即模型效果略有下降。
# <div align="center">
#   <img src="../images/weight-decay.png" width="75%" />
#   <br>
#   MQA
# </div>
#
#
# ### 9.3 GQA (Grouped-Query Attention)
#
# MQA 有点过于激进（从 $h$ 个 KV 砍到只剩 1 个），**GQA** 则是 MHA（标准多头）和 MQA 之间的一个折中方案。
#
# * **核心思想**：不把所有 Query 强行绑定到一个 KV 上，而是把 Query 分组。
# * **做法**：
#     * 将 Query Heads 分成若干个组（Group）。
#     * **组内共享**同一组 Key 和 Value。
#     * 例如：如果有 8 个 Query Heads，分成 4 组，那么只需要 4 个 KV Heads，而不是 MHA 的 8 个，也不是 MQA 的 1 个。
# * **优势**：
#     * 它提供了一个可调节的旋钮（Knob），让你在**模型效果（Expressiveness）**和**推理效率（Efficiency）**之间找平衡。
#     * 当 Group 数等于 Head 数时，它就是 MHA。
#     * 当 Group 数等于 1 时，它就是 MQA。
#
# <div align="center">
#   <img src="../images/weight-decay.png" width="75%" />
#   <br>
#   GQA
# </div>
#
#
#
#
# ### 9.4 性能对比与结论
#
# 实验数据（如 Llama 2/3 的采用）表明 GQA 是目前的最佳实践：
#
# 1.  **效果损失极小**：GQA 的 PPL 几乎与标准 MHA 持平（Ainslie et al., 2023），表现优于 MQA。
# 2.  **推理速度极大提升**：虽然只减少了一部分 KV Head，但在推理延迟（Time per sample）上，GQA 带来的加速非常明显，非常接近 MQA 的速度，远快于 MHA。
# 3.  **结论**：在现代大模型（特别是长上下文模型）中，使用 GQA 来压缩 KV Cache 已经成为标准操作。
#
#

# %% [markdown]
# ## 10. 稀疏与滑动窗口注意力 （Sparse / Sliding Window Attention）
#
# 随着模型上下文长度（Context Length）的不断增加，标准的全局注意力机制（Global Attention）面临着巨大的计算挑战。这一章节主要探讨了如何通过限制注意力的范围来降低计算成本，同时尽可能保持模型的长文本处理能力。
#
# ### 10.1 动机与早期尝试：稀疏注意力 (Sparse Attention)
#
# 我们知道，标准的 Transformer 注意力机制是**二次方复杂度 ($O(N^2)$)** 的。这意味着，如果上下文长度翻倍，计算量和显存占用会变成原来的四倍。当序列非常长时，这不仅昂贵，甚至在物理上是不可能的。
#
# 为了解决这个问题，早期的研究（如 OpenAI 的 Sparse Transformer, 2019）提出了**稀疏注意力**的概念。
# * **核心思想**：不再让每个 token 都去“看”之前所有的 token，而是设计一种稀疏的、结构化的注意力模式。
# * **权衡**：这本质上是在模型的表达能力（Expressiveness）和运行效率（Runtime）之间做权衡。
# * **常见模式**：
#     * **Strided（步幅式）**：每个 token 只关注固定间隔的几个 token。
#     * **Fixed（固定式）**：强制某些 token 只关注特定的区域。
#
# ### 10.2 滑动窗口注意力 (Sliding Window Attention, SWA)
#
# 在稀疏注意力的基础上，目前更流行且实用的一种变体是**滑动窗口注意力**（Mistral 等模型广泛采用）。
#
# * **工作机制**：
#     每个 token 不再关注整个历史上下文，而只关注它前面固定窗口大小（Window Size）内的 token（例如前 4096 个 token）。
#     * 这就把注意力矩阵从原本密密麻麻的下三角矩阵，变成了一条对角线附近的带状矩阵。
#     * 计算复杂度从 $O(N^2)$ 降低到了 $O(N \times W)$（线性复杂度），这对于处理超长文本至关重要。
#
# * **有效上下文长度 (Effective Context Length)**：
#     你可能会问，只看前 4096 个词，那更前面的信息岂不是丢了？
#     其实不然。虽然单层只能看一个窗口，但随着层数（Depth）的堆叠，信息的传递范围会像卷积神经网络的“感受野”一样不断扩大。
#     * 第 1 层的 token 关注了它之前的窗口。
#     * 第 2 层的 token 关注了第 1 层的输出（而第 1 层已经聚合了它之前的信息）。
#     * **结论**：通过堆叠网络深度，即便使用局部的滑动窗口，模型理论上也能拥有远超窗口大小的“有效上下文长度”。
#
# ### 10.3 当前的主流方案：交替注意力 (Interleaving Full and 'LR' Attention)
#
# 虽然滑动窗口能通过堆叠层数扩大感受野，但直接看全图（Full Attention）肯定还是表达力更强的的。为了兼顾效率和长距离依赖，现在的模型（如 Cohere Command A, Llama 4, Gemma 等）采用了一种混合策略。
#
# * **设计模式**：
#     不再是所有层都用滑动窗口，而是将全注意力层（Full Attention）和滑动窗口层（SWA）交替穿插。
#     * **例子**：在 Cohere Command A 中，每 **4** 层才插入一个 Full Attention 层，其余 3 层都是 Sliding Window Attention。
#     * Llama 4 和 Gemma 也有类似的设计（SWA + Full RoPE）。
#
# * **分工合作**：
#     这种架构让模型有了明确的分工：
#     * **短距离信息（Short-range info）**：主要靠滑动窗口层 + RoPE（旋转位置编码）来捕捉，处理局部的语法和逻辑，计算快、效率高。
#     * **长距离信息（Long-range info）**：主要靠那几个稀有的全注意力层来“一眼望到底”，或者配合 NoPE（无位置编码）等技术来捕捉全局关联。
#
# 这种“大部分时间省着用，关键时刻全开”的策略，是目前在受限算力下实现超长上下文处理的标准技巧（Standard Trick）。

# %% [markdown]
# ## 11. 线性注意力（Linear Attention）
#
# 虽然第 10 章提到的滑动窗口注意力（SWA）能有效降低计算量，但它本质上是以牺牲全局视野为代价的。为了在保持全局感受野的同时实现线性复杂度，研究者们提出了线性注意力（Linear Attention）机制。
#
# ### 11.1 核心原理：矩阵乘法的结合律
#
# 线性注意力的核心思想是将标准 Softmax 注意力中的非线性操作（Softmax）通过核函数（Kernel Function）近似，从而利用矩阵乘法的结合律来改变计算顺序。
#
# * **标准 Softmax 注意力**：
#     $$Attention(Q, K, V) = Softmax(QK^T)V$$
#     其计算顺序是先算 $QK^T$（得到 $N \times N$ 的矩阵），复杂度为 $O(N^2)$。
# * **线性注意力近似**：
#     利用核函数 $\phi(\cdot)$ 将相似度表示为特征映射的点积：$Sim(Q_i, K_j) = \phi(Q_i)^T \phi(K_j)$。
#     $$Attention(Q, K, V) = (\phi(Q)\phi(K)^T)V = \phi(Q)(\phi(K)^T V)$$
#     **改变顺序后**：先计算 $\phi(K)^T V$（得到 $d \times d$ 的矩阵），再左乘 $\phi(Q)$。计算复杂度从 $O(N^2)$ 降低到了 $O(N)$。
#
# ### 11.2 优势：从 Transformer 到 RNN 的桥梁
#
# 线性注意力不仅仅是计算速度的提升，它在数学形式上揭示了 Transformer 与循环神经网络（RNN）的深层联系：
#
# * **固定大小的状态（Fixed-size State）**：在推理生成阶段，线性注意力不需要存储随序列增长的 KV Cache，而是维护一个固定大小为 $d \times d$ 的状态矩阵$\phi(K)^T V$（Accumulated State）。
# * **推理速度极快**：在自回归解码时，线性注意力的显存占用是常数级的，不会随长度增加而爆炸，这使得它在处理超长序列时比标准 Transformer 快出几个数量级。
#
# ### 11.3 挑战与改进：性能与局限
#
# 尽管线性注意力在效率上具有绝对优势，但在实际应用中仍面临一些挑战：
#
# * **表达能力损失（Semantic Confusion）**：由于线性注意力不再具有 Softmax 的非线性特性，它在处理精确位置信息和局部模式建模上通常不如标准 Transformer。
# * **注入性问题（Injectivity）**：研究发现线性注意力可能出现“语义模糊”，即不同的 Query 可能会得到非常相似的输出，限制了模型的精细表征能力。
# * **混合架构趋势**：为了平衡性能与效率，现代研究（如 **Qwen-Next**, **HGRN-2**）倾向于使用混合架构，将线性注意力层与全注意力层以一定比例（如 3:1 或 6:1）进行组合，从而在降低成本的同时维持 Transformer 级别的性能表现。

# %% [markdown]
# ### 总结：2025 年的 LLM 架构画像
#
# 如果你今天要训练一个新的大模型，基于 Lecture 3 的内容，你的“默认配置”大概率是这样的：
#
# * **架构**：Decoder-only Transformer。
# * **归一化**：**RMSNorm** + **Pre-Norm**。
# * **激活函数**：**SwiGLU**（$d_{ff} \approx \frac{8}{3} d_{model}$）。
# * **位置编码**：**RoPE**。
# * **注意力**：**GQA**（推理加速）+ **Flash Attention**（训练加速）。
# * **偏置项**：**No Bias**（全连接层和 Norm 层都不加 bias）。
# * **正则化**：无 Dropout，只有 Weight Decay。
# * **稳定性**：可能加入 QK Norm 或 Logit Soft-capping。
