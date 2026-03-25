# %% [markdown]
# # GPUs
#
# [Thonk From First Principles](https://www.thonking.ai/)
#
# [GPU Mode](https://github.com/gpu-mode)
#
# ## 1. GPU 计算背景与硬件架构深度解析
#
# ### 1.1 为什么我们需要学习 GPU？(Context & Motivation)
# 在大模型时代，算力（Compute）与模型性能之间存在着极强的可预测关系。
# * **Scaling Laws（扩展定律）**：根据 Kaplan 等人的研究，随着算力投入的增加，模型的 Validation Loss 会呈现幂律下降（$L \propto C^{-0.048}$）。这意味着，只要堆砌更多的算力，模型效果几乎必然提升。
# * **硬件驱动进步**：更快的硬件、更高的利用率以及改进的并行化策略，是当前驱动 AI 进步的核心动力。没有 GPU 的扩展，就没有 LLM 的扩展。
# * **课程目标**：本章旨在打破 CUDA 和 GPU 的“黑盒”，理解如何编写快速算法，并深入剖析 GPU 在什么情况下会变慢（例如受到 Compute Intensity 的限制）。
#
# <div align="center">
#   <img src="../images/gpu-scaling.png" width="75%" />
#   <br>
#   Scaling law
# </div>

# %% [markdown]
# ### 1.2 硬件性能的演进：从摩尔定律到专用加速 (Hardware Evolution)
# 随着传统的 Dennard Scaling 在 2005 年左右失效（单核频率无法继续提升），处理器发展进入了多核与专用加速时代。过去 10 年间，单芯片推理性能提升了 **1000倍**，这主要得益于以下几个维度的技术压榨：
# 1.  **数值表示 (Number Representation)**：从 FP32 到 FP16、Int8，再到 TF32、BF16 甚至 FP8（Transformer Engine），带来了约 **16倍** 的提升。
# 2.  **复杂指令集 (Complex Instructions)**：引入 DP4、HMMA、IMMA 等专用指令，特别是 Tensor Cores 的出现，带来了约 **12.5倍** 的提升。
# 3.  **制程工艺 (Process)**：从 28nm 进化到 5nm，带来了约 **2.5倍** 的提升。
# 4.  **稀疏性 (Sparsity)**：利用结构化稀疏，带来了约 **2倍** 的提升。

# %% [markdown]
# ### 1.3 CPU vs GPU：设计哲学的本质差异
# CPU 和 GPU 在底层设计目标上截然不同：
# * **CPU (Low Latency Processor)**：
#     * **设计目标**：优化**延迟**。让单个线程跑得越快越好。
#     * **结构特点**：拥有巨大的缓存（Cache）和复杂的控制逻辑（Control），用于处理分支预测等复杂任务。ALU（算术逻辑单元）占比相对较小。
#     * **适用场景**：串行任务、复杂的逻辑判断。
# * **GPU (High Throughput Processor)**：
#     * **设计目标**：优化**吞吐量**。在单位时间内处理尽可能多的数据，不在乎单个线程的处理速度，而在乎整体的完成度。
#     * **结构特点**：
#         * **海量 ALU**：芯片大部分面积被计算单元占据。
#         * **简单的控制逻辑**：削减了分支预测和缓存的面积，不擅长复杂的逻辑跳转。
#         * **通过并行掩盖延迟**：当一组线程等待内存数据时，GPU 会立刻切换到另一组线程进行计算，从而掩盖内存读取的延迟。
#
# <div align="center">
# <img src="../images/gpu-cpu.png" width="75%">
# <br>
# GPU CPU 对比
# </div>

# %% [markdown]
# ### 1.4 GPU 架构解剖 (Anatomy of a GPU)
# 深入到 GPU 内部（以 NVIDIA A100/H100 为例），其核心组件包括：
# * **SM (Streaming Multiprocessor，流式多处理器)**：GPU 的核心计算模块。一个 GPU 包含很多个 SM（例如 GA100 有 128 个 SM）。
#     * 每个 SM 内部包含大量的 CUDA Cores（FP32/FP64/INT32）和 **Tensor Cores**。
#     * 拥有**寄存器堆 (Register File)**：速度最快，但极其昂贵。
#     * **L1 Cache / Shared Memory**：位于 SM 内部，速度非常快，是 GPU 编程中优化的关键（手动管理的缓存）。
#     * L2 cache位于芯片内部，而全局显存（global memory）则是位于 GPU 旁边的存储芯片
# * **HBM (High Bandwidth Memory)**：显存（Global Memory），位于 GPU 芯片旁边，容量大（如 40GB/80GB）但速度相对较慢。
# * **Tensor Cores**：专门为矩阵乘法设计的电路。在现代 GPU 上，矩阵乘法（Matmul）的运算速度比普通浮点运算快 **10倍以上**。
#
# <div align="center">
#   <img src="../images/gpu-sm.png" width="75%" />
#   <br>
#   流式多处理器
# </div>
#
# <div align="center">
#   <img src="../images/gpu-cache.png" width="75%" />
#   <br>
#   GPU中的存储设备
# </div>

# %% [markdown]
#  ### 1.5 GPU 执行模型 (Execution Model)
#
# 在 GPU 的执行模型中，有三个最重要的角色：
#
# 1.  **Threads (线程)**：
#     * **职责**：实际“干活”的单元。
#     * **特点**：并行执行。所有线程执行相同的指令，但处理不同的输入数据（即 **SIMT**, Single Instruction Multiple Threads）。
#     * **轻量级**：线程非常轻量，可以快速启动和停止（Context Switch 开销极低）。
#
# 2.  **Blocks (线程块)**：
#     * **定义**：一组线程的集合。
#     * **调度**：每个 Block 会被分配给一个 **SM (Streaming Multiprocessor)** 执行。
#     * **资源**：Block 内部拥有自己的 **Shared Memory (共享内存)**，块内的线程可以通过它进行通信。
#
# 3.  **Warp (线程束)**：
#     * **定义**：这是 GPU 硬件层面真正的**执行单位**。线程总是以 **32 个** 为一组（即一个 Warp）同时执行的。
#     * **调度机制**：
#         * 每个 SM 有 4 个 Warp 调度器（Warp Schedulers）。
#         * **SIMT 机制**：Warp 内的 32 个线程是“步调一致”的，它们在同一周期执行同一条指令。
#         * **隐藏延迟**：如果当前 Warp 需要等待数据（比如读取内存），调度器会瞬间切换到另一个就绪的 Warp 继续计算。这就是 GPU 为什么能通过高吞吐量来掩盖高延迟的核心原因。
#
# *注：虽然逻辑上 Block 组成了 Grid（整个 Kernel），但从执行效率和硬件调度的角度看，Warp 才是最关键的微观单位。*

# %% [markdown]
# ### 1.6 存储层级与“内存墙” (Memory Hierarchy & Memory Wall)
# 理解存储层级是编写高性能 GPU 代码的关键：
# * **速度层级**：
#     * **寄存器 (Registers)**：最快，私有（Per-thread）。
#     * **共享内存 (Shared Memory/L1)**：非常快 (~19 TB/s)，位于 SM 内部，所有线程块内的线程可见。
#     * **L2 Cache**：片上缓存，介于 SM 和 HBM 之间。
#     * **全局内存 (Global Memory/HBM)**：较慢 (~1.5 TB/s)，容量最大，所有 Block 可见。
#     * **系统内存 (CPU DRAM)**：最慢，通过 PCIe 连接。
#
# * **算力与带宽的剪刀差**：
#     * 在过去的 20 年里，硬件算力（FLOPS）增长了 **60,000倍**。
#     * 显存带宽（DRAM Bandwidth）仅增长了 **100倍**。
#     * **结论**：算力增长远快于带宽增长。这导致了现代 AI 负载往往是 **Memory Bound（受限于内存带宽）** 的，即计算单元经常在空转等待数据。因此，优化的核心往往在于**减少数据搬运**，而非减少计算量。

# %% [markdown]
# ### 1.7 TPU 
#
# 在讨论完 GPU 的架构后，我们很自然地会问：Google 的 TPU (Tensor Processing Unit) 和 GPU 有什么区别？
# 从宏观层面（High Level）来看，**GPU、TPU 以及大多数 AI 加速器的本质是非常相似的**，它们都是为了大规模并行计算而设计的。
#
# #### 1.7.1 TPU 的内部构造
# TPU 的设计更加专一，其内部组件可以与 GPU 进行类比：
#
# * **Scalar Unit (标量单元)**：
#     * **功能**：有点像控制中心或微型 CPU。
#     * **职责**：它负责“分发”指令给 VPU 和 MXU，自己处理一些简单的控制逻辑。
# * **Vector Unit (VPU, 向量单元)**：
#     * **功能**：处理非矩阵乘法的运算。
#     * **职责**：负责执行 **Elementwise Operations（逐元素操作）**，比如激活函数（ReLU, GELU）、加减法等。同时，它也负责将数据从 HBM 加载进 MXU。
# * **Matrix Multiply Unit (MXU, 矩阵乘法单元)**：
#     * **功能**：TPU 的心脏，相当于 GPU 的 Tensor Core，但通常更大。
#     * **职责**：专门执行矩阵乘法。它是芯片算力（FLOP/s）的主要来源。
# * **HBM (High Bandwidth Memory)**：
#     * 与 GPU 一样，用于存储权重、激活值、优化器状态和 Batch 数据。带宽决定了数据能多快地喂给计算单元。
#
# #### 1.7.2 TPU vs. GPU：核心差异
# 虽然组件相似，但在架构取舍上有所不同：
#
# 1.  **核心数量与大小**：
#     * **GPU**：拥有**很多**的 SM（流多处理器）。
#     * **TPU**：拥有**较少**的大型核心（TensorCores），但每个核心的矩阵乘法能力极强。
# 2.  **执行模型的简化 (No Warps)**：
#     * 这是 TPU 与 GPU 最显著的区别之一。
#     * **GPU**：依赖 **Warps (32线程)** 作为调度单位，通过快速切换 Warp 来隐藏延迟。
#     * **TPU**：**没有 Warps** 的概念，它只有 Blocks（任务块）。
#     * **影响**：这种设计简化了硬件控制逻辑，使得 TPU 在处理纯粹的矩阵乘法时效率极高，但在处理非矩阵乘法（Non-matmul）或需要复杂调度的任务时，其 Trade-off（权衡）与 GPU 不同。
#
# **总结**：TPU 就像是一个为了矩阵乘法而生的“偏科生”，它砍掉了 GPU 中为了图形渲染或通用计算保留的复杂控制（如 Warp 调度），换来了更极致的矩阵计算密度。

# %% [markdown]
# ## 2. 如何让 ML 负载在 GPU 上高速运行？
# 在理解了 GPU 的硬件架构后，我们面临的终极问题是：如何让机器学习任务跑得更快？核心挑战在于**显存带宽的瓶颈**。本章通过 **Roofline Model** 引出核心矛盾，并详细拆解了 6 大优化技巧。
#
# ### 2.1 核心指导思想：Roofline Model (屋顶线模型)
# 要优化性能，首先得知道瓶颈在哪。
# * **计算强度 (Operational Intensity, Ops/Byte)**：每从内存搬运 1 Byte 数据，能进行多少次浮点运算（FLOPs）。
# * **Roofline Model 图解**：
#     * **横轴**：计算强度。
#     * **纵轴**：吞吐量（GFLOPS）。
#     * **内存受限区 (Memory Bound)**：图左侧的斜线区域。此时性能受限于**带宽**。无论你的 GPU 算力多强，数据供不上，ALU 就在空转。
#     * **计算受限区 (Compute Bound)**：图右侧的水平线区域。此时性能受限于**峰值算力**。
# * **现状**：现代 ML 模型（尤其是 LLM）的大多数算子（如 Element-wise ops, Reduction）通常落在**内存受限区**。
# * **目标**：**避免内存瓶颈（Memory Bound）**，想办法向右移动（提高计算强度），或者向上突破（利用更快的存储层级）。
#
#
# <div align="center">
#   <img src="../images/gpu-roofline.png" width="50%" />
#   <br>
#   Roofline 模型
# </div>
#
#
# ### 2.2 技巧一：控制流分歧 (Control Divergence)
# 虽然这不是内存问题，但它严重影响执行效率。
# * **原理**：GPU 采用 **SIMT**（单指令多线程）模型。一个 Warp 内的 32 个线程必须在同一时刻执行同一条指令。
# * **问题**：如果你写了 `if-else` 分支：
#     * 线程 0-15 走 `if` 分支（执行 A, B）。
#     * 线程 16-31 走 `else` 分支（执行 X, Y）。
#     * **后果**：硬件无法同时执行 A 和 X。它只能先让所有线程走 `if` 路径（此时后半部分线程被**禁用/Masked out**），再让所有线程走 `else` 路径（前半部分线程被禁用）。
# * **代价**：执行时间变成了两个分支之和，硬件利用率直接腰斩。
# * **结论**：在 GPU Kernel 代码中，尽量避免写这种会导致线程走向不同路径的条件判断。
#
# ### 2.3 技巧二：低精度计算 (Low Precision)
# 简单粗暴但极其有效：**比特数越少，搬运越快，算得越快。**
# * **演进路线**：FP32 (4 Bytes) -> FP16/BF16 (2 Bytes) -> INT8 (1 Byte) -> FP4 (0.5 Byte)。
# * **双重收益**：
#     1.  **带宽收益**：FP16 比 FP32 数据量减半，意味着同样的带宽能搬运 2 倍的数据，直接提升了**计算强度 (Arithmetic Intensity)**。
#     2.  **算力收益**：Tensor Cores 在低精度下有专门的硬件加速电路，吞吐量成倍增加（例如 A100 的 INT8 算力是 FP32 的数倍）。
# * **应用场景**：
#     * 矩阵乘法：非常适合低精度。
#     * Pointwise ops (ReLU, Add 等)：大部分可以。
#     * **需要高精度的地方**：累加器（Accumulator，防止溢出）、Reductions（Sum, Softmax）、以及指数/对数运算（Exp, Log）通常需要保持 FP32 以维持数值稳定性。
#
# ### 2.4 技巧三：算子融合 (Operator Fusion)
# 这是 PyTorch 2.0 (`torch.compile`) 的核心魔法。
# * **工厂与仓库的比喻**：
#     * **GPU Compute (工厂)**：加工速度极快。
#     * **HBM Memory (仓库)**：距离工厂很远，运输慢。
# * **未融合 (Naive) 的问题**：
#     * 计算 `sin(x) + cos(x)`。
#     * 第一步：从 HBM 读 `x` -> 算 `sin` -> 写回 HBM (`tmp1`)。
#     * 第二步：从 HBM 读 `x` -> 算 `cos` -> 写回 HBM (`tmp2`)。
#     * 第三步：从 HBM 读 `tmp1`, `tmp2` -> 算 `add` -> 写回 HBM。
#     * **痛点**：大量时间浪费在把中间结果写回显存又读出来。
# * **融合后 (Fused Kernel)**：
#     * 从 HBM 读 `x` -> 在寄存器/L1 中一口气算完 `sin + cos + add` -> 写回 HBM。
#     * **收益**：内存读写次数大幅减少，带宽压力骤降。这对于 Element-wise 操作（如 LayerNorm, Softmax, GELU）的提速是决定性的。
#
# ### 2.5 技巧四：重计算 (Recomputation / Activation Checkpointing)
# 这是一个“以时间换空间”的策略，用来解决显存不足（OOM）并变相提升带宽效率。
# * **背景**：训练反向传播需要用到前向传播的激活值（Activations）。通常做法是把所有中间层的激活值存到 HBM 里（吃显存，且写回 HBM 慢）。
# * **策略**：
#     * **前向 (Forward)**：不算完所有激活值并存储，只存少量的“检查点 (Checkpoints)”。
#     * **反向 (Backward)**：当需要用到某层的激活值算梯度时，利用最近的检查点**临时重新计算 (Recompute)** 出来。
# * **反直觉的收益**：虽然计算量增加了（多算了一次前向），但因为减少了惊人的 HBM 读写（Memory Access），在内存受限的场景下，它反而可能**更快**！而且能训练更大的 Batch Size。
#
# ### 2.6 技巧五：内存合并 (Memory Coalescing)
# 这是写 CUDA Kernel 必须遵守的“交通规则”。
# * **硬件特性**：DRAM 的读写是**突发模式 (Burst Mode)** 的。你请求 1 个字节，它实际上会给你送来连续的一大块数据（比如 128 Bytes 的 Burst Section）。
# * **合并访问 (Coalesced)**：
#     * 如果在同一个 Warp 里，线程 0 访问地址 `X`，线程 1 访问 `X+1`，线程 2 访问 `X+2`...
#     * 硬件只需发起 **1 次** DRAM 请求，就能把这一大块数据全拿回来分给所有线程。效率 Max。
# * **未合并访问 (Uncoalesced)**：
#     * 如果线程 0 访问 `X`，线程 1 访问 `X+100`，线程 2 访问 `X+200`... (典型的**列优先**访问行存储矩阵的场景)。
#     * 硬件被迫发起 **32 次** 独立的 DRAM 请求。带宽利用率可能低至 1/32。
# * **教训**：在设计矩阵乘法或数据读取时，必须保证线程的访问模式在物理地址上是连续的。
#
# ### 2.7 技巧六：分块 Tiling (The Big One)
# 这是矩阵乘法（Matmul）优化的灵魂，也是 FlashAttention 的基石。
# * **核心思想**：利用高速的 **Shared Memory (SRAM)** 作为缓存。
# * **操作流程**：
#     1.  从慢速 HBM 中搬运一小块（Tile）数据 A 和 B 到快速 SRAM 中。
#     2.  所有线程反复利用 SRAM 里的这块数据进行多次计算（累加到 P）。
#     3.  计算完后，再搬运下一块。
# * **数学收益**：
#     * 朴素矩阵乘法：每个输入元素需要从 HBM 读取 $N$ 次。
#     * Tiling 矩阵乘法：每个输入元素只需从 HBM 读取 $N/T$ 次（$T$ 是块大小）。
#     * **结论**：全局内存访问量减少了 **T 倍**。这直接把任务从 Memory Bound 推向了 Compute Bound。
#
#
# ### 2.8 Tiling 的进阶难题：对齐与波浪量化 (Complexities)
# 理论很美好，工程很残酷。
#
# #### 2.8.1 内存对齐 (Alignment)
# * **问题**：DRAM 的 Burst Section 是固定的（比如 0-127, 128-255）。如果你的 Tile 大小或起始位置没对齐（比如从 1 开始读），一次读取就会跨越两个 Burst Section，导致各读了一半无效数据，浪费带宽。
# * **现象**：矩阵维度是 2 的幂次（如 2048）时通常很快；如果是奇数或素数，性能可能暴跌。
# * **解法**：Padding（填充）矩阵到合适的倍数（如 64 或 128 的倍数）。Karpathy 曾提到将 nanoGPT 的词表大小从 50257 改为 50304（64的倍数），速度直接提升 25%。
#
# #### 2.8.2 波浪量化 (Wave Quantization / Tail Effect)
# * **现象**：随着矩阵大小增加，性能曲线呈现锯齿状（周期性波动）。
# * **原因**：GPU 有很多 SM（比如 A100 有 108 个）。
#     * 假设你的任务被切分为 108 个 Tiles -> 完美，所有 SM 一起跑一轮（1 Wave）就结束。
#     * 假设切分为 109 个 Tiles -> 前 108 个 Tiles 跑完后，剩下的 **1 个 Tile** 只能由 1 个 SM 跑，其他 107 个 SM **围观（Idle）**。整体时间被拖慢了一倍。
# * **启示**：矩阵大小最好能让 Tile 总数整除 SM 数量，避免“尾巴”拖累整体利用率。
#
# ### 2.9 第二部分总结
# 让 GPU 飞驰的秘诀主要不在于算得更快，而在于**搬运得更少**：
# 1.  **Coalescing**：一次搬运一大块连续数据。
# 2.  **Fusion**：搬运一次，在寄存器里做多步计算。
# 3.  **Tiling**：搬运到 SRAM，反复利用。
# 4.  **Quantization**：把数据变小再搬运。
# 5.  **Recomputation**：少存点东西，下次用到再算，别占带宽。

# %% [markdown]
# ## 3. 理解 Flash Attention 
#
# Flash Attention (Dao et al.) 是近年来优化 Transformer 注意力机制最经典的案例。它通过极致的系统级优化，极大地加速了注意力计算。本章将运用我们之前学到的知识（特别是 Tiling 和重计算），来拆解 Flash Attention 到底快在哪里。
#
# ### 3.1 动机：标准 Attention 的内存瓶颈
# 首先来看看为什么标准的 Attention 慢。
# * **性能分析**：在 GPT-2 的注意力层性能分析中，我们发现一个反直觉的现象：**矩阵乘法（Matmul）** 虽然计算量大，但效率其实很高；反而是 **Dropout** 和 **Softmax** 这些计算量很小（Element-wise）的操作占据了大量时间。
# * **原因**：这些操作是典型的 **Memory Bound（受限于内存带宽）**。
#     * 标准实现需要反复读写 HBM（显存）：$Q \cdot K^T$ 算出 $S$ 矩阵（$N \times N$，巨大）写回 HBM $\rightarrow$ 读出来做 Softmax $\rightarrow$ 写回 $P$ 矩阵 $\rightarrow$ 读出来做 Dropout $\rightarrow$ 读出来乘 $V$。
#     * 这种反复搬运 $N \times N$ 级数据的过程直接撑爆了显存带宽。
# * **Flash Attention 的战绩**：
#     * **GFLOPS**：从 66.6 提升到 75.2（算力利用率更高）。
#     * **HBM R/W（显存读写量）**：从 **40.3 GB** 骤降至 **4.4 GB**（减少了近 10 倍！这是核心）。
#     * **Runtime**：从 41.7 ms 加速到 **7.3 ms**。
#
# ### 3.2 Attention 计算流回顾
# 标准的注意力计算包含三个步骤：
# 1.  **Matmul 1**：$S = Q \cdot K^T$。生成巨大的 $N \times N$ 分数矩阵。
# 2.  **Softmax**：$P = \text{Softmax}(S)$。对每一行做归一化，生成概率矩阵。
# 3.  **Matmul 2**：$O = P \cdot V$。生成最终输出。
#
# **痛点**：中间矩阵 $S$ 和 $P$ 的大小是 $N \times N$（序列长度的平方）。如果不做处理，它们必须被完整地写入 HBM 再读出，导致显存占用和带宽消耗随序列长度平方级增长。
#
# ### 3.3 Tiling 策略一：KQV 的分块计算
# 为了解决内存问题，Flash Attention 的第一招是 **Tiling（分块）**。
# 这部分其实就是标准的矩阵乘法 Tiling（参考第 2 章）：
# * 将 $Q, K, V$ 切分成小的 Block。
# * 将 Block 从 HBM 加载到高速的 **SRAM** 中。
# * 在 SRAM 内部计算一小块 $Q \cdot K^T$ 的结果。
# * **挑战**：虽然矩阵乘法可以分块，但中间夹着的 **Softmax** 比较难处理，因为它需要对整行数据求和（Normalization factor）才能算出最终概率，看起来似乎必须拥有整行数据才能开始计算。
#
# ### 3.4 Tiling 策略二：Online Softmax (增量计算)
# 这是 Flash Attention 的数学核心。为了让 Softmax 也能分块流式计算，我们需要用到 **Online Softmax** 技巧。
#
# * **标准 Safe Softmax**：
#     为了防止数值溢出，通常先减去最大值 $m$：
#     $$y_i = \frac{e^{x_i - m}}{\sum e^{x_j - m}}$$
#     这通常需要遍历数据两次（一次找最大值 $m$，一次求和）。
#
# * **Online Softmax (增量更新)**：
#     不仅可以一次遍历，还可以分块遍历。当我们处理一个新的分块时，如果发现了更大的最大值 $m_{new}$，我们可以利用**伸缩和（Telescoping Sum）**的技巧，去修正之前计算好的局部和，而不需要重新读取旧数据。
#     * **算法逻辑**：维护一个运行中的最大值 $m$ 和运行中的分母 $d$。每当读入新块，更新 $m$ 和 $d$，并动态调整当前的输出结果。
#     * **意义**：这使得我们可以**Tile-by-Tile（逐块）**地计算 Softmax，完全不需要把整个 $N \times N$ 矩阵存下来。
#
# ### 3.5 Flash Attention 的完整前向传播 (The Fused Kernel)
# 结合上述技术，Flash Attention 的执行流程（Algorithm 1）如下：
#
# 1.  **外层循环**：加载 $K, V$ 的 Block 到 SRAM。
# 2.  **内层循环**：加载 $Q$ 的 Block 到 SRAM。
# 3.  **SRAM 内计算**：
#     * 计算局部 $S_{tile} = Q_{tile} \cdot K_{tile}^T$。
#     * 利用 Online Softmax 更新当前行的最大值 $m$ 和分母 $d$。
#     * 计算局部输出 $O_{tile} = P_{tile} \cdot V_{tile}$。
#     * **Rescaling（重缩放）**：根据最新的 $m$ 和 $d$，对之前累加的 $O$ 进行修正。
#     * 公式：$O_{new} = \text{diag}(\dots)^{-1} O_{old} + \dots$ （通过保存统计量来修正，而非重读数据）。
# 4.  **写回**：只将最终计算好的 $O$（大小 $N \times d$）写回 HBM。
#
# **核心优势**：
# * **Kernel Fusion（算子融合）**：Matmul、Softmax、Mask、Dropout 全部融合在一个 Kernel 里做完。
# * **不存中间矩阵**：$N \times N$ 的 $S$ 和 $P$ 矩阵从未在 HBM 中真正产生过（只存在于 SRAM 的计算过程中）。
# * **Recomputation（重计算）**：在反向传播时，因为没有存 $S$ 和 $P$，需要重新用前向逻辑算一遍。虽然多算了，但因为省下了巨大的 HBM 读写，整体速度反而**更快**。
#
# ### 3.6 总结
# 回顾整节 GPU 课程，我们贯穿了从硬件到底层算法的逻辑：
#
# 1.  **硬件决定上限**：摩尔定律放缓，但专用加速器（Tensor Cores）和并行规模让算力持续爆炸（60000x 增长）。
# 2.  **内存是瓶颈**：显存带宽的增长（100x）远落后于算力。现代 AI 负载（特别是 LLM）的核心矛盾是 **Compute vs Data Movement**。
# 3.  **优化的终极心法**：
#     * **Coalescing**：保证内存访问连续，利用 DRAM Burst。
#     * **Tiling**：利用 SRAM 做缓存，增加数据的局部性复用。
#     * **Fusion**：将多个操作合并，减少 HBM 往返次数。
#     * **Flash Attention** 就是这些原则的集大成者：通过 Tiling 和重计算，用“多算一点”换取了“少读很多”，从而实现了性能的飞跃。
