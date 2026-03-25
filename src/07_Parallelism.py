#!/usr/bin/env python
# coding: utf-8

# # Parallism
# [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/pdf/2304.11277)
# 
# [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/pdf/2104.04473)
# 
# ## 1. LLM 训练的硬件基础与网络通信 (Basics of Networking)
# 
# ### 1.1 硬件性能的演进与“单卡极限”
# 
# 过去十年间，单芯片的推理性能提升了超过 **1000倍**。这种惊人的增长主要由以下四个因素驱动：
# 
# * **数值表示 (Number Representation):** 从 FP32 到 FP16、Int8，再到 TF32、BF16，提供了约 **16x** 的增益。
# * **复杂指令集 (Complex Instructions):** 专用指令集（如 DP4, HMMA, IMMA / Tensor Cores）提供了约 **12.5x** 的增益。
# * **制程工艺 (Process):** 从 28nm 进化到 5nm，提供了约 **2.5x** 的增益。
# * **稀疏性 (Sparsity):** 结构化稀疏带来了约 **2x** 的增益。
# 
# **然而，单 GPU 扩展存在“两堵墙”：**
# 1.  **显存墙 (Memory):** 模型规模的增长速度远超显存。从 2018 年的 ELMo (94M) 到 2020 年的 GPT-3 (175B) 和 Megatron-Turing NLG (530B)，参数量呈指数级上升，单张 GPU 的显存完全无法容纳这些大模型。
# 2.  **算力墙 (Compute):** 即使能装下，单卡的算力也无法支撑如此巨大的计算量。
# 
# <div align="center">
#   <img src="../images/para-size.png" width="50%" />
#   <br>
#   模型尺寸增长
# </div>

# ### 1.2 多机多卡并行架构 (Multi-GPU, multi-machine parallelism)
# 
# 为了突破单机限制，我们采用多机多卡并行策略。此时，**数据中心 (Datacenter)** 成为了新的计算单元 。这种架构依赖不同层级的互连技术：
# 
# #### 节点内并行 (Intra-node parallelism)
# * **连接方式:** 通过高速互连技术（High-speed interconnects）连接 。
# * **技术标准:** 使用 **NVLink 3.0**，提供高达 **400 GT/s per lane** 的带宽 。
# * **架构:** 典型的结构（如 DGX）中，GPU 之间通过 NVSwitch 实现高带宽互连 。
# 
# #### 节点间并行 (Inter-node parallelism)
# * **连接方式:** 跨机器的高速连接 。
# * **技术标准:**
#     * **HDR InfiniBand:** 带宽约为 **50 GT/s per lane** 。
#     * **PCI Express 4.0:** 带宽约为 **16 GT/s per lane**，主要用于 CPU 与 GPU 或网卡间的连接 。
# * **带宽差异:** 节点间带宽（InfiniBand/PCIe）显著低于节点内带宽（NVLink），因此需要通过策略将通信密集型任务保留在节点内 。
# 
# #### 网络拓扑设计差异 (TPU vs GPU)
# 不同的硬件设计选择了不同的通信层级设计 ：
# * **TPU 网络:** 采用环面网格 (Toroidal mesh) 结构，芯片直接互连 。
# * **GPU 网络:** 采用层级化交换结构。例如 DGX SuperPOD 使用 InfiniBand 的 Spine（骨干）和 Leaf（叶）交换机进行节点间连接，节点内则通过 NVSwitch 全互连，提供巨大的对分带宽 (Massive bisection bandwidth) 。
# 
# <div align="center">
# <img src="../images/para-tpu-gpu.png" width="75%"/>
# <br>
# TPU V.S. GPU
# </div>

# ### 1.3 集合通信原语 (Collective communication)
# 
# 在分布式训练中，GPU 间的数据交换依赖于一组标准化的通信原语 ：
# 
# * **All Reduce:** 所有节点均持有部分数据，操作后所有节点都获得所有数据的归约结果（如求和）。
# * **Reduce:** 所有节点的数据被归约并发送到单一的根节点 (Root) 。
# * **All Gather:** 所有节点收集其他所有节点的数据，最终每个节点都拥有完整的全局数据 。
# * **Broadcast:** 根节点将数据广播给所有其他节点 。
# * **Reduce Scatter:** 对数据进行归约，但结果被分散存储在各个节点上，每个节点只保留一部分结果 。
# 
# <div align="center">
# <img src="../images/para-operation.png" width="75%"/>
# <br>
# Communication
# </div>

# ### 1.4 All Reduce 的实现细节
# 
# All Reduce 是大模型训练中最关键的通信操作。在带宽受限的场景下，最优的实现方式是将 All Reduce 拆解为两个步骤 ：
# 
# 1.  **Reduce-Scatter:** 先对数据进行归约并分散到各卡 。
# 2.  **All-Gather:** 再将分散的结果收集到所有卡 。
# 
# <div align="center">
# <img src="../images/para-reduce.png" width="75%"/>
# <br>
# All Reduce
# </div>
# 
# **图解说明:**
# 假设有 4 张 GPU，每张卡持有数据块 A, B, C, D。
# * **Reduce-Scatter 阶段:** GPU 0 负责汇总 A 块的和，GPU 1 负责 B 块，以此类推。结束后，每张卡只有全局和的一部分 。
# * **All-Gather 阶段:** 各卡广播自己持有的那一部分和。结束后，每张卡都拥有了完整的 [Sum(A), Sum(B), Sum(C), Sum(D)] 。

# ### 1.5 总结
# 
# 进入多机扩展时代，我们对系统的核心诉求可以总结为以下三点 ：
# 1.  **线性内存扩展 (Linear memory scaling):** 最大可训练的模型参数量应随 GPU 数量线性增加。
# 2.  **线性计算扩展 (Linear compute scaling):** 模型的训练速度（FLOPS）应随 GPU 数量线性增加。
# 3.  **简单的集合通信原语 (Simple collective comms primitives):** 依赖标准化的通信操作来支撑复杂的并行策略 。

# In[2]:


import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, dim=1024, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.ReLU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

# 辅助函数：打印模型参数的形状和显存占用
def print_model_stats(model, name="Model"):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"[{name}] Total Parameter Size: {param_size:.2f} MB")

    # 打印前几个参数的形状，观察是否被切分
    print(f"[{name}] First Layer Weight Shape: {list(model.parameters())[0].shape}")


# ## 2. LLM 训练的并行方法
# 
# 本部分主要探讨如何并行化大语言模型（LLM）的训练。并行化主要基于三个核心思想：
# - **数据并行 (Data Parallelism)**
# - **模型并行 (Model Parallelism)** 
# - **激活/序列并行 (Activation/Sequence Parallelism)**。
# 
# ### 2.1 数据并行 (Data Parallelism)
# 
# #### 朴素的数据并行 (Naïve Data Parallelism)
# 对于训练中的并行而言，一个最简单的想法是将mini-batch进一步切分，每张计算卡只处理一部分数据。
# * **原理**：将大小为 $B$ 的 Batch 分割到 $M$ 台机器上，每台机器处理 $B/M$ 个样本。机器之间需要交换梯度以进行同步。
# * **计算扩展性**：良好，每个 GPU 处理的数据量减少了。
# * **通信开销**：每一轮 Batch 需要传输 2 倍参数量的在数据（All-Reduce梯度）。如果 Batch 很大，这个开销是可以接受的。
# 
# > **为什么通信成本是 2 * 参数量？**
# > 假设在一个 $N$ 卡集群上：
# > 1. **Scatter-Reduce 阶段**：每张卡需要向外传输比例为 $\frac{N-1}{N}$ 的梯度数据。
# > 2. **All-Gather 阶段**：每张卡把自己负责计算好的那一部分广播给其他卡，同样需要向外传输比例为 $(N-1) \times \frac{1}{N} = \frac{N-1}{N}$ 的数据量。
# >
# > **总通信量**为 $\frac{2(N-1)}{N} \times \Phi$（$\Phi$ 为参数量）。当 $N \to \infty$ 时，该值趋近于 $2\Phi$。
# 
# 
# **Naive DP 实现机制**：
# 1.  **Forward**: 各 GPU 独立计算，无通信。
# 2.  **Backward**: 各 GPU 独立计算局部梯度。
# 3.  **Sync**: 使用 **All-Reduce（Reduce-Scatter + All-Gather）** 原语对所有局部梯度求平均，确保每张卡获得相同的全局梯度。
# 4.  **Update**: 各 GPU 独立更新参数，保持模型副本一致。
# 
# **But ...**
# 
# 数据并行能够有效的实现计算扩展，但是每张卡仍然需要存储完整的模型状态，在内存上几乎没有扩展。

# #### ZeRO: 解决数据并行的内存问题
# ZeRO (Zero Redundancy Optimizer) 的核心思想是**切分昂贵的状态**，并利用 `Reduce-Scatter` 和 `All-Gather` 的等价性来替代 `All-Reduce`。
# <div align="center">
# <img src="../images/para-zero.png" width="75%"/>
# <br>
# ZERO 3
# </div>
# 
# 1.  **ZeRO Stage 1: 切分优化器状态 (Optimizer State Sharding)**
#     * **原理**：将优化器状态切分并分布到不同的 GPU 上。
# 
# 
#     * **流程**：
#         * Forward/Backward: 各卡存完整参数，算完整梯度。
#         * Scatter-Reduce 梯度: 梯度分块求和。每张卡只拿到自己负责的那 1/N 梯度。
#         * Optimizer Step: 各卡只更新自己负责的那 1/N 参数的优化器状态（省显存的关键），并更新这部分参数。
#         * All-Gather 参数: 各卡广播自己更新好的 1/N 参数，拼回完整模型。
#         * 通信量: 1Φ (梯度) + 1Φ (参数) = 2Φ (与 DDP 持平)。
#         * 收益: 优化器状态显存降低 N 倍。
# 
#     * **收益**：内存消耗显著降低，且在带宽受限的情况下通信成本与朴素数据并行相同（"免费"的优化）。
# 
# 
# 2.  **ZeRO Stage 2: 切分梯度 (Gradient Sharding)**
#     * **原理**：在切分优化器状态的基础上，进一步切分梯度。
#     * **流程**：
#         * Forward: 各卡存完整参数，跑前向计算。
#         * Backward + Reduce-Scatter: 反向传播时，**每算完一层**的梯度，立即进行 Reduce-Scatter。每张卡只保留自己负责的那 1/N 梯度，**剩下的梯度立即释放**。
#         * Optimizer Step: 各卡用自己持有的 1/N 梯度和优化器状态，更新自己负责的 1/N 参数。
#         * All-Gather 参数: 各卡广播自己更新好的 1/N 参数，拼回完整模型。
#         * 通信量: 1Φ (梯度) + 1Φ (参数) = 2Φ (与 DDP 持平)。
#     * **收益**: 梯度显存降低为 1/N + 优化器状态显存降低为 1/N。
# 
# 3.  **ZeRO Stage 3 (FSDP): 全切分 (Shard Everything)**
#     * **原理**：将参数、梯度、优化器状态全部切分。每张卡只存 1/N 的模型。
#     * **流程**：
#         * Forward: 计算某一层前，执行 **All-Gather** 临时拉取该层完整参数。计算完该层后，**立即释放**参数（只保留自己负责的 1/N）。
#         * Backward: 计算某一层梯度前，再次 **All-Gather** 拉取该层完整参数。算出梯度后，立即执行 **Reduce-Scatter**，只保留自己负责的 1/N 梯度，释放其余梯度和参数。
#         * Optimizer Step: 各卡用本地的 1/N 梯度更新本地的 1/N 参数。
#         * 通信量: 1Φ (Forward 参数) + 1Φ (Backward 参数) + 1Φ (Backward 梯度) = 3Φ (比 DDP 增加 50%)。
#     * **收益**: 实现了真正的显存线性扩展，单卡显存占用极低。
# 
# 
# <div align="center">
# <img src="../images/para-zero-communication.png" width="75%"/>
# <br>
# ZeRO 3 的通信过程
# </div>
# 
# **ZeRO 总结**：
# * **Stage 1 & 2**：通信成本低（2x 参数量），内存有一定优化但不完全线性。
# * **Stage 3**：通信成本稍高（3x 参数量），但解决了单卡存不下大模型的问题。
# 
# > **为什么 ZeRO 不是模型并行？**
# > 虽然 ZeRO 切分了参数，但它依然属于**数据并行**的范畴。
# > *   **计算逻辑**：ZeRO 的每张卡依然处理**不同的数据 Batch**，执行相同的计算图（尽管参数是临时拉取的）。
# > *   **模型并行 (MP)**：指的是每张卡处理**同一批数据**，但负责计算图的**不同部分**（如不同层或矩阵的不同块）。
# > *   **本质区别**：ZeRO 切分的是**存储**（为了省显存），而计算依然是数据并行的逻辑；MP 切分的是**计算逻辑**和**模型结构**。
# 
# 
# <div align="center">
# <img src="../images/para-zero-3.png" width="50%"/>
# <br>
# ZeRO 3 有效的减少了单卡显存占用
# </div>
# 
# 

# 
# #### FSDP / ZeRO Stage 3 的实际工作机制
# 理解 FSDP / ZeRO Stage 3 不仅仅是理解它“切分了所有东西”，更重要的是它如何通过动态管理来维持高效运行。它包含三个核心机制：
# 
# 1.  **增量式的计算与通信 (Incremental computation / communication)**：
#     系统不是一次性加载或通信所有参数，而是跟随计算图的执行步骤，“按需”请求当前层所需的参数分片。
# 
# 2.  **即用即抛 (Parameters / gradients are requested / sent and then immediately freed)**：
#     参数和梯度一旦完成其计算任务（前向或后向传播），就会立即从 GPU 显存中释放。这确保了显存中始终只保留当前极小一部分活跃参数，从而极大降低了峰值显存占用。
# 
# 3.  **核心优化：通信与计算重叠 (Overlapping communication and computation)**：
#     这是 FSDP 能够高效运行的关键，旨在解决频繁切分参数带来的通信延迟问题。
#     * **定义**：指在 GPU 执行当前层的计算任务（如前向传播）时，后台利用独立的通信流（Stream）并行地进行下一层参数的 `All-Gather` 传输。
#     * **预取 (Prefetching)**：当 GPU 正在计算第 $i$ 层时，系统会提前启动第 $i+1$ 层的参数获取。
#     * **掩盖延迟**：理想情况下，如果计算时间大于通信时间，参数传输的耗时就被完全“掩盖”在计算时间内。
#     * **工作流**：GPU 计算流执行 `FWD0` -> `FWD1` 的同时，GPU 通信流并行执行 `AG1` -> `AG2`。这种流水线机制使得虽然总通信量增加了，但实际拖慢训练速度的“显式通信时间”被最小化了。
# 
#     
# <div align="center">
# <img src="../images/para-fsdp.png" width="75%"/>
# <br>
# FSDP/ ZeRO 3 会同步执行通信与计算任务
# </div>

# ### 2.2 模型并行 (Model Parallelism)
# 
# 当数据并行遇到 Batch Size 扩展的收益递减或单卡内存不足时，我们需要模型并行。模型并行的核心是将参数切分到多个 GPU 上，并在 GPU 间传递激活值（Activations）。
# 
# #### 流水线并行 (Pipeline Parallelism)
# * **原理**：将模型的层（Layers）切分给不同的 GPU（例如 GPU 0 负责 Layer 0-1，GPU 1 负责 Layer 2-3）。
# * **朴素实现的缺陷**：利用率极低。在任一时刻，只有一个 GPU 在工作，其他 GPU 都在空转（Bubble 问题）。
# * **解决方案：微批处理 (Micro-batches)**
#     * 将一个大的 Batch 切分为多个微批次（Micro-batches）依次送入流水线。
#     * 这使得多个 GPU 可以同时处理不同微批次的计算，从而减少空转比例。
#     * **Bubble 比例**：$\frac{n_{stages}-1}{n_{micro}}$，因此需要较大的 Batch Size 才能有效填充流水线。
# * **调度策略**：
#     * **1F1B (One Forward One Backward)**：执行一次前向传播后立刻执行一次反向传播，以尽早释放激活值内存，减少峰值内存占用。
# * **适用场景**：通常用于节点间（Inter-node）并行，因为其通信量较小（点对点传输激活值），对带宽要求相对较低。
# 
# <div align="center">
# <img src="../images/para-pipeline.png" width="50%"/>
# <br>
# 大的batch size是减少 bubble的关键
# </div>
# 
# 

# **进阶优化：用带宽换利用率 (Trading communication bandwidth for utilization)**
# 标准的流水线调度（如 1F1B）仍会存在一定的 Bubble。为了进一步提升利用率，可以采用更复杂的调度策略。
# * **虚拟流水线 (Interleaved Stages)**：不再是将连续的层分配给一个设备（例如 Device 1 只负责 Layer 1-4），而是给每个设备分配多个不连续的阶段（例如 Device 1 同时负责 Stage 1 和 Stage 5）。
# * **权衡**：这种方式通过更细粒度的任务分配填充了原本闲置的时间窗口，从而提高了计算利用率；但代价是设备间的通信次数和总量会显著增加。本质上，这是一种**用通信带宽换取计算利用率**的策略。
# 
# <div align="center">
# <img src="../images/para-interleaved.png" width="75%"/>
# <br>
# 将layer中的计算进一步切分为交替运行的stage
# </div>
# 
# **零气泡流水线 ('Zero Bubble' Pipelining)**
# 为了将 Bubble 压缩到极致，可以对反向传播过程进行拆解。
# * **拆分反向传播**：将反向传播计算拆分为两个独立的部分：
#     1.  **计算输入梯度 ($\nabla_x L$)**：必须优先计算。因为上一级流水线（Previous Stage）急需这个梯度来开始它自己的反向传播。这部分是**关键路径**。
#     2.  **计算权重梯度 ($\nabla_W L$)**：依赖于输入 $x$ 和输出梯度 $\nabla_y L$。这部分计算结果只用于更新本地参数，**不影响其他 GPU**。
# * **填补空隙**：由于 $\nabla_w L$ 的计算没有紧迫的时间依赖，调度器可以将其灵活地安排在流水线的任意空闲时间（Bubble）中执行。这种“见缝插针”的策略可以几乎完全消除流水线中的气泡。

# #### 张量并行 (Tensor Parallelism)
# * **原理**：在层内部（Intra-layer）进行切分。将矩阵乘法 $Y = XA$ 拆解为子矩阵运算。
#     * **列并行 (Column Parallel)**：将权重矩阵 $A$ 按列切分，每个 GPU 计算一部分输出特征。
#     * **行并行 (Row Parallel)**：将权重矩阵 $B$ 按行切分，需要先对输入进行切分。
# 
# <div align="center">
# <img src="../images/para-tensor.png" width="75%"/>
# <br>
# Tensor 并行
# </div>
# 
# * **前向与后向**：前向传播中通常是一次恒等映射（Identity）配合一次 `All-Reduce`；后向传播则相反。
# 
# 
# #### 张量并行 vs. 流水线并行：优缺点对比 (Tensor parallel – pros and cons vs pipeline parallel)
# 如何选择张量并行还是流水线并行？我们需要权衡它们的特性：
# 
# * **优点 (Pros)**：
#     * **无流水线气泡 (No bubble)**：只要网络传输够快，计算就不需要等待，硬件利用率高。
#     * **实现简单 (Low complexity)**：通常只需要包装一下模型层即可，不需要对基础设施做大规模的重构。
#     * **对 Batch Size 不敏感**：不需要像流水线并行那样依赖巨大的 Batch Size 来掩盖气泡，在小 Batch 下表现更好。
# 
# * **缺点 (Cons)**：
#     * **通信量巨大**：这是张量并行的致命弱点。
#         * 流水线并行只需要在微批次之间点对点传输激活值（通信量级为 $b \times s \times h$）。
#         * 张量并行则需要在**每一层**都进行 All-Reduce 通信，通信量大概是流水线并行的 8 倍甚至更多（量级为 $8 \times b \times s \times h$ 乘以设备相关的系数）。
# 
# * **结论与适用场景**：
#     * 正是由于通信量极大的特点，张量并行**必须**运行在具有低延迟、高带宽互联环境的设备之间。
#     * 在实践中，它通常被严格限制在**单机内部 (Intra-node)** 使用，以利用 NVLink 这样极高速的互联协议（例如单机 8 卡范围内）。一旦跨越机器（Inter-node），通常就会切换到通信量更小的流水线并行。

# ### 2.3 激活与序列并行 (Activation & Sequence Parallelism)
# 
# #### 最后的复杂性：动态的激活内存 (A final complexity - Activation Memory)
# 在讨论并行策略时，我们往往关注静态的模型参数内存。然而在实际训练中，显存是动态变化的，其中**激活值 (Activations)** 占据了极大的比例。
# * **内存消耗**：我们以“Efficient Large-Scale Language Model Training on GPU Clusters
# Using Megatron-LM” 这篇论文中训练的 Megatron-LM模型为例，对于每一层 Transformer，激活内存大约为 $sbh(34 + 5\frac{as}{h})$。其中 $s$ 是序列长度，$b$ 是微批次大小，$h$ 是隐藏层维度，$a$ 是注意力头数。
# * **主要来源**：这里的系数（如 34）来自于注意力机制中的二次项、Dropout 以及各种中间变量。
# * **挑战**：如果不加以处理，随着序列长度的增加，激活内存会迅速撑爆显存。虽然像 Flash Attention 这样的技术可以通过重计算（Recomputation）去掉部分项（如 $5\frac{as}{h}$），但剩余部分依然庞大。
# 
# <div align="center">
# <img src="../images/para-dynamic.png" width="50%"/>
# <br>
# 训练过程中的内存动态变化
# </div>
# 
# **为什么是 $sbh(34 + 5\frac{as}{h})$？**
# 
# 1.  **线性增长部分 (系数 34 = 24 + 10)**：
#     * **可并行部分 (系数 24)**：
#         * 这部分内存可以通过张量并行 (Tensor Parallel) 随 GPU 数量 $t$ 线性减少 (即 $\frac{24}{t}$)。
#         * **MLP 部分 ($\approx 18$)**: 包含 Linear ($h \to 4h$)、GeLU 输入、Linear ($4h \to h$) 及其反向传播缓存。
#         * **Self-Attention 线性投影 ($\approx 6$)**: 包含 Q, K, V 投影输入及 Output 投影输入。
#     * **固定开销部分 (系数 10)**：
#         * 这部分在普通张量并行下无法切分，是内存瓶颈所在。
#         * **LayerNorm**: **4** (存储输入和输出等)。
#         * **Dropout**: **2**。
#         * **Attention 和 MLP 的输入 (Inputs)**: **4** (保留残差连接处的输入副本)。
# 
# 2.  **二次增长部分 (系数 5 的构成)**：
#     * 来源于注意力机制中的二次项，包含 Dropout。
#     * **Attention Score Matrix ($Q K^T$)**: 1 (存储 $[b, a, s, s]$ 矩阵)。
#     * **Softmax**: 1 (存储 Softmax 结果用于反向传播)。
#     * **Dropout on Attention**: 1 (存储 Mask)。
#     * **Gradients**: 反向传播时计算 $Q, K$ 梯度所需的中间状态 $\approx 2$。
#     * **总计**: $5 \times b \cdot a \cdot s^2 = 5 \times sbh \cdot \frac{as}{h}$。
#     * *注：这部分可以通过 Flash Attention 的重计算 (Recomputation) 消除。*

# 
# #### 张量并行下的激活瓶颈 (Activation under Tensor Parallel)
# 张量并行（TP）虽然切分了参数，但在处理激活内存时并不完美。在 TP 模式下，矩阵乘法部分的激活被切分了，内存随 TP 大小 $t$ 线性减少（即 $\frac{24}{t}$ 部分）。
# 
# 然而，LayerNorm ($4sbh$)、Dropout ($2sbh$) 以及注意力/MLP 的输入 ($4sbh$) 等操作，通常需要在每个 GPU 上保存完整的副本以进行计算。这导致每一层都会残留一个与 GPU 数量无关的 $10sbh$ 常数项。这意味着即使增加再多的 GPU，这部分内存开销也无法降低，阻碍了模型的线性扩展。
# 
# #### 序列并行：实现真正的线性扩展 (Sequence Parallelism)
# 为了消除上述的 $10sbh$ 瓶颈，我们可以进一步引入了**序列并行 (Sequence Parallelism)**。
# * **核心洞察**：LayerNorm 和 Dropout 等操作是沿着序列维度（Sequence Dimension）进行的逐点运算（Pointwise Ops）。这意味着它们也是可以被切分的，只要我们沿着序列轴切分即可。
# * **工作机制**：
#     * **前向传播**：将原本的 `All-Reduce` 操作拆解。在进入 LayerNorm 之前，使用 `Reduce-Scatter` 将完整的激活值沿序列维度切分给不同的 GPU；计算完成后，再通过 `All-Gather` 收集回完整的序列供后续的张量并行层使用。
#     * **后向传播**：执行相反的操作。
# * **最终效果**：
#     * **TP (Baseline)**：$sbh(10 + \frac{24}{t} + ...)$
#     * **TP + SP**：$sbh(\frac{34}{t} + ...)$。通过结合序列并行，原本无法切分的 $10sbh$ 项也被 $t$ 除掉了。
#     * **终极方案**：结合 **TP + SP + 选择性激活重计算 (Selective Activation Recomputation)**，可以将单层的激活内存压缩到 $sbh(\frac{34}{t})$ 甚至更低，实现了内存随设备数量的完美线性扩展。
# 
# #### 其他并行策略 (Other parallelism strategies)
# 除了上述主流的三种并行（数据、流水线、张量/序列），还有一些针对特定场景的策略：
# * **Context Parallel / Ring Attention**：专门用于处理超长上下文（Context）。它将极长的序列切分到多个 GPU 上，通过环状通信（Ring）来计算注意力机制，突破单卡序列长度的限制。
# * **Expert Parallel (MoE)**：混合专家模型（Mixture of Experts）的并行策略。将不同的“专家”（Experts，通常是 FFN 层）分配给不同的 GPU。数据在经过 Gate 选择后，会被发送到对应的 GPU 上进行处理（All-to-All Dispatch），处理完后再传回（All-to-All Combine）。

# ### 2.4 总结：3D 并行 (3D Parallelism)
# 
# 为了训练超大规模模型，通常会同时组合使用上述三种并行策略，称为 3D 并行。
# 
# **并行策略选择的经验法则**：
# 1.  **内存优先**：首先确保模型能放进显存。
#     * 使用 **张量并行 (TP)** 直到填满单个节点的 GPU（例如 8 卡）。
#     * 如果还放不下，使用 **流水线并行 (PP)** 跨节点扩展。
#     * 或者根据带宽情况使用 **ZeRO-3 (FSDP)**。
# 2.  **扩展性**：当模型放得下后，使用 **数据并行 (DP/ZeRO)** 来扩展到更多机器。
# 3.  **效率权衡**：
#     * **DP/ZeRO-1**：适合高带宽、不需要切分模型的情况。
#     * **FSDP (ZeRO-3)**：线性内存扩展，但通信开销大（1.5x）。
#     * **Pipeline**：线性内存扩展，通信少，但有 Bubble，需要大 Batch。
#     * **Tensor + Seq**：无 Bubble，但通信量极大，仅适合节点内高速互联。

# ## 3. 利用并行技术扩展与训练大型语言模型 (Scaling and training big LMs with parallelism)
# 
# 在理解了各种基础的并行原语后，本部分主要探讨如何将它们组合起来（即所谓的 "3D 并行"），以便在成千上万个 GPU 上训练万亿参数级别的模型。
# 
# ### 3.1 并行原语总结与权衡
# 
# 在构建训练架构时，我们需要在有限的资源（显存、带宽、Batch Size）之间进行权衡。以下是各主要并行策略的特性对比：
# 
# #### 各策略特性对比表
# * **DDP / ZeRO-1**
#     * **同步开销**：每个 Batch 同步一次。
#     * **显存扩展性**：无（每个 GPU 需存完整参数）。
#     * **带宽需求**：$2 \times$ 参数量。
#     * **Batch Size 扩展性**：线性。
#     * **易用性**：非常简单。
# * **FSDP (ZeRO-3)**
#     * **同步开销**：每个 FSDP block 执行 3 次通信。
#     * **显存扩展性**：线性（参数被完全切分）。
#     * **带宽需求**：$3 \times$ 参数量（比 DDP 高 1.5 倍）。
#     * **Batch Size 扩展性**：线性。
#     * **易用性**：非常简单（通常只需包装一下模型）。
# * **流水线并行 (Pipeline Parallelism)**
#     * **同步开销**：每个 Pipeline 阶段同步一次。
#     * **显存扩展性**：线性。
#     * **带宽需求**：仅传输激活值（点对点通信，量级极小）。
#     * **Batch Size 扩展性**：**无**（需要极大的 Batch Size 来填充气泡）。
#     * **易用性**：较难（需要处理切分和调度）。
# * **张量 + 序列并行 (Tensor + Seq Parallelism)**
#     * **同步开销**：每个 Transformer block 需同步 2 次。
#     * **显存扩展性**：线性。
#     * **带宽需求**：极高（每层都要进行 $8 \times$ 激活值的 All-Reduce）。
#     * **Batch Size 扩展性**：无影响。
#     * **易用性**：较难。

# ### 3.2 3D 并行策略与经验法则 (3D Parallelism & Rules of Thumb)
# 
# "3D 并行"指的是同时使用数据并行、张量并行和流水线并行。根据文献和实践，可以总结出以下组合策略的经验法则：
# 
# #### 规则 1：优先解决显存问题
# 直到模型能完整放入显存之前：
# 1.  **节点内 (Intra-node)**：首先使用 **张量并行 (TP)**。因为 TP 通信量极大，必须利用节点内的高速互联（如 NVLink）。通常 TP 大小设置为每台机器的 GPU 数量（例如 8）。
# 2.  **节点间 (Inter-node)**：如果单机显存不够，使用 **流水线并行 (PP)** 跨机器扩展。PP 通信量小，适合跨节点较慢的网络连接。
# 3.  **替代方案**：如果带宽允许，也可以直接使用 **ZeRO-3 (FSDP)** 来替代复杂的 TP/PP 组合。
# 
# #### 规则 2：利用数据并行扩展规模
# 当模型已经能塞进 GPU 集群后：
# * 使用 **数据并行 (DP)** 或 **ZeRO-1/2** 来利用剩余的 GPU 资源，进一步加速训练。
# 
# #### 规则 3：处理小 Batch Size
# * 如果全局 Batch Size 较小，导致通信占比过高，可以使用 **梯度累积 (Gradient Accumulation)**。通过在本地累积多次梯度的更新，变相增加 Batch Size，从而以计算时间换取更高的通信效率。

# ### 3.3 扩展策略实例分析：Megatron-LM
# 
# 通过分析 Megatron-LM 的扩展策略（从 1.7B 到 1T 参数），我们可以看到策略随模型规模的演进路径：
# 
# #### 演进路径
# 1.  **小模型 (1.7B - 7.5B)**：主要依赖数据并行。张量并行 (TP) 逐渐增加到 4 或 8。
# 2.  **中等模型 (18.4B - 76B)**：
#     * **张量并行 (TP)**：保持在 8（占满单机 GPU）。
#     * **流水线并行 (PP)**：开始引入，从 1 增加到 4。
#     * **数据并行 (DP)**：随着总 GPU 数增加而增加。
# 3.  **超大模型 (310B - 1T)**：
#     * **TP**：依然锁定在 8。
#     * **PP**：显著增加（例如 1T 模型用到 64 阶段）。
#     * **DP**：由于显存和计算限制，DP 的并行度反而下降（例如 1T 模型在 3072 张 GPU 上训练时，DP 大小仅为 6）。
# 
# #### 关键性能观察
# * **线性收益**：通过精细的 3D 并行配置（如 ZeRO-3 或 PTD-P），即使 GPU 数量增加到近 2000 张，每张 GPU 的吞吐量（TeraFLOP/s）依然能保持平稳，实现了算力的线性扩展。
# * **TP 的最优值**：实践表明，**TP=8** 通常是性能最佳点。这对应于单台服务器内全互联的 GPU 数量。
# 
# <div align="center">
# <img src="../images/para-scaling.png" width="75%">
# <br>
# Megtrao-LM 的扩展策略
# </div>

# ### 3.4 激活重计算 (Activation Recomputation)
# 
# 除了并行策略，**激活重计算**也是提升效率的关键技术。
# * **原理**：在前向传播时不保存所有激活值，而是在反向传播时重新计算它们。
# * **收益**：以约 33% 的额外计算开销为代价，换取了大量的显存节省。
# * **结果**：节省下来的显存允许我们使用**更大的 Batch Size**。实验数据显示，开启激活重计算后，由于 Batch Size 增大带来的吞吐量提升，往往超过了重计算带来的计算损耗，最终实现了更高的每秒序列处理速度。

# ### 3.5 业界大模型训练实战
# 
# 以下是近期几个知名大模型的实际训练架构配置：
# 
# #### 1. OLMo (AllenAI)
# * **策略**：主要依赖 **ZeRO / FSDP**。
# * **配置**：
#     * 通过 PyTorch FSDP 框架切分模型权重和优化器状态。
#     * **混合精度**：分片权重和优化器状态保持 FP32。在 Transformer 块的前向/后向计算时，才将权重动态转为 BF16。
#     * **Batch Size**：7B 模型使用 4M tokens 的全局 Batch，65B 模型使用 Warmup 策略（从 2M 增加到 16M）。
# 
# #### 2. DeepSeek
# * **框架**：HAI-LLM（轻量级高效框架）。
# * **并行组合**：集成了 **数据并行 + 张量并行 + 序列并行 + 1F1B 流水线并行**。
# * **优化细节**：
#     * 使用 **Flash Attention** 提升利用率。
#     * 使用 **ZeRO-1** 对数据并行组的优化器状态进行切分。
#     * **计算通信重叠**：极致优化，包括在 ZeRO-1 中重叠 Reduce-Scatter，在序列并行中重叠 All-Gather。
#     * **算子融合**：融合 LayerNorm、GEMM 和 Adam 更新。
# 
# #### 3. Yi (01.AI)
# * **核心挑战**：显存与通信。
# * **解决方案**：
#     * **ZeRO-1**：消除优化器状态的显存冗余。
#     * **TP + PP**：在计算节点内结合张量并行和流水线并行，避免跨节点的通信瓶颈。
#     * **拓扑感知 (Topology-aware)**：资源分配策略考虑到物理拓扑（Fat-tree），尽量减少跨层交换机的通信。
# 
# #### 4. Llama 3 405B (Meta)
# * **并行维度排序**：**[TP, CP, PP, DP]**。
#     * **最内层 (TP)**：对带宽和延迟要求最高，限制在单机内部。
#     * **最外层 (DP/FSDP)**：可以容忍较高的网络延迟（因为可以异步预取权重和归约梯度），因此跨越多跳网络部署。
# * **硬件挑战**：在 16,000+ GPU 的规模下，硬件故障是常态。
#     * 在 54 天的预训练中，发生了 400 多次意外中断。
#     * **78% 的中断源于硬件问题**，其中 GPU 故障（30.1%）和 HBM3 显存故障（17.2%）是主要原因。
# 
# #### 5. Gemma 2 (Google)
# * **硬件**：TPUv4, TPUv5e, TPUv5p。
# * **策略**：**ZeRO-3 + 模型并行 (TP+SP) + 数据并行**。
# * **架构**：
#     * 27B 模型在 6144 个 TPUv5p 芯片上训练。
#     * 使用 Jax 和 Pathways 的 "Single Controller" 编程范式。
#     * 优化器状态使用类似 ZeRO-3 的技术进一步切分。

# ### 3.6 总结 
# 
# 1.  **规模的必然性**：超过一定规模后，单卡无法训练，必须采用多 GPU、多节点并行。
# 2.  **没有万能钥匙**：不存在单一的最佳并行方案。实际的大规模训练通常需要同时结合数据并行、流水线并行和张量/序列并行（即 3D 并行）。
# 3.  **组合原则**：
#     * **TP** 用于单机内扩展显存和算力。
#     * **PP** 用于跨机扩展显存。
#     * **DP/ZeRO** 用于在大规模集群上线性扩展训练速度。
#     * 合理利用 **FSDP** 和 **激活重计算** 来平衡显存与通信开销。
