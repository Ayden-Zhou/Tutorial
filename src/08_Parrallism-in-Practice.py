#!/usr/bin/env python
# coding: utf-8

# # 分布式训练基础：多 GPU 通信与原语
# 
# 在上一章中，我们要解决的核心问题是如何在**单个 GPU** 内部通过算子融合（Fusion）和分块（Tiling）来减少内存访问，从而提升计算效率。
# 
# 本章我们将视野扩大到**多个 GPU**。当单个 GPU 无法容纳模型或计算速度不够快时，我们需要跨设备并行。
# - **计算 (Compute)**：算术逻辑单元 (ALU)，负责运算。
# - **数据 (Data)**：存储在内存中，离计算单元较远。
# 
# **统一的主题**：编排计算以避免数据传输成为瓶颈。我们需要通过复制（Replication）或切分（Sharding）来减少跨 GPU/跨节点的通信。
# 
# ## 1. 硬件层级与通信带宽
# 
# 我们在进行分布式编程时，必须对硬件的通信层级有清晰的认知。这是一个从“小而快”到“大而慢”的层级结构：
# 
# 1.  **Single node, single GPU (L1 cache / Shared Memory)**: 极快，但容量极小。
# 2.  **Single node, single GPU (HBM)**: 高带宽内存（如 H100 的 3.9 TB/s），是我们模型参数和激活值主要存放的地方。
# 3.  **Single node, multi-GPU (NVLink)**: 同一台机器内的 GPU 互联，速度非常快（如 H100 的 NVLink 总带宽约 900 GB/s）。
# 4.  **Multi-node, multi-GPU (NVSwitch/Ethernet)**: 跨机器通信，通常通过 NVSwitch 或以太网，带宽相对受限。
# 
# 
# 
# 我们将分两部分来讲解：
# 1.  **Part 1 (本节)**: 分布式通信与计算的构建块（原理与基准测试）。
# 2.  **Part 2 (下节)**: 分布式训练策略（数据并行、张量并行、流水线并行）。

# In[10]:


import torch
import time
import os
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Callable

# 假设 utils 已在本地路径，包含辅助函数
# 实际 Notebook 中通常需要定义或导入这些工具函数
from parallel_utils import spawn, int_divide, summarize_tensor, get_init_params, render_duration, get_device

# 检查 CUDA 可用性
if torch.cuda.is_available():
    os.system("nvidia-smi topo -m")


# ####  拓扑结构识别 
# 
# 通过执行 `nvidia-smi topo -m`，我们可以查看到当前系统中 GPU 之间的物理连接方式。这对于理解通信瓶颈至关重要。
# 
# **你可能会看到的关键连接类型：**
# *   **NV# (如 NV12)**: 表示 GPU 之间通过多条 **NVLink** 直连。这是最理想的情况，带宽极高（数百 GB/s）。
# *   **PHB / PIX / PXB**: 表示数据必须经过 **PCIe 总线**。带宽通常受限于 PCIe 版本（如 PCIe 4.0 x16 约为 31.5 GB/s），远低于 NVLink。
# *   **SYS**: 表示数据需要跨越 **不同的物理 CPU** 及其内存域（NUMA）。这是最慢的路径，因为数据需要经过 CPU 间的互联通路（如 QPI/UPI/Infinity Fabric）。
# 
# **其他：**
# *   **CPU Affinity**: 显示 GPU 物理连接到了哪些 CPU 核心。将计算进程绑定到对应的核心（Affinity）可以显著减少延迟。
# *   **NUMA Affinity**: 提示 GPU 靠近哪个内存节点。跨 NUMA 节点的内存访问（Remote Access）会导致性能下降。
# 

# ## 2. 集合通信操作 (Collective Operations)
# 
# 在分布式编程中，我们不直接管理点对点（Point-to-Point）的通信，而是使用更高层级的抽象，称为**集合通信操作**。它定义了跨多个节点（例如 256 个 GPU）的数据通信模式。
# 
# ### 2.1 核心术语
# * **World Size**: 设备的总数量（例如 4）。
# * **Rank**: 当前设备的唯一 ID（例如 0, 1, 2, 3）。
# 
# ### 2.2 常见原语
# 我们需要熟练掌握以下几种原语。理解它们的最好方式是看数据是如何流动的。
# 
# * **Broadcast**: 一个设备（Root）将数据发送给所有其他设备。
#     
# * **Scatter**: 一个设备将数据的不同部分分发给不同的设备（是 Gather 的逆操作）。
#     
# * **Gather**: 将所有设备的数据收集到一个设备上。
#     
# * **Reduce**: 将所有设备的数据进行归约运算（如 Sum, Min, Max），结果存放在一个设备上。
#     
# * **All-gather**: 所有设备都收集所有设备的数据（相当于 Gather 后再 Broadcast）。
#     
# * **Reduce-scatter**: 对数据进行归约，但结果分散在不同设备上（相当于 Reduce 后再 Scatter）。
#     
# * **All-reduce**: 所有设备都得到归约后的结果。
#     
# 

# ### 2.3 硬件与库支持
# 
# **NCCL (NVIDIA Collective Communication Library)**
# NCCL 将上述集合操作转化为 GPU 之间传输的底层数据包。它会自动检测硬件拓扑（NVLink, PCIe, Switches），优化路径，并启动 CUDA Kernel 进行收发。
# 
# **PyTorch Distributed (`torch.distributed`)**
# PyTorch 提供了干净的 Python 接口来调用 NCCL。
# * `dist.all_gather_into_tensor`
# * `dist.all_reduce`

# ### 2.4 代码实战：集合通信演示
# 
# 让我们通过代码来验证这些操作。我们需要先定义设置（Setup）和清理（Cleanup）函数，用于初始化进程组。

# In[11]:


get_ipython().run_cell_magic('writefile', 'collective_ops_demo.py', 'import torch\nimport torch.distributed as dist\nfrom parallel_utils import setup, cleanup, get_device\n\ndef collective_operations_main(rank: int, world_size: int):\n    """\n    此函数将在每个进程（rank = 0, ..., world_size - 1）中异步运行。\n    """\n    setup(rank, world_size)\n\n    # --- 1. All-reduce 演示 ---\n    dist.barrier()  # 等待所有进程到达此处，为了打印整齐\n\n    # 创建 tensor: Rank 0 -> [0, 1, 2, 3], Rank 1 -> [1, 2, 3, 4] ...\n    tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank \n\n    print(f"Rank {rank} [before all-reduce]: {tensor}", flush=True)\n\n    # 执行 All-reduce (Sum)，原地修改 tensor\n    # 结果应该是所有 rank 对应位置的元素之和\n    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)\n\n    print(f"Rank {rank} [after all-reduce]: {tensor}", flush=True)\n\n    # --- 2. Reduce-scatter 演示 ---\n    dist.barrier()\n\n    # 输入数据: 每个 rank 持有 world_size 长度的数据\n    input = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank\n    # 输出数据: 每个 rank 只接收归约后的一部分（标量）\n    output = torch.empty(1, device=get_device(rank)) \n\n    print(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", flush=True)\n\n    # 结果：Rank i 将持有所有 input[i] 的和\n    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)\n\n    print(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", flush=True)\n\n    # --- 3. All-gather 演示 ---\n    dist.barrier()\n\n    # 将上一步 Reduce-scatter 的输出作为输入\n    input_tensor = output\n    # 输出: 收集所有 rank 的数据，恢复成向量\n    output_tensor = torch.empty(world_size, device=get_device(rank)) \n\n    print(f"Rank {rank} [before all-gather]: input = {input_tensor}, output = {output_tensor}", flush=True)\n\n    dist.all_gather_into_tensor(output_tensor=output_tensor, input_tensor=input_tensor, async_op=False)\n\n    print(f"Rank {rank} [after all-gather]: input = {input_tensor}, output = {output_tensor}", flush=True)\n\n    # 验证结论：All-reduce = Reduce-scatter + All-gather\n\n    cleanup()\n')


# In[12]:


import torch.multiprocessing as mp
from collective_ops_demo import collective_operations_main # 从刚才生成的文件导入

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(collective_operations_main, args=(world_size,), nprocs=world_size, join=True)


# ## 3. 基准测试 (Benchmarking)
# 
# 了解原理后，我们需要测量实际的通信带宽。这将帮助我们理解通信开销。我们将主要测试 `All-reduce` 和 `Reduce-scatter` 的性能。
# 
# ### 3.1 All-reduce 基准测试
# 
# All-reduce 是数据并行（DDP）中最常用的操作（用于同步梯度）。

# In[14]:


get_ipython().run_cell_magic('writefile', 'benchmark_demo.py', 'import time\nimport torch\nimport torch.distributed as dist\nfrom parallel_utils import setup, cleanup, get_device, render_duration\n\ndef all_reduce_benchmark(rank: int, world_size: int, num_elements: int):\n    setup(rank, world_size)\n\n    # 创建一个大 Tensor\n    tensor = torch.randn(num_elements, device=get_device(rank))\n\n    # 1. Warmup (预热)\n    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)\n    if torch.cuda.is_available():\n        torch.cuda.synchronize()\n    dist.barrier()\n\n    # 2. 计时\n    start_time = time.time()\n    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)\n    if torch.cuda.is_available():\n        torch.cuda.synchronize()\n    # 注意：计时结束前不需要再 barrier，因为 all_reduce 本身包含隐式同步（对于结果正确性而言），\n    # 但为了测量所有进程完成的时间，barrier 是合理的。\n    dist.barrier() \n    end_time = time.time()\n\n    duration = end_time - start_time\n    # 只打印 Rank 0 的时间，避免刷屏\n    if rank == 0:\n        print(f"[all_reduce] Rank {rank}: duration {render_duration(duration)}", flush=True)\n\n    # 3. 计算有效带宽 (Bus Bandwidth)\n    # Ring All-reduce 的总线带宽公式： B = (2 * (N-1) / N) * Size / Time\n    size_bytes = tensor.element_size() * tensor.numel()\n    # 修正后的带宽计算公式\n    bus_bandwidth = (2 * (world_size - 1) / world_size) * size_bytes / duration\n\n    if rank == 0:\n        print(f"[all_reduce] Size: {size_bytes/1024**2:.2f} MB")\n        print(f"[all_reduce] Bus Bandwidth = {bus_bandwidth / 1024**3:.2f} GB/s", flush=True)\n\n    cleanup()\n')


# In[15]:


import torch
import torch.multiprocessing as mp
from benchmark_demo import all_reduce_benchmark # 从文件导入

if __name__ == "__main__":
    print("\n>>> Running All-Reduce Benchmark")
    n_gpus = torch.cuda.device_count()
    world_size = min(4, n_gpus)
    print(f"Using world_size={world_size}")

    # 数据量: 100M 个 float32 = 400MB
    mp.spawn(all_reduce_benchmark, args=(world_size, 100 * 1024**2), nprocs=world_size, join=True)


# ### 3.2 Reduce-scatter 基准测试
# 
# Reduce-scatter 常用于 ZeRO (Zero Redundancy Optimizer) 或 FSDP 等技术中，用于切分梯度或参数。

# In[16]:


get_ipython().run_cell_magic('writefile', 'benchmark_reduce_scatter_demo.py', 'import time\nimport torch\nimport torch.distributed as dist\nfrom parallel_utils import setup, cleanup, get_device, render_duration\n\ndef reduce_scatter_benchmark(rank: int, world_size: int, num_elements: int):\n    setup(rank, world_size)\n\n    # 创建输入：每个 Rank 都有一个 (world_size, num_elements) 的矩阵\n    # 也就是说，输入数据量是 all-reduce 的 world_size 倍\n    input = torch.randn(world_size, num_elements, device=get_device(rank)) \n    output = torch.empty(num_elements, device=get_device(rank))\n\n    # Warmup\n    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)\n    if torch.cuda.is_available():\n        torch.cuda.synchronize()\n        dist.barrier()\n\n    # 计时\n    start_time = time.time()\n    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)\n    if torch.cuda.is_available():\n        torch.cuda.synchronize()\n        dist.barrier()\n    end_time = time.time()\n\n    duration = end_time - start_time\n    # 只打印 Rank 0 的时间\n    if rank == 0:\n        print(f"[reduce_scatter] Rank {rank}: duration {render_duration(duration)}", flush=True)\n\n    # 计算带宽\n    dist.barrier()\n    data_bytes = input.element_size() * input.numel()\n    # Reduce-scatter 只需要发送数据进行归约，不需要像 All-reduce 那样再广播结果\n    # 发送量系数没有 2x\n    sent_bytes = data_bytes * (world_size - 1)\n    total_duration = world_size * duration\n    bandwidth = sent_bytes / total_duration\n\n    if rank == 0:\n        print(f"[reduce_scatter] Measured Bandwidth = {round(bandwidth / 1024**3, 2)} GB/s", flush=True)\n\n    cleanup()\n')


# In[17]:


import torch
import torch.multiprocessing as mp
from benchmark_reduce_scatter_demo import reduce_scatter_benchmark # 从文件导入

# 运行基准测试
if __name__ == "__main__":
    print("\n>>> Running Reduce-Scatter Benchmark")
    n_gpus = torch.cuda.device_count()
    world_size = min(4, n_gpus)
    print(f"Using world_size={world_size}")

    mp.spawn(reduce_scatter_benchmark, args=(world_size, 100 * 1024**2), nprocs=world_size, join=True)


# ### 总结 
# 
# 通过本节，我们建立了分布式训练的心理模型：
# 1.  **层级结构**：从 L1 缓存到跨节点以太网，带宽呈数量级下降。
# 2.  **集合通信**：使用 Broadcast, Reduce, All-gather 等原语来描述数据流，而不是手写 send/recv。
# 3.  **NCCL & PyTorch**：底层的库帮我们要么做到了拓扑感知，要么提供了易用的 Python 接口。
# 4.  **基准测试**：通信是昂贵的（GB/s 级别），而计算是极其快速的（TFLOPs 级别），因此我们需要精心设计并行策略。
# 
# 在下一部分中，我们将利用这些原语构建实际的三种并行策略：数据并行、张量并行和流水线并行。

# ## 4. 分布式训练策略 (Distributed Training Strategies)
# 
# 我们将通过“手动”编写 PyTorch 代码来实现三种主流的并行策略。这有助于消除高级库（如 DDP, Megatron-LM, DeepSpeed）的神秘感。
# 
# ### 4.1 数据并行 (Data Parallelism)
# 
# 这是最简单也最常用的策略。
# 
# * **切分策略**：沿着 **Batch（批次）** 维度切分数据。
# * **模型状态**：每个 Rank（GPU）持有完整的模型参数副本。
# * **计算流程**：
#     1.  每个 Rank 读取不同的数据分片。
#     2.  独立进行前向传播和反向传播，计算出**局部梯度**。
#     3.  **同步**：所有 Rank 之间执行 **All-reduce** 操作，对梯度取平均。
#     4.  更新参数（保证所有 Rank 的参数更新一致）。
# 
# 
# 
# 让我们先生成一些模拟数据：

# In[18]:


def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data


# 接下来是数据并行的核心实现。请注意代码中的 `dist.all_reduce`，这是 DDP (DistributedDataParallel) 的灵魂所在。

# In[ ]:


get_ipython().run_cell_magic('writefile', 'data_parallel_demo.py', 'import torch\nimport torch.nn.functional as F\nimport torch.distributed as dist\nfrom parallel_utils import setup, cleanup, get_device, int_divide, get_init_params, summarize_tensor\n\ndef data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):\n    setup(rank, world_size)\n\n    # --- 1. 数据分片 ---\n    # 每个 rank 获取数据的一个切片\n    # 在实际工程中，这里通常通过 DistributedSampler 来处理\n    batch_size = data.size(0)\n    num_dim = data.size(1)\n    local_batch_size = int_divide(batch_size, world_size)\n\n    start_index = rank * local_batch_size\n    end_index = start_index + local_batch_size\n\n    # 将属于本 rank 的数据移动到 GPU\n    local_data = data[start_index:end_index].to(get_device(rank))\n\n    # --- 2. 模型初始化 ---\n    # 关键点：每个 rank 都持有完整的参数副本 (params)\n    # 并且初始化必须完全一致（这里通过固定 seed 实现）\n    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]\n    optimizer = torch.optim.AdamW(params, lr=1e-3)\n\n    for step in range(num_steps):\n        # --- 3. 前向传播 (Local) ---\n        x = local_data\n        for param in params:\n            x = x @ param\n            x = F.gelu(x)\n\n        # Loss 是基于本地数据计算的，因此每个 rank 的 Loss 值不同\n        loss = x.square().mean()\n\n        # --- 4. 反向传播 (Local) ---\n        loss.backward()\n\n        # --- 5. 梯度同步 (Global) ---\n        # 这就是数据并行的核心：平均所有 worker 的梯度\n        # 这样能等效于在整个大 Batch 上进行的训练\n        for param in params:\n            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)\n\n        # --- 6. 参数更新 ---\n        # 由于梯度已经同步，所有 rank 的 optimizer step 行为将完全一致\n        optimizer.step()\n\n        print(f"[data_parallelism] Rank {rank}: step = {step}, "\n              f"loss = {loss.item():.4f}, "\n              f"params_sample = {[summarize_tensor(params[i]) for i in range(num_layers)]}", \n              flush=True)\n\n    cleanup()\n')


# In[ ]:


import torch
from data_parallel_demo import data_parallelism_main

# 启动演示
if __name__ == "__main__":
    print(">>> Running Data Parallelism")
    data = generate_sample_data()
    n_gpus = torch.cuda.device_count()
    world_size = min(4, n_gpus)
    print(f"Using world_size={world_size}")
    spawn(data_parallelism_main, world_size=world_size, data=data, num_layers=4, num_steps=1)


# 
# * 可以看到每个 Rank 的 `loss` 是不同的（因为数据不同）。
# * 但参数 `params` 在更新后保持完全一致（因为梯度被平均了）。
# 
# 
# ### 4.2 张量并行 (Tensor Parallelism)
# 
# 当模型太大，单个 GPU 甚至放不下参数时，我们需要将模型切开。张量并行（TP）通常用于切分矩阵乘法的大矩阵。
# 
# * **切分策略**：沿着 **Width（宽度/特征）** 维度切分权重矩阵。
# * **通信模式**：每一层计算都需要通信（通常是 All-gather 或 All-reduce）来重组激活值。
# 
# 
# 
# 在这个示例中，我们将演示一种简单的**列切分（Column Parallelism）**：
# 1.  我们将权重矩阵 $W$ 沿列切分，每个 Rank 持有 $W$ 的一部分列。
# 2.  输入 $X$ 是完整的。
# 3.  每个 Rank 计算 $Y_{partial} = X \cdot W_{local}$。
# 4.  此时每个 Rank 得到的是输出的一部分特征（Partial Output）。
# 5.  通过 **All-gather** 将所有部分的输出拼接起来，恢复完整的 $Y$，以便输入下一层。

# In[26]:


get_ipython().run_cell_magic('writefile', 'tensor_parallel_demo.py', 'import torch\nimport torch.nn.functional as F\nimport torch.distributed as dist\nfrom parallel_utils import setup, cleanup, get_device, int_divide, get_init_params, summarize_tensor\n\ndef tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):\n    setup(rank, world_size)\n\n    # 假设输入数据对所有 rank 可见（或者这里我们为了演示让所有 rank 拥有全量数据）\n    data = data.to(get_device(rank))\n    batch_size = data.size(0)\n    num_dim = data.size(1)\n\n    # 确定分片大小：将特征维度 num_dim 切分\n    local_num_dim = int_divide(num_dim, world_size)\n\n    # --- 1. 模型分片 ---\n    # 每个 rank 只持有 1/world_size 的参数\n    # 权重形状为: [num_dim, local_num_dim]\n    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]\n\n    # --- 2. 前向传播 ---\n    x = data\n    for i in range(num_layers):\n        # Local Compute: (batch, num_dim) @ (num_dim, local_num_dim) -> (batch, local_num_dim)\n        # 这一步只计算了输出向量的一部分切片\n        x = x @ params[i] \n        x = F.gelu(x)\n\n        # Communication: 需要收集所有 rank 的计算结果来恢复完整的向量\n        # 准备缓冲区\n        activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) \n                       for _ in range(world_size)]\n\n        # All-gather: 每个 rank 将自己的结果广播给所有人\n        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)\n\n        # 拼接: 将 [slice_1, slice_2, ...] 拼回 [full_vector]\n        # 结果形状回到: (batch_size, num_dim)\n        x = torch.cat(activations, dim=1)\n\n    print(f"[tensor_parallelism] Rank {rank}: output shape {summarize_tensor(x)}", flush=True)\n\n    # 注意：反向传播需要推导对应的梯度切分逻辑（此处留作练习）\n\n    cleanup()\n')


# In[27]:


from tensor_parallel_demo import tensor_parallelism_main
import torch

if __name__ == "__main__":
    print("\n>>> Running Tensor Parallelism")
    data = generate_sample_data()

    # 自动检测可用 GPU
    n_gpus = torch.cuda.device_count()
    world_size = min(4, n_gpus)
    print(f"Using world_size={world_size}")

    spawn(tensor_parallelism_main, world_size=world_size, data=data, num_layers=4)


# ### 4.3 流水线并行 (Pipeline Parallelism)
# 
# 另一种模型并行方式是将层“切断”。
# 
# * **切分策略**：沿着 **Depth（深度/层数）** 维度切分。例如，Rank 0 处理第 1-2 层，Rank 1 处理第 3-4 层。
# * **通信模式**：点对点通信（Point-to-Point, `send`/`recv`）。Rank $i$ 计算完后将激活值传给 Rank $i+1$。
# * **气泡（Bubble）**：为了减少等待时间，通常将 Batch 切分为更小的 **Micro-batches**。

# In[32]:


get_ipython().run_cell_magic('writefile', 'pipeline_parallel_demo.py', 'import torch\nimport torch.nn.functional as F\nimport torch.distributed as dist\nfrom parallel_utils import setup, cleanup, get_device, int_divide, get_init_params, summarize_tensor\n\ndef pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):\n    setup(rank, world_size)\n\n    # 数据准备\n    data = data.to(get_device(rank))\n    batch_size = data.size(0)\n    num_dim = data.size(1)\n\n    # --- 1. 层分片 ---\n    # 将总层数平分给各个 rank\n    local_num_layers = int_divide(num_layers, world_size)\n\n    # 每个 rank 初始化属于自己的那部分层\n    # Rank 0: Layers 0~1; Rank 1: Layers 2~3 (假设 total=4, world=2)\n    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]\n\n    # --- 2. Micro-batch 切分 ---\n    # 为了减少流水线气泡，将大 Batch 切小\n    micro_batch_size = int_divide(batch_size, num_micro_batches)\n\n    if rank == 0:\n        # 只有 Rank 0 拥有原始输入数据\n        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)\n    else:\n        # 后续 Rank 只需要分配接收缓存\n        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) \n                         for _ in range(num_micro_batches)]\n\n    # --- 3. 流水线执行 ---\n    for i, x in enumerate(micro_batches):\n        # A. 接收 (Recv)\n        # 如果不是第一个 stage，需要从上一个 rank 接收数据\n        if rank - 1 >= 0:\n            dist.recv(tensor=x, src=rank - 1)\n\n        # B. 计算 (Compute)\n        # 执行本 rank 负责的层\n        for param in local_params:\n            x = x @ param\n            x = F.gelu(x)\n\n        # C. 发送 (Send)\n        # 如果不是最后一个 stage，将结果发给下一个 rank\n        if rank + 1 < world_size:\n            print(f"[pipeline_parallelism] Rank {rank}: micro-batch {i} "\n                  f"sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)\n            dist.send(tensor=x, dst=rank + 1)\n        else:\n            # 最后一个 rank 输出最终结果\n            if i == 0: # 只打印一次作为示例\n                 print(f"[pipeline_parallelism] Rank {rank}: micro-batch {i} "\n                       f"final output {summarize_tensor(x)}", flush=True)\n\n    print("注：此处未展示通信与计算的重叠（Overlap），那是消除气泡的关键优化。")\n\n    cleanup()\n')


# In[33]:


from pipeline_parallel_demo import pipeline_parallelism_main
import torch


# 启动演示
if __name__ == "__main__":
    print("\n>>> Running Pipeline Parallelism")
    # 仅使用 2 个 GPU 演示流水线
    spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)


# ## 4.4 总结与展望 (Summary)
# 
# 通过本章的学习，我们从底层原语构建起了复杂的分布式系统。
# 
# 1.  **并行维度**：
#     * **Data**: 切分 Batch，使用 All-reduce 同步梯度。
#     * **Tensor**: 切分 Width，使用 All-gather/Reduce-scatter 同步激活值。
#     * **Pipeline**: 切分 Depth，使用 Send/Recv 传递中间结果。
#     * *(未涉及)* **Sequence**: 切分 Length（如 Ring Attention）。
# 
# 2.  **权衡**：
#     * 你可以选择**重计算 (Re-compute)** 以节省显存。
#     * 你可以选择将状态存在**内存 (CPU Memory)** 中（Offloading）。
#     * 你可以选择将状态存在**其他 GPU** 中并进行**通信 (Communicate)**。
#     
#     硬件在变快，但模型永远想要更大。因此，这种分层级的并行结构（Hierarchical Structure）将是未来系统设计的常态。
# 
# 3.  **遗漏的部分**：
#     * 更通用的模型结构（如 Attention 层的并行化）。
#     * 通信与计算的重叠（Overlap）：这是高性能训练的关键，但需要极其复杂的 bookkeeping。
#     * **JAX/TPU 生态**：例如 Levanter 等框架，允许用户仅定义模型和切分策略，编译器自动处理通信。而在 PyTorch 中，我们通常需要像刚才那样（或者通过 DeepSpeed/Megatron）显式构建。
# 
# 希望这份笔记能让你对大规模语言模型背后的“基础设施”有更直观的理解。
