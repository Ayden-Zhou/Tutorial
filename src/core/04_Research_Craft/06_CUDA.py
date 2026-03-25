# %% [markdown]
# # CUDA
#
#
# ## 1. 简介与环境配置
#
# ### 1.1 概览
#
# 我们将重点关注以下两点：
# 1.  **Benchmarking & Profiling（基准测试与性能分析）**：这是理解性能瓶颈的关键。
# 2.  **编写 Kernels（内核）**：我们将动手编写 GPU 内核。
#
# 为了获得最佳体验，建议在有 GPU 的环境中运行以下代码。
#
# 首先，我们需要进行一些必要的环境配置和工具函数导入，以便后续代码能够顺利运行。

# %%
import os

# 设置环境变量，以便在 CUDA 出错时获得更详细的报错信息
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
from typing import Callable
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl

import math



# === 基础工具函数 (来自 lecture_util 和 torch_util) ===

def get_device(index: int = 0) -> torch.device:
    """尝试使用 GPU，如果不可用则使用 CPU。"""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

def ensure_directory_exists(path: str):
    """确保目录存在"""
    if not os.path.exists(path):
        os.makedirs(path)

# 检查当前环境是否支持 GPU
if not torch.cuda.is_available():
    print("建议在 GPU 环境下运行本笔记以获得完整体验。")
else:
    print(f"当前使用的设备: {get_device()}")

def check_equal(f1, f2):
    x = torch.randn(2048, device=get_device())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def check_equal2(f1, f2):
    x = torch.randn(2048, 2048, device=get_device())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)



def round1(x: float) -> float:
    """Round to 1 decimal place."""
    return round(x, 1)


def mean(x: list[float]) -> float:
    return sum(x) / len(x)

# === 补充辅助函数 ===

def run_operation1(dim: int, operation: Callable) -> Callable:
    """
    辅助函数：用于测试一元操作 (如 gelu, softmax)。
    它创建一个随机的 dim x dim 矩阵 x，并返回一个无参函数，
    该函数调用 operation(x)。
    """
    # Setup: create one random dim x dim matrix
    x = torch.randn(dim, dim, device=get_device())
    
    # Return a function to perform the operation
    return lambda : operation(x)

# %% [markdown]
# ### 1.2 GPU 硬件架构回顾
#
# 在开始编写代码之前，我们需要复习一下 GPU 的硬件组成。了解硬件是写出高性能代码的前提。
#
# GPU 主要由以下几个核心部分组成（以 NVIDIA A100 为例）：
#
# * **计算单元 (Compute)**:
#     * **Streaming Multiprocessors (SMs)**: 这是 GPU 的核心计算引擎。例如 A100 拥有 108 个 SM。
# * **内存层次结构 (Memory)**:
#     * **DRAM (Global Memory)**: 显存，容量大但速度慢（A100: 80GB）。这是我们通常存放 Tensor 的地方。
#     * **L2 Cache**: 二级缓存，速度快于 DRAM（A100: 40MB）。
#     * **L1 Cache**: 一级缓存，位于每个 SM 内部，非常小但极快（A100: 每个 SM 192KB）。
#
# 我们可以编写一个简单的函数来查看当前 GPU 的具体规格。

# %%
def print_gpu_specs():
    """打印当前机器上所有 GPU 的规格信息"""
    if not torch.cuda.is_available():
        print("未检测到 GPU。")
        return

    num_devices = torch.cuda.device_count()
    print(f"检测到 {num_devices} 个 GPU 设备:")
    
    for i in range(num_devices):
        properties = torch.cuda.get_device_properties(i)
        print(f"设备 {i}: {properties.name}")
        print(f"  - 总显存: {properties.total_memory / 1024**3:.2f} GB")
        print(f"  - SM 数量 (MultiProcessor Count): {properties.multi_processor_count}")
        # 注意：CUDA Capability 版本也会影响支持的特性
        print(f"  - Compute Capability: {properties.major}.{properties.minor}")

print_gpu_specs()

# %% [markdown]
# ### 1.3 执行模型 (Execution Model)
#
# 理解了硬件，我们再来看看代码是如何在 GPU 上运行的。基本的执行逻辑是：我们需要对所有的索引 $i = 0, \dots, N-1$ 运行函数 $f(i)$。
#
# CUDA 的执行模型包含三个层级概念：
#
# 1.  **Thread (线程)**: 处理单个索引的基本单元（即执行一次 $f(i)$）。
# 2.  **Thread Block (线程块)**: 一组线程的集合，也被称为 CTA (Concurrent Thread Array)。**一个线程块会被调度到单个 SM 上执行**。
# 3.  **Grid (网格)**: 线程块的集合，代表了整个计算任务。
#
# > **Warp (线程束)**
# >
# >
# > 在 CUDA 编程逻辑中，我们通常只定义 **Grid** 和 **Block**。但在**硬件实际执行**时，SM 并不会以单个线程为单位进行调度，而是以 **Warp** 为单位。
# >
# > * **定义**: 一个 Warp 包含 **32 个连续的线程**（Thread 0-31 组成第一个 Warp，以此类推）。
# > * **SIMT 机制**: Warp 内的 32 个线程执行完全相同的指令（Single Instruction Multiple Threads）。
# > * **关键影响**: 
# >     * 如果代码中出现了 `if-else` 分支，导致 Warp 内的一部分线程走 `if`，另一部分走 `else`，就会发生 **Warp Divergence (分支发散)**，硬件必须串行执行这两个分支，导致性能下降。
# >     * 因此，Warp 是连接“软件定义的线程”与“硬件 SM”之间的桥梁。
#
# #### 为什么需要线程块 (Thread Blocks)？
# 核心原因是 **Shared Memory (共享内存)**。我们希望将读取相似数据的 $f(i)$ 任务分在一组。
# * **机制**: 同一个线程块内的线程共享一块极快的内存（Shared Memory，速度媲美 L1 Cache）。例如在 A100 上这块内存大约是 164KB。
# * **同步**: 我们可以在同一个块内的线程之间进行同步（Synchronize），以协调读写操作；但**跨线程块**之间通常无法直接同步。

# %% [markdown]
# ### 1.4 硬件与执行的交互
#
# 软件层面的“线程块”与硬件层面的“SM”是如何交互的？这直接关系到性能。
#
# #### Wave Quantization (波次量化)
# 线程块是分批次（Waves）调度到 SM 上执行的。
#
# * **问题**: 如果任务的最后一批次包含的线程块很少，无法填满所有 SM，那么就会导致部分 SM 空闲，利用率（Occupancy）降低。
# * **优化策略**: 尽量让线程块的总数量是 SM 数量的整数倍。
# * **经验法则**: 线程块的数量最好 $\ge 4 \times$ SM 的数量，以掩盖内存延迟。
# * **挑战**: 硬件的某些细节（如具体的调度逻辑、SM 的确切数量）对执行模型是隐藏的，这增加了优化的难度。
#
# #### Arithmetic Intensity (算术强度)
# 这是一个衡量计算与内存访问比例的指标：
#
# $$
# \text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes}}
# $$
#
# * **Compute-bound (计算受限)**: 算术强度高，性能瓶颈在于计算能力（是好事，说明充分利用了算力）。
# * **Memory-bound (内存受限)**: 算术强度低，性能瓶颈在于内存带宽（是坏事，大部分时间在等数据）。
#
# **通用规则**:
# * **矩阵乘法 (MatMul)** 通常是 Compute-bound 的。
# * **其他大部分操作**（如 Element-wise 操作、Reduction 等）通常是 Memory-bound 的。

# %% [markdown]
# ## 2. 基准测试与性能分析 (Benchmarking & Profiling)
#
# 在深度学习系统开发中，有一条黄金法则：**永远不要猜测性能，要测量它。**
#
# 我们通常通过两种方式来衡量性能：
# 1.  **Benchmarking (基准测试)**: 关注“宏观时间”，即完成某个操作需要多长时间（Wall-clock time）。它用于对比不同实现的快慢，以及理解性能随规模（Scaling）的变化。
# 2.  **Profiling (性能分析)**: 关注“微观细节”，即时间具体花在了哪里。它能帮我们看到底层调用了哪些 CUDA Kernel，以及 CPU 和 GPU 之间的交互。
#
# ### 2.1 准备工作：定义负载 (The Workload)
#
# 为了进行测试，我们需要一个具有代表性的计算负载。这里我们定义一个简单的多层感知机（MLP），包含 `Linear` 层和 `GeLU` 激活函数。这是一个非常通用的深度学习模块。
#
# 我们需要定义模型结构，以及辅助函数来执行“前向传播 + 反向传播”的完整流程。

# %%
# === 定义模型与辅助函数 ===

class MLP(nn.Module):
    """
    一个简单的 MLP: Linear -> GeLU -> Linear -> GeLU ...
    """
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        # 使用 ModuleList 堆叠层
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)
        return x

def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    """
    创建一个闭包函数，用于运行 MLP 的前向和反向传播。
    """
    # 1. 定义模型 (随机权重)
    model = MLP(dim, num_layers).to(get_device())

    # 2. 定义输入 (随机数据)
    x = torch.randn(batch_size, dim, device=get_device())

    def run():
        # 运行 num_steps 次 (注意：这里为了简化测试，没有包含优化器更新步骤)
        for step in range(num_steps):
            # Forward
            y = model(x).mean()
            # Backward
            y.backward()
    
    return run

def run_operation2(dim: int, operation: Callable) -> Callable:
    """辅助函数：用于测试二元操作 (如 matmul, add)"""
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    return lambda : operation(x, y)

def mean(x: list[float]) -> float:
    return sum(x) / len(x)

# %% [markdown]
# ### 2.2 基准测试 (Benchmarking)
#
# Benchmarking 的核心是测量端到端的时间。在 GPU 上测量时间有一个非常关键的细节：**异步执行**。
#
# GPU 的启动是异步的，Python 代码（CPU端）下发完指令后会立刻返回，而不用等待 GPU 执行完毕。因此，在计时结束前，必须调用 `torch.cuda.synchronize()` 来强制等待 GPU 完成所有任务。
#
# 此外，我们需要 **Warmup（预热）**。第一次运行代码时，PyTorch 可能需要进行内存分配、Kernel 编译或加载等初始化工作，这会干扰稳态性能的测量。

# %%
def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """
    对函数 `run` 进行基准测试。
    
    Args:
        num_warmups: 预热次数，忽略其时间。
        num_trials: 正式测试次数，取平均值以减少方差。
    """
    # 1. Warmup
    # 第一次运行通常较慢（编译、缓存填充等），我们只关心稳态性能
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 重要！等待 CUDA 线程结束

    # 2. 正式计时
    times: list[float] = [] 
    for trial in range(num_trials):
        start_time = time.time()

        run()  # 执行计算
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 重要！再次等待
            
        end_time = time.time()
        times.append((end_time - start_time) * 1000) # 转换为毫秒 (ms)

    mean_time = mean(times)
    print(f"[{description}] Mean time: {mean_time:.2f} ms")
    return mean_time

# === 测试 1: 简单的 sleep ===
# 验证我们的 benchmark 函数是否工作正常 (预期约 50ms)
benchmark("sleep", lambda : time.sleep(50 / 1000))

# %% [markdown]
# #### 探索矩阵乘法的 Scaling
# 让我们看看矩阵乘法（MatMul）的性能如何随着维度变化。矩阵乘法（Square Matrix Multiplication）的理论计算量是 $2N^3$ (FLOPs)。当矩阵维度 $N$ 翻倍时，计算量应该增加 8倍 ($2^3=8$)。但在 GPU 上这并不总是成立的，因为存在并行的“波次（Wave）”效应和缓存命中率的影响。

# %%
print("\n=== Benchmarking Matrix Multiplication ===")
if torch.cuda.is_available():
    dims = (1024, 2048, 4096, 8192) 
else:
    dims = (1024, 2048)

for dim in dims:
    benchmark(f"matmul(dim={dim})", run_operation2(dim=dim, operation=lambda a, b: a @ b))

# %% [markdown]
# #### 探索 MLP 的 Scaling
# 接下来，我们通过改变不同的超参数（步数、层数、Batch Size、维度），来观察 MLP 性能的变化规律。这有助于我们理解模型在不同轴上的扩展性。

# %%
print("\n=== Benchmarking MLP Scaling ===")
# 基础配置
dim = 256
num_layers = 4
batch_size = 256
num_steps = 2

# 1. 基准运行
benchmark("run_mlp (base)", run_mlp(dim, num_layers, batch_size, num_steps))
print("\n=== Benchmarking num_steps Scaling ===")
# 2. 扩展 num_steps (线性增长？)
for scale in (2, 4, 8, 16):
    benchmark(f"run_mlp({scale}x steps)", 
              run_mlp(dim, num_layers, batch_size, scale * num_steps))
print("\n=== Benchmarking num_layers Scaling ===")
# 3. 扩展 num_layers (线性增长？)
for scale in (2, 4, 8, 16):
    benchmark(f"run_mlp({scale}x layers)", 
              run_mlp(dim, scale * num_layers, batch_size, num_steps))
print("\n=== Benchmarking batch_size Scaling ===")
# 4. 扩展 batch_size (并行度增加，时间如何变化？)
for scale in (2, 4, 8, 16, 32, 64, 128, 256):
    benchmark(f"run_mlp({scale}x batch)", 
              run_mlp(dim, num_layers, scale * batch_size, num_steps))
print("\n=== Benchmarking dimension Scaling ===")
# 5. 扩展 dimension (计算量是平方增长？)
for scale in (2, 4, 8, 16, 32, 64):
    benchmark(f"run_mlp({scale}x dim)", 
              run_mlp(scale * dim, num_layers, batch_size, num_steps))

# %% [markdown]
# ### MLP 扩展性分析 (Scaling Analysis)
#
# *   **Steps & Layers (线性增长)**: 串行执行导致时间与深度呈严格线性关系。无法并行加速。
# *   **Batch Size (免费午餐)**: 在 32x 之前时间几乎不变。这是填充 GPU 空闲算力的过程，并未触及硬件瓶颈。
# *   **Dimension (指数爆炸)**: 计算量 $O(N^3)$ 和参数量 $O(N^2)$ 双重增长。64x 维度导致时间激增 60 倍，迅速撞上显存带宽和算力墙。

# %%
import gc
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ### 2.3 性能分析 (Profiling)
#
# Benchmarking 告诉我们要花多久，而 Profiling 告诉我们要**做什么**。PyTorch 内置的 Profiler 非常强大，它可以捕获 CPU 和 CUDA 之间的活动。
#
# 我们将编写一个 `profile` 函数，它利用 `torch.profiler` 记录执行过程，并按照 CUDA 执行时间排序打印出最耗时的操作。

# %%
def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    """
    使用 PyTorch Profiler 分析函数 `run`。
    """
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f"\n## Profiling: {description}")
    
    # 使用上下文管理器启动 Profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=with_stack, # 是否记录调用栈（用于生成 Flame Graph）
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 打印统计表
    # sort_by="cuda_time_total" 让我们可以一眼看到最耗时的 GPU 操作
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    print(table)

    # 如果需要可视化调用栈 (Flame Graph)，可以导出文件
    if with_stack:
        ensure_directory_exists("var")
        prof.export_stacks(f"var/stacks_{description}.txt", "self_cuda_time_total")
    
    return table

# %% [markdown]
# #### 剖析基础操作
# 让我们看看简单的加法（Add）和矩阵乘法（MatMul）在底层分别调用了什么内核。
#
# **观察重点：**
# 1.  **Kernel Name**: 注意看 CUDA Kernel 的名字。例如 MatMul 可能会调用 `cutlass_...`，这说明 PyTorch 底层使用了 NVIDIA 的 CUTLASS 库来优化线性代数运算。
# 2.  **Kernel 变化**: 不同大小的输入可能会触发不同的 Kernel 实现。

# %%
# 1. 分析加法 (Add)
add_function = lambda a, b: a + b
profile("add", run_operation2(dim=2048, operation=add_function))

# 2. 分析矩阵乘法 (MatMul)
matmul_function = lambda a, b: a @ b
profile("matmul", run_operation2(dim=2048, operation=matmul_function))

# 3. 分析小尺寸矩阵乘法
# 注意观察：小矩阵调用的 Kernel 可能与大矩阵不同
matmul_function_128 = lambda a, b: a @ b
profile("matmul(dim=128)", run_operation2(dim=128, operation=matmul_function_128))

# %% [markdown]
#
#
# 我们通过 PyTorch Profiler 收集了 `add`（向量加法）、`matmul`（大矩阵乘法）和 `matmul(dim=128)`（小矩阵乘法）的详细执行数据。这些数据生动地展示了 GPU 计算中的几种典型模式。
#
# -  操作名称。
#     *   `aten::add`, `aten::matmul`: PyTorch 的顶层 API 调用。
#     *   `cudaXXXXX`: CPU 向 GPU 发射内核的指令（开销所在）。
#     *   `sm80_xmma_gemm...`: 实际在 GPU 上运行的 CUDA Kernel（真正的计算工作）。
#
# **CPU Bound (CPU 瓶颈) - 向量加法**
# 在 `add` 操作中，我们观察到一个巨大的差异：
# *   **CPU 总时间**: 1.914 ms
# *   **GPU (CUDA) 实际计算时间**: 仅 24.992 us (微秒)
#
# 这是一个典型的 **Overhead Bound** 或 **CPU Bound** 场景。当我们对小数据量进行简单操作时，CPU 发号施令的时间远大于 GPU 真正干活的时间。
#
# **2. Compute Bound (计算密集型) - 大矩阵乘法**
# 在 `matmul` (默认维度可能较大) 中，情况发生了逆转：
# *   **GPU (CUDA) 实际计算时间**: 1.037 ms
# *   **核心 Kernel**: `sm80_xmma_gemm...` (这是调用 Tensor Core 的矩阵乘法内核)
#
# GPU 的执行时间显著增加，达到了毫秒级。此时，GPU 的 Tensor Core 正在全速运转。这是一个接近 **Compute Bound** 的场景。只有在这种情况下，我们才真正利用了 GPU 的算力。
#
# **Latency Bound (延迟受限) - 小矩阵乘法**
# 再看 `matmul(dim=128)` 的数据：
# *   **GPU 时间**: 仅 9.730 us
# *   **Kernel Launch (CPU)**: 28.281 us (甚至比 GPU 计算还慢)
#
# 对于 128x128 这种小矩阵，GPU 的计算瞬间完成。此时的性能瓶颈再次回到了 Kernel 的启动延迟上。在深度学习中，如果模型中充斥着这种极小的矩阵运算（例如某些特殊的 Attention Head 或 MoE 路由），GPU 的大部分时间都在“等待 CPU 发指令”，导致利用率极低。

# %% [markdown]
# #### 剖析复合操作
# 接下来看看更复杂的操作，比如 `cdist` (计算距离)、`gelu` (激活函数) 和 `softmax`。

# %%
# 分析 cdist（计算距离）
cdist_function = lambda a, b: torch.cdist(a, b)
profile("cdist", run_operation2(dim=2048, operation=cdist_function))

# 分析 gelu
gelu_function = lambda a, b: torch.nn.functional.gelu(a + b)
profile("gelu", run_operation2(dim=2048, operation=gelu_function))

# 分析 softmax
softmax_function = lambda a, b: torch.nn.functional.softmax(a + b, dim=-1)
profile("softmax", run_operation2(dim=2048, operation=softmax_function))

# %% [markdown]
# #### 剖析完整的 MLP
# 最后，我们对完整的 MLP 训练步骤进行 Profiling。设置 `with_stack=True` 可以记录 Python 代码的调用栈，这对于生成火焰图（Flame Graph）非常有用，能帮我们定位到底是哪一行 Python 代码触发了耗时的 CUDA 操作。

# %%
# 分析完整的 MLP
if torch.cuda.is_available():
    # 使用较大的参数以获得更清晰的 GPU Profile
    profile("mlp", run_mlp(dim=2048, num_layers=64, batch_size=1024, num_steps=2), with_stack=True)
else:
    profile("mlp", run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=2), with_stack=True)

# %% [markdown]
# 在完整的 MLP 训练循环中，性能开销的构成变得更加复杂。Profiler 捕捉到了正向传播（Forward）和反向传播（Backward）的全过程。
#
# **反向传播开销显著**:
# `AddmmBackward0` (矩阵乘法梯度的计算) 占据了大量时间。这符合预期，因为反向传播通常需要计算两个梯度（对 Input 和对 Weight），计算量约为正向传播的 2 倍。
#
# **Kernel 类型的多样性**:
# 我们看到了三种不同的 GEMM (矩阵乘法) Kernel 变体：
#     *   `tn` (Transpose-Normal): 对应 $W^T \cdot \text{grad}$，用于计算 Input 的梯度。
#     *   `nn` (Normal-Normal): 对应 $X \cdot W$，用于正向传播。
#     *   `nt` (Normal-Transpose): 对应 $\text{grad} \cdot X^T$，用于计算 Weight 的梯度。
#
# **结论**: 训练过程不仅仅是重复正向计算，它触发了完全不同的计算路径和 Kernel 调用。

# %% [markdown]
# ## 3. Kernel Fusion 与 CUDA 编程
#
# ### 3.1 Kernel Fusion 的动机 (Motivation)
#
# 为了理解为什么我们需要优化：
#
# * **DRAM (显存)** 就像是一个巨大的**仓库 (Warehouse)**：容量大，但存取速度慢，且距离远。
# * **SRAM (计算单元附近的缓存/寄存器)** 就像是**工厂 (Factory)**：空间小，但处理速度极快。
#
# **瓶颈在哪里？**
# 通常不在于工厂的加工速度（计算能力），而在于卡车在仓库和工厂之间往返运输原材料（数据）的速度（**显存带宽**）。
#
# **什么是 Kernel Fusion？**
# * **未融合 (Unfused)**: 从仓库取货 -> 加工步骤A -> 运回仓库 -> 从仓库取货 -> 加工步骤B -> 运回仓库。
# * **融合 (Fused)**: 从仓库取货 -> 在工厂里一次性完成步骤A和B -> 运回仓库。
#
# 显然，融合操作通过减少对 DRAM 的读写次数，能显著提升性能。
#
# 让我们以 **GeLU (Gaussian Error Linear Unit)** 激活函数为例。它的数学公式包含多个步骤：乘法、加法、Tanh、立方等。
#
# $$\text{GeLU}(x) \approx 0.5 \cdot x \cdot (1 + \tanh(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)))$$

# %%
# === 定义两种 GeLU 实现 ===

def pytorch_gelu(x: torch.Tensor):
    """
    1. PyTorch 原生实现 (Fused)
    通常由 C++ 编写的高度优化内核，一次性完成计算。
    """
    return torch.nn.functional.gelu(x, approximate="tanh")

def manual_gelu(x: torch.Tensor):
    """
    2. 手动 Python 实现 (Unfused)
    这会触发多个独立的 Kernel（乘法、加法、Tanh 等），
    导致频繁的显存读写。
    """
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

# 验证正确性
print("Checking correctness...")
check_equal(pytorch_gelu, manual_gelu)
print("Pass!")

# %% [markdown]
# #### 性能对比：Fused vs Unfused
# 让我们对这两者进行 Benchmark。你会发现，尽管计算逻辑完全相同，但性能差异巨大。

# %%
print("\n=== Benchmarking GeLU Implementations ===")
dim = 16384 # 使用较大的维度以凸显带宽瓶颈

# 1. 测量 Python 手动实现
manual_time = benchmark("manual_gelu", run_operation1(dim=dim, operation=manual_gelu))

# 2. 测量 PyTorch 原生实现
pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=dim, operation=pytorch_gelu))

if manual_time and pytorch_time:
    speedup = manual_time / pytorch_time
    print(f"\nSpeedup: {speedup:.2f}x (Fused is faster)")

# %% [markdown]
# **Profiling 视角**：
# 如果你对 `manual_gelu` 运行 `profile()`，你会看到它启动了一长串细碎的 CUDA kernel（如 `elementwise_kernel`, `mul`, `add`, `tanh` 等）。而 `pytorch_gelu` 只会启动一个高效的 fused kernel。

# %%
# 观察 manual_gelu 产生的碎片化 Kernels
profile("manual_gelu", run_operation1(dim=dim, operation=manual_gelu))

# %% [markdown]
# ### 3.2 编写 CUDA Kernels (The "Hard" Way)
#
# 既然 PyTorch 的原生 Kernel 这么快，那它是怎么写的呢？它是用 C++ 和 CUDA 编写的。
#
# 为了理解底层原理，我们将尝试自己编写一个 CUDA 版本的 GeLU。
#
# #### CUDA 编程模型回顾
# 我们在写 Kernel 时，主要是在写一个**线程 (Thread)** 需要做的事情。我们需要利用坐标系统来确定当前线程处理哪个数据：
# * `gridDim`, `blockDim`: 网格和块的大小。
# * `blockIdx`, `threadIdx`: 当前块和线程的索引。
#
# 通常的索引计算公式：
# `int i = blockIdx.x * blockDim.x + threadIdx.x;`
#
# #### 使用 `load_inline` 即时编译
# PyTorch 提供了 `load_inline` 工具，允许我们在 Python 脚本中直接嵌入 C++/CUDA 源代码，并在运行时编译加载。这对于学习和调试非常方便。

# %%
# === 定义 CUDA 源代码 ===

cuda_source = """
#include <cuda_runtime.h>
#include <math.h>

// CUDA Kernel 函数: __global__ 表示它在 GPU 上运行，被 CPU 调用
__global__ void gelu_kernel(const float* x, float* y, int n) {
    // 1. 计算全局唯一的线程索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. 边界检查: 防止越界访问
    if (i < n) {
        float val = x[i];
        // 3. 计算 GeLU (与 Python 版本逻辑一致)
        float tanh_val = tanh(0.79788456f * (val + 0.044715f * val * val * val));
        y[i] = 0.5f * val * (1.0f + tanh_val);
    }
}

// C++ Wrapper 函数: 负责申请显存、计算 Grid/Block 大小并启动 Kernel
torch::Tensor gelu(torch::Tensor x) {
    // 确保输入是连续的 CUDA Tensor
    auto x_cont = x.contiguous();
    auto y = torch::empty_like(x_cont);
    
    int n = x_cont.numel();
    const int threads = 1024; // 每个 Block 的线程数 (通常设为 128, 256, 512, 1024)
    // 计算需要多少个 Block: (n + threads - 1) / threads 是向上取整
    const int blocks = (n + threads - 1) / threads;

    // 启动 Kernel <<<grid, block>>>
    gelu_kernel<<<blocks, threads>>>(x_cont.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}
"""

cpp_source = "torch::Tensor gelu(torch::Tensor x);"

# %% [markdown]
# 接下来，我们编译并加载这个自定义模块。

# %%
def create_cuda_gelu():
    """编译并返回自定义的 CUDA GeLU 函数"""
    if not torch.cuda.is_available():
        return None
    
    ensure_directory_exists("var/cuda_gelu")
    
    # 动态编译 CUDA 代码
    gelu_module = load_inline(
        cuda_sources=[cuda_source],
        cpp_sources=[cpp_source],
        functions=["gelu"],
        extra_cflags=["-O2"], # 开启 O2 优化
        verbose=True,
        name="inline_gelu_v1",
        build_directory="var/cuda_gelu"
    )
    return gelu_module.gelu

# 创建并测试
print("\nCompiling CUDA kernel...")
cuda_gelu_func = create_cuda_gelu()

if cuda_gelu_func:
    print("Compilation successful!")
    # 验证正确性
    check_equal(cuda_gelu_func, manual_gelu)
    print("CUDA implementation is correct.")

# %% [markdown]
# #### 性能对比：CUDA vs Manual vs PyTorch
# 现在我们有了三个版本：
# 1.  **Manual (Python)**: 慢，显存读写多。
# 2.  **CUDA (Custom)**: 我们刚写的 C++ 版本，应该比 Python 快。
# 3.  **PyTorch (Native)**: 官方高度优化的版本。
#
# 我们的简单 CUDA 实现能打败 Python，但能打败 PyTorch 官方吗？

# %%
if cuda_gelu_func:
    print("\n=== Benchmarking with CUDA Kernel ===")
    benchmark("manual_gelu", run_operation1(dim=dim, operation=manual_gelu))
    benchmark("pytorch_gelu", run_operation1(dim=dim, operation=pytorch_gelu))
    benchmark("cuda_gelu", run_operation1(dim=dim, operation=cuda_gelu_func))

    print("\n=== Profiling CUDA Kernel ===")
    profile("cuda_gelu", run_operation1(dim=dim, operation=cuda_gelu_func))

# %% [markdown]
# **结论**：
# * **CUDA vs Python**: 我们的 CUDA 实现（`cuda_gelu`）通常比 Python 手动版（`manual_gelu`）快得多，因为它只需要读写显存各一次（Fused）。
# * **CUDA vs PyTorch**: 我们的实现通常略慢于或接近PyTorch 原生版。为什么？因为 PyTorch 的实现可能包含了更多优化，比如向量化加载（Vectorized Loads，一次读取 float4）、更精细的指令级优化等。
#
# 虽然 CUDA 很快，但编写它很痛苦（C++ 语法、指针管理、编译链）。这就是为什么我们需要下一章的主角——**Triton**。

# %% [markdown]
# ## 4. Triton 编程与自动编译
#
# ### 4.1 为什么选择 Triton？
#
# **Triton** 是由 OpenAI 开发的一种语言和编译器，旨在让 GPU 编程变得触手可及。
#
# *   **核心理念**: 用 Python 编写代码，但用 **Blocks (分块)** 的思维去思考。
# *   **优势**: 编译器会自动处理复杂的内存合并访问 (Memory Coalescing) 和共享内存管理 (Shared Memory Management)，大大降低了高性能算子开发的门槛。
#
# **Triton vs CUDA 的核心区别**：
# 在 CUDA 中，我们必须从**线程 (Thread)** 的角度思考问题（我是哪个线程？我要处理哪个数据？）。
# 而在 Triton 中，我们从**块 (Block)** 的角度思考问题。
#
# Triton 编译器会自动帮我们要处理很多令人头疼的硬件优化：
#
# | 特性 | CUDA (手动) | Triton (自动) |
# | :--- | :--- | :--- |
# | **Memory Coalescing** (内存合并访问) | 需要手动规划 | **编译器自动处理** |
# | **Shared Memory Management** (共享内存管理) | 需要手动分配和同步 | **编译器自动处理** |
# | **Scheduling within SMs** (SM 内部调度) | 手动 | **编译器自动处理** |
# | **Scheduling across SMs** (跨 SM 调度) | 手动 | 手动 |
#
# 这意味着用 Python 写的 Triton 代码，经过编译后，性能往往能媲美甚至超越手写的 CUDA 代码。

# %% [markdown]
# ### 4.2 编写 Triton Kernel：以 GeLU 为例
#
# 让我们用 Triton 重写 GeLU。你会发现代码结构非常像 Python，但引入了一些特殊的指针运算。
#
# **核心概念**：
# 1.  **装饰器 `@triton.jit`**: 告诉 Triton 这是一个需要编译并在 GPU 上运行的内核。
# 2.  **`tl.program_id(axis=0)`**: 获取当前 Block 的 ID。
# 3.  **指针运算**: 在 Triton 中，我们通过 `基地址 + 偏移量` 来生成指针向量。
# 4.  **Mask (掩码)**: 因为 Block 大小是固定的（例如 1024），处理边界时（例如数组长度不是 1024 的倍数）需要 Mask 来防止越界读写。

# %%
# 确保导入 triton
import triton
import triton.language as tl

@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton Kernel: 处理一个 Block 的数据
    """
    # 1. 获取当前 Block 的 ID
    pid = tl.program_id(axis=0)
    
    # 2. 计算当前 Block 的起始位置
    block_start = pid * BLOCK_SIZE
    
    # 3. 生成当前 Block 内部所有线程的偏移量 (0, 1, ..., BLOCK_SIZE-1)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 4. 创建 Mask: 只有在 num_elements 范围内的才处理
    mask = offsets < num_elements

    # 5. 读取数据 (Load)
    # 注意：这里 x_ptr + offsets 生成了一组指针，tl.load 会并行读取它们
    x = tl.load(x_ptr + offsets, mask=mask)

    # 6. 计算 GeLU (数学公式)
    # Triton 提供了标准的数学函数，如 tl.exp, tl.tanh (或者手动实现 tanh)
    # 近似公式: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # 7. 写回数据 (Store)
    tl.store(y_ptr + offsets, y, mask=mask)

def triton_gelu(x: torch.Tensor):
    """
    Python Wrapper: 负责准备 Tensor 和计算 Grid 大小，然后启动 Kernel
    """
    assert x.is_cuda
    # 确保内存连续，否则指针运算会出错
    x = x.contiguous() 

    # 分配输出空间
    y = torch.empty_like(x)

    # 确定 Grid 大小 (把数据切分成多少个 Block)
    num_elements = x.numel()
    block_size = 1024  # 自定义 Block 大小
    # triton.cdiv 是向上取整除法
    num_blocks = triton.cdiv(num_elements, block_size)

    # 启动 Kernel: grid 必须是 tuple
    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)

    return y

# %% [markdown]
# ### 4.3 深入底层：查看 PTX 汇编
#
# Triton 到底做了什么？它是解释器吗？
# 不，Triton 是一个**编译器**。它将 Python AST 解析并编译成 PTX（Parallel Thread Execution，即 GPU 汇编语言），最终交给 GPU 执行。
#
# 我们可以通过 inspecting 编译后的 Kernel 来查看生成的 PTX 代码。这能让我们确认编译器是否执行了预期的优化（例如向量化读取）。

# %%
def print_ptx(name: str, kernel):
    """打印 Triton Kernel 生成的 PTX 代码"""
    # 如果设置了 TRITON_INTERPRET=1 (调试模式)，则不会生成 PTX
    if os.environ.get("TRITON_INTERPRET") == "1":
        print("PTX is not generated when in interpret mode.")
        return

    # 强制触发一次编译以生成 Cache
    x = torch.randn(1024, device=get_device())
    y = torch.empty_like(x)
    kernel[(1,)](x, y, 1024, BLOCK_SIZE=1024)
    
    # 从 Triton 内部缓存中获取 PTX
    # 注意：这里访问内部 API 的方式可能随 Triton 版本变化
    try:
        kernel_cache = list(kernel.cache[0].values())[0]
        ptx_code = kernel_cache.asm["ptx"]
        print(f"\n=== PTX Code for {name} (Snippet) ===")
        print(ptx_code[:1000] + "\n... (truncated) ...") # 只打印前1000字符
    except Exception as e:
        print(f"Could not retrieve PTX: {e}")

if torch.cuda.is_available():
    print_ptx("triton_gelu", triton_gelu_kernel)

# %% [markdown]
# **观察 PTX 代码**:
# 如果你看到 `ld.global.v4.f32` 这样的指令，说明 Triton 自动帮我们将内存读取进行了向量化（一次读取 4 个 float），这是极其重要的性能优化，而在 CUDA 中我们需要手动通过 `float4` 类型来实现。
#
# ### 4.4 性能对决：Triton vs The World
#
# 现在我们有了四种 GeLU 实现，是时候一决高下了：
# 1.  **Manual (Python)**: 纯 PyTorch 操作符堆叠。
# 2.  **PyTorch (Native)**: 官方 C++ 实现。
# 3.  **CUDA (Custom)**: 我们上一章写的 C++ 实现。
# 4.  **Triton**: 刚写的 Python-like Kernel。

# %%
def benchmark_triton_suite():
    if not torch.cuda.is_available():
        return

    print("\n=== Benchmarking All GeLU Implementations ===")
    dim = 16384
    
    # 验证正确性
    print("Checking Triton correctness...")
    check_equal(triton_gelu, manual_gelu)
    print("Pass!")

    # 1. Manual
    benchmark("manual_gelu", run_operation1(dim, manual_gelu))
    
    # 2. PyTorch Native
    benchmark("pytorch_gelu", run_operation1(dim, pytorch_gelu))
    
    # 3. CUDA (如果上一章编译成功)
    try:
        cuda_gelu = create_cuda_gelu()
        if cuda_gelu:
            benchmark("cuda_gelu", run_operation1(dim, cuda_gelu))
    except Exception as e:
        print(f"CUDA benchmark skipped: {e}")

    # 4. Triton
    benchmark("triton_gelu", run_operation1(dim, triton_gelu))
    
    # Profiling
    profile("triton_gelu", run_operation1(dim, triton_gelu))

benchmark_triton_suite()

# %% [markdown]
# 通常你会发现：
# * **Triton ≈ PyTorch Native**: Triton 的性能非常接近官方高度优化的 C++ 实现。
# * **Triton vs CUDA**: 有时 Triton 甚至比我们要写几十行代码的“朴素” CUDA Kernel 还要快，因为它自动应用了 Thread Coarsening（线程粗化）等优化策略。
#
# ### 4.5 PyTorch 2.0 与 torch.compile
#
# 既然 Triton 这么好用，PyTorch 团队想：为什么不自动把用户的 Python 代码转换成 Triton 呢？
#
# 这就是 `torch.compile` 的由来。你不需要显式地写 Triton Kernel，只需要写普通的 PyTorch 代码，然后调用 `torch.compile`，PyTorch 就会在后台分析计算图，将其融合（Fusion）并生成高效的 Triton 代码。

# %%
def pytorch_compilation_demo():
    print("\n=== PyTorch 2.0 Compilation Demo ===")
    
    # 1. 拿回我们最慢的 Python 手动实现
    model = manual_gelu
    
    # 2. 一行代码进行编译！
    # 这会触发 TorchDynamo 分析图，Inductor 生成 Triton 代码
    compiled_gelu = torch.compile(model)

    print("Checking correctness of compiled model...")
    check_equal(compiled_gelu, manual_gelu)
    
    if not torch.cuda.is_available():
        return

    # 3. Benchmark
    # 第一次运行会触发编译（较慢），Benchmark 函数会自动 Warmup 跳过它
    benchmark("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu))
    
    # 4. Profile
    # 你会在 Profile 结果中看到类似 "triton_" 开头的 Kernel 名字
    profile("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu))

pytorch_compilation_demo()

# %% [markdown]
# **总结**：
# * **Manual**: 慢，适合原型开发。
# * **PyTorch Native**: 快，但修改底层逻辑很难。
# * **CUDA**: 最快且最灵活，但开发效率极低，门槛高。
# * **Triton**: 性能接近 CUDA，开发效率接近 Python。
# * **torch.compile**: **未来的默认选择**。它让你写 Python，却能享受到 Triton 的性能。

# %% [markdown]
# ## 5. 进阶 Triton 计算：Softmax 与矩阵乘法
#
# ### 5.1 Softmax：处理聚合操作 (Aggregation)
#
# Softmax 是注意力机制（Attention）的核心组件。它的数学形式是对矩阵的每一行进行归一化：
#
# $$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$
#
# **计算挑战**：
# 为了计算一行的 Softmax，我们需要：
# 1.  找到该行的最大值 $x_{max}$（为了数值稳定性，通常减去最大值）。
# 2.  计算所有元素的指数 $e^{x - x_{max}}$。
# 3.  计算指数之和 $\sum e^{...}$。
# 4.  每个元素除以和。
#
# 这意味着线程之间需要共享信息（比如行的总和）。
#
# #### Python 朴素实现
# 让我们先看一个简单的 PyTorch 实现。虽然代码很短，但它效率极低，因为它需要多次读写 DRAM（计算 Max -> 写回 -> 减去 -> 写回 -> Exp -> ...）。

# %%
def manual_softmax(x: torch.Tensor):
    """
    朴素的 Softmax 实现。
    
    性能瓶颈：
    多次扫描内存 (Reads/Writes)。
    总共大约需要 5MN + M 次读, 3MN + 2M 次写。
    """
    # x: (M, N)
    # 1. 计算每行的 Max (MN reads, M writes)
    x_max = x.max(dim=1)[0]

    # 2. 减去 Max (MN + M reads, MN writes)
    x_sub = x - x_max[:, None]

    # 3. 指数运算 (MN reads, MN writes)
    numerator = torch.exp(x_sub)

    # 4. 计算分母 Sum (MN reads, M writes)
    denominator = numerator.sum(dim=1)

    # 5. 除法归一化 (MN reads, MN writes)
    y = numerator / denominator[:, None]

    return y

# 测试数据
x_sample = torch.tensor([[5., 5, 5], [0, 0, 100]], device=get_device())
print("Manual Softmax Result:\n", manual_softmax(x_sample))

# %% [markdown]
# #### Triton Fused Softmax
# 在 Triton 中，我们可以将上述所有步骤融合到一个 Kernel 中。
#
# **策略**：
# * **Grid 设置**：我们将 Grid 设置为 `(M,)`，即**每一行分配给一个 Program (Block)**。
# * **Block 内部**：每个 Block 负责加载一整行的数据，在片上内存（SRAM）中完成 Max、Exp、Sum 的计算，最后只写回一次结果。
# * **`tl.max`, `tl.sum`**: Triton 提供了特定的 API 来在 Block 内部进行归约操作。

# %%
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    """
    Triton Softmax Kernel
    假设：BLOCK_SIZE 大于等于 num_cols (列数)，即一个 Block 能装下一整行。
    """
    # 1. 获取当前 Program 处理的是哪一行
    row_idx = tl.program_id(0)
    
    # 2. 生成列偏移量 (0, 1, ..., BLOCK_SIZE-1)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 3. 计算内存地址并加载数据
    # x_ptr 是基地址，row_idx * stride 定位到行首，col_offsets 定位到具体元素
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    
    # Mask 处理：如果列数小于 BLOCK_SIZE，超出部分填充 -inf (不影响 max)
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # 4. 计算 (都在 SRAM 中完成，无需读写 DRAM)
    # 找到最大值
    row_max = tl.max(x_row, axis=0)
    # 减去最大值并做指数
    numerator = tl.exp(x_row - row_max)
    # 求和
    denominator = tl.sum(numerator, axis=0)
    # 除法
    y_row = numerator / denominator

    # 5. 写回结果
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)


def triton_softmax(x: torch.Tensor):
    """Triton Softmax Wrapper"""
    y = torch.empty_like(x)
    M, N = x.shape
    
    # Block Size 设为大于 N 的最小 2 的幂次
    block_size = triton.next_power_of_2(N)
    
    # 启动 Kernel: 共有 M 个 Block，每个 Block 处理一行
    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )
    return y

# 验证正确性
print("\nChecking Triton Softmax correctness...")
if torch.cuda.is_available():
    check_equal2(lambda x: torch.nn.functional.softmax(x, dim=-1), triton_softmax)
    print("Pass!")

# %% [markdown]
# #### 性能对比
# 我们将对比以下实现：
# 1.  **Manual**: 朴素 Python 实现。
# 2.  **Compiled**: `torch.compile(manual_softmax)`。
# 3.  **PyTorch Native**: `torch.nn.functional.softmax`。
# 4.  **Triton**: 我们自定义的 Kernel。
#
# 注意观察：Triton 自定义实现和 `torch.compile` 生成的代码通常比朴素实现快得多，甚至能接近 PyTorch 的原生 C++ 实现。

# %%
def benchmark_softmax():
    if not torch.cuda.is_available():
        return

    print("\n=== Benchmarking Softmax ===")
    dim = 4096 # 矩阵大小 4096 x 4096
    
    benchmark("manual_softmax", run_operation1(dim, manual_softmax))
    
    compiled_softmax = torch.compile(manual_softmax)
    benchmark("compiled_softmax", run_operation1(dim, compiled_softmax))
    
    benchmark("pytorch_softmax", run_operation1(dim, lambda x: torch.nn.functional.softmax(x, dim=-1)))
    
    benchmark("triton_softmax", run_operation1(dim, triton_softmax))

benchmark_softmax()

# %% [markdown]
# ### 5.2 矩阵乘法：Tiling 与 存储层次 (Theory)
#
# 矩阵乘法 (MatMul) 是深度学习中计算量最大的操作。虽然我们在本节不直接编写 MatMul 的代码（因为逻辑较复杂），但理解其背后的优化原理至关重要。
#
# #### 1. 朴素实现的瓶颈
# 计算 $C = A \times B$。
# 对于 $C$ 中的每一个元素，我们需要读取 $A$ 的一行和 $B$ 的一列。
# * 如果 $A, B, C$ 都是 $N \times N$。
# * 读写总量大约是 $O(N^3)$ (因为每个点都要重新读)。
# * 这极度浪费显存带宽。
#
# #### 2. Tiling (分块/平铺)
# 优化核心思想：**利用 Shared Memory (共享内存)**。
#
# 我们可以将 $A$ 和 $B$ 切分成小的块 (Block/Tile)。
# * 将 $A$ 的一个小块和 $B$ 的一个小块加载到 **Shared Memory** 中。
# * Shared Memory 比显存快 10 倍以上。
# * 在 Shared Memory 上进行多次计算，直到用完这些数据。
# * **效果**：数据一旦从 DRAM 读进 SRAM，就被反复利用，极大地提高了**算术强度 (Arithmetic Intensity)**。
#
#
# #### 3. L2 Cache 优化 (Swizzling)
# 除了 Shared Memory，Triton 还会自动帮我们优化 **L2 Cache** 的命中率。
# * 如果我们按顺序处理 Block (行优先)，可能会导致访问 $B$ 矩阵时跳跃太大，导致 Cache Miss。
# * Triton 会按照一种特殊的顺序（Grouped / Swizzled）来调度 Block，使得内存访问局部性更好。
#
# ### 总结 (Summary)
#
# 到此为止，关于CUDA的核心内容已经介绍完毕。回顾一下我们学到的：
#
# 1.  **性能鸿沟**：Python/PyTorch 编程模型与底层硬件执行模型之间存在巨大差异。
# 2.  **测量**：Benchmark 测宏观时间，Profile 看微观内核。
# 3.  **Kernel Fusion**：减少显存读写是提升性能的关键（仓库 vs 工厂）。
# 4.  **Triton**：让我们能用 Python 编写 Block 级别的 GPU 代码，编译器自动处理了最难的内存合并与共享内存管理。
# 5.  **未来**：随着 `torch.compile` 的成熟，大部分时候我们不需要手写 Kernel，编译器会帮我们生成高效的 Triton 代码。
