#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os

# 如果当前目录是 'notebook'，则切换到父目录（项目根目录）
# 这样可以确保能够正确访问项目根目录下的 'var' 等文件夹
if os.path.basename(os.getcwd()) == 'notebook':
    os.chdir('..')

# 将当前目录添加到 path 中，以便可以导入根目录下的模块
sys.path.append(os.getcwd())

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import timeit
from jaxtyping import Float
from einops import rearrange, einsum, reduce

from facts import a100_flop_per_sec, h100_flop_per_sec
from references import zero_2019

def get_memory_usage(x: torch.Tensor):
    return x.numel() * x.element_size()

def same_storage(x: torch.Tensor, y: torch.Tensor):
    return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

def time_matmul(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the number of seconds required to perform `a @ b`."""

    # Wait until previous CUDA threads are done
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def run():
        # Perform the operation
        a @ b

        # Wait until CUDA threads are done
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Time the operation `num_trials` times
    num_trials = 5
    total_time = timeit.timeit(run, number=num_trials)

    return total_time / num_trials

def get_promised_flop_per_sec(device: str, dtype: torch.dtype) -> float:
    """Return the peak FLOP/s for `device` operating on `dtype`."""
    if not torch.cuda.is_available():
        print("No CUDA device available, so can't get FLOP/s.")
        return 1
    properties = torch.cuda.get_device_properties(device)

    if "A100" in properties.name or "A800" in properties.name:
        # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf")
        if dtype == torch.float32:
            return 19.5e12
        if dtype in (torch.bfloat16, torch.float16):
            return 312e12
        raise ValueError(f"Unknown dtype: {dtype}")

    if "H100" in properties.name or "H800" in properties.name:
        # https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")
        if dtype == torch.float32:
            return 67.5e12
        if dtype in (torch.bfloat16, torch.float16):
            return 1979e12 / 2  # 1979 is for sparse, dense is half of that
        raise ValueError(f"Unknown dtype: {dtype}")

    raise ValueError(f"Unknown device: {properties.name}")


# # 第二讲：Tensor, Model
# 
# 本讲概述：
# - 我们将讨论训练模型所需的所有**原语（primitives）**。
# - 我们将自底向上，从张量（tensors）到模型，再到优化器，最后是训练循环。
# - 我们将密切关注效率（**资源**的使用）。
# 
# 具体来说，我们会计算两类资源：
# - 显存 (Memory, GB)
# - 计算量 (Compute, FLOPs)
# 
# 让我们做一些简单的估算（napkin math）。
# 
# **问题**：在 1024 张 H100 上训练一个 70B 参数的模型（15T token）需要多久？

# 这里的 $6$ 是一个经验法则：**每个 Token 每个参数大约需要 6 次浮点运算**。
# 来源拆解（基于线性层 $Y=XW$）：
# 1. **前向传播 (Forward Pass)**:
#    - 计算 $Y = XW$（矩阵乘法）。
#    - 每次乘加运算算 2 次 FLOPs。
#    - 总计：$2 \times N \times D$ (参数量 $N$, Token数 $D$)。
# 2. **反向传播 (Backward Pass)**:
#    - 计算权重梯度 $\nabla W = X^T \nabla Y$。($2 \times N \times D$)
#    - 计算输入梯度 $\nabla X = \nabla Y W^T$。($2 \times N \times D$)
#    - 总计：$4 \times N \times D$。
# 
# **Total FLOPs** $\approx 2 + 4 = 6 \times N \times D$。

# In[58]:


total_flops = 6 * 70e9 * 15e12  # total_flops
assert h100_flop_per_sec == 1979e12 / 2
mfu = 0.5
flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24  # flops_per_day
days = total_flops / flops_per_day  # days
print(f"Days: {days}")


# **问题**：使用 8 张 H100 进行 AdamW 训练（朴素方法），最大能训练多大的模型？

# In[59]:


h100_bytes = 80e9  # h100_bytes
bytes_per_parameter = 4 + 4 + (4 + 4)  # parameters, gradients, optimizer state  bytes_per_parameter
num_parameters = (h100_bytes * 8) / bytes_per_parameter  # num_parameters
print(f"Num parameters: {num_parameters}")


# 注 1：我们这里朴素地使用了 float32 来存储参数和梯度。我们也可以使用 bf16 来存储参数和梯度（2+2 字节），并保留一份 float32 的参数副本（4 字节）。这虽然不能节省显存，但速度更快。[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054)
# 
# 注 2：这里没有计算激活值（activations）占用的显存（这取决于 batch size 和序列长度）。
# 
# 
# 我们不会详细讲解 Transformer 架构。
# 有很多优秀的资料：
# - [Assignment 1 handout](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf)
# - [Mathematical description](https://johnthickstun.com/docs/transformers.pdf)
# - [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
# - [Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
# 
# 相反，我们将使用更简单的模型。
# 
# 本讲的核心收获：
# - 机制：直截了当（就是 PyTorch）
# - 思维方式：资源计算（记得要做这件事）
# - 直觉：大方向（不涉及大模型）

# ## 显存计算 (Memory accounting)
# 
# 张量（Tensors）是存储所有内容的基本构建块：参数、梯度、优化器状态、数据、激活值。
# - [[PyTorch docs on tensors]](https://pytorch.org/docs/stable/tensors.html)
# 
# 你可以通过多种方式创建张量：

# In[60]:


x = torch.tensor([[1., 2, 3], [4, 5, 6]])
print(x)
x = torch.zeros(4, 8)  # 4x8 matrix of all zeros
print(x)
x = torch.ones(4, 8)  # 4x8 matrix of all ones
print(x)
x = torch.randn(4, 8)  # 4x8 matrix of iid Normal(0, 1) samples
print(x)


# 分配但不初始化值：

# In[61]:


x = torch.empty(4, 8)  # 4x8 matrix of uninitialized values
print(x)
# ...因为你可能想稍后使用自定义逻辑来设置这些值
nn.init.trunc_normal_(x, mean=0, std=1, a=-2, b=2)
print(x)


# 几乎所有东西（参数、梯度、激活值、优化器状态）都存储为浮点数。
# 
# ### float32
# - [[Wikipedia]](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)
# 
# <div align="center">
# <img src="../images/fp32.png" width="600">
# <br/>
# <i>float32</i>
# </div>
# 
# float32 数据类型（也称为 fp32 或单精度）是默认设置。
# 
# 传统上，在科学计算中，float32 是基准；在某些情况下你会使用双精度（float64）。
# 
# 但在深度学习中，你可以稍微“草率”一点。
# 
# 让我们检查一下这些张量的显存占用。
# 
# 显存占用取决于 (i) 值的数量 和 (ii) 每个值的数据类型。

# In[62]:


x = torch.zeros(4, 8)
print(x)
assert x.dtype == torch.float32  # Default type
assert x.numel() == 4 * 8
assert x.element_size() == 4  # Float is 4 bytes
assert get_memory_usage(x) == 4 * 8 * 4  # 128 bytes
print(get_memory_usage(x))


# GPT-3 前馈层中的一个矩阵：

# In[6]:


print(f"{get_memory_usage(torch.empty(12288 * 4, 12288)) / 1024 / 1024 / 1024} GB") # 2304 * 1024 * 1024  # 2.3 GB


# ...这确实很大！
# 
# ### float16
# - [[Wikipedia]](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)
# 
# <div align="center">
# <img src="../images/fp16.png" width="400">
# <br/>
# <i>float16</i>
# </div>
# 
# float16 数据类型（也称为 fp16 或半精度）减少了显存占用。

# In[64]:


x = torch.zeros(4, 8, dtype=torch.float16)
print(x)
assert x.element_size() == 2


# 然而，它的动态范围（特别是对于小数值）并不是很好。

# In[65]:


x = torch.tensor([1e-8], dtype=torch.float16)
print(x)
assert x == 0  # Underflow!


# 如果你在训练时遇到这种情况，会导致不稳定。
# 
# ### bfloat16
# - [[Wikipedia]](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
# 
# <div align="center">
# <img src="../images/bf16.png" width="400">
# <br/>
# <i>bfloat16</i>
# </div>
# 
# Google Brain 在 2018 年开发了 bfloat（brain floating point）来解决这个问题。
# 
# bfloat16 使用与 float16 相同的显存，但具有与 float32 相同的动态范围！
# 
# 唯一的缺点是分辨率较差，但这对于深度学习来说影响较小。

# In[66]:


x = torch.tensor([1e-8], dtype=torch.bfloat16)
print(x)
assert x != 0  # No underflow!


# 让我们比较不同数据类型的动态范围和显存占用：

# In[67]:


float32_info = torch.finfo(torch.float32)
print("float32:", float32_info)
float16_info = torch.finfo(torch.float16)
print("float16:", float16_info)
bfloat16_info = torch.finfo(torch.bfloat16)
print("bfloat16:", bfloat16_info)


# ### fp8
# 
# 2022 年，受机器学习工作负载的推动，FP8 被标准化。
# - [Link](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
# 
# <div align="center">
# <img src="https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/fp8_formats.png" width="400">
# <br/>
# <i>fp8 formats</i>
# </div>
# 
# H100 支持两种 FP8 变体：E4M3（范围 [-448, 448]）和 E5M2（范围 [-57344, 57344]）。
# - Reference: [FP8 Formats for Deep Learning](https://arxiv.org/pdf/2209.05433.pdf)
# 
# 对训练的影响：
# - 使用 float32 训练可行，但需要大量显存。
# - 使用 fp8、float16 甚至 bfloat16 训练有风险，可能会导致不稳定。
# - 解决方案（稍后讨论）：使用混合精度训练。

# ## 计算量计算 (Compute accounting)
# 
# 默认情况下，张量存储在 CPU 内存中。

# In[68]:


x = torch.zeros(32, 32)
assert x.device == torch.device("cpu")

# 然而，为了利用 GPU 的大规模并行性，我们需要将它们移动到 GPU 显存中。

# 让我们先看看是否有 GPU 可用。
if not torch.cuda.is_available():
    print("No GPU available")
else:
    num_gpus = torch.cuda.device_count()
    print(f"Num GPUs: {num_gpus}")
    for i in range(num_gpus):
        properties = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {properties.name}")

    memory_allocated = torch.cuda.memory_allocated()

    # Move the tensor to GPU memory (device 0).
    y = x.to("cuda:0")
    assert y.device == torch.device("cuda", 0)

    # Or create a tensor directly on the GPU:
    z = torch.zeros(32, 32, device="cuda:0")

    new_memory_allocated = torch.cuda.memory_allocated()
    memory_used = new_memory_allocated - memory_allocated
    assert memory_used == 2 * (32 * 32 * 4)  # 2 32x32 matrices of 4-byte floats
    print(f"Memory used: {memory_used} bytes")


# <div align="center">
# <img src="../images/cpu-gpu.png" width="400">
# <br/>
# <i>CPU vs GPU</i>
# </div>

# 大多数张量是通过对其他张量执行操作创建的。
# 每个操作都会产生一定的显存和计算后果。
# 
# PyTorch 中的张量是什么？
# PyTorch 张量是指向已分配内存的指针，并带有描述如何获取张量任何元素的元数据。
# - [[PyTorch docs]](https://pytorch.org/docs/stable/generated/torch.Tensor.stride.html)
# 
# <div align="center">
# <img src="https://martinlwx.github.io/img/2D_tensor_strides.png" width="400">
# <br/>
# <i>Tensor Strides</i>
# </div>

# In[ ]:


x = torch.tensor([
    [0., 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
])

# To go to the next row (dim 0), skip 4 elements in storage.
# 要转到下一行（维度 0），在存储中跳过 4 个元素。
assert x.stride(0) == 4

# To go to the next column (dim 1), skip 1 element in storage.
# 要转到下一列（维度 1），在存储中跳过 1 个元素。
assert x.stride(1) == 1

# To find an element:
# 查找元素：
r, c = 1, 2
index = r * x.stride(0) + c * x.stride(1)
print(f"Index: {index}")
assert index == 6


# 许多操作只是提供张量的不同**视图 (view)**。
# 这不会进行复制，因此一个张量中的突变会影响另一个张量。

# In[ ]:


x = torch.tensor([[1., 2, 3], [4, 5, 6]])
print("Original x:", x)

# Get row 0:
# 获取第 0 行：
y = x[0]
print("Row 0:", y)
assert torch.equal(y, torch.tensor([1., 2, 3]))
assert same_storage(x, y)

# Get column 1:
# 获取第 1 列：
y = x[:, 1]
print("Col 1:", y)
assert torch.equal(y, torch.tensor([2, 5]))
assert same_storage(x, y)

# View 2x3 matrix as 3x2 matrix:
# 将 2x3 矩阵视为 3x2 矩阵：
y = x.view(3, 2)
print("View 3x2:", y)
assert torch.equal(y, torch.tensor([[1, 2], [3, 4], [5, 6]]))
assert same_storage(x, y)

# Transpose the matrix:
# 转置矩阵：
y = x.transpose(1, 0)
print("Transposed:", y)
assert torch.equal(y, torch.tensor([[1, 4], [2, 5], [3, 6]]))
assert same_storage(x, y)

# Check that mutating x also mutates y.
# 检查修改 x 是否也会修改 y。
x[0][0] = 100
print("Mutated x:", x)
print("Affected y:", y)
assert y[0][0] == 100


# 请注意，某些视图是不连续的条目，这意味着无法进行进一步的视图操作。
# 可以先强制让张量在内存上变得连续 (contiguous)。

# In[ ]:


x = torch.tensor([[1., 2, 3], [4, 5, 6]])
y = x.transpose(1, 0)
assert not y.is_contiguous()
try:
    y.view(2, 3)
    assert False
except RuntimeError as e:
    assert "view size is not compatible with input tensor's size and stride" in str(e)

y = x.transpose(1, 0).contiguous().view(2, 3)
print("Contiguous view:", y)
assert not same_storage(x, y)


# 视图是免费的，复制则需要（额外的）显存和计算。
# 
# 这些操作对张量的每个元素应用某种运算，并返回相同形状的（新）张量。

# In[ ]:


x = torch.tensor([1, 4, 9])
assert torch.equal(x.pow(2), torch.tensor([1, 16, 81]))
assert torch.equal(x.sqrt(), torch.tensor([1, 2, 3]))
assert torch.equal(x.rsqrt(), torch.tensor([1, 1 / 2, 1 / 3]))  # i -> 1/sqrt(x_i)

assert torch.equal(x + x, torch.tensor([2, 8, 18]))
assert torch.equal(x * 2, torch.tensor([2, 8, 18]))
assert torch.equal(x / 0.5, torch.tensor([2, 8, 18]))

# `triu` takes the upper triangular part of a matrix.
# `triu` 获取矩阵的上三角部分。
x = torch.ones(3, 3).triu()
print("Upper triangular:", x)
assert torch.equal(x, torch.tensor([
    [1, 1, 1],
    [0, 1, 1],
    [0, 0, 1]],
))
# This is useful for computing an causal attention mask, where M[i, j] is the contribution of i to j.
# 这对于计算因果注意力掩码非常有用，其中 M[i, j] 是 i 对 j 的贡献。


# 最后，深度学习的“面包和黄油”：矩阵乘法。

# In[ ]:


x = torch.ones(16, 32)
w = torch.ones(32, 2)
y = x @ w
assert y.size() == torch.Size([16, 2])


# 通常，我们对批次中的每个示例和序列中的每个标记执行操作。
# 
# <div align="center">
# <img src="../images/batch-sequence.png" width="400">
# <br/>
# <i>Batch Sequence</i>
# </div>

# In[ ]:


x = torch.ones(4, 8, 16, 32)
w = torch.ones(32, 2)
y = x @ w
assert y.size() == torch.Size([4, 8, 16, 2])
# In this case, we iterate over values of the first 2 dimensions of `x` and multiply by `w`.
# 在这种情况下，我们遍历 `x` 前两个维度的值并乘以 `w`。


# 在 PyTorch 里写多维张量运算时，最容易踩坑的就是“我到底在换哪两个维度”。比如下面这个注意力里常见的写法，`transpose(-2, -1)` 并不直观：
# 
# ```python
# x = torch.ones(2, 2, 3)  # batch, seq, hidden
# y = torch.ones(2, 2, 3)  # batch, seq, hidden
# z = x @ y.transpose(-2, -1)  # batch, seq, seq
# # -2/-1 指倒数第二/第一维；代码一长就很容易看错
# ```
# 
# `einops` 提供了一套更“按名字写公式”的接口：你用字符串把每个维度标出来，库再据此完成变形、相乘、归约等操作。它的设计灵感来自爱因斯坦求和约定（Einstein summation notation）。
# 
# - [[Einops tutorial]](https://einops.rocks/1-einops-basics/)

# 
# 
# 爱因斯坦求和约定的规则其实很朴素：**同一个索引在一个项里出现两次，就默认对它求和**；只出现一次的索引，会保留在结果里。
# 
# 举个简单例子，假设我们有两个向量 $\mathbf{a}$ 和 $\mathbf{b}$：
# 
# $$ \mathbf{a} = [a_1, a_2, \dots, a_n], \quad \mathbf{b} = [b_1, b_2, \dots, b_n] $$
# 
# 它们的点积（dot product）通常写成：
# 
# $$ \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i $$
# 
# 在**爱因斯坦约定（Einstein Summation Convention）**下，可以省略求和号，写成：
# 
# $$ a_i b_i $$
# 
# 因为索引 $i$ 出现了两次，所以默认对 $i$ 进行求和。
# 
# 矩阵乘法也同理。对于 $\mathbf{C} = \mathbf{A}\mathbf{B}$，其中 $\mathbf{A} \in \mathbb{R}^{m \times k}$、$\mathbf{B} \in \mathbb{R}^{k \times n}$，传统写法是：
# 
# $$ C_{ij} = \sum_k A_{ik} B_{kj} $$
# 
# 爱因斯坦写法则是：
# 
# $$ C_{ij} = A_{ik} B_{kj} $$
# 
# 这里索引 $k$ 出现两次，所以被自动求和；而 $i$ 和 $j$ 各出现一次，因此出现在最终结果中。
# 
# 这套记法在深度学习里尤其有用：我们经常处理 `batch` / `sequence` / `head` / `hidden` 等高维张量，单靠 `.matmul()` 或 `.transpose()` 往往很难直观看出“哪些维度在对齐、哪些维度在合并”。而 `einsum` 就是把这条规则搬到了代码里：你通过字符串把维度关系写清楚，剩下的交给计算引擎去实现，代码的可读性和逻辑严密性都会显著提升。

# 那怎么更稳地“记住”每个维度分别是什么？最常见的做法是写注释；更进一步，可以把维度也写进类型提示里。
# 
# **传统写法（靠注释）**
# 
# ```python
# x = torch.ones(2, 2, 1, 3)  # batch seq heads hidden
# ```
# 
# **`jaxtyping` 的写法（把维度写到类型里）**
# 
# ```python
# x: Float[torch.Tensor, "batch seq heads hidden"] = torch.ones(2, 2, 1, 3)
# ```
# 
# 这类标注主要用于可读性与静态提示，本身不会在运行时强制检查维度是否匹配。

# 
# 
# 下面用几个小例子把传统写法和 `einops.einsum` 对比一下。

# In[ ]:


# Define two tensors:
# 定义两个张量：
x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)
y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)

# Old way:
# 旧方法：
z = x @ y.transpose(-2, -1)  # batch, sequence, sequence
print("Old way shape:", z.shape)

# New (einops) way:
# 新 (einops) 方法：
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
print("Einops way shape:", z.shape)

# Or can use `...` to represent broadcasting over any number of dimensions:
# 或者可以使用 `...` 表示对任意数量的维度进行广播：
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
print("Broadcasting shape:", z.shape)


# 你可以通过某些操作（例如 sum, mean, max, min）对单个张量进行归约 (reduce)。

# In[ ]:


x: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)

# Old way:
# 旧方法：
y = x.sum(dim=-1)
print("Old reduce shape:", y.shape)

# New (einops) way:
# 新 (einops) 方法：
y = reduce(x, "... hidden -> ...", "sum")
print("Einops reduce shape:", y.shape)


# 有时，一个维度代表两个维度，你想对其中一个进行操作。

# In[ ]:


x: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)
# ...where `total_hidden` is a flattened representation of `heads * hidden1`
# ...其中 `total_hidden` 是 `heads * hidden1` 的扁平化表示
w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)

# Break up `total_hidden` into two dimensions (`heads` and `hidden1`):
# 将 `total_hidden` 分解为两个维度（`heads` 和 `hidden1`）：
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
print("Rearranged x shape:", x.shape)

# Perform the transformation by `w`:
# 通过 `w` 执行变换：
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")
print("Transformed x shape:", x.shape)

# Combine `heads` and `hidden2` back together:
# 将 `heads` 和 `hidden2` 重新组合在一起：
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
print("Final x shape:", x.shape)


# 浏览完所有操作后，让我们检查一下它们的计算成本。
# 
# 浮点运算（FLOP）是像加法（x + y）或乘法（x * y）这样的基本运算。
# 
# 两个非常容易混淆的缩写（发音相同！）：
# - **FLOPs**：浮点运算次数（floating-point operations），是计算量的度量。
# - **FLOP/s**：每秒浮点运算次数（floating-point operations per second），也写作 FLOPS，用于衡量硬件速度。
# 
# **一些事实：**
# - GPT-3（2020）的训练大约消耗了 3.14e23 FLOPs。[链接](https://lambdalabs.com/blog/demystifying-gpt-3)
# - 有推测认为 GPT-4（2023）的训练大约需要 2e25 FLOPs。[链接](https://patmcguinness.substack.com/p/gpt-4-details-revealed)
# - 美国行政命令：任何使用 >= 1e26 FLOPs 训练的基础模型必须向政府报告（2025 年撤销）。
# 
# A100 的峰值算力约为 312 teraFLOP/s（见 [[spec]](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)）。
# 
# H100 的峰值算力约为 1979 teraFLOP/s（开启稀疏时），不使用稀疏时约为其 50%（见 [[spec]](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet)）。

# In[ ]:


assert a100_flop_per_sec == 312e12
assert h100_flop_per_sec == 1979e12 / 2

# 8 H100s for 2 weeks:
# 8 张 H100 运行 2 周：
total_flops = 8 * (60 * 60 * 24 * 7) * h100_flop_per_sec
print(f"Total FLOPs: {total_flops}")


# ### 线性模型 (Linear model)
# 作为一个简单示例，假设你有一个线性模型。
# - 我们有 n 个点
# - 每个点是 d 维的
# - 线性模型将每个 d 维向量映射到 k 个输出

# In[80]:


if torch.cuda.is_available():
    B = 16384  # Number of points
    D = 32768  # Dimension
    K = 8192   # Number of outputs
else:
    B = 1024
    D = 256
    K = 64

device = get_device()
x = torch.ones(B, D, device=device)
w = torch.randn(D, K, device=device)
y = x @ w

# We have one multiplication (x[i][j] * w[j][k]) and one addition per (i, j, k) triple.
# 每个 (i, j, k) 三元组有一个乘法 (x[i][j] * w[j][k]) 和一个加法。
actual_num_flops = 2 * B * D * K
print(f"Actual num FLOPs: {float(actual_num_flops):.3e}")


# ### 其他操作的 FLOPs
# - m x n 矩阵上的逐元素操作需要 O(m n) FLOPs。
# - 两个 m x n 矩阵的加法需要 m n FLOPs。
# 通常，对于足够大的矩阵，深度学习中遇到的任何其他操作都没有矩阵乘法那么昂贵。
# 
# 解释：
# - B 是数据点的数量
# - (D K) 是参数的数量
# - 前向传播的 FLOPs 是 2 * (Token数) * (参数量)
# 事实证明，这推广到了 Transformer（作为一阶近似）。
# 
# 我们的 FLOPs 计算如何转化为挂钟时间（秒）？
# 让我们计时！

# In[81]:


actual_time = time_matmul(x, w)
actual_flop_per_sec = actual_num_flops / actual_time
print(f"Actual time: {actual_time}")
print(f"Actual FLOP/s: {actual_flop_per_sec:.3e}")


# 每个 GPU 都有一个规格表，报告峰值性能。
# - A100 [[spec]](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)
# - H100 [[spec]](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet)
# 注意 FLOP/s 很大程度上取决于数据类型！

# In[82]:


promised_flop_per_sec = get_promised_flop_per_sec(device, x.dtype)
print(f"Promised FLOP/s: {promised_flop_per_sec}")

# ### Model FLOPs utilization (MFU)
# 定义：(actual FLOP/s) / (promised FLOP/s) [忽略通信/开销]
mfu = actual_flop_per_sec / promised_flop_per_sec
print(f"MFU: {mfu}")
# Usually, MFU of >= 0.5 is quite good (and will be higher if matmuls dominate)
# 通常，MFU >= 0.5 就很好了（如果矩阵乘法占主导地位，还会更高）


# 让我们用 bfloat16 做一次：

# In[83]:


x = x.to(torch.bfloat16)
w = w.to(torch.bfloat16)
bf16_actual_time = time_matmul(x, w)
bf16_actual_flop_per_sec = actual_num_flops / bf16_actual_time
bf16_promised_flop_per_sec = get_promised_flop_per_sec(device, x.dtype)
bf16_mfu = bf16_actual_flop_per_sec / bf16_promised_flop_per_sec
print(f"bfloat16 actual time: {bf16_actual_time}")
print(f"bfloat16 actual FLOP/s: {bf16_actual_flop_per_sec}")
print(f"bfloat16 promised FLOP/s: {bf16_promised_flop_per_sec}")
print(f"bfloat16 MFU: {bf16_mfu}")


# 注意：将 bfloat16 与 float32 进行比较，实际 FLOP/s 更高。
# 这里的 MFU 相当低，可能是因为承诺的 FLOPs 有点乐观。
# 
# ### 总结
# - 矩阵乘法占主导地位：(2 m n p) FLOPs
# - FLOP/s 取决于硬件 (H100 >> A100) 和数据类型 (bfloat16 >> float32)
# - 模型 FLOPs 利用率 (MFU): (actual FLOP/s) / (promised FLOP/s)

# 到目前为止，我们已经构建了张量（对应于参数或数据）并通过操作传递它们（前向）。
# 现在，我们要计算梯度（反向）。
# 
# 作为一个简单的例子，让我们考虑简单的线性模型：
# $y = 0.5 (x \cdot w - 5)^2$
# 
# 前向传播：计算损失

# In[84]:


x = torch.tensor([1., 2, 3])
w = torch.tensor([1., 1, 1], requires_grad=True)  # Want gradient
pred_y = x @ w
loss = 0.5 * (pred_y - 5).pow(2)

# Backward pass: compute gradients
# 反向传播：计算梯度
loss.backward()

# 在新版 PyTorch 中，访问非叶子节点的 .grad 属性会触发警告。
# 只有 requires_grad=True 的叶子节点（如权重 w）才会填充梯度。
# assert loss.grad is None  # 这行会触发 Warning
# assert pred_y.grad is None # 这行也会触发 Warning
assert x.grad is None
assert torch.equal(w.grad, torch.tensor([1, 2, 3]))
print("Gradients computed successfully")


# 让我们计算计算梯度的 FLOPs。
# 
# 重温我们的线性模型：

# In[85]:


if torch.cuda.is_available():
    # 同样调小维度以节省显存
    B = 4096
    D = 4096
    K = 2048
else:
    B = 1024
    D = 256
    K = 64

device = get_device()
x = torch.ones(B, D, device=device)
w1 = torch.randn(D, D, device=device, requires_grad=True)
w2 = torch.randn(D, K, device=device, requires_grad=True)

# Model: x --w1--> h1 --w2--> h2 -> loss
h1 = x @ w1
h2 = h1 @ w2
loss = h2.pow(2).mean()

# Recall the number of forward FLOPs:
# 回忆前向传播的 FLOPs 数量：
num_forward_flops = (2 * B * D * D) + (2 * B * D * K)
print(f"Forward FLOPs: {num_forward_flops:.3e}")


# ### 反向传播中的计算量 (FLOPs) 分析
# 
# 考虑一个简单的两层线性模型：
# $$x \xrightarrow{W_1} h_1 \xrightarrow{W_2} h_2 \to \mathcal{L} (\text{Loss})$$
# 
# **定义维度：**
# *   $x \in \mathbb{R}^{B \times D}$ （$B$ 为 Batch Size，$D$ 为输入维度）
# *   $W_1 \in \mathbb{R}^{D \times D}$，$W_2 \in \mathbb{R}^{D \times K}$
# *   $h_1 = x W_1 \in \mathbb{R}^{B \times D}$
# *   $h_2 = h_1 W_2 \in \mathbb{R}^{B \times K}$
# 
# 在反向传播中，我们需要计算梯度并沿计算图回传。以 $W_2$ 所在的层为例：
# 
# #### 1. 梯度的链式法则推导
# 
# 假设我们已经得到了上一层的梯度 $\frac{\partial \mathcal{L}}{\partial h_2} \in \mathbb{R}^{B \times K}$，我们需要计算：
# 
# *   **参数梯度 $\nabla_{W_2} \mathcal{L}$** (用于更新权重)：
#     根据索引表示：$\frac{\partial \mathcal{L}}{\partial W_{2, jk}} = \sum_{i=1}^{B} h_{1, ij} \cdot \frac{\partial \mathcal{L}}{\partial h_{2, ik}}$
#     **矩阵形式：** $\nabla_{W_2} \mathcal{L} = h_1^T \left( \frac{\partial \mathcal{L}}{\partial h_2} \right)$
#     *   **维度：** $(D \times B) \times (B \times K) = D \times K$
#     *   **FLOPs：** $2 \cdot B \cdot D \cdot K$ （每个元素涉及 $B$ 次乘法和 $B$ 次加法）
# 
# *   **激活梯度 $\nabla_{h_1} \mathcal{L}$** (用于向上传递梯度)：
#     根据索引表示：$\frac{\partial \mathcal{L}}{\partial h_{1, ij}} = \sum_{k=1}^{K} \frac{\partial \mathcal{L}}{\partial h_{2, ik}} \cdot W_{2, jk}$
#     **矩阵形式：** $\nabla_{h_1} \mathcal{L} = \left( \frac{\partial \mathcal{L}}{\partial h_2} \right) W_2^T$
#     *   **维度：** $(B \times K) \times (K \times D) = B \times D$
#     *   **FLOPs：** $2 \cdot B \cdot D \cdot K$ （每个元素涉及 $K$ 次乘法和 $K$ 次加法）
# 
# #### 2. 结论
# 
# 对于每一个线性层（Linear Layer）：
# 1.  **前向传播 (Forward)**：计算 $h = xW$，消耗 **$2BDK$** FLOPs。
# 2.  **反向传播 (Backward)**：涉及两次矩阵乘法（计算参数梯度和激活梯度），共消耗 **$4BDK$** FLOPs。
# 
# **总结：** 在深度学习中，反向传播的计算量大约是前向传播的 **2 倍**。
# 
# 一个很好的图形可视化：[The FLOPs Calculus of Language Model Training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4)
# 
# <div align="center">
# <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*VC9y_dHhCKFPXj90Qshj3w.gif" width="500">
# <br/>
# <i>FLOPs Calculus</i>
# </div>
# 
# 总结：
# - 前向传播：2 * (# 数据点) * (# 参数) FLOPs
# - 反向传播：4 * (# 数据点) * (# 参数) FLOPs
# - 总计：6 * (# 数据点) * (# 参数) FLOPs

# In[79]:


h1.retain_grad()  # For debugging
h2.retain_grad()  # For debugging
loss.backward()

num_backward_flops = 0
num_backward_flops += 2 * B * D * K
num_backward_flops += 2 * B * D * K
num_backward_flops += (2 + 2) * B * D * D

print(f"Backward FLOPs: {num_backward_flops:.3e}")
print(f"Ratio Backward/Forward: {num_backward_flops/num_forward_flops}")


# ## 模型 (Models)
# 
# 在 PyTorch 中，模型的权重通常存储为 `nn.Parameter` 对象。它们本质上是张量，但会被 `nn.Module` 自动识别为需要优化的参数。
# 
# ### 参数与初始化
# 
# 参数的初始化方式对模型的训练稳定性至关重要。如果我们直接使用标准正态分布来初始化权重，可能会遇到数值溢出的问题。
# 
# **为什么权重初始化（正态分布）与输入维度（Input Dim）有关？**
# 
# 在深度学习中，如果直接使用标准正态分布 $N(0, 1)$ 初始化权重，会由于**矩阵乘法的累加效应**导致数值稳定性问题。
# 
# 1. 方差爆炸（Variance Explosion）
# 假设一个神经元的运算为：
# $$y = \sum_{i=1}^{n} w_i x_i$$
# 其中 $n$ 是输入维度（input dim）。
# 
# 如果我们假设输入 $x_i$ 的方差为 1，权重 $w_i$ 也使用方差为 1 的标准正态分布：
# *   单个乘积 $w_i x_i$ 的方差约为 1。
# *   $n$ 个项相加后，输出 $y$ 的方差将变为 **$n$**。
# 
# **结论**：如果 $n=512$，输出的方差就扩大了 512 倍。随着网络层数加深，数值会迅速崩向无穷大（数值溢出）。
# 
# 2. 激活函数饱和（Saturation）
# 过大的输出值会使神经元落入激活函数的“饱和区”：
# *   **Sigmoid/Tanh**：在输入值很大时，梯度几乎为 0，导致**梯度消失**。
# *   **ReLU**：虽然缓解了饱和，但过大的数值仍会导致训练初期的**梯度爆炸**。
# 
# 

# In[7]:


input_dim = 16384
output_dim = 32

# 模型参数在 PyTorch 中存储为 `nn.Parameter` 对象
w = nn.Parameter(torch.randn(input_dim, output_dim))
assert isinstance(w, torch.Tensor)  # 表现得像张量
assert type(w.data) == torch.Tensor  # 访问底层的张量

# 让我们看看如果直接使用标准正态分布初始化会发生什么
x = nn.Parameter(torch.randn(input_dim))
output = x @ w
print(f"Output shape: {output.size()}")
print(f"First element of output: {output[0]:.3f}")
print(f"Standard deviation of output: {output.std():.3f}")
# 注意输出的每个元素的大小大约随着 sqrt(input_dim) 缩放


# 可以看到，输出的数值变得非常大（标准差接近 $\sqrt{\text{input\_dim}}$）。在深度神经网络中，如果每一层的输出都这样剧烈增长，梯度就会爆炸，导致训练极度不稳定。
# 
# 为了解决这个问题，我们需要一种方差与输入维度 `input_dim` 无关的初始化方式。最简单的方法就是将权重除以 $\sqrt{\text{input\_dim}}$。这本质上就是 Xavier 初始化（在常数因子范围内）。
# 
# 
# 常见的初始化方案：
# *   **Xavier 初始化**（针对 Tanh）：权重采样自 $N(0, \frac{1}{n})$。
# *   **Kaiming 初始化**（针对 ReLU）：考虑到 ReLU 会丢弃一半负值信号，权重采样自 $N(0, \frac{2}{n})$。
# 
# **总结**：将初始化进行维度相关的缩放，是为了**维持信号在层间传递时的方差一致性**，确保模型训练的数值稳定性。
# 
# 
# 
# 相关资料：
# - [Xavier 初始化论文](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
# - [关于多头注意力权重矩阵初始化的讨论](https://ai.stackexchange.com/questions/30491/is-there-a-proper-initialization-technique-for-the-weight-matrices-in-multi-head)

# In[4]:


# 我们想要一个对 `input_dim` 不敏感的初始化
# 为此，我们只需按 1/sqrt(input_dim) 进行缩放
w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
output = x @ w
print(f"First element of output after scaling: {output[0]:.3f}")
print(f"Standard deviation of output after scaling: {output.std():.3f}")

# 为了更加安全，我们通常会使用截断正态分布 (truncated normal distribution)，
# 将取值范围限制在 [-3, 3] 之间，以避免出现极端的异常值。
w = nn.Parameter(nn.init.trunc_normal_(torch.empty(input_dim, output_dim), std=1 / np.sqrt(input_dim), a=-3, b=3))


# ### 自定义模型
# 
# 在了解了参数和初始化后，我们可以开始构建更复杂的结构。下面我们通过 `nn.Module` 和 `nn.Parameter` 来构建一个简单的深层线性模型。在 PyTorch 中，`nn.Module` 是所有神经网络模块的基类，它能帮我们自动管理参数和子模块。

# In[13]:


class Linear(nn.Module):
    """简单的线性层（使用规范的 einsum）"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        # 使用 Xavier 初始化
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b = batch (批大小)
        # i = input_dim (输入维度)
        # o = output_dim (输出维度)
        # 'bi,io->bo' 表示：(batch, input) 乘以 (input, output) 得到 (batch, output)
        return torch.einsum('bi,io->bo', x, self.weight)

class Cruncher(nn.Module):
    """一个由多个线性层组成的简单模型"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            Linear(dim, dim)
            for i in range(num_layers)
        ])
        self.final = Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 逐层应用线性变换
        B, D = x.size()
        for layer in self.layers:
            x = layer(x)

        # 最后通过输出头
        x = self.final(x)
        assert x.size() == torch.Size([B, 1])

        # 移除最后一个维度
        x = x.squeeze(-1)
        assert x.size() == torch.Size([B])

        return x

def get_num_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())

D = 64  # 维度
num_layers = 2
model = Cruncher(dim=D, num_layers=num_layers)

# 检查参数大小
param_sizes = [
    (name, param.numel())
    for name, param in model.state_dict().items()
]
print("Parameter sizes:", param_sizes)
num_params = get_num_parameters(model)
print(f"Total parameters: {num_params}")

assert num_params == (D * D) * num_layers + D

# 记得将模型移动到 GPU（如果可用）
# device = get_device()
# model = model.to(device)

# 在一些数据上运行模型
B = 8  # Batch size
x = torch.randn(B, D)
y = model(x)
print(f"Output shape: {y.size()}")
assert y.size() == torch.Size([B])


# ## 训练循环
# 
# 在构建好模型后，我们需要准备数据、选择优化器，并编写训练循环。这一部分将介绍训练过程中的一些实用技巧和资源管理。
# 
# ### 随机性与可复现性
# 
# 随机性在深度学习中无处不在：参数初始化、Dropout、数据打乱等。为了便于调试和结果复现，我们通常会固定随机种子。
# 
# 通常建议在以下三个地方设置随机种子：
# 1. **PyTorch**：控制张量运算和模型初始化。
# 2. **NumPy**：控制数据处理中的随机性。
# 3. **Python Standard Library**：控制基础的随机操作。

# In[14]:


def set_seed(seed: int = 42):
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # NumPy
    np.random.seed(seed)

    # Python
    import random
    random.seed(seed)

    # 为了极致的可复现性（可能会稍微降低速度）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

set_seed(0)
print("Random seeds set.")


# ### 数据加载 (Data Loading)
# 
# 在语言模型训练中，数据通常是一系列整数（由分词器输出）。当数据量非常大（例如 LLaMA 的数据量达到几 TB）时，我们不能一次性将其全部加载进内存。
# 
# **实用技巧：**
# - **Memory Mapping (memmap)**：利用 NumPy 的 `memmap` 以“延迟加载”的方式访问磁盘上的数据。只有被访问到的部分才会被读入内存。
# - **固定内存 (Pinned Memory)**：如果使用 GPU 训练，将数据从 CPU 拷贝到 GPU 时，先将数据放在“固定内存”中可以加速传输。
# - **异步传输**：通过 `non_blocking=True` 让数据拷贝和 GPU 计算并行执行。

# In[ ]:


def get_batch(data: np.ndarray, batch_size: int, sequence_length: int, device: torch.device) -> torch.Tensor:
    # 随机选择 batch_size 个起始位置
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,))

    # 提取序列并转换为张量
    x = torch.stack([torch.from_numpy((data[start:start + sequence_length]).astype(np.int64)) for start in start_indices])

    # 如果是 CUDA，则使用固定内存加速传输
    if device.type == 'cuda':
        x = x.pin_memory()

    # 异步移动到 GPU
    return x.to(device, non_blocking=True)

# 模拟一些训练数据并序列化
orig_data = np.array([i for i in range(100)], dtype=np.int32)
orig_data.tofile("data.npy")

# 使用 memmap 延迟加载
data = np.memmap("data.npy", dtype=np.int32, mode='r')

# 获取一个批次
B, L = 2, 4
device = get_device()
x = get_batch(data, batch_size=B, sequence_length=L, device=device)
print(f"Batch shape: {x.shape}")
print(f"Batch data:\n{x}")


# ### 优化器 (Optimizers)
# 
# 优化器负责根据梯度更新模型参数。除了 PyTorch 自带的优化器，理解它们的内部逻辑也很重要。
# 
# 常用的优化器演进路线：
# 1. **SGD (Stochastic Gradient Descent)**
#    - **核心思想**：最基础的梯度下降。每个参数共用同一个全局学习率，沿着梯度的反方向迈出固定比例的一步。
#    - **局限性**：对所有参数“一视同仁”。在梯度较陡的方向容易震荡，而在平坦方向收敛极慢。且学习率（Step size）非常难以手动调节。
# 
# 2. **AdaGrad (Adaptive Gradient Algorithm)**
#    - **核心思想**：**自适应学习率**。它会记录每个参数历史梯度的平方和。梯度积累越大的参数（说明更新频繁），其有效学习率就会被缩小；反之则放大。
#    - **优点**：非常适合处理**稀疏数据**（某些特征出现频率极低）。
#    - **局限性**：由于分母上的历史梯度平方和是单调递增的，导致学习率会不断衰减直至趋近于 0，可能导致模型在未收敛时就提前“停止学习”。
# 
# 3. **RMSProp (Root Mean Square Propagation)**
#    - **核心思想**：改进了 AdaGrad 的衰减问题。它不再累积所有历史梯度，而是使用**指数移动平均 (EMA)** 仅保留最近一段时间的梯度信息。
#    - **优点**：解决了 AdaGrad 学习率过早消失的问题，在处理非平稳目标（如循环神经网络 RNN）时表现卓越。
# 
# 4. **Adam (Adaptive Moment Estimation)**
#    - **核心思想**：它同时结合了 **Momentum（动量）** 和 **RMSProp** 的优点。
#      - **一阶矩（动量）**：利用梯度的平均值来平滑更新方向，加速通过“山谷”地带。
#      - **二阶矩（RMSProp）**：利用梯度平方的平均值来调整每个参数的学习率。
# 
# 5. **AdamW (Adam with Weight Decay Fix)**
#    - **核心思想**：**权重衰减解耦**。在传统的 Adam 中，是直接对于正则化后的Loss求导，动量会影响正则的强度。AdamW 是对于无正则的Loss求导，然后进行衰减。问题在于Adam的更新步长需要根据二阶动量进行缩放（二阶动量越大，分母越大），这意味着更新越频繁的参数的衰减值也会被缩放的更小，这违背了正则的初衷。 
#    - **优点**：能够提供更有效的正则化效果，提升模型的**泛化能力**。
#    - **现状**：它是训练 **Transformer** 和大语言模型（LLM）的**行业标准**，如果你在训练这类模型，AdamW 通常是首选。
# 
# **AdamW** 
# 
# 假设我们当前的参数是 $w_{t-1}$，当前的梯度是 $g_t$，学习率是 $\eta$。
# 
# 第一步：更新一阶动量 (First Moment) $m_t$。计算动量而非单步的导数能够使局部最优附近的导数互相抵消，模型优化会继续朝着历史轨迹的方向进行。（$\beta_1=0.9$）
# 
# $m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$
# 
# 第二步：更新二阶动量 (Second Moment) $v_t$。计算二阶动量的目的是实现自适应更新率，梯度过大的时候见效更新率。（$\beta_2=0.999$）
# 
# $v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$
# 
# (注意：这里的 $g_t^2$ 是梯度元素的平方，即 element-wise square)
# 
# 第三步：偏差修正 (Bias Correction)。这是为了防止初始化时 $m_0$ 和 $v_0$ 为 0 导致的数值偏差，训练初期尤为重要
# 
# $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
# 
# 第四步：参数更新 (Update)，其中 weight decay部分是为了让模型参数始终保持一个较小的值，等同于一个 L2 正则化，用于防止模型过拟合。
# 
# $w_t = w_{t-1} - \eta \left( \underbrace{\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}}_{\text{Adam 部分}} + \underbrace{\lambda w_{t-1}}_{\text{Weight Decay 部分}} \right)$
# 
# 下面我们手动实现简单的 SGD 和 AdaGrad 优化器。

# In[ ]:


from typing import Iterable

class SimpleSGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super().__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 直接沿梯度负方向更新
                p.data -= lr * p.grad.data


# In[17]:


class SimpleAdaGrad(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super().__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data

                # 初始化或获取梯度平方累积值 g2
                if "g2" not in state:
                    state["g2"] = torch.zeros_like(grad)

                g2 = state["g2"]
                g2 += torch.square(grad)

                # 更新参数，分母增加 eps 防止除零
                p.data -= lr * grad / torch.sqrt(g2 + 1e-10)

print("Custom optimizers implemented.")


# ### 优化器对比演示：SimpleAdaGrad vs SGD
# 
# 我们通过最小化一个“长条形谷底”函数 $f(x, y) = x^2 + 10y^2$ 来观察不同：
# - **SGD**：对所有维度使用统一的学习率。在这个函数中，如果学习率过大，$y$ 方向容易震荡；如果过小，$x$ 方向收敛极慢。
# - **AdaGrad**：为每个维度维护独立的累积梯度平方，自动缩小梯度较大维度的步长，从而实现更平稳的收敛。

# In[18]:


import torch
import torch.nn as nn

# 1. 定义一个简单的非对称目标函数 f(x, y) = x^2 + 10y^2
def objective_function(params):
    x, y = params
    return x**2 + 10 * y**2

# 2. 初始化两组相同的参数
params_adagrad = nn.Parameter(torch.tensor([10.0, 10.0]))
params_sgd = nn.Parameter(torch.tensor([10.0, 10.0]))

# 3. 初始化优化器
lr = 0.5
opt_adagrad = SimpleAdaGrad([params_adagrad], lr=lr)
opt_sgd = torch.optim.SGD([params_sgd], lr=lr)

print(f"{'Step':<6} | {'AdaGrad (x, y)':<25} | {'SGD (x, y)':<25}")
print("-" * 65)

for step in range(11):
    # AdaGrad 更新
    opt_adagrad.zero_grad()
    loss_adagrad = objective_function(params_adagrad)
    loss_adagrad.backward()
    opt_adagrad.step()

    # SGD 更新
    opt_sgd.zero_grad()
    loss_sgd = objective_function(params_sgd)
    loss_sgd.backward()
    opt_sgd.step()

    if step % 2 == 0:
        adagrad_pos = f"({params_adagrad[0]:.2f}, {params_adagrad[1]:.2f})"
        sgd_pos = f"({params_sgd[0]:.2f}, {params_sgd[1]:.2f})"
        print(f"{step:<6} | {adagrad_pos:<25} | {sgd_pos:<25}")

print("-" * 65)
print("结论：SGD 在 y 方向（梯度大）产生了剧烈震荡，甚至可能发散；")
print("而 AdaGrad 通过自适应调整，在保持 x 方向推进的同时，有效抑制了 y 方向的震荡。")


# ### 训练状态下的资源计算
# 
# 在训练过程中，显存的占用不仅包括模型参数，还包括：
# 1. **参数 (Parameters)**：权重矩阵。
# 2. **梯度 (Gradients)**：与参数大小相同。
# 3. **优化器状态 (Optimizer States)**：例如 AdaGrad 需要存储历史梯度平方和（1 份副本），Adam 则需要存储一阶和二阶矩（2 份副本）。
# 4. **激活值 (Activations)**：前向传播中产生，用于反向传播计算梯度。激活值的大小通常与 `Batch Size` 和 `Sequence Length` 成正比。
# 
# **显存估算（以 float32 为例，每个值 4 字节）：**
# - 参数：$P \times 4$
# - 梯度：$P \times 4$
# - 优化器状态（AdaGrad）：$P \times 4$
# - 激活值：取决于模型深度和输入维度。
# 
# **计算量估算：**
# - 一个完整的 Step（前向 + 反向）大约需要 $6 \times B \times P$ 次 FLOPs。

# In[ ]:


# 使用刚才定义的模型
B, D, L = 2, 64, 2
model = Cruncher(dim=D, num_layers=2).to(device)
optimizer = SimpleAdaGrad(model.parameters(), lr=0.01)

# 模拟一个训练步骤
x = torch.randn(B, D, device=device)
y_true = torch.randn(B, device=device)

# 前向传播
y_pred = model(x)
loss = F.mse_loss(y_pred, y_true)

# 反向传播
loss.backward()

# 优化器更新
optimizer.step()

# 清空梯度（释放内存）
optimizer.zero_grad(set_to_none=True)

# 资源统计
num_params = get_num_parameters(model)
print(f"Number of parameters: {num_params}")

# 以 float32 计算
mem_params = num_params * 4 / 1024 # KB
mem_grads = num_params * 4 / 1024 # KB
mem_opt_state = num_params * 4 / 1024 # KB (AdaGrad 存储 1 个状态)

print(f"Estimated memory for params+grads+opt_state: {mem_params + mem_grads + mem_opt_state:.2f} KB")
print(f"Estimated compute per step: {6 * B * num_params / 1e6:.4f} MFLOPs")


# ### 训练循环与检查点 (Checkpointing)
# 
# 由于训练大型模型可能需要数周甚至数月，程序崩溃或硬件故障是不可避免的。因此，**定期保存检查点**至关重要。
# 
# 检查点通常包含：
# - **模型状态 (`model.state_dict()`)**：所有的参数权重。
# - **优化器状态 (`optimizer.state_dict()`)**：历史梯度信息、学习率、当前步数等。
# 
# 如果只保存模型而不保存优化器状态，重新训练时优化器会丢失历史状态（如 Adam 的动量），导致训练曲线出现剧烈跳变。

# In[ ]:


def train_one_step(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)
    y_pred = model(x)
    loss = F.mse_loss(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()

# 保存检查点示例
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}
torch.save(checkpoint, "model_checkpoint.pt")
print("Checkpoint saved to 'model_checkpoint.pt'")

# 加载检查点示例
loaded_checkpoint = torch.load("model_checkpoint.pt")
model.load_state_dict(loaded_checkpoint["model"])
optimizer.load_state_dict(loaded_checkpoint["optimizer"])
print("Checkpoint loaded.")


# ### 混合精度训练 (Mixed Precision Training)
# 
# 正如我们在张量部分看到的，`float32` 虽然准确但慢且耗显存，`bfloat16` 快且省显存但分辨率较低。
# 
# **混合精度训练**的思路是：
# - 在计算密集型操作（如矩阵乘法，涉及激活值）时使用较低精度（如 `bfloat16` 或 `fp8`）。
# - 在更新参数和累积梯度时保留一份 `float32` 的主权重（Master Weights），以保证数值稳定性。
# 
# PyTorch 提供了 `torch.amp` (Automatic Mixed Precision) 库来自动化这一过程。在 H100 等现代 GPU 上，甚至可以结合 NVIDIA 的 Transformer Engine 来支持 FP8 训练。

# In[ ]:


# 自动混合精度 (AMP) 的典型用法示例
# 在 H100 等 GPU 上通常使用 bf16
use_amp = torch.cuda.is_available()
device_type = 'cuda' if use_amp else 'cpu'
pt_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float32

# 创建缩放器（GradScaler），在使用 fp16 时非常重要，防止梯度下溢
# 注意：在使用 bf16 时通常不需要 GradScaler，因为 bf16 范围够大
scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and pt_dtype == torch.float16))

def train_step_amp(model, optimizer, x, y):
    optimizer.zero_grad(set_to_none=True)

    # 使用 autocast 开启自动混合精度
    with torch.amp.autocast(device_type=device_type, dtype=pt_dtype):
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y)

    # 梯度缩放处理
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

print(f"AMP implementation example added (Dtype: {pt_dtype}).")


# ## Appendix
# 
# 
# ### A. 理论直觉：为什么高维优化是可能的？
# 
# #### 1. 流形假设 (The Manifold Hypothesis)
# 我们在 Lecture 中处理的是 $16384$ 维甚至更高的向量。直觉告诉我们，高维空间是极其稀疏的（维度灾难），在这种空间里寻找规律看似是不可能的。
# 
# 
# **核心概念**：
# 虽然自然数据（图像、文本 embedding）的**表面维度 (Embedding Dimension)** 极高（例如 MNIST 图片是 $28 \times 28 = 784$ 维），但它们实际上分布在一个**内在维度 (Intrinsic Dimension)** 很低的流形 $\mathcal{M}$ 上（MNIST 的内在维度估计仅为 10-15 维）。
# 
# 
# * **观测到的证据 (Evidence)**：
#     1.  **隐空间插值 (Latent Interpolation)**：这是流形存在的最直观证据。
#         * **像素插值 (Pixel-wise)**：直接对两张人脸图片做像素平均，会得到一张模糊的“鬼图”。这是因为连接两点的直线穿过了流形弯曲的外部（高维空间的“空旷地带”），那里没有有效数据。
#         * **流形插值**：将图片映射到低维隐空间，取中点再解码，会得到一张清晰的、既像张三又像李四的脸。这说明我们在沿着低维流形的表面行走。
#     2.  **对抗样本 (Adversarial Examples)**：
#         * 在一张熊猫图上加上微小的噪声，模型就会识别为长臂猿。
#         * **解释**：这些精心设计的噪声实际上是沿着**垂直于流形**的方向推动了数据点。虽然在视觉上（流形切面）移动很小，但在高维的决策边界上，它已经掉出了“熊猫流形”的区域。
# 
# * **神经网络的本质：拓扑变换 (Topological Transformation)**：
#     * 可以将深度神经网络看作一台**拓扑变换机**。
#     * 输入是纠缠在一起的“瑞士卷”（数据在像素空间里混杂）。
#     * 每一层（ReLU, Conv）都在对空间进行拉伸、扭曲和折叠（**Manifold Unfolding**）。
#     * **目标**：把这个卷曲的流形铺平，使得不同类别的数据在最后线性可分。
# 
# * **启示**：
#     * **数据增强 (Data Augmentation)**：旋转、缩放等操作本质上是在探索流形的**局部结构**。我们在告诉模型：“沿着这个方向移动（比如变亮），依然在这个流形上（依然是只猫）”，这有助于模型描绘流形的边界。
#     * **距离度量 (Distance Metrics)**：在高维空间直接计算欧氏距离 (MSE) 往往效果不好，因为那是“直线距离”。更好的方式是在 Embedding 空间计算距离，这近似于流形上的**测地线距离 (Geodesic Distance)**。
#     * **生成式 AI (Diffusion Models)**：目前的 Diffusion 本质上就是**流形学习器**。加噪是把数据推离流形进入高维随机空间，去噪（Denoising）则是学习如何把高维空间中的随机点“落回”到那个低维的数据流形上。
# 
# ---
# 
# ### B. 工程与系统
# 
# 我们在正文中计算了 FLOPs，但在现实世界中，**Memory I/O (显存带宽)** 往往比 FLOPs (计算速度) 更早成为瓶颈。
# 
# #### 1. IO-Awareness: FlashAttention
# Transformer 的计算瓶颈核心在于 Attention 机制中的 $QK^T$ 操作产生的 $N \times N$ 矩阵。
# * **Naive Attention**：从 HBM (High Bandwidth Memory, 显存) 读取 $Q, K$，计算 $S = QK^T$，写回 HBM；读取 $S$，计算 Softmax，写回 HBM... **这种频繁的读写是巨大的浪费**。
# * **FlashAttention (Dao et al.)**：
#     * **思想**：**Tiling (分块)**。将 $Q, K, V$ 切成小块加载到 GPU 的 **SRAM** (L1 Cache, 极快但极小) 中。
#     * **技巧**：在 SRAM 中计算完 Softmax 和最终结果后，再写回 HBM。
#     * **结果**：虽然 FLOPs 没变（甚至稍微多了一点点，因为要重计算），但由于减少了 HBM 访问，速度提升了 3-10 倍，且显存占用从 $O(N^2)$ 降到了 $O(N)$。
# * **Research Tip**：现在几乎所有主流 LLM 训练都**强制**开启 FlashAttention。
# 
# #### 2. 梯度裁剪 (Gradient Clipping)
# 正文中提到了梯度爆炸。除了初始化，工程上最常用的防御手段是 `torch.nn.utils.clip_grad_norm_`。
# * **做法**：如果梯度的 L2 范数超过阈值（如 1.0），就按比例缩小整个梯度向量。
# * **为什么重要**：在训练早期或遇到“坏数据”时，Loss 会突然 Spike。裁剪能防止单次更新步长过大破坏模型已经学到的权重。
# 
# #### 3. 数值稳定性技巧 (Numerical Stability Tricks)
# 当你在使用 `fp16` 或 `bf16` 时，经常会遇到 `NaN` (Not a Number)。
# * **Log-Sum-Exp Trick**：在计算 Softmax 时：
#     $$\frac{e^{x_i}}{\sum e^{x_j}}$$
#     如果 $x_i$ 很大，$e^{x_i}$ 会溢出。
#     **Trick**：$\text{softmax}(x) = \text{softmax}(x - \max(x))$。减去最大值不会改变 Softmax 的结果，但保证了所有指数项都 $\le e^0 = 1$，避免溢出。
# * **Epsilon ($\epsilon$)**：在做除法或开根号时（如 LayerNorm, Adam），永远记得加一个 `1e-6` 或 `1e-10`。
#     $$\frac{x}{\sqrt{\sigma^2 + \epsilon}}$$
# 
# ---
# 
# ### C. 高性能算子：Triton 
# 
# #### 1. 什么是 Triton?
# Triton 是 OpenAI 开源的一种**类似 Python 的编程语言**，用于编写高效的 GPU 内核（Kernels）。它直接对标的是 NVIDIA 的 **CUDA C++**。
# 
# * **背景**：在 Triton 出现之前，如果你想写一个比 PyTorch 原生更快的算子（比如 FlashAttention），你必须精通 CUDA，手动管理线程、共享内存和寄存器。
# * **定位**：Triton 处于 PyTorch (高层) 和 CUDA (底层) 之间。它允许你用 Python 语法写出性能媲美甚至超越手写 CUDA 的代码。
# 
# #### 2. 核心哲学：基于块的编程 (Block-Based Programming)
# 这是 Triton 和 CUDA 最大的区别。
# 
# * **CUDA (Thread-based)**：你需要控制每一个**线程 (Thread)** 做什么（ThreadIdx.x）。
# * **Triton (Block-based)**：你操作的最小单位是**块 (Block)**。
#     * 在 Triton 里，你不再写 `scalar_a + scalar_b`。
#     * 你写的是 `block_a + block_b`。Triton 编译器会自动帮你把这个块操作映射到 GPU 的线程上。
# 
# 这种抽象极大地降低了开发难度，同时通过自动化优化（Coalesced memory access, Shared memory management）保持了高性能。
# 
# #### 3. 为什么它在今天如此重要？
# PyTorch 2.0 引入了 `torch.compile()`，其默认的后端 Inductor 就是基于 Triton 的。
# 
# 当你运行 `torch.compile(model)` 时：
# 1.  PyTorch 会捕获你的计算图。
# 2.  它会发现一系列细碎的操作（比如：`x * 2 + y - z`）。
# 3.  在 Eager Mode 下，这需要 GPU 读写 HBM（显存）三次。
# 4.  **Triton 的做法**：它会把这些操作**融合 (Fuse)** 成一个 Kernel。数据只从 HBM 读一次，在极快的 SRAM 中完成所有加减乘除，再写回 HBM。**这是 PyTorch 2.0 提速的主要来源。**
# 
# #### 4. Triton 代码长什么样？(以向量加法为例)
# Triton Kernel的伪代码：

# In[ ]:


import triton
import triton.language as tl

# @triton.jit 装饰器告诉编译器这是要在 GPU 上跑的代码
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,  # 指针指向显存(HBM)
    n_elements,                # 向量长度
    BLOCK_SIZE: tl.constexpr   # 元编程：块大小
):
    # 1. 确定当前程序处理的是哪一个数据块 (Program ID)
    pid = tl.program_id(axis=0)

    # 2. 计算当前块需要处理的数据偏移量
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 3. 边界检查 (Masking)：防止读取超过向量长度的内存
    mask = offsets < n_elements

    # 4. 加载数据：从 HBM 到 SRAM
    # 注意：这里直接加载了一个 BLOCK 的数据，而不是一个数
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 5. 计算：在 SRAM 中并行执行
    output = x + y

    # 6. 写回数据：从 SRAM 到 HBM
    tl.store(output_ptr + offsets, output, mask=mask)

