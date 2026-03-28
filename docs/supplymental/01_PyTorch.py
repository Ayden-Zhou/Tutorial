

# %% [python]
import math
import numpy as np
import torch
from matplotlib import pyplot as plt

# %% [markdown]
# # PyTorch 

# %% [markdown]
# ## 初始化 Tensor
#
# Tensor 可以用很多方式初始化。下面看几个例子：
#
# **直接从数据创建**
#
# Tensor 可以直接从数据创建，数据类型会自动推断。

# %%
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data) 
print(x_data)

# %% [markdown]
# **从另一个 tensor 创建**
#
# 新 tensor 会继承参数 tensor 的属性（形状、数据类型），除非你显式覆盖它们。

# %%
x_ones = torch.ones_like(x_data) # 保留 x_data 的属性
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # 覆盖 x_data 的数据类型
print(f"Random Tensor: \n {x_rand} \n")

# %% [markdown]
# **使用随机值或常数值创建**
#
# `shape` 是 tensor 维度的元组。下面这些函数里，它决定输出 tensor 的形状。

# %%
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# %% [markdown]
# ------------------------------------------------------------------------
#
# ## Tensor 的属性
#
# Tensor 的属性会描述它的形状、数据类型，以及它存放在哪个设备上。

# %%
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# %% [markdown]
# ## Tensor API 的基本操作
#
# PyTorch 提供了很多便捷的张量计算方法，比如 `x.clamp(0).pow(2)`（支持链式写法）。因此 PyTorch 里的代码通常更短。下面简单看一遍：
#
# **逐元素运算。** 大多数 tensor 运算都是简单的逐元素运算，也就是每个数组元素都做同样的数学操作。`x+y`、`x*y`、`x.abs()`、`x.pow(3)` 等都属于这一类。和 Matlab 不同，`*` 表示逐元素乘法，不是矩阵乘法。
#
# **默认返回拷贝。** 几乎所有操作，包括 `x.sort()` 这种，都会返回一个新的 tensor 副本，而不会覆盖输入 tensor。例外是以下划线结尾的函数，比如 `x.mul_(2)`，它会直接原地把 `x` 的内容乘 2。
#
# **常见的归约运算。** 还有一些常见操作，比如 `max`、`min`、`mean`、`sum`，会沿着一个或多个维度把数组压缩掉。在 PyTorch 里，你可以通过传入 `dim=n` 来指定要归约的维度。
#
#
# **线性代数计算** `torch.mm(a,b)` 是矩阵乘法，`torch.inverse(a)` 求逆，`torch.eig(a)` 求特征值，等等。
#
# 另一个要知道的是，PyTorch 通常会非常快，甚至在 CPU 上也常常比 NumPy 更快，因为它的实现会在底层强力并行化。NumPy 往往只用一个线程，而 PyTorch 在适合的时候会使用多个线程。
#
# 参见 [Tensor methods 的参考文档](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) 看看内置了哪些方法。下面先做一个简单的向量演示：

# %%
# 用 0 到 5 之间等间距的 101 个数构成一个向量。
x = torch.linspace(0, 5, 101)

# 打印 x 的前五个元素。
print(x[:5])

# 用向量做一些计算。
y1, y2 = x.sin(), x ** x.cos()
y3 = y2 - y1
y4 = y3.min()

# 打印并绘图展示这些结果。
print(f'The shape of x is {x.shape}')
print(f'The shape of y1=x.sin() is {y1.shape}')
print(f'The shape of y2=x ** x.cos() is {y2.shape}')
print(f'The shape of y3=y2 - y1 is {y3.shape}')
print(f'The shape of y4=y3.min() is {y4.shape}, a zero-d scalar')

plt.plot(x, y1, 'red', x, y2, 'blue', x, y3, 'green')
plt.axhline(y4, color='green', linestyle='--') # type: ignore
plt.show()

# %% [markdown]
# ## PyTorch Tensor 的维度顺序约定
#
# **多维数据的约定。** 一旦 tensor 不止一维，就需要决定各个轴的顺序。为了减少混乱，大多数数据处理都会遵循同一套全局约定。尤其是在 PyTorch 里，很多图像相关数据都是四维的，维度顺序通常写成 `data[batch_index, channel_index, y_position, x_position]`，也就是：
#
# * 第 0 维用于索引 batch 里的不同图像。
# * 第 1 维用于索引图像表示里的通道（例如 0,1,2 分别表示 R,G,B，或者更多通道）。
# * 第 2 维（如果存在）表示行位置（y 值，从上往下数）。
# * 第 3 维（如果存在）表示列位置（x 值，从左往右数）。
#
# 记住这个顺序有个经验：只在后面维度上变化的相邻元素，在内存里物理上会更近；因为它们经常一起参与运算，这有利于局部性。而第 0 维（batch 维）通常只是把彼此独立的数据点分组，彼此之间不会频繁组合，所以没必要放得很近。
#
# 不带网格几何结构的流式数据会去掉后面的维度；三维网格数据则会是 5 维，在 y 前面再加一个深度 z。这个四维轴顺序约定在 caffe 和 tensorflow 里也能见到。
#
# 可以用 `torch.cat([a, b, c])` 或 `torch.stack([a, b, c])` 把多个 tensor 拼成一个 batch tensor。（区别是：`cat` 不会增加新维度，只是在已有的第 0 维上拼接；`stack` 则会新加一个第 0 维作为 batch。）
#
# **多维线性运算的约定。** 存储矩阵权重或卷积权重时，会遵循线性代数里的约定：
# * 第 0 维（行数）对应输出通道维度
# * 第 1 维（列数）对应输入通道维度
# * 第 2 维（如果存在）是卷积核的 y 维度
# * 第 3 维（如果存在）是卷积核的 x 维度
#
# 由于这个约定假定通道按不同行来排，而数据约定把不同 batch 项放在不同行，因此在把数据送进线性代数运算之前，经常需要先做轴转置。
#
# **`permute` 和 `view`：不移动内存地重排数组。** `permute` 和 `view` 方法适合重新排列、拉平和展开各个轴。`x.permute(1,0,2,3).view(x.shape[1], -1)` 就是一个例子。它们只是在内存里改变这块数字的视图，而不会移动任何数字，所以速度很快。
#
# **有些重排还是需要拷贝。** 某些轴交换和拉平的组合无法在不拷贝数据的情况下完成；`x.contiguous()` 会把数据拷贝成当前视图对应的自然顺序；`x.reshape()` 也类似于 `view`，但在必要时会自动拷贝，这样你就不用自己判断了。参见 [Tensor.view 的文档](https://pytorch.org/docs/master/tensors.html#torch.Tensor.view)。
# %% [markdown]
# **标准的 NumPy 风格索引和切片：**

# %%
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[:, -1]}")
tensor[:,1] = 0
print(tensor)

# %% [markdown]
# **拼接 tensor** 你可以用 `torch.cat` 沿着指定维度拼接一串 tensor。也可以看一下 [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html)，这是另一个 tensor 拼接算子，但它和 `torch.cat` 有细微差别。

# %%
t1 = torch.cat([tensor, tensor, tensor], dim=0) # 沿着该维度拼接
print(tensor)
print(tensor.shape)
print(t1)
print(t1.shape)

t2 = torch.stack([tensor, tensor, tensor], dim=1) # 创建一个新的维度
print(tensor)
print(tensor.shape)
print(t2)
print(t2.shape)

# %% [markdown]
# **空维度**

# %%
t3 = tensor.unsqueeze(0)
print(tensor.shape)
print(t3.shape)
print(t3)
t4 = t3.squeeze(0)
print(tensor.shape)
print(t4.shape)
print(t4)

# %% [markdown]
# **算术运算**

# 这里计算的是两个 tensor 的矩阵乘法，y1、y2、y3 的值会相同。
# `tensor.T` 返回 tensor 的转置。
tensor = torch.Tensor([[1,2,5],[3,4,6]])
print(tensor.shape)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(y1, y1.shape)
print(y2, y2.shape)
print(y3, y3.shape)

# 这里计算的是逐元素乘法，z1、z2、z3 的值会相同。
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z1, z1.shape)
print(z2)
print(z3)

# %% [markdown]
# **单元素 tensor** 如果你有一个只有一个元素的 tensor，比如把一个 tensor 的所有值聚合成一个数，就可以用 `item()` 把它转成 Python 数值：

# %%
agg = tensor.sum()
print(agg, type(agg), agg.dtype)
print(agg.shape)
agg_item = agg.item()
print(agg_item, type(agg_item))

# %% [markdown]
# ## 5 爱因斯坦记号（Einsum）
#
# 矩阵乘法可以推广到任意维度的 tensor，但要把 tensor 各维度理清楚往往会让人头大。Einstein 记号的做法就是：给输入 tensor 的每个轴分配字母变量，然后明确写出哪些轴会出现在输出 tensor 里。比如，外积可以写成 i, j -> ij，而矩阵乘法可以写成 ij, jk -> ik。
#
# Einstein 记号本身也是一个持续发展的研究和编程语言设计主题：[这里有一篇关于 Einstein API 历史和未来的论文](https://openreview.net/pdf?id=oapKSVM2bcj)。
#
# 在 PyTorch 里，Einstein 记号可以用 einsum 表示。下面看看普通矩阵乘法写成 einsum 是什么样：

# %%

A = torch.randn(2,5)
B = torch.randn(5,3)

# 取消注释即可查看普通矩阵乘法
# print(torch.mm(A, B))

# 把普通矩阵乘法写成 einsum
print(torch.einsum('ij, jk -> ik', A, B))

# %% [markdown]
# ## 6 广播机制（Broadcasting）
# 如果两个 tensor 满足下面条件，就称它们是“可广播的”：
#
# * 每个 tensor 至少有一个维度。
# * 从最后一个维度开始向前比较时，对应维度的大小要么相等，要么其中一个是 1，要么其中一个维度不存在。
#
# 如果某个 PyTorch 操作支持 broadcast，那么它的 Tensor 参数可以自动扩展到相同大小，而不需要复制数据。
#
# 如果两个 tensor `x`, `y` 可广播，那么结果 tensor 的大小按下面规则计算：
#
# * 如果 `x` 和 `y` 的维度数不同，就在维度较少的 tensor 前面补 1，让它们长度一致。
# * 然后对每个维度取 `x` 和 `y` 在该维度上的最大值，作为结果维度大小。
#
# 更多细节见 [这里](https://pytorch.org/docs/stable/notes/broadcasting.html)。

# %%

x=torch.empty(5,7,3)
y=torch.empty(5,7,3)
# 同形状 tensor 一定可广播，也就是上面的规则一定成立。

x=torch.empty((0,))
y=torch.empty(2,2)
# x 和 y 不可广播，因为 x 甚至没有至少一个维度。

# 可以按最后几个维度对齐。
x=torch.empty(5,3,4,1)
y=torch.empty(  3,1,1)
# x 和 y 可广播。
# 第 1 个尾随维度：二者大小都是 1。
# 第 2 个尾随维度：y 的大小是 1。
# 第 3 个尾随维度：x 的大小等于 y 的大小。
# 第 4 个尾随维度：y 这里没有对应维度。

# 但下面这个不行：
x=torch.empty(5,2,4,1)
y=torch.empty(  3,1,1)
# x 和 y 不可广播，因为在第 3 个尾随维度上 2 != 3。

# 也可以按最后几个维度对齐，这样更容易读。
x=torch.empty(5,1,4,1)
y=torch.empty(  3,1,1)
print((x+y).size())

# 但这不是必须的：
x=torch.empty(1)
y=torch.empty(3,1,7)
print((x+y).size())

x=torch.empty(5,2,4,1)
y=torch.empty(3,1,1)
# print((x+y).size())

# %% [markdown]
# ## 7 自动微分（Autograd）
#
# 如果给 torch Tensor 加上 `x.requires_grad=True`，PyTorch 就会自动跟踪所有从 `x` 派生出来的 tensor 的计算历史。这样它就能求出任意标量结果相对于 `x` 各个分量变化的导数。
#
# <img src="https://raw.githubusercontent.com/davidbau/how-to-read-pytorch/6ce891301e79aa8e2164a703c08257a52b2d1ad3/notebooks/autograd-graph.png" style="max-width:100%">
#
# `torch.autograd.grad(output_scalar, [list of input_tensors])` 会计算列表中每个输入 tensor 分量对应的 `d(output_scalar) / d(input_tensor)`。要让它工作，输入 tensor 和输出必须处在同一个 `requires_grad=True` 计算图里。
#
# 在下面这个例子里，`x` 显式标记为 `requires_grad=True`，所以由 `x` 推导出来的 `y.sum()` 会自动带着计算历史，也就可以继续求导。

# %%
x = torch.linspace(0, 5, 100, requires_grad=True)
y = (x**2).cos()
s = y.sum()
[dydx] = torch.autograd.grad(s, [x])

plt.plot(x.detach(), y.detach(), label='y')
plt.plot(x.detach(), dydx, label='dy/dx')
plt.legend()
plt.show()

# %% [markdown]
# 对上面这个例子补充一句：因为向量空间里的各个分量彼此独立，所以当 `j != i` 时，`dy[j] / dx[i] == 0`。因此 `d(y.sum()) / dx[i] = dy[i] / dx[i]`，也就是说，计算标量和 `s` 的单个梯度向量，等价于计算逐元素导数 `dy/dx`。
#
# **把 tensor 从计算历史里分离出来。** 任何依赖于 `x` 的 tensor 都会是 `requires_grad=True`，并连接到完整的计算历史。但如果你把 tensor 转成普通 Python 数字，PyTorch 就看不到中间计算，也就没法继续求梯度。
#
# 为了避免在不知不觉中经过一个无法跟踪的非 PyTorch 数值，PyTorch 会阻止把 `requires_grad` tensor 直接转成不可追踪的数字。你需要先显式调用 `x.detach()` 或 `y.detach()`，告诉系统你要的是一个不带跟踪信息的引用，然后再把它拿去画图或当作普通数值使用。

# %% [markdown]
# ## 8 反向传播与 `.grad`
#
# 在典型神经网络里，我们面对的不是像上面那样只针对一个输入 `x` 的梯度，而是针对几十个甚至几百个已经标记为 `requires_grad=True` 的参数求梯度。把每个梯度输出和原始输入一一对应起来会比较麻烦。好在梯度和输入通常具有完全相同的形状，所以把梯度直接原地存到 tensor 自己身上就很自然。
#
# **用 `backward()` 把梯度写进 `.grad` 属性。** 为了简化这个常见操作，PyTorch 提供了 `y.backward()` 方法。它会计算 `y` 相对于所有被跟踪依赖的梯度，并把结果写到每个原始输入向量 `x` 的 `x.grad` 字段里，只要这个 `x` 被标记为 `requires_grad=True`。

# %%
x = torch.linspace(0, 5, 100, requires_grad=True)
y = (x**2).cos()
y.sum().backward()   # 把梯度写入下面的 grad 属性
print(x.grad)

plt.plot(x.detach(), y.detach(), label='y')
plt.plot(x.detach(), x.grad, label='dy/dx') # type: ignore
plt.legend()
plt.show()

# %% [markdown]
# ## 9 梯度累积、清零与推理时的内存节省
#
# **梯度累积。** 如果你的数据批次太大，没法一次性求出整批数据的梯度，通常可以把批次切成更小的片段，再把梯度加起来。因为梯度累积是常见模式，所以当参数的 `x.grad` 已经存在时，再调用 `.backward()` 不是错误；新的梯度会直接加到旧梯度上。
#
# **`zero_grad()`。** 这意味着在再次运行 `backward()` 之前，你需要先把 `x.grad` 的旧值清零，否则新梯度会叠加到旧梯度上。优化器提供了 `optim.zero_grad()`，用来一次性把所有待优化参数的梯度清掉。
#
# **推理时节省内存。** 如果你不打算训练网络，通常会把神经网络参数默认的 `requires_grad=True` 关掉。否则，每次前向运行都会生成带梯度属性的输出，并挂在一条很长的计算历史上，白白占用大量宝贵的 GPU 显存。
#
# 如果你完全不需要训练，可以遍历网络的所有参数，把它们设成 `requires_grad=False`。
#
# 另一种避免保留计算历史的方法，是把整段计算包进 `with torch.no_grad():` 代码块里。这样会关闭所有 autograd 机制，当然也意味着 `.backward()` 不会工作。
#
# **注意，这和 `net.eval()` 的作用不同。** `net.eval()` 只是把网络切到推理模式，让 batchnorm、dropout 等操作在训练和推理时表现不同；它不会影响 `requires_grad`。

# %%
# 设置可视化辅助函数。
# 下面这个单元定义 `plot_progress()`，用于绘制优化轨迹。

import matplotlib
from matplotlib import pyplot as plt

def plot_progress(bowl, track, losses):
    # 绘制目标函数的等高线，以及 x 和 y 的变化轨迹。
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 5))
    for size in torch.linspace(0.1, 1.0, 10):
        angle = torch.linspace(0, 6.3, 100)
        circle = torch.stack([angle.sin(), angle.cos()])
        ellipse = torch.mm(torch.inverse(bowl), circle) * size
        ax1.plot(ellipse[0,:], ellipse[1,:], color='skyblue')
    track = torch.stack(track).t()
    ax1.set_title('x 的优化轨迹')
    ax1.plot(track[0,:], track[1,:], marker='o')
    ax1.set_ylim(-1, 1)
    ax1.set_xlim(-1.6, 1.6)
    ax1.set_ylabel('x[1]')
    ax1.set_xlabel('x[0]')
    ax2.set_title('y 的变化轨迹')
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)) # type: ignore
    ax2.plot(range(len(losses)), losses, marker='o')
    ax2.set_ylabel('目标值')
    ax2.set_xlabel('迭代次数')
    fig.show()

from IPython.display import HTML
HTML('''<script>function toggle_code(){$('.rendered.selected div.input').toggle().find('textarea').focus();} $(toggle_code())</script>
<a href="javascript:toggle_code()">Toggle</a> the code for plot_progress.''')

# %% [markdown]
# ## 10 优化器
#
# 优化器的任务很简单：给定目标函数相对于一组输入参数的梯度，把参数沿着能让目标下降的方向稍微挪一点。它就是根据梯度，对每个参数做一个很小的修正。
#
# ### 10.1 手动实现梯度下降
#
# 你可以直接用 `loss.backward()` 计算损失对每个参数 `x` 的梯度，再用 `x -= learning_rate * x.grad` 把 `x` 往让损失变小的方向推一点。
#
# 下面是一个手动做梯度下降的例子：

# %%
import torch

x_init = torch.randn(2)
x = x_init.clone()

bowl = torch.tensor([[ 0.4410, -1.0317], [-0.2844, -0.1035]])
track, losses = [], []

for iter in range(21):
    x.requires_grad = True
    loss = torch.mm(bowl, x[:,None]).norm()
    loss.backward()
    with torch.no_grad():
        x = x - 0.1 * x.grad # type: ignore
    track.append(x.detach().clone())
    losses.append(loss.detach())

plot_progress(bowl, track, losses)

# %% [markdown]
# ### 10.2 内置优化算法
#
# PyTorch 还提供了不少现成的优化算法。
#
# 这些算法通常会在多次更新中加入一些技巧，让优化过程更快、更稳健。它们会根据当前目标函数曲面的形状，尝试调整更新方式。最简单的带动量方法是 SGD-with-momentum，对应 PyTorch 里的 `torch.optim.SGD`。
#
# ### 10.3 使用 SGD
#
# 使用 SGD 时，你需要先把目标算出来，并把所有参数的梯度填好，优化器才能执行一步更新。
#
# 1. 先把参数（这里是 `x`）设成 `x.requires_grad = True`，让 autograd 跟踪它。
# 2. 创建优化器，并告诉它要调整哪些参数（这里是 `[x]`）。
# 3. 在循环里先算目标，再调用 `loss.backward()` 把梯度写进 `x.grad`，最后调用 `optimizer.step()` 更新 `x`。
#
# **记得清零梯度。** 我们每次都先调用 `optimizer.zero_grad()`，把 `x.grad` 清零再重新算梯度；如果不这么做，新梯度会叠加到旧梯度上。

# %%
import torch

x = x_init.clone()
x.requires_grad = True
optimizer = torch.optim.SGD([x], lr=0.1, momentum=0.5)

bowl = torch.tensor([[ 0.4410, -1.0317], [-0.2844, -0.1035]])
track, losses = [], []

for iter in range(21):
    loss = torch.mm(bowl, x[:,None]).norm()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    track.append(x.detach().clone())
    losses.append(loss.detach())

plot_progress(bowl, track, losses)

# %% [markdown]
# ### 10.4 使用其他优化器
#
# 其他优化器的用法也类似。Adam 是一种很常用的自适应方法，通常不需要太多调参，可以直接替代普通 SGD。
#
# 有些更复杂的优化器，比如 LBFGS，需要你提供一个目标函数，让它自己反复调用并探测梯度。相关例子可以看 [hjmshi 的 LBFGS 示例](https://github.com/hjmshi/PyTorch-LBFGS/blob/master/examples/Other/lbfgs_tests.py#L129-L132)。

# %%
# 下面这段代码使用 Adam
x = x_init.clone()
x.requires_grad = True
optimizer = torch.optim.Adam([x], lr=0.1)

track, losses = [], []

for iter in range(21):
    loss = torch.mm(bowl, x[:,None]).norm()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    track.append(x.detach().clone())
    losses.append(loss.detach())

plot_progress(bowl, track, losses)

# %% [markdown]
# ### 10.5 其他技巧
#
# 1. **学习率调度。** 改善训练最简单、最有效的方法之一，是在训练过程中逐步降低学习率。学习率调度有很多不同策略，PyTorch 提供了一组 `torch.optim.lr_scheduler` 类，方便你直接套用。可以看 [Stack Overflow](https://stackoverflow.com/questions/48324152/) 或 [调度器文档](https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.StepLR)。
# 2. **多个优化器。** 有时你想优化的不止一个目标。最常见的做法，是把这些目标加权求和成一个总目标。但在某些场景下，你希望不同参数用不同目标来优化。这在对抗训练里很常见，比如 GAN 中两个网络彼此博弈。此时你可以为不同目标分别使用不同的优化器。可以参考 [GAN 实现示例](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py#L112-L113)。
 # %%
# %% [markdown]
# ## 11 神经网络模块
#
# PyTorch 使用 `torch.nn.Module` 类来表示神经网络。
#
# `Module` 本质上只是一个 **可调用** 的函数，但它可以：
#
# * 通过可训练的 `Parameter` tensor 来参数化，而且模块可以把这些参数列出来。
# * 由一组子 `Module` 组合而成，而这些子模块会继续贡献参数。
# * 通过列出命名参数和其他属性缓冲区来保存和加载。
#
# PyTorch 自带了几个基础网络模块，比如单层的 `Linear` 网络，或者把其他网络串起来的 `Sequential` 组合。当然，你也可以自己定义 `Module` 子类，只要声明 `Parameter` 属性，再用它们实现计算即可。
#
# 为了看清每个 `Module` 是怎样分担上面这些职责的，我们先看如何使用内置的 `Linear` 和 `Sequential` 模块。
#
# ### 11.1 使用 `torch.nn.Linear` 作为神经网络
#
# 线性层不只是一个好的入门例子，它本身就是所有神经网络最核心的工作马力，所以即使它很简单，也值得仔细看一遍。
#
# `torch.nn.Linear` 实现的是 `y = Ax + b` 这个函数：它接受 m 维输入 `x`，通过一个 n × m 的矩阵 `A`（具体数值叫做 `weight`）做乘法，再加上一个 n 维向量 `b`（具体数值叫做 `bias`），得到 n 维输出 `y`。我们可以像下面这样构造一个输入 3 维、输出 2 维的线性网络：

# %%
net = torch.nn.Linear(3, 2)
print(net)

# %% [markdown]
# 像任何 `Module` 一样，这个小网络也可以当作函数来运行。正如预期，当我们给它一个 3 维向量作为输入时，它会返回一个 2 维向量作为输出。

# %%
x = torch.tensor([[1.0, 0.0, 0.0]])
net(x) # wx+b

# %% [markdown]
# **PyTorch 网络默认按 batch 处理。** 上面向量数据里有双层嵌套，这是必须的，因为我们的 `Linear` 网络和普通的矩阵向量乘法不完全一样。按照惯例，PyTorch 的 `Module` 都是按批次处理数据的，所以如果你想喂给它一个 3 维向量，不能直接传一个向量，而要传一个只包含这一个向量的 singleton batch。
#
# 我们也可以一次送入多个输入，这里给出四个向量作为输入。网络会返回四个向量作为输出：

# %%
x_batch = torch.tensor([
    [1.0, 0. , 0. ],
    [0. , 1.0, 0. ],
    [0. , 0. , 1.0],
    [0. , 0. , 0. ],
])
print(x_batch.shape)
net(x_batch)

# %% [markdown]
# **参数默认是随机初始化的。** 这个线性层到底在计算什么怪东西？默认情况下，PyTorch 会把权重和偏置随机初始化。我们可以直接看这些随机参数。
#
# 当然，PyTorch 也提供了用正态分布或均匀分布初始化层的功能。关于权重初始化的更多信息见 [这里](https://pytorch.org/docs/stable/nn.init.html)。

# %%
print('weight is', net.weight)
print('bias is', net.bias)

# %% [markdown]
# **参数会被设成可用于 autograd 和优化的形式，并且可以枚举出来。** 上面可以看到，weight 和 bias 都是可训练参数，因为它们都属于 `Parameter` 类型。它们也都标记了 `requires_grad=True`，表示它们会参与 autograd 和训练优化。
#
# 这就是网络里仅有的两个可训练参数。我们可以通过 `net.named_parameters()` 把它们按名字列出来。

# %%
for name, param in net.named_parameters():
    print(f'{name} = {param}\n')

# %% [markdown]
# **可以通过保存 `state_dict` 来保存一个 Module。** `net.state_dict()` 和 `net.named_parameters()` 类似，但它返回的是对数据的分离引用，也就是 `requires_grad=False`，因此可以直接保存。对于更复杂的模块，`state_dict()` 里还可能包含一些保存网络状态所必需的、非可训练属性。

# %%
for k, v in net.state_dict().items():
    print(f'{k}: {v.type()}{tuple(v.shape)}')

import os
os.makedirs('checkpoints', exist_ok=True)
torch.save(net.state_dict(), 'checkpoints/linear.pth')

# %% [markdown]
# **可以用 `load_state_dict()` 重新载入保存好的 Module。** PyTorch 也提供了方便的 `torch.save` 和 `torch.load` 函数，用来把 `state_dict` 存到文件里再读回来。

# 之后如果要恢复状态：
# %%
net.load_state_dict(torch.load('checkpoints/linear.pth'))

# %% [markdown]
# ## 12 训练一个线性层
#
# 要训练一个网络，我们需要先定义一个分数，用来衡量当前结果离目标有多近。这个标量通常叫做 **objective** 或 **loss**。
#
# 比如，假设我们希望这个网络不管输入是什么，都输出 `[1, 1]`。那一个合理的损失函数就是和 `[1, 1]` 的均方距离，写成下面这样：

# %%
y_batch = net(x_batch)
loss = ((y_batch - torch.tensor([[1.0, 1.0]])) ** 2).sum(1).mean()
print(f'loss is {loss}')

# %% [markdown]
# 我们可以直接用 autograd 看看每个参数的小变化会怎样影响这个损失。

# %%
loss.backward()
print(f'weight is {net.weight} and grad is:\n{net.weight.grad}\n')
print(f'bias is {net.bias} and grad is:\n{net.bias.grad}\n')

# %% [markdown]
# **可以直接做最简单的梯度下降。** 为了改进这个层，我们可以用学习率 0.01 做最简单的梯度下降，也就是每个参数减去 0.01 倍的梯度。重复做下去，应该会越来越接近目标。
#
# 每当我们直接更新网络参数时，都需要用 `with torch.no_grad()` 临时关闭 autograd 机制。

# 下面这段保持为普通 Python 代码，兼容脚本执行。
# %%
net = torch.nn.Linear(3, 2)
log = []
for _ in range(10000):
    y_batch = net(x_batch)
    loss = ((y_batch - torch.tensor([[1.0, 1.0]])) ** 2).sum(1).mean()
    log.append(loss.item())
    net.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in net.parameters():
            p[...] -= 0.01 * p.grad # type: ignore
print(f'weight is {net.weight}\n')
print(f'bias is {net.bias}\n')

# %%
import matplotlib.pyplot as plt
plt.ylabel('loss')
plt.xlabel('iteration')
plt.plot(log)
plt.show()

# %% [markdown]
# 这样，我们就把一个简单的神经网络训练到了想要的目标，也就是输出一个常数 `[1.0, 1.0]`。经过几千次更新后，权重会接近零矩阵，而 bias 会变成 `[1.0, 1.0]`。
#
# **更现实的训练循环会在 GPU 上运行，使用更大的随机 batch 和现成的优化算法。**
#
# 下面我们重新创建一个随机初始化的 `Linear` 网络，并用 Adam 优化器在 GPU 上训练 1000 次。

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 下面这段也保持为普通 Python 代码，兼容脚本执行。
# %%
from torch.optim import Adam
net = torch.nn.Linear(3, 2)
optimizer = Adam(net.parameters(), lr=0.01)
# 把网络、目标值和训练输入都移到 GPU 上
net.to(device)
target = torch.tensor([[1.0, 1.0]], device=device)
log = []
for _ in range(1000):
    y_batch = net(torch.randn(100, 3, device=device))
    loss = ((y_batch - target) ** 2).sum(1).mean()
    log.append(loss.item())
    net.zero_grad()
    loss.backward()
    optimizer.step()
print(f'weight is {net.weight}\n')
print(f'bias is {net.bias}\n')

# %matplotlib inline
import matplotlib.pyplot as plt
plt.ylabel('loss')
plt.xlabel('iteration')
plt.plot(log)

# %% [markdown]
# ### 12.1 用可视化函数看网络输出
#
# 下面这个函数会画出分类目标和网络输出，方便我们直观看到模型学到了什么。

# %%
from matplotlib import pyplot as plt

def visualize_net(net, classify_target):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    grid = torch.stack([
        torch.linspace(-2, 2, 100)[None, :].expand(100, 100),
        torch.linspace(2, -2, 100)[:, None].expand(100, 100),
    ])
    x, y = grid
    target = classify_target(x, y)
    ax1.set_title('目标')
    ax1.imshow(target.float(), cmap='hot', extent=[-2,2,-2,2])
    ax2.set_title('网络输出')
    score = net(grid.permute(1, 2, 0).reshape(-1, 2).to(device)).softmax(1)
    ax2.imshow(score[:,1].reshape(100, 100).detach().cpu(), cmap='hot', extent=[-2,2,-2,2])
    plt.show()

# %% [markdown]
# ### 12.2 使用 `torch.nn.Sequential` 组合网络
#
# 和 `Linear` 不同，大多数网络都是由许多更小的网络组合出来的。最简单的方法就是把网络一个接一个串起来，让前一个模块的输出直接作为后一个模块的输入。例如，我们可以直接把几个 `Linear` 层串起来。
#
# **定义一个多层感知机。** 当然，如果只做线性变换，模型就不会比一个普通线性函数更有表达力。要在各层之间加一点非线性（例如把负数截断成 0，也就是 `ReLU`），我们就能得到一个 **多层感知机（Multilayer Perceptron）**，它是一个可以逼近任意函数的分段线性函数族。
#
# 下面看看如何把这个网络写成嵌套的 Sequential。

# %%
from collections import OrderedDict
from torch.nn import Linear, ReLU, Sequential

mlp = torch.nn.Sequential(OrderedDict([
    ('layer1', Sequential(Linear(2, 20), ReLU())),
    ('layer2', Sequential(Linear(20, 20), ReLU())),
    ('layer3', Sequential(Linear(20, 2)))
]))

print(mlp)
# 你也可以通过 `nn.Module` 的 `forward()` 函数来定义前向传播。

# %% [markdown]
# 上面这个例子里，我们嵌套了两层 Sequential。最外层里，我们定义并命名了三层。
#
# 每一层本身又是一个 Sequential：它先执行参数化的 `Linear` 运算，再接一个 `ReLU` 非线性截断。最里面这些步骤我们没有逐个命名，所以 Sequential 会自动给它们编号。
#
# **每个子模块都有完整限定名。** 我们可以通过 `net.named_modules()` 查看一个递归的子模块列表。

# %%
for n, c in mlp.named_modules():
    print(f'{n or "整个网络"} 是一个 {type(c).__name__}')

# %% [markdown]
# **一个模块的参数会包含它所有子模块的参数。** 我们可以通过名字把所有参数列出来看这一点。

# %%
for name, param in mlp.named_parameters():
    print(f'{name} has shape {tuple(param.shape)}')

# %% [markdown]
# 现在总共有六个参数：三层 `Linear` 各自对应一个 weight 和一个 bias。
#
# **训练一个分类器。** 这个稍微复杂一点的网络已经能表示更一般的函数了。比如，我们可以用这个结构来学习一个分类器函数。
#
# 假设我们要把平面上的点分成两类：在正弦曲线之上的是类 1，在正弦曲线之下的是类 0。下面是训练这个 MLP 的普通训练循环，使用 Adam 优化器。

# %%
device = 'cpu'
from torch.nn.functional import cross_entropy

def classify_target(x, y): # type: ignore
    return (y > (x * 3).sin()).long()

mlp.to(device)
optimizer = Adam(mlp.parameters(), lr=0.01)
for iteration in range(1024):
    in_batch = torch.randn(10000, 2, device=device)
    target_batch = classify_target(in_batch[:,0], in_batch[:,1])
    out_batch = mlp(in_batch)
    loss = cross_entropy(out_batch, target_batch)
    if iteration > 0:
        mlp.zero_grad()
        loss.backward()
        optimizer.step()
    if iteration == 2 ** iteration.bit_length() - 1:
        pred_batch = out_batch.max(1)[1]
        accuracy = (pred_batch == target_batch).float().sum() / len(in_batch)
        print(f'Iteration {iteration} accuracy: {accuracy}')
        visualize_net(mlp, classify_target)

# %% [markdown]
# **可以通过保存 `state_dict` 来保存一个网络。** 由于 `state_dict` 会把子模块的所有参数都收集起来，我们可以一次性全部保存。注意，因为每个参数都有完整限定名，如果我们只想加载其中一部分层，也可以手动挑出字典里的键并改名。

# %%
for k, v in mlp.state_dict().items():
    print(f'{k}: {v.dtype}{tuple(v.shape)}')

torch.save(mlp.state_dict(), 'checkpoints/mlp.pth')

# %% [markdown]
# ### 12.3 用 `forward` 定义自定义网络
#
# 有时你想把网络组件按比顺序堆叠更复杂的方式连接起来。
#
# 比如 [ResNet](https://arxiv.org/abs/1512.03385) 的核心观察是：如果不是学习一个任意线性变换，而是学习恒等映射的扰动，训练效果往往会好很多。也就是说，让某一层去学一个小的残差，而不是把整个答案从头学出来。
#
# 为了在我们这个三层小网络里用上残差技巧，就不能只靠一个整体的 `Sequential`；我们要自己写 `forward` 函数。它长这样：

class MyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Sequential(Linear(2, 20), ReLU())
        self.residual_layer2 = Sequential(Linear(20, 20), ReLU())
        self.layer3 = Linear(20, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = x + self.residual_layer2(x)
        x = self.layer3(x)
        return x

res_mlp = MyNetwork()
print(res_mlp)

# %% [markdown]
# ## 13 其他 `Module` 技巧
#
# `torch.nn.Module` 还有几个常用的“边角能力”，值得顺手记住。
#
# `torch.nn.Parameter` 用来包装可训练参数。你可以在 `__init__()` 里把某个张量包成 `Parameter`，然后挂到模块属性上，这样它就会被优化器自动处理。可以直接看 PyTorch 的 `Linear` 源码了解写法。这里 `in_features` 和 `out_features` 只是普通数值，不参与训练；真正会被训练的是 `weight` 和 `bias`。
#
# `module.training` 控制模块在训练态和推理态之间切换。有些模块在这两种模式下行为不同。比如 `Dropout` 在训练时会随机丢掉一部分通道，但在推理时会保留所有通道。对应地，`module.train()` 会把模块递归切到训练模式，`module.eval()` 会切到推理模式。`module.training` 这个布尔值就表示当前模式。
#
# `buffers` 也可以保存进模块里，但它们不一定要交给优化器更新。模块里并不是所有属性都适合做可训练参数。像 `BatchNorm` 就会在训练过程中统计均值和方差，再把这些统计量累积成内部状态，用来维持更稳定的激活分布。
#
# 另外，PyTorch 还提供了很多预定义模型结构。比如 `torchvision.models.resnet18(num_classes=100)` 会直接创建一个 ResNet-18 分类器，并配置成 100 类分类任务。

# %%
def classify_target(x, y):
    return (y > (x * 3).sin()).long()

res_mlp.to(device)
optimizer = Adam(res_mlp.parameters(), lr=0.01)
for iteration in range(1024):
    in_batch = torch.randn(10000, 2, device=device)
    target_batch = classify_target(in_batch[:,0], in_batch[:,1])
    out_batch = res_mlp(in_batch)
    loss = cross_entropy(out_batch, target_batch)
    if iteration > 0:
        res_mlp.zero_grad()
        loss.backward()
        optimizer.step()
    if iteration == 2 ** iteration.bit_length() - 1:
        pred_batch = out_batch.max(1)[1]
        accuracy = (pred_batch == target_batch).float().sum() / len(in_batch)
        print(f'Iteration {iteration} accuracy: {accuracy}')
        visualize_net(res_mlp, classify_target)

# %% [markdown]
# ## 14 数据集与 DataLoader
#
# 处理数据样本的代码很容易变得杂乱，也不利于维护。更合理的做法，是把数据集代码和模型训练代码尽量解耦，这样可读性和模块化都会更好。PyTorch 提供了两个基础数据抽象：`torch.utils.data.Dataset` 和 `torch.utils.data.DataLoader`。`Dataset` 负责存放样本及其标签，`DataLoader` 则在 `Dataset` 外面包一层可迭代接口，方便我们按批次取数据。
#
# PyTorch 的各个领域库也提供了一批预置数据集，比如 FashionMNIST。它们都继承自 `torch.utils.data.Dataset`，并为特定数据实现了额外接口。你可以用这些数据集来快速做原型验证和模型对比。相关文档分别在 [Image Datasets](https://pytorch.org/vision/stable/datasets.html)、[Text Datasets](https://pytorch.org/text/stable/datasets.html) 和 [Audio Datasets](https://pytorch.org/audio/stable/datasets.html)。

# %% [markdown]
# ### 14.1 加载数据集
#
# 下面演示如何从 TorchVision 加载 [Fashion-MNIST](https://research.zalando.com/project/fashion_mnist/fashion_mnist/) 数据集。Fashion-MNIST 来自 Zalando 的商品图片，共包含 60,000 个训练样本和 10,000 个测试样本。每个样本都是一张 28×28 的灰度图像，对应 10 个类别中的一个标签。
#
# 我们加载 [FashionMNIST Dataset](https://pytorch.org/vision/stable/datasets.html#fashion-mnist) 时，常用参数如下：
#
# - `root` 表示训练集和测试集的存放路径。
# - `train` 指定当前加载的是训练集还是测试集。
# - `download=True` 表示如果本地没有数据，就从网络下载。
# - `transform` 和 `target_transform` 分别定义输入特征和标签的变换。

# %%
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# %% [markdown]
# ### 14.2 遍历并可视化数据集
#
# 我们可以像访问列表一样手动索引 `Dataset`，比如 `training_data[index]`。下面用 `matplotlib` 随机展示训练集中的几个样本。

# %%
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx] # type: ignore
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# %% [markdown]
# ### 14.3 为自己的文件创建自定义数据集
#
# 自定义 `Dataset` 类通常要实现三个函数：`__init__`、`__len__` 和 `__getitem__`。下面这个例子里，FashionMNIST 风格的图片存放在目录 `img_dir` 中，标签则单独放在 CSV 文件 `annotations_file` 里。
#
# #### 14.3.1 `__init__`
#
# `__init__` 在实例化 `Dataset` 时只运行一次。我们会在这里初始化图片目录、标注文件，以及两个变换函数；变换的细节会在后面再讲。
#
# `labels.csv` 文件大致长这样：
#
#     tshirt1.jpg, 0
#     tshirt2.jpg, 0
#     ......
#     ankleboot999.jpg, 9
#
# #### 14.3.2 `__len__`
#
# `__len__` 返回这个数据集里样本的总数。
#
# #### 14.3.3 `__getitem__`
#
# `__getitem__` 会根据索引 `idx` 取出并返回一个样本。它先找到对应图片在磁盘上的路径，再用 `read_image` 读成张量，从 `self.img_labels` 里取出对应标签，必要时对图像和标签分别执行变换，最后返回 `(image, label)`。

# %%
import os
import pandas as pd
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # type: ignore
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# %% [markdown]
# ### 14.4 用 DataLoader 准备训练数据
#
# `Dataset` 一次只返回一个样本。真正训练模型时，我们通常希望按小批量取数据，在每个 epoch 重新打乱样本顺序来降低过拟合风险，并且利用 Python 的 `multiprocessing` 加速数据读取。
#
# `DataLoader` 就是把这些细节封装起来的可迭代接口。

# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# %% [markdown]
# ### 14.5 遍历 DataLoader
#
# 数据已经放进 `DataLoader` 之后，就可以按需遍历了。下面每次迭代都会返回一批 `train_features` 和 `train_labels`，其中 `batch_size=64`。由于我们设置了 `shuffle=True`，所有 batch 遍历完后数据顺序会被重新打乱。若想进一步控制数据读取顺序，可以参考 [Samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)。

# %%
# 显示图片和标签。

for train_features, train_labels in train_dataloader:
  print(f"Feature batch shape: {train_features.size()}")
  print(f"Labels batch shape: {train_labels.size()}")
  img = train_features[0].squeeze()
  label = train_labels[0]
  plt.imshow(img, cmap="gray")
  plt.show()
  print(f"Label: {label}")
  break
