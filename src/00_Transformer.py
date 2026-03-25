#!/usr/bin/env python
# coding: utf-8

# 
# <center><h1>图解 Transformer (The Annotated Transformer)</h1> </center>
# 
# 
# <center>
# <p><a href="https://arxiv.org/abs/1706.03762">Attention is All You Need
# </a></p>
# </center>
# 
# <div align="center"><img src="../images/00_authors.png" width="70%"/></div>
# 
# 
# 本文以逐行代码实现的形式展示了该论文的注释版本。它对原论文的一些章节进行了重新排序和删减，并在全文中添加了注释。
# 本文档本身是一个可运行的 Notebook，是一个完全可用的实现。
# 代码可在 [此处](https://github.com/harvardnlp/annotated-transformer/) 获取。
# 

# <h3> 目录 </h3>
# <ul>
# <li><a href="#prelims">准备工作 (Prelims)</a></li>
# <li><a href="#background">背景 (Background)</a></li>
# <li><a href="#part-1-model-architecture">第一部分：模型架构</a></li>
# <li><a href="#model-architecture">模型架构</a><ul>
# <li><a href="#encoder-and-decoder-stacks">编码器和解码器堆栈</a></li>
# <li><a href="#position-wise-feed-forward-networks">逐位置前馈网络</a></li>
# <li><a href="#embeddings-and-softmax">嵌入和 Softmax</a></li>
# <li><a href="#positional-encoding">位置编码</a></li>
# <li><a href="#full-model">完整模型</a></li>
# <li><a href="#inference">推理：</a></li>
# </ul></li>
# <li><a href="#part-2-model-training">第二部分：模型训练</a></li>
# <li><a href="#training">训练</a><ul>
# <li><a href="#batches-and-masking">批处理和掩码</a></li>
# <li><a href="#training-loop">训练循环</a></li>
# <li><a href="#training-data-and-batching">训练数据和批处理</a></li>
# <li><a href="#hardware-and-schedule">硬件和进度</a></li>
# <li><a href="#optimizer">优化器</a></li>
# <li><a href="#regularization">正则化</a></li>
# </ul></li>
# <li><a href="#a-first-example">第一个例子</a><ul>
# <li><a href="#synthetic-data">合成数据</a></li>
# <li><a href="#loss-computation">损失计算</a></li>
# <li><a href="#greedy-decoding">贪婪解码</a></li>
# </ul></li>
# <li><a href="#part-3-a-real-world-example">第三部分：真实世界示例</a>
# <ul>
# <li><a href="#data-loading">数据加载</a></li>
# <li><a href="#iterators">迭代器</a></li>
# <li><a href="#training-the-system">训练系统</a></li>
# </ul></li>
# <li><a href="#additional-components-bpe-search-averaging">附加组件：BPE、搜索、平均</a></li>
# <li><a href="#results">结果</a><ul>
# <li><a href="#attention-visualization">注意力可视化</a></li>
# <li><a href="#encoder-self-attention">编码器自注意力</a></li>
# <li><a href="#decoder-self-attention">解码器自注意力</a></li>
# <li><a href="#decoder-src-attention">解码器源注意力</a></li>
# </ul></li>
# <li><a href="#conclusion">结论</a></li>
# </ul>

# # 准备工作 (Prelims)
# 
# <a href="#background">跳过</a>

# In[ ]:


# !pip install -r requirements.txt


# In[ ]:


# # 在 Colab 中取消注释
# #
# !pip install -q torchdata==0.3.0 torchtext==0.12 spacy==3.2 altair GPUtil
# !python -m spacy download de_core_news_sm
# !python -m spacy download en_core_web_sm


# In[ ]:


import copy
import math
import os
import time
import warnings
from os.path import exists

import altair as alt
import GPUtil
import pandas as pd
import spacy
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchtext.datasets as datasets
from torch.nn.functional import log_softmax, pad
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator

# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True


# In[ ]:


# Some convenience helper functions used throughout the notebook


def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None


# > 我的评论以引用块形式呈现。主体文本均来自论文本身。

# # 背景

# 
# 减少顺序计算的目标也是 Extended Neural GPU、ByteNet 和 ConvS2S 的基础，它们都使用卷积神经网络作为基础构建块，并行计算所有输入和输出位置的隐藏表示。在这些模型中，关联两个任意输入或输出位置信号所需的操作数随位置间距离而增长，ConvS2S 呈线性增长，ByteNet 呈对数增长。这使得学习远距离位置之间的依赖关系变得更加困难。在 Transformer 中，这被减少到恒定数量的操作，尽管代价是由于平均注意力加权位置而导致有效分辨率降低，我们通过多头注意力（Multi-Head Attention）来抵消这种影响。
# 
# 自注意力（Self-attention），有时被称为内部注意力，是一种关联单个序列不同位置以计算序列表示的注意力机制。自注意力已成功应用于各种任务，包括阅读理解、摘要提取、文本蕴含和学习任务无关的句子表示。端到端记忆网络基于循环注意力机制而非序列对齐的循环，并已被证明在简单语言问答和语言建模任务上表现良好。
# 
# 然而，据我们所知，Transformer 是第一个完全依靠自注意力来计算其输入和输出表示而不使用序列对齐 RNN 或卷积的转导模型。

# # 第一部分：模型架构

# # 模型架构

# 
# 大多数具有竞争力的神经序列转导模型都具有编码器-解码器结构 [(引用)](https://arxiv.org/abs/1409.0473)。在这里，编码器将符号表示的输入序列 $(x_1, ..., x_n)$ 映射到连续表示序列 $\mathbf{z} = (z_1, ..., z_n)$。给定 $\mathbf{z}$，解码器随后每次生成一个符号的输出序列 $(y_1,...,y_m)$。在每一步中，模型都是自回归的 [(引用)](https://arxiv.org/abs/1308.0850)，在生成下一个符号时，将先前生成的符号作为额外输入。

# In[ ]:


class EncoderDecoder(nn.Module):
    """
    标准的编码器-解码器架构。是此模型及许多其他模型的基础。
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "接收并处理掩码后的源序列和目标序列。"
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# In[ ]:


class Generator(nn.Module):
    "定义标准的线性 + softmax 生成步骤。"

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


# 
# Transformer 遵循这种整体架构，编码器和解码器都使用堆叠的自注意力和逐点全连接层，分别如图 1 的左半部分和右半部分所示。

# <div align="center"><img src="../images/00_arch.png" width="70%"/></div>

# ## 编码器和解码器堆栈
# 
# ### 编码器
# 
# 编码器由 $N=6$ 个相同层的堆栈组成。

# In[ ]:


def clones(module, N):
    "产生 N 个相同的层。"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# In[ ]:


class Encoder(nn.Module):
    "核心编码器是由 N 层堆叠而成"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "依次将输入（和掩码）通过每一层。"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# 
# 我们在两个子层中的每一个周围都采用了残差连接 [(引用)](https://arxiv.org/abs/1512.03385)，随后是层归一化 [(引用)](https://arxiv.org/abs/1607.06450)。

# In[ ]:


class LayerNorm(nn.Module):
    "构建一个层归一化模块（详情见引用）。"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 
# 也就是说，每个子层的输出是 $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$，其中 $\mathrm{Sublayer}(x)$ 是由子层本身实现的函数。我们在将 dropout [(引用)](http://jmlr.org/papers/v15/srivastava14a.html) 应用于每个子层的输出之前，先将其添加到子层输入并进行归一化。
# 
# 为了方便这些残差连接，模型中的所有子层以及嵌入层产生的输出维度均为 $d_{\text{model}}=512$。

# In[ ]:


class SublayerConnection(nn.Module):
    """
    残差连接后接层归一化。
    注意：为了代码简洁，归一化是在前面，而不是在最后。
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "将残差连接应用于任何具有相同大小的子层。"
        return x + self.dropout(sublayer(self.norm(x)))


# 
# 每一层有两个子层。第一个是多头自注意力机制，第二个是简单的逐位置全连接前馈网络。

# In[ ]:


class EncoderLayer(nn.Module):
    "编码器由自注意力和前馈网络组成（定义如下）"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "遵循图 1（左）的连接方式。"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# ### 解码器
# 
# 解码器也由 $N=6$ 个相同层的堆栈组成。
# 

# In[ ]:


class Decoder(nn.Module):
    "具有掩码的通用 N 层解码器。"

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# 
# 除了每个编码器层中的两个子层之外，解码器还插入了第三个子层，该子层对编码器堆栈的输出执行多头注意力。与编码器类似，我们在每个子层周围采用残差连接，随后进行层归一化。

# In[ ]:


class DecoderLayer(nn.Module):
    "解码器由自注意力、源注意力和前馈网络组成（定义如下）"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "遵循图 1（右）的连接方式。"
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


# 
# 我们还修改了解码器堆栈中的自注意力子层，以防止当前位置注意到后续位置。这种掩码结合输出嵌入偏移一个位置的事实，确保对位置 $i$ 的预测只能依赖于小于 $i$ 位置的已知输出。
# 
# 

# In[ ]:


def subsequent_mask(size):
    "掩盖后续位置。"
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


# 
# > 下面的注意力掩码显示了每个目标词（行）允许查看的位置（列）。在训练期间，单词被阻止查看未来的单词。

# In[ ]:


def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


show_example(example_mask)


# ### 注意力 (Attention)
# 
# 注意力函数可以描述为将查询（query）和一组键值对（key-value pairs）映射到输出，其中查询、键、值和输出都是向量。输出计算为值的加权和，其中分配给每个值的权重由查询与相应键的兼容性函数计算。
# 
# 我们称这种特定的注意力为“缩放点积注意力”（Scaled Dot-Product Attention）。输入由维度为 $d_k$ 的查询和键以及维度为 $d_v$ 的值组成。我们计算查询与所有键的点积，每个除以 $\sqrt{d_k}$，然后应用 softmax 函数来获得值的权重。
# 
# 
# 
# <div align="center"><img src="../images/ModalNet-19.png" width="70%"/></div>

# 
# 在实践中，我们同时在一组查询上计算注意力函数，这些查询被打包成矩阵 $Q$。键和值也被打包成矩阵 $K$ 和 $V$。我们计算输出矩阵为：
# 
# $$
#    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
# $$

# In[ ]:


def attention(query, key, value, mask=None, dropout=None):
    "计算“缩放点积注意力”"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# 
# 两种最常用的注意力函数是加法注意力 [(引用)](https://arxiv.org/abs/1409.0473) 和点积（乘法）注意力。点积注意力与我们的算法相同，除了 $\frac{1}{\sqrt{d_k}}$ 的缩放因子。加法注意力使用具有单个隐藏层的全连接网络计算兼容性函数。虽然两者在理论复杂度上相似，但在实践中点积注意力更快且更节省空间，因为它可以使用高度优化的矩阵乘法代码来实现。
# 
# 
# 虽然对于较小的 $d_k$ 值，这两种机制的表现相似，但在没有缩放的情况下，对于较大的 $d_k$ 值，加法注意力的表现优于点积注意力 [(引用)](https://arxiv.org/abs/1703.03906)。我们怀疑对于较大的 $d_k$ 值，点积在幅度上增长得很大，从而将 softmax 函数推入梯度极小的区域（为了说明点积为什么会变大，假设 $q$ 和 $k$ 的分量是均值为 $0$、方差为 $1$ 的独立随机变量。那么它们的点积 $q \cdot k = \sum_{i=1}^{d_k} q_ik_i$ 的均值为 $0$，方差为 $d_k$。）。为了抵消这种影响，我们将点积缩放 $\frac{1}{\sqrt{d_k}}$。
# 
# 

# <div align="center"><img src="../images/ModalNet-20.png" width="70%"/></div>

# 
# 多头注意力允许模型在不同位置共同关注来自不同表示子空间的信息。对于单个注意力头，平均化会抑制这一点。
# 
# $$
# \mathrm{MultiHead}(Q, K, V) =
#     \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \\
#     \text{其中}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
# $$
# 
# 其中投影是参数矩阵 $W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$、$W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ 和 $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$。
# 
# 在这项工作中，我们采用 $h=8$ 个并行注意力层或头。对于其中的每一个，我们使用 $d_k=d_v=d_{\text{model}}/h=64$。由于每个头的维度降低，总计算成本与具有全维度的单头注意力相似。

# In[ ]:


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "传入模型大小和头数。"
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # 我们假设 d_v 始终等于 d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "实现图 2"
        if mask is not None:
            # 相同的掩码应用于所有 h 个头。
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) 批量执行从 d_model => h x d_k 的所有线性投影
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) 在批次中对所有投影向量应用注意力。
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 使用 view 进行“拼接”并应用最终的线性层。
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)


# ### 注意力在模型中的应用
# 
# Transformer 以三种不同的方式使用多头注意力：
# 1) 在“编码器-解码器注意力”层中，查询来自上一个解码器层，而存储的键和值来自编码器的输出。这允许解码器中的每个位置都关注输入序列中的所有位置。这模仿了序列到序列模型中典型的编码器-解码器注意力机制，例如 [(引用)](https://arxiv.org/abs/1609.08144)。
# 
# 
# 2) 编码器包含自注意力层。在自注意力层中，所有的键、值和查询都来自同一个地方，在本例中是编码器中前一层的输出。编码器中的每个位置都可以关注编码器前一层中的所有位置。
# 
# 
# 3) 类似地，解码器中的自注意力层允许解码器中的每个位置关注解码器中直到并包括该位置的所有位置。我们需要防止解码器中的左向信息流，以保持自回归属性。我们在缩放点积注意力内部通过屏蔽（设置为 $-\infty$）softmax 输入中对应于非法连接的所有值来实现这一点。

# ## 逐位置前馈网络
# 
# 除了注意力子层外，我们的编码器和解码器中的每一层都包含一个全连接的前馈网络，该网络分别且相同地应用于每个位置。这由两个线性变换组成，中间有一个 ReLU 激活。
# 
# $$\mathrm{FFN}(x)=\max(0, xW_1 + b_1) W_2 + b_2$$
# 
# 虽然线性变换在不同位置上是相同的，但它们在层与层之间使用不同的参数。描述它的另一种方式是两个内核大小为 1 的卷积。输入和输出的维度为 $d_{\text{model}}=512$，内层的维度为 $d_{ff}=2048$。

# In[ ]:


class PositionwiseFeedForward(nn.Module):
    "实现 FFN 方程。"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


# ## 嵌入和 Softmax
# 
# 与其他序列转导模型类似，我们使用学习到的嵌入将输入标记和输出标记转换为维度为 $d_{\text{model}}$ 的向量。我们还使用通常的学习线性变换和 softmax 函数将解码器输出转换为预测的下一个标记概率。在我们的模型中，我们在两个嵌入层和预 softmax 线性变换之间共享相同的权重矩阵，类似于 [(引用)](https://arxiv.org/abs/1608.05859)。在嵌入层中，我们将这些权重乘以 $\sqrt{d_{\text{model}}}$。

# In[ ]:


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# ## 位置编码
# 
# 由于我们的模型不包含循环和卷积，为了使模型能够利用序列的顺序，我们必须注入一些关于序列中标记相对或绝对位置的信息。为此，我们在编码器和解码器堆栈底部的输入嵌入中添加了“位置编码”。位置编码具有与嵌入相同的维度 $d_{\text{model}}$，因此两者可以相加。位置编码有很多选择，包括学习的和固定的 [(引用)](https://arxiv.org/pdf/1705.03122.pdf)。
# 
# 在这项工作中，我们使用不同频率的正弦和余弦函数：
# 
# $$PE_{(pos,2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
# 
# $$PE_{(pos,2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$
# 
# 其中 $pos$ 是位置，$i$ 是维度。也就是说，位置编码的每个维度都对应一个正弦曲线。波长形成从 $2\pi$ 到 $10000 \cdot 2\pi$ 的几何级数。我们选择这个函数是因为我们假设它能让模型轻松学习通过相对位置进行关注，因为对于任何固定的偏移量 $k$，$PE_{pos+k}$ 都可以表示为 $PE_{pos}$ 的线性函数。
# 
# 此外，我们将 dropout 应用于编码器和解码器堆栈中嵌入和位置编码的总和。对于基础模型，我们使用 $P_{drop}=0.1$ 的速率。
# 
# 

# In[ ]:


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


# 
# > 下面的位置编码将根据位置添加正弦波。波的频率和偏移对于每个维度都是不同的。

# In[ ]:


def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


show_example(example_positional)


# 
# 我们还尝试了使用学习到的位置嵌入 [(引用)](https://arxiv.org/pdf/1705.03122.pdf)，发现这两个版本产生的结果几乎相同。我们选择了正弦版本，因为它可能允许模型外推到比训练期间遇到的序列长度更长的序列。

# ## 完整模型
# 
# > 这里我们定义一个从超参数到完整模型的函数。

# In[ ]:


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "辅助函数：从超参数构建模型。"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # 这在他们的代码中很重要。
    # 使用 Glorot / fan_avg 初始化参数。
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ## 推理：
# 
# > 这里我们进行前向步骤来生成模型的预测。我们尝试使用我们的 transformer 来记忆输入。如你所见，由于模型尚未训练，输出是随机生成的。在下一个教程中，我们将构建训练函数并尝试训练我们的模型来记忆数字 1 到 10。

# In[ ]:


def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("未训练模型的预测示例:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


show_example(run_tests)


# # 第二部分：模型训练

# # 训练
# 
# 本节描述了我们模型的训练方案。

# 
# > 我们停下来做一个简短的插曲，介绍一些训练标准编码器解码器模型所需的工具。首先，我们定义一个批处理对象，它保存训练用的源句子和目标句子，并构建掩码。

# ## 批处理和掩码

# In[ ]:


class Batch:
    """用于在训练期间保存带掩码的数据批次的物体。"""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "创建一个掩码来隐藏填充和未来的单词。"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


# 
# > Next we create a generic training and scoring function to keep
# > track of loss. We pass in a generic loss compute function that
# > also handles parameter updates.

# ## 训练循环

# In[ ]:


class TrainState:
    """跟踪处理的步数、样本数和标记数"""

    step: int = 0  # 当前 epoch 中的步数
    accum_step: int = 0  # 梯度累积步数
    samples: int = 0  # 使用的样本总数
    tokens: int = 0  # 处理的标记总数


# In[ ]:


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """训练单个 epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


# ## 训练数据和批处理
# 
# 我们在标准的 WMT 2014 英德数据集上进行了训练，该数据集包含约 450 万个句子对。句子使用字节对编码（BPE）进行编码，源语言和目标语言共享约 37000 个标记的词汇表。对于英法翻译，我们使用了显著更大的 WMT 2014 英法数据集，包含 3600 万个句子，并将标记拆分为 32000 个字词（word-piece）词汇表。
# 
# 
# 句子对根据近似序列长度进行批处理。每个训练批次包含一组句子对，包含约 25000 个源标记和 25000 个目标标记。

# ## 硬件和进度
# 
# 我们在一台配备 8 块 NVIDIA P100 GPU 的机器上训练了我们的模型。对于使用整篇论文中描述的超参数的基础模型，每个训练步骤大约需要 0.4 秒。我们对基础模型进行了总计 100,000 步（约 12 小时）的训练。对于我们的大模型，每步时间为 1.0 秒。大模型训练了 300,000 步（3.5 天）。

# ## 优化器
# 
# 我们使用了 Adam 优化器 [(引用)](https://arxiv.org/abs/1412.6980)，其中 $\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$。我们在训练过程中根据以下公式改变学习率：
# 
# $$
# lrate = d_{\text{model}}^{-0.5} \cdot
#   \min({step\_num}^{-0.5},
#     {step\_num} \cdot {warmup\_steps}^{-1.5})
# $$
# 
# 这对应于在第一个 $warmup\_steps$ 训练步骤中线性增加学习率，之后与步数的平方根倒数成比例减少。我们使用了 $warmup\_steps=4000$。

# 
# > 注意：这部分非常重要。需要使用这种模型设置进行训练。

# 
# > 该模型在不同模型大小和优化超参数下的曲线示例。

# In[ ]:


def rate(step, model_size, factor, warmup):
    """
    我们必须将 LambdaLR 函数的 step 默认为 1，以避免零的负幂次方。
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


# In[ ]:


def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # 示例 1
        [512, 1, 8000],  # 示例 2
        [256, 1, 4000],  # 示例 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # 我们在 opts 列表中有 3 个示例。
    for idx, example in enumerate(opts):
        # 为每个示例运行 20000 个 epoch
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # 进行 20K 次虚拟训练步骤，保存每一步的学习率
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # 允许 altair 处理超过 5000 行的数据
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
        .interactive()
    )


example_learning_schedule()


# ## 正则化
# 
# ### 标签平滑 (Label Smoothing)
# 
# 在训练期间，我们采用了值为 $\epsilon_{ls}=0.1$ 的标签平滑 [(引用)](https://arxiv.org/abs/1512.00567)。这会损害困惑度（perplexity），因为模型学得更加不确定，但提高了准确率和 BLEU 分数。

# 
# > 我们使用 KL 散度损失来实现标签平滑。我们不使用 one-hot 目标分布，而是创建一个分布，该分布在正确单词上具有 `confidence`（置信度），而其余的 `smoothing`（平滑）质量分布在整个词汇表中。

# In[ ]:


class LabelSmoothing(nn.Module):
    "实现标签平滑。"

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


# 
# > 在这里我们可以看到一个示例，说明如何根据置信度将质量分布到单词中。

# In[ ]:


# 标签平滑示例。


def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color("target distribution:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


show_example(example_label_smoothing)


# 
# > 如果标签平滑对某个选择变得非常自信，它实际上会开始惩罚模型。

# In[ ]:


def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )


show_example(penalization_visualization)


# # 第一个例子
# 
# > 我们可以从尝试一个简单的复制任务开始。给定来自小词汇表的一组随机输入符号，目标是重新生成这些相同的符号。

# ## 合成数据
# 
# 

# In[ ]:


def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


# ## Loss Computation

# In[ ]:


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        return sloss.data * norm, sloss


# ## Greedy Decoding

# > This code predicts a translation using greedy decoding for simplicity.

# In[ ]:


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


# In[ ]:


# Train the simple copy task.


def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


# execute_example(example_simple_model)


# # 第三部分：真实世界示例
# 
# > 现在我们考虑一个使用 Multi30k 德英翻译任务的真实世界示例。这个任务比论文中考虑的 WMT 任务小得多，但它说明了整个系统。我们还展示了如何使用多 GPU 处理来使其变得非常快。

# ## 数据加载
# 
# > 我们将使用 torchtext 和 spacy 来加载数据集并进行分词。

# In[ ]:


# 加载 spacy 分词器模型，如果尚未下载，则下载它们


def load_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


# In[ ]:


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


# In[ ]:


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    print("Building German Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_de, index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


if is_interactive_notebook():
    # global variables used later in the script
    spacy_de, spacy_en = show_example(load_tokenizers)
    vocab_src, vocab_tgt = show_example(load_vocab, args=[spacy_de, spacy_en])


# 
# > Batching matters a ton for speed. We want to have very evenly
# > divided batches, with absolutely minimal padding. To do this we
# > have to hack a bit around the default torchtext batching. This
# > code patches their default batching to make sure we search over
# > enough sentences to find tight batches.

# ## Iterators

# In[ ]:


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)  # <s> token id
    eos_id = torch.tensor([1], device=device)  # </s> token id
    src_list, tgt_list = [], []
    for _src, _tgt in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab(src_pipeline(_src)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab(tgt_pipeline(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


# In[ ]:


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    # def create_dataloaders(batch_size=12000):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_iter, valid_iter, test_iter = datasets.Multi30k(language_pair=("de", "en"))

    train_iter_map = to_map_style_dataset(
        train_iter
    )  # DistributedSampler needs a dataset len()
    train_sampler = DistributedSampler(train_iter_map) if is_distributed else None
    valid_iter_map = to_map_style_dataset(valid_iter)
    valid_sampler = DistributedSampler(valid_iter_map) if is_distributed else None

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_iter_map,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


# ## Training the System

# In[ ]:


def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    config,
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.cuda(gpu)
    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda(gpu)

    train_dataloader, valid_dataloader = create_dataloaders(
        gpu,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config["warmup"]),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


# In[ ]:


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    from the_annotated_transformer import train_worker

    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    else:
        train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False)


def load_trained_model():
    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "file_prefix": "multi30k_model_",
    }
    model_path = "multi30k_model_final.pt"
    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt"))
    return model


if is_interactive_notebook():
    model = load_trained_model()


# 
# > 训练完成后，我们可以对模型进行解码以产生一组翻译。在这里，我们简单地翻译验证集中的第一个句子。这个数据集非常小，所以使用贪婪搜索的翻译相当准确。

# # 附加组件：BPE、搜索、平均

# 
# > 这主要涵盖了 transformer 模型本身。有四个方面我们没有明确涵盖。我们还在 [OpenNMT-py](https://github.com/opennmt/opennmt-py) 中实现了所有这些附加功能。
# 
# 

# 
# > 1) BPE/ Word-piece：我们可以使用一个库先将数据预处理为子词单元。参见 Rico Sennrich 的 [subword-nmt](https://github.com/rsennrich/subword-nmt) 实现。这些模型将转换训练数据，使其看起来像这样：

# ▁Die ▁Protokoll datei ▁kann ▁ heimlich ▁per ▁E - Mail ▁oder ▁FTP
# ▁an ▁einen ▁bestimmte n ▁Empfänger ▁gesendet ▁werden .

# 
# > 2) 共享嵌入：当使用具有共享词汇表的 BPE 时，我们可以在源/目标/生成器之间共享相同的权重向量。详情请参见 [(引用)](https://arxiv.org/abs/1608.05859)。要将其添加到模型中，只需执行以下操作：

# In[ ]:


if False:
    model.src_embed[0].lut.weight = model.tgt_embeddings[0].lut.weight
    model.generator.lut.weight = model.tgt_embed[0].lut.weight


# 
# > 3) 束搜索 (Beam Search)：这有点太复杂了，在这里无法涵盖。有关 pytorch 实现，请参见 [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py/)。
# >
# 

# 
# > 4) 模型平均：论文对最后 k 个检查点进行平均以产生集成效果。如果我们有一堆模型，我们可以在事后这样做：

# In[ ]:


def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


# # 结果
# 
# 在 WMT 2014 英德翻译任务上，大 transformer 模型（表 2 中的 Transformer (big)）比之前报告的最佳模型（包括集成模型）高出 2.0 BLEU 以上，建立了 28.4 的新最先进 BLEU 分数。该模型的配置列在表 3 的最后一行。在 8 块 P100 GPU 上训练耗时 3.5 天。即使是我们的基础模型也超过了所有之前发布的模型和集成模型，而训练成本仅为任何竞争模型的一小部分。
# 
# 在 WMT 2014 英法翻译任务上，我们的大模型实现了 41.0 的 BLEU 分数，优于所有之前发布的单一模型，训练成本不到之前最先进模型的 1/4。用于英法翻译的 Transformer (big) 模型使用的 dropout 率为 Pdrop = 0.1，而不是 0.3。
# 

# <div align="center">![](../images/results.png)</div>

# 
# 
# > 结合上一节中的附加扩展，OpenNMT-py 的复现版本在 EN-DE WMT 上达到了 26.9。在这里，我已将这些参数加载到我们的重新实现中。

# In[ ]:


# Load data and model for output checks


# In[ ]:


def check_outputs(
    valid_dataloader,
    model,
    vocab_src,
    vocab_tgt,
    n_examples=15,
    pad_idx=2,
    eos_string="</s>",
):
    results = [()] * n_examples
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(iter(valid_dataloader))
        rb = Batch(b[0], b[1], pad_idx)
        greedy_decode(model, rb.src, rb.src_mask, 64, 0)[0]

        src_tokens = [vocab_src.get_itos()[x] for x in rb.src[0] if x != pad_idx]
        tgt_tokens = [vocab_tgt.get_itos()[x] for x in rb.tgt[0] if x != pad_idx]

        print("Source Text (Input)        : " + " ".join(src_tokens).replace("\n", ""))
        print("Target Text (Ground Truth) : " + " ".join(tgt_tokens).replace("\n", ""))
        model_out = greedy_decode(model, rb.src, rb.src_mask, 72, 0)[0]
        model_txt = (
            " ".join(
                [vocab_tgt.get_itos()[x] for x in model_out if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Output               : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(n_examples=5):
    global vocab_src, vocab_tgt, spacy_de, spacy_en

    print("Preparing Data ...")
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=1,
        is_distributed=False,
    )

    print("Loading Trained Model ...")

    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(
        torch.load("multi30k_model_final.pt", map_location=torch.device("cpu"))
    )

    print("Checking Model Outputs:")
    example_data = check_outputs(
        valid_dataloader, model, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


# execute_example(run_model_example)


# ## 注意力可视化
# 
# > 即使使用贪婪解码器，翻译看起来也相当不错。我们可以进一步将其可视化，以查看在注意力的每一层发生了什么。

# In[ ]:


def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    "convert a dense matrix to a data frame with row and column indices"
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        # if float(m[r,c]) != 0 and r < max_row and c < max_col],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    df = mtx2df(
        attn[0, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )


# In[ ]:


def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn


def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    # ntokens = last_example[0].ntokens
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
            0,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(
        charts[0]
        # | charts[1]
        | charts[2]
        # | charts[3]
        | charts[4]
        # | charts[5]
        | charts[6]
        # | charts[7]
        # layer + 1 due to 0-indexing
    ).properties(title="Layer %d" % (layer + 1))


# ## 编码器自注意力

# In[ ]:


def viz_encoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]  # batch object for the final example

    layer_viz = [
        visualize_layer(
            model, layer, get_encoder, len(example[1]), example[1], example[1]
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )


show_example(viz_encoder_self)


# ## 解码器自注意力

# In[ ]:


def viz_decoder_self():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_self,
            len(example[1]),
            example[1],
            example[1],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


show_example(viz_decoder_self)


# ## 解码器源注意力

# In[ ]:


def viz_decoder_src():
    model, example_data = run_model_example(n_examples=1)
    example = example_data[len(example_data) - 1]

    layer_viz = [
        visualize_layer(
            model,
            layer,
            get_decoder_src,
            max(len(example[1]), len(example[2])),
            example[1],
            example[2],
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        & layer_viz[1]
        & layer_viz[2]
        & layer_viz[3]
        & layer_viz[4]
        & layer_viz[5]
    )


show_example(viz_decoder_src)


# # 结论
# 
#  希望这段代码对未来的研究有用。如有任何问题，请联系我们。
# 
# 
#  祝好，
#  Sasha Rush, Austin Huang, Suraj Subramanian, Jonathan Sum, Khalid Almubarak,
#  Stella Biderman
