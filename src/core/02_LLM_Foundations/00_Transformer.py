# %% [markdown]
# # Transformer
# Transformer [@vaswani2023attentionneed] 是当前最主流的序列建模架构，也是大规模语言模型（LLM）的核心结构。
# 自 2017 年提出以来，它逐渐取代 RNN 与 CNN，成为自然语言处理、视觉建模、语音建模乃至多模态学习中的基础组件。
# 当前主流模型（如 GPT、LLaMA 等）均基于 Transformer 架构进行扩展与规模化。

# 本教程的目标不是复述论文，而是：

# - 从零实现一个 极简但结构完整 的 Transformer
# - 清晰解释每一个张量的形状与信息流
# - 理解 mask、attention、multi-head 等机制背后的设计逻辑
#
#
# 我们将采用 decoder-only Transformer 作为核心结构，因为它是当前大模型的主流形式，也更适合理解自回归建模的本质。
#
# <div align="center">
# <img src="../images/00_arch.png" width="50%">
# <br>
# <b>Figure 1</b>: Transformer 架构[@vaswani2023attentionneed]
# </div>
#
# 在进入具体模块之前，我们先明确训练数据于训练目标的构造方式。

# %%
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as functional

RUN_EXAMPLES = True


def is_interactive_notebook() -> bool:
    """Return True when running in a Jupyter-like interactive shell."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    return shell is not None


def _should_run_examples() -> bool:
    # In scripts, __name__ == "__main__"; in notebooks, rely on IPython shell.
    return RUN_EXAMPLES and (__name__ == "__main__" or is_interactive_notebook())


def show_example(fn, args=None, kwargs=None):
    """Execute and return a demo function output when examples are enabled."""
    if not _should_run_examples():
        return None
    args = [] if args is None else args
    kwargs = {} if kwargs is None else kwargs
    return fn(*args, **kwargs)


def execute_example(fn, args=None, kwargs=None):
    """Execute a demo function for side effects when examples are enabled."""
    if not _should_run_examples():
        return None
    args = [] if args is None else args
    kwargs = {} if kwargs is None else kwargs
    fn(*args, **kwargs)
    return None


# %% [markdown]
# ## 训练数据与自回归目标

# 在理解 Transformer 结构之前，必须先明确一个问题：
# 模型在训练时到底在学什么？

# 在这里我们只讨论 decoder-only 模型，对于 decoder-only 语言模型的训练目标来说，训练目标是 **Next Token Prediction（下一个 token 预测）**。

# 给定一段 token 序列

# $$ x = (x_1, x_2, \dots, x_T) $$

# 模型的训练目标是最大化：

# $$ \sum_{t=1}^{T} \log P(x_t \mid x_{<t}) $$

# 也就是说：

# - 输入是前缀 $x_{<t}$
# - 目标是预测当前位置 $x_t$

# 这就是自回归（autoregressive）建模。
# 假设我们有一段文本：
# “attention is all you need”。
# 经过 tokenizer 得到：

# $$ [x_1, x_2, x_3, x_4, x_5] $$

# 在训练时，我们不会只预测最后一个 token，而是**同时预测所有位置**：

# | 上下文 | 预测目标 |
# | --- | --- |
# | $x_1$ | $x_2$ |
# | $x_1,x_2$ | $x_3$ |
# | $x_1,x_2,x_3$ | $x_4$ |
# | $x_1,x_2,x_3,x_4$ | $x_5$ |
#

# 在实现上，我们通常采用“右移一位”的方式构造标签。
# 输入：

# $$[x_1, x_2, x_3, x_4]$$

# 标签：

# $$[x_2, x_3, x_4, x_5]$$

# 模型一次前向传播即可得到所有位置的预测。
# 模型在每个位置 $t$ 产生一个对下一个 token 的预测分布：

# $$
# P(x_{t+1} \mid x_{\le t})
# $$

# 单个位置的 loss 是交叉熵：

# $$
# \ell_t = -\log P(x_{t+1} \mid x_{\le t})
# $$

# 整个序列的 loss 是对所有位置求平均：

# $$
# \frac{1}{N}
# \sum_{t \in \text{valid positions}}
# \ell_t
# $$

# 其中 $N$ 是有效 token 的数量（通常需要排除 padding）。
# 为了实现对于所有位置的同步训练，我们需要将输入和标签都转换为张量形式，并使用合适的掩码机制来确保对于每一个token的预测值依赖于其之前的所有token。
# 下一章节中我们首先介绍 tokenize 机制，接下来我们会在 self-attention 的实现中介绍如何进行掩码。


# %% [markdown]
# ## Tokenize（分词）
#
# Transformer 不直接处理字符串，而是处理整数序列 $[token\ id]$。
# 因此在进入 attention 之前，需要把原始文本稳定地映射为 $[token\ id]$，并保证训练与推理使用同一套规则。
# 语言是开放集合，模型输入必须是封闭接口。Tokenizer 将文本压缩到固定词表大小 $V$ 的离散符号集合上：每个 token 对应一个 $id \in [0, V)$。
# 后续模型只在 $id$ 序列上建模。
# 训练与推理必须使用同一套 tokenizer 与 vocab；否则输入分布不一致，会直接影响模型行为与可复现性。
#
# 分词后我们先得到一维 token id 张量 $x: [T]$，再加上 batch 维得到 $x_{batch}: [B, T]$。
# 接下来根据 token id 转换成 one-hot 矩阵，则得到 $x_{one\_hot}: [B, T, V]$，其中 $V$ 是词表大小。这个步骤实际实现中直接通过索引完成，以避免显式构造 one-hot 向量。
#
# Tiktoken 是当前 LLM 场景常见的 Tokenizer 库之一，下面用 tiktoken 做一个示例。


# %%
def example_tokenize_with_tiktoken():
    import tiktoken
    import torch
    import torch.nn.functional as functional

    text = "attention is all you need"
    encoding = tiktoken.get_encoding("cl100k_base")
    token_ids = encoding.encode(text)
    token_texts = [encoding.decode([token_id]) for token_id in token_ids]
    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    token_tensor_batched = token_tensor.unsqueeze(0)
    vocab_size = encoding.n_vocab
    token_tensor_btv = functional.one_hot(
        token_tensor_batched, num_classes=vocab_size
    ).to(torch.float32)

    print("text:", text)
    print("token_ids:", token_ids)
    print("tokens:", token_texts)
    print("token_tensor.shape [Seq]:", tuple(token_tensor.shape))
    print("token_tensor_batched.shape [B, T]:", tuple(token_tensor_batched.shape))
    print("vocab_size V:", vocab_size)
    print("token_tensor_btv.shape [B, T, V]:", tuple(token_tensor_btv.shape))


show_example(example_tokenize_with_tiktoken)

# %% [markdown]
# ### Padding（序列对齐）

# 在实际训练中，一个 batch 内的样本长度通常不同。
# 为了能够将它们堆叠成统一形状的张量 $x_{batch} \in \mathbb{R}^{B \times T}$，
# 需要将较短序列补齐到当前 batch 的最大长度 $T$。

# 设原始序列长度为 $L < T$，则补齐形式为：

# $$
# [x_1, x_2, \dots, x_L]
# \;\Rightarrow\;
# [x_1, x_2, \dots, x_L, \text{PAD}, \dots, \text{PAD}]
# $$

# 其中 $\text{PAD}$ 是一个特殊 token，
# 通常对应一个固定的 token id（例如 0）。

# 需要强调的是：
# padding 只是一种张量对齐手段，
# 它不携带语义信息，对应的 embedding 矢量也不会更新。
# 也不应该影响模型的预测结果。

# 因此后续在 attention 计算中，
# 需要显式屏蔽 padding 位置，使其不参与信息聚合。

# %% [markdown]
# ## Embedding
#
# ### 语义 Embedding
# Tokenize 之后，文本被表示为整数序列 token ids。此时每个 token 可以看作一个 $V$ 维的 one-hot 向量，其中 $V$ 是词表大小。
# one-hot 向量彼此正交，不包含语义结构，因此需要将其映射到连续向量空间。
# Embedding 的本质是一个查表操作（lookup table）。我们定义一个可学习矩阵 $W \in \mathbb{R}^{V \times d_{model}}$，其中 $V$ 为词表大小，$d_{model}$ 为隐藏维度。
# 对于一个 token id $i$，其对应向量为 $W[i]$。
# Embedding 的参数 $W$ 在训练过程中通过反向传播更新。语义相近的 token 会在向量空间中逐渐靠近，从而为后续 attention 提供可计算的结构。
# 在 Transformer 中，token embedding 通常会乘以 $\sqrt{d_{model}}$ 进行缩放，以保持数值尺度稳定。


# ### 位置 Embedding
#
# 在 Transformer 中，除了 token 的语义信息，还需要让模型知道序列中每个 token 的位置信息。因为 self-attention 机制本身是“无序”的，它只看 token 之间的相似度，不知道谁在前谁在后。
# 为此，我们引入位置编码（Positional Embedding），它为每个位置 $t$ 提供一个向量 $E_{pos}[t] \in \mathbb{R}^{d_{model}}$，并与 token embedding 相加：
# $$x = E_{token} + E_{pos}$$
#
# 这样，模型在计算 attention 时就能同时感知语义与顺序。
# Transformer 原文中采用的是固定正弦位置编码（Sinusoidal Positional Encoding）。
# 其定义如下：
#
# $$PE[t, 2i]   = sin(t / 10000^{2i / d_{model}})$$
# $$PE[t, 2i+1] = cos(t / 10000^{2i / d_{model}})$$
#
# 这种编码方式不需要训练参数，能够自然推广到任意长度。
# 它通过不同频率的正弦波让模型在连续空间中学习到相对位置信息。
# 在模型中，位置编码的加入方式为：
#
# $$x = (E_{token} * \sqrt{d_{model}}) + E_{pos}$$
#
# 其中 $\sqrt{d_{model}}$ 用于保持数值尺度一致。
# 这种设计让模型在无序的 attention 结构中恢复序列的顺序感。
# 值得一提的是，随着研究的发展，除了原始的正弦编码外，现在还出现了许多其他位置编码方式，例如可学习位置编码、相对位置编码、旋转位置编码（RoPE）等。
# %%
class TokenAndEmbedding(nn.Module):
    def __init__(self, V, d_model, max_T, pad_id=0):
        super().__init__()
        self.V = V
        self.d_model = d_model
        self.max_T = max_T
        self.pad_id = pad_id

        # token embedding: $W_{tok} \in \mathbb{R}^{V \times d_{model}}$
        # padding_idx 会让 $W_{tok}[pad\_id]$ 固定为 0，且不会被更新（更利于 mask 逻辑稳定）
        self.tok_emb = nn.Embedding(
            num_embeddings=V, embedding_dim=d_model, padding_idx=pad_id
        )

        # position embedding: $W_{pos} \in \mathbb{R}^{max\_T \times d_{model}}$
        self.pos_emb = nn.Embedding(num_embeddings=max_T, embedding_dim=d_model)

    def forward(self, token_ids):
        """
        token_ids: [B, T]，dtype 一般是 int64
        return:
            x: [B, T, d_model]
            attn_mask: [B, 1, 1, T]  (True 表示可见，False 表示被 mask；适配最常见 attention 写法)
        """
        B, T = token_ids.shape
        assert T <= self.max_T, "sequence length T must be <= max_T"

        # 1) token embedding lookup
        # tok: [B, T, d_model]
        tok = self.tok_emb(token_ids)

        # 2) position ids
        # pos_ids: [T] -> [1, T] -> [B, T] (广播也可以，但写全更直观)
        pos_ids = torch.arange(T, device=token_ids.device)  # [T]
        pos_ids = pos_ids.unsqueeze(0).repeat(B, 1)  # [B, T]

        # pos: [B, T, d_model]
        pos = self.pos_emb(pos_ids)

        # 3) combine + scale
        # 常见做法：tok * sqrt(d_model) + pos
        x = tok * (self.d_model**0.5) + pos

        # 4) attention mask（padding 位置不可见）
        # pad_mask: [B, T]，True 表示不是 pad（可见）
        pad_mask = token_ids != self.pad_id

        # 变形到 [B, 1, 1, T]，便于在 attention logits 上广播
        attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)

        return x, attn_mask


# %% [markdown]
# ## Attention
#
# 经过 embedding 之后，我们得到输入表示：
#
# $$
# x_0 \in \mathbb{R}^{B \times T \times d_{\text{model}}}
# $$
#
# 此时每个位置 $t$ 上的向量只包含该 token 的局部信息。
# 接下来需要让不同位置之间发生信息交互。
# 为此，Transformer 使用 Self-Attention 机制。

# <div align="center">
# <img src="../images/00_attn.png" width="50%">
# <br>
# <b>Figure 2</b>: 自注意力[@vaswani2023attentionneed]
# </div>
#
# 第一步，将输入映射为三个线性空间：
#
# $$
# Q = x_0 W_Q
# $$
#
# $$
# K = x_0 W_K
# $$
#
# $$
# V = x_0 W_V
# $$
#
# 其中
#
# $$
# W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}
# $$
#
# 得到：
#
# $$
# Q, K, V \in \mathbb{R}^{B \times T \times d_k}
# $$
#
# Self-Attention 的核心计算为：
#
# $$
# \text{softmax}
# \left(
# \frac{Q K^\top}{\sqrt{d_k}}
# \right)
# V
# $$
#
# 其中：
#
# $QK^\top$ 表示位置之间的相似度，
# $\sqrt{d_k}$ 用于数值稳定，
# softmax 使每个位置的权重归一化。
# 输出仍然是：
#
# $$
# x_1 \in \mathbb{R}^{B \times T \times d_k}
# $$
#
# 这一步的含义是：
# 每个 token 会根据与其他 token 的相似度，对所有位置的信息进行加权汇聚，从而形成“上下文相关表示”。

# %% [markdown]
# ### 注意力掩码（Attention Mask）
#
# 注意力掩码（Attention Mask）是用于在自注意力机制中屏蔽某些位置的注意力权重，从而实现对模型行为的控制。
# 总的来说，在 decoder-only的模型中，注意力掩码会在两种情形下发生：

# - 填充掩码：在训练时所有的文本被打包成了一个batch，但是每个文本的长度可能不同，因此需要使用填充掩码来屏蔽掉padding位置的注意力权重。
# - 因果掩码：在生成时，模型在生成第 $t$ 个 token 时，只能访问序列中前面的 $1, 2, \dots, t-1$ 个位置的信息，不能“偷看”未来的 token。因此需要使用因果掩码来屏蔽掉未来位置的注意力权重。

# #### 填充掩码（Padding Mask）
#
# 为了把不同长度的样本放进同一个 batch，我们会把短序列补齐到长度 $T$：

# $$
# [x_1, x_2, x_3, \text{PAD}, \text{PAD}]
# $$

# 这些 $\text{PAD}$ 不应该被任何位置关注到，否则会污染表示。
# 因此 padding mask 的规则是：如果位置 $j$ 是 padding，则对所有 $t$ 都屏蔽该 key：

#
# $$
# \mathrm{pad\_mask}_{t,j} =
# \begin{cases}
# 0, & x_j \neq \mathrm{PAD} \\
# -\infty, & x_j = \mathrm{PAD}
# \end{cases}
# $$
#

# 它的作用可以理解为：屏蔽所有人对 padding 的注意力（屏蔽的是 key 维）。


# #### 因果掩码（Causal Mask）
#

# 在自回归语言模型（例如 GPT 系列）中，模型在生成第 $t$ 个 token 时，只能访问序列中前面的 $1, 2, \dots, t-1$ 个位置的信息，不能“偷看”未来的 token。
# 为了实现这一约束，我们在 self-attention 的权重计算中引入 **causal mask**：

# $$
# \mathrm{mask}_{i,j} =
# \begin{cases}
# 0, & j \le i \\
# -\infty, & j > i
# \end{cases}
# $$

# 这样在计算
# $A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + \text{mask}\right)$
# 时，softmax 会自动将未来位置的权重压到 0，从而保证模型的预测严格遵守时间因果性。

# 在这里我们只实现 **decoder-only Transformer**，即只有自注意力（self-attention）层，没有 encoder-decoder 交互。
# 选择这种结构的原因是当前主流大模型（GPT、LLaMA 等）均采用 decoder-only 设计。
# 作为对比，**encoder-decoder Transformer**（原始 Transformer ）包含三种注意力形式：

# - **Encoder Self-Attention**：双向可见（无 causal mask），用于理解输入序列。
# - **Decoder Self-Attention**：单向可见（有 causal mask），用于生成输出。
# - **Cross-Attention**：decoder 读取 encoder 的输出，实现条件生成。


# 而在 decoder-only 模型中，所有注意力都是单向的，信息流完全由 causal mask 控制，从而实现严格的自回归生成。
# 在掌握注意力的计算方式和注意力掩码机制之后，我们就可以实现一个极简的注意力层。
# %%
class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, dropout_p=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k

        # 三个线性层：从 d_model -> d_k
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)

        # 输出投影层：从 d_k -> d_model
        self.W_O = nn.Linear(d_k, d_model, bias=False)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, pad_mask=None, causal=False):
        """
        x: [B, T, d_model]
        pad_mask: [B, T]，True 表示有效 token；False 表示 padding
        causal: 是否使用因果 mask（decoder 里需要）

        return:
            y: [B, T, d_model]
        """
        B, T, _ = x.shape

        # 1) 线性投影得到 Q,K,V
        Q = self.W_Q(x)  # [B, T, d_k]
        K = self.W_K(x)  # [B, T, d_k]
        V = self.W_V(x)  # [B, T, d_k]

        # 2) 打分：S = Q K^T / sqrt(d_k)
        # K.transpose(-2, -1): [B, d_k, T]
        S = torch.matmul(Q, K.transpose(-2, -1))  # [B, T, T]
        S = S / (self.d_k**0.5)

        # 3) mask：把“不可见位置”的分数置为 -inf，让 softmax 后权重为 0
        # 3.1 padding mask：如果 key 位置是 padding，则任何 query 都不应该 attend 到它
        if pad_mask is not None:
            # pad_mask: [B, T] -> key_mask: [B, 1, T]，用于广播到 [B, T, T]
            key_mask = pad_mask.unsqueeze(1)  # [B, 1, T]
            S = S.masked_fill(~key_mask, float("-inf"))

        # 3.2 causal mask：当前位置 t 只能看见 <= t 的位置
        if causal:
            # causal_mask: [T, T]，上三角为 True 表示要 mask
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            S = S.masked_fill(causal_mask, float("-inf"))

        # 4) softmax 得到注意力权重 A
        A = torch.softmax(S, dim=-1)  # [B, T, T]
        A = self.dropout(A)

        # 5) 加权求和得到输出
        x = torch.matmul(A, V)  # [B, T, d_k]

        return x


# %% [markdown]
# ### 如何理解 Self-Attention ？
#
# 按照我们之前的极简实现，我们可以把一个 self attention head 的计算过程拆成两步：
#
# 1) 对每个 token 做 value 投影（逐 token 的线性变换）
# $$v_i = W_V x_i$$
#
# 2) 用注意力权重把不同 token 的 value 混合（跨 token 的信息汇聚）
# $$y_i = \sum_j A_{i,j} v_j$$
#
# <div align="center">
# <img src="../images/00_qkv.png" width="100%">
# <br>
# <b>Figure 3</b>: QKV 计算
# </div>
#
# 可以看到，相比于 MLP层，self attention层最显著的特征是代表不同 token 的矢量之间发生了计算，而在 MLP 层中，所有的计算都是等价于逐 token 进行的。
# 因此直观上可以把它理解成两类操作的组合：
#
# - $A$ 决定“信息从哪里来、送到哪里去”
# - $W_V$ 决定“搬运什么信息”
#
# 如果你想看更完整、更系统的讨论（包括更一般的张量视角与更多例子），
# 可以参考 Anthropic 的 [Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)[@elhage2021mathematical] 系列研究：
#


# %% [markdown]
# ## Multi-Head Attention
#
# 单头 Self-Attention 的问题在于：
# 所有信息混合都发生在同一个子空间中。
# 如果一个 head 需要同时：
#
# - 处理语法依赖
# - 处理指代关系
# - 处理位置模式
#
# 那它必须在同一个 $d_{k}$ 维空间里完成所有工作。
#
# Multi-Head Attention 的核心思想是：
# 将 $d_{model}$ 划分为多个子空间，
# 在不同子空间中并行执行 attention，
# 然后再把结果拼接回来。
#
# <div align="center">
# <img src="../images/00_multi_heads.png" width="50%">
# <br>
# <b>Figure 4</b>: 多头注意力[@vaswani2023attentionneed]
# </div>
# 数学形式如下：
# 对第 $h$ 个 head
#
# $$Q_h = x W_Q^{(h)}$$
# $$K_h = x W_K^{(h)}$$
# $$V_h = x W_V^{(h)}$$
#
# 每个 head 独立计算：
#
# $$\mathrm{head}_h = \mathrm{softmax}\left(\frac{Q_h K_h^\top}{\sqrt{d_k}}\right) V_h$$
#
# 然后拼接所有 head：
#
# $$\mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_H)$$
#
# 最后做一次线性投影：
#
# $$x_1 = \mathrm{Concat}(\dots) W_O$$
#
# 其中：
# - $H$ 是 head 数
# - $d_{k} = d_{model} / H$
# 下面给出一个简单的实现。


# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 复用前面的单头实现：每个 head 都是一个 SelfAttention(d_model -> d_k)
        self.heads = nn.ModuleList(
            [SelfAttention(d_model=d_model, d_k=self.d_k) for _ in range(num_heads)]
        )

        # 拼接后统一写回 residual stream（d_model -> d_model）
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, pad_mask=None, causal=False):
        """
        x: [B, T, d_model]
        pad_mask: [B, T]，True 表示有效 token；False 表示 padding
        causal: 是否使用 causal mask

        return:
            y: [B, T, d_model]
        """
        # 每个 head 输出: [B, T, d_k]
        head_outs = [h(x, pad_mask=pad_mask, causal=causal) for h in self.heads]

        # 拼接: [B, T, H*d_k] = [B, T, d_model]
        x = torch.cat(head_outs, dim=-1)

        # 输出投影: [B, T, d_model]
        x = self.W_O(x)
        return x


# %% [markdown]
# ## Transformer
# 接下来，将若干层transformer block 以及 embedding, unembedding 矩阵拼接起来，就得到了一个极简的 decoder only transformer。
#


# %%
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        """
        x: [B, T, d_model]
        return: [B, T, d_model]
        """
        return self.fc2(self.act(self.fc1(x)))


# %%
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff)

        # Post-LN：每个子层之后做 LayerNorm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, pad_mask=None, causal=True):
        """
        x: [B, T, d_model]
        pad_mask: [B, T]，True 表示有效 token
        causal: 是否启用 causal mask（decoder-only 通常为 True）
        """
        # 1) Self-Attention 子层 + 残差 + LayerNorm
        x = self.ln1(x + self.attn(x, pad_mask=pad_mask, causal=causal))

        # 2) MLP 子层 + 残差 + LayerNorm
        x = self.ln2(x + self.ffn(x))

        return x


# %% [markdown]
# ## Transformer Block Architecture
#
# 在前面我们已经实现了注意力模块。现在把它和 MLP 组合起来，就得到 Transformer 的基本计算单元：**Transformer block**。
# 原始 Transformer（Attention Is All You Need）使用的是 **Post-LN** 结构。也就是说，每个子层先完成自己的变换，再与输入做残差相加，最后进行 LayerNorm：
#
# $$x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$$
#
# 一层 decoder-only block 由两个子层组成：
#
# 1) 多头自注意力（Self-Attention）
# $$x_1 = \mathrm{LayerNorm}(x_0 + \mathrm{MHA}(x_0))$$
#
# 2) 前馈网络（MLP / FFN）
# $$x_2 = \mathrm{LayerNorm}(x_1 + \mathrm{MLP}(x_1))$$
#
# 其中注意力子层负责 token 之间的信息交互，而 MLP 子层只在每个 token 内部做非线性变换。
# 下面给出一层 Transformer block 的实现。


# %%
class TransformerDecoderOnly(nn.Module):
    def __init__(self, V, d_model, num_heads, d_ff, num_layers=6, max_T=1024, pad_id=0):
        super().__init__()
        self.V = V
        self.d_model = d_model
        self.pad_id = pad_id

        self.embed = TokenAndEmbedding(V=V, d_model=d_model, max_T=max_T, pad_id=pad_id)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
                for _ in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(d_model, V, bias=False)

    def forward(self, token_ids, causal=True, return_last_logits=False):
        """
        token_ids: [B, T]
        return_last_logits:
            - False: 返回所有位置 logits: [B, T, V]（训练常用）
            - True : 只返回最后位置 logits: [B, V]（推理常用）
        """
        x, _ = self.embed(token_ids)  # [B, T, d_model]
        pad_mask = token_ids != self.pad_id  # [B, T]

        for blk in self.blocks:
            x = blk(x, pad_mask=pad_mask, causal=causal)  # [B, T, d_model]

        logits = self.lm_head(x)  # [B, T, V]

        if return_last_logits:
            return logits[:, -1, :]  # [B, V]
        return logits


# %% [markdown]
# ## Training Demo
#
# 作为最后一节，我们用已经实现好的 decoder-only Transformer 跑一个最小训练流程。
# 设定上做适当的简化：
#
# - 层数 $L=6$
# - 隐藏维度 $d_{model}=512$
#
# 训练的流程如下：
#
# - 输入文本：`attention is all youneed`
# - 用 `tiktoken` 做 tokenize 得到 token ids
# - forward 得到所有位置的 logits
# - 用 next-token prediction 计算 loss
# - 反向传播 + optimizer.step() 进行参数更新（演示若干步）

# %% [markdown]
#
# 我们用 GPT-2 的 BPE 词表（tiktoken 的 `gpt2` encoding）。
# 这样词表大小 $V$ 就直接等于 `enc.n_vocab`，可以用于初始化 embedding / lm head。

# %%
enc = tiktoken.get_encoding("gpt2")

text = "attention is all youneed"
ids = enc.encode(text)  # List[int]
token_ids = torch.tensor([ids], dtype=torch.long)  # [B=1, T]

V = enc.n_vocab
pad_id = enc.eot_token  # demo 里几乎用不到 padding，但模型接口需要一个 pad_id

token_ids, V, pad_id

# %% [markdown]
# （1）**初始化模型（$L=1, d_{model}=512$）**
# %%
d_model = 512
num_layers = 1
num_heads = 8  # 常见选择：512 / 8 = 64
d_ff = 2048  # 原文默认：d_ff = 4 * d_model
max_t = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerDecoderOnly(
    V=V,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    max_T=max_t,
    pad_id=pad_id,
).to(device)

token_ids = token_ids.to(device)

# %% [markdown]
#  （2）**一次 forward 得到所有位置的 next-token logits，然后算 loss**
#
# 对输入 $[x_1,\dots,x_T]$：
# - 模型输出 `logits[:, t]` 表示预测 $x_{t+1}$ 的分布
#
# 因此 loss 对齐方式是：
# - 预测：`logits[:, :-1, :]`
# - 标签：`token_ids[:, 1:]`
#


# %%
def next_token_loss(logits, token_ids, pad_id, ignore_index=-100):
    """
    logits:   [B, T, V]
    token_ids:[B, T]
    """
    pred_logits = logits[:, :-1, :]  # [B, T-1, V]
    labels = token_ids[:, 1:].clone()  # [B, T-1]

    # 如果 batch 里有 padding，需要把 PAD 位置忽略掉
    labels[labels == pad_id] = ignore_index

    B, Tm1, V = pred_logits.shape
    loss = functional.cross_entropy(
        pred_logits.reshape(B * Tm1, V),
        labels.reshape(B * Tm1),
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss


# %% [markdown]
# （3）**训练若干步**
#
# 这里只演示最核心的训练循环结构，不加入 dropout、warmup、clip、混合精度等额外工程细节

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()

for step in range(5):
    optimizer.zero_grad()

    logits = model(token_ids, causal=True)  # [1, T, V]
    loss = next_token_loss(logits, token_ids, pad_id=pad_id)

    loss.backward()
    optimizer.step()

    print(f"step={step:02d} | loss={loss.item():.4f}")

# %% [markdown]
# 到这里就完成了一个最小闭环：
# text -> tiktoken ids -> logits -> next-token loss -> backward -> update。

# %% [markdown]
# ## 参考文献
#
# ::: {#refs}
# :::
