#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os

# 切换到项目根目录，以便正确解析相对路径（如 var/files）
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
os.chdir(project_root)
sys.path.insert(0, project_root)

import regex
from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
import random

import tiktoken


# ## CS336：从零开始的语言模型
# 
# 

# ## 为什么要开设这门课程？
# 
# 
# 
# 问题：研究人员正在与底层技术**脱节**。
# 
# 8 年前，研究人员会自己实现并训练模型。
# 
# 6 年前，研究人员会下载一个模型（如 BERT）并微调。
# 
# 今天，研究人员只是提示一个闭源模型（如 GPT-4/Claude/Gemini）。
# 
# 抽象层级上移会提升生产力，但：
# 
# - 这些抽象会“漏水”（不同于编程语言或操作系统）。
# - 仍有需要“撕开技术栈”的基础研究。
# 
# 对这项技术的**完整理解**是进行**基础研究**的必要条件。
# 
# 核心目标：**通过构建来理解**。
# 

# ## 语言模型的工业化
# 
# <p align="center">
#   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Industrialisation.jpg/440px-Industrialisation.jpg" alt="工业化">
#   <br>
#   <i>19世纪工业革命时期的工厂：象征着现代大规模语言模型训练已演变为极高资本投入与算力集约的“重工业”模式</i>
# </p>
# 
# 据称 GPT-4 有 1.8T 参数。[报道](https://www.hpcwire.com/2024/03/19/the-generative-ai-future-is-now-nvidias-huang-says)
# 
# 据称 GPT-4 训练成本 1 亿美元。[报道](https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/)
# 
# xAI 用 20 万张 H100 训练 Grok，并在扩容。[报道](https://www.tomshardware.com/pc-components/gpus/elon-musk-is-doubling-the-worlds-largest-ai-gpu-cluster-expanding-colossus-gpu-cluster-to-200-000-soon-has-floated-300-000-in-the-past)
# 
# Stargate（OpenAI、NVIDIA、Oracle）4 年投资 5000 亿美元。[报道](https://openai.com/index/announcing-the-stargate-project/)
# 
# 此外，前沿模型的训练细节几乎没有公开。
# 
# 摘自 GPT-4 技术报告：[PDF](https://arxiv.org/pdf/2303.08774.pdf)
# 
# <p align="center">
#   <img src="../images/gpt4-no-details.png" alt="GPT-4 无细节">
#   <br>
#   <i>GPT-4 技术报告截图：模型架构、硬件、训练数据及计算量等关键细节均以“竞争格局”为由未予公开，反映了前沿模型日益封闭的行业现状</i>
# </p>
# 
# ## 规模不同，性质不同
# 
# 前沿模型并非我们可及。
# 
# 但构建小语言模型（<1B 参数）可能无法代表大语言模型。
# 
# 例 1：注意力与 MLP 的 FLOPs 占比随规模变化。[链接](https://x.com/stephenroller/status/1579993017234382849)
# 
# <p align="center">
#   <img src="../images/roller-flops.png" alt="FLOPs 占比">
#   <br>
#   <i>不同规模模型中注意力机制与 MLP 层的计算量（FLOPs）占比变化：随着参数量增加，MLP 的计算比重显著上升</i>
# </p>
# 
# 例 2：行为在规模中涌现。[论文](https://arxiv.org/pdf/2206.07682)
# 
# <p align="center">
#   <img src="../images/wei-emergence-plot.png" alt="涌现">
#   <br>
#   <i>能力涌现现象：某些复杂任务（如多步算术、翻译）的准确率在模型规模达到特定阈值前近乎随机，随后出现爆发式增长</i>
# </p>
# 
# ## 什么知识可以迁移到前沿模型？
# 
# 知识可以分为以下三个维度：
# 
# - **底层机制 (Mechanisms)**：即“事物运作的原理”（例如：Transformer 的内部构造、模型并行如何适配 GPU 架构）。
# - **工程理念 (Mindset)**：即“对待规模的准则”（例如：如何压榨硬件的每一分性能、如何严谨地遵循缩放律进行决策）。
# - **实践直觉 (Intuition)**：即“工程上的‘手感’”（例如：哪些数据处理方式或模型细微调整更能有效提升准确率）。
# 
# **底层机制**与**工程理念**具有极强的通用性，可以直接迁移到更大的模型开发中。
# 
# **实践直觉**则只能部分迁移，因为某些在小规模有效的经验，在超大规模下可能会失效。
# 
# ## 直觉？🤷
# 
# 一些设计决策目前还无法充分解释，只能来自实验。
# 
# 例：Noam Shazeer 提出 SwiGLU 的论文。[PDF](https://arxiv.org/pdf/2002.05202.pdf)
# 
# <p align="center">
#   <img src="../images/divine-benevolence.png" alt="神意般的善意">
#   <br>
#   <i>SwiGLU 激活函数的实验结果：论文标题“Divine Benevolence（神意般的善意）”戏称某些改进虽然有效但缺乏深层的理论解释</i>
# </p>
# 
# ## 苦涩的教训
# 
# 错误解读：规模才是一切，算法不重要。
# 
# 正确解读：**可扩展的算法**才重要。
# 
# ### 准确率 = 效率 × 资源
# 
# 事实上，在更大规模时效率更重要（浪费不起）。
# 
# [论文](https://arxiv.org/abs/2005.04305) 显示 2012-2019 年 ImageNet 上算法效率提升 44 倍。
# 
# 框架性问题：在既定算力与数据预算下，能构建的最佳模型是什么？
# 
# 换句话说，**最大化效率**！

# ## 语言模型的发展历程
# 
# ### 前神经时代（2010 年代以前）
# - **衡量英语熵的语言模型**：[Shannon (1950)](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf)
# - **N-gram 语言模型的大量研究**（用于机器翻译、语音识别）：[Brants et al. (2007)](https://aclanthology.org/D07-1090.pdf)
# 
# ### 神经组件（2010 年代）
# - **首个神经语言模型**：[Bengio et al. (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
# - **序列到序列 (Seq2seq) 建模**（用于机器翻译）：[Sutskever et al. (2014)](https://arxiv.org/pdf/1409.3215.pdf)
# - **Adam 优化器**：[Kingma & Ba (2014)](https://arxiv.org/pdf/1412.6980.pdf)
# - **注意力机制 (Attention)**：[Bahdanau et al. (2015)](https://arxiv.org/pdf/1409.0473.pdf)
# - **Transformer 架构**：[Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf)
# - **混合专家模型 (MoE)**：[Shazeer et al. (2017)](https://arxiv.org/pdf/1701.06538.pdf)
# - **模型并行技术**：[GPipe (2018)](https://arxiv.org/pdf/1811.06965.pdf), [ZeRO (2019)](https://arxiv.org/abs/1910.02054), [Megatron-LM (2019)](https://arxiv.org/pdf/1909.08053.pdf)
# 
# ### 早期基座模型（2010 年代后期）
# - **ELMo**：使用 LSTM 进行预训练，微调对下游任务有很大帮助 [Peters et al. (2018)](https://arxiv.org/abs/1802.05365)
# - **BERT**：使用 Transformer 进行预训练 [Devlin et al. (2018)](https://arxiv.org/abs/1810.04805)
# - **Google T5 (11B)**：将所有任务视为文本到文本 (Text-to-Text) 问题 [Raffel et al. (2019)](https://arxiv.org/pdf/1910.10683.pdf)
# 
# ### 拥抱规模化，趋于封闭
# - **OpenAI GPT-2 (1.5B)**：流畅的文本生成，首次展现零样本能力，采取阶段性发布策略 [Radford et al. (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
# - **缩放律 (Scaling Laws)**：为模型扩展提供了希望和可预测性 [Kaplan et al. (2020)](https://arxiv.org/pdf/2001.08361.pdf)
# - **OpenAI GPT-3 (175B)**：上下文学习 (In-context learning)，模型不再开源 [Brown et al. (2020)](https://arxiv.org/pdf/2005.14165.pdf)
# - **Google PaLM (540B)**：极大规模，但训练不够充分 [Chowdhery et al. (2022)](https://arxiv.org/pdf/2204.02311.pdf)
# - **DeepMind Chinchilla (70B)**：计算最优的缩放律 [Hoffmann et al. (2022)](https://arxiv.org/pdf/2203.15556.pdf)
# 
# ### 开放模型
# - **EleutherAI** 的开放数据集 ([The Pile](https://arxiv.org/pdf/2101.00027.pdf)) 与模型 ([GPT-J](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/))
# - **Meta OPT (175B)**：GPT-3 的复制尝试，遭遇大量 hardware 故障 [Zhang et al. (2022)](https://arxiv.org/pdf/2205.01068.pdf)
# - **Hugging Face / BigScience BLOOM**：专注于数据来源的透明度 [Workshop et al. (2022)](https://arxiv.org/abs/2211.05100)
# - **Meta Llama 系列**：[Llama (2023)](https://arxiv.org/pdf/2302.13971.pdf), [Llama 2 (2023)](https://arxiv.org/pdf/2307.09288.pdf), [Llama 3 (2024)](https://arxiv.org/abs/2407.21783)
# - **阿里巴巴 Qwen 系列**：[Qwen 2.5](https://arxiv.org/abs/2412.15115)
# - **DeepSeek 系列**：[DeepSeek-67B](https://arxiv.org/pdf/2401.02954.pdf), [DeepSeek-V2](https://arxiv.org/abs/2405.04434), [DeepSeek-V3](https://arxiv.org/pdf/2412.19437.pdf)
# - **AI2 OLMo 2**：[OLMo (2024)](https://arxiv.org/pdf/2402.00838.pdf), [OLMo 2 (2025)](https://arxiv.org/abs/2501.00656)
# 
# ### 开放程度的层级
# - **闭源模型**（如 GPT-4o）：仅提供 API 访问。
# - **开放权重模型**（如 DeepSeek）：提供模型权重，论文包含架构细节和部分训练细节，但不包含数据细节。
# - **开源模型**（如 OLMo）：提供权重和数据，论文包含绝大部分细节（但未必包含设计决策过程或失败的实验）。
# 
# ### 今日的前沿模型
# - **OpenAI o3**：[链接](https://openai.com/index/openai-o3-mini/)
# - **Anthropic Claude 3.7 Sonnet**：[链接](https://www.anthropic.com/news/claude-3-7-sonnet)
# - **xAI Grok 3**：[链接](https://x.ai/news/grok-3)
# - **Google Gemini 2.5**：[链接](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/)
# - **Meta Llama 3.3**：[链接](https://ai.meta.com/blog/meta-llama-3/)
# - **DeepSeek-R1**：[DeepSeek-R1 (2025)](https://arxiv.org/pdf/2501.12948.pdf)
# - **阿里巴巴 Qwen 2.5 Max**：[链接](https://qwenlm.github.io/blog/qwen2.5-max/)
# - **腾讯 Hunyuan-T1**：[链接](https://tencent.github.io/llm.hunyuan.T1/README_EN.html)

# ## 核心概念：一切为了效率
# 
# 在资源有限的情况下，如何训练出最佳模型？资源通常包括：**数据** + **硬件**（算力、内存、通信带宽）。
# 
# ### 效率驱动设计决策
# 
# 当下的语言模型开发受到算力高度约束，因此几乎所有的设计决策都旨在压榨硬件性能：
# 
# <p align="center">
#   <img src="../images/design-decisions.png" alt="设计决策" width="800">
#   <br>
#   <i>语言模型训练中的核心设计决策：涵盖了从分词、架构到数据处理和对齐的各个环节，每一项决策都旨在最大化计算与数据的利用效率</i>
# </p>
# 
# - **数据处理**：避免在低质量或无关数据上浪费宝贵的算力。
# - **分词 (Tokenization)**：直接处理原始字节虽然优雅，但在目前的模型架构下计算效率极低。
# - **模型架构**：许多改进（如 KV 缓存共享、滑动窗口注意力等）都是为了减少内存占用或 FLOPs。
# - **训练策略**：我们可以只训练一个 Epoch 就达到极好的效果！
# - **缩放律 (Scaling Laws)**：在小模型上节省算力，以便进行超参数调优。
# - **对齐 (Alignment)**：如果模型能更好地适配使用场景，那么较小的基座模型也能发挥巨大作用。
# 
# ---
# 
# ## 语言模型构建的全流程
# 
# ### 1. 基础构建 (Basics)
# **目标**：跑通全流程的基础版本。
# - **分词 (Tokenization)**：将字符串转换为整数序列（Tokens）。
# - **模型架构**：以原始 Transformer 为起点。
# - **训练**：优化器（AdamW 等）、学习率调度、超参数搜索。
# 
# <p align="center">
#   <img src="../images/transformer-architecture.png" alt="Transformer 架构" width="500">
#   <br>
#   <i>原始 Transformer 架构图：由编码器（Encoder）和解码器（Decoder）组成，是现代生成式语言模型的共同基石</i>
# </p>
# 
# ### 2. 系统优化 (Systems)
# **目标**：压榨硬件性能。
# - **算子核 (Kernels)**：编写 CUDA/Triton 核函数以最大化 GPU 利用率。
# - **并行技术**：数据并行、张量并行、流水线并行、序列并行。
# - **推理优化**：KV 缓存、推测解码。
# 
# <p align="center">
#   <img src="../images/prefill-decode.png" alt="Prefill 和 Decode" width="500">
#   <br>
#   <i>推理的两个阶段：Prefill（预填充，并行计算 Prompt 的表示）和 Decode（解码，逐个生成后续 Token），两者对硬件资源的需求特征迥异</i>
# </p>
# 
# ### 3. 缩放律 (Scaling Laws)
# **目标**：在小规模下做实验，预测大规模下的表现。
# 
# <p align="center">
#   <img src="../images/chinchilla-isoflop.png" alt="Chinchilla 缩放律" width="600">
#   <br>
#   <i>Chinchilla 实验结果：展示了在不同计算预算（IsoFLOPs 曲线）下，模型参数量与训练数据量的最优平衡关系</i>
# </p>
# 
# ### 4. 数据工程 (Data)
# **目标**：构建高质量的训练语料与评估体系。
# - **数据清洗**：HTML/PDF 转文本、质量过滤、去重（MinHash 等）。
# 
# <p align="center">
#   <img src="https://ar5iv.labs.arxiv.org/html/2101.00027/assets/pile_chart2.png" alt="The Pile 数据分布" width="500">
#   <br>
#   <i>The Pile 数据集构成：大规模、高质量且多样化的数据集（如 Wikipedia, ArXiv, GitHub）是训练强大语言模型的养料</i>
# </p>
# 
# ### 5. 对齐 (Alignment)
# **目标**：使模型真正可用并遵循指令。
# - **监督微调 (SFT)**：让模型学会遵循指令。
# - **偏好对齐**：RLHF、DPO、GRPO 等技术，提升安全性和风格。

# ## 分词 (Tokenization)
# 
# 原始文本通常表示为 Unicode 字符串。而语言模型是在 **Token**（通常表示为整数索引）序列上建立概率分布的。
# 
# <p align="center">
#   <img src="../images/tokenized-example.png" alt="分词示例" width="600">
#   <br>
#   <i>分词示例：一段文本被切分为多个 Token，每个 Token 对应词表中的一个唯一整数索引</i>
# </p>
# 
# 因此，我们需要：
# 1. **编码 (Encode)**：将字符串转换为 Token 序列。
# 2. **解码 (Decode)**：将 Token 序列还原为字符串。
# 
# **词表大小 (Vocabulary Size)** 是指可能出现的不同 Token 的总数。
# 
# ### Tokenizer 接口定义
# 
# 我们首先定义一个分词器的抽象基类：

# In[3]:


class Tokenizer(ABC):
    """分词器的抽象接口。"""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


# In[4]:


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """计算压缩比：字节数 / Token 数。"""
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens

def get_gpt2_tokenizer():
    # 使用 OpenAI 的 tiktoken 库获取 GPT-2 的分词器
    return tiktoken.get_encoding("gpt2")


# ### 分词器观察
# 
# 你可以通过 [tiktokenizer](https://tiktokenizer.vercel.app/?encoder=gpt2) 交互式地观察分词。
# 
# **一些关键观察点：**
# - 一个单词及其前面的空格通常被视为同一个 Token（例如 `" world"`）。
# - 单词出现在句首和句中可能会被编码为不同的 Token（例如 `"hello hello"`）。
# - 数字通常每隔几位会被切分。
# 
# 让我们看看 OpenAI 的 GPT-2 分词器的实际效果：

# In[5]:


tokenizer = get_gpt2_tokenizer()
string = "Hello, 🌍! 你好!"

# 编码
indices = tokenizer.encode(string)
print(f"Token 序列: {indices}")

# 解码还原
reconstructed_string = tokenizer.decode(indices)
print(f"还原字符串: {reconstructed_string}")

assert string == reconstructed_string

# 计算压缩比
ratio = get_compression_ratio(string, indices)
print(f"压缩比: {ratio:.2f}")


# ### 基于字符的分词 (Character-based tokenization)
# 
# Unicode 字符串是 Unicode 字符的序列。每个字符可以通过 `ord` 转换为码点（整数），通过 `chr` 转换回字符。
# 
# 例如：
# - `ord("a")` -> `97`
# - `ord("🌍")` -> `127757`
# 
# 我们来实现一个 `CharacterTokenizer`：

# In[6]:


class CharacterTokenizer(Tokenizer):
    """将字符串表示为 Unicode 码点序列。"""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))

tokenizer = CharacterTokenizer()
string = "Hello, 🌍! 你好!"
indices = tokenizer.encode(string)
print(f"Token 序列: {indices}")
print(f"还原字符串: {tokenizer.decode(indices)}")
print(f"压缩比: {get_compression_ratio(string, indices):.2f}")


# **基于字符分词的问题：**
# 1. **词表过大**：Unicode 字符约有 15 万个。
# 2. **稀疏性**：许多字符（如 🌍）非常罕见，这导致词表利用效率低下。
# 3. **序列长度**：压缩比通常为 1，意味着生成的 Token 序列非常长，会快速消耗模型的上下文窗口。

# ### 基于字节的分词 (Byte-based tokenization)
# 
# Unicode 字符串可以表示为字节序列，每个字节的值在 0 到 255 之间。最常用的编码方式是 [UTF-8](https://en.wikipedia.org/wiki/UTF-8)。
# 
# - 一些字符仅占用 1 个字节：`bytes("a", encoding="utf-8")` -> `b"a"`
# - 另一些字符（如表情或汉字）占用多个字节：`bytes("🌍", encoding="utf-8")` -> `b"\xf0\x9f\x8c\x8d"`
# 
# 我们来实现 `ByteTokenizer`：

# In[7]:


class ByteTokenizer(Tokenizer):
    """将字符串表示为字节序列。"""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")
        indices = list(map(int, string_bytes))
        return indices

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)
        string = string_bytes.decode("utf-8")
        return string

tokenizer = ByteTokenizer()
string = "Hello, 🌍! 你好!"
indices = tokenizer.encode(string)
print(f"Token 序列: {indices}")
print(f"还原字符串: {tokenizer.decode(indices)}")
print(f"压缩比: {get_compression_ratio(string, indices):.2f}")


# **基于字节分词的特点：**
# - **优点**：词表非常小且固定（只有 256 个值）。
# - **缺点**：压缩比非常糟糕（固定为 1），导致序列过长。在注意力机制计算复杂度随序列长度平方增长的背景下，这会导致严重的效率问题。
# 
# 目前有一些“无分词器 (Tokenizer-free)”的研究（如 [ByT5](https://arxiv.org/abs/2105.13626), [MEGABYTE](https://arxiv.org/pdf/2305.07185.pdf)），但在前沿大模型中尚未成为主流。

# ### 基于单词的分词 (Word-based tokenization)
# 
# 这是 NLP 的传统做法：将字符串拆分为单词。
# 
# 我们可以使用正则表达式来提取单词和标点：
# - 简单版本：`\w+|.`（保留字母数字序列或单个字符）
# - GPT-2 使用的高级版本如下：

# In[8]:


# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L23
GPT2_TOKENIZER_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

string = "I'll say supercalifragilisticexpialidocious!"

# 简单正则
simple_segments = regex.findall(r"\w+|.", string)
print(f"简单拆分: {simple_segments}")

# GPT-2 正则
gpt2_segments = regex.findall(GPT2_TOKENIZER_REGEX, string)
print(f"GPT-2 拆分: {gpt2_segments}")


# **基于单词分词的问题：**
# 
# 1. **词表爆炸（希普斯定律）**：根据 **希普斯定律 (Heaps' Law)**，词表大小 \( V \) 与语料库长度 \( N \) 成幂律关系 \( V = KN^\beta \)（通常 \(\beta \approx 0.5\)）。这意味着随着训练数据的增加，**词表大小会无限增长，永远不会收敛**。
# 2. **稀疏性**：许多单词非常罕见，出现频率极低（长尾效应），模型无法有效学习它们的表示。
# 3. **OOV (Out-of-Vocabulary) 问题**：由于词表必须截断（不可能无限大），训练时未见过的词或罕见词会被映射到一个特殊的 `[UNK]` Token。这不仅丢失了语义信息，还会严重干扰困惑度 (Perplexity) 的计算。
# 
# **为什么现代大模型不采用基于单词的分词？**
# 
# 除了上述理论上的词表无限增长问题，实际应用中还面临：
# - **形态变化**：如 "run", "runs", "running" 会被视为完全不同的词，无法共享词根语义。
# - **新词/专有名词**：如 "ChatGPT", "GPT-4o" 等在训练数据中可能未出现。
# - **多语言场景**：不同语言的"单词"边界定义差异巨大（如中文没有空格分隔）。
# 
# 因此，现代语言模型普遍采用 **子词分词 (Subword Tokenization)**，如 BPE，它在词表大小和序列长度之间取得了更好的平衡。

# ### 字节对编码 (Byte Pair Encoding, BPE)
# 
# **BPE** 是目前主流语言模型（如 GPT 系列）采用的分词方法，它在词表大小和序列长度之间取得了良好的平衡。
# 
# #### 历史背景
# 
# BPE 算法最初由 **Philip Gage** 在 1994 年提出，用于数据压缩。后来，**Sennrich et al. (2016)** 将其引入到神经机器翻译领域，替代了传统的基于单词的分词方法。**GPT-2** 进一步采用了字节级 BPE，使其成为现代语言模型的标准分词技术。
# 
# **核心思想**：通过**训练**分词器来自动确定词表，而不是预先定义固定的词表。
# 
# **基本直觉**：
# - **常见字符序列**（如 "the", "ing"）被表示为单个 Token，提高压缩比。
# - **罕见字符序列**（如专有名词、新词）被拆分为多个 Token，避免词表爆炸。
# 
# #### GPT-2 的实现方式
# 
# GPT-2 采用了**字节级 BPE**：
# 1. 首先将文本转换为 UTF-8 字节序列（每个字节 0-255）。
# 2. 从字节级别开始，不断合并出现频率最高的相邻 Token 对。
# 3. 这样可以在保持较小词表的同时，处理任意 Unicode 字符（包括多语言和表情符号）。
# 
# 接下来，我们将逐步实现 BPE 分词器的核心组件。

# #### 1. 辅助函数：合并相邻 Token 对
# 
# 在 BPE 训练过程中，我们需要一个函数来将序列中所有出现的特定相邻对替换为新的 Token 索引。

# In[9]:


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """返回 `indices`，其中所有出现的 `pair` 都被替换为 `new_index`。"""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


# #### 2. BPE 分词器定义
# 
# BPE 分词器需要两个核心组件：
# - **词表 (Vocab)**：将 Token 索引映射到字节串（用于解码）。
# - **合并规则 (Merges)**：记录训练过程中学到的合并操作（用于编码）。

# In[10]:


@dataclass(frozen=True)
class BPETokenizerParams:
    """定义 BPETokenizer 所需的参数。"""
    vocab: dict[int, bytes]             # 索引 -> 字节串
    merges: dict[tuple[int, int], int]  # (index1, index2) -> 新索引

class BPETokenizer(Tokenizer):
    """根据给定的合并规则和词表构建的 BPE 分词器。"""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        """将字符串编码为 Token 序列。"""
        indices = list(map(int, string.encode("utf-8")))
        # 注意：这是一个简化的缓慢实现，按训练时的合并顺序依次应用
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        """将 Token 序列解码为字符串。"""
        bytes_list = list(map(self.params.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8")
        return string


# 

# #### 3. BPE 训练算法
# 
# 训练 BPE 分词器的过程是一个迭代过程：
# 
# 1. **初始化**：从字节序列开始（每个字节 0-255 对应一个初始 Token）。
# 2. **统计频率**：统计所有相邻 Token 对的出现频率。
# 3. **合并最频对**：找到出现频率最高的相邻对，将其合并为一个新的 Token。
# 4. **更新词表**：将新 Token 添加到词表，并记录合并规则。
# 5. **重复**：重复步骤 2-4，直到达到指定的合并次数。
# 
# 通过这种方式，常见的字符序列（如 "the"）会被合并为单个 Token，而罕见序列则保持拆分状态。

# In[11]:


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:
    """
    训练 BPE 分词器。

    参数:
        string: 训练文本
        num_merges: 合并次数（决定最终词表大小）

    返回:
        BPETokenizerParams: 包含词表和合并规则的参数对象
    """
    # 从字节列表开始
    indices = list(map(int, string.encode("utf-8")))
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes

    for i in range(num_merges):
        # 统计所有相邻对的频率
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # 遍历所有相邻对
            counts[(index1, index2)] += 1

        if not counts:
            break

        # 找到最频繁的一对
        pair = max(counts, key=counts.get)
        index1, index2 = pair

        # 合并该对
        new_index = 256 + i
        merges[pair] = new_index
        vocab[new_index] = vocab[index1] + vocab[index2]
        indices = merge(indices, pair, new_index)

    return BPETokenizerParams(vocab=vocab, merges=merges)


# #### 4. 完整示例：训练与使用 BPE 分词器
# 
# 让我们通过一个完整的例子来演示 BPE 的训练和使用过程。

# In[12]:


# 使用示例文本训练 BPE 分词器
training_data = "the cat in the hat"
params = train_bpe(training_data, num_merges=3)

print(f"训练语料: '{training_data}'")
print(f"\n合并规则 (前 3 个):")
for i, (pair, new_index) in enumerate(list(params.merges.items())[:3]):
    pair_bytes = [params.vocab[pair[0]], params.vocab[pair[1]]]
    print(f"  合并 {i+1}: {pair_bytes} -> Token {new_index} ({params.vocab[new_index]})")

# 使用训练好的分词器
tokenizer = BPETokenizer(params)
test_string = "the hat"

# 编码
encoded = tokenizer.encode(test_string)
print(f"\n测试字符串: '{test_string}'")
print(f"Token 序列: {encoded}")

# 解码
decoded = tokenizer.decode(encoded)
print(f"还原字符串: '{decoded}'")
print(f"编码-解码一致性: {test_string == decoded}")

# 计算压缩比
ratio = get_compression_ratio(test_string, encoded)
print(f"压缩比: {ratio:.2f}")


# ### 分词方法总结
# 
# 我们对比了多种分词方法：
# 
# | 方法 | 词表大小 | 压缩比 | 主要问题 |
# |------|---------|--------|---------|
# | **基于字符** | ~150K | ~1 | 词表过大，稀疏性严重 |
# | **基于字节** | 256 | 1.0 | 序列过长，效率低下 |
# | **基于单词** | 无限增长 | 较高 | 词表爆炸（希普斯定律），OOV 问题 |
# | **BPE（子词）** | 可控制 | 良好 | 平衡词表大小与序列长度 |
# 
# **BPE 的优势**：
# - ✅ **固定词表大小**：通过控制合并次数，可以预先确定词表大小。
# - ✅ **处理未知词**：即使遇到训练时未见过的词，也能通过子词组合表示，无需 `[UNK]` Token。
# - ✅ **多语言支持**：字节级 BPE 可以处理任意 Unicode 字符。
# - ✅ **压缩比优化**：常见序列合并为单 Token，罕见序列保持拆分，在压缩比和词表大小之间取得平衡。
# 
# **当前实现的局限性**：
# - ⚠️ `encode()` 方法遍历所有合并规则，效率较低。实际应用中应只处理相关的合并。
# - ⚠️ 未处理特殊 Token（如 `<|endoftext|>`）。
# - ⚠️ 未使用预分词（pre-tokenization），如 GPT-2 的正则表达式。
# 
# **未来方向**：
# 虽然分词目前是语言模型的"必要之恶"，但未来可能直接基于字节进行建模。一些研究（如 ByT5、MEGABYTE）正在探索"无分词器"的方法，但在前沿大模型中尚未成为主流。

# ## Appendix: Minbpe 
# 
#  Karpathy 的 [minbpe](https://github.com/karpathy/minbpe) 库，尝试在基础 BPE 的基础上进行“工业级”改造。
# 
# 
# #### 1. 优化编码逻辑：按 Rank 贪心合并
# 
# 我们之前的 `BPETokenizer.encode()` 实现是一个简化的版本：它简单地按照训练得到的 `merges` 列表顺序，依次遍历所有规则。
# 
# 但真实的 BPE 编码（如 `tiktoken`）采用的是一种**贪心策略**：
# 1. **Rank 的概念**：训练时产生的 merge 规则是有顺序的（越早合并的 pair，优先级越高，Rank 越小）。
# 2. **动态选择**：编码时，我们不应该死板地遍历规则，而应该看**当前序列中实际出现了哪些 pair**，然后在这些 pair 中选择 **Rank 最小**（优先级最高）的一个进行合并。
# 3. **循环执行**：合并后，序列发生了变化，产生了新的 pair，于是我们重复上述过程，直到没有可合并的 pair 为止。
# 
# 这种方式不仅更严谨，也是后续实现正则分词（Regex Split）和特殊 Token（Special Tokens）的基础。

# In[13]:


from collections import defaultdict

def _get_stats(indices: list[int]) -> dict[tuple[int, int], int]:
    """
    统计当前序列里出现过的相邻 pair。
    注意：这里返回的计数其实不重要，我们只需要用它来枚举当前序列里到底有哪些 pair。
    """
    stats = defaultdict(int)
    for pair in zip(indices, indices[1:]):
        stats[pair] += 1
    return stats


# 接下来继承之前的 `BPETokenizer`，重写 `encode` 方法。核心变化在于把 `for` 循环遍历 merges 改成了 `while` 循环动态查找最优 pair。

# In[14]:


class BPETokenizerV2(BPETokenizer):
    """更贴近 minbpe 的实现：反复合并当前可用的最小 rank pair。"""

    def encode(self, string: str) -> list[int]:
        # 1. 初始状态：将字符串转换为 UTF-8 字节的整数列表
        indices = list(map(int, string.encode("utf-8")))
        merges = self.params.merges

        while len(indices) >= 2:
            # 2. 获取当前序列中所有的相邻 pair
            stats = _get_stats(indices)

            # 3. 找到当前 stats 里存在、且在 merges 表中 rank 最小（最早学到）的那个 pair
            #    如果没有找到（即当前所有 pair 都不在合并规则里），就停止
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))

            if pair not in merges:
                break

            # 4. 执行合并：将序列中所有的这个 pair 替换为新的 token id
            indices = merge(indices, pair, merges[pair])

        return indices


# 最后，写一个简单的玩具示例来验证一下。我们手动构造几条合并规则：先合并 `(a, b)`，再合并 `(ab, c)`。用新的 V2 分词器应该能正确处理这种依赖顺序。

# In[15]:


# 手动构造一个微型词表和规则
# 假设我们只想合并 "abc" 这一串
vocab = {i: bytes([i]) for i in range(256)}
a, b, c = ord("a"), ord("b"), ord("c")

# 规则 1: a + b -> 256 (rank 高，先合并)
# 规则 2: 256 + c -> 257 (rank 低，后合并)
merges = {(a, b): 256, (256, c): 257}
vocab[256] = vocab[a] + vocab[b]
vocab[257] = vocab[256] + vocab[c]

params = BPETokenizerParams(vocab=vocab, merges=merges)

# 对比测试
fast = BPETokenizerV2(params)
s = "abc"
encoded_ids = fast.encode(s)

print(f"原字符串: {s}")
print(f"编码结果: {encoded_ids} (应为 [257])")
print(f"解码验证: {fast.decode(encoded_ids) == s}")


# #### 2. Regex 预分词（Pre-Tokenization）
# 
# 在 Step 1 中，我们优化了 BPE 的合并策略，但目前的 `BPETokenizerV2` 仍然把整段文本看作一个超长的字节流。这意味着它可能会学到一些“跨越单词边界”的怪异 Token，比如把 `"dog."` 中的 `g.` 合并成一个 Token。
# 
# GPT 系列分词器（从 GPT-2 开始）引入了一个核心机制：**基于正则的预分词**。
# 
# 它的思路非常直观：
# 1. 先用一个精心设计的正则表达式，把文本切成一个个**独立的片段（Chunk）**。
# 2. **强制规定**：BPE 的合并操作**只能在每个 Chunk 内部进行**，绝不允许跨越 Chunk 边界。
# 
# 这样做的好处是显而易见的：标点符号、空格、单词会被自然地隔离开，Token 的语义边界更加清晰。这也是为什么我们现在的模型能很好地处理代码（代码中有大量标点和缩进），因为正则规则专门为这些场景做了优化。

# In[16]:


import regex

# GPT-4 的拆分规则
# 注意：这需要 python 的 `regex` 库（pip install regex），因为它用到了 \p{L} 等 Unicode 属性，
# Python 标准库的 re 模块不支持。
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# 编译 Pattern
compiled_pattern = regex.compile(GPT4_SPLIT_PATTERN)

# 看看它到底把文本切成了什么样
text = "Hello, world! 123. \n   Code: a = b"
chunks = compiled_pattern.findall(text)
print(chunks)


# 这个正则表达式初看非常复杂，但它其实是由几个简单的逻辑通过 `|`（或）拼接起来的。我们可以拆解来看：
# 
# 1. `'(?i:[sdmt]|ll|ve|re)`：**处理缩写**。比如 `I'll`, `you've`, `it's`。它会把 `'ll`, `'ve`, `'s` 切分出来，保证词根独立。
# 2. `[^\r\n\p{L}\p{N}]?+\p{L}+`：**处理单词**。`\p{L}` 代表所有 Unicode 字母。这里允许单词前面带一个非字母数字的符号（比如空格）。
# 3. `\p{N}{1,3}`：**处理数字**。它把数字切成最多 3 位一组。比如 `123456` 会被切成 `123` 和 `456`。这解释了为什么 GPT 在做大数运算时有时会犯错——因为它看到的不是一个完整的数字，而是几个碎片。
# 4. `\s*[\r\n]` 和 `\s+(?!\S)`：**处理空白和换行**。保证格式化信息（特别是代码缩进）被单独保留。
# 
# 有了这个 Pattern，我们就有了 BPE 的“硬边界”。

# In[17]:


class RegexTokenizer(BPETokenizerV2):
    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        self.pattern = pattern
        self.compiled_pattern = regex.compile(pattern)
        # 初始化空参数，等待 train() 填充
        super().__init__(BPETokenizerParams(vocab={}, merges={}))

    def train(self, text: str, vocab_size: int):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 1. 预分词：将文本切分为 chunks
        text_chunks = self.compiled_pattern.findall(text)

        # 2. 初始化：将每个 chunk 转换为 UTF-8 字节列表
        # ids_list 是一个列表的列表：[ [chunk1_bytes], [chunk2_bytes], ... ]
        ids_list = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            # 3. 统计全语料中所有 chunk 内部的 pair 频次
            stats = defaultdict(int)
            for chunk_ids in ids_list:
                # 注意：这里我们累加所有 chunk 的统计结果
                # 这样学到的规则是全局最优的
                for pair in zip(chunk_ids, chunk_ids[1:]):
                    stats[pair] += 1

            # 如果没有可合并的 pair 了（比如所有 chunk 都变成单个 token 了），提前结束
            if not stats:
                break

            # 4. 选出全局最高频 pair
            pair = max(stats, key=stats.get)

            # 5. 记录 merge 规则
            idx = 256 + i
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            # 6. 在所有 chunk 中执行合并
            ids_list = [merge(chunk_ids, pair, idx) for chunk_ids in ids_list]

        # 更新参数
        self.params = BPETokenizerParams(vocab=vocab, merges=merges)

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        """对单个 chunk 进行 BPE 编码"""
        indices = list(text_bytes)
        merges = self.params.merges
        while len(indices) >= 2:
            stats = _get_stats(indices)
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            indices = merge(indices, pair, merges[pair])
        return indices

    def encode_ordinary(self, text: str) -> list[int]:
        """核心：先 Regex 切分，再对每个 Chunk 独立 BPE"""
        text_chunks = self.compiled_pattern.findall(text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids


# #### 3. Special Tokens：打破规则
# 
# BPE 只能处理它见过的 bytes，但在实际的大模型应用中，我们经常需要一些具有特殊功能的标记，比如：
# - `<|endoftext|>`：表示文档结束。
# - `<|im_start|>` / `<|im_end|>`：在 Chat 模型中标记对话轮次。
# 
# 这些 Token 有一个特点：它们**绝不能被切分**。
# 
# 即使文本里出现了 `<|endoftext|>`，如果走普通的 BPE 流程，它可能会被切成 `<` `|` `endo` `ftext` `|` `>` 等一堆碎片，彻底失去语义。
# 
# 因此，`minbpe` 在编码时引入了**最高优先级的切分**：
# 1. **用户注册**：告诉分词器哪些字符串是特殊的（如 `{"<|endoftext|>": 100257}`）。
# 2. **优先切分**：在做 Regex 分词之前，先用 `re.split` 把这些特殊字符串完整地切出来。
# 3. **直接映射**：切出来的特殊字符串直接查表转换为 ID，**跳过后续所有的 Regex 和 BPE 流程**。
# 
# 这就是为什么你在调用 `tiktoken` 时经常看到 `allowed_special` 参数——它决定了哪些特殊字符是被当作指令执行，哪些只是被当作普通文本编码。

# #### 4.Demo
# 最后，让我们用一个极简的 Demo 来直观感受一下 `RegexTokenizer` 的效果。
# 
# 我们将对比它与普通分词器在处理**标点符号边界**时的不同，并演示 **Special Tokens** 是如何被优先识别的。注意观察 `chunks` 变量，它揭示了 BPE 算法眼中的“世界”是如何被正则表达式预先切碎的。

# In[18]:


# 1. 初始化一个 RegexTokenizer（复用前面的类定义）
tokenizer = RegexTokenizer()

# 2. 演示 Regex 分块（Pre-tokenization）
text = "Hello, world! 123"
print(f"原始文本: {repr(text)}")

# 看看 GPT-4 的正则把这句话切成了什么样
# 注意：标点符号（, !）被独立切分了，数字也被切分了
chunks = tokenizer.compiled_pattern.findall(text)
print(f"Regex 切分结果 (Chunks): {chunks}")
# 输出预期: ['Hello', ',', ' world', '!', ' ', '123']
# 解释: BPE 合并只能在这些 ['Hello', ',', ' world'...] 内部发生，永远不会跨越边界。


# 3. 演示 Special Tokens 的最高优先级
# 假设我们注册了一个特殊 Token
special_tokens = {"<|endoftext|>": 100257}
# 注意：这里我们还没实现完整的 register_special_tokens 方法，
# 在完整实现中，encode() 会先用 re.split(special_pattern) 切分。

s = "Hello<|endoftext|> world"
print(f"\n含特殊 Token 的文本: {repr(s)}")

# 模拟 encode() 中的第一步切分逻辑：
import regex as re
special_pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"
parts = re.split(special_pattern, s)

print(f"Special Token 切分结果: {parts}")
# 输出预期: ['Hello', '<|endoftext|>', ' world']
# 解释: <|endoftext|> 被完整保留，不会被切碎，也不会参与后续的 Regex 分词。


# In[19]:


# === End-to-End Demo: 从训练到特殊 Token 处理 ===

# 1. 准备训练语料
# 我们故意让 "hello" 和 "world" 频繁出现，同时在它们后面紧跟不同的标点
train_text = "hello world! " * 50 + "hello world. " * 50 + "hello world? " * 50
print(f"语料长度: {len(train_text)} 字符")


# In[20]:


# 2. 训练分词器
# 我们只学 3 个新 token (256, 257, 258)，看看它会学到什么
vocab_size = 256 + 3
tokenizer = RegexTokenizer()
tokenizer.train(train_text, vocab_size=vocab_size)


# In[21]:


# 3. 观察学到的规则
print("\n[训练结果]")
print("学到的 Merges:", tokenizer.params.merges)
# 预期行为：
# 由于 Regex 预分词把单词和标点（如 '!', '.', '?'）强制切开了，
# BPE 只能学到单词内部的组合（如 'he'+'ll', 'hell'+'o'），
# 而永远学不到跨单词+标点的组合（如 'd'+'!'）。
# 这就是 RegexTokenizer 的核心作用：保证 Token 的语义纯洁性。


# In[22]:


# 4. 测试编码 (验证 Regex 边界隔离)
test_text = "hello world!"
ids = tokenizer.encode_ordinary(test_text)
print(f"\n[编码测试]\n原文: {repr(test_text)}")
print(f"Token IDs: {ids}")
print(f"解码回看: {[tokenizer.params.vocab[i] for i in ids]}")
# 注意观察：'!' 应该被单独编码，没有粘在 'world' 后面


# In[23]:


# 5. 演示 Special Token
# 手动模拟一下 allowed_special="all" 的效果
special_tokens = {"<|endoftext|>": 100257}
s = "hello<|endoftext|>world"

# 简化的处理逻辑：先切 Special，再对剩下的部分做 encode_ordinary
parts = re.split(f"({'|'.join(re.escape(k) for k in special_tokens)})", s)
final_ids = []
for part in parts:
    if part in special_tokens:
        final_ids.append(special_tokens[part])
    else:
        final_ids.extend(tokenizer.encode_ordinary(part))

print(f"\n[Special Token 测试]\n原文: {repr(s)}")
print(f"最终 IDs: {final_ids}")
# 预期：中间包含 100257，且左右两侧的 hello/world 被正确处理

