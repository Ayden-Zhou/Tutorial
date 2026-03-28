# %% [markdown]
# ## 1. 数据过滤算法 (Filtering Algorithms)
#
# 在构建高质量语言模型的过程中，**数据选择与过滤**是提升模型性能的核心步骤。我们的目标是从海量的、质量参差不齐的原始数据中，提取出与目标任务最相关、质量最高的部分。
#
# ### 1.1 算法核心逻辑与构建模块
#
# 数据过滤的本质是一个搜索与匹配的过程：
# * **输入概念**：给定一小组**目标数据 (Target Data) $T$** 和海量的**原始数据 (Raw Data) $R$**。
# * **执行目标**：在 $R$ 中找到一个子集 $T'$，使得 $T'$ 在分布或特征上与 $T$ 尽可能相似。
#
#
# <div align="center">
#   <img src="../images/raw-target-schema.png" width="600" />
#   <p>原始数据与目标数据的过滤逻辑示意图</p>
# </div>
#
# #### 数据过滤的设计准则 (Desiderata)
#
# 在设计或选择过滤算法时，我们需要权衡以下两个核心指标：
# 1.  **泛化性 (Generalization)**：我们不希望 $T'$ 只是 $T$ 的简单复制，算法需要能从目标数据中泛化，筛选出具有相似语义但内容不同的新数据。
# 2.  **极致的性能 (Extreme Speed)**：由于原始数据 $R$ 的规模通常达到 TB 甚至 PB 级，过滤算法必须具备极高的计算效率，以便在大规模集群上运行。
#
# > **延伸阅读**：
# > 如果你想深入研究各种数据选择策略，可以参考这篇综述论文：[Data Selection for Language Models (Survey)](https://arxiv.org/abs/2402.16827)
#
#
#

# %% [markdown]
# ### 1.2 KenLM：高效的语言模型过滤工具
#
# **KenLM** 是工业界和科研界在进行大规模文本清洗时最常用的基准工具之一。它最初是为机器翻译任务开发的，因其极致的推理速度和内存效率而闻名。
#
# #### Kneser-Ney 平滑算法推导
#
# KenLM 的核心是基于 **n-gram** 统计模型，并采用了 **Kneser-Ney (KN) 平滑**技术。相比简单的极大似然估计（MLE），KN 平滑能更好地处理长尾词汇和未登录词问题。
#
# 其核心推导逻辑如下：
#
# 对于一个 $n$-gram，我们计算其概率 $P_{KN}(w_i | w_{i-n+1}^{i-1})$，它由两部分组成：当前词的折扣概率贡献和低阶模型的插值。
#
# 1. **绝对折扣 (Absolute Discounting)**：
#    $$P_{KN}(w_i | w_{i-n+1}^{i-1}) = \frac{\max(count(w_{i-n+1}^i) - d, 0)}{\sum_{w} count(w_{i-n+1}^{i-1}w)} + \lambda(w_{i-n+1}^{i-1}) P_{KN}(w_i | w_{i-n+2}^{i-1})$$
#    其中 $d$ 是折扣常数。
#
# 2. **连续概率 (Continuation Probability)**：
#    在低阶模型中，KN 不再计算词频，而是计算该词作为“新接续”出现的频率：
#    $$P_{continuation}(w_i) = \frac{|\{w_{i-1} : count(w_{i-1}, w_i) > 0\}|}{\sum_{w_j} |\{w_{i-1} : count(w_{i-1}, w_j) > 0\}|}$$
#
# #### KenLM 的特点
# * **算法简单**：基于计数和归一化（Count and Normalize），没有复杂的神经网络参数。
# * **速度极快**：采用高效的存储结构（如 Probing 或 Trie），非常适合作为预训练的第一道过滤关卡。
# * **相关资源**：
#     * [Kneser-Ney Smoothing - Wikipedia](https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing)
#     * [KenLM 官方文档/文章](https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing)（注：此链接指向KN平滑原理）

# %% [markdown]
# ### 1.3 核心概念：极大似然估计与平滑 (Concepts: MLE & Smoothing)
#
# 在使用 KenLM 进行过滤之前，我们需要理解 n-gram 语言模型背后的数学直觉。
#
# #### 极大似然估计 (Maximum Likelihood Estimation, MLE)
# 最基础的 n-gram 模型依赖于极大似然估计。假设我们使用一个 3-gram ($n=3$) 模型，要计算单词 "in" 出现在 "the cat" 之后的概率，可以通过计数来直接计算：
#
# $$P(\text{in} | \text{the cat}) = \frac{\text{count}(\text{the cat in})}{\text{count}(\text{the cat})}$$
#
# #### 稀疏性问题 (The Sparsity Problem)
# MLE 虽然直观，但在实际应用中面临一个巨大的挑战：**稀疏计数 (Sparse Counts)**。
# 对于较大的 $n$（例如 4-gram 或 5-gram），很多合法的词组在训练语料中可能从未出现过。这意味着分子的计数为 0，导致预测概率为 0。这显然是不合理的，因为“未在训练集中出现”并不代表“在语言中不可能发生”。
#
# #### 解决方案：Kneser-Ney 平滑
# 为了解决稀疏性问题，我们引入 **Kneser-Ney 平滑**。它的核心思想利用低阶 n-gram 的信息来弥补高阶 n-gram 的缺失。
#
# * **直觉理解**：如果我们需要估计 $P(\text{in} | \text{the cat})$，但 "the cat in" 从未出现过，我们不应直接给它零概率，而是应该参考 $P(\text{in} | \text{cat})$ 的概率。
# * 也就是说，$P(\text{in} | \text{the cat})$ 的估计值应该部分依赖于 $P(\text{in} | \text{cat})$。
#
# > **参考资料**
# > * [Kneser-Ney smoothing - Wikipedia](https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing)
# > * [相关文章介绍](https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing)
#
#
#
# ### 1.4 代码实战：使用 KenLM 计算困惑度 (Implementation)
#
# 理解了原理后，我们来实际运行一下 KenLM。我们将加载一个在 Wikipedia 上预训练好的英文 n-gram 模型，并计算不同文本的**困惑度 (Perplexity)**。困惑度越低，说明模型认为该句子越自然（或者说质量越高/越符合训练分布）。
#
# 首先，定义模型路径和下载链接：

# %%
import os
import requests
import shutil
from io import BytesIO

def download_file(url: str, filename: str):
    """下载文件，如果已存在则跳过"""
    if not os.path.exists(filename):
        print(f"Downloading {url} to {filename}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        response = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0"
        })
        with open(filename, "wb") as f:
            shutil.copyfileobj(BytesIO(response.content), f)

# 使用
model_url = "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.arpa.bin"
model_path = "../var/en.arpa.bin"  # 相对于 notebook/ 目录

# download_file(model_url, model_path)

# %% [markdown]
# 接下来，加载 KenLM 模型。

# %%
# 加载模型
model = kenlm.Model(model_path)

# %% [markdown]
# 为了评估文本质量，我们需要定义一个计算函数。这里有几个细节需要注意：
# 1.  **预处理**：KenLM 对输入格式有一定要求，通常需要添加句首 `<s>` 和句尾 `</s>` 标记。这里的实现是一个简单的 "Hack" 版本。
# 2.  **归一化**：`model.score` 返回的是对数概率总和。为了避免长文档因为 token 多而得分极低（绝对值大），我们需要用 token 数量进行归一化，计算出困惑度（Perplexity）。
#
# 公式如下：
# $$\text{Perplexity} = \exp\left(-\frac{\text{Score}}{N}\right)$$

# %%
def compute(content: str):
    # Hacky preprocessing: 简单的预处理，处理标点并添加句子边界标记
    content = "<s> " + content.replace(",", " ,").replace(".", " .") + " </s>"

    # 计算 log p(content)
    # 这里的 score 是 log10 概率
    score = model.score(content)

    # Perplexity normalizes by number of tokens to avoid favoring short documents
    # 归一化：除以 token 数量，避免长短文档不公平比较
    num_tokens = len(list(model.full_scores(content)))
    perplexity = math.exp(-score / num_tokens)

    return score, perplexity

# %% [markdown]
# #### 过滤效果测试
# 现在我们输入几种不同类型的文本，观察 KenLM 的打分情况：
# 1.  **高质量文本**：斯坦福大学的历史介绍。
# 2.  **特定领域文本**：课程评分政策（虽合乎语法，但可能与维基百科的分布略有不同）。
# 3.  **乱码/无意义文本**：键盘乱敲的字符。
# 4.  **重复文本**：重复单词。
#
# 请关注 `perplexity` 的值：

# %%
# 测试案例 1: 高质量的自然语言文本
score, perplexity = compute("Stanford University was founded in 1885 by Leland and Jane Stanford as a tribute to the memory of their only child, Leland Stanford Jr.")
print(f"Sample 1 - Score: {score:.2f}, Perplexity: {perplexity:.2f}")

# 测试案例 2: 领域特定的自然语言文本
score, perplexity = compute("If you believe that the course staff made an objective error in grading, you may submit a regrade request on Gradescope within 3 days after the grades are released.")
print(f"Sample 2 - Score: {score:.2f}, Perplexity: {perplexity:.2f}")

# 测试案例 3: 无意义的乱码
score, perplexity = compute("asdf asdf asdf asdf asdf")
print(f"Sample 3 - Score: {score:.2f}, Perplexity: {perplexity:.2f}")

# 测试案例 4: 极其重复的文本
score, perplexity = compute("the the the the the the the the the the the the the the the the")
print(f"Sample 4 - Score: {score:.2f}, Perplexity: {perplexity:.2f}")

# %% [markdown]
# ### 1.5 进阶过滤策略：CCNet Pipeline
#
# KenLM 虽然高效，但它是一个“粗糙”的工具。它只能告诉我们句子是否通顺（Perplexity），却很难判断内容质量的高低。为了在海量数据中进一步提纯，CCNet 提出了一种经典的流水线策略，并在后续被 LLaMA 等大模型沿用。
#
# #### 核心策略
# CCNet 的处理单元是**段落 (Paragraphs)**，而非整篇文章。其核心逻辑非常直接：
# 1.  **分段计算**：计算每个段落的困惑度（Perplexity）。
# 2.  **排序与截断**：按困惑度从小到大排序。
# 3.  **保留头部**：只保留困惑度最低（最自然）的 **前 1/3** 的段落。
#
# 这种方法在效率和质量之间取得了很好的平衡。尽管 KenLM 模型相对简单（Crude），但通过这种阈值过滤，能有效去除极其嘈杂的文本。
#
# > **参考资料**
# > * [CCNet Paper: Extracting High Quality Monolingual Datasets from Web Crawl Data](https://arxiv.org/pdf/1911.00359)
#
#
#
# ### 1.6 FastText 分类器 (FastText Classifier)
#
# 除了基于困惑度的过滤，我们通常还需要训练一个二分类器（Quality Classifier），用于直接判断一段文本是“高质量”还是“低质量”。在这里，**FastText** 是最经典的基准模型。
#
# #### 为什么选择 FastText？
# FastText 由 Facebook AI Research (FAIR) 提出，其设计目标非常明确：**在文本分类任务上，获得与深度神经网络相当的准确率，但速度要快几个数量级。**
#
# 对于数据过滤任务而言，我们需要处理 TB 级别的数据，推理速度是第一要素。BERT 或 Llama 虽然准确，但在 CPU 集群上跑数据清洗太慢了。
#
# > **参考资料**
# > * [FastText Paper: Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759)
#
#
#
# ### 1.7 架构对比：从词袋到 FastText (Model Architecture)
#
# 为了理解 FastText 的高效性，我们需要对比一下基础的词袋模型（Bag of Words, BoW）和 FastText 的改进。
#
# #### Baseline: 传统的词袋分类模型
# 假设我们有一个简单的分类任务。
# * $L$: 输入文本长度
# * $V$: 词表大小 (Vocabulary size)
# * $K$: 类别数 (Number of classes)
#
# 最直接的做法是学习一个从词表直接映射到类别的权重矩阵 $W \in \mathbb{R}^{V \times K}$。模型将输入中所有词的嵌入取平均，然后通过 Softmax 输出。
#
#  **存在的问题：**
# 参数量为 $V \times K$。当词表很大（例如 100k+）且类别很多时，这个单层矩阵会变得非常巨大，计算和存储都很昂贵。
#
# #### FastText 的改进：引入隐层 (Low-rank Approximation)
# FastText 引入了一个隐层维度 $H$ (Hidden dimension)，将大矩阵分解。
# 1.  先将词映射到低维空间 $H$: $W_{in} \in \mathbb{R}^{V \times H}$
# 2.  再从 $H$ 映射到类别 $K$: $W_{out} \in \mathbb{R}^{H \times K}$
#
# 总参数量从 $V \times K$ 降低到了 $H \times (V + K)$。由于 $H$ 通常很小（例如 10-100），这极大地减少了计算量。
#
# #### 训练技巧
# 为了进一步提升速度，FastText 在工程实现上还有两个关键优化：
# 1.  **异步并行 SGD (Parallelized, asynchronous SGD)**：允许多个线程同时更新参数，不加锁，牺牲微小的数学精确度换取极大的速度提升。
# 2.  **学习率调度**：采用线性插值的方式，将学习率从初始值线性衰减到 0。
#     * [代码实现参考 (GitHub)](https://github.com/facebookresearch/fastText/blob/main/src/fasttext.cc#L653)
#
#
# ### 1.8 特征工程：N-grams 与 Hashing Trick
#
# FastText 的另一个核心贡献在于特征表示。它不仅仅使用单词（Unigram），还使用了 **Bag of n-grams** 来捕获局部词序信息。
#
# 例如，对于句子 `x`：
#
# x_tokens = ["the", "cat", "in", "the", "hat"]
# # Bigrams 示例
# bigrams = ["the cat", "cat in", "in the", "the hat"]
#
# #### 词表爆炸问题
# 如果我们把所有出现的 bi-grams、tri-grams 都加入词表，词表大小 $V$ 会呈指数级增长，且理论上是无界的（Unbounded）。这会导致内存溢出。
#
# #### 解决方案：Hashing Trick
# FastText 巧妙地使用了 **Hashing Trick**。它不维护一个显式的“n-gram 词典”，而是直接将 n-gram 字符串通过 Hash 函数映射到固定数量的桶（Bins）中。
#
# $$\text{Index} = \text{Hash}(\text{n-gram}) \pmod{\text{NumBins}}$$
#
# 这样，无论 n-gram 有多少，特征空间的维度永远被限制在 `num_bins` 大小（例如 1000 万）。
#
# **Python 模拟实现：**

# %%
import mmh3 # 需要安装: pip install mmh3

x = ["the cat", "cat in", "in the", "the hat"]

# 在工业界实践中，num_bins 通常设为 10^7 左右
num_bins = 8 

# 将每个 bigram 映射为一个整数索引
hashed_x = [mmh3.hash(bigram) % num_bins for bigram in x]

print(f"Original bigrams: {x}")
print(f"Hashed indices:   {hashed_x}")

# %% [markdown]
# #### 总结：用于数据过滤的 FastText
# 在数据过滤场景下，我们的任务通常非常简单：
# * **类别 $K=2$**：高质量 (Good) vs 低质量 (Bad)。
# * **模型本质**：当 $K=2$ 且 $H$ 很小时，FastText 本质上就是一个高效的**线性分类器**，建立在 n-gram 哈希特征之上。
#
# 虽然我们完全可以使用 BERT 或 Llama 作为分类器来获得更好的性能，但在处理万亿级别的 Token 时，FastText 提供的“足够好”的精度与“极快”的速度，使其成为数据清洗流水线中不可替代的一环。

# %% [markdown]
# ### 1.9 DSIR：基于重要性重采样的数据选择
#
# 在 FastText 之后，学术界开始思考一个更本质的问题：我们不仅仅想区分“好”与“坏”，我们更希望从海量的原始数据（Raw Data）中，选出一个子集，使其分布尽可能接近我们要解决的目标任务（Target Data）。
#
# 2023 年提出的 **DSIR (Data Selection for Language Models via Importance Resampling)** 就是这一思路的代表作。
#
# > **参考资料**
# > * [DSIR Paper: Data Selection for Language Models via Importance Resampling](https://arxiv.org/abs/2302.03169)
#
# <div align="center">
#   <img src="https://www.jinghong-chen.net/content/images/size/w1200/2023/12/Screenshot-2023-12-24-at-17.41.38.png" width="600" />
#   <p>图 1.2：DSIR 流程示意图 - 利用重要性权重从 Raw Data 中重采样</p>
# </div>
#
# #### 1.9.1 问题设定 (Setup)
# * **目标数据集 $D_p$ (Target)**：高质量、但规模较小的数据（例如 Wikipedia，或者某个特定领域的教科书）。
# * **提议/原始数据集 $D_q$ (Proposal/Raw)**：海量、但包含大量噪声的数据（例如 CommonCrawl）。
#
# #### 1.9.2 演进思路
# **Take 1: 理想化的重要性采样**
# 理论上，我们可以这样做：
# 1.  在 $D_p$ 上训练一个语言模型，学习目标分布 $p$。
# 2.  在 $D_q$ 上训练一个语言模型，学习原始分布 $q$。
# 3.  对于 $D_q$ 中的每个样本 $x$，计算重要性权重 $w(x) = \frac{p(x)}{q(x)}$。
# 4.  根据权重 $w$ 对 $D_q$ 进行重采样。
#
# **Problem (困难)**：目标数据 $D_p$ 通常太小了，无法训练出一个足够好且不过拟合的生成式语言模型（比如 GPT-2 级别）。用过拟合的模型计算出的 $p(x)$ 是不可靠的。
#
# **Take 2: DSIR 的解决方案 (Hashed n-grams)**
# DSIR 的核心洞见是：**不要训练复杂的神经网络，回归简单的特征统计。**
# 它再次利用了 FastText 中的 **Hashed n-grams** 技术来估计分布 $p$ 和 $q$。
#
# 让我们用代码来模拟 DSIR 如何利用 Hash Bin 来估计文本的概率：

# %%
import mmh3
import numpy as np

# 假设这是一小段目标训练数据
training_text = "the cat in the hat"
num_bins = 4 # 在实际应用中，这里通常是 10^4 到 10^7

def get_hashed_ngrams(text: str):
    """
    将文本分割为 n-gram (这里简化为 unigram) 并映射到 hash bins
    """
    ngrams = text.split(" ") 
    return [mmh3.hash(ngram) % num_bins for ngram in ngrams]

# 1. 获取目标数据的特征表示
training_hashed_ngrams = get_hashed_ngrams(training_text)
print(f"Hashed Features: {training_hashed_ngrams}")

# 2. 估计目标分布 p (简单的频率计数)
# 统计每个 bin 出现的频率，作为该特征的概率估计
probs = [training_hashed_ngrams.count(x) / len(training_hashed_ngrams) for x in range(num_bins)]
print(f"Bin Probabilities (Model p): {probs}")

# 3. 计算新文本在目标分布 p 下的概率
# 假设特征之间独立 (Naive Bayes 假设)，将各特征概率相乘
hashed_ngrams = get_hashed_ngrams("the text")
prob = np.prod([probs[x] for x in hashed_ngrams])

print(f"Probability of 'the text' under p: {prob}")

# %% [markdown]
# #### 1.9.3 结果与对比
# DSIR 在 GLUE 基准测试上的表现略优于启发式的分类器（如 FastText）。
#
# <div align="center">
#   <img src="../images/dsir-results.png" width="700" />
#   <p>图 1.3：DSIR 与其他方法的性能对比</p>
# </div>
#
# **DSIR vs FastText:**
# * **更具原则性 (Principled)**：FastText 只是学习把数据分开（二分类），而 DSIR 试图建模数据的分布，能更好地捕捉数据的**多样性 (Diversity)**。
# * **计算复杂度相当**：两者都依赖于 Hashed n-grams，处理大规模数据时都非常快。
# * **改进空间**：两者都可以通过使用更好的特征模型（不仅仅是 n-gram）来进一步提升。
#
#
#
# ### 1.10 深入理解：重要性采样 (Importance Sampling)
#
# 为了彻底理解 DSIR 的数学原理，我们需要回顾一下蒙特卡洛方法中的**重要性采样**。这是从一个分布中估计另一个分布属性的核心技术。
#
# #### 算法流程
# 假设我们想要从目标分布 $p$ 中采样，但我们只能从提议分布 $q$ 中获取样本（因为 $D_q$ 是我们拥有的海量数据）。
#
# 1.  **Setup**: 定义目标 $p$ 和提议 $q$。
# 2.  **Sample**: 从 $q$ 中抽取 $n$ 个样本。
# 3.  **Reweight**: 计算每个样本的权重 $w_i = \frac{p(x_i)}{q(x_i)}$。如果一个样本在 $p$ 中概率高但在 $q$ 中概率低，它的权重就会很大。
# 4.  **Resample**: 根据归一化后的权重 $w$，重新抽取样本。
#
# 我们用一段 Python 代码来完整模拟这个过程：

# %%
import numpy as np

# 1. 定义分布
# 词表: [0, 1, 2, 3]
vocabulary = [0, 1, 2, 3]
# 目标分布 p: 我们希望更多地采样到 '2' 和 '3'
p = [0.1, 0.2, 0.3, 0.4]
# 提议分布 q: 但原始数据主要由 '0' 和 '1' 组成
q = [0.4, 0.3, 0.2, 0.1]

print(f"Target Distribution p: {p}")
print(f"Proposal Distribution q: {q}")

# 2. 从 q 中采样 (模拟获取原始数据)
n = 100
samples = np.random.choice(vocabulary, p=q, size=n)
print(f"\nInitial samples from q (first 20): {samples[:20]}")
# 你会发现这里有很多 0 和 1

# 3. 计算重要性权重 w = p(x) / q(x)
# 对应 DSIR 中: weight = p_model(doc) / q_model(doc)
w = [p[x] / q[x] for x in samples]

# 归一化权重
z = sum(w)
w_normalized = [w_i / z for w_i in w]

# 4. 根据权重重采样 (Resampling)
final_samples = np.random.choice(samples, p=w_normalized, size=n)

print(f"Resampled data (first 20): {final_samples[:20]}")
# 现在的样本分布应该更接近 p (更多 2 和 3)，尽管它们最初来自 q

# %% [markdown]
# ### 1.11 算法总结与通用框架 (Summary & Framework)
#
# 到目前为止，我们介绍了三种主流的数据过滤算法：**KenLM**, **fastText**, 和 **DSIR**。尽管它们在数学实现上有所不同，但本质上都遵循同一个**通用过滤框架**。
#
# #### 通用框架定义
# 给定目标数据集 $T$ (Target) 和原始数据集 $R$ (Raw)，我们的任务是找到 $R$ 的一个子集，使其与 $T$ 相似。
#
# 1.  **建模与评分**：基于 $R$ 和 $T$ 估计某种模型，并导出一个评分函数 $\text{score}(x)$。
# 2.  **筛选/采样**：根据分数保留 $R$ 中的样本。
#
# #### 框架的实例化 (Instantiations)
#
# 我们可以用这个框架来统一描述前面学到的算法：
#
# | 算法 | 类型 | 评分函数 $\text{score}(x)$ | 筛选策略 |
# | :--- | :--- | :--- | :--- |
# | **KenLM** | 生成式模型 | $p_T(x)$ <br> (基于目标数据的似然概率) | **阈值截断**：保留 $\text{score}(x) \ge \tau$ 的样本 |
# | **fastText** | 判别式分类器 | $p(T \mid x)$ <br> (分类器认为 $x$ 属于目标类别的概率) | **阈值截断**：保留 $\text{score}(x) \ge \tau$ 的样本 |
# | **DSIR** | 重要性重采样 | $\frac{p_T(x)}{p_R(x)}$ <br> (目标分布与原始分布的比率) | **概率重采样**：以正比于 $\text{score}(x)$ 的概率采样 $x$ |
#
#
# ## 2. 过滤应用实战 (Filtering Applications)
#
# 掌握了核心算法后，我们来看看这些工具在实际的大模型预训练数据处理中是如何被使用的。同一套过滤机制（Machinery）可以被灵活应用于各种不同的任务。
#
# ### 2.1 语言识别 (Language Identification, LID)
#
# 最基础的过滤任务就是**语言识别**：从混合语料中提取特定语言（如英语）的文本。
#
# #### 为什么要进行语言筛选？为什么不直接做多语言？
# 既然现在的模型都很强大，为什么我们还要费力去清洗出纯英语或其他特定语言的数据？
# 1.  **数据质量 (Data Quality)**：对于任何特定语言，整理和清洗出高质量数据的难度很大。针对一种语言优化清洗规则通常比针对所有语言更容易。
# 2.  **计算资源限制 (Compute Constraints)**：在计算资源有限的情况下（Tokens 总量恒定），这是一个零和博弈。分配给低质量语言的 Token 越多，留给高质量语言或核心语言的计算预算就越少。
#
# #### 多语言模型的权衡
# 不同的模型在多语言策略上选择了不同的路线：
# * **BLOOM**：英语数据占比仅 30%。事后分析认为其英语能力受到影响，部分原因是英语训练不足（Undertrained）。([相关论文](https://arxiv.org/pdf/2303.03915))
# * **Frontier Models (GPT-4, Claude, Llama 等)**：通常是重度多语言的（Heavily Multilingual），但前提是它们有足够的计算资源来保证每种语言都得到充分训练。
#
# ### 2.2 实战：使用 fastText 进行语言识别
#
# fastText 是目前工业界进行语言识别的事实标准工具。
# * **开箱即用**：官方提供了预训练好的分类器。
# * **覆盖广**：支持 176 种语言。
# * **数据源**：基于 Wikipedia, Tatoeba (翻译网站) 和 SETimes 等多语言语料训练。
#
# > **参考资料**
# > * [FastText Language Identification 文档](https://fasttext.cc/docs/en/language-identification.html)
#
# **应用示例**：
# 著名的开源数据集构建项目 **Dolma** 就使用了 fastText，策略是保留 $p(\text{English}) \ge 0.5$ 的网页。
#
# 让我们加载模型并测试一下：

# %%
import fasttext
import os

# 1. 定义模型下载地址和路径
# 注意：lid.176.bin 大约 126MB
model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
model_path = "../var/lid.176.bin"

# 简单检查并下载模型 (Linux/Mac 环境下使用 wget，Windows 用户需手动下载)
if not os.path.exists(model_path):
    if not os.path.exists("var"):
        os.makedirs("var")
    print(f"Downloading model to {model_path}...")
    os.system(f"wget {model_url} -O {model_path}")

# 2. 加载模型
# Suppress warning: fasttext 可能会报一些加载警告，属正常现象
model = fasttext.load_model(model_path)

# %%
# 3. 进行预测
# predict 返回格式通常为 (('__label__en',), array([0.98]))

def quick_predict(text):
    labels, probs = model.predict([text])
    # 格式化输出，只取第一个 label 和概率
    lang = labels[0][0].replace("__label__", "")
    prob = probs[0][0]
    return f"Lang: {lang:<4} | Prob: {prob:.4f} | Text: {text[:50]}..."

print("--- Standard Tests ---")
print(quick_predict("The quick brown fox jumps over the lazy dog."))  # English
print(quick_predict("Auf dem Wasser zu singen"))  # German
print(quick_predict("Bonjour!"))  # French
print(quick_predict("Feliz Navidad / Próspero año y felicidad / I wanna wish you a Merry Christmas"))  # Spanish + English (Code-switching)

print("\n--- Edge Cases ---")
# 重复文本
print(quick_predict("The quick brown fox jumps over the lazy dog. " * 2))
# 非正式英语 / 网络用语
print(quick_predict("OMG that movie was 🔥🔥! So dope 😎🤘!")) 
# LaTeX 代码 (通常会被识别为英文，因为命令是英文单词)
print(quick_predict("The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$."))
# 编程代码 (C++)
print(quick_predict("for (int i = 0; i < 10; i++)"))

# %% [markdown]
# ### 2.3 局限性与挑战 (Caveats)
#
# 虽然 fastText 非常好用，但在处理某些情况时仍需小心：
# * **短文本 (Short sequences)**：文本太短时特征不足，容易误判。
# * **低资源语言**：训练数据少的语言准确率较低。
# * **方言误伤**：可能会错误地过滤掉英语的某些方言（Dialects）。
# * **相似语言**：对于马来语 (Malay) 和印尼语 (Indonesian) 这种极度相似的语言，区分难度很大。
# * **混合语/代码切换 (Code-switching)**：例如上面 "Spanish + English" 的例子，模型只能给出一个主标签，这一定义本身就是模糊的。
#
#
# ### 2.4 案例研究：OpenMathText
#
# 为了展示如何组合使用上述技术，我们来看一个具体的案例：**OpenMathText**。
# 这是一个旨在从 CommonCrawl 中提取高质量数学文本的项目。
#
# > **参考资料**
# > * [OpenMathText Paper](https://arxiv.org/pdf/2310.06786)
#
# **Pipeline 设计：**
# OpenMathText 使用了一个多级过滤漏斗：
# 1.  **基于规则 (Rules)**：首先检查是否包含 LaTeX 命令等明显的数学特征。
# 2.  **KenLM 过滤**：在 ProofPile (高质量数学数据) 上训练 KenLM。保留困惑度 (Perplexity) $< 15000$ 的文本。
# 3.  **fastText 分类**：训练一个二分类器预测“是否为数学写作”。
#     * 如果是数学内容，保留阈值为 $0.17$。
#     * 如果不是数学内容，保留阈值为 $0.8$（意味着只有极其确定的非数学文本才会被当作通用文本保留，或者反过来理解：即使分类器认为不是数学，但置信度不高，也可能先通过）。
#
# **结果**：
# 该项目最终生成了 **14.7B tokens** 的数据。实验表明，使用这些精选数据训练出的 **1.4B 参数模型**，其数学推理能力优于那些使用 **20倍数据量** 训练出来的模型。
#
# 这再次证明了数据过滤的核心信条：**Quality > Quantity.**

# %% [markdown]
# ### 2.5 质量过滤策略 (Quality Filtering Strategies)
#
# 在完成了语言识别后，我们面临的最大挑战是如何定义“质量”。在业界，关于是否使用**基于模型的过滤 (Model-based Filtering)** 存在两种流派：
#
# 1.  **启发式流派**：倾向于不使用复杂模型，仅靠规则和去重。代表模型包括 **C4, Gopher, RefinedWeb, FineWeb, Dolma**。
# 2.  **模型过滤流派**：训练一个分类器来筛选数据。代表模型包括 **GPT-3, LLaMA, DCLM**。目前，这正逐渐成为一种主流标准（Becoming the norm）。
#
# 让我们来看看几个经典模型的过滤方案：
#
# ####  GPT-3 的分类器策略
# GPT-3 的数据处理是模型过滤的早期经典案例。
#
# * **正样本 (Positives)**：被认为是高质量的数据源，包括 Wikipedia, WebText2, Books1, Books2。
# * **负样本 (Negatives)**：未经过滤的 CommonCrawl 样本。
#
# **方法**：
# 他们使用 Spark 的 Tokenizer 提取单词特征，训练了一个**线性分类器 (Linear Classifier)** 来区分正负样本。
#
# > **参考资料**
# > * [GPT-3 Paper](https://arxiv.org/pdf/2005.14165)
# > * [Spark Tokenizer Features](https://spark.apache.org/docs/latest/ml-features#tokenizer)
#
# **采样策略 (Pareto Sampling)**：
# GPT-3 并没有简单地设定一个阈值（比如 score > 0.5），而是根据分类器的打分进行**随机采样**。为了在保留高质量数据的同时不完全丢弃低分数据（因为分类器可能会犯错，或者低分数据包含罕见知识），他们使用了帕累托分布（Pareto Distribution）。
#
# <div align="center">
#   <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Probability_density_function_of_Pareto_distribution.svg/325px-Probability_density_function_of_Pareto_distribution.svg.png" width="400" />
#   <p>图 2.1：帕累托分布概率密度函数示意图</p>
# </div>
#
# 我们可以用一段简单的 Python 代码来模拟这种基于分数的随机保留机制：

# %%
import numpy as np

def keep_document(score: float) -> bool:
    """
    根据文档得分决定是否保留该文档。
    score 越高，被保留的概率越大；但即使 score 较低，也有非零概率被保留。
    这里 alpha=9 是 GPT-3 论文中的超参数。
    """
    return np.random.pareto(9) > 1 - score

# 模拟测试
print(f"High score (0.99) kept? {keep_document(0.99)}")
print(f"Low score (0.10) kept? {keep_document(0.10)}")

# %% [markdown]
# #### LLaMA / RedPajama 的引用过滤
# LLaMA 在构建其训练集（后来被 RedPajama 复现）时，采用了一种更有趣的“正样本”定义方式：
#
# * **核心假设**：如果一个网页被维基百科（Wikipedia）引用了，那么它很可能是高质量的。
# * **正样本**：Wikipedia 参考文献中链接到的页面。
# * **负样本**：随机的 CommonCrawl 页面。
# * **策略**：训练分类器，保留被分类为正的文档。
#
# > **参考资料**
# > * [LLaMA Paper](https://arxiv.org/pdf/2302.13971)
#
# #### Phi-1: "Textbooks Are All You Need"
# 微软的 Phi-1 系列模型代表了数据质量的另一个极端：**合成数据与教科书级质量**。他们的理念是：如果你用极高质量的数据（如教科书），即使是小模型（1.3B）也能达到惊人的效果。
#
# > **参考资料**
# > * [Phi-1 Paper](https://arxiv.org/pdf/2306.11644)
#
#

# %% [markdown]
# ## 3. 数据去重 (Deduplication)
#
# 在收集完海量数据后，我们往往会发现其中充斥着大量的重复内容。去重（Deduplication）不仅能显著减少训练成本，还能提升模型性能。通常，我们将重复数据分为两类：
#
# 1.  **精确重复 (Exact duplicates)**：完全一样的内容。例如镜像网站、GitHub 上的 Fork 项目等。
# 2.  **近似重复 (Near duplicates)**：内容非常相似，仅有几个 Token 的差异。
#
# #### 近似重复的常见来源
# 近似重复在网络语料中非常普遍，常见的例子包括：
# * 服务条款 (Terms of Service) 和 许可证声明 (Licenses)。
# * **公式化写作 (Formulaic writing)**：复制粘贴的内容，或者由模板生成的文本。
# * 复制粘贴时产生的微小格式差异。
#
# <div align="center">
#   <img src="https://d3i71xaburhd42.cloudfront.net/4566c0d22ebf3c31180066ab23b6c445aeec78d5/5-Table1-1.png" width="600" />
#   <p>图 3.1：近似重复文本的示例</p>
# </div>
#
# **一个惊人的例子**：
# 在 C4 数据集中，下面这段产品描述竟然被重复了 **61,036 次**：
# > “by combining fantastic ideas, interesting arrangements, and follow the current trends in the field of that make you more inspired and give artistic touches. We’d be honored if you can apply some or all of these design in your wedding. believe me, brilliant ideas would be perfect if it can be applied in real and make the people around you amazed!
#
# #### 为什么要进行去重？
# 去重并不仅仅是为了节省存储空间，研究表明它对语言模型训练至关重要：
# * **提升训练效率**：数据量变少了，训练同样的 Epoch 所需的 FLOPs 更少。
# * **避免记忆化 (Avoid Memorization)**：模型不再死记硬背训练数据，这有助于缓解版权纠纷和隐私泄露的风险。
#
# > **参考资料**
# > * [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/pdf/2107.06499)
#
# #### 去重的设计空间 (Design Space)
# 在设计去重流水线时，我们需要回答三个核心问题：
# 1.  **处理单元 (What is an item?)**：是针对句子、段落还是整篇文档去重？
# 2.  **匹配策略 (How to match?)**：是要求完全精确匹配，还是寻找公共子序列，亦或是计算公共子项的比例？
# 3.  **处理动作 (What action to take?)**：发现重复后，是全部删除，还是只保留一份？
#
# #### 核心挑战
# 去重的本质是将一个项与其他所有项进行比较。如果我们有 $N$ 个文档，两两比较的复杂度是 $O(N^2)$。对于十亿级别的网页数据，这是不可接受的。因此，我们需要寻找 **线性时间复杂度 (Linear time)** 的算法来扩展到大规模数据上。
#
#
#
# ### 3.1 哈希函数基础 (Hash Functions)
#
# 为了实现高效的比对，我们离不开**哈希函数**。
#
# * **定义**：哈希函数 $h$ 将一个任意长度的数据项（Item）映射为一个固定长度的哈希值（整数或字符串）。
# * **特性**：哈希值的数据规模远小于原始数据。
# * **冲突 (Collision)**：当 $x \neq y$ 但 $h(x) = h(y)$ 时，我们称发生了哈希冲突。
#
# #### 效率与抗冲突性的权衡
# 在选择哈希算法时，我们通常面临效率（Speed）和抗冲突性（Collision Resistance）的权衡：
#
# 1.  **加密哈希函数 (Cryptographic hash functions)**：例如 SHA-256。
#     * 特点：抗冲突性极强，几乎不可能找到两个不同的输入产生相同的哈希值。
#     * 缺点：计算速度慢。常用于比特币挖矿、数字签名等安全领域。
# 2.  **非加密哈希函数**：例如 **DJB2, MurmurHash, CityHash**。
#     * 特点：计算速度极快。
#     * 缺点：抗冲突性较弱（但在去重场景下通常可以接受）。常用于哈希表（Hash Tables）实现。
#
# > **参考资料**
# > * [Which hashing algorithm is best for uniqueness and speed?](https://softwareengineering.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed)
#
# 在我们的课程中，为了处理大规模数据，我们将使用 **MurmurHash**（具体使用 `mmh3` 库）。
#
# 下面是一个简单的调用示例：

# %%
import mmh3

# 计算字符串 "hello" 的哈希值
h = mmh3.hash("hello")
print(f"Hash value of 'hello': {h}")

# %% [markdown]
# ### 3.2 精确去重 (Exact Deduplication)
#
# 精确去重是最简单的去重形式，其核心思想是：只有当两个文本完全一模一样（字节级匹配）时，才认为它们是重复的。
#
# #### 简单示例 (A Simple Example)
# 让我们定义一个简单的去重任务：
# 1.  **处理单元 (Item)**：整个字符串。
# 2.  **匹配策略 (Match)**：精确匹配 (Exact match)。
# 3.  **处理动作 (Action)**：每组重复项中只保留一个，删除其余的。
#
# 我们在 Python 中通过 `itertools.groupby` 来模拟这个过程。这种“排序 (Sort) -> 分组 (Group) -> 归约 (Reduce)”的逻辑正是 MapReduce 范式的基础，因此这种方法非常容易在分布式系统（如 Spark, Hadoop）上并行化和扩展。

# %%
import mmh3
import itertools

# 模拟原始数据，包含重复的 'hello'
items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]
print(f"Original items: {items}")

# 1. 计算哈希并排序
# 注意：itertools.groupby 要求输入是已排序的，否则无法正确分组
# 这里我们根据 item 的哈希值进行排序
sorted_items = sorted(items, key=mmh3.hash)

# 2. 分组
# key=mmh3.hash 意味着哈希值相同的会被分到同一组
hash_items = itertools.groupby(sorted_items, key=mmh3.hash)

# 3. 去重 (保留每组的第一个)
deduped_items = [next(group) for h, group in hash_items]

print(f"Deduped items:  {deduped_items}")

# %% [markdown]
# #### 优缺点分析
# * **优点 (Pro)**：简单、语义清晰、**高精度 (High precision)**（绝不会误删不同的内容）。
# * **缺点 (Con)**：无法处理**近似重复 (Near duplicates)**。例如 "hello" 和 "hello "（多一个空格）会被视为两个完全不同的项。
#
#
#
# #### 工业界案例：C4 数据集
# Google 的 T5 模型所使用的 **C4 (Colossal Clean Crawled Corpus)** 数据集就使用了精确去重策略，但它的处理粒度更细。
#
# > **参考资料**
# > * [C4 / T5 Paper: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683v4)
#
# **C4 的去重设计：**
# 1.  **处理单元**：**3句话的跨度 (3-sentence spans)**。它不是对整篇文章去重，而是滑动窗口式的。
# 2.  **匹配策略**：精确匹配。
# 3.  **处理动作**：如果发现某个 3 句的跨度在数据集中出现过（比如引用了相同的样板文字），就将这部分删除。
#
# **警告 (Caveat)**：
# 这种做法有一个明显的副作用。如果从文档中间删除了一个重复的 3 句跨度，剩下的文档可能会变得**不连贯 (Incoherent)**。原本流畅的段落中间可能会突然出现断层，这对训练语言模型的连贯性生成能力是有害的。

# %% [markdown]
# ### 3.3 布隆过滤器 (Bloom Filter)
#
# 当我们需要处理的数据量极大（例如数十亿个 URL），且内存资源有限时，存储所有数据的哈希值可能都变得昂贵。这时，我们需要一种**高效的、近似的**数据结构来进行成员资格测试（Set Membership Test）。
#
# **布隆过滤器 (Bloom Filter)** 就是这样一个完美的工具。
#
# #### 核心特性
# * **内存极度高效 (Memory efficient)**：不需要存储原始数据，只存储位数组（Bit Array）。
# * **支持更新**：可以随时添加新元素。
# * **不支持删除**：通常情况下，标准布隆过滤器不支持删除操作（因为无法确定某一位是否由该元素独占）。
# * **零假阴性 (No False Negatives)**：如果它返回 "No"，那么该元素**绝对**不在集合中。
# * **存在假阳性 (False Positives)**：如果它返回 "Yes"，该元素**很有可能**在集合中，但有极小的概率其实不在（误报）。
# * **可控误差**：通过增加哈希函数的数量或位数组的大小，可以将假阳性率以指数级速度降低。
#
#
# #### 尝试 1：朴素的哈希表 (Naive Approach)
#
# 首先，我们尝试构建一个极其简单的版本：只使用 **1 个哈希函数** 和一个非常小的 **位数组 (Bins)**。
#
# 我们要测试的目标是：
# 1.  构建一个包含 `items` 的表。
# 2.  查询 `non_items`（这些词不在原集合中）。
# 3.  统计有多少 `non_items` 被错误地判断为“存在”（假阳性）。
#
# 为了让代码可运行，我们需要先定义构建和查询的辅助函数：

# %%
import mmh3
import math

# 定义测试数据
items = ["the", "cat", "in", "the", "hat"]
non_items = ["what", "who", "why", "when", "where", "which", "how"]

# 辅助函数：构建表 (1个哈希函数)
def build_table(items, m):
    table = [0] * m
    for item in items:
        # 使用 murmurhash3，取模 m
        h = mmh3.hash(item) % m
        table[h] = 1
    return table

# 辅助函数：查询表 (1个哈希函数)
def query_table(table, item, m):
    h = mmh3.hash(item) % m
    return table[h]

# 辅助函数：统计 True 的个数
def count(results, target=1):
    return sum(1 for x in results if x == target)

# %% [markdown]
# 现在我们设置一个很小的 $m=8$，看看会发生什么：

# %%
m = 8  # 只有 8 个桶 (Bit slots)

# 1. 构建
table = build_table(items, m)

# 2. 验证：所有在集合中的元素都应该查得到
for item in items:
    assert query_table(table, item, m) == 1

# 3. 测试不在集合中的元素 (False Positive Test)
result = {item: query_table(table, item, m) for item in non_items}
print(f"Query Results for non-items: {result}")

# 4. 计算错误率
num_mistakes = count(result.values(), 1)
# 这里计算的是：在所有被判定为“存在”的请求中，有多少是错的（或者简单的错误计数）
# 注：为了演示，这里使用简单的比率
total_queries = len(items) + num_mistakes # 分母根据具体定义可能不同，此处沿用原逻辑
false_positive_rate = num_mistakes / (len(items) + num_mistakes)

print(f"Number of mistakes (False Positives): {num_mistakes}")
print(f"False Positive Rate: {false_positive_rate:.2f}")

# %% [markdown]
# #### 问题分析
# 在 $m$ 很小且只有 1 个哈希函数的情况下，**哈希冲突 (Collision)** 会非常频繁。
# * **朴素方案**：如果只想降低冲突，我们通常需要线性增加 $m$ 的大小。误差概率随内存大小多项式级下降 $O(1/m)$。
# * **更好方案**：引入多个哈希函数 $k$。
#
#
#
# #### 尝试 2：引入 K 个哈希函数 (The Power of K)
#
# 布隆过滤器的核心魔法在于使用 **$k$ 个独立的哈希函数**。
# * **写入时**：将一个元素通过 $k$ 个哈希函数映射到 $k$ 个位置，将这些位置全部置为 1。
# * **查询时**：只有当这 $k$ 个位置**全部**为 1 时，我们才认为该元素可能存在。
#
# 只要其中有一个位置是 0，该元素就绝对不存在。这种机制让误报率随计算量（$k$）呈**指数级下降**。
#
# 让我们实现支持 $k$ 个哈希函数的版本：

# %%
# 辅助函数：构建表 (k个哈希函数)
def build_table_k(items, m, k):
    table = [0] * m
    for item in items:
        for seed in range(k):
            # 使用 seed 来模拟不同的哈希函数
            h = mmh3.hash(item, seed=seed) % m
            table[h] = 1
    return table

# 辅助函数：查询表 (k个哈希函数)
def query_table_k(table, item, m, k):
    for seed in range(k):
        h = mmh3.hash(item, seed=seed) % m
        if table[h] == 0:
            return 0  # 只要有一位不是1，就肯定不存在
    return 1

# %% [markdown]
# 现在我们将哈希函数的数量增加到 $k=2$，保持 $m=8$ 不变：

# %%
k = 2  # 使用 2 个哈希函数
table = build_table_k(items, m, k)

# 1. 验证
for item in items:
    assert query_table_k(table, item, m, k) == 1

# 2. 测试
result = {item: query_table_k(table, item, m, k) for item in non_items}
print(f"Query Results for non-items (k={k}): {result}")

# 3. 计算错误率
num_mistakes = count(result.values(), 1)
false_positive_rate = num_mistakes / (len(items) + num_mistakes)

print(f"Number of mistakes: {num_mistakes}")
print(f"False Positive Rate: {false_positive_rate:.2f}")

# %% [markdown]
# **结论**：通过简单地增加哈希函数的数量，我们在不增加内存开销（$m$ 不变）的情况下，显著降低了假阳性率。这就是布隆过滤器在数据去重和缓存过滤中如此流行的原因。

# %% [markdown]
# ### 3.4 误报率的数学分析 (False Positive Rate Analysis)
#
# 我们在工程中不能只靠“感觉”来设定参数。为了设计一个高效的布隆过滤器，我们需要理解 $m$（桶的数量）、$n$（插入元素的数量）和 $k$（哈希函数的数量）是如何共同决定误报率的。
#
# > **参考资料**
# > * [Bloom filter - Wikipedia](https://en.wikipedia.org/wiki/Bloom_filter)
#
# #### 数学推导
# 假设哈希函数是完全独立的，且哈希结果均匀分布。
#
# **参数设定：**
# * $m$: 位数组的长度 (Number of bins)
# * $k$: 哈希函数的数量
# * $n$: 插入元素的数量
#
# 我们来一步步推导一个不在集合中的元素被误报为“存在”的概率。
#
# 1.  **单次插入的概率**：
#     当我们向长度为 $m$ 的数组中插入 1 个元素，使用 1 个哈希函数时，某个特定的位被置为 1 的概率是 $\frac{1}{m}$。
#     反之，该位**保持为 0** 的概率是：
#     $$P(\text{bit}_i = 0) = 1 - \frac{1}{m}$$
#
# 2.  **$k$ 次哈希后的概率**：
#     如果我们使用 $k$ 个哈希函数处理这 1 个元素，该位仍然保持为 0 的概率是：
#     $$P(\text{bit}_i = 0 | 1 \text{ item}) = \left(1 - \frac{1}{m}\right)^k$$
#
# 3.  **$n$ 个元素插入后的概率**：
#     当我们插入全部 $n$ 个元素后，该位仍然保持为 0 的概率是：
#     $$P(\text{bit}_i = 0 | n \text{ items}) = \left(1 - \frac{1}{m}\right)^{kn}$$
#
#     因此，该位**变为 1** 的概率 $p$ 为：
#     $$p = P(\text{bit}_i = 1) = 1 - \left(1 - \frac{1}{m}\right)^{kn}$$
#
# 4.  **误报率 (False Positive Rate, FPR)**：
#     现在我们查询一个**不在**集合中的新元素。为了发生误报，该元素的 $k$ 个哈希值对应的位置必须**全部**都已经被置为 1。
#     $$FPR = p^k = \left( 1 - \left(1 - \frac{1}{m}\right)^{kn} \right)^k$$
#
# 让我们用 Python 代码来验证这个计算过程，并寻找最优解：

# %%
import math

# 示例参数
m = 1000   # Bins 数量
k = 10     # 哈希函数数量
n = 100    # 插入元素数量

# 1. 某个特定位在 1 次哈希操作后被置为 1 的概率
p_1_hash = 1 / m 

# 2. 某个特定位在处理完 1 个元素(k次哈希)后被置为 1 的概率
# 这里我们计算它保持为 0 的概率然后取反
p_1_item = 1 - (1 - 1 / m) ** k

# 3. 某个特定位在处理完 n 个元素后被置为 1 的概率 (这也是数组中 1 的密度)
p_n_items = 1 - (1 - 1 / m) ** (k * n)

# 4. 误报率 (FPR): 查询一个新元素时，其 k 个哈希位置全部碰巧是 1 的概率
fpr = p_n_items ** k

print(f"对于 m={m}, n={n}, k={k}:")
print(f"数组饱和度 (Density): {p_n_items:.4f}")
print(f"理论误报率 (FPR): {fpr:.10f}")

# %% [markdown]
# #### 最优的 K 值 (Optimal K)
# 这就引出了一个核心的 Trade-off：
# * $k$ 太小：数组中 1 的密度低，但只要碰巧撞上几个 1 就会误报。
# * $k$ 太大：虽然每次查询要求更多匹配，但数组会被迅速填满（全是 1），导致误报率上升，且计算成本增加。
#
# 数学上可以证明，当数组中 0 和 1 的比例各占一半（即 $p \approx 0.5$）时，效率最高。
# 此时最优的 $k$ 值为：
# $$k = \ln(2) \cdot \frac{m}{n} \approx 0.7 \cdot \frac{m}{n}$$
#
# 在最优 $k$ 下，误报率简化为：
# $$FPR \approx (0.5)^k = (0.6185)^{m/n}$$
#
# **应用案例：Dolma 数据集**
# Dolma 在处理段落级去重时，为了保证极高的精度，将误报率设定为 $10^{-15}$。这是一个极高的标准，意味着几乎不可能误删数据。

# %%
# 计算最优 k
optimal_k = math.log(2) * (m / n)

# 计算在最优 k 下的理论误报率
optimal_fpr = 0.5 ** optimal_k

print(f"最优 k 值: {optimal_k:.2f}")
print(f"最优参数下的 FPR: {optimal_fpr:.10f}")

# %% [markdown]
#
#
# ### 3.5 工业级实现：使用 Bitarray (Implementation Details)
#
# 在前面的示例中，我们使用了 Python 的 `list` 来模拟位数组。但在实际生产环境中，`list` 的内存开销太大（一个整数至少占用 28 字节）。
#
# 为了达到极致的内存效率，我们应该使用 `bitarray` 库，它能真正以 **bit** 为单位存储数据。
#
# > **注意**：运行以下代码可能需要安装库：`pip install bitarray`
#
# #### 构建表 (Build Table)
# 我们将之前的逻辑封装为更通用的函数。

# %%
try:
    from bitarray import bitarray
except ImportError:
    # Fallback if bitarray is not installed, though memory usage will be higher
    print("bitarray not installed, using list mock.")
    def bitarray(n): return [0] * n

def build_bloom_filter(items, num_bins, k, seeds):
    """
    构建布隆过滤器
    items: 数据列表
    num_bins: 位数组大小 (m)
    k: 哈希次数
    seeds: 随机种子列表，用于模拟 k 个不同的哈希函数
    """
    # 初始化位数组，全部置 0
    table = bitarray(num_bins)
    if isinstance(table, list): # mock list logic
        for i in range(num_bins): table[i] = 0
    else:
        table.setall(0)
        
    for item in items:
        for i in range(k):
            # 使用第 i 个种子进行哈希
            seed = seeds[i]
            h = mmh3.hash(item, seed=seed) % num_bins
            table[h] = 1
            
    return table

# %% [markdown]
# #### 查询表 (Query Table)
# 查询逻辑同样需要遍历 $k$ 次哈希。使用 `all()` 函数可以简洁地表达“必须全部为 1”的逻辑。

# %%
def query_bloom_filter(table, item, num_bins, k, seeds):
    """
    查询布隆过滤器
    返回: 1 (可能存在), 0 (绝对不存在)
    """
    # 检查所有 k 个哈希位置是否都为 1
    # 如果有一个是 0，all() 返回 False，int(False) 为 0
    return int(all(
        table[mmh3.hash(item, seed=seeds[i]) % num_bins] 
        for i in range(k)
    ))

# %% [markdown]
#
#
# ## 4. 近似去重与 MinHash (Approximate Deduplication)
#
#
#
# 到目前为止，我们解决的都是“这个字符串是否完全出现过”的问题（Exact Deduplication）。但在 Web 数据中，更常见的问题是：
#
# * “这篇文章是否只是修改了几个词？”
# * “这段代码是否只是改变了变量名？”
#
# 这时，我们需要**近似成员资格 (Approximate Set Membership)** 检测。为了做到这一点，我们首先需要定义什么是“相似”。这就引出了 **Jaccard 相似度** 和 **MinHash** 算法。

# %% [markdown]
# ### 4.1 Jaccard 相似度 (Jaccard Similarity)
#
# 在处理“近似重复”时，我们首先需要定义什么是“相似”。对于文本数据，最常用的度量标准是 **Jaccard 相似度**。
#
# 我们将文档视为 Token（或 n-gram）的集合。
# * **定义**：两个集合 $A$ 和 $B$ 的 Jaccard 相似度是它们交集的大小除以并集的大小。
#     $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
#
# 让我们用 Python 直观地计算一下：

# %%
# 定义两个集合，它们大部分元素相同，只有 '4' 和 '5' 不同
A = {"1", "2", "3", "4"}
B = {"1", "2", "3", "5"}

def compute_jaccard(A, B):
    intersection = len(A & B)
    union = len(A | B)
    return intersection / union

jaccard = compute_jaccard(A, B)
print(f"Jaccard Similarity: {jaccard}")
# 解释: 交集是 {1,2,3} (3个), 并集是 {1,2,3,4,5} (5个), 结果应为 3/5 = 0.6

# %% [markdown]
# **近似重复的定义**：
# 在工程上，我们通常定义：如果两个文档的 Jaccard 相似度大于某个阈值（Threshold，例如 0.8），则视它们为**近似重复 (Near Duplicates)**。
#
# **算法挑战**：
# 如果我们要对十亿级别的网页进行去重，计算任意两篇文档的 Jaccard 相似度需要 $O(N^2)$ 的复杂度，这是不可接受的。我们需要一种线性时间的算法。
#
#
#
# ### 4.2 MinHash：将相似度转换为哈希碰撞
#
# 为了解决计算效率问题，我们引入 **MinHash** 算法。
# 通常情况下，哈希函数的设计目标是避免冲突（Collision）。但在 MinHash 中，我们的目标截然相反：我们设计一个随机哈希函数 $h$，使得两个集合发生哈希冲突的概率**恰好等于**它们的 Jaccard 相似度。
#
# $$\text{Pr}[h(A) = h(B)] = J(A, B)$$
#
# #### 核心实现
# MinHash 的计算非常简单：
# 1.  对集合中的每个元素进行哈希。
# 2.  取这些哈希值中的**最小值**作为该集合的签名。

# %%
import mmh3

def minhash(S: set, seed: int):
    """
    计算集合 S 的 MinHash 值。
    seed 用于模拟不同的随机哈希函数。
    """
    # 对集合中所有元素哈希，取最小值
    return min(mmh3.hash(x, seed) for x in S)

# %% [markdown]
# #### 为什么 MinHash 有效？(原理直觉)
# 我们可以通过**特征矩阵 (Characteristic Matrix)** 和**随机置换 (Permutation)** 来理解。
# [Image of MinHash characteristic matrix permutation]
#
# 1.  **随机置换**：一个哈希函数本质上定义了所有可能元素的一个随机排序（Permutation）。
# 2.  **首个元素**：MinHash 取最小值，等价于问：“在这个随机排序中，哪个元素最先出现？”
# 3.  **概率分析**：
#     * 对于集合 $A$ 和 $B$，所有元素的并集 $A \cup B$ 中的任何一个元素都有均等的概率在随机排序中排在第一位。
#     * 如果排在第一位的元素属于交集 $A \cap B$（即图例中的 1, 2, 3），那么 $A$ 和 $B$ 在这个位置都会有相同的值，即 $\min(A) = \min(B)$。
#     * 如果排在第一位的元素属于差集（即图例中的 4, 5），那么其中一个集合会包含它，另一个不会，导致 $\min(A) \neq \min(B)$。
#
# 因此，$\min(A) = \min(B)$ 的概率就是交集大小占并集大小的比例，即 Jaccard 相似度。
#
# #### 验证 MinHash
# 让我们生成 100 个随机哈希函数，看看统计出来的碰撞率是否接近真实的 Jaccard 相似度。

# %%
# 辅助函数：统计 True 的个数
def count(results, target=True):
    return sum(1 for x in results if x == target)

n = 100  # 生成 100 个随机哈希函数
# 检查这 100 次 MinHash 中，A 和 B 有多少次是一样的
matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]

# 估算的 Jaccard 相似度 = 匹配次数 / 总次数
estimated_jaccard = count(matches, True) / len(matches)

print(f"Estimated Jaccard: {estimated_jaccard}")
print(f"Actual Jaccard:    {jaccard}")

# 验证误差是否在允许范围内
assert abs(estimated_jaccard - jaccard) < 0.2 # 放宽一点范围因为 n=100 样本较小

# %% [markdown]
# 虽然我们可以哈希文档了，但仅仅一次碰撞并不能确切告诉我们 $J(A, B) > \text{threshold}$。我们需要更强的判别力。
#
#
#
# ### 4.3 局部敏感哈希 (LSH)
#
# MinHash 解决了“将相似度转化为概率”的问题，但它还是太随机了。
# * 如果我们只用 1 个 MinHash 函数：碰撞概率等于相似度。如果相似度是 0.8，我们有 20% 的概率漏掉它。
# * 如果我们平均使用 $n$ 个 MinHash：虽然估计准了，但我们需要两两比较 $n$ 个哈希值，计算量依然很大。
#
# **目标**：我们希望构造一种结构，使得：
# * 如果 $J(A, B)$ 很高（例如 > 0.8），它们**几乎一定**碰撞。
# * 如果 $J(A, B)$ 很低（例如 < 0.2），它们**几乎不**碰撞。
#
# 我们需要**锐化 (Sharpen)** 概率曲线。
#
# #### 解决方案：分段与行 (Bands and Rows)
# 我们将 $n$ 个 MinHash 函数分成 $b$ 个**条带 (Bands)**，每个条带包含 $r$ 个**行 (Rows)** 的哈希函数。
# $$n = b \times r$$
#
#
# **LSH 的碰撞规则**：
# 如果 $A$ 和 $B$ 在**任意一个 (Some)** 条带中的**所有 (All)** $r$ 个哈希值都完全一致，我们就认为 $A$ 和 $B$ 是候选的近似重复对。
#
# 这构成了 **AND-OR** 的逻辑结构：
# 1.  **内部 (AND)**：在一个条带内，必须 $r$ 个哈希值都匹配。这降低了低相似度文档碰撞的概率。
# 2.  **外部 (OR)**：只要有 $b$ 个条带中的任意一个满足条件即可。这提高了高相似度文档被检出的概率。
#
# 让我们设定参数并计算概率：

# %%
n = 12      # 总共 12 个哈希函数
b = 3       # 分成 3 个条带 (Bands)
r = 4       # 每个条带 4 个哈希函数 (Rows)
# 满足 n = b * r

# %% [markdown]
# #### 概率锐化公式
# 给定 Jaccard 相似度 $s$ (sim)，$A$ 和 $B$ 被判定为碰撞（Candidate）的概率是多少？
#
# 1.  在一个特定的条带内，所有 $r$ 个哈希值都匹配的概率：$s^r$
# 2.  在一个特定的条带内，不匹配（至少有一个不同）的概率：$1 - s^r$
# 3.  在所有 $b$ 个条带中，全都不匹配的概率：$(1 - s^r)^b$
# 4.  **最终碰撞概率**（至少有一个条带匹配）：
#     $$P(\text{collision}) = 1 - (1 - s^r)^b$$
#
# 这个函数会呈现出优美的 **S 形曲线 (S-Curve)**。

# %%
def get_prob_collision(sim, b, r):
    """
    计算 LSH 碰撞概率
    sim: Jaccard 相似度
    b: bands 数量
    r: rows 数量
    """
    # 1. 某个特定 band 内所有 r 个 hash 都 match 的概率
    prob_match = sim ** r
    
    # 2. 至少有一个 band match 的概率 (1 减去 所有 band 都不 match 的概率)
    prob_collision = 1 - (1 - prob_match) ** b
    
    return prob_collision

# %% [markdown]
# ### 4.4 LSH 参数调优与直觉 (Parameter Tuning & Intuition)
#
# 理解 LSH 的关键在于理解 $b$ (条带数) 和 $r$ (每条带的哈希数) 如何改变碰撞概率曲线。我们的目标是构建一个 **S 形曲线 (S-Curve)**，在阈值以下概率极低，在阈值以上概率极高。
#
# 首先，让我们代入一组具体数值计算一下。假设 Jaccard 相似度为 0.8，我们要计算在 $b=5, r=10$ 时的碰撞概率：

# %%
# 使用上一节定义的函数
# def get_prob_collision(sim, b, r): ...

prob_collision = get_prob_collision(sim=0.8, b=5, r=10)
print(f"Similarity=0.8, b=5, r=10 -> Collision Probability: {prob_collision:.4f}")

# %% [markdown]
# <div align="center">
#   <img src="https://cdn.sanity.io/images/vr8gru94/production/b470799575b8e77911bacb8500977afef06d6c85-1280x720.png" width="600" />
#   <p>图 4.1：典型的 LSH 概率 S 形曲线</p>
# </div>
#
# #### 参数敏感性分析
# 为了看清 S 曲线的移动规律，我们定义一组相似度点 `sims`，并观察改变 $b$ 和 $r$ 时概率的变化。
#
# **1. 基准情况 ($b=10, r=10$)**

# %%
sims = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
probs_base = {sim: get_prob_collision(sim=sim, b=10, r=10) for sim in sims}

print("Base Case (b=10, r=10):")
for s, p in probs_base.items():
    print(f"  Sim: {s:.2f} -> Prob: {p:.4f}")

# %% [markdown]
# **2. 增加 r (Rows): 曲线右移 -> 更严格**
# 如果我们保持 $b$ 不变，增加 $r$ 到 20：
# * **直觉**：每个条带内需要连续匹配的哈希值变多了 (AND 逻辑变强)，这使得碰撞变得更难。
# * **结果**：曲线向右移动，意味着我们需要更高的相似度才能触发碰撞。

# %%
probs_high_r = {sim: get_prob_collision(sim=sim, b=10, r=20) for sim in sims}

print("\nHarder to match (Increased r=20):")
for s, p in probs_high_r.items():
    print(f"  Sim: {s:.2f} -> Prob: {p:.4f}")

# %% [markdown]
# **3. 增加 b (Bands): 曲线左移 -> 更宽松**
# 如果我们保持 $r$ 不变 (在 r=20 的基础上)，增加 $b$ 到 20：
# * **直觉**：我们有了更多的条带 (OR 逻辑变强)，也就是有更多次机会去“撞大运”。
# * **结果**：曲线向左移动，即使相似度稍低，也更容易被检出。

# %%
probs_high_b = {sim: get_prob_collision(sim=sim, b=20, r=20) for sim in sims}

print("\nEasier to match (Increased b=20):")
for s, p in probs_high_b.items():
    print(f"  Sim: {s:.2f} -> Prob: {p:.4f}")

# %% [markdown]
# <div align="center">
#   <img src="https://cdn.sanity.io/images/vr8gru94/production/aace49fa240778e8ecf6e85ad08a2de7f5385566-1280x720.png" width="600" />
#   <p>图 4.2：不同参数下 LSH 曲线的陡峭程度与位置变化</p>
# </div>
#
#
#
# ### 4.5 寻找“阈值”：相变点在哪里？
#
# 在实际论文或工程（如 GPT-3 或 Deduplication Paper）中，我们经常看到巨大的参数设置，比如 $n=9000$。我们该如何快速估算这些参数对应的相似度阈值？
#
# > **参考案例**
# > [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/pdf/2107.06499)
# > 该论文使用了以下设置：
# > $b = 20, r = 450$ (总共 $n = 9000$ 个哈希函数)
#
# 我们可以通过数学推导找到 S 曲线发生“相变”的阈值点 $t$。
# 这个阈值通常定义为概率曲线开始急剧上升的拐点，近似满足：
# $$s^r \approx \frac{1}{b}$$
# 由此可得阈值近似公式：
# $$\text{threshold} \approx \left(\frac{1}{b}\right)^{\frac{1}{r}}$$
#
# 让我们计算一下论文中设置的理论阈值：

# %%
# 论文中的参数设置
b = 20
r = 450

# 计算近似阈值
threshold = (1 / b) ** (1 / r)
print(f"Theoretical Threshold: {threshold:.4f}")

# %% [markdown]
# **在这个阈值点上会发生什么？**
#
# 1.  **单个 Band 匹配的概率**：
#     根据我们的近似推导，$s^r$ 应该等于 $1/b$。

# %%
prob_match = (1 / b)
print(f"Prob of one band matching (s^r): {prob_match:.4f}")

# %% [markdown]
# 2.  **最终发生碰撞的概率**：
#     代入公式 $1 - (1 - \frac{1}{b})^b$。
#     当 $b$ 较大时，根据极限公式 $\lim_{x\to\infty} (1 - 1/x)^x = 1/e$，碰撞概率约为：
#     $$P \approx 1 - \frac{1}{e} \approx 0.632$$

# %%
prob_collision = 1 - (1 - 1 / b) ** b
print(f"Prob of collision at threshold: {prob_collision:.4f}")
print(f"Approximation (1 - 1/e): {1 - 1/2.71828:.4f}")

# %% [markdown]
# **结论**：
# 通过调整 $b$ 和 $r$，我们可以精确控制去重的力度。对于 $b=20, r=450$，我们设定的阈值约为 **0.935**。这意味着 Jaccard 相似度高于 0.935 的文档有很大概率被检测出来，而低于此值的则会被忽略。

# %% [markdown]
# ## 5. 课程总结与回顾 (Summary & Recap)
#
# 至此，我们已经完成了关于**数据处理与清洗**的核心课程。为了更好地消化这些内容，让我们将本节课（Mechanics）与上一节课（Datasets）的内容串联起来。
#
# ### 5.1 全局视野：从源头到成品
#
# 在**上一节课**中，我们宏观地综述了用于训练语言模型的数据集生态：
# * **数据流转 (Pipeline)**：
#     * **Live Service**: 实时服务产生数据（例如 GitHub 上的代码提交）。
#     * **Dump/Crawl**: 数据被抓取或导出存档（例如 GH Archive）。
#     * **Processed Data**: 经过清洗后的成品数据（例如 The Stack）。
# * **处理环节**:
#     从原始的 HTML/文本到最终的训练语料，中间经历了 HTML 提取、语言/质量/毒性过滤以及去重等关键步骤。
#
# 而在**本节课**中，我们将镜头拉近，深入探讨了支撑上述流程的**底层机制 (Mechanics)**：
# * **过滤算法**：如何区分好数据和坏数据。
# * **过滤应用**：如何将算法应用于特定的业务目标（语言、质量、安全性）。
# * **去重技术**：如何在海量数据规模下高效地通过哈希技术进行去重。
#
#
#
# ### 5.2 核心工具箱 (The Toolkit)
#
# 现在，你的工具箱里已经配备了以下几类强大的武器：
#
# 1.  **算法构建模块 (Algorithmic Tools)**
#     * **KenLM**: 基于 n-gram 的统计模型，通过困惑度（Perplexity）快速筛选通顺文本。
#     * **fastText**: 高效的线性分类器，用于快速判断文本类别。
#     * **DSIR**: 基于重要性重采样的方法，用于从原始分布中筛选出符合目标分布的数据。
#
# 2.  **过滤应用场景 (Applications)**
#     * **语言识别 (Language ID)**：使用 fastText 等工具区分 176+ 种语言。
#     * **质量过滤 (Quality)**：从启发式规则到基于模型的分类器（如 GPT-3, LLaMA 的分类器）。
#     * **毒性过滤 (Toxicity)**：使用分类器剔除有害、色情或仇恨言论。
#
# 3.  **大规模去重 (Deduplication)**
#     * **Bloom Filters**: 以极低的内存占用进行集合成员测试（概率型数据结构）。
#     * **MinHash**: 将集合相似度转化为哈希碰撞概率。
#     * **LSH (局部敏感哈希)**：通过 Band/Row 的设计，锐化碰撞概率曲线，实现线性时间复杂度的近似近邻搜索。
#
#
# ### 5.3 结语：从机制到直觉
#
# > **"Now you have the tools (mechanics), just have to spend time with data (intuitions)."**
#
# 你现在已经掌握了处理数据的**机械原理 (Mechanics)**。你知道了怎么算 Jaccard 相似度，怎么训练 fastText 分类器，怎么调 LSH 的参数。
#
# 但数据科学不仅仅是算法。下一步，你需要做的是**花时间与数据相处**。
# * 去观察那些被过滤掉的样本，看看分类器为什么会犯错。
# * 去阅读那些高困惑度的文本，理解数据的噪声分布。
# * 去调整去重的阈值，感受 Precision 和 Recall 的权衡。
#
# 只有将这些工具与对数据的敏锐**直觉 (Intuitions)** 结合，你才能真正构建出高质量的训练语料库。
