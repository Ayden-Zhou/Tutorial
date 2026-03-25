# %% [markdown]
# ## 1. 模型评估 (Evaluation)
#
# 在语言模型的研究与开发中，**评估 (Evaluation)** 的核心定义是：给定一个**固定的模型**，衡量它究竟有多“好”？
#
# 在深入具体指标之前，我们需要明确几个核心要点：
# * **没有唯一的标准**：不存在一种“放之四海而皆准”的评估方式。你需要根据想要衡量的目标（如推理能力、安全性、成本等）来选择合适的评估手段。
# * **回归实例**：不要只盯着总分，务必观察模型在具体样本上的表现和预测结果。
# * **多维度考量**：评估应涵盖能力（Capabilities）、安全性（Safety）、成本（Costs）和真实感（Realism）。
# * **明确规则**：在对比时，要清晰界定比较的对象是评估方法本身，还是具体的模型系统。

# %% [markdown]
# ### 1.1 评估呈现：我们能看到什么？
#
# 目前社区最直观的评估方式是 **Benchmark（基准测试）分数**。
#
# #### 基准测试成绩单 (Benchmark Scores)
#
# 现代语言模型（如 DeepSeek-R1, Llama4 等）通常会在一系列类似的基准测试上进行评估，例如 MMLU（大规模多任务语言理解）和 MATH（数学推理）。
#
# <div align="center">
#   <img src="../images/deepseek-r1-benchmarks.png" width="800" />
#   <p>DeepSeek R1 Benchmarks</p>
# </div>
#
# <div align="center">
#   <img src="../images/llama4-benchmarks.png" width="800" />
#   <p>Llama4 Benchmarks</p>
# </div>
#
# <div align="center">
#   <img src="https://www.datocms-assets.com/64837/1741887109-instruct-1.png" width="800" />
#   <p>1741887109 Instruct 1</p>
# </div>
#
# 面对这些数据，我们需要思考：这些基准测试具体测试了什么？这些数字背后的实际业务含义是什么？
#
# #### 能力榜单与成本权衡
#
# 除了单纯的分数，我们还需要参考权威的第三方排行榜，并重点关注**成本 (Costs)**。
#
# <div align="center">
#   <img src="../images/helm-capabilities-leaderboard.png" width="1000" />
#   <p>Helm Capabilities Leaderboard</p>
# </div>
#
# * **[[HELM capabilities]](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard)**：斯坦福大学推出的全方位模型评估框架。
#
# <div align="center">
#   <img src="../images/artificial-analysis.png" width="800" />
#   <p>Artificial Analysis</p>
# </div>
#
# * **[[Artificial Analysis]](https://artificialanalysis.ai/)**：提供了性能与价格比的深度分析。
#
# #### 市场与用户选择
#
# 另一种评估视角是：如果一个模型好用，人们自然会选择它并为其付费。
#
# <div align="center">
#   <img src="../images/openrouter.png" width="600" />
#   <p>OpenRouter Rankings</p>
# </div>
#
# * **[[OpenRouter]](https://openrouter.ai/rankings)**：通过实际 API 使用流量反映模型的受欢迎程度。
#
# <div align="center">
#   <img src="../images/chatbot-arena-leaderboard.png" width="800" />
#   <p>Chatbot Arena Leaderboard</p>
# </div>
#
# * **[[Chatbot Arena]](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)**：基于众测的“竞技场”，通过人类真实偏好进行 ELO 排名。

# %% [markdown]
# ### 1.2 “体感” (Vibes)
#
# 在硬性的指标之外，研究者们经常提到 “Vibes” —— 即模型给人的直观感觉。
#
# <div align="center">
#   <img src="../images/demis-gemini-2.5.png" width="500" />
#   <p>Demis Hassabis 关于 Gemini 2.5 的分享</p>
# </div>
#
# * 参考链接：[Demis Hassabis on X](https://x.com/demishassabis/status/1919779362980692364)
#
# #### 评估的困境
#
# 当前评估领域正面临某种程度的“危机”，当模型在公开基准测试上接近满分时，我们该如何定义下一步的进步？
#
# <div align="center">
#   <img src="../images/karpathy-crisis.png" width="600" />
#   <p>Andrej Karpathy 关于评估危机的见解</p>
# </div>

# %% [markdown]
# ## 2. 如何看待评估 (How To Think About Evaluation)
#
# 在开始写代码跑分之前，我们需要先矫正对“评估”的认知。
#
# 你可能会误以为评估只是一个机械的过程：拿一个现成的模型，扔进去一些 prompt，然后对输出结果算个平均分。**但实际上，评估是一个非常深刻且丰富的话题，它甚至在某种程度上决定了语言模型的未来走向。**
#
# ### 2.1 评估的目的
#
# 首先要明确的是，**不存在所谓的“唯一真理”式的评估（There is no one true evaluation）**。评估方式的选择完全取决于你试图回答什么问题。
#
# 通常，不同的角色有不同的评估目标：
#
# 1.  **用户或企业决策者**：为了特定的用例（例如客户服务聊天机器人），需要在模型 A 和模型 B 之间做出购买决策。
# 2.  **研究人员**：想要衡量模型的**原始能力（Raw Capabilities）**，比如通过测试来衡量“智能”。
# 3.  **商业与政策制定者**：我们需要理解模型带来的**收益与危害（Benefits + Harms）**。
# 4.  **模型开发者**：需要获取反馈以改进模型。
#
# 在上述每一种情况中，核心挑战在于如何将一个抽象的**目标 (Goal)** 转化为一个具体的、可执行的评估方案。
#
# ### 2.2 评估框架 (Framework)
#
# 为了系统地进行评估，我们可以构建一个包含四个维度的思考框架：
#
# #### 1. 输入是什么？ (What are the inputs?)
#
# 评估的数据集决定了评估的上限。你需要思考：
# * **覆盖率**：评估覆盖了哪些用例？
# * **长尾难度**：数据集中是否有代表性的、处于长尾分布的**困难输入（Difficult inputs）**？
# * **适配性**：输入数据是否针对模型进行了适配（例如，是否涵盖了多轮对话的情况）？
#
# #### 2. 如何调用模型？ (How do you call the language model?)
#
# 同一个模型，调用方式不同，表现可能天差地别。
# * **Prompting**：你如何提示（Prompt）模型？
# * **增强能力**：语言模型是否使用了思维链（Chain-of-Thought）、外部工具或 RAG（检索增强生成）等技术？
# * **评估对象**：我们到底是在评估**语言模型本身**，还是一个**代理系统（Agentic System）**？
#     * *注意：模型开发者通常关注前者，而终端用户通常关注后者。*
#
# #### 3. 如何评估输出？ (How do you evaluate the outputs?)
#
# 拿到模型的生成结果后，量化质量是一个复杂的工程。
# * **参考质量**：用于对比的参考答案（Reference outputs）本身是否是无误的？
# * **指标选择**：你使用什么指标？（例如代码生成中常用的 `pass@k`）。
# * **成本因素**：如何将成本（推理成本 + 训练成本）纳入考量？
# * **非对称错误**：如何处理**非对称错误（Asymmetric errors）**？
#     * *例子：在闲聊中说错话可能无伤大雅，但在医疗场景下的幻觉（Hallucinations）代价是巨大的。*
# * **开放式生成**：如何处理没有标准答案（Ground Truth）的开放式生成任务？
#
# #### 4. 如何解读结果？ (How to interpret the results?)
#
# 最后，面对跑分出来的数字，我们该如何理解？
# * **数字的含义**：如何解读一个具体的数字（例如 91% 的准确率）——这是否意味着模型已经准备好部署了？
# * **泛化能力**：面对训练集和测试集可能存在的**重叠（Data Contamination/Overlap）**，我们如何评估模型的真实泛化能力？
# * **归因**：我们评估的到底是最终得到的**模型（Model）**，还是产生这个模型的**方法（Method）**？
#
# **总结**：如你所见，在进行评估时，我们需要通过这一系列层层递进的问题来审视我们的方案，这远比单纯地“跑个分”要复杂得多。

# %% [markdown]
# ## 3. 困惑度 (Perplexity)
#
# 在评估语言模型时，我们首先要回顾一个基本概念：语言模型本质上是关于 Token 序列的概率分布 $p(x)$。
#
# **困惑度 (Perplexity, PPL)** 是衡量模型对某个数据集 $D$ 预测能力的经典指标。其数学定义如下：
#
# $$
# \text{Perplexity}(D) = P(D)^{-\frac{1}{|D|}} = \left( \frac{1}{P(x_1, \dots, x_N)} \right)^{\frac{1}{N}}
# $$
#
# 直观地说，它衡量的是模型是否给予了数据集 $D$ 较高的概率。
# * 在**预训练 (Pre-training)** 阶段，我们的优化目标就是最小化训练集上的困惑度。
# * 最直接的评估方式，自然是计算测试集（Test Set）上的困惑度。
#
#

# %% [markdown]
# ### 3.1 评估范式的演变
#
# #### 标准数据集与传统评估
# 在早期研究中，有一些标准的数据集被广泛使用：
# * **Penn Treebank (WSJ)**：华尔街日报语料。
# * **WikiText-103**：维基百科语料。
# * **One Billion Word Benchmark (1BW)**：源自机器翻译任务（WMT11），包含欧洲议会、联合国文档和新闻。
#
# 通常的做法是：在一组特定的数据集（训练集划分）上训练，然后在**同一个数据集**（测试集划分）上进行评估。
# * 例如：纯 CNNs+LSTMs 架构的模型在 One Billion Word Benchmark 上的进展（困惑度从 51.3 降至 30.0）。
#     * [相关论文 Link](https://arxiv.org/abs/1602.02410)
#
# #### GPT-2 带来的转变：分布外评估 (OOD)
# GPT-2 引入了一种新的评估范式。它在 **WebText**（约 40GB 文本，源自 Reddit 链接的网页）上进行训练，然后在上述标准数据集上进行 **零样本 (Zero-shot)** 评估。
#
# 这实际上是一种**分布外 (Out-of-distribution)** 评估，但其核心理念是：如果训练数据足够庞大且多样，它应该能覆盖各种领域。
#
# <div align="center">
#   <img src="../images/gpt2-perplexity.png" width="800" />
#   <p>GPT-2 Perplexity Results</p>
# </div>
#
# **观察结论**：
# * 这种迁移（Transfer）在小数据集上效果很好。
# * 但在非常大的特定数据集（如 1BW）上，效果不如专门在该数据集上训练的模型。

# %% [markdown]
# ### 3.2 为什么我们仍然需要困惑度？
#
# 自从 GPT-2 和 GPT-3 之后，语言模型的研究论文逐渐转向关注**下游任务的准确率 (Downstream Task Accuracy)**。但这并不意味着困惑度过时了。
#
# **困惑度的独特价值：**
# 1.  **更平滑 (Smoother)**：相比于任务准确率，困惑度曲线更平滑，非常适合用于拟合**缩放定律 (Scaling Laws)**。
# 2.  **普适性 (Universal)**：这是我们训练模型时使用的原生目标，而任务准确率可能会遗漏模型能力的一些细微之处。
# 3.  **条件困惑度 (Conditional Perplexity)**：我们也可以在下游任务上测量条件困惑度（这也常用于 Scaling Laws 的研究）。
#     * [相关论文 Link](https://arxiv.org/abs/2412.04403)
#
# #### 给“榜单”维护者的警告
# 如果你在运行一个模型排行榜（Leaderboard），需要注意：**评估器需要信任语言模型**。
# * **任务准确率**：我们可以把模型当成黑盒，只取其生成的文本输出，计算所需的指标。
# * **困惑度**：我们需要模型输出具体的概率值，并且必须信任这些概率之和为 1。这在过去（处理 `<UNK>` 未知词标记时）尤其令人头疼。
#
#
#
# ### 3.3 “困惑度最大主义者”视角 (The Perplexity Maximalist View)
#
# 有一种极端的观点认为，困惑度就是一切：
# * 假设真实的数据分布是 $t$，模型分布是 $p$。
# * 理论上可能的最佳困惑度是 $H(t)$（真实分布的熵），这仅在 $p = t$ 时通过。
# * 如果我们能完美地建模真实分布（即拥有了 $t$），那么我们理论上就能解决所有任务。
# * **结论**：只要不断降低困惑度，我们最终将实现通用人工智能 (AGI)。
#
# **局限性 (Caveat)**：这可能不是通往 AGI 最**高效**的路径。因为降低困惑度可能意味着模型花费大量容量去拟合数据中无关紧要的噪音部分，而不是核心逻辑。

# %% [markdown]
# ### 3.4 本质上属于“困惑度”的任务
#
# 有些任务虽然形式上不是计算 PPL，但其核心思想与困惑度高度相似（预测下一个词或填空）：
#
# #### 完形填空任务 (Cloze tasks) - LAMBADA
# 要求模型预测句子的最后一个词，测试其对长距离依赖的理解。
# * [LAMBADA Link](https://arxiv.org/abs/1606.06031)
#
# <div align="center">
#   <img src="../images/lambada.png" width="800" />
#   <p>Lambada Task</p>
# </div>
#
# #### HellaSwag
# 常识推理任务，选择最合理的句子结尾。
# * [HellaSwag Link](https://arxiv.org/pdf/1905.07830)
#
# <div align="center">
#   <img src="../images/hellaswag.png" width="600" />
#   <p>Hellaswag Task</p>
# </div>

# %% [markdown]
# ## 4. 知识类基准测试 (Knowledge Benchmarks)
#
# 评估语言模型的另一个重要维度是考察其**世界知识 (World Knowledge)** 的储备量。这类测试通常要求模型回答需要特定领域事实支持的问题。
#
# 以下是目前主流的几个知识类基准测试：
#
# ### 4.1 大规模多任务语言理解 (MMLU)
#
# **Massive Multitask Language Understanding (MMLU)** 是目前最知名的综合性知识评估数据集之一。
#
# * **学科覆盖**：涵盖 57 个学科（例如数学、美国历史、法律、道德等），均为多项选择题。
# * **数据来源**：题目由研究生和本科生从互联网上的免费公开资源中收集整理。
# * **核心实质**：MMLU 实质上是在测试模型的**知识储备**，而非单纯的语言理解能力。
# * **评估方法**：最初是在 GPT-3 上使用少样本提示 (Few-shot prompting) 进行评估的。
#
# <div align="center">
#   <img src="../images/mmlu.png" width="800" />
#   <p>MMLU 示例</p>
# </div>
#
# * [[HELM MMLU 可视化预测结果]](https://crfm.stanford.edu/helm/mmlu/latest/)
#
#
#
# ### 4.2 MMLU-Pro
#
# 随着模型能力的提升，原版 MMLU 的区分度逐渐降低，因此推出了更难的版本 **MMLU-Pro**。
#
# * [相关论文 Link](https://arxiv.org/abs/2406.01574)
# * **去噪**：移除了原 MMLU 中含有噪声或过于简单的问题。
# * **增加难度**：将选项从 4 个扩展到了 10 个，大幅降低了随机猜测的成功率。
# * **评估方式**：使用思维链 (Chain of Thought, CoT) 进行评估，给予模型更多的推理机会。
# * **结果分析**：在这一基准下，模型的准确率普遍下降了 16% 到 33%，这表明该测试尚未饱和，更能区分顶尖模型的差异。
#
# <div align="center">
#   <img src="../images/mmlu-pro.png" width="800" />
#   <p>MMLU-Pro 示例</p>
# </div>
#
# * [[HELM MMLU-Pro 可视化预测结果]](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro)
#
#
# ### 4.3 研究生级防谷歌搜索问答 (GPQA)
#
# **Graduate-Level Google-Proof Q&A (GPQA)** 旨在挑战模型在高度专业化领域的推理能力。
#
# * [相关论文 Link](https://arxiv.org/abs/2311.12022)
# * **作者团队**：题目由来自 Upwork 的 61 位博士级专家编写。
#
# <div align="center">
#   <img src="../images/gpqa.png" width="800" />
#   <p>GPQA 难度对比</p>
# </div>
#
# * **性能对比**：
#     * **领域博士专家**：准确率约为 65%。
#     * **非专家（允许使用 Google 30分钟）**：准确率仅为 34%。这意味着即使有人类智慧加持和搜索引擎，普通人也难以回答这些问题。
#     * **GPT-4**：准确率约为 39%。
# * [[HELM GPQA 可视化预测结果]](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa)
#
#
#
# ### 4.4 人类最后的考试 (Humanity's Last Exam)
#
# 这是一个极具挑战性的新基准测试，旨在设定这一代 AI 的天花板。
#
# * [相关论文 Link](https://arxiv.org/abs/2501.14249)
# * **规模与形式**：包含 2500 个问题，涵盖多模态、多学科，题型包括多项选择和简答题。
#
# <div align="center">
#   <img src="../images/hle-examples.png" width="800" />
#   <p>Humanity's Last Exam 示例</p>
# </div>
#
# * **众包激励**：为了保证题目质量，项目组提供了 50 万美元的奖金池，并给予题目创作者论文共同作者 (co-authorship) 的身份。
# * **筛选流程**：所有题目都经过了前沿 LLM 的“过滤”（如果当前最好的模型能做对，该题就会被剔除），并经过多轮人工审查。
#
# <div align="center">
#   <img src="../images/hle-pipeline.png" width="800" />
#   <p>HLE 筛选流程</p>
# </div>
#
# <div align="center">
#   <img src="../images/hle-results.png" width="800" />
#   <p>HLE 评估结果</p>
# </div>
#
# * [[最新排行榜]](https://agi.safe.ai/)

# %% [markdown]
# ## 5. 指令遵循基准测试 (Instruction Following Benchmarks)
#
# 到目前为止，我们评估的主要是结构化程度较高的任务（如多项选择题）。
# 然而，随着 ChatGPT 的普及，**指令遵循 (Instruction Following)** 成为了评估的重点：即模型能否仅仅根据用户的指令完成任务。
#
# 这里的核心挑战在于：**如何评估开放式的回答 (Open-ended response)？** 当没有唯一的标准答案时，我们该如何量化模型的好坏？
#
# ### 5.1 Chatbot Arena (聊天机器人竞技场)
#
# **Chatbot Arena** 采用了众包和 ELO 等级分机制，是目前最接近“真实人类偏好”的排行榜。
#
# * [相关论文 Link](https://arxiv.org/abs/2403.04132)
#
# #### 运作机制
# 1.  **随机用户输入**：互联网上的随机用户输入 Prompt。
# 2.  **盲测**：用户同时收到两个随机（匿名）模型的回答。
# 3.  **人类投票**：用户根据回答质量判断哪个更好（Win/Tie/Loss）。
# 4.  **ELO 评分**：基于成对比较的结果计算 ELO 分数。
#
# **特点**：它是**动态 (Live)** 的而非静态的数据集，并且能够随时容纳新发布的模型。
#
# <div align="center">
#   <img src="../images/chatbot-arena-leaderboard.png" width="800" />
#   <p>Chatbot Arena Leaderboard</p>
# </div>
#
# * [[Chatbot Arena 排行榜]](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)
#
#

# %% [markdown]
#
# ### 5.2 指令遵循评估 (IFEval)
#
# **IFEval** 试图通过通过“可验证的约束”来解决开放式评估的难题。
#
# * [相关论文 Link](https://arxiv.org/abs/2311.07911)
#
# <div align="center">
#   <img src="../images/ifeval-categories.png" width="600" />
#   <p>IFEval Categories</p>
# </div>
#
# #### 核心思想
# * **合成约束**：在指令中加入简单的合成约束（例如：“写一个故事，必须包含超过 400 个单词”或“回复必须全部大写”）。
# * **自动验证**：这些约束可以被程序自动验证。
# * **局限性**：它虽然能精确判断模型是否遵守了格式约束，但无法评估回复内容的**语义质量**。此外，这些指令通常较为简单，约束条件也略显人工化 (Artificial)。
#
# * [[HELM IFEval 可视化预测结果]](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/ifeval)

# %% [markdown]
# ### 5.3 AlpacaEval
#
# **AlpacaEval** 是一种基于“模型评估模型” (LLM-as-a-Judge) 的自动化评估框架。
#
# * [相关链接 Link](https://tatsu-lab.github.io/alpaca_eval/)
#
# #### 评估流程
# * **数据源**：包含来自不同来源的 805 条指令。
# * **裁判**：通常使用 GPT-4 Preview 作为裁判。
# * **指标**：计算模型回复相对于 GPT-4 Preview 的**胜率 (Win Rate)**。
# * **潜在偏差**：这种方法存在潜在的偏差（Bias），例如裁判模型可能更偏向于自己生成的风格或更冗长的回答。
#
# <div align="center">
#   <img src="../images/alpacaeval-leaderboard.png" width="600" />
#   <p>AlpacaEval Leaderboard</p>
# </div>

# %% [markdown]
# ### 5.4 WildBench
#
# **WildBench** 旨在提供比 AlpacaEval 更贴近真实场景且更严谨的自动化评估。
#
# * [相关论文 Link](https://arxiv.org/pdf/2406.04770)
#
# #### 核心特点
# * **数据来源**：从 100 万条真实的人类-聊天机器人对话中筛选出的 1024 个高质量样本。
# * **双重裁判机制**：
#     * 使用 GPT-4 Turbo 配合**检查清单 (Checklist)** 进行评分（类似于让裁判使用思维链 CoT 进行评判）。
#     * 同时也使用 GPT-4 作为裁判。
# * **与人类对齐**：WildBench 与 Chatbot Arena 的相关性高达 **0.95**。
#     * *注：Chatbot Arena 经常被视为评估其他 Benchmark 有效性的“事实标准” (Sanity check)。*
#
# <div align="center">
#   <img src="../images/wildbench.png" width="800" />
#   <p>WildBench Framework</p>
# </div>
#
# * [[HELM WildBench 可视化预测结果]](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench)

# %% [markdown]
# ## 6. 代理基准测试 (Agent Benchmarks)
#
# 在前面的章节中，我们主要评估的是模型在单次交互中的表现。而在 **Agent** 评估中，我们关注的是那些需要**使用工具**（例如运行代码、浏览网页）并且需要经过**一段时间的迭代**才能完成的任务。
#
# 我们可以给 Agent 下一个定义：
# $$\text{Agent} = \text{Language Model} + \text{Agent Scaffolding}$$
# 这里的 **Agent Scaffolding（代理脚手架）** 指的是决定如何调用和使用语言模型的逻辑框架。
#
# 以下是几个代表性的 Agent 基准测试：
#
# ### 6.1 SWE-bench (软件工程)
#
# **SWE-bench** 旨在评估大模型解决真实世界软件工程问题的能力。
#
# * [相关论文 Link](https://arxiv.org/abs/2310.06770)
# * **数据规模**：包含来自 12 个流行 Python 代码仓库的 2294 个任务。
# * **任务形式**：给定整个代码库 (Codebase) 和一个问题描述 (Issue description)，Agent 需要提交一个 Pull Request (PR) 来修复该 Issue。
# * **评估指标**：通过新增加的或现有的**单元测试 (Unit tests)** 来验证修复是否成功。这是非常硬核且客观的评估标准。
#
# <div align="center">
#   <img src="../images/swebench.png" width="800" />
#   <p>SWE-bench 概览</p>
# </div>
#
#
#
# ### 6.2 CyBench (网络安全)
#
# **CyBench** 是一个专注于网络安全领域的基准测试，用来衡量 Agent 在攻防场景下的能力。
#
# * [相关论文 Link](https://arxiv.org/abs/2408.08926)
# * **任务内容**：包含 40 个夺旗赛 (Capture the Flag, CTF) 任务。
# * **难度衡量**：使用“首次解决时间” (First-solve time) 作为衡量任务难度的标准。这意味着模型不仅要能解题，还要足够快。
#
# <div align="center">
#   <img src="../images/cybench.png" width="800" />
#   <p>CyBench 任务结构</p>
# </div>
#
# <div align="center">
#   <img src="../images/cybench-agent.png" width="800" />
#   <p>CyBench Agent 交互流程</p>
# </div>
#
# <div align="center">
#   <img src="../images/cybench-results.png" width="800" />
#   <p>CyBench 评估结果</p>
# </div>
#
#
#
# ### 6.3 MLE-bench (机器学习工程)
#
# **MLE-bench** 考察的是 Agent 是否具备像人类机器学习工程师那样的实战能力。
#
# * [相关论文 Link](https://arxiv.org/abs/2410.07095)
# * **任务来源**：精选了 75 个 **Kaggle 竞赛**。
# * **能力要求**：这要求 Agent 具备端到端的机器学习能力，包括数据预处理、模型训练、调参以及生成提交文件等。
#
# <div align="center">
#   <img src="../images/mlebench.png" width="800" />
#   <p>MLE-bench 框架</p>
# </div>
#
# <div align="center">
#   <img src="../images/mlebench-results.png" width="800" />
#   <p>MLE-bench 评估结果</p>
# </div>

# %% [markdown]
# ## 7. 纯推理基准测试 (Pure Reasoning Benchmarks)
#
# 回顾我们之前讨论的所有评估任务（从 MMLU 到 Agent），它们都有一个共同点：都需要依赖大量的**语言能力**和**世界知识**。
#
# 这引出了一个深刻的问题：**我们能否将“推理 (Reasoning)”从“知识 (Knowledge)”中剥离出来？**
#
# 很多研究者认为，推理能力捕捉到了一种比单纯记忆事实 (Memorizing facts) 更纯粹的智能形式。为了衡量这种“流体智力”，Francois Chollet 在 2019 年引入了 ARC (Abstraction and Reasoning Corpus) 挑战。
#
# ### 7.1 ARC-AGI 挑战
#
# * **[[ARC-AGI 官网]](https://arcprize.org/arc-agi)**
#
# ARC 的核心在于要求模型通过极少量的示例（Few-shot），观察并抽象出网格变换的规律，然后应用到新的测试用例上。这几乎不依赖任何外部文本知识。
#
# #### ARC-AGI-1
#
# 这是最初版本的挑战，旨在测试系统对核心先验知识（如对象持久性、对称性等）的掌握程度。
#
# <div align="center">
#   <img src="https://arcprize.org/media/images/arc-task-grids.jpg" width="800" />
#   <p>ARC 任务网格示例</p>
# </div>
#
# 目前的模型在这上面的进展（Leaderboard）：
#
# <div align="center">
#   <img src="https://arcprize.org/media/images/oseriesleaderboard.png" width="800" />
#   <p>o1-series Leaderboard</p>
# </div>
#
# #### ARC-AGI-2
#
# 随着模型能力的提升，更难的版本 ARC-AGI-2 被提出。
#
# <div align="center">
#   <img src="https://arcprize.org/media/images/blog/arc-agi-2-unsolved-1.png" width="800" />
#   <p>ARC-AGI-2 未解决难题示例</p>
# </div>

# %% [markdown]
# ## 8. 安全基准测试 (Safety Benchmarks)
#
# 当我们在讨论 AI 的“安全性”时，我们到底在讨论什么？这就好比汽车工业中的碰撞测试，我们需要一套标准来衡量模型在极端或恶意情况下的表现。
#
# <div align="center">
#   <img src="https://www.team-bhp.com/forum/attachments/road-safety/2173645d1625144681-will-crash-test-rating-change-if-higher-variant-chosen-images-30.jpeg" width="500" />
#   <p>汽车碰撞测试类比</p>
# </div>
#
# * **[[HELM Safety: 精选的安全基准测试集]](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard)**
#
#
#
# ### 8.1 HarmBench
#
# **HarmBench** 是一个专注于评估模型是否会产生有害行为的基准测试。
#
# * [相关论文 Link](https://arxiv.org/abs/2402.04249)
# * **内容基础**：基于 **510 种有害行为**，这些行为违反了法律或社会规范。
# * **评估资源**：
#     * [[HELM 上的 HarmBench 榜单]](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard/harm_bench)
#     * [[安全失败示例 (Safety failure example)]](https://crfm.stanford.edu/helm/safety/latest/#/runs/harm_bench:model=anthropic_claude-3-7-sonnet-20250219?instancesPage=4) —— *查看模型在特定诱导下如何突破防线。*
#
#
#
# ### 8.2 AIR-Bench
#
# **AIR-Bench** 侧重于从监管和政策合规的角度进行评估。
#
# * [相关论文 Link](https://arxiv.org/abs/2407.17436)
# * **设计依据**：基于监管框架（Regulatory frameworks）和公司政策。
# * **分类体系**：将风险细分为 **314 个类别**，共包含 5694 个 Prompt。这提供了一个非常细粒度的风险图谱。
#
#
#
# * [[HELM AIR-Bench 榜单]](https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard)
#
#
# ### 8.3 越狱攻击 (Jailbreaking)
#
# 语言模型通常经过训练（RLHF 等）来拒绝有害指令。然而，**越狱 (Jailbreaking)** 旨在绕过这些防御机制。
#
# * **自动优化攻击**：贪婪坐标梯度 (Greedy Coordinate Gradient, **GCG**) 是一种可以自动优化 Prompt 以绕过安全检查的技术。
#     * [相关论文 Link](https://arxiv.org/pdf/2307.15043)
# * **迁移性 (Transferability)**：研究发现，针对开源模型（如 Llama）生成的对抗性攻击，往往可以直接迁移攻击闭源模型（如 GPT-4）。这意味着开源模型的弱点可能成为整个生态系统的安全隐患。
#
# <div align="center">
#   <img src="../images/gcg-examples.png" width="800" />
#   <p>GCG 攻击示例</p>
# </div>
#
#
#
# ### 8.4 部署前测试 (Pre-deployment testing)
#
# 为了应对潜在风险，各国政府机构开始介入监管环节。
#
# * **机构合作**：美国安全研究所 (US Safety Institute) 与英国 AI 安全研究所 (UK AI Safety Institute) 正在开展合作。
# * **测试流程**：目前（主要是自愿性质的），AI 公司会在模型发布前赋予安全研究所访问权限。研究所运行评估并向公司生成一份报告。
#     * [[查看报告样本]](https://www.nist.gov/system/files/documents/2024/12/18/US_UK_AI%20Safety%20Institute_%20December_Publication-OpenAIo1.pdf)
#
#
# ### 8.5 什么是真正的“安全”？ (But what is safety?)
#
# 安全并不是一个非黑即白的各种指标，它具有高度的复杂性：
#
# #### 语境相关性 (Contextual)
# 安全的定义高度依赖于语境：政治、法律和社会规范在不同国家和地区截然不同。一个在某个文化中安全的回答，在另一个文化中可能是冒犯的。
#
# #### 拒绝 vs. 能力 (Refusal vs. Capability)
# 初看之下，人们可能认为“安全”就是“拒绝回答”，但这与“能力”是对立的。然而事实并非总是如此：
# * **幻觉 (Hallucinations)**：在医疗场景中，减少幻觉既提高了系统的**能力**，也提高了系统的**安全性**。二者在此处是统一的。
#
# #### 风险构成的两个维度
# 降低模型安全性通常涉及两个因素：**能力 (Capabilities)** + **倾向性 (Propensity)**。
# * 一个系统可能有能力做坏事，但它拒绝这样做（低倾向性）。
# * **对于 API 模型**：**倾向性**更重要。只要模型拒绝了恶意请求，它通常被视为安全的。
# * **对于开源权重模型**：**能力**更重要。因为如果模型具备某种危险能力（如制造生化武器的知识），即便它被微调为拒绝回答，攻击者也可以轻易地通过微调（Fine-tune）移除这些安全限制。
#
# #### 双重用途 (Dual-use)
# 这是一个棘手的领域。
# * 例如，一个在 **CyBench** 上表现出色的网络安全 Agent，既可以帮助修补系统漏洞（防御），也可以被用来入侵系统（攻击）。
# * **思考**：CyBench 被安全研究所用来进行安全评估，但本质上，它难道不是一个能力评估吗？**能力即风险**，这是高阶 AI 安全面临的核心悖论。

# %% [markdown]
# ## 9. 真实性 (Realism)
#
# 语言模型在实际生产环境中的使用量已经达到了惊人的规模：
#
# * **OpenAI**: 每天处理超过 1000 亿 (100B) 个 Token。
# * **Cursor**: 代码补全量已达到 10 亿 (1B) 行。
#
# <div align="center">
#   <img src="../images/openai-100b-tokens.png" width="600" />
#   <p>OpenAI: 100B Tokens/Day</p>
# </div>
#
# * [[Tweet source]](https://x.com/sama/status/1756089361609981993)
#
# <div align="center">
#   <img src="../images/cursor-1b-lines.png" width="600" />
#   <p>Cursor: 1B Lines</p>
# </div>
#
# * [[Tweet source]](https://x.com/amanrsanger/status/1916968123535880684)
#
# 然而，我们必须面对一个现实：**大多数现有的基准测试（如 MMLU）与真实世界的使用场景相去甚远。**
#
# 虽然来自真实用户的实时流量（Live traffic）最能反映真实情况，但其中也包含了大量的“垃圾”请求，这并不完全是我们想要优化的目标。为了更好地理解这种差异，我们可以将 Prompt 分为两类：
#
# 1.  **测验 (Quizzing)**：用户其实**知道**答案，他们只是在测试系统能否答对（类似于标准化考试）。目前的很多 Benchmark 属于这一类。
# 2.  **求知 (Asking)**：用户**不知道**答案，他们试图利用系统来获取信息或解决问题。
#
# 显然，**“求知 (Asking)”** 才是更具现实意义、更能为用户产生价值的场景。
#
#
#
# ### 9.1 Clio (Anthropic)
#
# 为了弥补这一差距，Anthropic 推出了 **Clio**，尝试利用语言模型来分析真实的用户数据。
#
# * [相关论文 Link](https://arxiv.org/abs/2412.13678)
#
# Clio 的核心思路是分析并分享人们实际上在问什么，从而提取出通用的模式，让评估更贴近真实需求。
#
# <div align="center">
#   <img src="../images/clio-table4.png" width="700" />
#   <p>Clio: 分析用户意图分布</p>
# </div>
#
#
# ### 9.2 MedHELM
#
# 医疗领域是真实性评估的一个典型案例。
#
# * [相关论文 Link](https://arxiv.org/abs/2412.13678)
#
# 以往的医疗基准测试主要基于标准化考试（如执业医师资格考试）。但 **MedHELM** 采取了不同的路径：
#
# * **数据来源**：不再是试卷，而是由 29 位临床医生提供的 121 个具体临床任务。
# * **数据构成**：混合了私有数据和公共数据集，涵盖了真实的临床决策过程。
#
#
# * [[MedHELM 榜单]](https://crfm.stanford.edu/helm/medhelm/latest/#/leaderboard)
#
# **总结**：这一方向虽然充满希望，但我们不得不面对一个矛盾：**真实性 (Realism)** 和 **隐私 (Privacy)** 往往是难以兼得的。越是真实的数据（如真实病历），隐私风险越高。

# %% [markdown]
# ## 10. 评估的有效性 (Validity)
#
# 即便我们选对了评估指标，设计了合理的任务，还有一个根本性的问题悬在头顶：**我们如何知道我们的评估是有效的？**
#
# ### 10.1 训练集与测试集的重叠 (Train-test overlap)
#
# 这是评估领域目前面临的最大危机之一——数据污染 (Data Contamination)。
#
# * **机器学习的第一课**：永远不要在测试集上训练模型。这是铁律。
# * **前大模型时代**：在 ImageNet 或 SQuAD 的时代，我们有定义非常清晰的训练集/测试集划分 (Train-test splits)。
# * **现在的情况**：大模型是在整个互联网上训练的，而且模型发布者通常**不告诉**公众他们具体用了哪些数据。这就导致我们根本不知道模型是不是已经“见过”了考题。
#
# 面对这个问题，目前主要有两条解决路径：
#
# #### 路径 1：尝试从模型端推断重叠 (Infer overlap from model)
#
# 我们可以通过技术手段来“侦测”模型是否见过测试数据。一种有效的方法是利用数据点的**可交换性 (Exchangeability)**。
# * 如果模型对测试数据的预测表现出异常的确定性或特定模式，可能暗示了数据泄露。
# * [相关论文 Link](https://arxiv.org/pdf/2310.17623)
#
# <div align="center">
#   <img src="../images/contamination-exchangeability.png" width="600" />
#   <p>利用可交换性检测数据污染</p>
# </div>
#
# #### 路径 2：建立报告规范 (Encourage reporting norms)
#
# 另一种路径是依靠社区规范和行业自律。
# * 模型提供商应该主动报告训练集与测试集的重叠情况（例如报告置信区间），而不是让用户去猜。
# * [相关论文 Link](https://arxiv.org/abs/2410.08385)
#
#
#
# ### 10.2 数据集质量 (Dataset quality)
#
# 除了模型“作弊”，另一个影响有效性的因素是**考题本身出错了**。现有的基准测试往往包含噪声、错误甚至无法完成的任务。
#
# #### 修正现有基准：SWE-Bench Verified
#
# OpenAI 针对 SWE-Bench 进行了清洗和修正，发布了 **SWE-Bench Verified**。
# * 他们发现原版中许多任务不仅对模型难，对人类来说也是定义不清的。
# * [OpenAI 官方介绍](https://openai.com/index/introducing-swe-bench-verified/)
# * [相关博客](https://openai.com/index/introducing-swe-bench-verified/)
#
# #### 创建“白金版”基准 (Platinum versions)
#
# 社区开始致力于打造更高质量的、经得起推敲的“白金版”基准测试，去除低质量数据。
# * [相关论文 Link](https://arxiv.org/abs/2502.03461)
#
# <div align="center">
#   <img src="https://pbs.twimg.com/media/GjICXQlWkAAYnDS?format=jpg&name=4096x4096" width="700" />
#   <p>数据集清洗对性能评估的影响</p>
# </div>
#
# <div align="center">
#   <img src="https://pbs.twimg.com/media/GjICcGQXYAAM4o1?format=jpg&name=4096x4096" width="800" />
#   <p>高质量基准测试示例</p>
# </div>

# %% [markdown]
# ## 11. 我们到底在评估什么？ (What Are We Evaluating)
#
# 在深入各种复杂的指标之前，我们需要回归原点，问自己一个根本性的问题：**我们到底在评估什么？** 换句话说，这场“游戏”的规则究竟是什么？
#
# ### 11.1 评估范式的转移
#
# **前大模型时代 (Pre-foundation models)**
# 在过去，我们评估的核心对象是 **方法 (Methods)**。
# * 大家严格遵守标准化的训练集和测试集划分（Standardized train-test splits）。
# * 比如在 ImageNet 竞赛中，数据是固定的，比拼的是谁的架构（ResNet, VGG 等）更优秀。
#
# **今天 (Today)**
# 现在，我们更多是在评估 **模型/系统 (Models/Systems)**。
# * 规则变成了“Anything goes”（各显神通）。
# * 只要能拿出更好的结果，你可以用更多的私有数据、更强的算力集群。这使得纯粹的算法比较变得更加困难。
#
#
# ### 11.2 例外：回归对“方法”的评估
#
# 虽然“拼模型”是主流，但社区中仍有一些项目致力于控制变量，回归对**方法论**的评估，以推动纯粹的技术进步。
#
# #### nanoGPT Speedrun
# Andrej Karpathy 发起的挑战，旨在回归对**计算效率**的极致追求。
# * **规则**：给定**固定**的数据集。
# * **目标**：看谁能用最少的**计算时间**（Compute time）达到特定的验证集 Loss。
# * 这迫使参与者去优化算法和工程实现，而不是单纯堆算力。
#
# <div align="center">
#   <img src="../images/karpathy-nanogpt-speedrun.png" width="600" />
#   <p>Karpathy nanoGPT Speedrun</p>
# </div>
#
# * [相关推文 Link](https://x.com/karpathy/status/1846790537262571739)
#
# #### DataComp-LM
# 这是一个专注于**数据处理方法**的基准测试。
# * [相关论文 Link](https://arxiv.org/abs/2406.11794)
# * **规则**：给定一个庞大的原始数据集 (Raw dataset)。
# * **目标**：设计最好的数据过滤和处理策略，使用标准化的训练流程 (Standard training pipeline)，看谁能训练出准确率最高的模型。
#
#
# ### 11.3 总结：明确游戏规则
#
# 这两种评估导向各有价值，但服务于不同的受众：
#
# 1.  **评估方法 (Evaluating Methods)**：旨在鼓励研究人员进行**算法创新**。
# 2.  **评估模型/系统 (Evaluating Models/Systems)**：直接服务于**下游用户**，告诉用户哪个成品最好用。
#
# **结论**：无论我们选择哪条路，最重要的是要**清晰地定义游戏规则**。混淆了这两者（例如拿一个使用了更多训练数据的模型去声称其算法更优），只会让评估失去意义。
