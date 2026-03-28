# Progress

## Current Status
### 主要变动
- 已对 `rough-draft` 的图片规则做最小更新：首次插图、初始图位放置以及尽量收敛到最终 `images/` 路径，统一放在 `rough-draft` 阶段处理；没有同步扩散到其他 skill。
- 已将 `/.harness/spec.md` 补成固定执行顺序，并明确图片处理规则：图片首次插入跟随 `lecture-drafting`，`lecture-revision` 只整理已有图片，图片路径默认写相对目标文档的路径。
- 按用户要求先升级了 `lecture-curation` 输出规范：从“只在 `##` 级别记录 `主要参考`”改为“`##` 级别保留主参考，`###` 级别默认增加轻量 `小节参考`”。
- 新规范的核心是：`##` 级别继续写相对路径 + 精确 heading + 用途；`###` 级别只写相对路径 + 精确 heading，默认不重复长的“用于……”说明，除非该小节真的混合多个来源且容易混淆。
- `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 现已完成整讲 `## 1-7` 的 rough draft；其中 `## 4-7` 的主线已经按用户要求压回 `generalization_problem.md`，`generalization.md` 主要承担直觉图像、现代反例和 modern insight 的补充角色。
- `## 4` 现已从 outline 扩成完整粗稿：按“学习 vs 优化 -> 多项式回归中的欠拟合/过拟合 -> 多个插值解并存 -> `star` 问题 -> 文件柜与样本级记忆 -> 为什么偏向简单解”的链条展开，并插入了 `two_fits_comparison`、`filing_cabinet_model`、`sample_level_memorization` 三张图。
- `## 3` 已补齐“宽度直觉 -> ReLU 分段线性 -> 折点递推上界 -> 锯齿函数深度分离”的完整链条，明确把“通用逼近”与“表达效率”区分开来，并把 `K_L \le (2n)^L` 与 Telgarsky 式深度分离结论写入正文。
- 新的 `###` 级 `小节参考` 规范已经应用到这讲的 `## 4-7`：所有 `### 4.x`、`5.x`、`6.x`、`7.x` 都保留了轻量 source anchor，后续 drafting / revision 可直接按小节回源。
- 按用户要求用 3 个 `rough-drafter` subagent 完成了 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的 `## 5-7` rough draft，并由 integrator 做了 lecture-level merge。
- `## 5` 现已扩成完整粗稿：围绕“数据、先验、假设空间如何缩小搜索”展开，并插入了 1 张 `Occam` 图。
- `## 6` 现已扩成完整粗稿：按“经典 U 型/VC 直觉 -> 参数数不是有效容量 -> 随机标签 -> 显式正则化不足 -> 双下降”的顺序展开，并插入了 4 张现成图片。
- `## 7` 现已扩成完整粗稿：按“结构偏置 -> Transformer -> 软偏置 -> simple/compression bias -> 训练流程 -> practice as prior -> 理论转向 -> overparameterization -> PAC-Bayes/compression -> 组合式与 jagged generalization”的顺序展开，并插入了 9 张现成图片。
- merge pass 额外做了三件事：删除了 `## 5` 里重复的 `source-anchor` 注释、给 `## 6` 和 `## 7` 各补了一小段 integrator 过渡、把 `## 6-7` 的图片块统一成 `figure/figcaption` 结构。
- 明确了 tutorial skill 中“删除脚手架”的阶段分工：`rough-draft` / `lecture-drafting` 默认保留 `本节目标`、`主要参考` 与最小 handoff scaffolding，`writing-revision` / `lecture-revision` 则默认负责删除这些 reader-facing rough-draft 脚手架。
- 按用户要求对 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 执行了一次 section-scoped `$lecture-drafting` orchestration：显式使用了 2 个 `rough-drafter` subagent，现已把 `## 1-2` 合并成 rough draft，同时保留 `## 3+` 仍为 outline。
- 完成 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的 `## 2 逼近视角：从通用逼近到构造性逼近` rough draft，保持 `## 1`、`## 3` 及之后未改动。
- 这次 `## 2` 补齐了通用逼近的存在性边界、Lipschitz 下的分箱逼近、一维 `if`/指示函数到 ReLU 局部凸起的构造链，以及高维下的维数灾难和 proof-summary / handoff notes。
- 验证：使用 `sed -n '119,190p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 和 `rg -n '^## 2 |^### 2\\.|^## 3 |handoff|revise-note|主要参考：' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 抽查 section 边界，确认修改只落在目标区块内。
- 完成 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的 `## 1 总览：逼近问题与泛化问题的区分` 粗草稿：补入了逼近/泛化的概念区分、经验风险与总体风险的分离、近似误差/训练误差/泛化误差的角色分工，并插入了 `文件柜模型` 与“两种都拟合了数据”的现成图片。
- 这次只改了 `## 1`，保留 `## 2` 及之后章节不动；同时保留 section-level `主要参考` 和一个面向下一节的简短 handoff note，方便后续继续用 section-scoped drafting 接着写。
- 验证：通过 `sed -n` 抽查了修改后的 `## 1` 片段，确认图片路径、数学公式和 handoff/revise notes 都落在目标 section 内，没有触碰后续章节。
- 按用户要求重组了 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的 generalization 部分（第 4–7 节）：从原来的“直觉反例 / 经典理论 / 现代现象 / 当前解释”平铺结构，改为“插值不等于规律 / 数据-先验-假设空间三种工具 / 经典复杂度图景的断裂点 / 深网泛化的现有线索与开放问题”。
- 这次重组显式提取了此前没有放进框架主线的 insight，包括：样本级记忆比总体规律学习更容易、学习是在搜索空间里找“真相之针”、数据/先验/假设空间分别对应软约束与硬约束、复杂度度量本身是开放问题、以及 `simple + spikes` / 组合式泛化 / jagged generalization 等现代深网现象。
- 验证：重新阅读 `generalization.md` 与 `generalization_problem.md` 的相关节，并用 `sed -n` 抽查目标讲义，确认新的 `## 4–7` 标题、`###` 小节和 `主要参考` 锚点已经对应到新增主线。
- 将 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的 `主要参考` 从“文件 + 页段”改为“文件 + 精确 source heading + 用途”，使后续 rough draft 能直接回到对应 section，而不是只靠页段猜测。
- 同步更新 `.agents/skills/lecture-curation/SKILL.md` 与 `.agents/skills/lecture-curation/references/output-format.md`：`lecture-curation` 现在要求在 `##` 节级别记录“相对路径 + 精确 heading anchor + 该来源承担的教学作用”，不再默认使用 `pp. X–Y`。
- 验证：用 `rg` 和 `sed` 交叉检查了讲义文件、skill 正文和 output-format，确认三处表述已经对齐，且旧的 `page range` / `pp. X–Y` 约束已被移除。
- 按用户要求重写了 tutorial skill 的图片工作流：现成图片的首次插入前移到 `rough-draft`，后续 `writing-revision` / `lecture-revision` 默认基于讲稿内已有图片整理，不再把图片当成默认后置阶段。
- 更新了 `.harness/spec.md`、`rough-draft`、`lecture-drafting`、`writing-revision`、`lecture-revision`、`demo-insertion` 的正文及相关格式参考，统一了“图片属于 rough draft 技术素材”的边界。
- 将 `Figure Curation` 在 spec 中改写为 optional 增量补图流程，只在 rough draft 之后用户新增图片资源时启用。
- 验证：用 `rg` 交叉检查了相关 skill 文本，确认 `rough-draft`、两个 revision skill、`demo-insertion` 与 `spec` 对图片职责的表述已对齐。
- 按用户要求继续处理 `docs/core/01_Deep_Learning/01_PyTorch.py` 的 750 行之后内容：把 `Module` 技巧和 `Datasets/DataLoaders` 两大块改成中文 percent 脚本格式，并把章节连续编号接到 `## 13`、`## 14` 及其子章节。
- 删除了该范围内的练习提示和残留英文三引号注释，补回缺失的 `# %%` cell 分隔，保持脚本可继续按 percent 格式拆分。
- 验证：`python -m py_compile docs/core/01_Deep_Learning/01_PyTorch.py` 通过。
- 按用户要求完成 `docs/core/01_Deep_Learning/01_PyTorch.py` 500-750 行讲义注释的中文化与 percent 化，补齐了 `## 11` 到 `### 12.3` 的连续编号，并删除了该范围内的练习段。
- 这次更新覆盖了 `torch.nn.Linear`、参数与 `state_dict`、线性层训练闭环、GPU 训练可视化、`torch.nn.Sequential`、分类训练和 `forward` 自定义网络的导语；脚本语法检查通过。
- 按用户要求完成 `docs/core/01_Deep_Learning/01_PyTorch.py` 250-500 行讲义注释的中文化与 percent 化，补齐了 `##/###` 编号，并删除了该范围内的 `### Exercise` 练习段。
- 这次更新覆盖了 `Einsum`、`Broadcasting`、`Autograd`、`backward()`、梯度累积与推理内存、优化器等内容；保持代码可运行，语法检查通过。
- 按用户要求完成 `docs/core/01_Deep_Learning/01_PyTorch.py` 前 300 行讲义注释的中文化，并转成 `# %% [markdown]` / `# %%` 的 percent 脚本格式；保留代码逻辑不变。
- 已用 `python -m py_compile docs/core/01_Deep_Learning/01_PyTorch.py` 做过语法验证，通过。
- 新增一个仓库内发布脚手架：`src/sync_public_repo.py` 负责把当前仓库的 `docs/`、`images/` 单向镜像同步到开源仓库 `/home/data/users/zhouyf/myprojects/Tutorial-Site`。
- 在 `justfile` 增加了 `just sync-public`，作为正式同步入口。
- 对 `docs/core/01_Deep_Learning/01_Pytorch_and_Tensor.py` 执行了一次整讲级 `lecture-revision` orchestration：显式孵化了 4 个 `writing-reviser` subagent，按 `## 2`、`## 3-4`、`## 5-6`、`## 7-8` 分工修订，再由主 agent 统一做整讲 integration。
- 将上一轮 rough draft 进一步收紧成更像讲义正文的 tutorial prose：移除了 reader-facing 的 `主要参考`、`本节目标`、`取舍说明` 和大部分协作脚手架，保留了 `tensor -> 形状与内存 -> autograd -> nn.Module -> 最小训练闭环 -> 常见工程接口` 这条主线。
- 按用户要求为 `docs/core/01_Deep_Learning/01_Pytorch_and_Tensor.py` 的 torch / nn 相关操作补了首次出现注释，原则是只在第一次出现处点明作用，后续重复调用保持简洁。
- 做了一轮 audit-only 清理：统一 lecture opening 与 closing，清掉 HTML handoff/source 注释残留，修复局部 merge 残留，并确认最终文件不再暴露 revision scaffold。

### 未完成
- 这次只修改了 skill 正文和格式参考；若后续希望 skill 触发条件也显式反映新图片流程，还需要再同步相关 frontmatter `description` / `openai.yaml`。
- 尚未做 notebook/percent-format 渲染预览；当前只完成脚本级验证。
- `Tensor 与 NumPy 的互转关系` 仍然是 text-first 小节；如果后续决定显式加入 `numpy` 依赖，可以再补 runnable demo。

### 当前 blocker
- 无。
## Message to Human
- 无

## History Logs
### 2026-03-28
- 修复 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 中另一条 display math 的渲染兼容性问题：将
  `\|f_\theta(x^{(i)})-y^{(i)}\|_2^2+\lambda \|\theta\|_2^2`
  改为
  `\left\|f_\theta(x^{(i)})-y^{(i)}\right\|_2^2+\lambda \left\|\theta\right\|_2^2`。
- 这是最小改动，只改了该公式本身，没有调整周边正文或编号。
- 验证：`sed -n '463,474p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`

### 2026-03-28
- 修复 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 中一条 display math 的渲染兼容性问题：将
  `\lvert f_\theta(x^{(i)})-y^{(i)}\rvert^{0.25}` 改为更稳的
  `\left|f_\theta(x^{(i)})-y^{(i)}\right|^{0.25}`。
- 这是最小改动，只改了该公式本身，没有调整周边正文或编号。
- 验证：`sed -n '418,432p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`

### 2026-03-28
- 按用户要求只做最小 skill 改动：更新 `.agents/skills/rough-draft/SKILL.md` 与 `.agents/skills/rough-draft/references/draft-format.md`，明确 rough draft 是默认的首次插图阶段，并在这一阶段完成初始图位放置与尽量收敛到最终 `images/` 路径。
- 这次没有修改 `lecture-drafting`、`lecture-revision`、`writing-revision` 等其他 skill，刻意把改动限制在 `rough-draft` 链路内。

### 2026-03-28
- 将 `/.harness/spec.md` 补充为固定执行顺序，并把图片处理规则写清楚：图片首次插入放在 `lecture-drafting`，`lecture-revision` 只整理已有图片，路径默认使用相对目标文档的写法；只有目标文档位置或图片资源目录变化时才改路径。

### 2026-03-28
- 按用户要求执行 section-scoped `$rough-draft docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md ## 4`，只在目标区块内把 `## 4` 从 outline 扩成完整粗稿，未触碰 `## 5+` 的正文。
- `## 4` 的组织顺序按当前 lecture plan 落地为：
  - `4.1` 学习目标与优化目标的区分，显式写出经验风险与总体风险
  - `4.2` 多项式回归中的欠拟合与过拟合，保留最小公式骨架与 U 型直觉
  - `4.3-4.4` 多个插值解并存、训练集上更贴不等于更接近真规律，并引入 `star` 问题
  - `4.5` 文件柜模型与样本级记忆更容易
  - `4.6` 收束到“泛化真正要解释的是为什么偏向简单解”
- 本次插入 3 张现成图片，均采用居中 `figure/figcaption` 结构：
  - `intuitive_ideas_about_generalization_two_fits_comparison.png`
  - `intuitive_ideas_about_generalization_filing_cabinet_model.png`
  - `intuitive_ideas_about_generalization_sample_level_memorization.png`
- 保留了 `本节目标`、`主要参考`、`小节参考`，并新增 1 条 local `revise-note` 与 1 条面向 `## 5` 的 `handoff`，符合 rough-draft 阶段不提前清理脚手架的规则。
- 同步更新 `.harness/progress.md` 的 `Current Status`：当前这讲 `## 1-7` 都已进入 rough-draft 阶段，下一步自然是 `writing-revision` / `lecture-revision`。
- 验证：
  - `sed -n '240,330p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`
  - `sed -n '330,430p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`
  - `sed -n '430,462p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`
  - `rg -n '^## 4 |^### 4\\.|^## 5 |^小节参考：|<figure>|figcaption|handoff|revise-note' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`
  - 未运行测试；本次是讲义 section rough draft，不涉及可执行代码

### 2026-03-28
- 按用户要求执行 `$lecture-drafting docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的 `## 5-7`，并显式使用了 3 个 `rough-drafter` subagent：
  - worker A 负责 `## 5`
  - worker B 负责 `## 6`
  - worker C 负责 `## 7`
- orchestration brief 明确了：
  - `## 5` 以 `generalization_problem.md` 的 `11.3`、`11.5` 为主 spine，只把 `Occam` 当补充
  - `## 6` 先立经典图景，再用 `generalization.md` 提供随机标签、复杂度失灵、显式正则化不足和双下降
  - `## 7` 承担 modern insights，不再重复本科主线骨架
  - images 按 section 级 ownership 分配，避免 worker 全量扫图
- worker 合并结果：
  - `## 5` 完成 `5.1-5.4` 粗稿，并插入 `generalization_theory_occams_razor.png`
  - `## 6` 完成 `6.1-6.5` 粗稿，并插入 `bias_variance_tradeoff`、`cifar10_train_accuracy`、`cifar10_test_accuracy`、`double_descent_schematic` 四张图
  - `## 7` 完成 `7.1-7.10` 粗稿，并插入结构偏置、soft bias、version space、parameter-function map、overparameterization、GPT-2 bounds、compositional/jagged generalization 等九张图
- integrator merge pass 额外做了 lecture-level cleanup：
  - 删除 `## 5` 中 4 处重复的 `source-anchor` HTML 注释
  - 在 `## 6` 与 `## 7` 的 section opening 各补 1 段跨节过渡
  - 将 `## 6-7` 全部图片块统一为居中 `figure/figcaption`
  - 保留 `本节目标`、`主要参考`、`小节参考`、必要 `handoff/revise-note`，不提前做 revision 阶段清理
- 验证：
  - `rg -n '^## 5|^## 6|^## 7|^### 5\\.|^### 6\\.|^### 7\\.|^小节参考：|<figure>|figcaption|source-anchor' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`
  - `sed -n '306,840p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`
  - 未运行测试；本次是文稿 rough draft orchestration，不涉及可执行代码
- 按用户要求把新的 `lecture-curation` 规范实际应用到 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md:311` 之后的 `## 4-7`。
- 具体修改是：在 `### 4.1-4.6`、`### 5.1-5.4`、`### 6.1-6.5`、`### 7.1-7.10` 下逐一补入 `小节参考：`，格式严格采用“相对路径 + 精确 heading”，只在确实混合来源的小节使用两行轻量锚点。
- 这次没有改动 `## 4-7` 的结构、标题和 `##` 级 `主要参考`，只把 subsection-level drafting anchor 补全。
- 验证：`sed -n '311,520p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`；`rg -n '^### (4|5|6|7)\\.|^小节参考：' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`，确认 `## 4-7` 下每个 `###` 都有对应的 `小节参考`。
- 按用户要求先设计新的 `lecture-curation` 输出规范，而不是立即批量改讲义：更新了 `.agents/skills/lecture-curation/SKILL.md` 和 `.agents/skills/lecture-curation/references/output-format.md`。
- 规则变化是：
  - `##` 级别继续保留 `主要参考`，格式仍为“相对路径 + 精确 heading + 用途”
  - `###` 级别默认新增 `小节参考`
  - `小节参考` 采用轻量格式，只写“相对路径 + 精确 heading”，除非该小节确实混合多来源且需要消歧
- 在 skill 正文里同步写明：review 时要检查每个 `###` 是否已有足够精确的 drafting anchor；在 format reference 里加入了 `小节参考` 的单源和双源示例。
- 验证：`rg -n '小节参考|lightweight|###-level|##-level' .agents/skills/lecture-curation/SKILL.md .agents/skills/lecture-curation/references/output-format.md`；`sed -n '1,260p' .agents/skills/lecture-curation/references/output-format.md`。
- 按用户要求继续补强 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的 generalization outline，把此前判断“值得补”的 4 条 insight 显式挂进结构：
  - `4.4 训练集上更贴的解，不一定更接近真实规律`
  - `6.4 显式正则化不是深网泛化的全部解释`
  - `7.7 从“数假设”到“刻画哪些解更可能出现”`
  - `7.8 过参数化为何可能同时扩大空间并增强简单性偏置`
- 同时把用户点名的两个点升格成更显眼的小节：
  - `7.2 Transformer 的结构偏置为何常被低估`
  - `7.6 也许“深度学习实践”本身就是我们的先验`
- 为了容纳这些新增点，`## 4` 扩成 `4.1-4.6`，`## 6` 扩成 `6.1-6.5`，`## 7` 扩成 `7.1-7.10`，但仍保持 framework，不重新引入 rough-draft 正文。
- 验证：`sed -n '311,410p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`；`rg -n '^### 4\\.|^### 6\\.|^### 7\\.' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`。
- 按用户要求执行 `$lecture-curation`，重排 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的 `## 4-7` outline，并明确允许直接删除不该保留的填充正文。
- `## 4` 现在改为以 `generalization_problem.md` 为主线的五步结构：学习目标与优化目标的区分、多项式回归中的欠拟合与过拟合、多个插值解为何让泛化变得困难、文件柜模型与“记忆不等于规律”、以及“为什么偏向简单解”的本节收束。
- `## 5` 保留为“三种工具”主线，但删除了上一轮 rough draft 正文，只保留 `数据 / 先验 / 假设空间 / 三者取舍` 的 outline。
- `## 6` 改成“先立经典图景，再给深网反例”的顺序：经典 U 型与 VC 直觉、参数数量不是有效容量、随机标签实验、双下降。
- `## 7` 保留现代解释线索链条：结构归纳偏置、软偏置、简单性偏置、优化与实践、理论化尝试、不完整性。
- 直接删除了 `## 4-7` 里上一轮 rough draft 的正文、图片块和 source/handoff 注释，使文件重新符合 `lecture-curation` 的 framework 边界。
- 验证：`sed -n '311,520p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`；`rg -n '^## [4-7] |^### [4-7]\\.|<div align=\"center\">|<img src=|<!-- source:|<!-- handoff:' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`。
- 按用户要求把 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的 `## 3-4` 从 outline 扩成 rough draft，继续沿用 section-scoped `lecture-drafting` 工作流，但这次没有额外扩到 `## 5+`。
- `## 3` 增补了四段主线：为什么宽度直觉看起来合理、为什么 ReLU 网络可从分段线性/折点视角分析、为什么折点数对宽度与深度呈现加法式 vs 指数式差异，以及为什么锯齿函数给出了深度分离的典型例子。
- `## 4` 增补了文件柜模型、样本级记忆更容易、振荡解与平滑解并存、多项式回归/`\star` 例子的连接，并插入了 `sample_level_memorization`、`oscillating_fit`、`smooth_fit` 三张 reference 目录内已有图片；为避免重复，没有再次插入第 `1` 节已经出现的 `filing_cabinet_model` 和 `two_fits_comparison`。
- 验证：`sed -n '190,340p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`；`rg -n '^## 3 |^### 3\\.|^## 4 |^### 4\\.|主要参考：|intuitive_ideas_about_generalization_' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`。
- 按用户要求把“`本节目标` / `主要参考` 何时删除”的分工显式写进相关 skill：更新了 `rough-draft`、`lecture-drafting`、`writing-revision`、`lecture-revision` 及其格式参考，统一成“rough draft 保留脚手架，revision 删除脚手架”。
- 验证：运行 `rg -n '本节目标|主要参考|rough-draft scaffolding|goal/reference scaffolding|reader-facing cleanup belongs to revision|default stage to remove|leftover rough-draft section scaffolding' .agents/skills/rough-draft/SKILL.md .agents/skills/rough-draft/references/draft-format.md .agents/skills/lecture-drafting/SKILL.md .agents/skills/writing-revision/SKILL.md .agents/skills/writing-revision/references/revision-format.md .agents/skills/lecture-revision/SKILL.md`，确认分工表述已在 drafting / revision 两侧同步落盘。
- 按用户要求执行 section-scoped `$lecture-drafting`：显式使用 2 个 `rough-drafter` subagent，分别起草 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的 `## 1` 与 `## 2`，再由主 agent 合并回同一文件。
- 合并后的结果是：`## 1` 加入了逼近/泛化区分、经验风险与总体风险公式，以及两张现成图片；`## 2` 加入了 `2.1-2.5` 的粗稿正文与 `#### 2.5.x` 的证明总结，同时把 `主要参考` 扩到 `## 证明总结与讨论。` 以覆盖新增内容。
- 验证：运行 `rg -n '^## [12] |^### [12]\\.|^#### 2\\.5\\.|主要参考：|handoff|revisi' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`，并用 `sed -n '1,190p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 抽查，确认 `## 1-2` 已落盘、`## 3+` 仍为 outline、section-level anchors 与 handoff notes 保留。
- 按 `lecture-drafting` 的 section-scoped 模式完成 `## 2` 的 rough draft：把通用逼近、Lipschitz 分箱、`if`/指示函数到 ReLU 局部凸起的机制链、以及高维维数灾难写成了可直接进入后续 revision 的粗草稿。
- 该 section 只使用了 `approximation.md`，并保留了 section-level `主要参考`、局部 `revise-note` 和面向 `## 3` 的 handoff note；没有插入新图片，也没有改动 `## 1` 或 `## 3+`。
- 验证：`sed -n '119,190p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`，以及 `rg -n '^## 2 |^### 2\\.|^## 3 |handoff|revise-note|主要参考：' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`。
- 按 `lecture-drafting` 的 section-scoped workflow 完成 `## 1` 的 rough draft：引入逼近 vs 泛化区分、经验风险/总体风险公式、文件柜模型和两种拟合曲线图片，作为后续 `## 2` 的承接。
- 验证：使用 `sed -n` 抽查 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的 `## 1` 区块，确认修改只落在目标 section 内，未影响 `## 2+`。
- 按用户要求重组 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的第 `4–7` 节，将 generalization 部分改为四段链条：`插值并不等于学到规律`、`数据/先验/假设空间如何缩小搜索`、`经典复杂度图景为何不足以解释深网`、`深网泛化的现有线索与开放问题`。
- 这次显式吸收了 `generalization.md` 和 `generalization_problem.md` 里之前没有被提到结构层的 insight：`样本级记忆比总体规律更容易`、`真相之针/路灯` 搜索框架、三种工具的软硬约束区分、`复杂度怎么量` 本身是开放问题，以及 `simple + spikes`、组合式泛化和 jagged generalization。
- 验证：再次用 `sed -n '55,190p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 抽查修改后的 generalization 部分，确认新的 `本节目标`、`主要参考` 与 `###` 小节已经落盘。
- 按用户要求把 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的 `主要参考` 升级为“文件 + 精确 source heading + 用途”的格式，逐节标出了对应的 `##` / `###` 来源，而不再使用宽泛的页段描述。
- 同步修改 `.agents/skills/lecture-curation/SKILL.md` 与 `.agents/skills/lecture-curation/references/output-format.md`，将 `lecture-curation` 的参考记录规范从 `page range` 改为 `exact source heading anchors`。
- 验证：运行 `rg -n '主要参考：|pp\\. |page range|exact source heading|heading anchors' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md .agents/skills/lecture-curation/SKILL.md .agents/skills/lecture-curation/references/output-format.md`，并用 `sed -n` 抽查三处文件，确认新格式已生效且规则文本一致。
- 按用户要求实现了 tutorial skill 的图片工作流调整：将现成图片的首次插入前移到 `rough-draft`，把 `Figure Curation` 在 `.harness/spec.md` 中改写为 rough draft 之后的 optional 增量补图流程。
- 更新了 `rough-draft` 与其 `draft-format`，要求在当前 lecture/topic scope 内读取同级 `images/` 目录、直接插入有教学价值的图片，并统一 image block 的相对路径、caption 与 `revise-note` 约定。
- 更新了 `lecture-drafting` 与其 orchestration format，增加 section 级图片资源分配、merge 时的重复图片清理，以及 worker 不应全量扫图的约束。
- 更新了 `writing-revision` / `lecture-revision` 及其参考格式，改成“优先整理已有 image block，仅在异常情况下才留 picture placeholder”，并补上整讲级图片去重、位置协调与 caption 收口规则。
- 更新了 `demo-insertion` 边界：已有静态图片默认属于 rough-draft 工作流，demo skill 不再接管现成图片的默认使用。
- 验证：运行 `rg -n '首次插图|images/|image block|picture placeholder|figure-curation|Figure Curation|增量图片|duplicated images|静态图片|rough-draft workflow' .harness/spec.md .agents/skills/rough-draft/SKILL.md .agents/skills/rough-draft/references/draft-format.md .agents/skills/lecture-drafting/SKILL.md .agents/skills/lecture-drafting/references/orchestration-format.md .agents/skills/writing-revision/SKILL.md .agents/skills/writing-revision/references/revision-format.md .agents/skills/lecture-revision/SKILL.md .agents/skills/lecture-revision/references/orchestration-format.md .agents/skills/demo-insertion/SKILL.md`，确认关键职责表述已覆盖并对齐。

### 2026-03-27
- 按用户要求继续处理 `docs/core/01_Deep_Learning/01_PyTorch.py` 的 750 行之后内容：把 `Module` 技巧和 `Datasets/DataLoaders` 两大块改成中文 percent 脚本格式，并把章节连续编号接到 `## 13`、`## 14` 及其子章节。
- 删除了该范围内的练习提示和残留英文三引号注释，补回缺失的 `# %%` cell 分隔，保持脚本可继续按 percent 格式拆分。
- 验证：`python -m py_compile docs/core/01_Deep_Learning/01_PyTorch.py` 通过。

### 2026-03-27
- 按用户要求继续处理 `docs/core/01_Deep_Learning/01_PyTorch.py` 的 500-750 行：把 `torch.nn.Linear`、参数与 `state_dict`、训练闭环、GPU 可视化、`torch.nn.Sequential`、分类训练和 `forward` 自定义网络导语改成中文 percent 脚本格式，并按 `## 11`、`### 12.1`、`### 12.2`、`### 12.3` 重新编号。
- 删除了该范围内的 `### Exercise` 练习段和对应 `TODO` 注释，同时去掉了 IPython magic 的残留说明注释。
- 验证：`python -m py_compile docs/core/01_Deep_Learning/01_PyTorch.py` 通过。

### 2026-03-27
- 按用户要求继续处理 `docs/core/01_Deep_Learning/01_PyTorch.py` 的 250-500 行：把 `Einsum`、`Broadcasting`、`Autograd`、`反向传播`、`梯度累积/清零/推理内存`、`优化器` 等段落改成中文 percent 脚本格式，并按 `## 5` 到 `### 10.5` 重新编号。
- 删除了该范围内的 `### Exercise` 练习段和对应 `TODO` 注释。
- 修复了边界处的模块小节标题与字符串边界，保证脚本仍可被 `py_compile` 正常解析。
- 验证：`python -m py_compile docs/core/01_Deep_Learning/01_PyTorch.py` 通过。

### 2026-03-27
- 按用户要求把 `docs/core/01_Deep_Learning/01_PyTorch.py` 前 300 行的英文讲义注释改成中文，并整理为 percent 脚本格式：加入了 `# %% [markdown]` / `# %%` 单元标记，保留代码不变，只做文本层和单元层的整理。
- 涉及的范围包括主题 1、Tensor 初始化、Tensor 属性、基础操作、维度顺序约定、索引切片、拼接、空维度、算术运算、单元素 tensor、Einsum 和 Broadcasting 的开头说明。
- 验证：`python -m py_compile docs/core/01_Deep_Learning/01_PyTorch.py` 通过。

### 2026-03-27
- 按用户要求为 `docs/core/01_Deep_Learning/01_Pytorch_and_Tensor.py` 补充 torch 操作注释：对首次出现的 `torch.tensor`、`reshape/view`、`transpose/permute`、autograd、`nn.Module`、`nn.Linear`、`SGD`、`Dataset/DataLoader` 等调用添加简短行内说明，其余重复调用不加注释，避免讲义代码过密。
- 修正了两处最初误落在 markdown 示例里的注释，把 `zero_grad` / `step` 的说明移回实际代码行，并补上 `zeros_like` 的首次说明。
- 验证：`uv run python docs/core/01_Deep_Learning/01_Pytorch_and_Tensor.py` 通过。
- 按用户要求新增 `just sync-public` 发布脚手架：实现 `src/sync_public_repo.py`，将当前仓库 `docs/`、`images/` 镜像同步到 `/home/data/users/zhouyf/myprojects/Tutorial-Site`；脚本支持 `--target-root` 覆盖默认目标，便于在 `/tmp` 做无副作用验证。
- 同步语义为目录级镜像：目标仓库中 `docs/`、`images/` 里源端不存在的条目会删除，源端存在的文件会递归复制覆盖。
- Installed `numpy`, `einops`, and `jaxtyping` with `uv add`; cleaned `references/01_Deep_learning/01_Pytorch_and_Tensor/Tensor.py` by inlining the FLOP constants and removing the unused broken `zero_2019` import so the reference script compiles standalone.
- Finished lecture-revision integration for `docs/core/01_Deep_Learning/01_Pytorch_and_Tensor.py`: locally tightened `## 3-4`, re-ran scaffold audit, `uv run python -m py_compile`, and full `uv run python` end-to-end check; lecture remains runnable, with the existing NumPy warning unchanged.
- 按用户要求对 `docs/core/01_Deep_Learning/01_Pytorch_and_Tensor.py` 执行 `$lecture-revision`，并显式孵化了 4 个 `writing-reviser` subagent；integrator 自己保留了 `## 1` 开场、`## 9` 收束、跨 section 桥接和 audit-only 清理所有权。
- revision scope 切分为：worker A 负责 `## 2`，worker B 负责 `## 3-4`，worker C 负责 `## 5-6`，worker D 负责 `## 7-8`；整讲 opening / conclusion 与全局术语统一由主 agent 处理。
- integration pass 中做了 lecture-level cleanup：移除了 reader-facing 的 `主要参考` / `本节目标` / `取舍说明`，清掉 HTML comments 与 source-aware residue，统一了 tutorial-author voice，并把 `NumPy` 依赖继续压在 runnable path 之外。
- 处理了一个中断带来的 merge 问题：上一回合被用户中断后，部分 worker 临时产物没有保留下来，因此重新在仓库内 `.tmp_revision/` 目录收集片段，再完成最终整合；同时对 `## 4` 开头残留的旧 scaffold 做了定点修补。
- 验证：`uv run python -m py_compile docs/core/01_Deep_Learning/01_Pytorch_and_Tensor.py` 通过；`rg -n '主要参考|本节目标|取舍说明|参考材料|slides 上|<!--|TODO|这一节真正想说明的是' docs/core/01_Deep_Learning/01_Pytorch_and_Tensor.py` 未命中；`uv run python docs/core/01_Deep_Learning/01_Pytorch_and_Tensor.py` 跑通，训练闭环、Dropout 演示和 DataLoader 示例均执行成功。
- 运行时仍有一条来自 `torch` 的 warning：当前环境缺少 `numpy`，导致 NumPy bridge 初始化失败；这不影响整讲脚本执行，但说明若后续要把 `Tensor <-> NumPy` 小节升级成 runnable demo，最好显式把 `numpy` 纳入 `uv` 依赖。

### 2026-03-26
- 按用户要求对 `docs/core/01_Deep_Learning/01_Pytorch_and_Tensor.py` 执行 `$lecture-drafting`，并显式孵化了 3 个 `rough-drafter` subagent；integrator 自己保留了 `## 1` 开场、`## 9` 小结和跨 worker 的 merge cleanup 所有权。
- section ownership 切分为：worker A 负责 `## 2-3`（tensor 基本表示、形状思维），worker B 负责 `## 4-6`（内存/dtype/device、autograd、`nn.Module`），worker C 负责 `## 7-8`（最小训练闭环、常见接口与工程习惯）。
- 主 agent 合并时做了最小 lecture-level cleanup：补写整讲开头与结尾，统一术语和 CPU-first 叙述，保留 `NumPy` / `einops` 为 text-first 边界说明，避免把当前仓库没有声明的额外依赖强行写进 runnable path。
- 针对可运行性做了两处收口：将 `NumPy` 互转收缩为文字说明与 revise-note；随机种子示例保留 `random + torch` 的最小版本，从而与当前 `uv` 环境保持一致。
- 验证：`uv run python -m py_compile docs/core/01_Deep_Learning/01_Pytorch_and_Tensor.py` 通过；`uv run python docs/core/01_Deep_Learning/01_Pytorch_and_Tensor.py` 跑通，最小训练闭环、Dropout train/eval 演示、DataLoader 示例均执行成功。
- 运行时有一条来自 `torch` 的 warning：当前环境缺少 `numpy`，导致 NumPy bridge 初始化失败；这不影响本讲脚本执行，但说明如果后续想把 `Tensor <-> NumPy` 小节升级成 runnable demo，最好显式把 `numpy` 纳入 `uv` 依赖。

### 2026-03-26
- 按用户要求对 `docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md` 执行 `$lecture-revision`，并显式孵化了 2 个 `writing-reviser` subagent；用户额外约束为“保留已经手动调整过的章节顺序，不再改顺序，只改章节号和衔接段落”。
- 读取整讲当前结构后，确认旧编号与新顺序发生错位：顶层顺序实际为 `1 / 3 / 3.6 / 4 / 2 / 5 / 6 / 7 / 9 / 10`，并据此将 worker 切分为“前半讲 `1-4`”与“后半讲 `5-10`”两个 disjoint scope。
- 主 agent 合并两路修订结果时，保留了用户当前 section 顺序，统一将顶层编号改写为 `## 1` 到 `## 10`，并同步修正了 `### 2.x -> 5.x`、`### 5.x -> 6.x`、`### 6.x -> 7.x` 等局部编号。
- 主 agent 在 integration pass 中补写了关键过渡，包括 `4.5 -> 5`、`5 -> 6`、`6 -> 7`、`7 -> 8`，使 reordered lecture 读起来仍像一讲连续教程，而不是把旧顺序硬拼在一起。
- audit-only 验证：运行 `rg -n '^## |^### ' docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md`，确认顶层与小节编号连续；运行 `rg -n '## 3\.6|### 2\.|### 3\.|### 5\.|### 6\.|第 14 章|主要参考|参考材料里|slides 上|本节目标|这一节真正想说明的是' docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md` 未命中，未发现旧编号或脚手架残留。
- 局部人工抽查：检查了 `4.5 -> 5` 与 `7 -> 8` 附近段落，确认新的过渡句已落盘；未进行渲染或格式化。
