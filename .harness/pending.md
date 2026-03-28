## Pending Task

### Pending Infrastructure: Demo Figure Pipeline

Goal:
- 为由 demo 代码直接生成的教学图补一层统一脚手架，而不是把这部分复杂度立即压进当前 `Skill 6: Demo Insertion`。
- 支持在 lecture 需要 loss 曲线、决策边界、attention heatmap 或其他代码生成图时，完成“运行程序 -> 记录数据 -> 统一画图风格 -> 保存到 `images/<part name>/` -> 再插回讲义”的闭环。

Why pending:
- 如果要把这件事做得可复用，仅靠当前 lecture 内联 code block 还不够；通常还需要统一的运行入口、数据记录格式、命名规则、图片落盘路径和绘图 style。
- 这已经更像一层实验/可视化基础设施，而不是当前 `Demo Insertion` 里顺手补几行 `matplotlib`。
- 在没有这层脚手架之前，`Skill 6` 仍以“直接插入最小可运行 demo，必要时允许最小 figure-from-code”作为主职责，而不是默认承担稳定的图表产线。

Likely requirements later:
- 一个轻量但统一的运行入口，用来执行 demo 并产出可复现结果。
- 一个极简数据记录约定，用来保存 loss、accuracy、boundary grid、attention map 等绘图输入。
- 一套统一的 plotting style，避免每讲各自画出风格漂移很大的图。
- 明确图片输出目录与命名约定，例如 `images/<part name>/`。
- 讲义中的 image block 与生成脚本、输出图片之间的可追溯关系。

Current decision:
- 暂不为这件事新建独立 skill，也暂不扩展当前 `demo-insertion` 到完整图表产线。
- 先把它记为 pending infrastructure；等这类需求足够多时，再决定是扩展 `Skill 6` 还是额外抽出一层专门脚手架。


### 7. Skill: Figure Curation (Optional)

Goal:
- 在 rough draft 之后处理“增量图片需求”，而不是承担默认主线中的首次插图。
- 当用户后续又补充了一批本地图片或网页图片，且现有讲稿里还没有这些图时，再做一次资源驱动的筛选与补图。

When to use:
- 当 rough draft 已经完成，并且其中已经插入了该 lecture 当前已知的大部分关键图片。
- 当用户后续又提供了一批新的本地图片或网页链接，希望判断这些新增图片是否值得补进现有讲稿时。
- 当我们需要对新增图片做一次集中筛选，而不想重新走整轮 rough draft 时。

Artifact flow:
- In: existing lecture draft + newly provided figure resources
- Out: optional figure curation plan / incremental inserted image blocks

Responsibilities:
- 读取新增图片资源，判断哪些图真正值得补进现有讲稿，哪些图虽然存在但不值得再插。
- 将选中的新增图片映射到具体 section 或段落，说明它为什么补在这里、希望读者从图里看到什么。
- 对本地图片，插入指向仓库内已有图片的 image block。
- 对网页图片，在用户已经提供网页的前提下，定位适合插入的具体图片 URL，并插入指向该图片 URL 的 image block。
- 为每张新增保留图片补最小必要的 caption、引导语或观察提示，使图片与上下文叙述连起来。

Non-goals:
- 不承担默认流程中的首次图片插入；默认首次插图发生在 Skill 2 的 rough draft 阶段。
- 不生成新的图片、曲线或示意图；需要代码生成的图交给 Skill 6。
- 不从零开始做大规模开放式搜图；默认只处理用户后续给定的本地图资源或网页。
- 不因为“手头有图”就强行往讲义里补图。
- 不扩展 lecture 的理论覆盖范围。

Decision rules:
- 当新增图片能明显改善读者对某个局部机制、现象或结构的理解，并且现有讲稿里还没有同等效果的图时，才考虑补入。
- 当新增图片只是重复现有讲稿里已经插入的图，或者只是重复正文已经足够清楚的内容，通常不补入。
- 当多张新增图片表达的是同一个意思时，优先保留最直接、最干净、与当前 section 最贴合的一张。
- 当网页中有多张候选图时，应优先选择来源稳定、主题聚焦、与正文叙述粒度一致的图片。

Inputs:
- 单个 lecture 的现有 draft。
- 用户后续提供的新增本地图片资源。
- 用户后续提供的网页链接；必要时从网页中定位具体图片 URL。
- 当前 lecture 的 section 目标与上下文。

Outputs:
- 新增候选图片资源的筛选结果。
- 建议补图的位置清单。
- 每张新增保留图片的教学目的、caption 或引导语。
- 指向本地图片路径或具体网页图片 URL 的 image block。
- 不建议补入的图片及原因。

Suggested artifacts:
- `specs/lectures/<lecture_name>_figures.md`
- `specs/lectures/<lecture_name>_image_blocks.md`

### Workflow Boundary

- Skill 1 负责单个 lecture 的资料筛选、内容取舍与 section 结构设计。
- Skill 2 负责在既定 lecture plan 下写出一版材料完整的 lecture 粗草稿，并在当前 lecture/topic 范围内读取同级 `images/` 资源，把真正有教学价值的现成图片直接插入粗稿。
- Skill 3 负责整讲级的 rough-draft orchestration：可并行分派多个 `rough-drafter` subagent，由它们执行 `Skill 2`，并在最后统一合并成一版材料完整的 lecture 粗草稿。
- Skill 4 负责单 section 或单 lecture局部范围内的重写、删改和教学风格上的 revise。
- Skill 5 负责整讲级的 revision orchestration：可并行分派多个 `Skill 4` worker，并在最后统一处理上一讲顺承、section 桥接、术语一致性、去重复，以及一次 audit-only 二次检查。
- Skill 6 负责在最后阶段按理解缺口判断是否需要 demo，并把极简、可运行的 tutorial demos 直接生成并插入 lecture。
- Skill 7 只在 rough draft 之后出现“新增图片资源需要补入”的情况下启用；它不是默认主线中的首次插图阶段。
- Skill 1 先出方案，再通过用户交互收敛，不默认一次性定稿。
- Skill 1 不写教程正文。
- Skill 2 不重做 lecture 层面的内容取舍，除非发现明显缺口；但它必须把应保留的材料真正写进粗草稿，而不是只给提纲。这里的“材料”默认包括当前 scope 内真正有教学价值的现成图片，而不只是文字 references。
- Skill 2 不负责 lecture-level 的并行 drafting orchestration 与整讲 merge；这些任务由 Skill 3 统一负责。
- Skill 3 负责唯一一次 lecture-level drafting brief，并确保 references 分配、图片资源分配、全局术语和 source anchor 规则只在整讲层面统一处理，不由各 section worker 重复承担。
- Skill 3 只做 rough-draft 阶段所需的最小 merge cleanup，不把整讲提前 polish 成讲义。
- Skill 4 默认基于 Skill 2 或 Skill 3 产出的粗草稿完成局部 revise，不再重新通读参考资料；只有在来源冲突、证据不足、caption 明显不成立，或粗稿明显漏掉关键材料时才回源核对。
- Skill 4 不负责 lecture-level 的上一讲顺承与整讲 integration；这些任务由 Skill 5 统一负责。
- Skill 4 可以为保证局部认知连续性少量补写桥接段、最小例子或 code block placeholder；它默认整理、移动、删减 rough draft 中已经插入的图片，而不是把图片重新退回为默认占位符。
- Skill 5 负责唯一一次 lecture-level continuity brief，并确保上一讲顺承只在整讲层面处理，不由各 section worker 重复承担。
- Skill 5 负责整讲层面的图片去重、位置协调和 caption 风格统一；它可以建议哪里还缺例子或 demo，但不负责默认主线中的首次图片筛选。
- Skill 5 的 audit-only pass 只允许窄范围、低风险的 lecture-level 清理；如果问题需要第二轮完整重写，应明确暴露为后续任务，而不是在 audit 中静默完成。
- Skill 6 不负责扩展 lecture 的理论覆盖范围；它只在必要位置直接生成并插入极简、可运行的 demo 或代码生成图，不接管现成静态图片资源的默认使用流程。
- Skill 7 不负责扩展 lecture 的理论覆盖范围；它只在新增图片资源出现时，判断哪些新增图值得补入，以及这些图应如何与正文配合。
