# Progress

## Current Status
### 主要变动
- 收紧 `lecture-revision`：新增 audit-only 二次检查步骤，只允许拦截写作脚手架、课程结构元话语、source-aware 残留、记号不一致与公式转义污染等窄范围 lecture-level 问题，明确不升级为第二轮完整 revision。
- 新建项目内专用 `lecture-drafting` skill，负责并行分派多个 `rough-draft` worker，并在最后合并成一版整讲 rough draft。
- 并发调度 2 个 `writing-reviser` subagent，按 `## 1-4` 与 `## 5-8` 分工完成 `docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md` 的 lecture-level revision，并由主 agent 合并回正式文档。
- 新建项目内 Codex subagent 配置 `.codex/agents/writing-reviser.toml`，将其约束为只执行仓库里的 `writing-revision` skill。
- 新建项目内 Codex subagent 配置 `.codex/agents/rough-drafter.toml`，将其约束为只执行仓库里的 `rough-draft` skill。
- 将项目内 skill 目录统一整理到 `./.agents/skills/`。
- 新建并完成 `lecture-curation` skill，明确了单 lecture 结构策划的工作流、标题命名规则、section 级主要参考记录格式，以及“按教学价值抽取结构而不是摘抄原文”的原则。
- 生成了第二讲 outline 草稿 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`，并将标题体系收紧为更接近讲义大纲的命名风格。
- 更新了 `lecture-curation` 的标题规则与 output-format 示例，默认偏向“概念块 / 机制 / 边界 / 局限”式标题，避免口语化问句标题。
- 在 `docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md` 中新增“优化与训练稳定性” section，并顺延后续编号。
- 在 `docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md` 末尾补充了与当前 Lecture 1 最相关的 MIT 6.7960 / CS182 推荐阅读。
- 将 `docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md` 从 outline 展开为一版完整 rough draft：`## 1` 到 `## 8` 已补成连续中文正文，并按 section 加入 `主要参考`、必要的 source anchors，以及少量 `handoff` / `revise-note`。
- 收紧 `rough-draft` skill：明确 rough draft 不仅要保留理解路径，也必须保留 section 的关键公式、最小符号集、局部推导骨架和机制链，避免把技术核心压缩成纯直觉 prose。
- 修复了 `commit-progress` 的 frontmatter 描述，使其能够通过当前 skill 校验脚本。

### 未完成
- 第二讲 outline 仍可能需要继续根据用户偏好微调标题粒度和课程风格。
- 第一讲已完成一轮 `writing-revision`，正文从 source-aware rough draft 收紧为更面向读者的 teaching draft；下一步主要是 demo insertion 与必要的局部补图/补例。

### 当前 blocker
- 无。

## Message to Human
- 无

## History Logs
### 2026-03-26
- 更新 `.agents/skills/lecture-revision/SKILL.md` 与 `.harness/spec.md`：为整讲 revision 增加 audit-only 二次检查机制，限定其只做窄范围 residue audit 与低风险清理，不做第二轮完整重写。
- 新建 `.agents/skills/lecture-drafting/`，将其定义为整讲级 rough-draft orchestration skill：负责并行分派多个 `rough-draft` worker，并在最后统一做 section merge、术语一致性、source anchor 风格收口与最小必要连接。
- 收紧 `rough-draft` 的 frontmatter 与正文边界，明确其不负责 lecture-level 的并行 drafting orchestration 和最终 merge；这些任务改由 `lecture-drafting` 负责。
- 同步更新 `.harness/spec.md`：新增 `Skill 3: Lecture Drafting`，并将 `Writing Revision / Lecture Revision / Demo Insertion` 顺延为 `Skill 4 / 5 / 6`，明确 rough-draft 阶段与 revision 阶段各自的 orchestration owner。

### 2026-03-26
- 新建 `.agents/skills/lecture-revision/`，将其定义为整讲级 revision orchestration skill：负责并行分派多个 `writing-revision` worker，并在最后统一做上一讲顺承、section 间桥接、术语一致性与去重复。
- 收紧 `writing-revision` 的 frontmatter 与正文边界，删除其对上一讲顺承和 lecture-level continuity 的默认所有权，改为专注 section-local rewrite。
- 同步更新 `.harness/spec.md`：新增 `Skill 4: Lecture Revision`，并将 `Demo Insertion` 顺延为 `Skill 5`；明确 `Skill 3` 负责局部 rewrite，`Skill 4` 负责整讲 integration。

### 2026-03-26
- 并发调度 2 个 `writing-reviser` subagent，分别负责 `## 1-4` 与 `## 5-8`，产出独立 section-scoped revision 草稿后由主 agent 清洗 LaTeX 转义污染、合并回 `docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md`。
- 新建 `.codex/agents/writing-reviser.toml`，设置 `model = "gpt-5.4"`、`model_reasoning_effort = "medium"`、`sandbox_mode = "workspace-write"`，并在 `developer_instructions` 中强约束该 subagent 只执行 `.agents/skills/writing-revision/`。
- 收紧 `.harness/spec.md` 与 `.agents/skills/writing-revision/SKILL.md`：为 `Skill 3` 新增默认的上一讲顺承检查，约定同文件夹内编号减一的 lecture 视为上一讲。
- 明确这一检查只服务于开场承接、术语/记号一致性与去重复；若联系较弱或当前任务是 section-scoped revision，则只做最小必要检查，不额外大段重读或重写上一讲内容。

### 2026-03-26
- 收紧 `.harness/spec.md` 与 `.agents/skills/writing-revision/`：把“不要暴露写作脚手架”“不要在正文里提参考材料/课件来源”写成 `Skill 3` 的显式约束。
- 在 `writing-revision` 的 checklist 与 format reference 中加入反例约束，明确应避免 `本节目标`、`这一节真正想说明的是`、`参考材料里` 这类不自然表述。
- 对 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 再做一轮 revision，删除各 section 的 `本节目标` 块，并把 `这一节...`、`参考材料...` 改写成直接面向读者的讲义叙述。

### 2026-03-26
- 用户指出 Lecture 1 rough draft 过于偏向理解性 prose，关键公式与机制展开不足；据此回查 `.harness/spec.md`、`.agents/skills/rough-draft/SKILL.md` 与当前文稿，确认问题既有执行层因素，也有 skill 规则偏向“直觉优先、公式从简”的因素。
- 更新 `.agents/skills/rough-draft/SKILL.md`：在 source map、must survive、drafting 与 heuristics 中补入 `关键公式`、`最小符号集`、`局部推导骨架`、`机制链` 等显式要求，并禁止把 section 的技术核心压缩成 intuition-only prose。
- 更新 `.agents/skills/rough-draft/references/draft-format.md`：加入 `关键公式 / 机制骨架` 示例块与 `Technical Core` 说明，让 rough-draft agent 有统一落点来保留公式和机制。
- 静态验证：重新阅读 skill 与 format 文档，确认新增约束已经覆盖“读源时抽取 technical core”“写 draft 时保留公式和符号”“格式上提供明确落点”这三层。
- 基于更新后的 rough-draft 规则，只重跑了 Lecture 1 中最需要补技术骨架的 `## 3-5`；当前文稿已在这些 section 中补入 `关键公式 / 机制骨架`、最小符号定义、经验风险目标、两层 MLP 的反向传播骨架，以及最小二分类训练闭环。
- 进一步收紧 rough-draft 的公式格式约束：要求默认使用标准 Markdown 数学环境，行内公式用 `$...$`、独立公式用 `$$...$$`，反引号只用于代码样式字面量，不再作为公式容器。
- 将 `docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md` 中主要技术段落的公式格式统一到标准数学环境：把误用反引号包裹的变量、方程、损失和梯度链改成 `$...$`，并保留路径、API 名与真正代码字面量的反引号。
- 静态验证：回读 `docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md` 的 `## 3-5`，确认新增公式、机制链与 section 拼接落盘，未发现 heredoc 引入的转义污染。

### 2026-03-25
- 按用户要求并发调度了 4 个 rough-drafter 子 agent，按 `1-2 / 3-4 / 5-6 / 7-8` 分工为 `docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md` 起草正文。
- 其中首个 `7-8` 子 agent 返回了不符合要求的总结型消息，没有提供可合并的 section markdown；随后补发 recovery 子 agent，成功收回 `## 7-8` 的 rough-draft section blocks。
- 将四路 section 产物合并回 `docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md`，保留现有顶部信息与 `## 9 推荐阅读`，并把 `## 1-8` 展开成连续 rough draft。
- 静态验证：运行 `rg -n '^## ' docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md`，确认 `## 1` 到 `## 9` 结构完整；运行 `sed -n '1,120p'` 与 `tail -n 40` 检查正文开头、结尾、`主要参考` 块和关键 source anchors 已落盘。
- 未做渲染、格式化或 `writing-revision`，当前产物仍按 rough-draft 阶段管理。

### 2026-03-25
- 收紧 `.harness/spec.md` 中 `Skill 3: Writing Revision` 的边界，明确其主产物应直接落到 lecture 文档，而不是元规则文档。
- 加入 `Revision acceptance checklist`，要求逐 section 回答“这一节在解决什么问题”、显式桥接前后内容，并检查是否保留了 `Skill 2` 中真正高价值的 insight。
- 明确 `Skill 3` 可以留下空的 code block placeholder 或 picture placeholder 来补齐理解闭环，但不最终决定代码演示实现。
- 加入语气约束：正文应站在 tutorial 作者角度，而不是辅助 agent 或编辑者角度；协作信息应放进注释或 revision notes。
- 在 workflow boundary 中补充：`Skill 3` 可为认知连续性少量补写桥接段、最小例子和 placeholder，但不扩展理论覆盖范围，也不重做大规模资料抽取。

### 2026-03-25
- 新建 `.codex/agents/rough-draft.toml`，设置 `model = "gpt-5.4"`、`model_reasoning_effort = "high"`、`sandbox_mode = "workspace-write"`。
- 在 `developer_instructions` 中强约束该 subagent 只执行 `.agents/skills/rough-draft/`，遇到其他任务时拒绝并交回主 agent。
- 保持最小改动：未新增 `.codex/config.toml`，沿用 Codex 默认 agent 并发与深度设置。

### 2026-03-25
- 在 `docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md` 末尾新增 `## 9 推荐阅读`，整理了 MIT 6.7960 与 CS182 中与 Lecture 1 最相关的 11 个阅读入口。

### 2026-03-25
- 在 `docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md` 中新增 `## 6 优化与训练稳定性`，包含学习率、初始化、优化器、batch size、归一化/残差、训练曲线信号五个子标题。
- 将原 `## 6 非线性与网络深度的作用` 与 `## 7 深度学习中的三个基本问题` 顺延为 `## 7` 与 `## 8`。

### 2026-03-25
- 新建 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`，基于 `3_approximation_2025.pdf` 与 `7_generalization_2025.pdf` 生成第二讲 outline。
- 将第二讲标题由口语化问句改为更像讲义的命名方式，例如“通用逼近的含义与边界”“经典泛化理论及其局限”。
- 更新 `.agents/skills/lecture-curation/SKILL.md` 与 `.agents/skills/lecture-curation/references/output-format.md`，加入“默认使用讲义式标题、避免口语化 heading 模板”的约束与正反例。
- 运行 `python3 /home/zhouyf/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/lecture-curation`，结果为 `Skill is valid!`。

### 2026-03-25
- 将 `.agents/commit-progress` 和 `.agents/lecture-curation` 移动到 `.agents/skills/`。
- 完成 `.agents/skills/lecture-curation/SKILL.md` 与 `.agents/skills/lecture-curation/references/output-format.md`。
- 运行 `python3 /home/zhouyf/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/lecture-curation`，结果为 `Skill is valid!`。
- 运行 `python3 /home/zhouyf/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/commit-progress`，发现 frontmatter 描述含 angle brackets；做最小修复后再次校验，结果为 `Skill is valid!`。
