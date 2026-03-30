# Progress

## Current Status
### 主要变动
- `.agents/skills/writing-revision/SKILL.md` 又做了一轮小幅压缩：从 69 行降到 66 行，主要是收紧措辞、合并模式说明和最终检查表述；这份文件现在已经接近“只剩高价值规则”的压缩下限。
- `.agents/skills/lecture-drafting/SKILL.md` 已按 `simplify-skill-md` 压缩：从 100 行缩到 82 行，保留整讲粗草稿并行起草所需的 section allocation、image ownership、merge-pass 和 rough-stage 边界规则，合并了重复的 worker 约束和 merge checklist。
- `.agents/skills/rough-draft/SKILL.md` 已按 `simplify-skill-md` 压缩：从 109 行缩到 81 行，保留 rough draft 阶段最关键的覆盖优先、技术核心保留、初次图片插入和 section-scoped mergeability 规则，合并了重复的 workflow / heuristics / boundaries 表述。
- `.agents/skills/lecture-revision/SKILL.md` 已按 `simplify-skill-md` 压缩：从 109 行缩到 84 行，保留整讲并行修订所需的 lecture-level 协调规则，合并了重复的 worker 约束、integration checklist 和 audit 边界。
- 已创建一个全局 skill：`/home/zhouyf/.codex/skills/simplify-skill-md`。它只处理单个现有 `SKILL.md`，默认直接精简正文，不扩展到整个 skill 目录；规则取向为激进压缩，但保留 frontmatter、触发覆盖、核心流程和必要的资源导航。
- `.agents/skills/writing-revision/SKILL.md` 已按 context engineering 取向做了一轮压缩：从 122 行缩到 69 行，保留范围判断、读文件顺序、修订规则、占位与交接、边界和最终检查，删掉了重复 checklist、重复边界和低性价比写作理念展开。
- `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 当前已完成整讲 `## 1-7` 的 rough draft。
- 本次已按确认方案重排该讲 `## 4-7` 的子章节顺序，并补了必要的衔接句，使“插值问题 -> 搜索空间收缩 -> 经典复杂度失效 -> 深网泛化线索”这条主线连续下来。
- `## 7` 中原先混在一起的三类内容已拆开：
  - 结构偏置、Transformer 偏置、软偏置并入 `## 5`
  - 过参数化、假设分布、PAC-Bayes 与压缩界并入 `## 6`
  - `## 7` 只保留版本空间、优化偏置、实践先验、组合式泛化与未解问题

### 未完成
- 这讲尚未进入 `writing-revision` / `lecture-revision`。

### 当前 blocker
- 无。

## Message to Human
- 无

## History Logs
### 2026-03-28
- 按用户要求再次使用 `$simplify-skill-md` 处理 `.agents/skills/writing-revision/SKILL.md`。
- 这次改动较小，主要是：
  - 去掉 frontmatter `description` 尾部多余空格
  - 合并 `Scope` 里的模式说明
  - 进一步收紧 `Revision Rules`、`Figures, Placeholders, Handoff` 和 `Final Check` 的措辞
  - 将行数从 `69` 压到 `66`
- 验证：
  - `python3 /home/zhouyf/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/writing-revision`
  - `wc -l .agents/skills/writing-revision/SKILL.md`
  - `git diff -- .agents/skills/writing-revision/SKILL.md`

### 2026-03-28
- 按用户要求使用 `$simplify-skill-md` 精简 `.agents/skills/lecture-drafting/SKILL.md`。
- 注意：用户写的是 `lecture-draft`，仓库里实际 skill 目录是 `lecture-drafting`，本次按实际路径处理。
- 主要调整：
  - 删除独立的 `Overview`
  - 将 lecture-level drafting brief、worker constraints、merge decisions 合并到 `Read Order` / `Worker Orchestration` / `Merge Pass`
  - 删除重复的 `Worker Assignment Rules` 和 `Merge Checklist`
  - 补了简短 `Final Check`
  - 最终行数从 `100` 压到 `82`
- 验证：
  - `python3 /home/zhouyf/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/lecture-drafting`
  - `wc -l .agents/skills/lecture-drafting/SKILL.md`
  - `git diff -- .agents/skills/lecture-drafting/SKILL.md`

### 2026-03-28
- 按用户要求使用 `$simplify-skill-md` 精简 `.agents/skills/rough-draft/SKILL.md`。
- 主要调整：
  - 删除独立的 `Overview` 和单独的 `Writing Heuristics`
  - 将覆盖优先、技术核心、图片插入、local revise guidance 合并到 `Drafting Rules`
  - 将 section-scoped mergeability 约束并入 `Section-Scoped Mode`
  - 补了简短 `Final Check`，删除重复边界表述
  - 最终行数从 `109` 压到 `81`
- 验证：
  - `python3 /home/zhouyf/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/rough-draft`
  - `wc -l .agents/skills/rough-draft/SKILL.md`
  - `git diff -- .agents/skills/rough-draft/SKILL.md`

### 2026-03-28
- 按用户要求使用 `$simplify-skill-md` 精简 `.agents/skills/lecture-revision/SKILL.md`。
- 主要调整：
  - 删除独立的 `Overview` 和大段 `Integration Checklist`
  - 将 orchestration、worker rules、integration、audit 的重复约束合并成 `Scope` / `Read Order` / `Worker Orchestration` / `Integration and Audit`
  - 保留 lecture-level 高价值规则：previous-lecture continuity brief、worker 边界、整讲集成责任、audit-only pass 的窄修补边界
  - 将行数从 `109` 压到 `84`
- 验证：
  - `python3 /home/zhouyf/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/lecture-revision`
  - `wc -l .agents/skills/lecture-revision/SKILL.md`
  - `git diff -- .agents/skills/lecture-revision/SKILL.md`

### 2026-03-28
- 根据用户确认的边界，创建全局 skill `simplify-skill-md` 到 `/home/zhouyf/.codex/skills/simplify-skill-md`。
- 设计选择：
  - 仅作用于单个现有 `SKILL.md`
  - 默认直接改写，不输出单独诊断
  - 写成通用 skill，不绑定本仓库
  - 压缩策略偏激进：优先删除正文中的描述性、重复性语言
- 实现结果：
  - 用 `init_skill.py` 初始化 skill
  - 将模板 `SKILL.md` 改写为可执行版本，正文分成 `Scope`、`Read First`、`Compression Rules`、`Rewrite Procedure`、`Final Check`、`Output`
  - 没有创建额外的 `scripts/`、`references/`、`assets/`
- 验证：
  - `python3 /home/zhouyf/.codex/skills/.system/skill-creator/scripts/quick_validate.py /home/zhouyf/.codex/skills/simplify-skill-md`
  - `find /home/zhouyf/.codex/skills/simplify-skill-md -maxdepth 2 -type f | sort`

### 2026-03-28
- 按用户要求精简 `.agents/skills/writing-revision/SKILL.md`，目标是把它收缩成“触发后最小可执行说明”，减少对 context window 的占用。
- 主要调整：
  - 删除 `Overview`、独立的 `Section-Scoped Revision`、大段 `Revision Checklist`
  - 将重复的写作原则、边界和口吻要求合并进 `Scope` / `Read Order` / `Revision Rules`
  - 保留高价值的非显然规则：以 rough draft 为主、仅在 correctness 风险下回源、默认教学推进顺序、section-scoped 不越界、已有图片优先修补
  - 将行数从 `122` 压到 `69`
- 验证：
  - `python3 /home/zhouyf/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/writing-revision`
  - `wc -l .agents/skills/writing-revision/SKILL.md`
  - `git diff -- .agents/skills/writing-revision/SKILL.md`

### 2026-03-28
- 按用户确认的最终方案，重排了 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 的 `## 4-7`：
  - `4.6` 改为 `简单性偏好`
  - `## 5` 改为 `搜索空间收缩`，吸收原 `7.1-7.3`
  - `## 6` 改为 `经典复杂度的局限`，吸收原 `7.8-7.9`
  - `## 7` 改为 `深网泛化线索`，只保留版本空间、优化偏置、实践先验、组合式泛化和未解问题
- 同步补写了多处节间过渡句，避免重排后出现“刚讲过又回头补”的阅读跳跃。
- 验证：
  - `rg -n '^## |^### ' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`
  - `sed -n '380,760p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`
  - `sed -n '760,840p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`

### 2026-03-28
- 按用户要求继续扫 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md` 里可能不稳的绝对值/范数公式写法，把 3 处 `|...|` / `\|...\|` 统一改成更稳的 `\left|...\right|` / `\left\|...\right\|`：
  - `|f(x)-g(x)|\le\varepsilon`
  - `|g(x)-g(x')|\le L\|x-x'\|`
  - `\int_{[0,1]^d}|f(x)-g(x)|\,dx\le 2\epsilon`
- 这次刻意没有改 `|\mathcal{H}|` 这类集合基数写法，因为它通常渲染稳定，且语义上不是范数兼容性问题。
- 验证：
  - `sed -n '68,92p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`
  - `sed -n '140,150p' docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`

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
