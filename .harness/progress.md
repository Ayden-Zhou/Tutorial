# Progress

## Current Status
### 主要变动
- 新建项目内专用 `rough-draft` skill，负责从 lecture plan 与参考资料直接写出材料完整的 lecture 粗草稿。
- 在 `rough-draft` 中加入 section-scoped drafting 约束，兼容后续多个 subagent 并行撰写不同章节并再合并。
- 将 `.harness/spec.md` 中 `Skill 2` 从 `Skeleton Drafting` 重定义为 `Rough Draft`，改为直接产出材料完整的 lecture 粗草稿。
- 相应重写 `Skill 3`，明确其默认只消费 `Skill 2` 粗草稿做重写与教学化 revise，而不是重新大规模翻参考资料。
- 将项目内 skill 目录统一整理到 `./.agents/skills/`。
- 新建并完成 `lecture-curation` skill，明确了单 lecture 结构策划的工作流、标题命名规则、section 级主要参考记录格式，以及“按教学价值抽取结构而不是摘抄原文”的原则。
- 生成了第二讲 outline 草稿 `docs/core/01_Deep_Learning/02_Approximation_and_Generalization.md`，并将标题体系收紧为更接近讲义大纲的命名风格。
- 更新了 `lecture-curation` 的标题规则与 output-format 示例，默认偏向“概念块 / 机制 / 边界 / 局限”式标题，避免口语化问句标题。
- 修复了 `commit-progress` 的 frontmatter 描述，使其能够通过当前 skill 校验脚本。

### 未完成
- 第二讲 outline 仍可能需要继续根据用户偏好微调标题粒度和课程风格。
- `rough-draft` 目前只完成了结构校验，尚未用真实 lecture 草稿任务做一次前向试跑。

### 当前 blocker
- 无。

## Message to Human
- 无

## History Logs
### 2026-03-25
- 使用 `init_skill.py` 在 `.agents/skills/rough-draft` 初始化新 skill，并补全 `SKILL.md`、`agents/openai.yaml` 与 `references/draft-format.md`。
- 将 `rough-draft` 定义为项目内专用的粗草稿生成 skill：输入 lecture plan 与参考资料，输出材料完整但未精修的 lecture 草稿。
- 在 `rough-draft` 中加入 section-scoped drafting 规则，要求只改指定 section、保留 handoff 注释与来源锚点，以兼容多个 subagent 并行写不同章节。
- 运行 `python3 /home/zhouyf/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/rough-draft`，结果为 `Skill is valid!`。

### 2026-03-25
- 重写 `.harness/spec.md` 中 `Skill 2` 与 `Skill 3`：将 `Skill 2` 改名为 `Rough Draft`，目标从 skeleton 产出调整为“先写一版材料完整的 lecture 粗草稿”；将 `Skill 3` 改为默认基于该粗草稿完成重写与教学化 revise。
- 更新 `.harness/spec.md` 中 `Workflow Boundary`：明确 `Skill 2` 负责把应保留材料真正写进粗草稿，`Skill 3` 默认不再重新通读参考资料，仅在来源冲突、证据不足或材料缺失时回源。
- 运行 `git diff -- .harness/spec.md .harness/progress.md` 作为本次规范修改的范围核对。

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
