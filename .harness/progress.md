# Progress

## Current Status
### 主要变动
- 将项目内 skill 目录统一整理到 `./.agents/skills/`。
- 新建并完成 `lecture-curation` skill，明确了单 lecture 结构策划的工作流、标题命名规则、section 级主要参考记录格式，以及“按教学价值抽取结构而不是摘抄原文”的原则。
- 修复了 `commit-progress` 的 frontmatter 描述，使其能够通过当前 skill 校验脚本。

### 未完成
- 尚未用真实 lecture 任务对 `lecture-curation` 做一次前向试跑。

### 当前 blocker
- 无。

## Message to Human
- 无

## History Logs
### 2026-03-25
- 将 `.agents/commit-progress` 和 `.agents/lecture-curation` 移动到 `.agents/skills/`。
- 完成 `.agents/skills/lecture-curation/SKILL.md` 与 `.agents/skills/lecture-curation/references/output-format.md`。
- 运行 `python3 /home/zhouyf/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/lecture-curation`，结果为 `Skill is valid!`。
- 运行 `python3 /home/zhouyf/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/commit-progress`，发现 frontmatter 描述含 angle brackets；做最小修复后再次校验，结果为 `Skill is valid!`。
