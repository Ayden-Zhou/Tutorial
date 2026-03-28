# Output Format

Use this reference when writing the lecture framework file.

## Heading Rules

- Use `# <lecture title>` for the lecture title.
- Write the H1 title in Chinese.
- Do not include the lecture number in H1.
- Use `## <n> <section name>` for section titles.
- Use `### <n.m> <subsection name>` for subsection titles.
- Use `#### <n.m.k> <subsubsection name>` only when further splitting is genuinely useful.
- Keep numbering contiguous and update all later headings after structural edits.

## Title Naming Rules

**`##` 级标题（section）**
- 短名词短语或动名词短语，通常 2–5 个词。
- 不用冒号。不用疑问句（"为何/如何/为什么"句式）。
- 命名这一节"是什么内容"，不陈述"这节的论点或结论"。

**`###` 级标题（subsection）**
- 短名词短语，通常 2–4 个词。
- 如果用冒号，冒号后只能跟一个短限定词（一个名词短语），不能是完整解释或句子。
- 不用疑问句。不用"XX 说明了什么"、"XX 为何/如何 YY"等动宾展开句式。
- **冒号检查**：写完标题后，如果含冒号，删掉冒号后半段，看剩下的部分能否独立表达主题。能独立则删掉后半段；不能独立则重写整个标题为纯名词短语。

**通用规则**
- 名称来自教学任务，不来自源文件措辞。
- 避免口语化开头：`我们到底在问什么`、`为什么已经有启发`、`还能往哪里找解释`。
- 兄弟标题尽量保持语法一致。
- 优先简洁中文，不照搬教材术语。
- 有高价值直觉解释时，提升为独立标题，但不靠增加层级来凑对称。

Good section titles:
- `## 1 逼近与泛化：问题的区分`（H1 级允许冒号，冒号后是短限定词）
- `## 4 收缩搜索空间的三种工具`
- `## 6 经典复杂度理论的边界`
- `### 4.3 多个插值解并存`
- `### 6.1 U 型图景的隐含前提`
- `### 7.5 优化过程作为先验`

Weak section titles (do not use):
- `## 5 泛化视角：数据、先验与假设空间如何缩小搜索`（冒号后是完整解释句）
- `## 6 泛化视角：经典复杂度图景为何不足以解释深网`（疑问式展开）
- `### 5.3 假设空间作为硬约束：搜索更快，但可能把真解排除在外`（冒号后是结论句）
- `### 7.2 Transformer 的结构偏置为何常被低估`（疑问句式）
- `## 1 我们到底在学什么`（口语化疑问）

## Section Block Format

Record the main supporting references at the `##` section level, using exact source heading anchors.

- For each reference, include:
  - the relative path to the source file,
  - the exact source heading(s) or heading range that support this section,
  - a short phrase describing what that source contributes.
- Prefer source headings over synthetic page ranges. If the heading already contains a page label such as `### 24 title`, cite that heading text directly.
- If a reference contributes to multiple sections, record it separately under each with the relevant heading anchors and role.

Use this pattern:

```md
## 1 AI 任务与学习问题的转化

本节目标：
- 让读者先把”学习问题”理解成从数据中拟合输入到输出的映射。

主要参考：
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/1_intro_2025.md`：`## 监督学习问题`，`### 输入与输出`；用于建立任务抽象
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/lec-5.md`：`## 为什么需要学习`；用于补充“规则难以手写”的动机

### 1.1 输入输出映射的基本形式

### 1.2 数据、标签与目标
```

## Subsection Block Format

Record a lightweight source anchor at the `###` subsection level by default.

- Use `小节参考：` directly under the `###` heading.
- At this level, keep the anchor minimal:
  - relative path to the source file,
  - exact source heading(s) that mainly support this subsection.
- Do not repeat a long “用于……” explanation at every `###` unless the subsection truly mixes multiple sources and the distinction matters for later drafting.
- When one `###` clearly comes from one main source idea, a single anchor line is preferred.

Use this pattern:

```md
### 7.2 Transformer 的结构偏置

小节参考：
- `../../../references/01_Deep_learning/02_Approximation_and_Generalization/generalization.md`：`### 40 Idea #1: Architectural symmetries`
```

When a subsection genuinely merges two sources, use the lightest two-line form that still disambiguates:

```md
### 4.3 多个插值解并存

小节参考：
- `../../../references/.../generalization_problem.md`：`### 11.2.2 以多项式回归为视角理解泛化`
- `../../../references/.../generalization.md`：`### 11 这两种都”拟合了数据”`
```

Keep `小节参考` lightweight. It is a drafting anchor, not a second `主要参考` block.

## Relative Path Rule

- Write reference paths relative to the output Markdown file, not relative to the repository root.
- Prefer full relative paths over bare filenames so later drafting steps can reopen the exact sources without guessing.
- Recompute the relative path if the output file location changes.

Example:
- Output file: `docs/core/01_Deep_Learning/01_Deep_Learning_Foundations.md`
- Reference file: `references/01_Deep_learning/01_Deep_learing_Foundatiosn/1_intro_2025.pdf`
- Recorded path: `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/1_intro_2025.pdf`

## Scope Reminder

- Keep the document at the framework level.
- Use short bullets for `本节目标` or `取舍说明` only when they help future drafting.
- Keep `##`-level references richer than `###`-level references: section blocks explain the section spine, subsection blocks only point drafting to the right source passage.
- Do not write full explanatory paragraphs in this step.
