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

- Name each heading after the teaching job it performs, not after the source file's wording.
- Default to lecture-note style phrasing rather than conversational prompts.
- Prefer concise noun phrases or mechanism/comparison/judgment forms such as `分箱逼近的基本思路`、`通用逼近的含义与边界`、`深度与宽度的表达效率`.
- Use question-style headings only when the question itself is pedagogically central and still reads naturally in a printed outline.
- Avoid casual heading openings such as `我们到底在问什么问题`、`为什么已经有启发`、`还能往哪里找解释`.
- Keep sibling headings grammatically consistent when possible.
- Prefer concise Chinese phrasing over translated textbook jargon.
- Promote an intuitive explanation into its own heading when it materially improves understanding.
- Do not create extra levels only for visual symmetry.

Good section titles:
- `## 1 AI 任务与学习问题的转化`
- `## 4 训练闭环的工作机制`
- `## 7 经典泛化理论及其局限`
- `### 4.3 反向传播的梯度来源`

Weak section titles:
- `## 1 我们到底在学什么`
- `## 2 为什么这个方法已经有启发`
- `## 3 还能往哪里找解释`
- `## 4 Other Topics`

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
### 7.2 Transformer 的结构偏置为何常被低估

小节参考：
- `../../../references/01_Deep_learning/02_Approximation_and_Generalization/generalization.md`：`### 40 Idea #1: Architectural symmetries`
```

When a subsection genuinely merges two sources, use the lightest two-line form that still disambiguates:

```md
### 4.3 多个插值解为何让泛化变得困难

小节参考：
- `../../../references/.../generalization_problem.md`：`### 11.2.2 以多项式回归为视角理解泛化`
- `../../../references/.../generalization.md`：`### 11 这两种都“拟合了数据”`
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
