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
- Prefer titles that answer a question, define a mechanism, or mark a clear learning step.
- Keep sibling headings grammatically consistent when possible.
- Prefer concise Chinese phrasing over translated textbook jargon.
- Promote an intuitive explanation into its own heading when it materially improves understanding.
- Do not create extra levels only for visual symmetry.

Good section titles:
- `## 1 从 AI 任务到学习问题`
- `## 4 训练闭环是怎么工作的`
- `### 4.3 反向传播：梯度从哪里来`

Weak section titles:
- `## 1 Introduction`
- `## 2 Definitions`
- `## 3 Other Topics`

## Section Block Format

Record the main supporting references at the `##` section level.

Use this pattern:

```md
## 1 从 AI 任务到学习问题

本节目标：
- 让读者先把“学习问题”理解成从数据中拟合输入到输出的映射。

主要参考：
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/1_intro_2025.pdf`
- `../../../references/01_Deep_learning/01_Deep_learing_Foundatiosn/lec-5.pdf`

### 1.1 什么叫“从输入到输出的映射”

### 1.2 数据、标签与目标
```

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
- Do not write full explanatory paragraphs in this step.
