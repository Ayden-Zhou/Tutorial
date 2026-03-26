# Draft Format

Use this reference when writing rough lecture drafts for this project.

## Output Modes

Choose one of these modes before writing.

### Full-Lecture Mode

Use when the user wants the whole lecture drafted.

Recommended shape:

```md
# 讲义标题

## 1 节标题

主要参考：
- `relative/path/to/ref_a.pdf`
- `relative/path/to/ref_b.pdf`

这里开始写连续粗草稿正文……

关键公式 / 机制骨架（按需保留）：
- 行内公式用 `$y = Wx + b$`
- 独立公式可写成：

  $$
  L(\theta) = ...
  $$
- 这里用 1-3 行说明符号含义、公式在本节里负责什么机制

### 1.1 小节标题

继续写正文……

<!-- revise-note: 这里保留了两个例子，Skill 3 决定保留哪个 -->
```

### Section-Scoped Mode

Use when the user wants only one named `##` or `###` block drafted.

Recommended shape:

```md
## 3 某个 section 标题

主要参考：
- `relative/path/to/ref_a.pdf`
- `relative/path/to/ref_c.pdf`

正文……

关键公式 / 机制骨架（按需保留）：
- 行内公式示例：`$...$`
- 独立公式或局部推导骨架示例：`$$...$$`
- 这里用最小公式集、符号表或局部推导骨架保住本节的技术核心

<!-- handoff: 默认承接上一节已经解释过的经验风险最小化；下一节可能需要补一小段从泛化现象过渡到理论解释的桥接句 -->

<!-- revise-note: 这一段保留了两个不同来源的机制解释，后续需要合并 -->
```

## Heading Rules

- Follow the lecture plan's structure.
- Keep the H1 title in Chinese.
- Do not include the lecture number in H1.
- Use `## <n> <section name>` for section titles.
- Use `### <n.m> <subsection name>` for subsection titles.
- Keep numbering contiguous only when you are drafting the full lecture or the user explicitly asks for renumbering.
- In section-scoped mode, do not renumber untouched parts of the document.
- When a section's core content depends on formulas or symbolic definitions, keep a small `关键公式 / 机制骨架` block or weave the same material directly into the prose.
- The goal is not formal completeness; the goal is to preserve enough equations, symbols, and local computation steps that later revision does not need to reconstruct the technical core from scratch.
- Use `$...$` for inline math and `$$...$$` for display math. Do not use backticks as the default formula container; backticks are for literal code-style tokens.

## Source Anchors

Record the main supporting references at the section level, then add local anchors only where they will help later revision.

Use these rules:
- Prefer relative paths from the output Markdown file.
- Keep a short `主要参考` block near the top of each drafted `##` section.
- Add inline source anchors only for material that is likely to need checking, merging, or selective preservation.
- Prefer compact anchor forms such as `来源：ref_a p.12` or `来源：ref_b sec.3.1` inside HTML comments or short parenthetical notes.
- Do not clutter every paragraph with anchors if a section-level block is enough.

Examples:
- `<!-- source: ../../../references/.../3_approximation_2025.pdf p.14 -->`
- `（来源：../../../references/.../7_generalization_2025.pdf sec.2）`

## Technical Core

When rough-drafting technical sections:
- Preserve the minimum formula set needed to keep the section mechanically correct.
- Name the symbols that the reader must track locally.
- Keep short derivation skeletons or update equations when they carry the actual mechanism.
- Do not turn equations into vague prose if the equation is the cleanest statement of the idea.
- If a formula should visually render as mathematics in the lecture, write it in math delimiters rather than code spans.

## Revise Notes

Use short HTML comments for local revise guidance.

Examples:
- `<!-- revise-note: 这里定义先于直觉，Skill 3 需要重排 -->`
- `<!-- revise-note: 这一段信息全，但语气很像资料摘录 -->`
- `<!-- handoff: 下一节需要把这里的经验现象接到 margin-based intuition -->`

## Parallel-Friendly Rules

When drafting only part of a lecture for later merge:
- Keep scope boundaries visible.
- Avoid editing global intro or outro text unless requested.
- Keep terminology choices consistent with the nearby existing draft when possible.
- Make merge-sensitive assumptions explicit in `handoff` comments instead of changing distant sections.
- Preserve useful redundancy inside the scoped section; deduplication belongs mostly to revision.
