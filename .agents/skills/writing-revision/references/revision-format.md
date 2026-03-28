# Revision Format

Use this reference when rewriting rough lecture drafts for this project.

## Output Modes

Choose one mode before revising.

### Full-Lecture Mode

Use when the user wants the whole lecture revised.

Recommended shape:

```md
# 讲义标题

## 1 节标题

这里开始写修订后的正文。正文应直接面向读者，而不是面向协作者；也不要写成 `本节目标` 之类的脚手架说明。

### 1.1 小节标题

继续写正文……

<!-- revise-note: 这一节仍然缺一个更短的反例 -->
```

### Section-Scoped Mode

Use when the user wants only one named `##` or `###` block revised.

Recommended shape:

````md
## 3 某个 section 标题

修订后的正文……

<div align="center">
  <img src="../../../references/01_Deep_learning/.../images/double_descent.png" width="50%">
  <br>
  double descent 示意图
</div>

这里直接告诉读者应该观察什么，而不是只把图丢在这里。

```python
# code block placeholder: 这里后续要用一个最小例子展示经验风险下降但测试误差不一定同步下降
```

<!-- handoff: 下一节需要把这里的经验现象接到一般化界的直觉 -->
````

## Placeholder Rules

Use placeholders only when they genuinely help preserve the teaching path.

### Code Block Placeholder

Use a fenced code block with a brief comment describing what the future demo should reveal.

Example:

```python
# code block placeholder: 这里后续要展示 width 增大后分段线性拐点数如何变化
```

### Picture Placeholder

Use a short blockquote or HTML comment only when a figure is clearly needed, but the rough draft did not preserve one and you cannot safely reconstruct it from the current draft.

Examples:

```md
> picture placeholder: 这里需要一张对比图，展示随机标签与真实标签训练时训练/测试误差曲线的差异
```

```md
<!-- picture placeholder: 这里后续补一张 sawtooth composition 的示意图 -->
```

## Comment Rules

Use short HTML comments only for collaborator-facing notes.

Examples:
- `<!-- revise-note: 这里已经顺过语气，但还缺一个最小反例 -->`
- `<!-- handoff: 下一节默认承接这一节已经解释过的 bias-variance tension -->`

Do not leave these notes in normal prose paragraphs.

## Source Rules

- Do not add `主要参考` blocks to the lecture body.
- Remove rough-draft `主要参考` blocks and similar section-goal scaffolding during revision unless the user explicitly wants to preserve them.
- Keep source information only when it has real revision value.
- Prefer HTML comments or separate revision notes over reader-facing reference lists.

## Image Rules

- Prefer revising existing image blocks over replacing them with placeholders.
- Convert image blocks to the centered HTML figure format when you revise them, even if the rough draft used a different wrapper.
- Keep each image close to the paragraph or subsection it supports.
- Tighten captions and lead-in sentences so the reader knows what to observe.
- Remove an image only when it is redundant, misleading, or no longer fits the revised teaching path.
- If an image still needs later review, leave a short HTML comment after it instead of turning the section back into a picture placeholder.

## Voice Rules

- Keep the body in tutorial-author voice.
- Prefer `我们先看`, `下面来看`, `这里可以先记住`.
- Avoid phrases like `这里需要补充`, `后续可以考虑`, `建议 agent`, `this section should`.
- Avoid scaffold phrases like `本节目标`, `这一节真正想说明的是`, `这一段想留下的结论是`.
- Avoid source-aware phrases like `参考材料里`, `课件上`, `slides 这里`.

## Parallel-Friendly Rules

When revising only one section for later merge:
- Keep scope boundaries explicit.
- Do not rewrite distant sections without a direct reason.
- Match nearby terminology when possible.
- Put merge-sensitive concerns into `handoff` comments rather than broad rewrites.
