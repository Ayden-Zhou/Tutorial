---
name: writing-revision
description: Revise a project-specific lecture rough draft into clearer, more teachable tutorial prose. Use when one lecture or one named section/subsection already has a material-complete rough draft and needs local restructuring, deduplication, bridge sentences inside the lecture, author-voice cleanup, or placeholder planning for later demos/pictures without redoing large-scale source extraction. For multi-subagent lecture-wide revision, previous-lecture continuity, or final whole-lecture integration, use `lecture-revision` instead.
---

# Writing Revision

## Overview

Rewrite an existing rough lecture draft for this tutorial project into a clearer teaching draft. Preserve the draft's strongest insights, examples, and mechanism explanations while improving reader-facing flow, author voice, and local coherence.

## Workflow

1. Confirm the revision scope.
- Work on one lecture only.
- Resolve the target file before revising.
- Distinguish between two modes:
  - full-lecture mode: revise the whole lecture draft.
  - section-scoped mode: revise only the named `##` or `###` block.
- If the user names only a section, preserve untouched neighboring sections unless the user explicitly asks for broader rewriting.
- If there is no rough draft yet, stop and hand the task back to `$rough-draft`.

2. Read the draft before making decisions.
- Read the current target markdown file first.
- In section-scoped mode, also read the parent heading and adjacent sibling headings so local bridges stay consistent.
- Treat the rough draft as the main working artifact.
- Reopen original references only when one of these is true:
  - sources conflict,
  - a key claim looks unsupported,
  - a necessary explanation is clearly missing from the draft.

3. Decide what should survive the rewrite.
- Preserve high-value insight, good examples, crisp bridge sentences, and compact mechanism explanations from the rough draft.
- Remove redundancy, flat paraphrase, and paragraphs that explain only what something is without clarifying why it matters.
- Prefer a stronger explanation path over preserving rough-draft order.
- Do not keep section-level `主要参考` blocks in the lecture body.
- Keep source anchors only when they are useful for later checking or selective preservation, and prefer comments or separate revision notes over reader-facing reference blocks.
- Keep the lecture inside the existing lecture-plan boundary unless the user explicitly asks to expand it.

4. Rewrite for reader cognition, not source order.
- Follow the default movement:
  - problem or phenomenon -> mechanism intuition -> necessary abstraction -> verification, implication, or application
- Reorder paragraphs when that improves the teaching path.
- Introduce formal definitions, notation, or formulas only after the reader has a local intuition.
- Use minimal bridging sentences between sections and subsection openings whenever the topic shift would otherwise feel abrupt.
- Treat section goals as internal scaffolding, not reader-facing prose.
- Keep the tutorial in Chinese by default, except for terms that are genuinely clearer in English.

5. Keep the voice inside the tutorial.
- Write from the tutorial author's point of view, not from the assistant's or editor's point of view.
- Prefer phrases like `我们先看`, `下面来看`, `这里可以看到`, `先别急着下定义`.
- Do not leave agent-facing prose in the body such as `这里需要补充`, `后续可考虑`, `建议模型`, or `TODO for revision`.
- Do not expose writing-scaffold prose in the body such as `本节目标`, `这一节真正想说明的是`, `这一段想留下的结论是`, or `下面我们将证明`.
- Write the lecture as self-contained text; do not mention reference materials, slides, or source files in the body. If a source has a good analogy or explanation, rewrite it as the lecture's own narration.
- Put collaborator-facing notes only in HTML comments or in a separate revision-notes artifact.

6. Handle examples, code, and figures carefully.
- Bind abstract explanation to at least one of:
  - a minimal example,
  - a counterexample,
  - a figure description,
  - a `picture placeholder`,
  - a `code block placeholder`
- If understanding clearly depends on a later demo or figure, leave a short placeholder in the draft instead of forcing premature implementation.
- Suggest demo needs, but do not decide final code-block insertion or write the final teaching code here.

7. Leave a clean handoff.
- If you create placeholders, make them explicit and local.
- If a section still has a gap after revision, leave a short `revise-note` or put it in revision notes.
- Keep handoff notes short; the lecture body should remain reader-facing.

## Section-Scoped Revision

Treat section-scoped revision as a first-class mode.

- Revise only the requested section or subsection.
- Preserve established heading structure and numbering unless the user explicitly asks for structural changes.
- Do not silently rewrite the lecture-wide introduction or conclusion from inside a section-scoped task.
- Keep terminology compatible with nearby untouched sections when possible.
- If a global inconsistency is discovered, note it briefly instead of rewriting distant sections without permission.

## Revision Checklist

Before finishing, check each revised section against this list:

- Can you state in one sentence what problem this section is solving, without exposing that sentence as a `本节目标` label in the body?
- Does the section establish a question, phenomenon, example, or intuition before heavy abstraction?
- Are repeated points merged rather than restated?
- Is there an explicit bridge from the previous section when the topic changes?
- Did the rewrite preserve the rough draft's highest-value insight, example, or explanation path?
- If code or a figure is needed for understanding but not yet implemented, is there a clear local placeholder?
- Does the prose sound like a tutorial author speaking to learners rather than an agent talking to collaborators?
- Does the body avoid scaffold phrases like `本节`, `这一节`, or source-aware phrases like `参考材料里`, unless there is a truly unavoidable rhetorical reason?

## Boundaries

- Produce a revised teaching draft, not a final polished publication.
- Do not re-run large-scale source extraction unless the draft is insufficient for correctness.
- Do not flatten the draft so aggressively that its strongest examples or insights disappear.
- Do not turn the lecture body into editorial commentary.
- Do not insert reader-facing `主要参考` sections into the lecture body.
- Do not expose section-goal scaffolding in the lecture body.
- Do not mention source materials in the lecture body.
- Do not take ownership of lecture-wide previous-lecture continuity or whole-lecture integration in this skill; hand those tasks to `lecture-revision`.
- Do not insert final demo code blocks or decide final visualization content in this step.

## Resource

- `references/revision-format.md`: placeholder conventions, comment style, and full-lecture vs section-scoped output guidance.
