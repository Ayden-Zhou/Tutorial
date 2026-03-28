---
name: writing-revision
description: Revise a project-specific lecture rough draft into clearer, more teachable tutorial prose. Use when one lecture or one named section/subsection already has a material-complete rough draft and needs local restructuring, deduplication, bridge sentences inside the lecture, author-voice cleanup, or placeholder planning for later demos/pictures without redoing large-scale source extraction. For multi-subagent lecture-wide revision, previous-lecture continuity, or final whole-lecture integration, use `lecture-revision` instead.
---

# Writing Revision

## Overview

Rewrite an existing rough lecture draft for this tutorial project into a clearer teaching draft. Preserve the draft's strongest insights, examples, and mechanism explanations while improving reader-facing flow, author voice, and local coherence.

Treat revision as a material-selection step as well as a prose rewrite: choose what survives, what gets cut, and how the content is reorganized around a teaching path rather than a source inventory.

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
- The rough draft is expected to be materially complete. Reopen original references only when one of these is true:
  - sources conflict in a way that cannot be resolved from the draft alone,
  - a key claim looks unsupported and the gap would mislead the reader,
  - an explanation critical for correctness is clearly absent from the draft,
  - an inserted image or caption appears misleading enough that it cannot be safely repaired from the draft alone.
- If the draft appears thin overall, flag it rather than silently patching it by re-extracting sources.

3. Decide what should survive the rewrite.
- Preserve high-value insight, good examples, crisp bridge sentences, and compact mechanism explanations from the rough draft.
- Remove redundancy, flat paraphrase, and paragraphs that explain only what something is without clarifying why it matters.
- When the rough draft contains parallel versions of the same explanation from multiple sources, select the one with the strongest teaching value and remove the others. Do not merge all versions together or keep them all for completeness.
- Preserve paradoxes, known failure cases, and open questions that appear in the draft. Do not smooth them over to make the narrative cleaner; they are part of the intellectual content and help the reader build accurate judgment.
- Prefer a stronger explanation path over preserving rough-draft order.
- Treat rough-draft scaffolding cleanup as part of revision: this is the default stage to remove reader-facing `本节目标`, `主要参考`, and other temporary drafting labels once their function has been absorbed into the prose.
- Do not keep section-level `主要参考` blocks in the lecture body.
- Keep source anchors only when they are useful for later checking or selective preservation, and prefer comments or separate revision notes over reader-facing reference blocks.
- Keep the lecture inside the existing lecture-plan boundary unless the user explicitly asks to expand it.

4. Rewrite for reader cognition, not source order.
- Follow the default movement:
  - problem or phenomenon -> mechanism intuition -> necessary abstraction -> verification, implication, or application
- Reorder paragraphs when that improves the teaching path.
- When multiple sources cover the same concept, organize the section around a conceptual framework first, then use individual sources or methods as examples or supporting evidence. Do not organize by source order.
- Introduce formal definitions, notation, or formulas only after the reader has a local intuition. A formula should make precise what was just explained in words, not introduce a new idea cold. If a formula would introduce something with no prior intuition in the section, build the intuition first.
- Prefer lower formula density when prose can carry the explanation equally well; formulas are support, not the main carrier of understanding.
- When a concept or mechanism has a clean analogy or a natural "what this really is" restatement, include it after the technical detail as a compression sentence. Do not force analogies where none fit naturally.
- Use minimal bridging sentences between sections and subsection openings whenever the topic shift would otherwise feel abrupt.
- Treat section goals as internal scaffolding, not reader-facing prose.
- Keep the tutorial in Chinese by default, except for terms that are genuinely clearer in English.

5. Keep the voice inside the tutorial.
- Write from the tutorial author's point of view, not from the assistant's or editor's point of view.
- Write with varied sentence rhythm: state conclusions and key insights in short, direct sentences (under 20 characters); use medium-length sentences for mechanism explanation; reserve lists for pipelines, steps, or parallel alternatives. Avoid repeating a fixed set of transition phrases.
- Express evaluative judgment: mark which findings are significant (`值得注意的是`), which tradeoffs are critical (`这里真正的问题是`), and which claims remain uncertain (`目前还不完全清楚`). Do not report all content with equal weight.
- Do not leave agent-facing prose, exposed scaffold prose, or source-aware narration in the body. Rewrite useful source explanations as the lecture's own narration.
- Put collaborator-facing notes only in HTML comments or in a separate revision-notes artifact.

6. Handle examples, code, and figures carefully.
- Bind abstract explanation to at least one of:
  - a minimal example,
  - a counterexample,
  - a figure description,
  - an inserted image block,
  - a `code block placeholder`
- Treat image blocks that already exist in the rough draft as first-class teaching material. Tighten their lead-in, caption, placement, or surrounding prose before deciding that a placeholder is needed.
- Use a picture placeholder only as an exception when the section clearly needs a figure, the rough draft failed to preserve one, and you cannot safely reconstruct it from the current draft.
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
- If a global inconsistency is discovered, note it briefly instead of rewriting distant sections without permission.

## Revision Checklist

Before finishing, check each revised section against this list:

- Can you state in one sentence what problem this section is solving, without exposing that sentence as a `本节目标` label in the body?
- Does the section establish a question, phenomenon, example, or intuition before heavy abstraction?
- Are repeated points merged rather than restated?
- Is there an explicit bridge from the previous section when the topic changes?
- Did the rewrite preserve the rough draft's highest-value insight, example, or explanation path?
- If the draft contained parallel versions of the same concept from multiple sources, was only the strongest one kept?
- Does every formula appear after a prose intuition that motivates it, not before?
- Were paradoxes, failure cases, or open questions in the draft preserved rather than smoothed over?
- If the draft already contained an image, is it still attached to the right paragraph, and does the caption tell the reader what to look at?
- If code or a figure is still needed for understanding but not yet available, is there a clear local placeholder?
- Does the prose sound like a tutorial author speaking to learners rather than an agent talking to collaborators?
- Does the body avoid scaffold phrases like `本节`, `这一节`, or source-aware phrases like `参考材料里`, unless there is a truly unavoidable rhetorical reason?
- Do section headings name a phenomenon, problem, or insight — rather than just a topic label?

## Boundaries

- Produce a revised teaching draft, not a final polished publication.
- Do not re-run large-scale source extraction unless the draft is insufficient for correctness.
- Do not flatten the draft so aggressively that its strongest examples or insights disappear.
- Do not turn the lecture body into editorial commentary.
- Do treat revision as the default stage for deleting rough-draft goal/reference scaffolding, unless the user explicitly asks to keep it.
- Do not take ownership of lecture-wide previous-lecture continuity or whole-lecture integration in this skill; hand those tasks to `lecture-revision`.
- Do not insert final demo code blocks or decide final visualization content in this step.
- Do not replace an existing useful image with a placeholder just because the surrounding prose still needs work.

## Resource

- `references/revision-format.md`: placeholder conventions, comment style, and full-lecture vs section-scoped output guidance.
