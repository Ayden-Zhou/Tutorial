---
name: writing-revision
description: Revise a project-specific lecture rough draft into clearer, more teachable tutorial prose. Use when one lecture or one named section/subsection already has a material-complete rough draft and needs local restructuring, deduplication, bridge sentences inside the lecture, author-voice cleanup, or placeholder planning for later demos/pictures without redoing large-scale source extraction.
---

# Writing Revision

Revise an existing rough lecture draft into a clearer teaching draft. Treat revision as selection plus rewriting: preserve the draft's strongest insight, examples, and mechanism explanations, cut weak repetition and scaffolding, and reorganize the material around a reader-facing teaching path.

## Scope

- Work on one lecture only.
- Resolve the target markdown file before revising.
- Support two modes: full-lecture or section-scoped (`##` / `###` only).
- In section-scoped mode, keep neighboring sections untouched unless the user explicitly asks for broader rewriting.
- If there is no rough draft yet, stop and hand the task back to `$rough-draft`.

## Read Order

- Read the current target file first.
- In section-scoped mode, also read the parent heading and adjacent sibling headings so local bridges and terminology stay consistent.
- Treat the rough draft as the main working artifact.
- Reopen original references only when needed for correctness:
  - the draft contains conflicting claims you cannot resolve locally,
  - a key claim looks unsupported,
  - a necessary explanation is clearly missing,
  - an existing image or caption looks misleading enough that you cannot safely fix it from the draft alone.
- If the draft is thin overall, flag that gap instead of silently re-extracting the whole topic from sources.

## Revision Rules

- Preserve the rough draft's highest-value insight, example, bridge, and compact mechanism explanation.
- Remove redundancy, flat paraphrase, and reader-facing drafting scaffolding such as `本节目标`, `主要参考`, and similar process labels.
- When the draft contains parallel versions of the same explanation, keep the version with the strongest teaching value and delete the rest instead of merging everything together.
- Preserve failure cases, paradoxes, and open questions that carry real intellectual content. Do not smooth them away just to make the narrative cleaner.
- Reorder aggressively when needed. Prefer a better teaching path over rough-draft order or source order.
- Use the default progression `problem or phenomenon -> mechanism intuition -> necessary abstraction -> verification, implication, or application`.
- Introduce definitions, notation, and formulas only after local intuition is established. Prefer prose when it explains the point just as well.
- Write in tutorial-author voice for learners, in Chinese by default. Do not leave agent-facing prose, source-aware narration, or editorial commentary in the lecture body.
- In section-scoped mode, do not silently rewrite the lecture-wide introduction, conclusion, or distant sections. If you discover a global inconsistency, leave a short note instead.

## Figures, Placeholders, Handoff

- Bind abstract explanation to at least one concrete support: a minimal example, a counterexample, a figure description, an existing image block, or a `code block placeholder`.
- Treat images already present in the draft as first-class teaching material. Tighten their lead-in, placement, and caption before using a placeholder.
- Use a picture placeholder only when a figure is clearly necessary and you cannot safely reconstruct one from the current draft.
- Suggest future demo needs, but do not write final teaching code or decide final visualization content in this step.
- If a local gap remains after revision, leave a short `revise-note` or `handoff` comment. Keep collaborator-facing notes short and out of normal prose.

## Boundaries

- Produce a revised teaching draft, not a final polished publication.
- Do not re-run large-scale source extraction unless the draft is insufficient for correctness.
- Do not take ownership of lecture-wide previous-lecture continuity or whole-lecture integration in this skill; hand those tasks to `lecture-revision`.
- Do not replace an existing useful image with a placeholder just because the surrounding prose still needs work.

## Final Check

- Each revised section starts from a question, phenomenon, or intuition before heavy abstraction.
- Repeated points and drafting scaffolding are gone.
- The strongest insight, example, and failure case still survive.
- Existing images stay attached to the right paragraph, or any placeholder is explicit and local.

## Resource

- `references/revision-format.md`: output shape, placeholder conventions, comment style, and image-format rules.
