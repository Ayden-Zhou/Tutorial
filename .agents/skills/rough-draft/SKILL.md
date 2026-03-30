---
name: rough-draft
description: Write a project-specific rough lecture draft from an existing lecture plan plus reference materials. Use when the user wants a material-complete but not yet polished draft for one lecture or one named section/subsection, especially when preserving source insights, examples, mechanism explanations, formulas, and source anchors matters more than final prose quality. For multi-subagent whole-lecture drafting or final merge of parallel section drafts, use `lecture-drafting` instead.
---

# Rough Draft

Translate reference materials into a comprehensive Chinese rough draft guided by the lecture plan. Optimize for coverage, not selection: preserve all in-scope material faithfully so later revision can decide what survives.

## Scope

- Work on one lecture only.
- Resolve the target output file before drafting.
- Support two modes:
  - full-lecture mode: write the rough draft for the whole lecture.
  - section-scoped mode: write only the named section or subsection the user specifies.
- If the target file already exists, read it first and treat it as the current draft instead of overwriting blindly.
- If the user does not identify a lecture plan, stop and ask for it.
- If the user asks for section-scoped drafting, stop and confirm the exact section boundary only when it cannot be inferred from the plan or existing draft.

## Read Order

- Read the lecture plan first and treat it as the scope boundary.
- Read every reference file the user provides for the requested scope. When the lecture plan records page ranges (`pp. X–Y`) for a reference, read those pages directly instead of scanning the whole file.
- Check whether the current lecture or topic folder has a nearby `images/` directory. When it exists, treat it as part of the source materials for the same scope rather than as a later optional add-on.
- Reopen adjacent sections from the existing draft when section-scoped work depends on them for continuity.
- While reading, note everything within scope that should survive into the rough draft: explanations, examples, counterexamples, formulas, symbols, derivation skeletons, mechanism descriptions, algorithmic steps, computation chains, and images with clear teaching value.
- Do not judge what is low-value at this step. If in doubt, keep it.

## Drafting Rules

- Translate all in-scope content: motivations, intuitions, examples, counterexamples, mechanism explanations, formulas, symbol definitions, update equations, objective functions, derivation skeletons, and algorithmic steps.
- When multiple sources cover the same point differently, preserve all materially distinct versions side by side. Revision will choose.
- Write continuous prose, not just headings and bullets.
- Follow [references/draft-format.md](references/draft-format.md) for output shape, heading rules, source-anchor blocks, and section-scoped handoff notes.
- Keep the draft clearly rough: it may be redundant, uneven in tone, or locally awkward, but it should already contain the important material.
- Keep the lecture in Chinese by default, except for terms that are genuinely clearer in English.
- Write bridge sentences when needed so later section-by-section revision still has visible continuity.
- Keep the technical core explicit. When a section depends on formulas, symbolic definitions, update equations, or short derivation chains, include them together with a brief explanation of what each symbol or step is doing.
- Use standard Markdown math delimiters for formulas that should render as mathematics: inline math uses `$...$`, display equations or short derivation blocks use `$$...$$`. Do not wrap formulas in backticks unless you are referring to the literal source text or showing non-rendering pseudocode.
- Reserve backticks for code, file paths, API names, config keys, and other literal identifiers rather than mathematical expressions.
- When a nearby image clearly helps preserve geometric intuition, trend judgment, structural impression, or an experimental picture already being discussed, insert it now. First-time image insertion belongs to rough drafting.
- Normalize inserted images into centered HTML figure blocks with explicit captions. Keep them near the paragraph they support.
- Do not insert an image just because a file exists. Skip images that merely repeat the prose, need long detours to explain, or do not clearly belong to the local teaching path.
- If several nearby images express the same point, keep the most direct one.
- Keep source anchors for translated claims, examples, and formulas when later revision is likely to need traceability.
- Keep image paths traceable and relative to the target markdown file. When practical, point to the lecture's intended final location under `images/`.
- Keep section-level rough-draft scaffolding such as `主要参考` and short `handoff` / `revise-note` comments when they help later merge or revision.
- Leave short local notes for structural weakness, unresolved source conflicts, retained parallel explanations, or images that may need relocation or caption tightening.

## Section-Scoped Mode

- Draft only the requested section or subsection.
- Preserve the heading structure already established by the plan or existing draft.
- Do not renumber untouched sibling sections unless the user explicitly asks for a structural rewrite.
- Read the immediate parent section and adjacent sibling headings before drafting so the local section can connect cleanly.
- Add a short handoff note when useful: what this section assumes, what the next section may build on, or where later merge work may need a bridge.
- If mergeability matters, keep scope boundaries explicit, avoid rewriting global intro or conclusion text, and avoid changing terminology outside the requested section.

## Boundaries

- Produce a rough draft, not a polished lecture.
- Do not take ownership of lecture-wide parallel drafting orchestration or final merge; hand those tasks to `lecture-drafting`.
- Do not downgrade into a skeleton, bullet list, or material dump.
- Translate faithfully; the goal at this step is coverage, not prose quality.
- Do not decide final demo-code insertion in this step.
- Do not generate new figures or write code just to make a picture; only use existing images that already belong to the current scope.
- You may make the minimum path adjustment needed for an inserted image to point at its intended final location under `images/`, but do not expand the task into large-scale image asset migration.
- Do not broaden the lecture beyond the lecture plan unless the user explicitly asks for an expansion.
- Do not silently replace a section's technical core with intuition-only prose when the source relies on formulas, symbolic definitions, or explicit computation steps.

## Final Check

- The draft stays inside the lecture plan's scope and does not silently select only the polished-looking material.
- The technical core survives in explicit form where the source depends on formulas, symbols, or computation steps.
- Useful images from the current scope are inserted only when they materially support the local teaching path.
- The draft remains mergeable: headings, numbering, terminology, and handoff notes still support later revision or integration.

## Resource

- `references/draft-format.md`: output shape, source-anchor conventions, and section-scoped handoff format.
