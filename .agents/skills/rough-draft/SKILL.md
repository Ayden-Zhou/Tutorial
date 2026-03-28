---
name: rough-draft
description: Write a project-specific rough lecture draft from an existing lecture plan plus reference materials. Use when the user wants a material-complete but not yet polished draft for one lecture or one named section/subsection, especially when preserving source insights, examples, mechanism explanations, formulas, and source anchors matters more than final prose quality. For multi-subagent whole-lecture drafting or final merge of parallel section drafts, use `lecture-drafting` instead.
---

# Rough Draft

## Overview

Translate reference materials into a comprehensive Chinese rough draft for this tutorial project, guided by the lecture plan. The goal at this step is coverage, not selection: preserve all in-scope material faithfully so later revision can choose what survives.

## Workflow

1. Confirm the drafting scope.
- Work on one lecture only.
- Resolve the target output file before drafting.
- Distinguish between two modes:
  - full-lecture mode: write the rough draft for the whole lecture.
  - section-scoped mode: write only the named section or subsection the user specifies.
- If the target file already exists, read it first and treat it as the current draft instead of overwriting blindly.
- If the user does not identify a lecture plan, stop and ask for it.
- If the user asks for section-scoped drafting, stop and confirm the exact section boundary only when it cannot be inferred from the plan or existing draft.

2. Read the planning artifact and references.
- Read the lecture plan first and treat it as the scope boundary.
- Read every reference file the user provides for the requested scope. When the lecture plan records page ranges (`pp. X–Y`) for a reference, read those pages directly instead of scanning the whole file.
- Check whether the current lecture or topic folder has a nearby `images/` directory. When it exists, treat it as part of the source materials for the same scope rather than as a later optional add-on.
- Reopen adjacent sections from the existing draft when section-scoped work depends on them for continuity.
- While reading, note everything within the lecture plan's scope that should be translated or preserved: explanations, examples, counterexamples, formulas, symbols, derivation skeletons, mechanism descriptions, algorithmic steps, computation chains, and images with clear teaching value.
- Do not judge what is "low-value" at this step. If in doubt, translate it.

3. Translate comprehensively.
- Translate all content within the lecture plan's scope: motivations, intuitions, examples, counterexamples, mechanism explanations, formulas, symbol definitions, update equations, objective functions, derivation skeletons, and algorithmic steps.
- When multiple sources cover the same point differently, translate all versions and keep them side by side. Writing revision will choose.
- When a nearby image can clearly help the reader build geometric intuition, trend judgment, structural impression, or an experimental picture that the current section is already discussing, insert it directly into the rough draft with a short lead-in or caption. This is the default stage for first-time image insertion.
- Normalize every inserted image into a centered HTML figure block with an explicit caption; do not flatten it into a bare Markdown image.
- Do not insert an image just because a file exists. Skip images that merely repeat the prose, are hard to understand without long detours, or do not clearly belong to the current section.
- When several nearby images express the same point, keep the one that is most direct and most tightly coupled to the local teaching path.
- Do not compress, summarize, or skip in-scope content at this step. If an explanation seems repetitive, translate it anyway.
- Keep source anchors for all translated claims, examples, and formulas so writing revision can trace back if needed.
- Keep image paths traceable. Use relative paths from the target markdown file and add only the minimum caption or observation hint needed for later revision. When possible, converge the reference to the lecture's final image location under the repository `images/` directory instead of leaving it on a temporary source path.
- Keep section-level rough-draft scaffolding such as `本节目标`, `主要参考`, and short `handoff` / `revise-note` comments when they still help later merge or revision. Do not strip them here just to make the draft look finished; reader-facing cleanup belongs to revision.
- Only skip content that is clearly outside the lecture plan's scope.

4. Draft directly in markdown.
- Write continuous prose, not just headings and bullets.
- Follow [references/draft-format.md](references/draft-format.md) for output shape, heading rules, source-anchor blocks, and section-scoped handoff notes.
- Keep the draft clearly rough: it may be redundant, uneven in tone, or locally awkward, but it should already contain the important material.
- Keep the lecture in Chinese by default, except for terms that are genuinely clearer in English.
- Write bridge sentences when needed so later section-by-section revision still has visible continuity.
- When a section depends on formulas or symbolic definitions, include them in the rough draft together with a short explanation of what each symbol or computation step is doing.
- When a section depends on an existing image for intuition, place the image block near the paragraph it supports instead of deferring it to a later curation stage. Do the initial image placement here rather than inserting the figure first and expecting revision to relocate it later.
- Use standard Markdown math delimiters for formulas that should render as mathematics: inline math uses `$...$`, display equations or short derivation blocks use `$$...$$`. Do not wrap formulas in backticks unless you are referring to the literal source text or showing non-rendering pseudocode.
- Reserve backticks for code, file paths, API names, config keys, and other literal identifiers rather than mathematical expressions.
- When a section depends on a mechanism chain, keep the chain explicit rather than replacing it with intuition-only prose.

5. Leave revise guidance for the next step.
- Mark places where the draft is structurally weak, repetitive, too abstract, or missing a good local bridge.
- Point out where later writing revision should decide between multiple retained explanations or examples.
- Point out any unresolved conflicts across sources.
- Point out when an inserted image is useful but may need relocation, caption tightening, or comparison against a nearby alternative.
- Keep these notes short and local so they help revision without turning the draft into commentary.

## Section-Scoped Drafting

Treat section-scoped drafting as a first-class mode, not as an afterthought.

- Draft only the requested section or subsection.
- Preserve the heading structure already established by the plan or existing draft.
- Do not renumber untouched sibling sections unless the user explicitly asks for a structural rewrite.
- Read the immediate parent section and adjacent sibling headings before drafting so the local section can connect cleanly.
- Add a short handoff note when useful:
  - what this section assumes from the previous section,
  - what the next section will likely build on,
  - where a future merge may need a transition sentence.
- If multiple subagents could draft different sections in parallel, optimize for mergeability:
  - keep scope boundaries explicit,
  - avoid rewriting global introduction or conclusion text unless asked,
  - avoid changing terminology choices outside the requested section,
  - make assumptions visible instead of silently normalizing the whole document.

## Writing Heuristics

- Prefer material that answers why a concept exists, what problem it solves, or how a reader should build intuition for it.
- Prefer preserving a good explanation path over preserving source order.
- Preserve source-specific insight even when the current wording is not final.
- Keep formulas and formal definitions whenever they are needed to make the section technically usable, not only when they add local flavor.
- Treat nearby `images/` resources as part of the section's technical material when they are the fastest way to preserve a phenomenon, shape, comparison, or structural intuition.
- Prefer rendered math over code styling for equations: use `$...$` and `$$...$$` consistently so later revision does not need to normalize formula formatting.
- Prefer a minimal but explicit technical core: objective function, key variable meanings, update equations, and short derivation skeletons beat prose-only summaries.
- If a mechanism is easiest to preserve as a short equation chain or algorithm sketch, keep that structure in the rough draft and explain it briefly in words.
- When a source has a genuinely strong turn of explanation, preserve the idea and anchor it, even if the final prose will later be rewritten.

## Boundaries

- Produce a rough draft, not a polished lecture.
- Do not delete section-goal or section-reference scaffolding merely for polish; those removals belong mostly to `writing-revision` / `lecture-revision`.
- Do not take ownership of lecture-wide parallel drafting orchestration or final merge; hand those tasks to `lecture-drafting`.
- Do not downgrade into a skeleton, bullet list, or material dump.
- Translate faithfully; the goal at this step is coverage, not prose quality.
- Do not decide final demo-code insertion in this step.
- Do not generate new figures or write code just to make a picture; only use existing images that already belong to the current scope.
- You may make the minimum path adjustment needed for an inserted image to point at its intended final location under `images/`, but do not expand the task into large-scale image asset migration.
- Do not broaden the lecture beyond the lecture plan unless the user explicitly asks for an expansion.
- Do not silently replace a section's technical core with intuition-only prose when the source relies on formulas, symbolic definitions, or explicit computation steps.

## Resource

- `references/draft-format.md`: output shape, source-anchor conventions, and section-scoped handoff format.
