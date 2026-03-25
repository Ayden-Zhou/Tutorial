---
name: rough-draft
description: Write a project-specific rough lecture draft from an existing lecture plan plus reference materials. Use when the user wants a material-complete but not yet polished draft for one lecture or one named section/subsection, especially when preserving source insights, examples, mechanism explanations, and source anchors matters more than final prose quality.
---

# Rough Draft

## Overview

Write a material-complete rough draft for this tutorial project from a settled lecture plan and the provided references. Preserve high-value explanations, insights, examples, and mechanism descriptions in draft form so later revision can focus on reorganization and teaching quality rather than re-reading source files.

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
- Read every reference file the user provides for the requested scope.
- Reopen adjacent sections from the existing draft when section-scoped work depends on them for continuity.
- Build a compact source map while reading:
  - which explanations are worth preserving,
  - which examples or counterexamples are worth carrying forward,
  - which mechanism descriptions are essential,
  - which passages are low-value repetition.
- Prefer extracting insight and explanation paths over extracting isolated facts.

3. Decide what must survive into the rough draft.
- Preserve the material that Skill 3 would be unhappy to lose: good motivations, crisp intuitions, informative examples, strong bridge sentences, and compact mechanism explanations.
- Preserve enough redundancy that later revision can cut safely.
- Compress low-value repetition, notation boilerplate, and exhaustive detail that does not help the teaching path.
- Keep source anchors for important claims, explanations, examples, and especially for any wording or framing that may need to be revisited.
- Distinguish source-derived material from your own connective tissue.

4. Draft directly in markdown.
- Write continuous prose, not just headings and bullets.
- Follow [references/draft-format.md](references/draft-format.md) for output shape, heading rules, source-anchor blocks, and section-scoped handoff notes.
- Keep the draft clearly rough: it may be redundant, uneven in tone, or locally awkward, but it should already contain the important material.
- Keep the lecture in Chinese by default, except for terms that are genuinely clearer in English.
- Write bridge sentences when needed so later section-by-section revision still has visible continuity.
- If the user requests section-scoped mode, modify only the requested section area and preserve untouched neighboring sections unless the user explicitly asks for broader edits.

5. Leave revise guidance for the next step.
- Mark places where the draft is structurally weak, repetitive, too abstract, or missing a good local bridge.
- Point out where Skill 3 should decide between multiple retained explanations or examples.
- Point out any unresolved conflicts across sources.
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
- Keep formulas and formal definitions only when they support the mechanism being explained in this scope.
- When several sources cover the same point, keep the version with the strongest teaching value and record the others as support if still useful.
- When a source has a genuinely strong turn of explanation, preserve the idea and anchor it, even if the final prose will later be rewritten.

## Boundaries

- Produce a rough draft, not a polished lecture.
- Do not downgrade into a skeleton, bullet list, or material dump.
- Do not copy long stretches of source wording.
- Do not decide final demo-code insertion in this step.
- Do not broaden the lecture beyond the lecture plan unless the user explicitly asks for an expansion.
- Do not silently drop a high-value example or insight just to make the draft sound smoother.

## Resource

- `references/draft-format.md`: output shape, source-anchor conventions, and section-scoped handoff format.
