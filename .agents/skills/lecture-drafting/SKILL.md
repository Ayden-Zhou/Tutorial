---
name: lecture-drafting
description: Coordinate a full rough lecture draft by splitting sections across multiple `rough-draft` workers, then merging them into one material-complete lecture draft. Use when one lecture is large enough to benefit from parallel section drafting, when references should be allocated section by section, or when one agent should own lecture-level mergeability constraints instead of making every rough-draft worker rewrite the whole lecture.
---

# Lecture Drafting

## Overview

Draft one whole lecture by orchestrating multiple section-level `rough-draft` workers and then merging their outputs into a single material-complete rough lecture draft. Own the lecture-level decisions that should not be duplicated across section workers: section boundaries, reference allocation, intro/outro ownership, source-anchor conventions, terminology consistency, and final merge cleanup.

## Workflow

1. Confirm the orchestration scope.
- Work on one lecture only.
- Use this skill only when the task is whole-lecture or clearly multi-section.
- If the user wants only one section or subsection drafted, hand the task to `$rough-draft` instead of orchestrating workers.
- Resolve the target lecture file before doing anything else.
- If the target file already exists, read it first and treat it as the current draft state.

2. Build the lecture-level drafting context.
- Read the lecture plan first and treat it as the primary scope boundary.
- Read the current lecture file when it already exists.
- Read the provided references and decide which ones mainly support which sections.
- Build a short drafting brief before spawning workers:
  - which sections each worker will own,
  - which references are primary for each owned section,
  - what terminology and notation should remain consistent across the lecture,
  - what source-anchor style and handoff-note style workers should follow,
  - which global text only the integrating agent should touch, such as lecture opening, lecture conclusion, or shared framing paragraphs.
- Keep the brief short; it is a mergeability constraint, not a second lecture plan.

3. Split the lecture into worker-owned sections.
- Partition the lecture into disjoint `##` sections by default.
- Use `###`-level splitting only when a `##` section is too large or the user explicitly asks for finer parallelism.
- Give each worker a clear write boundary.
- Tell each worker they are not responsible for lecture-wide introduction, conclusion, terminology normalization, or whole-lecture deduplication.
- Tell each worker to keep local section quality high while avoiding edits outside their assigned scope.

4. Spawn `rough-draft` workers.
- Use `rough-draft` as the worker skill for section-local drafting.
- Pass each worker only the minimum context needed:
  - the target file,
  - the lecture plan,
  - the assigned section boundary,
  - the references most relevant to that section,
  - nearby headings or adjacent sections when needed,
  - the drafting brief only as a mergeability constraint, not as a request to rewrite lecture-wide framing.
- Make ownership explicit so workers do not draft the same section.
- Ask workers to preserve useful local source anchors and handoff notes that still help later revision.

5. Merge the lecture yourself.
- Merge worker outputs back into the lecture draft in lecture-plan order.
- Own all lecture-level merge decisions yourself:
  - consistent heading hierarchy,
  - consistent terminology and notation across sections,
  - removal of obvious duplicated setup created by parallel workers,
  - retention of source anchors that are still useful for later revision,
  - minimal connective tissue when two neighboring sections would otherwise join too abruptly.
- Keep the merged draft clearly rough; do not over-polish the prose in this skill.
- Do not leave worker-management commentary in the lecture body.

6. Finish with a whole-lecture pass.
- Check whether the lecture now reads like one material-complete rough draft rather than several section drafts stitched together.
- Verify that no worker accidentally drafted a second lecture introduction or conclusion.
- Verify that all major sections from the lecture plan are represented.
- Leave only the smallest necessary handoff notes or rough-draft guidance for later writing revision.

## Worker Assignment Rules

- Give each worker disjoint ownership.
- Tell workers they are not alone in the codebase and should not revert others' edits.
- Tell workers to draft only their assigned section.
- Tell workers not to rewrite the lecture introduction, lecture conclusion, or global terminology choices unless explicitly instructed.
- Tell workers to preserve local technical core, formulas, examples, and source anchors when they matter for later revision.

## Merge Checklist

Before finishing, check these lecture-level questions:

- Does the merged file still follow the lecture plan section order?
- Did each worker stay inside its scope, or does the draft still contain competing versions of the same setup?
- Are terminology, notation, and source-anchor conventions consistent enough that later revision will not need to rediscover them?
- Did the merge pass remove only obvious duplication, without prematurely polishing away useful rough-draft redundancy?
- Does the final file read as one material-complete rough draft rather than a material dump or a bundle of disconnected section drafts?

## Boundaries

- Use this skill for lecture-level drafting orchestration and merge, not for isolated one-section drafting.
- Do not make every section worker read every reference file when section-specific allocation is enough.
- Do not let multiple workers compete over the lecture introduction, conclusion, or global terminology.
- Do not upgrade the merge pass into full writing revision; keep the lecture clearly in rough-draft stage.
- Do not decide final demo insertion here; keep that for the demo skill.

## Resource

- `references/orchestration-format.md`: drafting brief format, worker assignment template, and merge-pass checklist.
