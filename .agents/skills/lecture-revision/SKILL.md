---
name: lecture-revision
description: Coordinate a full lecture revision by splitting sections across multiple `writing-reviser` subagents that execute `writing-revision`, then integrating the whole lecture into one coherent draft. Use when a whole lecture benefits from parallel section revision, when one agent should own cross-section bridges and terminology consistency, or when previous-lecture continuity should be handled once at lecture level instead of by every section reviser.
---

# Lecture Revision

## Overview

Revise one whole lecture by orchestrating section-level `writing-reviser` subagents, integrating their outputs into one coherent draft, and then running one narrow audit-only pass. Own the lecture-level decisions that should not be duplicated across section workers.

## Workflow

1. Confirm the orchestration scope.
- Work on one lecture only.
- If the user wants only one section or subsection revised, hand the task to `$writing-revision` instead of orchestrating workers.
- Resolve the target lecture file before doing anything else.

2. Build the lecture-level context.
- Read the full current lecture draft first.
- Read the lecture plan if it exists.
- Identify the previous lecture as the file in the same folder whose leading lecture number is current-minus-one, and read it when it exists.
- Build a short continuity brief before spawning workers:
  - what the current lecture should assume from the previous lecture,
  - what terminology and notation must stay consistent,
  - what should not be re-explained,
  - what kind of opening bridge the integrated draft should use.
- Keep this brief short; it is a constraint, not a second outline.

3. Split the lecture into worker-owned sections.
- Partition the lecture into disjoint `##` sections by default.
- Use `###`-level splitting only when a `##` section is too large or the user explicitly asks for finer parallelism.
- Give each worker a clear write boundary.

4. Spawn `writing-reviser` workers.
- Use the `writing-reviser` subagent for section-local rewriting; it should execute the repository's `writing-revision` skill.
- Pass each worker only the minimum context needed:
  - the target file,
  - the assigned section boundary,
  - nearby headings or adjacent sections when needed,
  - the continuity brief only as terminology/duplication guidance, not as a request to write lecture-wide bridges.
- Make ownership explicit so workers do not rewrite the same section.
- Ask workers to preserve useful local image blocks, placeholders, and notes that still help later integration.

5. Integrate the lecture yourself.
- Merge worker outputs back into the lecture draft.
- Own all lecture-level integration decisions yourself:
  - the opening transition from the previous lecture,
  - bridges between major sections,
  - consistent terminology and notation across sections,
  - removal of duplicated explanations created by parallel workers,
  - removal of duplicated images or captions created by parallel workers,
  - relocation of image blocks that ended up attached to the wrong paragraph after merge,
  - smoothing abrupt tone or abstraction jumps across section boundaries.
- Keep the integrated draft in tutorial-author voice.
- Do not leave worker-management commentary in the lecture body.

6. Run an audit-only pass.
- After integration, do one explicit audit-only pass over the whole lecture.
- The goal of this pass is detection-first and cleanup-second: catch lecture-level residue without turning the pass into a second full rewrite.
- This pass may directly fix only narrow, mechanical, low-risk issues such as:
  - writing-scaffold prose,
  - leftover rough-draft section scaffolding such as `本节目标` or `主要参考`,
  - course-structure meta commentary such as `把这一节放在...` or `前面讲了...后面会...`,
  - source-aware residue such as `主要参考`, `参考材料里`, or `slides 上`,
  - duplicated framing created by parallel workers,
  - notation inconsistency,
  - math-format or escape corruption,
  - duplicated image blocks or obviously stale image captions,
  - short bridge sentences that are obviously abrupt after merge.
- If a problem would require substantial local rewriting, re-arguing the section, or structural reorganization, do not silently perform a second full revision pass; note the issue briefly instead.

7. Finish with a whole-lecture pass.
- Verify the integrated draft reads like one lecture, that lecture-level continuity appears only once, and that the audit-only pass stayed narrow.
- Leave only the smallest necessary placeholders or revision notes.

## Worker Assignment Rules

- Give each worker disjoint ownership.
- Tell workers they are not alone in the codebase and should not revert others' edits.
- Tell workers to revise only their assigned section.
- Tell workers not to add lecture-wide framing, previous-lecture recap, or global terminology changes unless explicitly instructed.

## Integration Checklist

Before finishing, check these lecture-level questions:

- Does the lecture opening naturally pick up from the previous lecture when that connection matters?
- Is previous-lecture continuity written once at lecture level, rather than repeated by multiple sections?
- Do adjacent sections connect cleanly, without duplicated setup or abrupt jumps?
- Are notation, term choices, and problem framing consistent across all revised sections?
- Do adjacent sections still use image blocks coherently, without duplicated figures or captions left over from local revisions?
- Did section workers stay inside their scopes, or does the merged lecture still contain conflicting framing?
- Does the merged lecture still contain scaffold prose, course-structure meta commentary, source-aware residue, or obvious math-format corruption?
- Did the final audit stay narrow, or did it accidentally become a second full rewrite?
- Does the final lecture read as one coherent tutorial chapter rather than a bundle of separately revised sections?

## Boundaries

- Do not make every section worker read and own previous-lecture continuity.
- Do not let workers compete over the lecture introduction, conclusion, or global terminology.
- Do not redo every section locally after workers return; focus on integration and lecture-level fixes.
- Do not turn the audit-only pass into a second full revision pass.
- During the audit-only pass, only make narrow, mechanical, low-risk fixes; if a problem needs broad rewriting, note it instead of silently rewriting large sections.
- Do not decide final demo insertion here; keep that for the demo skill.

## Resource

- `references/orchestration-format.md`: continuity brief format, worker assignment template, and integration-pass checklist.
