---
name: lecture-revision
description: Coordinate a full lecture revision by splitting sections across multiple `writing-reviser` subagents that execute `writing-revision`, then integrating the whole lecture into one coherent draft. Use when a whole lecture benefits from parallel section revision, when one agent should own cross-section bridges and terminology consistency, or when previous-lecture continuity should be handled once at lecture level instead of by every section reviser.
---

# Lecture Revision

Revise one whole lecture by orchestrating section-level `writing-reviser` subagents, integrating their outputs into one coherent draft, and then running one narrow audit-only pass. Own lecture-level decisions once instead of duplicating them across workers.

## Scope

- Work on one lecture only.
- If the user wants only one section or subsection revised, hand the task to `$writing-revision` instead.
- Resolve the target lecture file before doing anything else.

## Read Order

- Read the full current lecture draft first.
- Read the lecture plan if it exists.
- Identify the previous lecture as the file in the same folder whose leading lecture number is current-minus-one, and read it when it exists.
- Build a short continuity brief before spawning workers. Keep only:
  - what the current lecture should assume from the previous lecture,
  - what terminology and notation must stay consistent,
  - what should not be re-explained,
  - what kind of opening bridge the integrated draft should use.
- Keep the brief short. It is a constraint, not a second outline.

## Worker Orchestration

- Partition the lecture into disjoint `##` sections by default.
- Use `###`-level splitting only when a `##` section is too large or the user explicitly asks for finer parallelism.
- Give each worker a clear write boundary.
- Use the `writing-reviser` subagent for section-local rewriting; it should execute the repository's `writing-revision` skill.
- Pass each worker only the minimum context needed:
  - the target file,
  - the assigned section boundary,
  - nearby headings or adjacent sections when needed,
  - the continuity brief only as terminology/duplication guidance, not as a request to write lecture-wide bridges.
- Make ownership explicit. Workers should revise only their assigned section, should not revert others' edits, and should not add lecture-wide framing, previous-lecture recap, or global terminology changes unless explicitly instructed.
- Ask workers to preserve useful local image blocks, placeholders, and notes that still help later integration.

## Integration and Audit

- Merge worker outputs back into the lecture draft.
- Own lecture-level integration decisions yourself:
  - the opening transition from the previous lecture,
  - bridges between major sections,
  - consistent terminology and notation across sections,
  - removal of duplicated explanations, image blocks, or captions created by parallel workers,
  - relocation of image blocks that ended up attached to the wrong paragraph after merge,
  - smoothing abrupt tone or abstraction jumps across section boundaries.
- Keep the integrated draft in tutorial-author voice.
- Do not leave worker-management commentary in the lecture body.
- After integration, do one explicit audit-only pass over the whole lecture.
- Use this pass for detection first and cleanup second. Do not turn it into a second full rewrite.
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
- If a problem would require substantial local rewriting, re-arguing a section, or structural reorganization, note it briefly instead of silently doing another full pass.

## Boundaries

- Do not make every section worker read and own previous-lecture continuity.
- Do not let workers compete over the lecture introduction, conclusion, or global terminology.
- Do not redo every section locally after workers return; focus on integration and lecture-level fixes.
- Do not decide final demo insertion here; keep that for the demo skill.

## Final Check

- The lecture opening picks up from the previous lecture only where that connection matters, and that continuity appears once at lecture level.
- Adjacent sections connect cleanly, with consistent terminology, notation, and image use after merge.
- The merged lecture does not read like separately revised fragments.
- The audit-only pass stayed narrow and removed only residue or low-risk issues.

## Resource

- `references/orchestration-format.md`: continuity brief format, worker assignment template, and integration-pass checklist.
