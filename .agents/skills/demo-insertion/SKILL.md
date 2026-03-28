---
name: demo-insertion
description: Insert project-specific runnable teaching demos into an existing lecture draft. Use when a lecture or one named section/subsection is already structurally stable and needs final code blocks that directly close a real understanding gap, demonstrate a mechanism, validate a derivation, compare behaviors, or produce a minimal teaching figure from code. Do not use for rough drafting, prose revision, lecture-wide orchestration, or curation of existing images from local folders or webpages; use `rough-draft`, `writing-revision`, `lecture-revision`, or `figure-curation` instead.
---

# Demo Insertion

## Overview

Insert final teaching demos into this tutorial project's lecture markdown. Treat demos as a last-mile teaching asset: decide whether code is truly needed, then write the minimal runnable code block directly into the lecture and tighten the surrounding prose so the reader knows what to run and what to look for.

## Workflow

1. Confirm the insertion scope.
- Work on one lecture only.
- Resolve the target markdown file before editing.
- Distinguish between two modes:
  - full-lecture mode: inspect the whole lecture and insert demos where needed.
  - section-scoped mode: inspect only the named `##` or `###` block and its nearby context.
- If the user names only one section, do not silently add demos to distant sections.

2. Read the lecture before writing any code.
- Read the current lecture draft first.
- In section-scoped mode, also read the parent heading and adjacent sibling headings so the local demo does not break nearby flow.
- Treat existing `code block placeholder` markers as strong hints, not automatic insertion commands.
- If the lecture still reads like a rough draft, or the prose path is not stable enough to know what the demo should teach, stop and hand the task back to `writing-revision` or `lecture-revision`.

3. Decide whether a demo is actually needed.
- Insert a demo only when code will help the reader observe a phenomenon, understand a mechanism, compare two behaviors, or validate a compact derivation.
- Do not insert a demo when the surrounding text, math, or an existing figure already makes the point cleanly.
- Prefer one sharp demo over multiple repetitive ones.
- If the lecture is missing a static image that should already have been preserved from existing references, do not take ownership of that gap here; hand it back to `rough-draft`, `writing-revision`, or a user-requested incremental figure pass.

4. Design the smallest runnable experiment.
- Keep only the minimum data, parameters, and control flow needed to expose the idea.
- Prefer short helper functions over one long script when the logic has two or three natural steps.
- Keep code blocks runnable and local to the lecture.
- If the natural output is a tiny curve, scatter plot, or geometric sketch produced by the code, include it as part of the demo rather than splitting it into a separate figure workflow.
- Make the expected observation obvious from the code and the surrounding text.

5. Write the surrounding teaching prose.
- Add a short lead-in before each inserted demo:
  - what question this demo answers,
  - what the reader should pay attention to,
  - why running it is worthwhile.
- Add a short follow-up after the demo when the result is not self-evident.
- Keep the voice inside the tutorial. Write as the tutorial author speaking to learners, not as an editor leaving notes.
- Do not leave behind placeholder markers once the final demo is inserted.

6. Apply the project code style.
- Make the code look like runnable pseudocode: minimal, readable, and mechanically correct.
- Do not add defensive programming, logging, configuration systems, CLI entrypoints, testing scaffolds, or library-style abstraction.
- If you define helper functions, prefer keyword arguments on the main data flow.
- Add type hints to nontrivial helper functions when that improves readability.
- If the demo uses tensors and changes tensor shape, add trailing shape comments on the shape-changing lines.
- Avoid explicit loops over tensor dimensions; prefer batched ops, `einsum`, or vectorized code when relevant.
- Do not hardcode `.to("cuda")`; use device-aware code when device placement matters.
- Use names that teach the idea, such as `student_logits`, `margin_values`, or `bin_centers`, instead of generic names like `x`, `tmp`, or `res` unless the math is genuinely clearer that way.

7. Leave a clean lecture.
- Keep only the demos that materially improve understanding.
- If you choose not to implement a previously suggested placeholder, either remove it or replace it with a short explanation in revision notes.
- Keep the markdown reader-facing; collaborator notes should stay in comments or separate notes, not in the lecture body.

## Full-Lecture and Section-Scoped Modes

Treat both modes as first-class.

- In full-lecture mode, scan the lecture for the few places where demos genuinely unlock understanding.
- In section-scoped mode, only insert demos inside the requested scope unless a tiny neighboring bridge is necessary.
- In section-scoped mode, do not renumber unrelated headings or rewrite distant demo decisions.
- If a lecture-wide demo plan is inconsistent, note it briefly instead of silently rewriting the whole lecture from inside one section task.

## Demo Heuristics

Prefer demos for these cases:
- a theorem or claim becomes intuitive after one tiny numerical experiment,
- a representation or decision boundary is easier to grasp after plotting it,
- two architectures, losses, or optimization choices are best understood by comparing behavior,
- the lecture asks the reader to believe a mechanism that is easy to observe directly.

Avoid demos for these cases:
- pure terminology, historical framing, or reading guidance,
- material that would require too much setup for too little insight,
- code that merely rephrases the prose without creating a new observation,
- material already explained cleanly by an existing image in the lecture draft.

## Demo Checklist

Before finishing, check each inserted demo against this list:
- Does this code block answer a concrete teaching question?
- Is it runnable as written inside the project tutorial context?
- Is the setup minimal enough that the reader can see the mechanism without wading through boilerplate?
- Is there a short lead-in telling the reader what to observe?
- If the result is not visually obvious, is there a short follow-up explaining the observation?
- If the code produces a figure, is the figure generation minimal and directly tied to the point of the section?
- Does the code avoid engineering scaffolding, defensive branches, and unnecessary abstraction?
- If the demo uses tensor code, does it follow the project's tensor and device conventions?

## Boundaries

- Produce final lecture-facing demo code, not placeholders.
- Do not do large-scale prose revision here; hand that work back to `writing-revision` or `lecture-revision`.
- Do not curate existing images from local folders or webpages; the default place for those images is the rough-draft workflow, not this skill.
- Do not expand the lecture's theoretical scope just to justify a demo.
- Do not turn tutorial code into reusable library code.
- Do not leave reader-facing `TODO`, `本节目标`, or collaborator notes in the lecture body.

## Resource

- `references/demo-format.md`: lecture insertion patterns, section-scoped conventions, and demo block shapes for this project.
