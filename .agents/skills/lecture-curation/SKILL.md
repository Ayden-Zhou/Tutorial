---
name: lecture-curation
description: Plan the structure of a single lecture from multiple reference files and user instructions, then write the lecture framework directly to the target Markdown file under docs/. Use when the user provides PDFs, notes, slides, papers, or other reference paths for one lecture and needs help deciding what to cover, how to organize sections and subsections, how to preserve high-value intuitive explanations as structure, and which reference files mainly support each section.
---

# Lecture Curation

## Overview

Read the provided references for one lecture, compare their teaching value, and write a lecture framework directly to the target Markdown file. Preserve the value of intuitive explanations by turning them into structure when appropriate, instead of copying source wording.

## Workflow

1. Confirm the lecture scope.
- Work on one lecture only.
- Resolve the target output file under `docs/`.
- If the target file already exists, read it first and treat it as the current draft instead of overwriting blindly.
- If the instruction does not identify a single lecture target, stop and ask for clarification.

2. Read the source materials.
- Read every reference file the user provides.
- Build a short source map for each file: main theme, high-value intuitive explanations, problem motivations, mechanism explanations, examples, and low-value detail.
- Compare overlap and disagreement across sources before deciding structure.

3. Design the lecture structure.
- Write the lecture title in Chinese.
- Do not include the lecture number in the H1 title.
- Use section and subsection boundaries to reflect teaching value, not the table of contents of any one source.
- Default to lecture-note style headings rather than spoken-language prompts.
- `##` headings should usually name a conceptual block, mechanism, result, comparison, or limitation.
- `###` headings should usually name a concept, construction, example, conclusion, or caveat.
- Promote a source idea into its own `##` or `###` when it gives readers a much better intuition, motivation, comparison, or mechanism view.
- Keep the lecture focused on the current topic. Do not expand into a whole-course outline.
- Avoid casual heading templates such as `我们到底在问什么问题`、`为什么已经有启发`、`还能往哪里找解释` unless the user explicitly wants a conversational style.

4. Write the framework.
- Write directly to the target Markdown file.
- Follow [references/output-format.md](references/output-format.md) for heading numbering, title naming, and section-level reference formatting.
- Record the main supporting references under each `##` section using relative paths from the output document.
- Use concise placeholders such as `本节目标` or `取舍说明` when they help the next drafting step.
- Do not paste long source text. Abstract and reorganize instead.

5. Review the result.
- Check that the section order follows a coherent learning path.
- Check that every major section has explicit supporting references.
- Check that the framework preserves high-value intuitive content even when the source treated it as a side note.
- Check that the headings can stand alone in a printed outline and still read like a lecture handout.
- Check that the output is still a framework, not a full lecture draft.

## Selection Heuristics

- Prefer content that helps readers understand why a concept exists, what problem it solves, or how to build an intuition for it.
- Treat intuitive explanations, minimal examples, contrasts, and common misunderstandings as high-value signals for structure design.
- Promote these high-value signals into explicit sections or subsections when they materially improve the lecture.
- Downweight content that mainly repeats definitions, notation, history, or exhaustive detail without improving understanding of the current lecture goal.
- When several sources cover the same point, keep the source that explains it most clearly and use the others as support.
- Preserve disagreements only when the difference itself is pedagogically useful.
- Prefer heading names that would still make sense if extracted into a syllabus or printed lecture outline.

## Boundaries

- Produce a lecture framework, not polished lecture prose.
- Do not add runnable code blocks in this step.
- Do not decide final demo code placement in this step.
- Do not expand the lecture beyond the current user-provided scope.
- Do not copy source wording except for short file paths or unavoidable terms.

## Resource

- `references/output-format.md`: heading rules, title naming rules, and the per-section reference block format.
