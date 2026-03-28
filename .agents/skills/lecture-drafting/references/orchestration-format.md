# Lecture Drafting Orchestration Format

## Drafting Brief

Keep the lecture-level brief short and operational.

```md
## Drafting Brief
- lecture scope: docs/core/.../02_xxx.md
- worker split:
  - worker A: ## 1-3
  - worker B: ## 4-6
- image ownership:
  - worker A: `references/.../images/fit_curves/`
  - worker B: `references/.../images/architecture_bias/`
- terminology to keep stable:
  - empirical risk -> 经验风险
  - population risk -> 总体风险
- source-anchor rule:
  - keep anchors only for claims, formulas, or explanation paths worth revisiting later
- global text owned only by integrator:
  - lecture opening
  - lecture conclusion
  - section-to-section connective sentences when they cross worker boundaries
```

## Worker Assignment Template

```md
Use the `rough-drafter` subagent on [target file]. It should execute `$rough-draft` within its assigned scope.

Your ownership:
- write scope: ## 3 to ## 4
- do not edit outside this scope

Inputs:
- lecture plan: [path]
- references for your scope:
  - [path A]
  - [path B]
- images for your scope:
  - [image dir or image file A]
  - [image dir or image file B]
- nearby context to read for continuity:
  - previous heading
  - next heading

Constraints:
- keep the local section material-complete, not polished
- preserve formulas, symbols, mechanism chains, and high-value examples
- insert only the images that clearly support your owned section; do not scan unrelated image folders
- keep useful source anchors and short handoff notes
- do not rewrite lecture-wide introduction, conclusion, or global terminology
```

## Merge Pass

During merge, prefer the lightest lecture-level cleanup that improves mergeability:

- keep lecture-plan order intact
- remove only obvious duplicated setup
- unify heading depth and terminology
- preserve source anchors that still help later revision
- remove only obvious duplicated images
- keep image blocks near the paragraphs they support
- add only minimal transition prose across worker boundaries
- leave deeper prose cleanup to `writing-revision`

## Anti-Patterns

Avoid these mistakes:

- giving every worker the full reference set when section-scoped subsets are enough
- giving every worker every image directory when section-scoped subsets are enough
- letting multiple workers rewrite the same intro or shared framing paragraph
- normalizing the whole lecture into polished prose during the merge pass
- dropping technical core or source anchors just because two section drafts use different local styles
