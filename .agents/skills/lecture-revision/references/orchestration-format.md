# Orchestration Format

Use this reference when coordinating a lecture-wide revision with multiple `writing-reviser` workers. Each worker should execute the `writing-revision` skill locally.

## Continuity Brief

Create a very short lecture-level continuity brief before spawning workers.

Recommended shape:

```md
## Continuity Brief

- previous lecture: `01_Foo.md`
- carry over:
  - readers already know concept A and notation B
  - do not re-explain mechanism C in full
- opening bridge:
  - start from X, then narrow to Y
- terminology locks:
  - keep using `经验风险` rather than alternate labels
```

Keep it short. It exists to prevent duplicate lecture-wide framing, not to become a second outline.

## Worker Assignment Template

Recommended assignment shape:

```md
## Worker Assignment

- scope: `## 3 深度与宽度的表达效率`
- ownership: revise this section only
- preserve:
  - local placeholders
  - useful local examples
- do not do:
  - previous-lecture recap
  - whole-lecture introduction rewrite
  - global terminology changes
- continuity constraints:
  - keep terminology consistent with the continuity brief
  - avoid repeating material already settled earlier in the lecture
```

## Integration Pass

After workers return, handle these yourself:

- opening continuity from the previous lecture
- bridges between `##` sections
- terminology and notation normalization
- duplicate setup paragraphs created by parallel workers
- removal of conflicting lecture-level framing

## Anti-Patterns

Avoid these failure modes:

- every worker adds its own recap of the previous lecture
- multiple workers rewrite the lecture introduction
- workers rename core terms differently in different sections
- the lead integrator redoes all local prose instead of focusing on integration
