# Progress Notes

Use this reference only when syncing `.harness/progress.md` into module notes during `$commit-progress`.

## Goal

Preserve only module-level, long-lived information that is not reliably inferable from code, types, docstrings, inline comments, or tests.

## Linear Flow

1. Read the current repository's `.harness/progress.md`.
2. Inspect the current modified paths and collect touched `src/<module>/` directories.
3. For each touched module, open `src/<module>/notes.md` if it exists.
4. Extract candidate facts from `progress.md` that belong to that module.
5. Keep only facts that fit one of the fixed sections below.
6. Merge them into `src/<module>/notes.md` by section.
7. Remove duplicates or equivalent restatements.
8. Skip modules with no qualifying facts.

## Fixed Sections

Only use these sections:

- `## Scope`
- `## External Assumptions`
- `## Verification`
- `## Design Constraints`

Do not create empty sections.

## Keep

Keep only:

- External prerequisites such as environment variables, data locations, or runtime dependencies
- Verification rules or verification preconditions that will matter again later
- Non-obvious module constraints whose rationale is not recoverable from code alone

## Drop

Drop:

- Implementation details visible in code
- Facts visible in type hints, docstrings, inline comments, or tests
- Per-commit file lists
- Current blockers, TODOs, temporary states, or future plans
- Unverified guesses
- Changelog-style narration

## Minimality Test

Keep a candidate only if removing it would make a future maintainer more likely to make a mistake in setup, verification, or boundary interpretation because the code does not reveal it.

## Merge Rules

- Update only `src/<module>/notes.md`
- Merge by section instead of rewriting the whole file
- Prefer short bullets or short sentences
- Preserve existing valid notes
- Deduplicate by meaning, not only by exact string match
