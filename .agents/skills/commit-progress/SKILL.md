---
name: commit-progress
description: Read the current repository's `.harness/progress.md` plus the current git diff, sync long-lived module notes into `src/module-name/notes.md`, draft a commit summary, commit the current repo changes, and then run the current repository's `.harness/cli/reset-progress`. Use only when the user explicitly invokes `$commit-progress`; do not trigger from ordinary natural-language requests about commits, summaries, notes syncs, or progress resets.
---

# Commit Progress

## Overview

Use this skill to turn the current repository worktree into a single commit whose message includes both a fresh summary and the current `progress.md` snapshot. Before committing, sync long-lived module knowledge from `progress.md` into touched `src/<module>/notes.md` files. After the commit succeeds, reset `progress.md` so the next work cycle starts cleanly.

## Workflow

1. Confirm the trigger and repo state.
- Only proceed after an explicit `$commit-progress` invocation.
- Resolve the current git repository root from the working directory.
- Read `<repo-root>/.harness/progress.md`.
- Inspect `git status --short` and `git diff --stat`.
- If there is nothing meaningful to commit, stop and say so.
- Stop if the current directory is not inside a git repository.

2. Sync module notes from progress.
- Inspect the modified paths and collect touched `src/<module>/` directories.
- For each touched module, update `src/<module>/notes.md` before drafting the commit message.
- Follow [references/progress_notes.md](references/progress_notes.md) for the extraction and merge rules.
- Skip modules with no qualifying long-lived information in `progress.md`.

3. Draft the commit message.
- Write a concise one-line subject based on `progress.md` and the current diff.
- Add a short bullet summary of the main changes.
- Mention notes synchronization when it materially changed any `src/<module>/notes.md`.
- Append the full current `progress.md` content verbatim so the commit message carries the workflow state snapshot.
- Write the message to a temporary file instead of a shell variable.

Use this layout:

```text
<subject line>

Summary:
- <main change>
- <verification or important note>

Progress snapshot:
<repo-root>/.harness/progress.md
<full progress.md content>
```

4. Commit and reset.
- Run `python3 <repo-root>/.agents/skills/commit-progress/scripts/commit_from_message.py <message-file>`.
- This helper auto-discovers `<repo-root>` from the current working directory, performs `git add .`, verifies that staged changes exist, commits with `git commit -F`, and then runs `<repo-root>/.harness/cli/reset-progress`.
- Pass `--repo-root <repo-root>` only when you need to override auto-discovery.
- Do not push unless the user separately asks for it.

5. Report back.
- Quote the commit subject you used.
- Summarize the committed files from `git show --stat --oneline -1`.
- Mention any `src/<module>/notes.md` files updated from `progress.md`.
- Confirm that `<repo-root>/.harness/cli/reset-progress` ran.

## Stop Rules

- Stop if the user did not explicitly invoke `$commit-progress`.
- Stop if `<repo-root>/.harness/progress.md` is missing or empty enough that no useful summary can be written.
- Stop if `.harness/cli/reset-progress` is missing.
- Stop if `git add .` still leaves no staged changes.
- Treat commit failures, merge conflicts, and missing git identity as repo-state issues; report them clearly instead of retrying blindly.

## Resource

- `scripts/commit_from_message.py`: deterministic `git add . -> git commit -F <message-file> -> .harness/cli/reset-progress` helper.
- `references/progress_notes.md`: rules for syncing long-lived module notes from `progress.md`.
