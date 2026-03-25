#!/usr/bin/env python3
"""Commit the current repo state from a prepared message file, then reset progress."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def discover_repo_root(start_path: Path) -> Path:
    """Resolve the git repository root from the current working tree.

    Args:
        start_path: Directory to start the git root lookup from.

    Returns:
        Absolute repository root path.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start_path,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr_text = result.stderr.strip() or "not inside a git repository"
        raise SystemExit(f"Unable to discover git repository root from {start_path}: {stderr_text}")
    return Path(result.stdout.strip()).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run git add/commit from a prepared message file, then reset progress.",
    )
    parser.add_argument("message_file", help="Path to the prepared commit message file")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repository root override containing .git and .harness/cli/reset-progress",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root is not None
        else discover_repo_root(Path.cwd())
    )
    message_file = Path(args.message_file).resolve()
    reset_progress = repo_root / ".harness" / "cli" / "reset-progress"

    if not (repo_root / ".git").exists():
        raise SystemExit(f"Not a git repository root: {repo_root}")
    if not message_file.is_file():
        raise SystemExit(f"Commit message file not found: {message_file}")
    if not reset_progress.is_file():
        raise SystemExit(f"Missing reset-progress script: {reset_progress}")

    run(["git", "add", "."], cwd=repo_root)

    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_root,
        check=False,
    )
    if staged.returncode == 0:
        raise SystemExit("No staged changes to commit.")
    if staged.returncode != 1:
        raise SystemExit("Unable to determine staged git diff state.")

    run(["git", "commit", "-F", str(message_file)], cwd=repo_root)
    run([str(reset_progress)], cwd=repo_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
