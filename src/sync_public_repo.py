from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_ROOT = Path("/home/data/users/zhouyf/myprojects/Tutorial-Site")
SYNC_DIR_NAMES = ("docs", "images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror docs/ and images/ into the public Tutorial-Site repo."
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
        help="Target public repo root. Defaults to Tutorial-Site.",
    )
    return parser.parse_args()


def remove_extra_entries(*, source_dir: Path, target_dir: Path) -> None:
    for target_entry in target_dir.iterdir():
        source_entry = source_dir / target_entry.name
        if source_entry.exists():
            continue
        if target_entry.is_dir():
            shutil.rmtree(target_entry)
            continue
        target_entry.unlink()


def copy_source_entries(*, source_dir: Path, target_dir: Path) -> None:
    for source_entry in source_dir.iterdir():
        target_entry = target_dir / source_entry.name
        if source_entry.is_dir():
            target_entry.mkdir(parents=True, exist_ok=True)
            sync_tree(source_dir=source_entry, target_dir=target_entry)
            continue
        shutil.copy2(source_entry, target_entry)


def sync_tree(*, source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    remove_extra_entries(source_dir=source_dir, target_dir=target_dir)
    copy_source_entries(source_dir=source_dir, target_dir=target_dir)


def main() -> None:
    args = parse_args()
    target_root = args.target_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    for dir_name in SYNC_DIR_NAMES:
        source_dir = PROJECT_ROOT / dir_name
        target_dir = target_root / dir_name
        sync_tree(source_dir=source_dir, target_dir=target_dir)
        print(f"synced {source_dir} -> {target_dir}")


if __name__ == "__main__":
    main()
