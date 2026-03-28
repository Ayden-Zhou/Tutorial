from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebook"
OUTPUT_DIR = ROOT / "src"


def normalize_source(source: str | list[str]) -> str:
    if isinstance(source, list):
        return "".join(source)
    return source


def format_markdown_cell(source: str) -> str:
    lines = source.splitlines()
    rendered = ["# %% [markdown]"]
    if not lines:
        rendered.append("#")
    else:
        for line in lines:
            rendered.append(f"# {line}" if line else "#")
    return "\n".join(rendered)


def format_code_cell(source: str) -> str:
    rendered = ["# %%"]
    if source:
        rendered.append(source.rstrip("\n"))
    return "\n".join(rendered)


def notebook_to_percent_script(notebook: dict) -> str:
    chunks: list[str] = []
    for cell in notebook.get("cells", []):
        source = normalize_source(cell.get("source", ""))
        cell_type = cell.get("cell_type")
        if cell_type == "markdown":
            chunks.append(format_markdown_cell(source))
        elif cell_type == "code":
            chunks.append(format_code_cell(source))
        else:
            chunks.append(format_markdown_cell(source))
    return "\n\n".join(chunks) + "\n"


def export_notebook(notebook_path: Path) -> Path:
    output_path = OUTPUT_DIR / f"{notebook_path.stem}.py"
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        body = (
            "# %% [markdown]\n"
            f"# Source notebook could not be converted: {notebook_path.name}\n"
            f"# Reason: {exc}\n"
        )
    else:
        body = notebook_to_percent_script(notebook)

    output_path.write_text(body, encoding="utf-8")
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    notebook_paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebook_paths:
        raise SystemExit(f"No notebooks found in {NOTEBOOK_DIR}")

    for notebook_path in notebook_paths:
        output_path = export_notebook(notebook_path)
        print(f"{notebook_path.relative_to(ROOT)} -> {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
