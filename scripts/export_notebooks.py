from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert import PythonExporter
from nbformat.reader import NotJSONError


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebook"
OUTPUT_DIR = ROOT / "src"


def export_notebook(notebook_path: Path, exporter: PythonExporter) -> Path:
    output_path = OUTPUT_DIR / f"{notebook_path.stem}.py"
    try:
        notebook = nbformat.read(notebook_path, as_version=4)
    except (NotJSONError, OSError) as exc:
        body = (
            f"# Source notebook could not be converted: {notebook_path.name}\n"
            f"# Reason: {exc}\n"
        )
    else:
        body, _ = exporter.from_notebook_node(notebook)

    output_path.write_text(body, encoding="utf-8")
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exporter = PythonExporter()

    notebook_paths = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebook_paths:
        raise SystemExit(f"No notebooks found in {NOTEBOOK_DIR}")

    for notebook_path in notebook_paths:
        output_path = export_notebook(notebook_path, exporter)
        print(f"{notebook_path.relative_to(ROOT)} -> {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
