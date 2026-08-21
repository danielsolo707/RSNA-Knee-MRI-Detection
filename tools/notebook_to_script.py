"""Convert code cells from a Jupyter notebook into a plain Python script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("script", type=Path)
    args = parser.parse_args()

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    chunks: list[str] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        chunks.append(f"# %% [notebook cell {index}]\n{source.rstrip()}\n")

    args.script.parent.mkdir(parents=True, exist_ok=True)
    args.script.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
