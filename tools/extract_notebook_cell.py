"""Extract one notebook code cell, optionally dropping a leading cell magic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("cell", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--drop-first-line", action="store_true")
    args = parser.parse_args()

    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    cell = notebook["cells"][args.cell]
    if cell.get("cell_type") != "code":
        raise ValueError(f"cell {args.cell} is not a code cell")
    source = cell.get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    if args.drop_first_line:
        source = source.split("\n", 1)[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source.rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
