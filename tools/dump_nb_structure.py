"""Dump cell structure of a notebook (one line per cell, source truncated)."""
import json, sys

def summarize(path, max_cells=200, truncate=90):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    print(f"{path}: {len(cells)} cells")
    for i, c in enumerate(cells[:max_cells]):
        src = "".join(c.get("source", []))
        first = " ".join(src.split())[:truncate]
        print(f"[{i:03d}] {c.get('cell_type','?'):8s} len={len(src):7d} | {first}")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
        print()
