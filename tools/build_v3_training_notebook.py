#!/usr/bin/env python3
"""Package the audited V3 B3 modules into a self-contained Kaggle notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODULES = (
    "efficientnet_b3_public_repro_v1.py",
    "efficientnet_b3_public_repro_v2_anatomy.py",
    "efficientnet_b3_public_repro_v4_t4.py",
)


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": hashlib.sha1(source.encode("utf-8")).hexdigest()[:12],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--label-file", default="labels_v3.csv")
    args = parser.parse_args()

    cells = [
        code_cell(
            "from pathlib import Path\n"
            "Path('/kaggle/working/v3_b3_source').mkdir(parents=True, exist_ok=True)\n"
        )
    ]
    for name in MODULES:
        source = (args.source_dir / name).read_text(encoding="utf-8")
        cells.append(
            code_cell(
                f"%%writefile /kaggle/working/v3_b3_source/{name}\n{source}"
            )
        )

    driver = f'''import json
import shutil
import subprocess
import sys
from pathlib import Path

competition = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
if not competition.is_dir():
    matches = list(Path("/kaggle/input").glob("**/train_series.csv"))
    if not matches:
        raise FileNotFoundError("competition train_series.csv is unavailable")
    competition = matches[0].parent

def unique_file(pattern):
    matches = sorted(Path("/kaggle/input").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {{pattern}}, found {{matches}}")
    return matches[0]

labels = unique_file("**/{args.label_file}")
folds = unique_file("**/folds_v3.csv")
checkpoint = unique_file("**/tf_efficientnet_b3_aa-84b4657e.pth")
output = Path("/kaggle/working/v3_b3_fold{args.fold}")
command = [
    sys.executable,
    "/kaggle/working/v3_b3_source/efficientnet_b3_public_repro_v4_t4.py",
    "--train-csv", str(competition / "train.csv"),
    "--series-csv", str(competition / "train_series.csv"),
    "--v3-labels", str(labels),
    "--v3-folds", str(folds),
    "--image-root", str(competition / "train_series"),
    "--backbone-checkpoint", str(checkpoint),
    "--output-dir", str(output),
    "--fold", "{args.fold}",
]
print("Launching:", " ".join(command), flush=True)
cache = Path("/kaggle/working/v3_1_uint8_cache")
try:
    subprocess.run(command, check=True)
finally:
    cache_files = [path for path in cache.rglob("*.npz")] if cache.is_dir() else []
    audit = {{
        "cache_files": len(cache_files),
        "cache_bytes": sum(path.stat().st_size for path in cache_files),
        "cache_removed_before_notebook_output": True,
    }}
    output.mkdir(parents=True, exist_ok=True)
    (output / "cache_audit.json").write_text(json.dumps(audit, indent=2) + "\\n")
    print("cache audit:", audit, flush=True)
    if cache.is_dir():
        shutil.rmtree(cache)
'''
    cells.append(code_cell(driver))
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
