#!/usr/bin/env python3
"""Build Kaggle notebooks for parallel cache + full B3 fold training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_SRC = ROOT / "kaggle" / "v3_1_cache" / "build_cache_parallel.py"
TRAIN_SRC_DIR = ROOT / "kaggle" / "v3_b3_fold0"
MODULES = (
    "efficientnet_b3_public_repro_v1.py",
    "efficientnet_b3_public_repro_v2_anatomy.py",
    "efficientnet_b3_public_repro_v4_t4.py",
    "efficientnet_b3_public_repro_v5_full.py",
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


def write_notebook(path: Path, cells: list[dict]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def build_cache_notebook(output: Path, image_size: int, workers: int) -> None:
    source = CACHE_SRC.read_text(encoding="utf-8")
    driver = f'''import subprocess
import sys
from pathlib import Path

competition = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
if not competition.is_dir():
    matches = list(Path("/kaggle/input").glob("**/train_series.csv"))
    if not matches:
        raise FileNotFoundError("competition input unavailable")
    competition = matches[0].parent

command = [
    sys.executable,
    "/kaggle/working/build_cache_parallel.py",
    "--train-csv", str(competition / "train.csv"),
    "--series-csv", str(competition / "train_series.csv"),
    "--image-root", str(competition / "train_series"),
    "--output-dir", "/kaggle/working/v3_1_cache",
    "--image-size", "{image_size}",
    "--cache-slices", "12",
    "--workers", "{workers}",
]
print("Launching:", " ".join(command), flush=True)
subprocess.run(command, check=True)
'''
    write_notebook(
        output,
        [
            code_cell(f"%%writefile /kaggle/working/build_cache_parallel.py\n{source}"),
            code_cell(driver),
        ],
    )


def build_fold_notebook(output: Path, fold: int, label_file: str) -> None:
    cells = [
        code_cell(
            "from pathlib import Path\n"
            "Path('/kaggle/working/v3_b3_source').mkdir(parents=True, exist_ok=True)\n"
        )
    ]
    for name in MODULES:
        source = (TRAIN_SRC_DIR / name).read_text(encoding="utf-8")
        cells.append(code_cell(f"%%writefile /kaggle/working/v3_b3_source/{name}\n{source}"))

    driver = f'''import json
import subprocess
import sys
from pathlib import Path

competition = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
if not competition.is_dir():
    matches = list(Path("/kaggle/input").glob("**/train_series.csv"))
    if not matches:
        raise FileNotFoundError("competition train_series.csv unavailable")
    competition = matches[0].parent

def unique_file(pattern):
    matches = sorted(Path("/kaggle/input").glob(pattern))
    if len(matches) != 1:
        # Prefer the v3 labels dataset path when multiple exist.
        preferred = [m for m in matches if "rsna-knee-v3-labels" in str(m)]
        if len(preferred) == 1:
            return preferred[0]
        raise RuntimeError(f"expected one {{pattern}}, found {{matches}}")
    return matches[0]

labels = unique_file("**/{label_file}")
folds = unique_file("**/folds_v3.csv")
checkpoint = unique_file("**/tf_efficientnet_b3*.pth")
output = Path("/kaggle/working/v3_b3_fold{fold}")
command = [
    sys.executable,
    "/kaggle/working/v3_b3_source/efficientnet_b3_public_repro_v5_full.py",
    "--train-csv", str(competition / "train.csv"),
    "--series-csv", str(competition / "train_series.csv"),
    "--v3-labels", str(labels),
    "--v3-folds", str(folds),
    "--image-root", str(competition / "train_series"),
    "--backbone-checkpoint", str(checkpoint),
    "--output-dir", str(output),
    "--fold", "{fold}",
    # GPU budget wave-1 + T4 VRAM: batch1 / workers0 / smaller backbone chunks.
    # Fold0 v3 died SIGKILL ~47m (OOM). 8 slices subsampled from 12-slice cache.
    # See docs/gpu-budget-28h.md.
    "--image-size", "288",
    "--train-slices", "8",
    "--valid-slices", "8",
    "--batch-size", "1",
    "--workers", "0",
    "--backbone-chunk", "6",
    "--frozen-epochs", "1",
    "--unfrozen-epochs", "3",
    "--require-cache",
]
print("Launching:", " ".join(command), flush=True)
subprocess.run(command, check=True)
print("artifacts:", sorted(p.name for p in output.glob("*")), flush=True)
'''
    cells.append(code_cell(driver))
    write_notebook(output, cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "kaggle" / "v5_production")
    args = parser.parse_args()
    out = args.out_dir
    build_cache_notebook(out / "cache_288" / "rsna-knee-v5-cache-288.ipynb", 288, 4)
    for fold in range(5):
        build_fold_notebook(
            out / f"b3_fold{fold}" / f"rsna-knee-v5-b3-fold{fold}.ipynb",
            fold,
            f"labels_v3_fold{fold}.csv",
        )
    print("done")


if __name__ == "__main__":
    main()
