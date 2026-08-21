#!/usr/bin/env python3
"""Build V6 B3 fold notebooks: longer train (1+7 epochs) for real .90 path.

Same T4-safe VRAM knobs as V5 wave-1 (batch1, workers0, chunk6, 8 slices,
require-cache) but more unfrozen epochs so the CNN can approach the public
8-epoch recipe instead of the 1+3 underfit that soft-OOF'd at ~0.77.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SRC_DIR = ROOT / "kaggle" / "v3_b3_fold0"
MODULES = (
    "efficientnet_b3_public_repro_v1.py",
    "efficientnet_b3_public_repro_v2_anatomy.py",
    "efficientnet_b3_public_repro_v4_t4.py",
    "efficientnet_b3_public_repro_v5_full.py",
)


def code_cell(source: str) -> dict:
    if not source.endswith("\n"):
        source += "\n"
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


def build_fold_notebook(
    output: Path,
    fold: int,
    label_file: str,
    frozen_epochs: int,
    unfrozen_epochs: int,
    train_slices: int,
) -> None:
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
        preferred = [m for m in matches if "rsna-knee-v3-labels" in str(m) or "labels_v3" in str(m)]
        if len(preferred) == 1:
            return preferred[0]
        # Prefer exact fold label basename match
        exact = [m for m in matches if m.name == pattern.split("/")[-1]]
        if len(exact) == 1:
            return exact[0]
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
    "--image-size", "288",
    "--train-slices", "{train_slices}",
    "--valid-slices", "{train_slices}",
    "--batch-size", "1",
    "--workers", "0",
    "--backbone-chunk", "6",
    "--frozen-epochs", "{frozen_epochs}",
    "--unfrozen-epochs", "{unfrozen_epochs}",
    "--gold-weight", "3.0",
    "--expert-sample-fraction", "0.20",
    "--unfrozen-backbone-blocks", "4",
    "--model-family", "efficientnet_b3_noflip_v6_long",
    "--require-cache",
]
print("V6 Launching:", " ".join(command), flush=True)
subprocess.run(command, check=True)
print("artifacts:", sorted(p.name for p in output.glob("*")), flush=True)
'''
    cells.append(code_cell(driver))
    write_notebook(output, cells)


def write_meta(path: Path, kernel_id: str, title: str, code_file: str) -> None:
    meta = {
        "id": kernel_id,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": ["danielsolo1770/rsna-knee-v3-labels-folds"],
        "kernel_sources": ["danielsolo1770/rsna-knee-v5-parallel-uint8-cache-288"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ["timm/tf-efficientnet/PyTorch/tf-efficientnet-b3/1"],
        "machine_shape": "NvidiaTeslaT4",
    }
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "kaggle" / "v6_production")
    parser.add_argument("--frozen-epochs", type=int, default=1)
    parser.add_argument("--unfrozen-epochs", type=int, default=7)
    parser.add_argument("--train-slices", type=int, default=8)
    args = parser.parse_args()
    out = args.out_dir
    for fold in range(5):
        nb_name = f"rsna-knee-v6-b3-fold{fold}.ipynb"
        build_fold_notebook(
            out / f"b3_fold{fold}" / nb_name,
            fold,
            f"labels_v3_fold{fold}.csv",
            args.frozen_epochs,
            args.unfrozen_epochs,
            args.train_slices,
        )
        write_meta(
            out / f"b3_fold{fold}" / "kernel-metadata.json",
            f"danielsolo1770/rsna-knee-v6-b3-fold{fold}",
            f"rsna-knee-v6-b3-fold{fold}",
            nb_name,
        )
    print(
        f"done V6: frozen={args.frozen_epochs} unfrozen={args.unfrozen_epochs} "
        f"slices={args.train_slices}"
    )


if __name__ == "__main__":
    main()
