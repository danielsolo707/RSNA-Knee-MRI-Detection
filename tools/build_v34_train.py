"""Build Kaggle GPU trainer: V34 fold-0 CoAtNet on the public Raptor 336px corpus.

Gold-58 is held out (the public trainer default). Does NOT write submission.csv.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DST = REPO / "kaggle" / "train_v34_raptor_f0"
TRAINER = (REPO / "tools" / "train_raptor.py").read_text(encoding="utf-8")
TRAINER = TRAINER.replace('if __name__ == "__main__":\n    main()\n', "")
LABELS = REPO / "artifacts" / "raptor_labels" / "labels_llm_soft.parquet"

LAUNCH = r'''
import os, sys, shutil, base64
from pathlib import Path
_LABEL_B64 = """__LABEL_B64__"""
lab = Path("/kaggle/working/labels_llm_soft.parquet")
if not lab.is_file():
    lab.write_bytes(base64.b64decode(_LABEL_B64))
print("labels", lab, "bytes", lab.stat().st_size, flush=True)
print("inputs:", os.listdir("/kaggle/input"), flush=True)
for d in os.listdir("/kaggle/input"):
    p = f"/kaggle/input/{d}"
    try:
        print(" ", d, "->", os.listdir(p)[:12], flush=True)
    except Exception as e:
        print(" ", d, "ERR", e, flush=True)

sys.argv = [
    "train_raptor.py",
    "--arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
    "--res", "384",
    "--epochs", "12",
    "--bs", "2",
    "--k", "12",
    "--k_eval", "24",
    "--grad_ckpt",
    "--ckpt", "timm",
    "--norm", "imagenet",
    "--tag", "v34f0",
    "--labels", str(lab),
    "--out_dir", "/kaggle/working",
    "--workers", "2",
    "--log_every", "10",
    "--resume", "",
]
print("argv", sys.argv, flush=True)
main()
'''


def main() -> None:
    import base64
    DST.mkdir(parents=True, exist_ok=True)
    b64 = base64.b64encode(LABELS.read_bytes()).decode("ascii")
    cell = TRAINER + "\n\n" + LAUNCH.replace("__LABEL_B64__", b64)
    compile(cell, "v34", "exec")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# V34 fold-0 owned CoAtNet on Raptor 336px corpus\n",
                    "\n",
                    "Train only. Gold-58 held out. Never writes `submission.csv`.\n",
                    "T4: fp16 + grad checkpoint + bs=2. Gate on gold macro-AUC.\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": cell.splitlines(keepends=True),
                "outputs": [],
                "execution_count": None,
            },
        ],
    }
    (DST / "rsna-knee-v34-raptor-f0.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    if LABELS.is_file():
        shutil.copy2(LABELS, DST / "labels_llm_soft.parquet")
    meta = {
        "id": "danielsolo1770/rsna-knee-v34-raptor-f0",
        "title": "rsna-knee-v34-raptor-f0",
        "code_file": "rsna-knee-v34-raptor-f0.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [
            "dreaddevelopment/knee-raptor-corpus",
            "dreaddevelopment/knee-raptor-corpus-ext",
        ],
        "kernel_sources": [],
        "competition_sources": [
            "rsna-knee-abnormality-detection",
        ],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("V34 kernel ready:", DST)


if __name__ == "__main__":
    main()
