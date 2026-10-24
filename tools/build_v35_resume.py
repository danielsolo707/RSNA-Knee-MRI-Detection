"""Build a clean V35 fold-0 resume kernel from the fixed trainer.

The existing train_v35_raptor_f0_resume.ipynb was left in an inconsistent state
by incremental text patches (fail-closed guard present but --require_resume flag
missing -> would AttributeError at runtime). This rebuilds it deterministically
from tools/train_raptor.py with:

  * --resume  (auto-discovers raptor_ft_v34f0.pt under /kaggle/input)
  * --require_resume  (FAIL-CLOSED: abort if the fold-0 checkpoint is not found)
  * kernel_sources -> v34 so the fold-0 checkpoint comes in as a kernel output

Run: python tools/build_v35_resume.py
"""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DST = REPO / "kaggle" / "train_v35_raptor_f0_resume"
TRAINER = (REPO / "tools" / "train_raptor.py").read_text(encoding="utf-8")
# do not execute main() when imported as a module; the launch cell calls it.
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
    "--resume", "",          # auto-discovers raptor_ft_v34f0.pt under /kaggle/input
    "--require_resume",      # FAIL-CLOSED if discovered checkpoint is missing/invalid
]
print("argv", sys.argv, flush=True)
main()
'''


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    b64 = base64.b64encode(LABELS.read_bytes()).decode("ascii")
    cell = TRAINER + "\n\n" + LAUNCH.replace("__LABEL_B64__", b64)
    compile(cell, "v35", "exec")  # syntax gate before writing
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# V35 fold-0 resume: CoAtNet-384, Raptor 336px corpus\n",
                    "\n",
                    "Continues the V34 fold-0 trainer from its saved best-epoch checkpoint. ",
                    "Train only; gold-58 held out; never writes `submission.csv`.\n",
                    "`--require_resume` fails-closed if the V34 checkpoint is not retrievable\n",
                    "(aborts instead of silently retraining from scratch on a paid GPU run).\n",
                ],
            },
            {"cell_type": "code", "metadata": {}, "source": cell.splitlines(keepends=True),
             "outputs": [], "execution_count": None},
        ],
    }
    (DST / "rsna-knee-v35-raptor-f0-resume.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    if LABELS.is_file():
        shutil.copy2(LABELS, DST / "labels_llm_soft.parquet")
    meta = {
        "id": "danielsolo1770/rsna-knee-v35-raptor-f0-resume",
        "title": "rsna-knee-v35-raptor-f0-resume",
        "code_file": "rsna-knee-v35-raptor-f0-resume.ipynb",
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
        "kernel_sources": [
            "danielsolo1770/rsna-knee-v34-raptor-f0",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("clean V35 resume kernel ready:", DST)


if __name__ == "__main__":
    main()