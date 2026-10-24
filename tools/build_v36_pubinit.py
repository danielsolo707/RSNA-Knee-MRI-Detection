"""Build V36: fine-tune the PUBLIC strong checkpoint instead of training from ImageNet.

Why this exists
---------------
V34/V35 start CoAtNet-384 from timm ImageNet weights on the Raptor corpus. After 2
epochs they reach gold-58 macro 0.760, and the public recipe needs ~12 epochs to get
to ~0.917. That is ~54 h of T4 per fold - more than a week of quota.

But `raptor_ft_coatnet_v5_full_swa.pt` (the primary arm of our own 0.936 submission,
hosted at `dreaddevelopment/raptor-knee-maxspan`) is ALREADY trained on this exact
corpus and objective. It saves in the very same dict format that `--resume` expects:

    {"model": state_dict, "arch": str, "res": int}

verified against `load_model()` in the V33 notebook (which does
`model.load_state_dict(ck["model"], strict=True)` and reads `ck["arch"]`, `ck["res"]`).

So fine-tuning FROM it skips ~10 epochs of GPU for free.

Guards added for this run
-------------------------
* `--eval_first` (new flag, added to tools/train_raptor.py) scores the public
  checkpoint on gold-58 BEFORE training and adopts it as the 'best' baseline. Without
  it the first epoch becomes 'best' by default and the run is uninterpretable.
* `--require_resume` fail-closed: abort rather than silently train from ImageNet on a
  paid run if the public checkpoint is missing or unreadable.
* Low LR (bb 1e-5 / head 1e-4 vs 3e-5 / 1e-3 for from-scratch) - the model is converged.
* `--epochs 2` and `--k 10` so the run fits the 9 h Kaggle session cap with margin
  (~7.5 h; V34 measured 4.5 h/epoch at k=12).

Run: python tools/build_v36_pubinit.py
"""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DST = REPO / "kaggle" / "ft_v36_public_init"
TRAINER = (REPO / "tools" / "train_raptor.py").read_text(encoding="utf-8")
TRAINER = TRAINER.replace('if __name__ == "__main__":\n    main()\n', "")
LABELS = REPO / "artifacts" / "raptor_labels" / "labels_llm_soft.parquet"

PUB_CKPT = "/kaggle/input/raptor-knee-maxspan/raptor_ft_coatnet_v5_full_swa.pt"

LAUNCH = r'''
import os, sys, base64, glob
from pathlib import Path
_LABEL_B64 = """__LABEL_B64__"""
lab = Path("/kaggle/working/labels_llm_soft.parquet")
if not lab.is_file():
    lab.write_bytes(base64.b64decode(_LABEL_B64))
print("labels", lab, "bytes", lab.stat().st_size, flush=True)

# Fail-fast inventory: confirm the public checkpoint is really attached and readable.
# NOTE: Kaggle mounts datasets at /kaggle/input/datasets/<owner>/<slug>/ - a hardcoded
# /kaggle/input/<slug>/... path silently resolves to nothing (cost us one probe run).
# Resolve by glob, like V33's find_weight_file() does.
hits = sorted(glob.glob("/kaggle/input/**/raptor_ft_coatnet_v5_full_swa.pt", recursive=True))
print("checkpoints visible under /kaggle/input:", hits, flush=True)
if not hits:
    raise SystemExit(
        "FATAL: raptor_ft_coatnet_v5_full_swa.pt not found under /kaggle/input. "
        "Attach dreaddevelopment/raptor-knee-maxspan. Refusing to start so the GPU "
        "session is not wasted."
    )
PUB_CKPT = hits[0]

sys.argv = [
    "train_raptor.py",
    "--arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",  # overridden by ckpt
    "--res", "384",                                        # overridden by ckpt
    "--epochs", "2",
    "--bs", "2",
    "--k", "10",
    "--k_eval", "24",
    "--grad_ckpt",
    "--ckpt", "timm",
    "--norm", "imagenet",
    "--tag", "v36pub",
    "--labels", str(lab),
    "--out_dir", "/kaggle/working",
    "--workers", "2",
    "--log_every", "10",
    "--bb_lr", "1e-5",        # LOW: fine-tuning an already-converged model
    "--head_lr", "1e-4",
    "--max_wall_h", "8.3",     # stop cleanly before Kaggle's 9 h cap so outputs commit
    "--ckpt_every_min", "15",  # resumable checkpoint every 15 min (epoch ~= 4 h)
    "--resume", PUB_CKPT,
    "--require_resume",   # FAIL-CLOSED if the public checkpoint is missing/invalid
    "--eval_first",       # score the public ckpt on gold-58 BEFORE training
]
print("argv", sys.argv, flush=True)
main()
'''


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    b64 = base64.b64encode(LABELS.read_bytes()).decode("ascii")
    cell = TRAINER + "\n\n" + LAUNCH.replace("__LABEL_B64__", b64).replace("__PUB_CKPT__", PUB_CKPT)
    compile(cell, "v36", "exec")  # syntax gate before writing

    md = [
        "# V36: fine-tune the public 0.935 checkpoint (not from ImageNet)\n",
        "\n",
        "Starts from `raptor_ft_coatnet_v5_full_swa.pt` — the primary CoAtNet arm of our own\n",
        "0.936 submission (dataset `dreaddevelopment/raptor-knee-maxspan`). V34/V35 started from\n",
        "timm ImageNet weights and needed ~12 epochs to reach gold ~0.917; this starts there.\n",
        "\n",
        "- `--eval_first` prints the public checkpoint's gold-58 macro **before** training and\n",
        "  makes it the baseline, so 'did fine-tuning help?' is answerable.\n",
        "- `--require_resume` aborts rather than silently training from scratch on a paid run.\n",
        "- Low LR (1e-5 / 1e-4); 2 epochs at k=10 to fit the 9 h session cap.\n",
        "- Train only; gold-58 held out; never writes `submission.csv`.\n",
    ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": md},
            {"cell_type": "code", "metadata": {}, "source": cell.splitlines(keepends=True),
             "outputs": [], "execution_count": None},
        ],
    }
    (DST / "rsna-knee-v36-public-init.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    if LABELS.is_file():
        shutil.copy2(LABELS, DST / "labels_llm_soft.parquet")

    meta = {
        "id": "danielsolo1770/rsna-knee-v36-public-init",
        "title": "rsna-knee-v36-public-init",
        "code_file": "rsna-knee-v36-public-init.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "dreaddevelopment/knee-raptor-corpus",
            "dreaddevelopment/knee-raptor-corpus-ext",
            "dreaddevelopment/raptor-knee-maxspan",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("V36 public-init fine-tune kernel ready:", DST)
    print("  init checkpoint:", PUB_CKPT)


if __name__ == "__main__":
    main()
