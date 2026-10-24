"""Build V38: fine-tune the public v5_full_swa checkpoint on the FUSED PSEUDO-LABELS.

Same recipe as V36 (fine-tune, not from-scratch: LR 1e-5/1e-4, 2 epochs, k=10,
eval_first baseline, require_resume fail-closed) but the training labels are
`artifacts/pseudo_labels/pseudo_labels_final_w60.parquet` instead of the raw
raptor soft labels.

Why these labels are better (all measured out-of-sample on gold-58, see
artifacts/pseudo_labels/fusion_report.json):

    raptor soft labels (v4_blend proxy)      0.8927
    dual-CoAtNet arm fusion                  0.9171
    blend 40% labels + 60% arm fusion        0.9292   <- what V38 trains on

+0.036 label AUC over the raw labels, ~9x the effect size that failed to
transfer in V37 (n=58 noise scale), so it should survive measurement noise.
The arm component is in-sample on the 4,349 train studies (the public
checkpoints were trained there) - mitigated by the 40% label blend at a single
global ratio, never per-target (V37 lesson).

Gate: eval_first prints the v5_full_swa baseline (expect ~0.9129). The
fine-tune must beat that or the run is a no-submit. Best checkpoint is selected
by gold-58 macro, so distillation-that-hurts cannot silently win.

Run: python tools/build_v38_pseudo_ft.py
"""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DST = REPO / "kaggle" / "ft_v38_pseudo_ft"
TRAINER = (REPO / "tools" / "train_raptor.py").read_text(encoding="utf-8")
TRAINER = TRAINER.replace('if __name__ == "__main__":\n    main()\n', "")
LABELS = REPO / "artifacts" / "pseudo_labels" / "pseudo_labels_final_w60.parquet"

LAUNCH = r'''
import os, sys, base64, glob
from pathlib import Path
_LABEL_B64 = """__LABEL_B64__"""
lab = Path("/kaggle/working/pseudo_labels_final_w60.parquet")
if not lab.is_file():
    lab.write_bytes(base64.b64decode(_LABEL_B64))
print("labels", lab, "bytes", lab.stat().st_size, flush=True)

# Fail-fast inventory: confirm the public checkpoint is really attached and readable.
# Resolve by glob - Kaggle mounts datasets at /kaggle/input/datasets/<owner>/<slug>/
# and hardcoded /kaggle/input/<slug>/ paths silently resolve to nothing.
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
    "--ft_epochs", "2",     # resume-aware: run 2 epochs AFTER start_ep (public SWA ckpt
                            # has epoch=[14,12,15] -> start_ep=16; plain --epochs 2 would
                            # make range(16,2) empty = silent no-training run)
    "--bs", "2",
    "--k", "10",
    "--k_eval", "24",
    "--grad_ckpt",
    "--ckpt", "timm",
    "--norm", "imagenet",
    "--tag", "v38pseudo",
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
    if not LABELS.is_file():
        raise SystemExit(f"missing {LABELS} - run tools/fuse_pseudo_labels.py first")
    DST.mkdir(parents=True, exist_ok=True)
    b64 = base64.b64encode(LABELS.read_bytes()).decode("ascii")
    cell = TRAINER + "\n\n" + LAUNCH.replace("__LABEL_B64__", b64)
    compile(cell, "v38", "exec")  # syntax gate before writing

    md = [
        "# V38: fine-tune the public 0.935 checkpoint on fused pseudo-labels\n",
        "\n",
        "Same recipe as V36 (fine-tune from `raptor_ft_coatnet_v5_full_swa.pt`, LR 1e-5/1e-4,\n",
        "2 epochs at k=10) but the training labels are the two-arm pseudo-label blend:\n",
        "\n",
        "| labels | gold-58 AUC |\n",
        "|---|---|\n",
        "| raw raptor soft labels | 0.8927 |\n",
        "| dual-CoAtNet arm fusion | 0.9171 |\n",
        "| **40% labels + 60% fusion (this run)** | **0.9292** |\n",
        "\n",
        "- `--eval_first` prints the public checkpoint's gold-58 macro **before** training and\n",
        "  makes it the baseline (expect ~0.9129); the fine-tune must beat it.\n",
        "- `--require_resume` aborts rather than silently training from scratch on a paid run.\n",
        "- Best checkpoint selected by gold-58 macro, so the gate is enforced in-run.\n",
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
    (DST / "rsna-knee-v38-pseudo-ft.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    shutil.copy2(LABELS, DST / "pseudo_labels_final_w60.parquet")

    meta = {
        "id": "danielsolo1770/rsna-knee-v38-pseudo-ft",
        "title": "rsna-knee-v38-pseudo-ft",
        "code_file": "rsna-knee-v38-pseudo-ft.ipynb",
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
    print("V38 pseudo-label fine-tune kernel ready:", DST)
    print("  labels:", LABELS.name, LABELS.stat().st_size, "bytes")
    print("  init checkpoint: raptor_ft_coatnet_v5_full_swa.pt (attached dataset)")


if __name__ == "__main__":
    main()
