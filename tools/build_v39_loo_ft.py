"""Build V39: fine-tune the public v5_full_swa checkpoint on LEAVE-ONE-OUT pseudo-labels.

V38 trained v5 on a teacher that INCLUDED v5's own in-sample predictions
(0.6 * rank(fused v4+v5) + 0.4 * rank(soft)). Gold-58 collapsed 0.9131 -> 0.9012
in one epoch and never recovered (artifacts/v38_out, receipt best_gold = the
eval_first baseline). Self-distillation: the student was pushed toward a
sharpened copy of its own overfit in-sample predictions = confirmation bias.

V39 fixes the teacher: the pseudo-labels are built from the v4 arm ONLY
(leave-one-out wrt the student) blended with the soft labels:

    pseudo = (1-w) * rank(soft) + w * rank(v4 preds),   w = 0.60

Measured out-of-sample on gold-58 (tools/fuse_pseudo_labels.py, report
artifacts/pseudo_labels/fusion_report_LOO_v4.json):

    soft labels only                    0.8927
    v4 arm preds only                   0.9141
    LOO blend (this run, w=0.60)        0.9269

The teacher is now independent of the student (no v5 component), so the label
signal can transfer instead of reinforcing the student's own errors. +0.034
label AUC over soft, still ~8x the V37 effect that failed to transfer on n=58.

Everything else is identical to V38 (single-variable change = the teacher):
resume v5_full_swa, LR 1e-5/1e-4, 2 epochs, k=10, k_eval=24, eval_first,
require_resume, best-checkpoint-by-gold. Gate: the fine-tune must beat the
eval_first baseline (~0.9131) or the run is a no-submit (best ckpt = baseline).

Run: python tools/build_v39_loo_ft.py
"""
from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DST = REPO / "kaggle" / "ft_v39_loo_ft"
TRAINER = (REPO / "tools" / "train_raptor.py").read_text(encoding="utf-8")
TRAINER = TRAINER.replace('if __name__ == "__main__":\n    main()\n', "")
LABELS = REPO / "artifacts" / "pseudo_labels" / "pseudo_labels_LOO_v4_w60.parquet"
LABEL_NAME = "pseudo_labels_LOO_v4_w60.parquet"

LAUNCH = r'''
import os, sys, base64, glob
from pathlib import Path
_LABEL_B64 = """__LABEL_B64__"""
lab = Path("/kaggle/working/__LABEL_NAME__")
if not lab.is_file():
    lab.write_bytes(base64.b64decode(_LABEL_B64))
print("labels", lab, "bytes", lab.stat().st_size, flush=True)

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
    "--tag", "v39loo",
    "--labels", str(lab),
    "--out_dir", "/kaggle/working",
    "--workers", "2",
    "--log_every", "10",
    "--bb_lr", "1e-5",        # LOW: fine-tuning an already-converged model (same as V38;
    "--head_lr", "1e-4",      # the single variable changed in V39 is the teacher)
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
        raise SystemExit(
            f"missing {LABELS} - build it with:\n"
            "  python tools/fuse_pseudo_labels.py --w-v5 0.0 --w-v4 1.0 --blend-w 0.6 "
            f"--ref-parquet artifacts/pseudo_labels/__no_ref_loo__.parquet --out {LABELS}")
    DST.mkdir(parents=True, exist_ok=True)
    b64 = base64.b64encode(LABELS.read_bytes()).decode("ascii")
    cell = TRAINER + "\n\n" + LAUNCH.replace("__LABEL_B64__", b64).replace(
        "__LABEL_NAME__", LABEL_NAME)
    compile(cell, "v39", "exec")  # syntax gate before writing

    md = [
        "# V39: fine-tune the public 0.935 checkpoint on LEAVE-ONE-OUT pseudo-labels\n",
        "\n",
        "V38 trained v5 on a teacher that included v5's OWN in-sample predictions and\n",
        "collapsed (gold 0.9131 -> 0.9012, best = the eval_first baseline). V39 fixes the\n",
        "teacher: pseudo-labels come from the **v4 arm only** (leave-one-out wrt the\n",
        "student) blended with the soft labels, so the teacher is independent of the\n",
        "student.\n",
        "\n",
        "| teacher labels | gold-58 AUC |\n",
        "|---|---|\n",
        "| raw raptor soft labels | 0.8927 |\n",
        "| v4 arm preds only | 0.9141 |\n",
        "| **LOO blend 40% soft + 60% v4 (this run)** | **0.9269** |\n",
        "| (V38's self-inclusive blend, for reference) | 0.9292 |\n",
        "\n",
        "Single-variable change vs V38 = the teacher. LR, epochs, k, gates all identical.\n",
        "\n",
        "- `--eval_first` prints the public checkpoint's gold-58 macro **before** training and\n",
        "  makes it the baseline (expect ~0.9129); the fine-tune must beat it.\n",
        "- `--require_resume` aborts rather than silently training from scratch on a paid run.\n",
        "- Best checkpoint selected by gold-58 macro, so the gate is enforced in-run; a\n",
        "  distillation that hurts returns the baseline checkpoint (no-submit).\n",
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
    (DST / "rsna-knee-v39-loo-ft.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    shutil.copy2(LABELS, DST / LABEL_NAME)

    meta = {
        "id": "danielsolo1770/rsna-knee-v39-loo-ft",
        "title": "rsna-knee-v39-loo-ft",
        "code_file": "rsna-knee-v39-loo-ft.ipynb",
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
    print("V39 LOO pseudo-label fine-tune kernel ready:", DST)
    print("  labels:", LABELS.name, LABELS.stat().st_size, "bytes")
    print("  teacher: v4 arm only (leave-one-out) + soft, gold-58 label AUC 0.9269")
    print("  init checkpoint: raptor_ft_coatnet_v5_full_swa.pt (attached dataset)")


if __name__ == "__main__":
    main()
