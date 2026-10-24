"""Build V43: train OUR OWN arm from scratch on Kaggle (route B, Kaggle-side).

What this kernel does
---------------------
Trains the v31 DINOv2-S attention-MIL architecture from the ImageNet-pretrained
backbone (no competition weights, no public fork) on the LOO pseudo-labels:

  * train:  4,349 studies with LOO teacher labels (pseudo_labels_LOO_v4_w60.csv)
  * valid:  the 58 gold studies with their TRUE labels (gold never enters training)

So the gold-58 macro AUC this run reports is an HONEST measure of the arm -
unlike V33's gold numbers, which are contaminated by gold-tuning.

GATE: gold-58 macro >= ~0.87 -> fuse with V33 and submit (the V42 verdict says
a weak arm dilutes; we only blend once this arm is competitive).

Why it fits in the 9 h session
------------------------------
  * corpus cache is prebuilt and attached (danielsolo1770/rsna-knee-v31-cache-280,
    the same dataset the fold3/fold4 kernels trained on) - no DICOM reading here;
  * proven Kaggle config from the fold kernels: batch 16, accum 1, workers 2,
    T4; fold4 did 24 epochs comfortably inside the cap;
  * wall-budget guards added to v31probe.py: never starts an epoch it cannot
    finish (30-min reserve), hard-stops mid-epoch at 8-min reserve, writes a
    resumable checkpoint every 20 min - whatever Kaggle does, /kaggle/working
    commits best.pt + last.pt + history.json + receipt.json + gold_preds.npz.

Repo convention honored: this trainer NEVER writes submission.csv.

Run: python tools/build_v43_owned_train.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DST = REPO / "kaggle" / "train_owned_v43"

_trainer = (REPO / "tools" / "v31probe.py").read_text(encoding="utf-8")
# strip the __main__ block: inside a notebook cell __name__ IS "__main__" and
# main() would fire with default Colab paths before our launcher runs.
TRAINER_SRC = _trainer.replace('if __name__=="__main__":\n    main()', "").replace(
    'if __name__ == "__main__":\n    main()', "")
assert "__main__" not in TRAINER_SRC, "failed to strip the __main__ block"

LAUNCH = '''
import torch as _gc_torch
assert _gc_torch.cuda.is_available(), "no CUDA device - kernel must be GPU-enabled"
_gc_cap = _gc_torch.cuda.get_device_capability()
_gc_name = _gc_torch.cuda.get_device_name()
print(f"[gpu] {_gc_name} | capability sm_{_gc_cap[0]}{_gc_cap[1]}", flush=True)
assert _gc_cap[0] >= 7, (
    f"GPU {_gc_name} (sm_{_gc_cap[0]}{_gc_cap[1]}) is too old for the base-image "
    f"torch (needs sm_70+); kernel metadata must pin machine_shape=NvidiaTeslaT4"
)

import glob, os
print("inputs:", os.listdir("/kaggle/input"), flush=True)
for d in os.listdir("/kaggle/input"):
    p = f"/kaggle/input/{d}"
    try:
        print(" ", d, "->", os.listdir(p)[:8], flush=True)
    except Exception as e:
        print(" ", d, "ERR", e, flush=True)

hits = glob.glob("/kaggle/input/**/x_f16.dat", recursive=True)
assert hits, "V31 cache (x_f16.dat) not found - attach danielsolo1770/rsna-knee-v31-cache-280"
cache_dir = os.path.dirname(hits[0])
print("cache at", cache_dir, flush=True)

lab_hits = glob.glob("/kaggle/input/**/pseudo_labels_LOO_v4_w60.csv", recursive=True)
assert lab_hits, "LOO labels not found - attach danielsolo1770/rsna-knee-v43-labels"
labels_csv = lab_hits[0]
print("LOO labels at", labels_csv, flush=True)

cand = glob.glob("/kaggle/input/**/train.csv", recursive=True)
comp = [p for p in cand if "rsna-knee-abnormality-detection" in p]
assert cand, "competition train.csv (gold truth) not found"
gold_csv = (comp or cand)[0]
print("gold truth at", gold_csv, flush=True)

cfg = Cfg(
    cache_dir=cache_dir,
    out_dir="/kaggle/working/v43_out",
    labels_csv=labels_csv,
    gold_holdout=True,
    gold_csv=gold_csv,
    epochs=26,
    batch=16,
    accum=1,
    lr_head=3e-4,
    lr_bb_top=5e-5,
    llrd=0.7,
    patience=6,
    warmup=2,
    ema_decay=0.999,
    num_workers=2,
    max_wall_h=8.3,       # graceful stop before the 9 h session cap
    ckpt_every_min=20.0,  # resumable checkpoint against mid-run kills
)
main(cfg)
'''

PIN = ("import subprocess, sys\n"
       "subprocess.run([sys.executable,'-m','pip','install','-q',"
       "'transformers==4.44.2','accelerate<1.0'], check=True)\n"
       "print('pinned transformers 4.44.2 (newer breaks Dinov2 LayerScale)')\n")

CELL = PIN + "\n" + TRAINER_SRC + "\n\n" + LAUNCH


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    compile(CELL, "v43train", "exec")
    md = [
        "# V43: train our OWN arm from scratch (DINOv2-S on LOO labels)\n",
        "\n",
        "v31 architecture from the pretrained backbone, trained only on the 4,349\n",
        "LOO pseudo-labelled studies; validated on the 58 gold studies with TRUE\n",
        "labels (gold never enters training) -> honest gold-58 macro AUC.\n",
        "\n",
        "Gate: gold-58 >= ~0.87, then fuse with V33. Trainer never writes\n",
        "submission.csv. Wall-budget guards keep the run inside the 9 h cap and\n",
        "commit best.pt / last.pt / history.json / receipt.json / gold_preds.npz.\n",
    ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": md},
            {"cell_type": "code", "metadata": {}, "source": CELL.splitlines(keepends=True),
             "outputs": [], "execution_count": None},
        ],
    }
    (DST / "rsna-knee-v43-owned-train.ipynb").write_text(json.dumps(nb), encoding="utf-8")

    meta = {
        "id": "danielsolo1770/rsna-knee-v43-owned-train",
        "title": "rsna-knee-v43-owned-train",
        "code_file": "rsna-knee-v43-owned-train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,   # pip-pin transformers 4.44.2 (fold4 lesson)
        # v1 (2026-08-31) landed on a P100 (sm_60); the base-image torch only
        # supports sm_70+ -> "CUDA error: no kernel image". Pin T4 like V33/V42.
        "machine_shape": "NvidiaTeslaT4",
        "kernel_sources": ["danielsolo1770/rsna-knee-v31-cache-280"],
        "dataset_sources": ["danielsolo1770/rsna-knee-v43-labels"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n",
                                              encoding="utf-8")
    print("V43 owned-train kernel ready:", DST)
    print("  gate: gold-58 macro >= ~0.87 (honest; gold never in training)")
    print("  needs dataset: danielsolo1770/rsna-knee-v43-labels "
          "(pseudo_labels_LOO_v4_w60.csv)")


if __name__ == "__main__":
    main()
