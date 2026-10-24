"""Build a ~15-min kernel that scores EVERY public CoAtNet checkpoint on gold-58.

Why this is worth doing before any 8-hour training run
-----------------------------------------------------
It buys offline fusion research for the price of a coffee break:

  * gives us the gold-58 macro AUC of each public arm - numbers we do NOT have
    (V33's locally stored log is the 3-study dry run only);
  * dumps each arm's per-study, per-target prediction vector on the 58 gold studies;
  * `tools/blend_gold.py` then searches rank-fusions **offline, for free**, over every
    combination - including the third checkpoint (`v4_full_swa`) that nothing currently
    loads. That tests Track A's whole premise with zero GPU and zero submits.

The three checkpoints:

  v5_full_swa  dreaddevelopment/raptor-knee-maxspan     primary arm of our 0.936
  v4_full      dreaddevelopment/raptor-knee-widedense   complement arm of our 0.936
  v4_full_swa  dreaddevelopment/raptor-knee-widedense   **loaded by nothing at all**

Run: python tools/build_arm_gold_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DST = REPO / "kaggle" / "probe_arm_gold"
TRAINER = (REPO / "tools" / "train_raptor.py").read_text(encoding="utf-8")
TRAINER = TRAINER.replace('if __name__ == "__main__":\n    main()\n', "")

LAUNCH = r'''
# ---- score every public CoAtNet checkpoint on the 58 gold studies ----
import os, gc, glob, json
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

dev = "cuda" if torch.cuda.is_available() else "cpu"
use_cuda = dev == "cuda"
bf16 = use_cuda and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if bf16 else torch.float16
print(f"device {dev} | gpus {torch.cuda.device_count()} | bf16={bf16}", flush=True)

print("checkpoints visible:", glob.glob("/kaggle/input/**/raptor_ft_coatnet_*.pt", recursive=True),
      flush=True)

# --- gold-58 setup (mirrors main() so the eval geometry is identical) ---
ids = _open_ids()
id2row = {u: i for i, u in enumerate(ids)}
idset = set(ids)
_tcsv = os.path.join(RSNA, "train.csv")
if not os.path.exists(_tcsv):
    _tcsv = _find("train.csv")
tr = pd.read_csv(_tcsv); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
gold_ids = [u for u in gold_df.index if u in idset]
labels = {u: gold_df.loc[u, LAB].values.astype(np.float32) for u in gold_ids}
print("gold studies available:", len(gold_ids), flush=True)

_roi_kw = dict(roi=False, roi_mode="compartment", roi_pad=0.06, roi_overlap=0.12,
               roi_sbox=None, roi_cen=None)

K_EVAL = 24


def find_ckpt(fname, required=True):
    """Kaggle mounts datasets under /kaggle/input/datasets/<owner>/<slug>/.

    Hardcoding /kaggle/input/<slug>/... silently misses them (this cost us one probe run).
    V33's inference kernel works because it globs - do the same here.
    """
    hits = sorted(glob.glob(f"/kaggle/input/**/{fname}", recursive=True))
    if hits:
        return hits[0]
    if required:
        raise RuntimeError(f"{fname} not found under /kaggle/input")
    return None


ARMS = [
    ("v5_full_swa", "raptor_ft_coatnet_v5_full_swa.pt"),
    ("v4_full",     "raptor_ft_coatnet_v4_full.pt"),
    ("v4_full_swa", "raptor_ft_coatnet_v4_full_swa.pt"),
]

preds, summary, Y_all = {}, {}, None
for name, fname in ARMS:
    path = find_ckpt(fname, required=False)
    if not path:
        print(f"[skip] {name}: {fname} not found", flush=True)
        continue
    print(f"[{name}] -> {path}", flush=True)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    res = int(ck.get("res", 384))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(dev)
    ds = StudyWindows(HERE, gold_ids, id2row, labels, res, K_EVAL, train=False,
                      norm="imagenet", **_roi_kw)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate)
    P, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            with torch.autocast(dev, dtype=amp_dtype, enabled=use_cuda):
                o = torch.sigmoid(model(x.to(dev)).float())
            P.append(o.cpu().numpy()); Y.append(y.numpy())
    P = np.concatenate(P); Y_all = np.concatenate(Y)
    aucs = {}
    for j, t in enumerate(LAB):
        if len(set(Y_all[:, j].astype(int))) > 1:
            aucs[t] = float(roc_auc_score(Y_all[:, j], P[:, j]))
    macro = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    summary[name] = {"macro": macro, "per_target": aucs, "arch": arch, "res": res,
                     "file": os.path.basename(path)}
    preds[name] = P
    print(f"{name:12s} res={res} macro={macro:.4f}", flush=True)
    print("   weakest: " + ", ".join(f"{k}={v:.3f}" for k, v in
                                     sorted(aucs.items(), key=lambda kv: kv[1])[:4]), flush=True)
    del model, ck, ds, dl
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

if preds:
    np.savez("/kaggle/working/arm_gold_preds.npz",
             **{f"pred_{k}": v for k, v in preds.items()},
             truth=Y_all, ids=np.array(gold_ids), lab=np.array(LAB))
    json.dump({"arms": summary, "n_gold": len(gold_ids), "k_eval": K_EVAL, "lab": LAB,
               "summary_macro": {k: v["macro"] for k, v in summary.items()}},
              open("/kaggle/working/arm_gold_preds.json", "w"), indent=1)
    print("saved arm_gold_preds.npz / arm_gold_preds.json", flush=True)
    print("MACRO:", json.dumps({k: round(v['macro'], 4) for k, v in summary.items()}), flush=True)
else:
    print("NO ARMS EVALUATED - check the attached datasets", flush=True)
'''


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    cell = TRAINER + "\n\n" + LAUNCH
    compile(cell, "armprobe", "exec")

    md = [
        "# Probe: score every public CoAtNet checkpoint on gold-58\n",
        "\n",
        "No training. Loads all three public checkpoints, evaluates each on the 58 gold\n",
        "studies at k_eval=24, and dumps per-study prediction vectors.\n",
        "\n",
        "Purpose: `tools/blend_gold.py` can then search rank-fusions **offline and free**,\n",
        "including `v4_full_swa` - a checkpoint that nothing currently loads. This tests the\n",
        "whole 'add a third vote' premise without spending a submit.\n",
        "\n",
        "Expected cost: ~15 min GPU.\n",
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
    (DST / "arm-gold-probe.ipynb").write_text(json.dumps(nb), encoding="utf-8")

    meta = {
        "id": "danielsolo1770/arm-gold-probe",
        "title": "arm-gold-probe",
        "code_file": "arm-gold-probe.ipynb",
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
            "dreaddevelopment/raptor-knee-widedense",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n",
                                              encoding="utf-8")
    print("arm-gold probe ready:", DST)
    print("  evaluates: v5_full_swa, v4_full, v4_full_swa")


if __name__ == "__main__":
    main()
