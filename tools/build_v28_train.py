"""Build V28 fold-0 DINOv2 partial-FT trainer (no competition submit)."""
from __future__ import annotations

import json
from pathlib import Path

DST = Path(r"C:\Users\ASUS\Desktop\DanielEmpire\ML\RSNA Knee Abnormality Detection\kaggle\train_v28_dino_ft")

SOURCE = r'''
# V28 fold-0 DINOv2-small partial fine-tune.
# Target path to 0.940: owned 5-fold model, unfrozen last blocks, 224px, 3-slice 2.5D.
# This job is TRAIN ONLY. It never writes a competition submission.csv.
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
T0 = time.time()
SMOKE = False
IMG = 224
CROP_MM = 160.0
N_SLICE = 3
N_SLOT = 3
SLOTS = [("Sagittal", True), ("Coronal", True), ("Axial", True)]
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
EPOCHS = 2 if SMOKE else 6
UNFREEZE_LAST = 4
LR_HEAD = 2e-4
LR_BACKBONE = 2e-5
BATCH = 4
FOLD = 0
SEED = 20260817
ALIVE = 3.8 * 3600


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_dir(*names):
    root = Path("/kaggle/input")
    hits = []
    if not root.is_dir():
        return None
    for p in root.rglob("*"):
        if p.is_dir() and p.name in names:
            hits.append(p)
    return hits[0] if hits else None


def find_file(name):
    root = Path("/kaggle/input")
    if not root.is_dir():
        return None
    for p in root.rglob(name):
        if p.is_file():
            return p
    return None


COMP = find_dir("rsna-knee-abnormality-detection")
assert COMP is not None, "competition data missing"
log(f"comp={COMP}")

import cv2
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"device={DEV}")


def load_soft_labels():
    files = []
    for name in ("report_labels_v2.csv", "llm_labels_v2.csv", "labels_llm_gpt56sol.csv"):
        p = find_file(name)
        if p is None:
            log(f"label missing: {name}")
            continue
        df = pd.read_csv(p, dtype={"StudyInstanceUID": str})
        keep = ["StudyInstanceUID"] + [c for c in TARGETS if c in df.columns]
        files.append(df[keep].set_index("StudyInstanceUID"))
        log(f"label {name} rows={len(df)}")
    assert files, "no label tables"
    idx = files[0].index
    for f in files[1:]:
        idx = idx.union(f.index)
    out = pd.DataFrame(index=idx)
    for t in TARGETS:
        cols = [f[t] for f in files if t in f.columns]
        stacked = pd.concat(cols, axis=1)
        out[t] = stacked.mean(axis=1)
    train = pd.read_csv(COMP / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train.dropna(subset=TARGETS, how="all")
    if len(gold):
        g = gold.set_index("StudyInstanceUID")[TARGETS].apply(pd.to_numeric, errors="coerce")
        out.loc[g.index.intersection(out.index), TARGETS] = g.loc[g.index.intersection(out.index)]
        log(f"gold overwrite {len(g)}")
    out = out.reset_index().rename(columns={"index": "StudyInstanceUID"})
    return out


def report_group(uid: str) -> str:
    return hashlib.md5(uid.encode()).hexdigest()[:8]


def physical_order(sdir: Path):
    keyed = []
    for f in sdir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            iop = np.array(ds.ImageOrientationPatient, dtype=np.float64).reshape(2, 3)
            normal = np.cross(iop[0], iop[1])
            ipp = np.array(ds.ImagePositionPatient, dtype=np.float64)
            key = float(np.dot(ipp, normal))
        except Exception:
            try:
                key = float(getattr(ds, "InstanceNumber", 0) or 0)
            except Exception:
                key = 0.0
        keyed.append((key, f))
    keyed.sort()
    return [f for _, f in keyed]


def read_crop(path: Path):
    try:
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        arr *= float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        arr += float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    except Exception:
        return None
    try:
        ps = float(ds.PixelSpacing[0])
    except Exception:
        ps = CROP_MM / max(arr.shape)
    half = int(round(CROP_MM / max(ps, 1e-3) / 2))
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    y0, y1 = max(0, cy - half), min(arr.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(arr.shape[1], cx + half)
    crop = arr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    lo, hi = np.percentile(crop, [1, 99])
    crop = np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1)
    img = cv2.resize(crop, (IMG, IMG), interpolation=cv2.INTER_AREA)
    return img


def pick_fs(series_df, plane):
    sub = series_df[series_df.Anatomical_Plane == plane]
    if sub.empty:
        return None
    scored = sub.copy()
    if "Fat_Suppression" in scored.columns:
        scored["_fs"] = pd.to_numeric(scored.Fat_Suppression, errors="coerce").fillna(0)
    elif "Fluid_Sensitive" in scored.columns:
        scored["_fs"] = pd.to_numeric(scored.Fluid_Sensitive, errors="coerce").fillna(0)
    else:
        scored["_fs"] = 0
    return scored.sort_values("_fs", ascending=False).iloc[0]


def build_study(uid, recs, series_root):
    x = np.zeros((N_SLOT, N_SLICE, IMG, IMG), np.float32)
    m = np.zeros(N_SLOT, np.uint8)
    rows = pd.DataFrame(recs)
    for i, (plane, _fs) in enumerate(SLOTS):
        rec = pick_fs(rows, plane)
        if rec is None:
            continue
        sdir = series_root / uid / str(rec.SeriesInstanceUID)
        if not sdir.is_dir():
            continue
        files = physical_order(sdir)
        if not files:
            continue
        lo, hi = 0.20, 0.80
        i0 = int(round(lo * (len(files) - 1)))
        i1 = int(round(hi * (len(files) - 1)))
        mid = (i0 + i1) // 2
        picks = [max(0, mid - 1), mid, min(len(files) - 1, mid + 1)]
        for j, p in enumerate(picks[:N_SLICE]):
            img = read_crop(files[p])
            if img is not None:
                x[i, j] = img
                m[i] = 1
    return x, m


class KneeDS(Dataset):
    def __init__(self, uids, y, recs, series_root):
        self.uids = list(uids)
        self.y = y
        self.recs = recs
        self.root = series_root

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        x, m = build_study(uid, self.recs.get(uid, []), self.root)
        # 3 slots x 3 slices -> 9ch then mean-pool slots in the model
        return (
            torch.from_numpy(x),
            torch.from_numpy(m),
            torch.from_numpy(self.y[i].astype(np.float32)),
        )


def load_dinov2():
    from transformers import AutoModel

    cands = [
        Path("/kaggle/input/models/metaresearch/dinov2/pytorch/small/1"),
        Path("/kaggle/input/metaresearch/dinov2/pytorch/small/1"),
    ]
    src = None
    for c in cands:
        if c.exists():
            src = str(c)
            break
    if src is None:
        p = find_dir("small")
        src = str(p) if p else None
    log(f"dinov2 source={src}")
    model = AutoModel.from_pretrained(src, local_files_only=True)
    return model


class PartialFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = load_dinov2()
        dim = int(self.enc.config.hidden_size)
        n_blocks = len(self.enc.encoder.layer)
        for p in self.enc.parameters():
            p.requires_grad = False
        for layer in self.enc.encoder.layer[-UNFREEZE_LAST:]:
            for p in layer.parameters():
                p.requires_grad = True
        log(f"dinov2 dim={dim} blocks={n_blocks} unfrozen_last={UNFREEZE_LAST}")
        self.slot_proj = nn.Linear(dim, 256)
        # RSNA 2024 lumbar 1st (Avengers): attention MIL beat mean-pool on LB.
        self.query = nn.Parameter(torch.randn(len(TARGETS), 256) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        self.aux = nn.Linear(256, len(TARGETS))

    def forward(self, x, m):
        # x: B,S,C,H,W  C=3 adjacent slices as RGB
        b, s, c, h, w = x.shape
        inp = x.reshape(b * s, c, h, w)
        rgb = inp if c == 3 else inp.mean(1, keepdim=True).repeat(1, 3, 1, 1)
        rgb = (rgb - 0.45) / 0.22
        out = self.enc(pixel_values=rgb)
        tok = self.slot_proj(out.last_hidden_state[:, 0]).reshape(b, s, -1)
        mask = m.unsqueeze(-1)
        tok = tok * mask
        attn = torch.einsum("bsd,td->bst", tok, self.query) / 16.0
        attn = attn.masked_fill(m.unsqueeze(-1) == 0, -1e4)
        wgt = torch.softmax(attn, dim=1)
        pooled = torch.einsum("bst,bsd->btd", wgt, tok)
        main = self.head(pooled).squeeze(-1)
        aux = self.aux(tok.mean(1))
        return main, aux


def macro_auc(y, p):
    scores = []
    for j in range(y.shape[1]):
        if len(np.unique(y[:, j] > 0.5)) < 2:
            continue
        try:
            scores.append(roc_auc_score((y[:, j] > 0.5).astype(int), p[:, j]))
        except Exception:
            continue
    return float(np.mean(scores)) if scores else float("nan")


def main():
    labels = load_soft_labels()
    train_csv = pd.read_csv(COMP / "train.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(COMP / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    uids = [u for u in train_csv.StudyInstanceUID.astype(str) if u in set(labels.StudyInstanceUID)]
    labels = labels.set_index("StudyInstanceUID").loc[uids].reset_index()
    groups = np.array([report_group(u) for u in labels.StudyInstanceUID])
    y = labels[TARGETS].to_numpy(np.float32)
    y = np.clip(y, 0, 1)
    gkf = GroupKFold(5)
    folds = list(gkf.split(np.arange(len(uids)), groups=groups))
    tr_idx, va_idx = folds[FOLD]
    if SMOKE:
        tr_idx = tr_idx[:80]
        va_idx = va_idx[:40]
        log("SMOKE subset on")
    recs = {s: g.to_dict("records") for s, g in series.groupby("StudyInstanceUID")}
    series_root = COMP / "train_series"
    ds_tr = KneeDS(labels.StudyInstanceUID.iloc[tr_idx], y[tr_idx], recs, series_root)
    ds_va = KneeDS(labels.StudyInstanceUID.iloc[va_idx], y[va_idx], recs, series_root)
    log(f"fold{FOLD} train={len(ds_tr)} val={len(ds_va)}")
    net = PartialFT().to(DEV)
    params = [
        {"params": [p for n, p in net.named_parameters() if p.requires_grad and "enc." in n], "lr": LR_BACKBONE},
        {"params": [p for n, p in net.named_parameters() if p.requires_grad and "enc." not in n], "lr": LR_HEAD},
    ]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV.type == "cuda")
    work = Path("/kaggle/working/v28_fold0")
    work.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(EPOCHS):
        if time.time() - T0 > ALIVE:
            log("alive limit, stop")
            break
        net.train()
        losses = []
        loader = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0)
        for x, m, t in loader:
            if time.time() - T0 > ALIVE:
                break
            x, m, t = x.to(DEV), m.to(DEV), t.to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=DEV.type == "cuda"):
                logit, aux = net(x, m)
                loss = F.binary_cross_entropy_with_logits(logit, t)
                loss = loss + 0.25 * F.binary_cross_entropy_with_logits(aux, t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        net.eval()
        ys, ps = [], []
        with torch.no_grad():
            for x, m, t in DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=0):
                x, m = x.to(DEV), m.to(DEV)
                with torch.cuda.amp.autocast(enabled=DEV.type == "cuda"):
                    pred = torch.sigmoid(net(x, m)[0]).cpu().numpy()
                ys.append(t.numpy())
                ps.append(pred)
        yv = np.concatenate(ys)
        pv = np.concatenate(ps)
        auc = macro_auc(yv, pv)
        log(f"ep{ep} loss={np.mean(losses):.4f} val_auc={auc:.4f}")
        if auc > best and math.isfinite(auc):
            best = auc
            torch.save({"model": net.state_dict(), "auc": best, "epoch": ep, "fold": FOLD}, work / "best.pt")
            pd.DataFrame(pv, columns=TARGETS).assign(StudyInstanceUID=list(ds_va.uids)).to_csv(work / "val_preds.csv", index=False)
    receipt = {
        "fold": FOLD,
        "best_val_auc": best,
        "smoke": SMOKE,
        "img": IMG,
        "crop_mm": CROP_MM,
        "unfreeze_last": UNFREEZE_LAST,
        "epochs": EPOCHS,
        "n_train": len(ds_tr),
        "n_val": len(ds_va),
        "seconds": time.time() - T0,
    }
    (work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    log(json.dumps(receipt))
    # refuse to emit a competition submission
    sub = Path("/kaggle/working/submission.csv")
    if sub.exists():
        sub.unlink()
        log("removed accidental submission.csv")


main()
'''


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    compile(SOURCE, "v28", "exec")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": [SOURCE],
                "outputs": [],
                "execution_count": None,
            }
        ],
    }
    (DST / "rsna-knee-v28-dino-ft-fold0.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    meta = {
        "id": "danielsolo1770/rsna-knee-v28-dino-ft-fold0",
        "title": "rsna-knee-v28-dino-ft-fold0",
        "code_file": "rsna-knee-v28-dino-ft-fold0.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "pilkwang/rsna-knee-llm-labels",
            "stevenleehans/rsna-knee-llm-report-labels",
            "lixin73/rsna-knee-llm-report-labels-sol56",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
        "machine_shape": "NvidiaTeslaT4",
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("v28 trainer ready", DST)


if __name__ == "__main__":
    main()
