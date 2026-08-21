"""Build a CPU Kaggle kernel that writes the V30 5-depth 224px cache (no GPU, no submit)."""
from __future__ import annotations

import json
from pathlib import Path

DST = Path(r"C:\Users\ASUS\Desktop\DanielEmpire\ML\RSNA Knee Abnormality Detection\kaggle\cache_v30_224")

SOURCE = r'''
# V30 224px 5-depth FS cache. CPU only. Never writes submission.csv.
# Output: /kaggle/working/v30_cache/{x_f16.dat,mask.dat,y.npy,uids.txt,manifest.json}
from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

cv2.setNumThreads(1)
T0 = time.time()
IMG = 224
CROP_MM = 160.0
N_PER_PLANE = 5
N_SLICE = 1
PLANES = ["Sagittal", "Coronal", "Axial"]
N_SLOT = len(PLANES) * N_PER_PLANE
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
ALIVE = 8.2 * 3600
OUT = Path("/kaggle/working/v30_cache")
OUT.mkdir(parents=True, exist_ok=True)


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


def find_prev():
    root = Path("/kaggle/input")
    if not root.is_dir():
        return None
    for p in root.rglob("progress.json"):
        d = p.parent
        if (d / "x_f16.dat").is_file() and (d / "mask.dat").is_file():
            try:
                rec = json.loads(p.read_text())
            except Exception:
                rec = {}
            if rec.get("status") == "complete" and rec.get("n") == 4407:
                continue
            return d
    return None


COMP = find_dir("rsna-knee-abnormality-detection")
assert COMP is not None, "competition data missing"
log(f"comp={COMP}")


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
    return out.reset_index().rename(columns={"index": "StudyInstanceUID"})


def physical_order(sdir: Path):
    keyed = []
    for f in sdir.glob("*.dcm"):
        ds = None
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            iop = np.array(ds.ImageOrientationPatient, dtype=np.float64).reshape(2, 3)
            normal = np.cross(iop[0], iop[1])
            ipp = np.array(ds.ImagePositionPatient, dtype=np.float64)
            key = float(np.dot(ipp, normal))
        except Exception:
            try:
                key = float(getattr(ds, "InstanceNumber", 0) or 0) if ds is not None else 0.0
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
    for pi, plane in enumerate(PLANES):
        rec = pick_fs(rows, plane)
        if rec is None:
            continue
        sdir = series_root / uid / str(rec.SeriesInstanceUID)
        if not sdir.is_dir():
            continue
        files = physical_order(sdir)
        if not files:
            continue
        lo, hi = 0.15, 0.85
        idxs = np.linspace(lo, hi, N_PER_PLANE)
        for j, frac in enumerate(idxs):
            slot = pi * N_PER_PLANE + j
            fi = int(round(float(frac) * (len(files) - 1)))
            fi = min(max(fi, 0), len(files) - 1)
            img = read_crop(files[fi])
            if img is not None:
                x[slot, 0] = img
                m[slot] = 1
    return x, m


def main():
    prev = find_prev()
    if prev is not None and not (OUT / "x_f16.dat").is_file():
        log(f"resume-copy from {prev}")
        for name in ("x_f16.dat", "mask.dat", "progress.json"):
            src = prev / name
            if src.is_file():
                shutil.copy2(src, OUT / name)
    labels = load_soft_labels()
    train_csv = pd.read_csv(COMP / "train.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(COMP / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    uids = [u for u in train_csv.StudyInstanceUID.astype(str) if u in set(labels.StudyInstanceUID)]
    labels = labels.set_index("StudyInstanceUID").loc[uids]
    y = np.clip(labels[TARGETS].to_numpy(np.float32), 0, 1)
    recs = {s: g.to_dict("records") for s, g in series.groupby("StudyInstanceUID")}
    series_root = COMP / "train_series"
    n = len(uids)
    log(f"n={n} slots={N_SLOT}")
    x_path = OUT / "x_f16.dat"
    m_path = OUT / "mask.dat"
    prog_path = OUT / "progress.json"
    start = 0
    if prog_path.is_file() and x_path.is_file() and m_path.is_file():
        try:
            start = int(json.loads(prog_path.read_text())["done"])
            log(f"resume cache at {start}")
        except Exception:
            start = 0
    xmm = np.memmap(x_path, dtype=np.float16, mode="r+" if start else "w+", shape=(n, N_SLOT, N_SLICE, IMG, IMG))
    mmm = np.memmap(m_path, dtype=np.uint8, mode="r+" if start else "w+", shape=(n, N_SLOT))
    n_ok = 0
    for i, uid in enumerate(uids):
        if i < start:
            continue
        if time.time() - T0 > ALIVE:
            prog_path.write_text(json.dumps({"done": i, "n": n, "status": "alive_stop"}) + "\n")
            xmm.flush()
            mmm.flush()
            log(f"alive stop at {i}/{n}")
            return
        xx, mm = build_study(uid, recs.get(uid, []), series_root)
        xmm[i] = xx.astype(np.float16)
        mmm[i] = mm
        n_ok += int(mm.sum() > 0)
        if i % 50 == 0:
            xmm.flush()
            mmm.flush()
            prog_path.write_text(json.dumps({"done": i + 1, "n": n, "n_ok": n_ok}) + "\n")
            log(f"{i+1}/{n} ok_slots_last={int(mm.sum())}")
        if i % 200 == 0:
            gc.collect()
    xmm.flush()
    mmm.flush()
    np.save(OUT / "y.npy", y)
    (OUT / "uids.txt").write_text("\n".join(uids) + "\n", encoding="utf-8")
    man = {
        "n": n,
        "x_shape": [n, N_SLOT, N_SLICE, IMG, IMG],
        "dtype": "float16",
        "img": IMG,
        "crop_mm": CROP_MM,
        "recipe": "v30-224-5depth-3fs-planes",
        "targets": TARGETS,
        "n_ok_last_pass": n_ok,
        "seconds": time.time() - T0,
        "status": "complete",
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    prog_path.write_text(json.dumps({"done": n, "n": n, "status": "complete"}) + "\n")
    log(json.dumps(man))
    sub = Path("/kaggle/working/submission.csv")
    if sub.exists():
        sub.unlink()


main()
'''


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    compile(SOURCE, "v30_cache", "exec")
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
    (DST / "rsna-knee-v30-cache-224.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    meta = {
        "id": "danielsolo1770/rsna-knee-v30-cache-224",
        "title": "rsna-knee-v30-cache-224",
        "code_file": "rsna-knee-v30-cache-224.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [
            "pilkwang/rsna-knee-llm-labels",
            "stevenleehans/rsna-knee-llm-report-labels",
            "lixin73/rsna-knee-llm-report-labels-sol56",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("v30 cache kernel ready", DST)


if __name__ == "__main__":
    main()
