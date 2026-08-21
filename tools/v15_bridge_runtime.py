#!/usr/bin/env python3
"""Build V15 bridge volumes [3,16,256,256] for all 4,407 training studies (CPU kernel).

Same recipe as inference_runtime.py (fluid-sensitive FS series per plane, physical
order, central 16-slice window, resize 256, per-volume uint8 min/max). Output:
chunked npz + manifest with SHA256 per chunk, published as a private dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

PLANES = ["AXIAL", "CORONAL", "SAGITTAL"]
DEPTH = 16
SIZE = 256
DEPTH_INDICES = np.array([0, 2, 4, 6, 9, 11, 13, 15], dtype=np.int64)
CHUNK = 64  # studies per npz chunk
WORKERS = 8
UID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def truthy(value) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def dicom_files(series_dir: Path) -> list:
    return sorted(
        p for p in series_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".dcm"
    )


def choose_series(series_rows, study_uid: str, plane: str, root: Path):
    import pandas as pd

    rows = series_rows.get(str(study_uid))
    if rows is None:
        return None
    candidates = []
    for row in rows.itertuples(index=False):
        row_plane = str(getattr(row, "Anatomical_Plane", "")).strip().upper()
        if row_plane != plane:
            continue
        series_uid = str(getattr(row, "SeriesInstanceUID"))
        folder = root / "train_series" / str(study_uid) / series_uid
        if not folder.is_dir():
            continue
        files = dicom_files(folder)
        if not files:
            continue
        fluid = truthy(getattr(row, "Fluid_Sensitive", False))
        fatsat = truthy(getattr(row, "Fat_Suppression", False))
        enough = len(files) >= DEPTH
        score = (4 * fluid, 3 * fatsat, 2 * enough, min(len(files), 128))
        candidates.append((score, series_uid, folder, files, fluid, fatsat))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def header_key(path: Path):
    import pydicom

    projection = instance = None
    try:
        dataset = pydicom.dcmread(
            path, stop_before_pixels=True, force=True,
            specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"],
        )
        try:
            ipp = np.asarray(dataset.ImagePositionPatient[:3], dtype=np.float64)
            iop = np.asarray(dataset.ImageOrientationPatient[:6], dtype=np.float64)
            projection = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            projection = None
        try:
            instance = float(dataset.InstanceNumber)
        except Exception:
            instance = None
    except Exception:
        pass
    return projection, instance


def physical_order(files: list):
    import numpy as _np

    rows = [(p, *header_key(p), pos) for pos, p in enumerate(files)]
    usable_projection = sum(_np.isfinite(r[1]) for r in rows if r[1] is not None)
    usable_instance = sum(_np.isfinite(r[2]) for r in rows if r[2] is not None)
    need = max(2, int(_np.ceil(0.8 * len(rows))))
    if usable_projection >= need:
        finite = [r[1] for r in rows if r[1] is not None and _np.isfinite(r[1])]
        fallback = float(_np.median(finite))
        rows.sort(key=lambda r: (r[1] if r[1] is not None else fallback, r[3]))
        method = "geometry_projection"
    elif usable_instance >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float("inf"), r[3]))
        method = "instance_number"
    else:
        rows.sort(key=lambda r: r[0].name)
        method = "filename_fallback"
    return [r[0] for r in rows], method


def central_window(files: list) -> list:
    if len(files) >= DEPTH:
        center = len(files) // 2
        return files[center - DEPTH // 2: center + DEPTH // 2]
    if not files:
        return []
    missing = DEPTH - len(files)
    left = missing // 2
    right = missing - left
    return [files[0]] * left + files + [files[-1]] * right


def read_slice(path: Path):
    import cv2
    import pydicom

    try:
        dataset = pydicom.dcmread(path, force=True)
        image = dataset.pixel_array.astype(np.float32)
        image = image * float(getattr(dataset, "RescaleSlope", 1) or 1)
        image = image + float(getattr(dataset, "RescaleIntercept", 0) or 0)
        if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            image = image.max() + image.min() - image
        return image
    except Exception:
        return None


def read_volume(files: list):
    import cv2

    selected = central_window(files)
    images = [read_slice(p) for p in selected]
    good = [i for i, im in enumerate(images) if im is not None]
    if not good:
        return None, len(selected)
    for i, im in enumerate(images):
        if im is None:
            images[i] = images[min(good, key=lambda other: abs(other - i))]
    resized = np.stack(
        [cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR) for im in images],
        axis=0,
    ).astype(np.float32)
    lo, hi = float(resized.min()), float(resized.max())
    if not np.isfinite(lo + hi) or hi <= lo:
        return None, len(selected)
    normalized = np.clip((resized - lo) / (hi - lo), 0.0, 1.0)
    volume = np.rint(normalized * 255.0).astype(np.uint8)
    if volume.shape != (DEPTH, SIZE, SIZE):
        raise RuntimeError(f"volume bridge shape failed: {volume.shape}")
    return volume, len(selected) - len(good)


def process_study(study_uid: str, series_rows, root: Path):
    volume = np.zeros((len(PLANES), DEPTH, SIZE, SIZE), dtype=np.uint8)
    presence = np.zeros(len(PLANES), dtype=np.float32)
    receipt = {"StudyInstanceUID": str(study_uid), "planes": {}}
    for plane_index, plane in enumerate(PLANES):
        chosen = choose_series(series_rows, str(study_uid), plane, root)
        if chosen is None:
            receipt["planes"][plane] = {"present": False, "reason": "no_series"}
            continue
        score, series_uid, _folder, files, fluid, fatsat = chosen
        ordered, order_method = physical_order(files)
        plane_volume, decode_fills = read_volume(ordered)
        if plane_volume is None:
            receipt["planes"][plane] = {
                "present": False, "reason": "decode_or_constant_failure",
                "SeriesInstanceUID": series_uid,
            }
            continue
        volume[plane_index] = plane_volume
        presence[plane_index] = 1.0
        receipt["planes"][plane] = {
            "present": True,
            "SeriesInstanceUID": series_uid,
            "source_slices": len(files),
            "fluid_sensitive": bool(fluid),
            "fat_suppression": bool(fatsat),
            "selection_score": list(score),
            "order_method": order_method,
            "decode_fills": int(decode_fills),
        }
    return volume, presence, receipt


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_root(default_root: Path) -> Optional[Path]:
    if (default_root / "train.csv").is_file() and (default_root / "train_series.csv").is_file():
        return default_root
    base = Path("/kaggle/input")
    if base.is_dir():
        for depth1 in sorted(p for p in base.iterdir() if p.is_dir()):
            for cand in [depth1] + sorted(p for p in depth1.iterdir() if p.is_dir()):
                if (cand / "train.csv").is_file() and (cand / "train_series.csv").is_file():
                    return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/kaggle/input/rsna-2024-knee-competition"))
    ap.add_argument("--out-dir", type=Path, default=Path("/kaggle/working/v15_bridge"))
    ap.add_argument("--train-csv", type=Path, default=None,
                    help="train.csv path (default: <root>/train.csv)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all studies")
    args = ap.parse_args()

    import pandas as pd
    import pydicom  # noqa: F401  (import check)

    root = args.root
    if args.train_csv is None:
        root = _resolve_root(root)
        if root is None:
            raise RuntimeError(
                "competition mount not found: need train.csv + train_series.csv "
                "under --root (or nested one level under /kaggle/input)"
            )
    train_csv = args.train_csv or (root / "train.csv")
    train = pd.read_csv(train_csv, dtype={UID: str})
    if UID not in train or train[UID].isna().any() or not train[UID].is_unique:
        raise RuntimeError("train identity contract failed")
    series = pd.read_csv(root / "train_series.csv", dtype={UID: str, "SeriesInstanceUID": str})
    series_rows = {str(s): g.copy() for s, g in series.groupby(UID, sort=False)}

    uids = train[UID].astype(str).tolist()
    if args.limit > 0:
        uids = uids[: args.limit]

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "recipe": "V55-bridge",
        "planes": PLANES,
        "depth": DEPTH,
        "size": SIZE,
        "depth_indices": DEPTH_INDICES.tolist(),
        "stored_depth": int(len(DEPTH_INDICES)),
        "chunk": CHUNK,
        "studies": len(uids),
        "chunks": [],
        "started": time.time(),
    }

    chunks = [uids[i: i + CHUNK] for i in range(0, len(uids), CHUNK)]
    receipts = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for chunk_index, chunk_uids in enumerate(chunks):
            processed = list(pool.map(lambda u: process_study(u, series_rows, root), chunk_uids))
            volumes = np.stack([item[0] for item in processed])  # (n,3,16,256,256)
            # Store only the depth indices the student reads: halves dataset size.
            volumes = volumes[:, :, DEPTH_INDICES]  # (n,3,8,256,256)
            presence = np.stack([item[1] for item in processed])
            receipts.extend(item[2] for item in processed)

            path = out / f"chunk_{chunk_index:04d}_volume.npy"
            np.save(path, volumes, allow_pickle=False)
            presence_path = out / f"chunk_{chunk_index:04d}_presence.npy"
            np.save(presence_path, presence, allow_pickle=False)
            digest = sha256_bytes(path.read_bytes())
            presence_digest = sha256_bytes(presence_path.read_bytes())
            manifest["chunks"].append(
                {"index": chunk_index, "studies": len(chunk_uids),
                 "study_start": chunk_index * CHUNK,
                 "study_stop": chunk_index * CHUNK + len(chunk_uids),
                 "volume_file": path.name, "volume_bytes": int(path.stat().st_size),
                 "volume_sha256": digest,
                 "presence_file": presence_path.name,
                 "presence_bytes": int(presence_path.stat().st_size),
                 "presence_sha256": presence_digest}
            )
            log(f"chunk {chunk_index}/{len(chunks)} done bytes={manifest['chunks'][-1]['volume_bytes']}")

    present = sum(r["planes"][p].get("present", False) for r in receipts for p in PLANES)
    manifest["present_plane_volumes"] = present
    manifest["elapsed_seconds"] = round(time.time() - manifest["started"], 3)
    manifest_path = out / "v15_bridge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_sha256"] = sha256_bytes(manifest_path.read_bytes())

    receipt = {
        "status": "V15_BRIDGE_BUILD_COMPLETE",
        "studies": len(uids),
        "chunks": len(chunks),
        "present_plane_volumes": present,
        "elapsed_seconds": manifest["elapsed_seconds"],
        "manifest_sha256": manifest["manifest_sha256"],
    }
    receipt_path = out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    log("V15_BRIDGE_BUILD_COMPLETE " + json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())