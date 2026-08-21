#!/usr/bin/env python3
"""Parallel, resume-safe uint8 MRI cache for RSNA Knee V3 production training.

Writes one compressed npz per study:
  images: uint8 [3, slices, H, W]
  mask:   uint8 [3, slices]

Designed for Kaggle CPU notebooks. Sequential decoding of 4,407 multi-series
studies is too slow for a single session; this uses a process pool and atomic
per-study writes so interrupted runs resume cleanly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

UID = "StudyInstanceUID"
PLANES = ("Sagittal", "Coronal", "Axial")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def natural_key(path: Path):
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


def pick_series_slots(series: pd.DataFrame, count: int = 3) -> list[str | None]:
    required = {
        "SeriesInstanceUID",
        "Anatomical_Plane",
        "Fluid_Sensitive",
        "Fat_Suppression",
    }
    missing = required.difference(series.columns)
    if missing:
        raise ValueError(f"series table missing columns: {sorted(missing)}")
    frame = series.copy()
    for column in ("Fluid_Sensitive", "Fat_Suppression"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["_plane"] = frame["Anatomical_Plane"].map(
        {plane: index for index, plane in enumerate(PLANES)}
    ).fillna(3)
    preferred = frame.loc[frame["Fluid_Sensitive"].eq(1)].sort_values(
        ["_plane", "Fat_Suppression", "SeriesInstanceUID"],
        ascending=[True, False, True],
    )
    selected: list[str] = []
    seen_planes: set[str] = set()
    for _, row in preferred.iterrows():
        plane = str(row["Anatomical_Plane"])
        if plane not in seen_planes:
            selected.append(str(row["SeriesInstanceUID"]))
            seen_planes.add(plane)
        if len(selected) == count:
            break
    ordered = frame.sort_values(
        ["Fluid_Sensitive", "Fat_Suppression", "_plane", "SeriesInstanceUID"],
        ascending=[False, False, True, True],
    )
    for series_uid in ordered["SeriesInstanceUID"].astype(str):
        if len(selected) == count:
            break
        if series_uid not in selected:
            selected.append(series_uid)
    return selected + [None] * (count - len(selected))


def load_dicom_volume(
    series_dir: Path, image_size: int, max_series_slices: int
) -> np.ndarray:
    slices = []
    for path in sorted(series_dir.glob("*.dcm"), key=natural_key):
        try:
            dataset = pydicom.dcmread(str(path), force=True)
            image = dataset.pixel_array.astype(np.float32)
            image *= float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
            image += float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
            if str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1":
                image = float(np.max(image)) - image
            instance = getattr(dataset, "InstanceNumber", None)
            try:
                instance = int(instance) if instance is not None else None
            except (TypeError, ValueError):
                instance = None
            location = getattr(dataset, "SliceLocation", None)
            if location is None:
                position = getattr(dataset, "ImagePositionPatient", None)
                location = position[2] if position is not None and len(position) >= 3 else None
            try:
                location = float(location) if location is not None else None
            except (TypeError, ValueError):
                location = None
            slices.append((instance, location, path.name, image))
        except Exception:
            continue
    if not slices:
        raise RuntimeError(f"no decodable slices in {series_dir}")
    slices.sort(
        key=lambda item: (
            0 if item[0] is not None else 1 if item[1] is not None else 2,
            item[0] if item[0] is not None else 0,
            item[1] if item[1] is not None else 0.0,
            item[2],
        )
    )
    volume = np.stack([item[3] for item in slices]).astype(np.float32, copy=False)
    low, high = np.percentile(volume, [1.0, 99.0])
    high = max(float(high), float(low) + 1.0)
    volume = np.clip((volume - float(low)) / (high - float(low)), 0.0, 1.0)
    if len(volume) > max_series_slices:
        start = (len(volume) - max_series_slices) // 2
        volume = volume[start : start + max_series_slices]
    resized = []
    for image in volume:
        height, width = image.shape
        if height != width:
            side = max(height, width)
            padded = np.zeros((side, side), dtype=image.dtype)
            top, left = (side - height) // 2, (side - width) // 2
            padded[top : top + height, left : left + width] = image
            image = padded
        resized.append(
            cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        )
    return np.rint(np.stack(resized) * 255.0).astype(np.uint8)


def uniform_indices(length: int, count: int) -> np.ndarray:
    if length <= count:
        return np.concatenate(
            [np.arange(length), np.full(count - length, length - 1)]
        ).astype(np.int64)
    return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)


def _worker_build_one(payload: dict) -> dict:
    """Process-pool worker: build one study cache file if missing/invalid."""
    uid = payload["uid"]
    destination = Path(payload["destination"])
    image_root = Path(payload["image_root"])
    image_size = int(payload["image_size"])
    cache_slices = int(payload["cache_slices"])
    max_series_slices = int(payload["max_series_slices"])
    shape = (3, cache_slices, image_size, image_size)

    if destination.is_file():
        try:
            with np.load(destination, allow_pickle=False) as cached:
                if cached["images"].shape != shape:
                    raise ValueError("shape")
                decoded = int(cached["mask"].sum())
            return {
                "uid": uid,
                "status": "cached",
                "decoded_slots": decoded,
                "errors": [],
            }
        except Exception:
            destination.unlink(missing_ok=True)

    series = pd.DataFrame(payload["series_rows"])
    images, masks, errors = [], [], []
    for series_uid in pick_series_slots(series, 3):
        if series_uid is None:
            images.append(np.zeros((cache_slices, image_size, image_size), np.uint8))
            masks.append(np.zeros(cache_slices, np.uint8))
            errors.append("missing-slot")
            continue
        try:
            volume = load_dicom_volume(
                image_root / uid / series_uid, image_size, max_series_slices
            )
            images.append(volume[uniform_indices(len(volume), cache_slices)])
            masks.append(np.ones(cache_slices, np.uint8))
        except Exception as error:
            images.append(np.zeros((cache_slices, image_size, image_size), np.uint8))
            masks.append(np.zeros(cache_slices, np.uint8))
            errors.append(f"{series_uid}:{type(error).__name__}")

    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, images=np.stack(images), mask=np.stack(masks))
    os.replace(temporary, destination)
    return {
        "uid": uid,
        "status": "built",
        "decoded_slots": int(np.stack(masks).sum()),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--series-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--cache-slices", type=int, default=12)
    parser.add_argument("--max-series-slices", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.image_size < 64 or args.cache_slices < 1:
        raise ValueError("invalid image_size / cache_slices")

    train = pd.read_csv(args.train_csv, dtype={UID: str})
    series = pd.read_csv(
        args.series_csv,
        dtype={UID: str, "SeriesInstanceUID": str},
    )
    if len(train) != 4407 or train[UID].duplicated().any():
        raise ValueError("train UID contract failed")

    grouped = {uid: rows for uid, rows in series.groupby(UID, sort=False)}
    cache_dir = args.output_dir / "studies"
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    jobs = []
    for uid in train[UID].astype(str):
        rows = grouped.get(uid)
        if rows is None:
            raise ValueError(f"no series metadata for {uid}")
        jobs.append(
            {
                "uid": uid,
                "destination": str(cache_dir / f"{uid}.npz"),
                "image_root": str(args.image_root),
                "image_size": args.image_size,
                "cache_slices": args.cache_slices,
                "max_series_slices": args.max_series_slices,
                "series_rows": rows.to_dict(orient="list"),
            }
        )

    failures: dict[str, list[str]] = {}
    decoded_slots: dict[str, int] = {}
    completed = 0
    workers = max(1, int(args.workers))

    print(
        json.dumps(
            {
                "status": "START",
                "rows": len(jobs),
                "image_size": args.image_size,
                "cache_slices": args.cache_slices,
                "workers": workers,
            }
        ),
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker_build_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            uid = result["uid"]
            decoded_slots[uid] = int(result["decoded_slots"])
            if result["errors"]:
                failures[uid] = list(result["errors"])
            if completed % 50 == 0 or completed == len(jobs):
                progress = {
                    "status": "BUILDING" if completed < len(jobs) else "VERIFYING",
                    "completed_rows": completed,
                    "total_rows": len(jobs),
                    "elapsed_seconds": time.time() - started,
                    "failure_rows": len(failures),
                    "rate_per_minute": completed / max((time.time() - started) / 60.0, 1e-6),
                }
                atomic_json(args.output_dir / "cache_progress.json", progress)
                print(json.dumps(progress), flush=True)

    files = sorted(cache_dir.glob("*.npz"))
    if len(files) != len(train):
        raise RuntimeError(f"cache file count mismatch: {len(files)} != {len(train)}")
    order = {uid: index for index, uid in enumerate(train[UID].astype(str))}
    if {path.stem for path in files} != set(order):
        raise RuntimeError("cache UID set mismatch")

    expected_shape = [3, args.cache_slices, args.image_size, args.image_size]
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in sorted(files, key=lambda item: order[item.stem]):
        with np.load(path, allow_pickle=False) as cached:
            if list(cached["images"].shape) != expected_shape:
                raise RuntimeError(f"bad shape for {path.stem}: {cached['images'].shape}")
        file_hash = sha256(path)
        total_bytes += path.stat().st_size
        aggregate.update(f"{path.stem},{path.stat().st_size},{file_hash}\n".encode())

    manifest = {
        "format": "rsna-knee-v3-1-uint8-cache-v2-parallel",
        "status": "COMPLETE",
        "rows": len(files),
        "shape_per_study": expected_shape,
        "dtype": "uint8",
        "image_size": args.image_size,
        "cache_slices": args.cache_slices,
        "max_series_slices": args.max_series_slices,
        "workers": workers,
        "total_bytes": total_bytes,
        "elapsed_seconds": time.time() - started,
        "failure_rows": failures,
        "decoded_slice_count_by_uid": decoded_slots,
        "aggregate_sha256": aggregate.hexdigest(),
        "input_sha256": {
            str(args.train_csv): sha256(args.train_csv),
            str(args.series_csv): sha256(args.series_csv),
        },
        "module_sha256": sha256(Path(__file__)),
    }
    atomic_json(args.output_dir / "cache_manifest.json", manifest)
    slim = {k: v for k, v in manifest.items() if k != "decoded_slice_count_by_uid"}
    print(json.dumps(slim, indent=2), flush=True)


if __name__ == "__main__":
    main()
