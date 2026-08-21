#!/usr/bin/env python3
"""Build a persistent, manifest-verified uint8 MRI cache for V3.1 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom


UID = "StudyInstanceUID"
PLANES = ("Sagittal", "Coronal", "Axial")
IMAGE_SIZE = 224
CACHE_SLICES = 12
MAX_SERIES_SLICES = 64


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


def load_dicom_volume(series_dir: Path) -> np.ndarray:
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
    if len(volume) > MAX_SERIES_SLICES:
        start = (len(volume) - MAX_SERIES_SLICES) // 2
        volume = volume[start : start + MAX_SERIES_SLICES]
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
            cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA,
            )
        )
    return np.rint(np.stack(resized) * 255.0).astype(np.uint8)


def uniform_indices(length: int, count: int) -> np.ndarray:
    if length <= count:
        return np.concatenate(
            [np.arange(length), np.full(count - length, length - 1)]
        ).astype(np.int64)
    return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)


def write_study_cache(
    uid: str,
    rows: pd.DataFrame,
    image_root: Path,
    destination: Path,
) -> tuple[int, list[str]]:
    images, masks, errors = [], [], []
    for series_uid in pick_series_slots(rows, len(PLANES)):
        if series_uid is None:
            images.append(np.zeros((CACHE_SLICES, IMAGE_SIZE, IMAGE_SIZE), np.uint8))
            masks.append(np.zeros(CACHE_SLICES, np.uint8))
            errors.append("missing-slot")
            continue
        try:
            volume = load_dicom_volume(image_root / uid / series_uid)
            images.append(volume[uniform_indices(len(volume), CACHE_SLICES)])
            masks.append(np.ones(CACHE_SLICES, np.uint8))
        except Exception as error:
            images.append(np.zeros((CACHE_SLICES, IMAGE_SIZE, IMAGE_SIZE), np.uint8))
            masks.append(np.zeros(CACHE_SLICES, np.uint8))
            errors.append(f"{series_uid}:{type(error).__name__}")
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, images=np.stack(images), mask=np.stack(masks))
    os.replace(temporary, destination)
    return int(np.stack(masks).sum()), errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--series-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

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
    failures: dict[str, list[str]] = {}
    decoded_slots: dict[str, int] = {}

    for index, uid in enumerate(train[UID].astype(str), 1):
        destination = cache_dir / f"{uid}.npz"
        expected_rows = grouped.get(uid)
        if expected_rows is None:
            raise ValueError(f"no series metadata for {uid}")
        if destination.is_file():
            try:
                with np.load(destination, allow_pickle=False) as cached:
                    if cached["images"].shape != (3, CACHE_SLICES, IMAGE_SIZE, IMAGE_SIZE):
                        raise ValueError("shape")
                    decoded_slots[uid] = int(cached["mask"].sum())
                continue
            except Exception:
                destination.unlink(missing_ok=True)
        decoded_slots[uid], errors = write_study_cache(
            uid, expected_rows, args.image_root, destination
        )
        if errors:
            failures[uid] = errors
        if index % 100 == 0 or index == len(train):
            progress = {
                "status": "BUILDING" if index < len(train) else "VERIFYING",
                "completed_rows": index,
                "total_rows": len(train),
                "elapsed_seconds": time.time() - started,
                "failure_rows": len(failures),
            }
            atomic_json(args.output_dir / "cache_progress.json", progress)
            print(json.dumps(progress), flush=True)

    files = sorted(cache_dir.glob("*.npz"))
    if len(files) != len(train):
        raise RuntimeError(f"cache file count mismatch: {len(files)} != {len(train)}")
    order = {uid: index for index, uid in enumerate(train[UID].astype(str))}
    if {path.stem for path in files} != set(order):
        raise RuntimeError("cache UID set mismatch")
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in sorted(files, key=lambda item: order[item.stem]):
        file_hash = sha256(path)
        total_bytes += path.stat().st_size
        aggregate.update(f"{path.stem},{path.stat().st_size},{file_hash}\n".encode())
    manifest = {
        "format": "rsna-knee-v3-1-uint8-cache-v1",
        "status": "COMPLETE",
        "rows": len(files),
        "shape_per_study": [3, CACHE_SLICES, IMAGE_SIZE, IMAGE_SIZE],
        "dtype": "uint8",
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
    print(json.dumps({key: value for key, value in manifest.items() if key != "decoded_slice_count_by_uid"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
