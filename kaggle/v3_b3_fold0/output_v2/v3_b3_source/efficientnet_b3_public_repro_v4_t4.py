#!/usr/bin/env python3
"""T4-bounded, no-flip EfficientNet-B3 training for RSNA Knee.

This is an independent reproduction of the public .903 recipe.  It keeps the
audited v2 model and report-label contracts, but corrects two important input
details disclosed by that notebook: non-square slices are padded before resize,
and no horizontal flip is ever applied to laterality-specific targets.  V3 runs
one fold per Kaggle job, allowing the full 288px / 12-slice / eight-epoch recipe
to stay inside the code-competition runtime budget.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _load_v2():
    path = Path(__file__).with_name("efficientnet_b3_public_repro_v2_anatomy.py")
    if not path.is_file():
        raise FileNotFoundError(f"audited v2 dependency is absent: {path}")
    name = "rsna_knee_effnet_b3_public_repro_v2_t4_dependency"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import audited v2 dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = _load_v2()
BASE = V2.BASE
TARGETS, PLANES, UID = V2.TARGETS, V2.PLANES, V2.UID
TrainConfig = V2.TrainConfig
TrainingPolicy = V2.TrainingPolicy
pick_series_slots = V2.pick_series_slots
uniform_slice_indices = V2.uniform_slice_indices
stochastic_slice_indices = V2.stochastic_slice_indices
make_model_class = V2.make_model_class
_training_imports = V2._training_imports


def _natural_key(path: Path):
    import re

    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    )


def load_dicom_volume(series_dir: Path, image_size: int, max_slices: int = 64):
    """Decode like the .903 inference path, including square padding."""
    cv2, pydicom, _, _, _, _, _ = _training_imports()
    slices = []
    for path in sorted(Path(series_dir).glob("*.dcm"), key=_natural_key):
        try:
            ds = pydicom.dcmread(str(path), force=True)
            image = ds.pixel_array.astype(np.float32)
            image *= float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            image += float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
                image = float(np.max(image)) - image
            instance = getattr(ds, "InstanceNumber", None)
            try:
                instance = int(instance) if instance is not None else None
            except (TypeError, ValueError):
                instance = None
            location = getattr(ds, "SliceLocation", None)
            if location is None:
                position = getattr(ds, "ImagePositionPatient", None)
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
    lo, hi = np.percentile(volume, [1.0, 99.0])
    hi = max(float(hi), float(lo) + 1.0)
    volume = np.clip((volume - float(lo)) / (hi - float(lo)), 0.0, 1.0)
    if len(volume) > int(max_slices):
        start = (len(volume) - int(max_slices)) // 2
        volume = volume[start : start + int(max_slices)]

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
            cv2.resize(image, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
        )
    return np.rint(np.stack(resized) * 255.0).astype(np.uint8).astype(np.float32) / 255.0


def make_dataset_class():
    """Fixed-plane dataset with intensity jitter and explicitly no flip."""
    _, _, _, torch, _, _, Dataset = _training_imports()

    class NoFlipKneeStudyDataset(Dataset):
        def __init__(self, rows, series, image_root, config, training, seed):
            self.rows = rows.reset_index(drop=True)
            self.series = {
                str(uid): group.copy() for uid, group in series.groupby(UID, sort=False)
            }
            self.image_root = Path(image_root)
            self.config = config
            self.training = bool(training)
            self.seed = int(seed)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row, config = self.rows.iloc[index], self.config
            uid = str(row[UID])
            count = config.train_slices if self.training else config.valid_slices
            visit_seed = (
                int(torch.randint(0, 2**31 - 1, (1,)).item()) if self.training else 0
            )
            rng = np.random.default_rng(self.seed + 1_000_003 * index + visit_seed)
            empty = pd.DataFrame(
                columns=[
                    "SeriesInstanceUID",
                    "Anatomical_Plane",
                    "Fluid_Sensitive",
                    "Fat_Suppression",
                ]
            )
            selected = pick_series_slots(self.series.get(uid, empty), len(PLANES))
            planes, masks = [], []
            for series_uid in selected:
                if series_uid is None:
                    planes.append(
                        np.zeros((count, config.image_size, config.image_size), np.float32)
                    )
                    masks.append(np.zeros(count, bool))
                    continue
                volume = load_dicom_volume(
                    self.image_root / uid / str(series_uid),
                    config.image_size,
                    config.max_series_slices,
                )
                indices = (
                    stochastic_slice_indices(len(volume), count, rng)
                    if self.training
                    else uniform_slice_indices(len(volume), count)
                )
                sampled = volume[indices].astype(np.float32)
                if self.training:
                    sampled = np.clip(sampled * rng.uniform(0.88, 1.12), 0.0, 1.0)
                    sampled = sampled ** rng.uniform(0.88, 1.12)
                planes.append((sampled - 0.449) / 0.226)
                masks.append(np.ones(count, bool))
            return {
                "uid": uid,
                "images": torch.from_numpy(np.stack(planes)[:, :, None]),
                "mask": torch.from_numpy(np.stack(masks)),
                "target": torch.tensor(
                    [row[f"target::{target}"] for target in TARGETS], dtype=torch.float32
                ),
                "weight": torch.tensor(
                    [row[f"train_weight::{target}"] for target in TARGETS],
                    dtype=torch.float32,
                ),
            }

    return NoFlipKneeStudyDataset


# V2.train_one_fold resolves these names from its own module globals.
V2.load_dicom_volume = load_dicom_volume
V2.make_dataset_class = make_dataset_class


def load_v3_label_table(
    train: pd.DataFrame, labels: pd.DataFrame, folds: pd.DataFrame
) -> pd.DataFrame:
    """Convert the audited V3 artifacts to the trainer's leakage-safe schema."""
    required_train = {UID, "Report", *TARGETS}
    required_labels = {UID, *TARGETS, *(f"{target}__weight" for target in TARGETS)}
    required_folds = {UID, "fold", "report_group", "is_gold"}
    for name, frame, required in (
        ("train", train, required_train),
        ("labels_v3", labels, required_labels),
        ("folds_v3", folds, required_folds),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame[UID].duplicated().any():
            raise ValueError(f"{name} contains duplicate {UID} rows")

    expected = set(train[UID].astype(str))
    for name, frame in (("labels_v3", labels), ("folds_v3", folds)):
        observed = set(frame[UID].astype(str))
        if observed != expected:
            raise ValueError(
                f"{name} UID mismatch: missing={len(expected - observed)}, "
                f"extra={len(observed - expected)}"
            )

    base = train[[UID, *TARGETS]].copy()
    label_columns = [UID]
    for target in TARGETS:
        label_columns.extend([target, f"{target}__weight"])
    out = base.merge(labels[label_columns], on=UID, how="inner", suffixes=("__gold", "__pseudo"))
    out = out.merge(folds[[UID, "fold", "report_group", "is_gold"]], on=UID, how="inner")
    out["fold"] = pd.to_numeric(out["fold"], errors="raise").astype(np.int8)
    out["is_expert"] = pd.to_numeric(out["is_gold"], errors="raise").astype(np.int8)
    if sorted(out["fold"].unique().tolist()) != list(range(5)):
        raise ValueError(f"unexpected folds: {sorted(out['fold'].unique().tolist())}")
    if int(out["is_expert"].sum()) != int(train[list(TARGETS)].notna().all(axis=1).sum()):
        raise ValueError("is_gold does not match the expert labels in train.csv")

    result = out[[UID, "report_group", "is_expert", "fold"]].copy()
    for target in TARGETS:
        pseudo = pd.to_numeric(out[f"{target}__pseudo"], errors="raise")
        weight = pd.to_numeric(out[f"{target}__weight"], errors="raise")
        gold = pd.to_numeric(out[f"{target}__gold"], errors="coerce")
        if not pseudo.between(0.0, 1.0).all():
            raise ValueError(f"V3 pseudo-labels outside [0, 1] for {target}")
        if not weight.between(0.0, 3.0, inclusive="both").all():
            raise ValueError(f"V3 weights outside [0, 3] for {target}")
        result[f"pseudo::{target}"] = pseudo.astype(np.float32)
        result[f"weight::{target}"] = weight.astype(np.float32)
        result[f"gold::{target}"] = gold
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train-csv",
        "series-csv",
        "v3-labels",
        "v3-folds",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)

    config = replace(
        TrainConfig(),
        image_size=288,
        train_slices=12,
        valid_slices=32,
        batch_size=2,
        workers=4,
        backbone_chunk=12,
        frozen_epochs=1,
        unfrozen_epochs=7,
        frozen_lr=8e-4,
        unfrozen_lr=8e-5,
        gold_weight=2.0,
        backbone="efficientnet_b3",
    )
    policy = replace(
        TrainingPolicy(),
        expert_sample_fraction=0.15,
        unfrozen_backbone_blocks=3,
        model_family="efficientnet_b3_noflip_t4_v4",
    )
    paths = [
        args.train_csv,
        args.series_csv,
        args.v3_labels,
        args.v3_folds,
    ]
    frames = [pd.read_csv(path) for path in paths]
    labels = load_v3_label_table(frames[0], frames[2], frames[3])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output_dir / "fold_labels.csv", index=False)
    manifest = {
        "status": "PREPARED",
        "config": asdict(config),
        "training_policy": asdict(policy),
        "rows": len(labels),
        "expert_rows": int(labels["is_expert"].sum()),
        "fold_counts": labels["fold"].value_counts().sort_index().to_dict(),
        "fold_expert_counts": labels.loc[
            labels["is_expert"].eq(1), "fold"
        ].value_counts().sort_index().to_dict(),
        "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "no_horizontal_flip": True,
        "square_padding_before_resize": True,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if args.prepare_only:
        return
    if args.fold is None or args.image_root is None or args.backbone_checkpoint is None:
        parser.error("training requires --fold, --image-root, and --backbone-checkpoint")
    if not 0 <= args.fold < config.folds:
        parser.error(f"--fold must be in [0, {config.folds - 1}]")
    print(
        "saved",
        V2.train_one_fold(
            labels,
            frames[1],
            args.image_root,
            args.output_dir,
            args.fold,
            args.backbone_checkpoint,
            config,
            policy,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
