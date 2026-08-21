#!/usr/bin/env python3
"""Production EfficientNet-B3 fold trainer for RSNA Knee top-50 / top-10 path.

Uses the audited v2 anatomy model + v4 no-flip / square-pad contracts, but:
  * full 288px / 12-slice recipe (not the 224 smoke)
  * fold-cross-fitted V3.1 labels (labels_v3_fold{N}.csv)
  * optional prebuilt uint8 cache (mounted Kaggle dataset or kernel output)
  * epoch checkpoints + full held-out-fold OOF
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _load_v4():
    path = Path(__file__).with_name("efficientnet_b3_public_repro_v4_t4.py")
    if not path.is_file():
        raise FileNotFoundError(f"v4 dependency missing: {path}")
    name = "rsna_knee_effnet_b3_v4_dependency"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V4 = _load_v4()
V2 = V4.V2
BASE = V4.BASE
TARGETS, PLANES, UID = V4.TARGETS, V4.PLANES, V4.UID
TrainConfig = V4.TrainConfig
TrainingPolicy = V4.TrainingPolicy
load_v3_label_table = V4.load_v3_label_table
prepare_full_fold_rows = V4.prepare_full_fold_rows


def resolve_cache_root(explicit: Optional[Path]) -> Path:
    """Prefer an attached complete cache, else working directory."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env = os.environ.get("RSNA_CACHE_ROOT", "").strip()
    if env:
        candidates.append(Path(env))

    root = Path("/kaggle/input")
    if root.is_dir():
        for manifest in root.rglob("cache_manifest.json"):
            candidates.append(manifest.parent)
        for studies in root.rglob("studies"):
            if studies.is_dir() and any(studies.glob("*.npz")):
                candidates.append(studies.parent)

    candidates.extend(
        [
            Path("/kaggle/working/v3_1_cache"),
            Path("/kaggle/working/v3_1_uint8_cache"),
        ]
    )

    best: Optional[Path] = None
    best_count = -1
    for candidate in candidates:
        if candidate is None:
            continue
        studies = candidate / "studies"
        if not studies.is_dir():
            continue
        count = sum(1 for _ in studies.glob("*.npz"))
        if count > best_count:
            best, best_count = candidate, count
    if best is not None and best_count > 0:
        return best
    return Path("/kaggle/working/v3_1_uint8_cache")


def patch_cache_root(cache_root: Path) -> None:
    """Force v4 dataset class to read the production cache."""
    V4.CACHE_ROOT = cache_root
    # v4 make_dataset_class closes over CACHE_ROOT at call time via module global.
    # Re-bind and rebuild dataset factory.
    original = V4.make_dataset_class

    def make_dataset_class():
        # Temporarily ensure CACHE_ROOT is current
        V4.CACHE_ROOT = cache_root
        return original()

    V4.make_dataset_class = make_dataset_class
    V2.make_dataset_class = make_dataset_class
    V2.load_dicom_volume = V4.load_dicom_volume


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-csv", "series-csv", "v3-labels", "v3-folds", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--train-slices", type=int, default=8)
    parser.add_argument("--valid-slices", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--backbone-chunk", type=int, default=6)
    parser.add_argument("--frozen-epochs", type=int, default=1)
    # Wave-1 after OOM: 1+3 epochs, batch1, 8 slices (docs/gpu-budget-28h.md).
    # Wave-2 / V6: prefer 1+7 to approach the public 8-epoch recipe without OOM.
    parser.add_argument("--unfrozen-epochs", type=int, default=3)
    parser.add_argument("--gold-weight", type=float, default=2.0)
    parser.add_argument("--expert-sample-fraction", type=float, default=0.15)
    parser.add_argument("--unfrozen-backbone-blocks", type=int, default=3)
    parser.add_argument(
        "--model-family",
        type=str,
        default="efficientnet_b3_noflip_v5_wave1",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--require-cache", action="store_true")
    args = parser.parse_args(argv)

    train_started = __import__("time").time()
    cache_root = resolve_cache_root(args.cache_root)
    patch_cache_root(cache_root)

    config = replace(
        TrainConfig(),
        image_size=int(args.image_size),
        train_slices=int(args.train_slices),
        valid_slices=int(args.valid_slices),
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        backbone_chunk=int(args.backbone_chunk),
        frozen_epochs=int(args.frozen_epochs),
        unfrozen_epochs=int(args.unfrozen_epochs),
        frozen_lr=8e-4,
        unfrozen_lr=8e-5,
        gold_weight=float(args.gold_weight),
        backbone="efficientnet_b3",
    )
    policy = replace(
        TrainingPolicy(),
        expert_sample_fraction=float(args.expert_sample_fraction),
        unfrozen_backbone_blocks=int(args.unfrozen_backbone_blocks),
        model_family=str(args.model_family),
    )

    # Ensure prepare_fold_rows is the leakage-safe full-fold variant.
    V2.BASE.prepare_fold_rows = prepare_full_fold_rows
    BASE.prepare_fold_rows = prepare_full_fold_rows

    paths = [args.train_csv, args.series_csv, args.v3_labels, args.v3_folds]
    frames = [pd.read_csv(path) for path in paths]
    labels = load_v3_label_table(frames[0], frames[2], frames[3])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output_dir / "fold_labels.csv", index=False)

    cache_files = list((cache_root / "studies").glob("*.npz")) if (cache_root / "studies").is_dir() else []
    manifest = {
        "status": "PREPARED",
        "config": asdict(config),
        "training_policy": asdict(policy),
        "rows": len(labels),
        "expert_rows": int(labels["is_expert"].sum()),
        "fold_counts": labels["fold"].value_counts().sort_index().to_dict(),
        "fold_expert_counts": labels.loc[labels["is_expert"].eq(1), "fold"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "inputs": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "no_horizontal_flip": True,
        "square_padding_before_resize": True,
        "fold_cross_fitted_labels": True,
        "full_heldout_fold_oof": True,
        "cache_root": str(cache_root),
        "cache_study_files": len(cache_files),
        "require_cache": bool(args.require_cache),
        "gpu_budget_note": "wave1_1plus4_epochs_28h_plan",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)

    if args.require_cache and len(cache_files) < 4000:
        raise RuntimeError(
            f"require-cache set but only {len(cache_files)} study caches under {cache_root}"
        )

    if args.prepare_only:
        return
    if args.fold is None or args.image_root is None or args.backbone_checkpoint is None:
        parser.error("training requires --fold, --image-root, and --backbone-checkpoint")
    if not 0 <= args.fold < config.folds:
        parser.error(f"--fold must be in [0, {config.folds - 1}]")

    destination = V2.train_one_fold(
        labels,
        frames[1],
        args.image_root,
        args.output_dir,
        args.fold,
        args.backbone_checkpoint,
        config,
        policy,
    )
    elapsed = __import__("time").time() - train_started
    timing = {
        "fold": int(args.fold),
        "elapsed_seconds": elapsed,
        "elapsed_hours": elapsed / 3600.0,
        "epochs_total": int(args.frozen_epochs) + int(args.unfrozen_epochs),
        "checkpoint": str(destination),
        "projected_five_fold_hours": 5.0 * (elapsed / 3600.0),
        "gpu_budget_remaining_hint": "stop_if_five_fold_projection_gt_20h",
    }
    (args.output_dir / f"fold{args.fold}_timing.json").write_text(
        json.dumps(timing, indent=2, sort_keys=True) + "\n"
    )
    print("saved", destination, flush=True)
    print(json.dumps(timing, indent=2, sort_keys=True), flush=True)
    if timing["projected_five_fold_hours"] > 20.0:
        print(
            "WARNING: five-fold projection exceeds 20 GPU hours; "
            "reduce unfrozen-epochs for remaining folds.",
            flush=True,
        )


if __name__ == "__main__":
    main()
