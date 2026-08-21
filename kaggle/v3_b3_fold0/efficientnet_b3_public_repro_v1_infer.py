#!/usr/bin/env python3
"""Five-fold inference for the public EfficientNet-B3 reproduction package."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("effnet_public_repro", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_submission(path: Path, uids, targets, predictions) -> None:
    rows = []
    for uid in uids:
        values = predictions.get(str(uid), np.full(len(targets), 0.5, dtype=float))
        rows.append({"StudyInstanceUID": uid, **dict(zip(targets, map(float, values)))})
    pd.DataFrame(rows, columns=["StudyInstanceUID", *targets]).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--series-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs=5, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-hours", type=float, default=8.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()

    module = load_module(args.module)
    _, _, _, torch, nn, _, _ = module._training_imports()
    if not torch.cuda.is_available() and not args.allow_cpu_smoke:
        raise RuntimeError("GPU is required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(0) >= (8, 0)
        else torch.float16
    )
    Model = module.make_model_class()
    models, configs, folds = [], [], []
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        folds.append(int(checkpoint.get("fold", -1)))
        config = module.TrainConfig(**checkpoint["config"])
        model = Model(config.backbone, checkpoint=None, backbone_chunk=config.backbone_chunk)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().to(device)
        models.append(model)
        configs.append(config)
    if set(folds) != set(range(5)):
        raise ValueError(f"expected exactly folds 0..4, got {folds}")
    if len({json.dumps(vars(config), sort_keys=True) for config in configs}) != 1:
        raise ValueError("fold checkpoint configs disagree")
    config = configs[0]

    test = pd.read_csv(args.test_csv)
    series = pd.read_csv(args.series_csv)
    if test[module.UID].duplicated().any():
        raise ValueError("test contains duplicate study IDs")
    grouped = {str(uid): group.copy() for uid, group in series.groupby(module.UID, sort=False)}
    empty = pd.DataFrame(columns=[
        "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"
    ])
    uids = test[module.UID].astype(str).tolist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission.csv"
    predictions, errors, series_errors = {}, {}, {}
    eval_slices, active_models = config.valid_slices, len(models)
    started = time.time()

    for index, uid in enumerate(uids):
        try:
            slots = module.pick_series_slots(grouped.get(uid, empty), len(module.PLANES))
            images, masks = [], []
            for series_uid in slots:
                if series_uid is None:
                    images.append(np.zeros((eval_slices, config.image_size, config.image_size), np.float32))
                    masks.append(np.zeros(eval_slices, bool))
                    continue
                try:
                    volume = module.load_dicom_volume(
                        args.image_root / uid / series_uid,
                        config.image_size,
                        config.max_series_slices,
                    )
                except Exception as exc:
                    series_errors[f"{uid}/{series_uid}"] = repr(exc)
                    images.append(np.zeros((eval_slices, config.image_size, config.image_size), np.float32))
                    masks.append(np.zeros(eval_slices, bool))
                    continue
                chosen = module.uniform_slice_indices(len(volume), eval_slices)
                sampled = (volume[chosen].astype(np.float32) - 0.449) / 0.226
                images.append(sampled)
                masks.append(np.ones(eval_slices, bool))
            mask = np.stack(masks)
            if not mask.any():
                raise RuntimeError("no decodable series")
            image_tensor = torch.from_numpy(np.stack(images)[:, :, None])[None].to(device)
            mask_tensor = torch.from_numpy(mask)[None].to(device)
            probability = torch.zeros(len(module.TARGETS), device="cpu")
            with torch.no_grad():
                for model in models[:active_models]:
                    amp = (
                        torch.autocast("cuda", dtype=amp_dtype)
                        if device.type == "cuda" else contextlib.nullcontext()
                    )
                    with amp:
                        logits = model(image_tensor, mask_tensor)
                    probability += torch.sigmoid(logits.float()).cpu().squeeze(0)
            value = (probability / active_models).numpy()
            if not np.isfinite(value).all():
                raise RuntimeError("non-finite prediction")
            predictions[uid] = value
        except Exception as exc:
            errors[uid] = repr(exc)
            predictions[uid] = np.full(len(module.TARGETS), 0.5, dtype=float)

        done = index + 1
        if done % args.checkpoint_every == 0 or done == len(uids):
            write_submission(submission_path, uids, module.TARGETS, predictions)
            projected = (time.time() - started) / done * len(uids)
            if done >= args.checkpoint_every and projected > args.budget_hours * 3600:
                if eval_slices > 16:
                    eval_slices = 16
                elif active_models > 3:
                    active_models = 3
            print(
                f"{done}/{len(uids)} projected_hours={projected / 3600:.2f} "
                f"slices={eval_slices} folds={active_models} errors={len(errors)}",
                flush=True,
            )

    write_submission(submission_path, uids, module.TARGETS, predictions)
    meta = {
        "status": "PASS" if not errors else "PASS_WITH_FALLBACKS",
        "studies": len(uids),
        "errors": errors,
        "series_errors": series_errors,
        "wall_seconds": time.time() - started,
        "eval_slices_final": eval_slices,
        "active_models_final": active_models,
        "checkpoints": [str(path) for path in args.checkpoints],
        "folds": folds,
    }
    (args.output_dir / "infer_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
