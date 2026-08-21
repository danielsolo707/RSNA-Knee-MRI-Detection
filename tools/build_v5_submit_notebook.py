#!/usr/bin/env python3
"""Build V5 submission notebook: V1 DINO path + gold-gated B3 rank blend.

Clones the verified 0.899 V1 notebook and appends a final cell that:
  1. Saves DINO `submission.csv` as `dino_submission.csv`
  2. Runs five-fold EfficientNet-B3 inference from private checkpoints
  3. Applies gold-LOO blend weights (dino-only or 50/50 rank mean per target)
  4. Overwrites `submission.csv`
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_NB = ROOT / "kaggle" / "submission_v1" / "rsna-knee-v1-public-0899.ipynb"
OUT_DIR = ROOT / "kaggle" / "submission_v5_dino_b3"
TRAIN_SRC = ROOT / "kaggle" / "v3_b3_fold0"
CKPT_DIR = ROOT / "artifacts" / "b3_checkpoints_v5"
MODULES = (
    "efficientnet_b3_public_repro_v1.py",
    "efficientnet_b3_public_repro_v2_anatomy.py",
    "efficientnet_b3_public_repro_v4_t4.py",
)


BLEND_CELL = r'''
# === V5: gold-gated DINO + B3 rank blend (appended after V1 DINO path) ===
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BLEND_TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
UID = "StudyInstanceUID"


def _rank_pct_matrix(pred: np.ndarray) -> np.ndarray:
    out = np.zeros_like(pred, dtype=np.float64)
    for t in range(pred.shape[1]):
        values = pred[:, t]
        order = values.argsort(kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
        sorted_vals = values[order]
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            if j > i:
                avg = 0.5 * (i + j) + 1.0
                ranks[order[i : j + 1]] = avg
            i = j + 1
        out[:, t] = ranks / len(values)
    return out


def _find_one(pattern: str) -> Path | None:
    hits = sorted(Path("/kaggle/input").glob(pattern))
    return hits[0] if hits else None


def _load_blend_weights() -> dict:
    path = _find_one("**/blend_report_gold.json") or _find_one("**/blend_report.json")
    if path is None:
        raise FileNotFoundError("blend_report_gold.json not found under /kaggle/input")
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("weights") or payload
    print("blend weights from", path, flush=True)
    return weights


def _copy_b3_sources() -> Path:
    """Prefer dataset-shipped sources; fall back to writing from working tree if present."""
    out = Path("/kaggle/working/v3_b3_source")
    out.mkdir(parents=True, exist_ok=True)
    names = [
        "efficientnet_b3_public_repro_v1.py",
        "efficientnet_b3_public_repro_v2_anatomy.py",
        "efficientnet_b3_public_repro_v4_t4.py",
    ]
    for name in names:
        src = _find_one(f"**/{name}")
        if src is None:
            raise FileNotFoundError(f"missing B3 source module {name}")
        dest = out / name
        if not dest.exists():
            shutil.copy2(src, dest)
    return out


def _load_module(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # Ensure sibling imports resolve when v4 loads v2 via with_name paths
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def _find_competition() -> Path:
    direct = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
    if direct.is_dir():
        return direct
    hits = list(Path("/kaggle/input").glob("**/test_series.csv"))
    if not hits:
        raise FileNotFoundError("competition test_series.csv not found")
    return hits[0].parent


def _find_checkpoints() -> list[Path]:
    paths = []
    for fold in range(5):
        hit = _find_one(f"**/fold{fold}_final.pt")
        if hit is None:
            raise FileNotFoundError(f"fold{fold}_final.pt not found")
        paths.append(hit)
    return paths


def run_b3_infer() -> pd.DataFrame:
    t0 = time.time()
    src_dir = _copy_b3_sources()
    # Load v4 which pulls v2/v1 anatomy + model
    v4 = _load_module(src_dir / "efficientnet_b3_public_repro_v4_t4.py", "effnet_v4_infer")
    _, _, _, torch_mod, nn, _, _ = v4._training_imports()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU required for B3 inference")
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability(0) >= (8, 0)
        else torch.float16
    )
    Model = v4.make_model_class()
    ckpts = _find_checkpoints()
    models, folds = [], []
    config = None
    for path in ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        folds.append(int(ck.get("fold", -1)))
        cfg_dict = ck["config"]
        cfg = v4.TrainConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
        model = Model(cfg.backbone, checkpoint=None, backbone_chunk=cfg.backbone_chunk)
        model.load_state_dict(ck["model"], strict=True)
        model.eval().to(device)
        models.append(model)
        config = cfg
        print(f"loaded {path.name} fold={ck.get('fold')}", flush=True)
    if set(folds) != set(range(5)):
        raise ValueError(f"expected folds 0..4, got {folds}")

    comp = _find_competition()
    test = pd.read_csv(comp / "test.csv")
    series = pd.read_csv(comp / "test_series.csv")
    image_root = comp / "test_series"
    if not image_root.is_dir():
        # some layouts nest series under test/
        alt = list(comp.glob("**/test_series"))
        image_root = alt[0] if alt else image_root
    grouped = {str(uid): g.copy() for uid, g in series.groupby(UID, sort=False)}
    empty = pd.DataFrame(
        columns=["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]
    )
    uids = test[UID].astype(str).tolist()
    eval_slices = int(config.valid_slices)
    image_size = int(config.image_size)
    predictions = {}
    errors = {}
    for i, uid in enumerate(uids):
        try:
            slots = v4.pick_series_slots(grouped.get(uid, empty), len(v4.PLANES))
            images, masks = [], []
            for series_uid in slots:
                if series_uid is None:
                    images.append(np.zeros((eval_slices, image_size, image_size), np.float32))
                    masks.append(np.zeros(eval_slices, bool))
                    continue
                try:
                    volume = v4.load_dicom_volume(
                        image_root / uid / str(series_uid),
                        image_size,
                        config.max_series_slices,
                    )
                except Exception:
                    images.append(np.zeros((eval_slices, image_size, image_size), np.float32))
                    masks.append(np.zeros(eval_slices, bool))
                    continue
                chosen = v4.uniform_slice_indices(len(volume), eval_slices)
                sampled = (volume[chosen].astype(np.float32) - 0.449) / 0.226
                images.append(sampled)
                masks.append(np.ones(eval_slices, bool))
            mask = np.stack(masks)
            if not mask.any():
                raise RuntimeError("no decodable series")
            image_tensor = torch.from_numpy(np.stack(images)[:, :, None])[None].to(device)
            mask_tensor = torch.from_numpy(mask)[None].to(device)
            probability = torch.zeros(len(v4.TARGETS), device="cpu")
            with torch.no_grad():
                for model in models:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        logits = model(image_tensor, mask_tensor)
                    probability += torch.sigmoid(logits.float()).cpu().squeeze(0)
            value = (probability / len(models)).numpy()
            if not np.isfinite(value).all():
                raise RuntimeError("non-finite prediction")
            predictions[uid] = value
        except Exception as exc:
            errors[uid] = repr(exc)
            predictions[uid] = np.full(len(v4.TARGETS), 0.5, dtype=float)
        if (i + 1) % 50 == 0 or (i + 1) == len(uids):
            elapsed = time.time() - t0
            print(
                f"B3 {i+1}/{len(uids)} elapsed={elapsed/60:.1f}m errors={len(errors)}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = []
    for uid in uids:
        row = {UID: uid}
        for t, v in zip(v4.TARGETS, predictions[uid]):
            row[t] = float(v)
        rows.append(row)
    b3 = pd.DataFrame(rows)[[UID, *BLEND_TARGETS]]
    b3.to_csv("b3_submission.csv", index=False)
    meta = {
        "studies": len(uids),
        "errors": len(errors),
        "error_sample": dict(list(errors.items())[:10]),
        "elapsed_seconds": time.time() - t0,
        "checkpoints": [str(p) for p in ckpts],
        "eval_slices": eval_slices,
        "image_size": image_size,
    }
    Path("b3_infer_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    return b3


def blend_dino_b3(dino: pd.DataFrame, b3: pd.DataFrame, weights: dict) -> pd.DataFrame:
    dino = dino.set_index(UID)
    b3 = b3.set_index(UID)
    # align to dino index (test order)
    uids = dino.index.astype(str).tolist()
    D = dino.reindex(uids)[BLEND_TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    B = b3.reindex(uids)[BLEND_TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(D).all():
        raise RuntimeError("DINO predictions contain non-finite values")
    if not np.isfinite(B).all():
        # fill missing B3 with DINO so blend degrades safely
        bad = ~np.isfinite(B)
        B[bad] = D[bad]
    Rd = _rank_pct_matrix(D)
    Rb = _rank_pct_matrix(B)
    out = np.zeros_like(D, dtype=np.float64)
    used = {"dino_only": [], "equal_rank": [], "b3_only": []}
    for t, name in enumerate(BLEND_TARGETS):
        w = weights.get(name) or weights.get(name.replace("'", "’")) or {"dino": 1.0, "b3": 0.0}
        wd = float(w.get("dino", 0.0))
        wb = float(w.get("b3", 0.0))
        if wb <= 0 and wd > 0:
            out[:, t] = Rd[:, t]
            used["dino_only"].append(name)
        elif wd <= 0 and wb > 0:
            out[:, t] = Rb[:, t]
            used["b3_only"].append(name)
        else:
            total = wd + wb
            out[:, t] = (wd * Rd[:, t] + wb * Rb[:, t]) / max(total, 1e-12)
            used["equal_rank"].append(name)
    result = pd.DataFrame({UID: uids})
    for t, name in enumerate(BLEND_TARGETS):
        result[name] = out[:, t]
    Path("v5_blend_meta.json").write_text(json.dumps(used, indent=2) + "\n")
    print("blend target policy:", json.dumps(used), flush=True)
    return result


def v5_blend_main():
    dino_path = Path("submission.csv")
    if not dino_path.is_file():
        raise FileNotFoundError("V1 path did not write submission.csv")
    shutil.copy2(dino_path, "dino_submission.csv")
    dino = pd.read_csv("dino_submission.csv")
    print("DINO submission", dino.shape, flush=True)
    weights = _load_blend_weights()
    # If all dino-only, skip B3 to save GPU (still write report)
    needs_b3 = any(float((weights.get(t) or {}).get("b3", 0.0)) > 0 for t in BLEND_TARGETS)
    if not needs_b3:
        print("all targets dino-only; keeping DINO submission", flush=True)
        return
    b3 = run_b3_infer()
    blended = blend_dino_b3(dino, b3, weights)
    blended.to_csv("submission.csv", index=False)
    print("wrote blended submission.csv", blended.shape, flush=True)
    print(blended.head().to_string(), flush=True)


try:
    v5_blend_main()
except Exception:
    traceback.print_exc()
    # Keep DINO submission if blend fails — never wipe a valid 0.899 path.
    if Path("dino_submission.csv").is_file() and not Path("submission.csv").is_file():
        shutil.copy2("dino_submission.csv", "submission.csv")
    print("V5 blend failed; left existing submission.csv if present", flush=True)
'''


def code_cell(source: str) -> dict:
    # notebook source is list of lines
    if not source.endswith("\n"):
        source = source + "\n"
    return {
        "cell_type": "code",
        "id": hashlib.sha1(source.encode("utf-8")).hexdigest()[:12],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    nb = json.loads(V1_NB.read_text(encoding="utf-8"))
    # Append blend cell after V1 main()
    nb["cells"].append(code_cell(BLEND_CELL.lstrip("\n")))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_nb = OUT_DIR / "rsna-knee-v5-dino-b3-blend.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("wrote", out_nb)

    meta = {
        "id": "danielsolo1770/rsna-knee-v5-dino-b3-gold-blend",
        "title": "RSNA Knee V5 DINO+B3 gold-gated blend",
        "code_file": "rsna-knee-v5-dino-b3-blend.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "pilkwang/pilkwang-public-dataset-for-notebooks-figures",
            "pilkwang/rsna-knee-llm-labels",
            "pilkwang/rsna-knee-weights",
            "danielsolo1770/rsna-knee-v5-b3-checkpoints",
        ],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [
            "metaresearch/dinov2/PyTorch/small/1",
            "timm/tf-efficientnet/PyTorch/tf-efficientnet-b3/1",
        ],
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote kernel-metadata.json")

    # Stage dataset payload
    ds = ROOT / "artifacts" / "kaggle_dataset_b3_checkpoints"
    if ds.exists():
        shutil.rmtree(ds)
    ds.mkdir(parents=True)
    for f in range(5):
        shutil.copy2(CKPT_DIR / f"fold{f}_final.pt", ds / f"fold{f}_final.pt")
    for name in MODULES:
        shutil.copy2(TRAIN_SRC / name, ds / name)
    gold = CKPT_DIR / "blend_report_gold.json"
    if not gold.is_file():
        gold = ROOT / "artifacts" / "blend_v5" / "blend_report_gold.json"
    shutil.copy2(gold, ds / "blend_report_gold.json")
    # optional full oof for auditing (not required at infer)
    oof = CKPT_DIR / "b3_full_oof.csv"
    if oof.is_file():
        shutil.copy2(oof, ds / "b3_full_oof.csv")
    (ds / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "rsna-knee-v5-b3-checkpoints",
                "id": "danielsolo1770/rsna-knee-v5-b3-checkpoints",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("staged dataset at", ds)
    for p in sorted(ds.iterdir()):
        print(" ", p.name, p.stat().st_size)


if __name__ == "__main__":
    main()
