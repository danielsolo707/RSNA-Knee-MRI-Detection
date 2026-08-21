#!/usr/bin/env python3
"""Build self-contained V5 submit notebook (no new dataset upload required).

- Clones V1 DINO 0.899 notebook
- Writes B3 source modules via %%writefile
- Appends gold-gated DINO+B3 blend cell
- Checkpoints come from kernel_sources of completed fold notebooks
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_NB = ROOT / "kaggle" / "submission_v1" / "rsna-knee-v1-public-0899.ipynb"
OUT_DIR = ROOT / "kaggle" / "submission_v5_dino_b3"
TRAIN_SRC = ROOT / "kaggle" / "v3_b3_fold0"
GOLD_PATH = ROOT / "artifacts" / "blend_v5" / "blend_report_gold.json"
MODULES = (
    "efficientnet_b3_public_repro_v1.py",
    "efficientnet_b3_public_repro_v2_anatomy.py",
    "efficientnet_b3_public_repro_v4_t4.py",
)


def code_cell(source: str) -> dict:
    if not source.endswith("\n"):
        source += "\n"
    return {
        "cell_type": "code",
        "id": hashlib.sha1(source.encode("utf-8")).hexdigest()[:12],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def blend_cell_source(weights: dict, gold_meta: dict) -> str:
    weights_json = json.dumps(weights, sort_keys=True)
    dino_m = gold_meta.get("dino_gold_macro")
    pol_m = gold_meta.get("policy_gold_macro")
    return f'''# === V5 gold-gated DINO + B3 rank blend ===
from __future__ import annotations

import json
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
GOLD_WEIGHTS = json.loads({weights_json!r})


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


def _find_one(pattern: str):
    hits = sorted(Path("/kaggle/input").glob(pattern))
    return hits[0] if hits else None


def _load_module(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {{path}}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    if str(path.parent) not in sys.path:
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


def _find_checkpoints():
    paths = []
    for fold in range(5):
        hit = _find_one(f"**/fold{{fold}}_final.pt")
        if hit is None:
            raise FileNotFoundError(f"fold{{fold}}_final.pt not found under /kaggle/input")
        paths.append(hit)
    return paths


def run_b3_infer() -> pd.DataFrame:
    t0 = time.time()
    src_dir = Path("/kaggle/working/v3_b3_source")
    v4 = _load_module(src_dir / "efficientnet_b3_public_repro_v4_t4.py", "effnet_v4_infer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU required for B3 inference")
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(0) >= (8, 0) else torch.float16
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
        print(f"loaded {{path}} fold={{ck.get('fold')}}", flush=True)
    if set(folds) != set(range(5)):
        raise ValueError(f"expected folds 0..4, got {{folds}}")

    comp = _find_competition()
    test = pd.read_csv(comp / "test.csv")
    series = pd.read_csv(comp / "test_series.csv")
    image_root = comp / "test_series"
    if not image_root.is_dir():
        alt = list(comp.glob("**/test_series"))
        image_root = alt[0] if alt else image_root
    grouped = {{str(uid): g.copy() for uid, g in series.groupby(UID, sort=False)}}
    empty = pd.DataFrame(columns=["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"])
    uids = test[UID].astype(str).tolist()
    eval_slices = int(config.valid_slices)
    image_size = int(config.image_size)
    predictions, errors = {{}}, {{}}
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
                        image_root / uid / str(series_uid), image_size, config.max_series_slices
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
                raise RuntimeError("non-finite")
            predictions[uid] = value
        except Exception as exc:
            errors[uid] = repr(exc)
            predictions[uid] = np.full(len(v4.TARGETS), 0.5, dtype=float)
        if (i + 1) % 50 == 0 or (i + 1) == len(uids):
            print(
                f"B3 {{i+1}}/{{len(uids)}} elapsed={{(time.time()-t0)/60:.1f}}m errors={{len(errors)}}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    rows = []
    for uid in uids:
        row = {{UID: uid}}
        for t, v in zip(v4.TARGETS, predictions[uid]):
            row[t] = float(v)
        rows.append(row)
    b3 = pd.DataFrame(rows)[[UID, *BLEND_TARGETS]]
    b3.to_csv("b3_submission.csv", index=False)
    Path("b3_infer_meta.json").write_text(
        json.dumps(
            {{
                "studies": len(uids),
                "errors": len(errors),
                "elapsed_seconds": time.time() - t0,
                "checkpoints": [str(p) for p in ckpts],
            }},
            indent=2,
        )
        + "\\n"
    )
    return b3


def blend_dino_b3(dino: pd.DataFrame, b3: pd.DataFrame, weights: dict) -> pd.DataFrame:
    dino = dino.set_index(UID)
    b3 = b3.set_index(UID)
    uids = dino.index.astype(str).tolist()
    D = dino.reindex(uids)[BLEND_TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    B = b3.reindex(uids)[BLEND_TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(D).all():
        raise RuntimeError("DINO non-finite")
    bad = ~np.isfinite(B)
    if bad.any():
        B[bad] = D[bad]
    Rd, Rb = _rank_pct_matrix(D), _rank_pct_matrix(B)
    out = np.zeros_like(D, dtype=np.float64)
    used = {{"dino_only": [], "equal_rank": [], "b3_only": []}}
    for t, name in enumerate(BLEND_TARGETS):
        w = weights.get(name) or {{"dino": 1.0, "b3": 0.0}}
        wd, wb = float(w.get("dino", 0.0)), float(w.get("b3", 0.0))
        if wb <= 0:
            out[:, t] = Rd[:, t]
            used["dino_only"].append(name)
        elif wd <= 0:
            out[:, t] = Rb[:, t]
            used["b3_only"].append(name)
        else:
            out[:, t] = (wd * Rd[:, t] + wb * Rb[:, t]) / max(wd + wb, 1e-12)
            used["equal_rank"].append(name)
    result = pd.DataFrame({{UID: uids}})
    for t, name in enumerate(BLEND_TARGETS):
        result[name] = out[:, t]
    Path("v5_blend_meta.json").write_text(json.dumps(used, indent=2) + "\\n")
    print("blend policy", used, flush=True)
    return result


def v5_blend_main():
    if not Path("submission.csv").is_file():
        raise FileNotFoundError("V1 did not write submission.csv")
    shutil.copy2("submission.csv", "dino_submission.csv")
    dino = pd.read_csv("dino_submission.csv")
    print("DINO", dino.shape, flush=True)
    Path("blend_report_gold.json").write_text(
        json.dumps(
            {{
                "weights": GOLD_WEIGHTS,
                "policy": "gold_loo_gain_ge_0.005",
                "dino_gold_macro": {dino_m},
                "policy_gold_macro": {pol_m},
            }},
            indent=2,
        )
        + "\\n"
    )
    b3 = run_b3_infer()
    blended = blend_dino_b3(dino, b3, GOLD_WEIGHTS)
    blended.to_csv("submission.csv", index=False)
    print("wrote blended submission.csv", blended.shape, flush=True)


try:
    v5_blend_main()
except Exception:
    traceback.print_exc()
    if Path("dino_submission.csv").is_file():
        shutil.copy2("dino_submission.csv", "submission.csv")
        print("blend failed; restored DINO submission.csv", flush=True)
'''


def main() -> None:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    weights = gold["weights"]
    v1 = json.loads(V1_NB.read_text(encoding="utf-8"))
    cells = list(v1["cells"])
    cells.append(
        code_cell(
            "from pathlib import Path\n"
            "Path('/kaggle/working/v3_b3_source').mkdir(parents=True, exist_ok=True)\n"
        )
    )
    for name in MODULES:
        text = (TRAIN_SRC / name).read_text(encoding="utf-8")
        cells.append(code_cell(f"%%writefile /kaggle/working/v3_b3_source/{name}\n{text}"))
    cells.append(code_cell(blend_cell_source(weights, gold)))

    nb = {
        "cells": cells,
        "metadata": v1.get(
            "metadata",
            {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
            },
        ),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_nb = OUT_DIR / "rsna-knee-v5-dino-b3-blend.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("wrote", out_nb, "cells", len(cells), "bytes", out_nb.stat().st_size)

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
        ],
        "kernel_sources": [
            "danielsolo1770/rsna-knee-v5-b3-fold0",
            "danielsolo1770/rsna-knee-v5-b3-fold1",
            "danielsolo1770/rsna-knee-v5-b3-fold2",
            "danielsolo1770/rsna-knee-v5-b3-fold3",
            "danielsolo1770/rsna-knee-v5-b3-fold4",
        ],
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
    print(
        "gold policy macro",
        gold.get("policy_gold_macro"),
        "vs dino",
        gold.get("dino_gold_macro"),
    )


if __name__ == "__main__":
    main()
