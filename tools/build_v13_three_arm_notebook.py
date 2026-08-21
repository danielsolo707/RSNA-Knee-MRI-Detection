#!/usr/bin/env python3
"""Build V13: fixed three-arm DINO + E9 Rad-FS + E11 Rad-NOFS.

The notebook always recreates the verified DINO parent first. It then recreates
V9 from pinned E9 heads and keeps V9 as the fallback before attempting E11.
V13 may replace V9 only when the fixed external three-arm recipe passes the
independent OOF, grouped-fold, bootstrap, artifact, and schema gates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_NB = ROOT / "kaggle" / "submission_v1" / "rsna-knee-v1-public-0899.ipynb"
OUT_DIR = ROOT / "kaggle" / "submission_v13_three_arm"
OUT_NB = OUT_DIR / "rsna-knee-v13-three-arm-gated.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

V13_ARM = r'''
# === V13: fixed DINO + E9 Rad-FS + E11 Rad-NOFS ==============================
# Fixed external recipe, no in-kernel fitting of blend weights:
#   ten targets: 0.30 DINO + 0.35 E9 + 0.35 E11
#   Baker's/Fracture: DINO only
# V9 is materialised first and becomes the durable fallback before E11 runs.

import gc as _v13_gc
import hashlib as _v13_hashlib
import json as _v13_json
import shutil as _v13_shutil
import time as _v13_time
import traceback as _v13_tb
from pathlib import Path as _V13Path

import numpy as _v13_np
import pandas as _v13_pd
import torch as _v13_torch
import torch.nn as _v13_nn

_V13_TARGETS = list(TARGETS)
_V13_UID = "StudyInstanceUID"
_V13_WORK = _V13Path("/kaggle/working")
_V13_PRIMARY = _V13_WORK / "submission.csv"
_V13_DINO = _V13_WORK / "submission_dino_v1.csv"
_V13_V9 = _V13_WORK / "submission_v9_parent.csv"
_V13_AUDIT = _V13_WORK / "v13_three_arm_audit.json"
_V13_OUT = _V13_WORK / "rsna_v13"
_V13_OUT.mkdir(parents=True, exist_ok=True)

_V13_TOKEN_DIM = 2048
_V13_HEAD_DIM = 512
_V13_RUNTIME_CAP = 4.50 * 3600
_V13_MIN_E11_SECONDS = 0.45 * 3600
_V13_MIN_GAIN = 0.005
_V13_MIN_BOOTSTRAP_WIN = 0.80
_V13_MIN_POSITIVE_FOLDS = 3
_V13_MAX_SOFT_DROP = 0.005
_V13_PRESERVED = {"Baker's", "Fracture"}
_V13_E9_SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
]
_V13_E11_SLOTS = [
    ("SAG_NOFS", "Sagittal", None, False),
    ("COR_NOFS", "Coronal", None, False),
    ("AX_NOFS", "Axial", None, False),
    ("SAG_FS", "Sagittal", None, True),
]
_V13_SHA = {
    "encoder": "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734",
    "dino_oof": "62d47ba4c0c8347b5b24e7fd2aa517aae0fd6d4656fd5d829fd3a50f0159909c",
    "e9_heads": "0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa",
    "e9_oof": "f5b1f02029af98740565bec5b6a76d8471b4b6a6039d09ea410368230cb1f03f",
    "e11_heads": "df98518d0a8f7cc506b4b806da03c3fca18c8f8f743e813c5401759b45e0f1fe",
    "e11_oof": "f5415485f57103fe86370e3d4bba420485a7497ef199d40545a20f6e53a0f0e2",
}


def _v13_log(message):
    try:
        log(f"[V13] {message}")
    except Exception:
        print(f"[V13] {message}", flush=True)


def _v13_check_deadline(tag, reserve_seconds=0.0):
    elapsed = _v13_time.time() - T0
    remaining = _V13_RUNTIME_CAP - elapsed
    if remaining <= float(reserve_seconds):
        raise TimeoutError(
            f"{tag}: end-to-end runtime cap; elapsed={elapsed:.1f}s "
            f"remaining={remaining:.1f}s reserve={float(reserve_seconds):.1f}s"
        )
    return remaining


def _v13_sha(path):
    h = _v13_hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _v13_find(name, expected_sha=None):
    found = []
    for path in _V13Path("/kaggle/input").rglob(name):
        if path.is_file() and "train_series" not in path.parts and "test_series" not in path.parts:
            found.append(path)
    if not found:
        raise FileNotFoundError(name)
    if expected_sha is not None:
        matches = [path for path in found if _v13_sha(path) == expected_sha]
        if len(matches) != 1:
            raise RuntimeError(f"{name}: expected exactly one hash match, got {len(matches)}")
        return matches[0]
    if len(found) != 1:
        raise RuntimeError(f"{name}: ambiguous inputs: {[str(path) for path in found]}")
    return found[0]


def _v13_rank(values):
    return _v13_pd.DataFrame(_v13_np.asarray(values, dtype=float)).rank(
        method="average", pct=True
    ).to_numpy(float)


def _v13_auc(y, score):
    y = _v13_np.asarray(y, dtype=float)
    score = _v13_np.asarray(score, dtype=float)
    positive = y >= 0.5
    negative = ~positive
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = _v13_np.argsort(score, kind="mergesort")
    ranks = _v13_np.empty(len(score), dtype=float)
    ranks[order] = _v13_np.arange(1, len(score) + 1, dtype=float)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _v13_macro(y, pred):
    scores = [_v13_auc(y[:, j], pred[:, j]) for j in range(pred.shape[1])]
    scores = [score for score in scores if _v13_np.isfinite(score)]
    return float(_v13_np.mean(scores)) if scores else float("nan")


def _v13_validate(frame, expected_ids):
    if frame.columns.tolist() != [_V13_UID] + _V13_TARGETS:
        raise RuntimeError(f"submission schema drift: {frame.columns.tolist()}")
    ids = frame[_V13_UID].astype(str).tolist()
    expected = list(map(str, expected_ids))
    if ids != expected or len(ids) != len(set(ids)):
        raise RuntimeError("submission ID order/uniqueness drift")
    values = frame[_V13_TARGETS].to_numpy(float)
    if not _v13_np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise RuntimeError("invalid submission values")


def _v13_load_oof(path, train):
    frame = _v13_pd.read_csv(path, dtype={_V13_UID: str})
    if frame[_V13_UID].duplicated().any():
        raise RuntimeError(f"duplicate OOF IDs in {path}")
    aligned = train[[_V13_UID]].merge(frame, on=_V13_UID, how="left", validate="one_to_one")
    pred = aligned[_V13_TARGETS].apply(_v13_pd.to_numeric, errors="coerce").to_numpy(float)
    if not _v13_np.isfinite(pred).all():
        raise RuntimeError(f"non-finite OOF after alignment: {path}")
    return pred, aligned["fold"].to_numpy(int)


def _v13_three_arm(dino, e9, e11):
    dino_r, e9_r, e11_r = _v13_rank(dino), _v13_rank(e9), _v13_rank(e11)
    candidate = 0.30 * dino_r + 0.35 * e9_r + 0.35 * e11_r
    for j, target in enumerate(_V13_TARGETS):
        if target in _V13_PRESERVED:
            candidate[:, j] = dino_r[:, j]
    return candidate


def _v13_gate(train):
    dino_path = _v13_find("oof.npz", _V13_SHA["dino_oof"])
    e9_path = _v13_find("v52_oof.csv", _V13_SHA["e9_oof"])
    e11_path = _v13_find("v52_e11_oof.csv", _V13_SHA["e11_oof"])
    with _v13_np.load(dino_path, allow_pickle=True) as bundle:
        ids = [str(value) for value in bundle["ids"].tolist()]
        dino = bundle["pred"].astype(float)
        derived = bundle["y_derived"].astype(float)
        npz_gold = bundle["gold_mask"].astype(bool)
        if bundle["targets"].astype(str).tolist() != _V13_TARGETS:
            raise RuntimeError("DINO OOF target order drift")
    if ids != train[_V13_UID].astype(str).tolist():
        raise RuntimeError("DINO OOF study order drift")
    e9, fold9 = _v13_load_oof(e9_path, train)
    e11, fold11 = _v13_load_oof(e11_path, train)
    if not _v13_np.array_equal(fold9, fold11):
        raise RuntimeError("E9/E11 fold assignments differ")
    gold = train[_V13_TARGETS].notna().all(axis=1).to_numpy()
    if int(gold.sum()) != 58 or not _v13_np.array_equal(gold, npz_gold):
        raise RuntimeError(f"gold contract drift: {int(gold.sum())}")
    y = train.loc[gold, _V13_TARGETS].to_numpy(float)
    dino_g, e9_g, e11_g = dino[gold], e9[gold], e11[gold]
    parent = 0.80 * _v13_rank(dino_g) + 0.20 * _v13_rank(e9_g)
    candidate = _v13_three_arm(dino_g, e9_g, e11_g)
    parent_score = _v13_macro(y, parent)
    candidate_score = _v13_macro(y, candidate)
    gain = candidate_score - parent_score
    folds = fold9[gold]
    fold_receipts = []
    for fold in sorted(set(folds.tolist())):
        mask = folds == fold
        pscore, cscore = _v13_macro(y[mask], parent[mask]), _v13_macro(y[mask], candidate[mask])
        fold_receipts.append({"fold": int(fold), "rows": int(mask.sum()), "v9": pscore,
                              "candidate": cscore, "gain": cscore - pscore})
    positive_folds = sum(record["gain"] > 0 for record in fold_receipts)
    rng = _v13_np.random.default_rng(13035)
    deltas = []
    for _ in range(5000):
        sample = rng.integers(0, len(y), size=len(y))
        pscore = _v13_macro(y[sample], parent[sample])
        cscore = _v13_macro(y[sample], candidate[sample])
        if _v13_np.isfinite(pscore) and _v13_np.isfinite(cscore):
            deltas.append(cscore - pscore)
    deltas = _v13_np.asarray(deltas, dtype=float)
    soft_parent = _v13_macro(derived, 0.80 * _v13_rank(dino) + 0.20 * _v13_rank(e9))
    soft_candidate = _v13_macro(derived, _v13_three_arm(dino, e9, e11))
    bootstrap_win = float(_v13_np.mean(deltas > 0))
    promote = bool(
        gain >= _V13_MIN_GAIN
        and positive_folds >= _V13_MIN_POSITIVE_FOLDS
        and bootstrap_win >= _V13_MIN_BOOTSTRAP_WIN
        and soft_candidate >= soft_parent - _V13_MAX_SOFT_DROP
    )
    receipt = {
        "v9_gold_macro": parent_score,
        "candidate_gold_macro": candidate_score,
        "gold_gain": gain,
        "fold_results": fold_receipts,
        "positive_folds": int(positive_folds),
        "bootstrap_valid": int(len(deltas)),
        "bootstrap_win": bootstrap_win,
        "bootstrap_mean_delta": float(deltas.mean()),
        "bootstrap_q025": float(_v13_np.quantile(deltas, 0.025)),
        "soft_v9_macro": soft_parent,
        "soft_candidate_macro": soft_candidate,
        "soft_delta": soft_candidate - soft_parent,
        "gate_pass": promote,
        "oof_hashes": {"dino": _v13_sha(dino_path), "e9": _v13_sha(e9_path), "e11": _v13_sha(e11_path)},
    }
    _v13_log(f"gate V9={parent_score:.5f} V13={candidate_score:.5f} gain={gain:+.5f} "
             f"folds={positive_folds}/5 boot={bootstrap_win:.3f} soft={soft_candidate-soft_parent:+.5f}")
    return receipt


class _V13Head(_v13_nn.Module):
    def __init__(self, n_slot, n_slices=8):
        super().__init__()
        self.n_slot = int(n_slot)
        self.n_slices = int(n_slices)
        self.project = _v13_nn.Sequential(
            _v13_nn.LayerNorm(_V13_TOKEN_DIM),
            _v13_nn.Linear(_V13_TOKEN_DIM, _V13_HEAD_DIM),
            _v13_nn.GELU(),
        )
        self.plane = _v13_nn.Parameter(_v13_torch.randn(n_slot, _V13_HEAD_DIM) * 0.01)
        self.position = _v13_nn.Parameter(_v13_torch.randn(n_slices, _V13_HEAD_DIM) * 0.01)
        self.query = _v13_nn.Parameter(_v13_torch.randn(len(_V13_TARGETS), _V13_HEAD_DIM) * 0.02)
        self.attn = _v13_nn.MultiheadAttention(_V13_HEAD_DIM, 8, dropout=0.10, batch_first=True)
        self.fuse = _v13_nn.Sequential(
            _v13_nn.LayerNorm(_V13_HEAD_DIM * 4),
            _v13_nn.Linear(_V13_HEAD_DIM * 4, _V13_HEAD_DIM),
            _v13_nn.GELU(),
            _v13_nn.Dropout(0.15),
        )
        self.weight = _v13_nn.Parameter(_v13_torch.randn(len(_V13_TARGETS), _V13_HEAD_DIM) * 0.02)
        self.bias = _v13_nn.Parameter(_v13_torch.zeros(len(_V13_TARGETS)))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), self.n_slot, self.n_slices, _V13_HEAD_DIM)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        key_padding = mask <= 0
        empty = key_padding.all(1)
        if empty.any():
            key_padding = key_padding.clone()
            key_padding[empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(query, token, token, key_padding_mask=key_padding,
                                     need_weights=False)[0]
        denom = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdims=True) / denom
        mean = mean.expand(-1, len(_V13_TARGETS), -1)
        fused = self.fuse(_v13_torch.cat(
            [attended, mean, _v13_torch.abs(attended - mean), attended * mean], -1
        ))
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


def _v13_load_encoder(device):
    from torchvision.models import resnet50
    path = _v13_find("ResNet50.pt", _V13_SHA["encoder"])

    class Encoder(_v13_nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _v13_nn.Sequential(*list(resnet50(weights=None).children())[:-2])
        def forward(self, image):
            return self.backbone(image).mean(dim=(2, 3))

    model = Encoder()
    state = _v13_torch.load(str(path), map_location="cpu", weights_only=True)
    if state and all(str(key).startswith("backbone.") for key in state):
        model.load_state_dict(state, strict=True)
    else:
        cleaned = {}
        for key, value in state.items():
            key = str(key).replace("module.", "", 1)
            if key.startswith("fc."):
                continue
            cleaned[key if key.startswith("backbone.") else "backbone." + key] = value
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"encoder state mismatch missing={missing[:5]} unexpected={unexpected[:5]}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@_v13_torch.inference_mode()
def _v13_encode(cache, slot_mask, device):
    n, slots, slices, height, width = cache.shape
    features = _v13_np.zeros((n, slots * slices, _V13_TOKEN_DIM), _v13_np.float16)
    token_mask = _v13_np.repeat(slot_mask[:, :, None], slices, axis=2).reshape(n, -1)
    valid = _v13_np.flatnonzero(token_mask.reshape(-1) > 0)
    flat = cache.reshape(-1, height, width)
    model = _v13_load_encoder(device)
    batch = 64
    for start in range(0, len(valid), batch):
        _v13_check_deadline("RadImageNet encoder batch")
        index = valid[start:start + batch]
        image = _v13_torch.from_numpy(flat[index]).to(device).float().div_(127.5).sub_(1.0)
        image = image.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        with _v13_torch.autocast("cuda", enabled=True):
            encoded = model(image)
        if encoded.shape[1:] != (_V13_TOKEN_DIM,):
            raise RuntimeError(f"encoder shape drift: {tuple(encoded.shape)}")
        features.reshape(-1, _V13_TOKEN_DIM)[index] = encoded.float().cpu().numpy().astype(_v13_np.float16)
    del model
    _v13_torch.cuda.empty_cache()
    return features, token_mask.astype(_v13_np.float32)


def _v13_load_states(name, expected_sha, n_slot):
    path = _v13_find(name, expected_sha)
    payload = _v13_torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("targets", _V13_TARGETS) != _V13_TARGETS:
        raise RuntimeError(f"target order drift in {name}")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError(f"expected five folds in {name}")
    states = []
    for record in folds:
        state = record.get("state_dict") if isinstance(record, dict) else None
        if state is None or "plane" not in state or int(state["plane"].shape[0]) != n_slot:
            raise RuntimeError(f"head slot/state drift in {name}")
        states.append(state)
    return states, str(path)


@_v13_torch.inference_mode()
def _v13_predict_states(states, features, masks, n_slot, device):
    all_index = _v13_np.arange(len(features), dtype=_v13_np.int64)
    fold_predictions = []
    for fold_index, state in enumerate(states):
        _v13_check_deadline(f"RadImageNet head fold {fold_index}")
        head = _V13Head(n_slot=n_slot).to(device)
        head.load_state_dict(state, strict=True)
        chunks = []
        for start in range(0, len(all_index), 64):
            _v13_check_deadline(f"RadImageNet head fold {fold_index} batch")
            index = all_index[start:start + 64]
            feature = _v13_torch.from_numpy(features[index]).to(device)
            mask = _v13_torch.from_numpy(masks[index]).to(device)
            with _v13_torch.autocast("cuda", enabled=True):
                chunks.append(_v13_torch.sigmoid(head(feature, mask)).float().cpu())
        fold_predictions.append(_v13_torch.cat(chunks).numpy())
        del head
        _v13_torch.cuda.empty_cache()
    return _v13_np.mean(_v13_np.stack(fold_predictions), axis=0)


def _v13_test_arm(test, test_series, plane_map, slots, crop_mm, heads_name, heads_sha, device, tag, min_fill):
    global SLOTS, N_SLOT, IMG, CACHE_IMG, CACHE_SLICES, CROP_MM, SLICE_BAND, GROUP, N_GROUP, RULES, TIME_BUDGET
    saved = {name: globals()[name] for name in (
        "SLOTS", "N_SLOT", "IMG", "CACHE_IMG", "CACHE_SLICES", "CROP_MM",
        "SLICE_BAND", "GROUP", "N_GROUP", "RULES", "TIME_BUDGET"
    )}
    try:
        SLOTS = list(slots)
        N_SLOT = len(slots)
        IMG = CACHE_IMG = 224
        CACHE_SLICES = 8
        GROUP = 1
        N_GROUP = 8
        CROP_MM = float(crop_mm)
        SLICE_BAND = (0.12, 0.88)
        RULES = dict(RULES_LEGACY)
        TIME_BUDGET = min(float(saved["TIME_BUDGET"]), _V13_RUNTIME_CAP)
        headers = annotate(walk("test_series"))
        selected = pick_slots(headers, plane_map)
        studies, cache, slot_mask = build_cache(
            selected, plane_map, lat_of(headers, f"{tag} "), tag
        )
        by_uid = {str(uid): i for i, uid in enumerate(studies)}
        missing = [str(uid) for uid in test[_V13_UID] if str(uid) not in by_uid]
        if missing:
            raise RuntimeError(f"{tag}: {len(missing)} test studies missing from cache")
        order = _v13_np.asarray([by_uid[str(uid)] for uid in test[_V13_UID]], dtype=_v13_np.int64)
        cache, slot_mask = cache[order], slot_mask[order]
        fill = {slot[0]: float(slot_mask[:, i].mean()) for i, slot in enumerate(slots)}
        overall_fill = float(slot_mask.mean())
        empty_studies = int((slot_mask.sum(axis=1) < 1).sum())
        if empty_studies:
            raise RuntimeError(f"{tag}: {empty_studies} studies have no usable slot")
        if overall_fill < float(min_fill):
            raise RuntimeError(f"{tag}: slot fill {overall_fill:.1%} below {float(min_fill):.1%}")
        _v13_check_deadline(f"{tag}: before encoding")
        features, masks = _v13_encode(cache, slot_mask, device)
        del cache, slot_mask, headers, selected
        _v13_gc.collect()
        states, source = _v13_load_states(heads_name, heads_sha, len(slots))
        pred = _v13_predict_states(states, features, masks, len(slots), device)
        _v13_check_deadline(f"{tag}: after inference")
        del features, masks
        _v13_gc.collect()
        if not _v13_np.isfinite(pred).all():
            raise RuntimeError(f"{tag}: non-finite predictions")
        return pred, fill, source
    finally:
        for name, value in saved.items():
            globals()[name] = value


def run_v13():
    audit = {
        "status": "DINO_ONLY",
        "promote": False,
        "recipe": "ten targets 0.30 DINO + 0.35 E9 + 0.35 E11; Baker's/Fracture DINO",
        "public_parent_reference": 0.906,
        "evidence_boundary": "OOF gate is diagnostic; only Kaggle public score is leaderboard evidence.",
    }
    if not _V13_PRIMARY.is_file():
        raise FileNotFoundError("DINO parent submission missing")
    _v13_shutil.copy2(_V13_PRIMARY, _V13_DINO)
    baseline = _v13_pd.read_csv(_V13_DINO, dtype={_V13_UID: str})
    train = _v13_pd.read_csv(ROOT / "train.csv", dtype={_V13_UID: str})
    test = _v13_pd.read_csv(ROOT / "test.csv", dtype={_V13_UID: str})
    _v13_validate(baseline, test[_V13_UID])
    audit["dino_submission_sha256"] = _v13_sha(_V13_DINO)

    device = _v13_torch.device("cuda:0" if _v13_torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V13 requires CUDA")
    test_series = _v13_pd.read_csv(
        ROOT / "test_series.csv", dtype={_V13_UID: str, "SeriesInstanceUID": str}
    )
    train_series = _v13_pd.read_csv(
        ROOT / "train_series.csv", dtype={_V13_UID: str, "SeriesInstanceUID": str}
    )
    both = _v13_pd.concat([train_series, test_series], ignore_index=True)
    plane_map = dict(zip(both["SeriesInstanceUID"], both["Anatomical_Plane"]))

    # Build V9 first. From this point forward every E11 failure preserves 0.906.
    e9_test, e9_fill, e9_source = _v13_test_arm(
        test, test_series, plane_map, _V13_E9_SLOTS, 10_000.0,
        "v52_radimagenet_heads.pt", _V13_SHA["e9_heads"], device, "v13-e9-test", 0.80
    )
    dino_rank, e9_rank = _v13_rank(baseline[_V13_TARGETS].to_numpy(float)), _v13_rank(e9_test)
    v9 = baseline.copy()
    v9[_V13_TARGETS] = 0.80 * dino_rank + 0.20 * e9_rank
    _v13_validate(v9, test[_V13_UID])
    v9.to_csv(_V13_V9, index=False)
    v9.to_csv(_V13_OUT / "submission_v9_parent.csv", index=False)
    _v13_shutil.copy2(_V13_V9, _V13_PRIMARY)
    audit.update({"status": "V9_PRESERVED", "e9_fill": e9_fill, "e9_heads": e9_source,
                  "v9_submission_sha256": _v13_sha(_V13_V9)})

    # Materialize the scored parent before any diagnostic-only gate can fail.
    audit["gate"] = _v13_gate(train)
    if not audit["gate"]["gate_pass"]:
        audit["status"] = "OOF_REJECT_KEEP_V9"
        return audit
    remaining = _v13_check_deadline("before E11", _V13_MIN_E11_SECONDS)
    audit["seconds_before_e11"] = remaining

    e11_test, e11_fill, e11_source = _v13_test_arm(
        test, test_series, plane_map, _V13_E11_SLOTS, 130.0,
        "v52_e11_heads.pt", _V13_SHA["e11_heads"], device, "v13-e11-test", 0.45
    )
    candidate_values = 0.30 * dino_rank + 0.35 * e9_rank + 0.35 * _v13_rank(e11_test)
    for j, target in enumerate(_V13_TARGETS):
        if target in _V13_PRESERVED:
            candidate_values[:, j] = dino_rank[:, j]
    candidate = baseline.copy()
    candidate[_V13_TARGETS] = candidate_values
    _v13_validate(candidate, test[_V13_UID])
    _v13_check_deadline("before V13 selection/write")
    selected = _V13_OUT / "submission_v13_three_arm.csv"
    candidate.to_csv(selected, index=False)
    _v13_shutil.copy2(selected, _V13_PRIMARY)
    audit.update({
        "status": "V13_SELECTED",
        "promote": True,
        "e11_fill": e11_fill,
        "e11_heads": e11_source,
        "selected_path": str(selected),
        "selected_sha256": _v13_sha(selected),
        "elapsed_seconds": float(_v13_time.time() - T0),
    })
    return audit


# Apply the 4.5-hour ceiling to the parent as well as both added arms. All
# components share T0, so the cap is end-to-end rather than per-arm.
TIME_BUDGET = min(float(TIME_BUDGET), _V13_RUNTIME_CAP)

_V13_PARENT_OK = False
_V13_PARENT_ERROR = None
try:
    main()
    _V13_PARENT_OK = True
except LabelSourceError:
    traceback.print_exc()
    raise
except Exception as _v13_parent_error:
    traceback.print_exc()
    _V13_PARENT_ERROR = f"{type(_v13_parent_error).__name__}: {_v13_parent_error}"
    print("DINO parent failed; refusing to emit an emergency/chance submission")

_v13_receipt = {"status": "NOT_RUN", "promote": False}
if not _V13_PARENT_OK:
    _v13_receipt = {
        "status": "PARENT_ERROR_NO_SUBMISSION",
        "promote": False,
        "error": _V13_PARENT_ERROR,
    }
else:
    try:
        _v13_receipt = run_v13()
    except Exception as _v13_error:
        _v13_tb.print_exc()
        _v13_receipt = {
            "status": "V13_ERROR_KEEP_BEST_AVAILABLE",
            "promote": False,
            "error": f"{type(_v13_error).__name__}: {_v13_error}",
            "traceback": _v13_tb.format_exc()[-5000:],
        }
        try:
            if _V13_V9.is_file():
                _v13_shutil.copy2(_V13_V9, _V13_PRIMARY)
                _v13_receipt["fallback"] = "V9"
            elif _V13_DINO.is_file():
                _v13_shutil.copy2(_V13_DINO, _V13_PRIMARY)
                _v13_receipt["fallback"] = "DINO"
        except Exception:
            pass

_V13_AUDIT.write_text(
    _v13_json.dumps(_v13_receipt, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)

print("V13 AUDIT", _v13_json.dumps({
    key: _v13_receipt.get(key) for key in (
        "status", "promote", "selected_sha256", "fallback", "elapsed_seconds", "error"
    )
}, sort_keys=True, default=str))
log("done")
'''


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


def main() -> None:
    notebook = json.loads(V1_NB.read_text(encoding="utf-8"))
    cells = list(notebook["cells"])
    final_index = None
    for index in range(len(cells) - 1, -1, -1):
        source = "".join(cells[index].get("source", []))
        if cells[index].get("cell_type") == "code" and "main()" in source and "try:" in source:
            final_index = index
            break
    if final_index is None:
        raise RuntimeError("V1 final execution cell not found")
    cells = cells[:final_index] + [code_cell(V13_ARM)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps({
        "cells": cells,
        "metadata": notebook.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 5,
    }, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": "danielsolo1770/rsna-knee-v13-three-arm-gated",
        "title": "rsna-knee-v13-three-arm-gated",
        "code_file": OUT_NB.name,
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
            "marwanmath/resnet-50-radimagenet-marwan",
            "antoinegg1/rsna-knee-e9-radimagenet-heads-v15",
            "antoinegg1/rsna-knee-e11-diverse-heads-v20",
        ],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
        "machine_shape": "NvidiaTeslaT4",
    }
    OUT_META.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_NB}")
    print(f"wrote {OUT_META}")


if __name__ == "__main__":
    main()
