#!/usr/bin/env python3
"""Build V11: own RadImageNet query heads on V3 fold-cross labels, dual-gated vs V9.

Pipeline:
  1) V1 DINO package → fallback
  2) Encode train+test with frozen RadImageNet (3 FS slots / 224 / 8)
  3) Train 5-fold FoundationQueryHead on labels_v3_fold{k}.csv + folds_v3.csv
  4) Reconstruct V9 parent OOF = rank-blend(DINO, public e9 Rad, α=0.20)
  5) Dual-gate own Rad vs V9 parent; promote only if nested gold ≥ parent + 0.002
  6) Else keep V9 parent test file if e9 available, else DINO
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_NB = ROOT / "kaggle" / "submission_v1" / "rsna-knee-v1-public-0899.ipynb"
OUT_DIR = ROOT / "kaggle" / "submission_v11_own_rad"
OUT_NB = OUT_DIR / "rsna-knee-v11-own-rad-gated.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

RAD_ARM = r'''
# === V11: own Rad heads on V3 labels, dual-gated vs V9 parent =================
# Parent public reference: V9 DINO+e9 Rad α=0.20 → 0.906
# Promote only if nested gold OOF ≥ V9 parent + 0.002 (stricter after V10 null).

import gc as _v11_gc
import hashlib as _v11_hashlib
import json as _v11_json
import shutil as _v11_shutil
from pathlib import Path as _V11Path

import numpy as _v11_np
import pandas as _v11_pd
import torch as _v11_torch
import torch.nn as _v11_nn
import torch.nn.functional as _v11_F

_V11_TARGETS = list(TARGETS)
_V11_UID = "StudyInstanceUID"
_V11_WORK = _V11Path("/kaggle/working")
_V11_PRIMARY = _V11_WORK / "submission.csv"
_V11_DINO = _V11_WORK / "submission_dino_v1.csv"
_V11_V9 = _V11_WORK / "submission_v9_parent.csv"
_V11_AUDIT = _V11_WORK / "v11_rad_audit.json"
_V11_OUT = _V11_WORK / "rsna_rad_v11"
_V11_TOKEN_DIM = 2048
_V11_HEAD_DIM = 512
_V11_BASE_ALPHA = 0.20
_V11_MIN_GAIN = 0.002  # stricter than V9/V10
_V11_SOFT_FLOOR = 0.82
_V11_ALPHA_GRID = _v11_np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], dtype=float)
_V11_RAD_SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
]
_V11_EXPECTED_CKPT = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"


def _v11_log(msg):
    try:
        log(f"[V11] {msg}")
    except Exception:
        print(f"[V11] {msg}", flush=True)


def _v11_sha(path):
    h = _v11_hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _v11_find_file(name):
    root = _V11Path("/kaggle/input")
    if not root.is_dir():
        raise FileNotFoundError(name)
    for p in root.rglob(name):
        parts = set(p.parts)
        if "train_series" in parts or "test_series" in parts:
            continue
        if p.is_file():
            return p
    raise FileNotFoundError(name)


def _v11_rank(values):
    frame = _v11_pd.DataFrame(_v11_np.asarray(values, dtype=_v11_np.float64))
    return frame.rank(method="average", pct=True).to_numpy(_v11_np.float64)


def _v11_auc_1d(y, s):
    y = _v11_np.asarray(y, dtype=float)
    s = _v11_np.asarray(s, dtype=float)
    hard = (y >= 0.5).astype(float)
    pos = hard == 1
    neg = hard == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = _v11_np.argsort(s, kind="mergesort")
    ranks = _v11_np.empty(len(s), dtype=float)
    ranks[order] = _v11_np.arange(1, len(s) + 1, dtype=float)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _v11_macro_auc(y, pred):
    vals = []
    for j in range(y.shape[1]):
        a = _v11_auc_1d(y[:, j], pred[:, j])
        if a == a:
            vals.append(a)
    return float(_v11_np.mean(vals)) if vals else float("nan")


def _v11_validate_sub(frame, expected_ids):
    cols = [_V11_UID] + _V11_TARGETS
    if frame.columns.tolist() != cols:
        raise RuntimeError(f"schema drift: {frame.columns.tolist()}")
    ids = frame[_V11_UID].astype(str).tolist()
    exp = list(map(str, expected_ids))
    if ids != exp or len(ids) != len(set(ids)):
        raise RuntimeError("study identity/order drift")
    vals = frame[_V11_TARGETS].to_numpy(float)
    if not _v11_np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
        raise RuntimeError("invalid probability values")


def _v11_blend_ranks(base_rank, rad_rank, alpha):
    a = float(alpha)
    return (1.0 - a) * base_rank + a * rad_rank


class _V11Head(_v11_nn.Module):
    def __init__(self, n_slot=3, n_slices=8, n_targets=12):
        super().__init__()
        self.n_slot = n_slot
        self.n_slices = n_slices
        self.project = _v11_nn.Sequential(
            _v11_nn.LayerNorm(_V11_TOKEN_DIM),
            _v11_nn.Linear(_V11_TOKEN_DIM, _V11_HEAD_DIM),
            _v11_nn.GELU(),
        )
        self.plane = _v11_nn.Parameter(_v11_torch.randn(n_slot, _V11_HEAD_DIM) * 0.01)
        self.position = _v11_nn.Parameter(_v11_torch.randn(n_slices, _V11_HEAD_DIM) * 0.01)
        self.query = _v11_nn.Parameter(_v11_torch.randn(n_targets, _V11_HEAD_DIM) * 0.02)
        self.attn = _v11_nn.MultiheadAttention(_V11_HEAD_DIM, 8, dropout=0.10, batch_first=True)
        self.fuse = _v11_nn.Sequential(
            _v11_nn.LayerNorm(_V11_HEAD_DIM * 4),
            _v11_nn.Linear(_V11_HEAD_DIM * 4, _V11_HEAD_DIM),
            _v11_nn.GELU(),
            _v11_nn.Dropout(0.15),
        )
        self.weight = _v11_nn.Parameter(_v11_torch.randn(n_targets, _V11_HEAD_DIM) * 0.02)
        self.bias = _v11_nn.Parameter(_v11_torch.zeros(n_targets))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), self.n_slot, self.n_slices, _V11_HEAD_DIM)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        key_padding = mask <= 0
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(
            query, token, token, key_padding_mask=key_padding, need_weights=False
        )[0]
        denom = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdims=True) / denom
        mean = mean.expand(-1, query.shape[1], -1)
        fused = self.fuse(
            _v11_torch.cat(
                [attended, mean, _v11_torch.abs(attended - mean), attended * mean], -1
            )
        )
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


def _v11_load_encoder(device):
    from torchvision.models import resnet50

    ckpt = _v11_find_file("ResNet50.pt")
    sha = _v11_sha(ckpt)
    if sha != _V11_EXPECTED_CKPT:
        _v11_log(f"RadImageNet sha mismatch (continuing): {sha}")

    class Enc(_v11_nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _v11_nn.Sequential(*list(resnet50(weights=None).children())[:-2])

        def forward(self, image):
            return self.backbone(image).mean(dim=(2, 3))

    model = Enc()
    state = _v11_torch.load(str(ckpt), map_location="cpu", weights_only=True)
    if state and all(str(k).startswith("backbone.") for k in state):
        model.load_state_dict(state, strict=True)
    else:
        cleaned = {}
        for k, v in state.items():
            k = str(k).replace("module.", "", 1)
            if k.startswith("backbone."):
                cleaned[k] = v
            elif not k.startswith("fc."):
                cleaned["backbone." + k] = v
        try:
            model.load_state_dict(cleaned, strict=False)
        except Exception:
            msd = model.state_dict()
            filtered = {
                k: v for k, v in cleaned.items() if k in msd and msd[k].shape == v.shape
            }
            model.load_state_dict(filtered, strict=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    _v11_log(f"RadImageNet encoder loaded sha={sha[:12]}")
    return model, sha


@_v11_torch.inference_mode()
def _v11_encode(cache, slot_mask, device):
    n, slots, slices, h, w = cache.shape
    features = _v11_np.zeros((n, slots * slices, _V11_TOKEN_DIM), _v11_np.float16)
    token_mask = _v11_np.repeat(slot_mask[:, :, None], slices, axis=2).reshape(n, -1)
    valid = _v11_np.flatnonzero(token_mask.reshape(-1) > 0)
    flat = cache.reshape(-1, h, w)
    model, sha = _v11_load_encoder(device)
    batch = 64 if device.type == "cuda" else 8
    for b0 in range(0, len(valid), batch):
        ix = valid[b0 : b0 + batch]
        x = _v11_torch.from_numpy(flat[ix]).to(device).float().div_(127.5).sub_(1.0)
        x = x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        with _v11_torch.autocast("cuda", enabled=device.type == "cuda"):
            feat = model(x)
        features.reshape(-1, _V11_TOKEN_DIM)[ix] = feat.float().cpu().numpy().astype(
            _v11_np.float16
        )
        if b0 % (batch * 40) == 0:
            _v11_log(f"encoded {b0}/{len(valid)} slices")
        if time.time() - T0 > TIME_BUDGET - 1.0 * 3600:
            raise TimeoutError("encode approaching time budget")
    del model
    if device.type == "cuda":
        _v11_torch.cuda.empty_cache()
    return features, token_mask.astype(_v11_np.float32), sha


@_v11_torch.inference_mode()
def _v11_predict(model, features, masks, indices, device, batch=64):
    model.eval()
    out = []
    for b0 in range(0, len(indices), batch):
        ix = indices[b0 : b0 + batch]
        x = _v11_torch.from_numpy(features[ix]).to(device)
        m = _v11_torch.from_numpy(masks[ix]).to(device)
        with _v11_torch.autocast("cuda", enabled=device.type == "cuda"):
            out.append(_v11_torch.sigmoid(model(x, m)).float().cpu())
    return _v11_torch.cat(out).numpy()


def _v11_load_v3_fold_tables(train):
    """Return fold_id[n], list of (y,w) per fold k for training with labels_v3_fold{k}."""
    folds = _v11_pd.read_csv(_v11_find_file("folds_v3.csv"), dtype={_V11_UID: str})
    if "fold" not in folds.columns:
        raise RuntimeError("folds_v3.csv missing fold column")
    folds = train[[_V11_UID]].merge(folds, on=_V11_UID, how="left")
    if folds["fold"].isna().any():
        raise RuntimeError("folds_v3 missing some train UIDs")
    fold_id = folds["fold"].astype(int).to_numpy()
    gold = train[_V11_TARGETS].notna().all(axis=1).to_numpy()
    per_fold = []
    for k in range(5):
        lab = _v11_pd.read_csv(
            _v11_find_file(f"labels_v3_fold{k}.csv"), dtype={_V11_UID: str}
        )
        lab = train[[_V11_UID]].merge(lab, on=_V11_UID, how="left")
        y = lab[_V11_TARGETS].apply(_v11_pd.to_numeric, errors="coerce").to_numpy(
            _v11_np.float32
        )
        wcols = [f"{t}__weight" for t in _V11_TARGETS]
        if all(c in lab.columns for c in wcols):
            w = lab[wcols].apply(_v11_pd.to_numeric, errors="coerce").to_numpy(
                _v11_np.float32
            )
        else:
            w = _v11_np.ones_like(y, dtype=_v11_np.float32) * 0.5
        # gold overrides
        y[gold] = train.loc[gold, _V11_TARGETS].to_numpy(_v11_np.float32)
        w[gold] = 3.0
        nan = ~_v11_np.isfinite(y)
        y[nan] = 0.5
        w[nan] = 0.05
        w = _v11_np.clip(w, 0.05, 3.0)
        per_fold.append((y, w))
        _v11_log(f"loaded labels_v3_fold{k}: finite={(~nan).mean():.3f}")
    return fold_id, per_fold, gold


def _v11_load_dino_oof():
    path = _v11_find_file("oof.npz")
    data = _v11_np.load(path, allow_pickle=True)
    ids = [str(x) for x in data["ids"].tolist()]
    if "pred" in data.files:
        pred = data["pred"].astype(_v11_np.float64)
    elif "ours" in data.files:
        pred = data["ours"].astype(_v11_np.float64)
    else:
        raise RuntimeError("oof.npz missing pred")
    return ids, pred


def _v11_load_e9_oof(train):
    """Public e9 Rad OOF for V9 parent reconstruction (3-slot only)."""
    try:
        path = _v11_find_file("v52_oof.csv")
    except FileNotFoundError:
        return None
    df = _v11_pd.read_csv(path, dtype={_V11_UID: str})
    df = train[[_V11_UID]].merge(df, on=_V11_UID, how="left")
    if not all(t in df.columns for t in _V11_TARGETS):
        return None
    arr = df[_V11_TARGETS].apply(_v11_pd.to_numeric, errors="coerce").to_numpy(float)
    if not _v11_np.isfinite(arr).all():
        _v11_log("e9 OOF has non-finite; rejecting")
        return None
    _v11_log(f"loaded e9 OOF from {path}")
    return arr


def _v11_train_fold(features, masks, y, weights, train_idx, val_idx, fold, device):
    from torch.utils.data import DataLoader, Dataset

    class Rows(Dataset):
        def __init__(self, indices):
            self.indices = _v11_np.asarray(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, k):
            i = int(self.indices[k])
            return features[i], masks[i], y[i], weights[i]

    model = _V11Head().to(device)
    opt = _v11_torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=3e-3)
    gen = _v11_torch.Generator().manual_seed(int(SEED) + 300 + fold)
    loader = DataLoader(
        Rows(train_idx),
        batch_size=48,
        shuffle=True,
        generator=gen,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    best, best_auc, stale = None, -1.0, 0
    for epoch in range(30):
        model.train()
        for x, m, target, weight in loader:
            x, m = x.to(device), m.to(device)
            target, weight = target.to(device), weight.to(device)
            with _v11_torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x, m)
                raw = _v11_F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
                loss = (raw * weight).sum() / weight.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            _v11_nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pred = _v11_predict(model, features, masks, val_idx, device)
        score = _v11_macro_auc(y[val_idx], pred)
        _v11_log(f"fold {fold} ep {epoch}: soft-val AUC {score:.5f}")
        if score > best_auc + 2e-4:
            best_auc, stale = score, 0
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 6:
                break
        if time.time() - T0 > TIME_BUDGET - 0.75 * 3600:
            _v11_log("time budget near end during head train")
            break
    return best, best_auc


def run_v11_own_rad():
    audit = {
        "status": "DINO_OR_V9_PRESERVED",
        "promote": False,
        "min_gain": _V11_MIN_GAIN,
        "parent_recipe": "V9 DINO+e9 Rad global alpha=0.20",
        "public_parent_score_reference": 0.906,
        "evidence_boundary": (
            "Nested 58-label OOF is local only. "
            "submission.csv overwritten only when promote=true vs V9 parent."
        ),
    }
    _V11_OUT.mkdir(parents=True, exist_ok=True)
    if not _V11_PRIMARY.is_file():
        raise FileNotFoundError("submission.csv missing before V11 arm")
    _v11_shutil.copy2(_V11_PRIMARY, _V11_DINO)
    audit["dino_sha256"] = _v11_sha(_V11_DINO)

    device = _v11_torch.device("cuda:0" if _v11_torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("V11 requires CUDA")
    remaining = TIME_BUDGET - (time.time() - T0)
    audit["remaining_seconds_at_start"] = remaining
    if remaining < 3.0 * 3600:
        raise TimeoutError(f"only {remaining/60:.1f} min left; need ~3h+ for train encode")

    global SLOTS, N_SLOT, IMG, CACHE_IMG, CACHE_SLICES, CROP_MM, SLICE_BAND, GROUP, N_GROUP
    _saved = {
        "SLOTS": SLOTS,
        "N_SLOT": N_SLOT,
        "IMG": IMG,
        "CACHE_IMG": CACHE_IMG,
        "CACHE_SLICES": CACHE_SLICES,
        "CROP_MM": CROP_MM,
        "SLICE_BAND": SLICE_BAND,
        "GROUP": GROUP,
        "N_GROUP": N_GROUP,
    }
    try:
        SLOTS = list(_V11_RAD_SLOTS)
        N_SLOT = 3
        IMG = CACHE_IMG = 224
        CACHE_SLICES = 8
        GROUP = 1
        N_GROUP = 8
        CROP_MM = 10_000.0
        SLICE_BAND = (0.12, 0.88)

        train = _v11_pd.read_csv(ROOT / "train.csv", dtype={_V11_UID: str})
        test = _v11_pd.read_csv(ROOT / "test.csv", dtype={_V11_UID: str})
        train_series = _v11_pd.read_csv(
            ROOT / "train_series.csv", dtype={_V11_UID: str, "SeriesInstanceUID": str}
        )
        test_series = _v11_pd.read_csv(
            ROOT / "test_series.csv", dtype={_V11_UID: str, "SeriesInstanceUID": str}
        )
        both = _v11_pd.concat([train_series, test_series], ignore_index=True)
        plane_map = dict(zip(both["SeriesInstanceUID"], both["Anatomical_Plane"]))

        fold_id, per_fold_labels, gold = _v11_load_v3_fold_tables(train)
        audit["n_gold"] = int(gold.sum())
        audit["fold_counts"] = {str(k): int((fold_id == k).sum()) for k in range(5)}

        # Train encode
        _v11_log("building train FS cache @224")
        htr = annotate(walk("train_series"))
        slots_tr = pick_slots(htr, plane_map)
        st_tr, Ctr, Mtr = build_cache(
            slots_tr, plane_map, lat_of(htr, "v11-train "), "v11-train"
        )
        tr_by = {str(u): i for i, u in enumerate(st_tr)}
        tr_order = _v11_np.array(
            [tr_by[str(u)] for u in train[_V11_UID]], dtype=_v11_np.int64
        )
        Ctr, Mtr = Ctr[tr_order], Mtr[tr_order]
        features, token_mask, enc_sha = _v11_encode(Ctr, Mtr, device)
        del Ctr, Mtr, htr
        _v11_gc.collect()
        audit["encoder_sha256"] = enc_sha
        audit["train_tokens"] = int(token_mask.sum())

        # Train 5 folds with fold-cross labels
        own_oof = _v11_np.zeros((len(train), len(_V11_TARGETS)), dtype=_v11_np.float32)
        fold_states = []
        fold_scores = {}
        for k in range(5):
            y_k, w_k = per_fold_labels[k]
            tr = _v11_np.where(fold_id != k)[0]
            va = _v11_np.where(fold_id == k)[0]
            state, score = _v11_train_fold(
                features, token_mask, y_k, w_k, tr, va, k, device
            )
            if state is None:
                raise RuntimeError(f"fold {k} produced no checkpoint")
            head = _V11Head().to(device)
            head.load_state_dict(state, strict=True)
            own_oof[va] = _v11_predict(head, features, token_mask, va, device)
            fold_states.append(state)
            fold_scores[str(k)] = float(score)
            del head
            _v11_torch.cuda.empty_cache()
        del features, token_mask
        _v11_gc.collect()
        audit["fold_soft_val_auc"] = fold_scores

        # Soft OOF vs binary soft labels from fold0 labels mean of all? use labels_v3.csv
        lab_global = _v11_pd.read_csv(
            _v11_find_file("labels_v3.csv"), dtype={_V11_UID: str}
        )
        lab_global = train[[_V11_UID]].merge(lab_global, on=_V11_UID, how="left")
        y_soft = lab_global[_V11_TARGETS].apply(_v11_pd.to_numeric, errors="coerce").to_numpy(
            float
        )
        y_soft[~_v11_np.isfinite(y_soft)] = 0.5
        # gold hard for gold rows
        y_soft[gold] = train.loc[gold, _V11_TARGETS].to_numpy(float)
        soft_own = _v11_macro_auc(y_soft, own_oof)
        audit["own_soft_macro_auc"] = soft_own
        soft_ok = soft_own >= _V11_SOFT_FLOOR
        audit["soft_floor"] = _V11_SOFT_FLOOR
        audit["soft_gate_ok"] = bool(soft_ok)
        _v11_log(f"own soft OOF macro {soft_own:.5f} floor={_V11_SOFT_FLOOR} ok={soft_ok}")

        # Save OOF + heads
        oof_frame = _v11_pd.DataFrame(own_oof, columns=_V11_TARGETS)
        oof_frame.insert(0, _V11_UID, train[_V11_UID])
        oof_frame["fold"] = fold_id
        oof_frame["is_gold"] = gold.astype(_v11_np.uint8)
        oof_frame.to_csv(_V11_OUT / "v11_own_oof.csv", index=False)
        _v11_torch.save(
            {"folds": [{"fold": i, "state_dict": s} for i, s in enumerate(fold_states)]},
            _V11_OUT / "v11_own_heads.pt",
        )

        # DINO + e9 for V9 parent
        dino_ids, dino_pred = _v11_load_dino_oof()
        dino_index = {u: i for i, u in enumerate(dino_ids)}
        e9_oof = _v11_load_e9_oof(train)
        if e9_oof is None:
            raise RuntimeError("e9 v52_oof.csv required to reconstruct V9 parent")

        # Align gold
        gold_uids = train.loc[gold, _V11_UID].astype(str).tolist()
        gold_y_all = train.loc[gold, _V11_TARGETS].to_numpy(float)
        train_index = {str(u): i for i, u in enumerate(train[_V11_UID])}
        dino_rows, own_rows, e9_rows, keep_y, gfolds = [], [], [], [], []
        for uid, gy in zip(gold_uids, gold_y_all):
            if uid not in dino_index or uid not in train_index:
                continue
            dino_rows.append(dino_index[uid])
            own_rows.append(train_index[uid])
            e9_rows.append(train_index[uid])
            keep_y.append(gy)
            gfolds.append(int(fold_id[train_index[uid]]))
        if len(dino_rows) < 50:
            raise RuntimeError(f"too few gold rows aligned: {len(dino_rows)}")

        gold_y = _v11_np.asarray(keep_y, float)
        dino_rank = _v11_rank(dino_pred[_v11_np.array(dino_rows)])
        e9_rank = _v11_rank(e9_oof[_v11_np.array(e9_rows)])
        own_rank = _v11_rank(own_oof[_v11_np.array(own_rows)])
        gfolds = _v11_np.asarray(gfolds, int)

        v9_parent = _v11_blend_ranks(dino_rank, e9_rank, _V11_BASE_ALPHA)
        v9_score = _v11_macro_auc(gold_y, v9_parent)
        dino_score = _v11_macro_auc(gold_y, dino_rank)
        e9_score = _v11_macro_auc(gold_y, e9_rank)
        own_score = _v11_macro_auc(gold_y, own_rank)
        audit["dino_gold_rank_macro"] = dino_score
        audit["e9_gold_rank_macro"] = e9_score
        audit["own_gold_rank_macro"] = own_score
        audit["v9_parent_gold_macro"] = v9_score
        _v11_log(
            f"gold DINO={dino_score:.5f} e9={e9_score:.5f} own={own_score:.5f} "
            f"V9parent={v9_score:.5f}"
        )

        # Nested alpha selection for own vs DINO, then compare to V9 parent
        nested = _v11_np.zeros_like(dino_rank)
        choices = []
        fold_gains = []
        for outer in sorted(set(gfolds.tolist())):
            tr = gfolds != outer
            va = ~tr
            if int(va.sum()) == 0 or int(tr.sum()) < 8:
                continue
            scored = []
            for alpha in _V11_ALPHA_GRID:
                blend = _v11_blend_ranks(dino_rank[tr], own_rank[tr], alpha)
                pen = 0.005 * float(alpha)
                scored.append(_v11_macro_auc(gold_y[tr], blend) - pen)
            best = int(_v11_np.argmax(scored))
            alpha = float(_V11_ALPHA_GRID[best])
            choices.append(alpha)
            nested[va] = _v11_blend_ranks(dino_rank[va], own_rank[va], alpha)
            parent_va = _v11_macro_auc(gold_y[va], v9_parent[va])
            cand_va = _v11_macro_auc(gold_y[va], nested[va])
            fold_gains.append(float(cand_va - parent_va))

        nested_score = _v11_macro_auc(gold_y, nested)
        final_alpha = float(_v11_np.median(_v11_np.asarray(choices))) if choices else 0.0
        final_oof = _v11_blend_ranks(dino_rank, own_rank, final_alpha)
        final_score = _v11_macro_auc(gold_y, final_oof)
        positive = int(sum(g > 0 for g in fold_gains))
        supported = bool(
            soft_ok
            and final_alpha > 0
            and nested_score >= v9_score + _V11_MIN_GAIN
            and final_score >= v9_score + _V11_MIN_GAIN
            and positive >= max(2, len(fold_gains) // 2)
        )
        audit["alpha_choices"] = choices
        audit["final_alpha"] = final_alpha
        audit["nested_gold_macro"] = nested_score
        audit["final_gold_macro"] = final_score
        audit["fold_gains_vs_v9"] = fold_gains
        audit["positive_outer"] = positive
        audit["selection_supported"] = supported
        _v11_log(
            f"gate V9={v9_score:.5f} nested={nested_score:.5f} final={final_score:.5f} "
            f"alpha={final_alpha} pos={positive}/{len(fold_gains)} supported={supported}"
        )

        # Build test predictions for V9 parent always; own if supported
        _v11_log("building test FS cache @224")
        hte = annotate(walk("test_series"))
        slots_te = pick_slots(hte, plane_map)
        st_te, Cte, Mte = build_cache(
            slots_te, plane_map, lat_of(hte, "v11-test "), "v11-test"
        )
        te_by = {str(u): i for i, u in enumerate(st_te)}
        te_order = _v11_np.array(
            [te_by[str(u)] for u in test[_V11_UID]], dtype=_v11_np.int64
        )
        Cte, Mte = Cte[te_order], Mte[te_order]
        test_features, test_mask, _ = _v11_encode(Cte, Mte, device)
        del Cte, Mte, hte
        _v11_gc.collect()

        # Need e9 test preds for V9 parent file: load e9 heads if present
        e9_states = None
        try:
            head_path = _v11_find_file("v52_radimagenet_heads.pt")
            payload = _v11_torch.load(str(head_path), map_location="cpu", weights_only=False)
            folds = payload.get("folds")
            if isinstance(folds, list) and len(folds) == 5:
                sts = [rec.get("state_dict") for rec in folds]
                if all(s is not None for s in sts) and sts[0].get("plane") is not None:
                    if int(sts[0]["plane"].shape[0]) == 3:
                        e9_states = sts
                        _v11_log(f"e9 heads for test parent from {head_path}")
        except Exception as exc:
            _v11_log(f"e9 heads unavailable for test parent: {exc}")

        dino_test = _v11_pd.read_csv(_V11_DINO, dtype={_V11_UID: str})
        _v11_validate_sub(dino_test, test[_V11_UID])
        dino_test_rank = _v11_rank(dino_test[_V11_TARGETS].to_numpy(float))

        if e9_states is not None:
            e9_preds = []
            all_te = _v11_np.arange(len(test), dtype=_v11_np.int64)
            for st in e9_states:
                head = _V11Head().to(device)
                head.load_state_dict(st, strict=True)
                e9_preds.append(
                    _v11_predict(head, test_features, test_mask, all_te, device)
                )
                del head
                _v11_torch.cuda.empty_cache()
            e9_test = _v11_np.mean(_v11_np.stack(e9_preds), axis=0)
            e9_test_rank = _v11_rank(e9_test)
            v9_frame = dino_test.copy()
            v9_frame[_V11_TARGETS] = _v11_blend_ranks(
                dino_test_rank, e9_test_rank, _V11_BASE_ALPHA
            )
            _v11_validate_sub(v9_frame, test[_V11_UID])
            v9_frame.to_csv(_V11_V9, index=False)
            v9_frame.to_csv(_V11_OUT / "submission_v9_parent.csv", index=False)
            audit["v9_parent_sha256"] = _v11_sha(_V11_V9)
            _v11_shutil.copy2(_V11_V9, _V11_PRIMARY)
        else:
            _v11_shutil.copy2(_V11_DINO, _V11_PRIMARY)
            audit["v9_parent_note"] = "e9 heads missing; primary=DINO until promote"

        if not soft_ok:
            audit["status"] = "SOFT_REJECT_KEEP_PARENT"
            return audit
        if not supported:
            audit["status"] = "OOF_REJECTED_KEEP_V9_PARENT"
            return audit

        # Own test infer + blend with DINO
        own_preds = []
        all_te = _v11_np.arange(len(test), dtype=_v11_np.int64)
        for st in fold_states:
            head = _V11Head().to(device)
            head.load_state_dict(st, strict=True)
            own_preds.append(
                _v11_predict(head, test_features, test_mask, all_te, device)
            )
            del head
            _v11_torch.cuda.empty_cache()
        own_test = _v11_np.mean(_v11_np.stack(own_preds), axis=0)
        own_only = _v11_pd.DataFrame(own_test, columns=_V11_TARGETS)
        own_only.insert(0, _V11_UID, test[_V11_UID])
        own_only.to_csv(_V11_OUT / "submission_own_rad_only.csv", index=False)
        own_test_rank = _v11_rank(own_test)
        blended = dino_test.copy()
        blended[_V11_TARGETS] = _v11_blend_ranks(
            dino_test_rank, own_test_rank, final_alpha
        )
        _v11_validate_sub(blended, test[_V11_UID])
        out_path = _V11_OUT / f"submission_v11_alpha_{int(round(1000*final_alpha)):03d}.csv"
        blended.to_csv(out_path, index=False)
        _v11_shutil.copy2(out_path, _V11_PRIMARY)
        audit["status"] = "CANDIDATE_SELECTED"
        audit["promote"] = True
        audit["selected_sha256"] = _v11_sha(_V11_PRIMARY)
        audit["selected_path"] = str(out_path)
        _v11_log(f"PROMOTED V11 own Rad alpha={final_alpha}")
        return audit
    finally:
        for k, v in _saved.items():
            globals()[k] = v


try:
    main()
except LabelSourceError:
    traceback.print_exc()
    raise
except Exception:
    traceback.print_exc()
    t = pd.read_csv(find_root() / "test.csv")
    for c in TARGETS:
        t[c] = 0.5
    t.to_csv("submission.csv", index=False)
    print("wrote fallback submission.csv")

_v11_receipt = {"status": "RAD_NOT_RUN", "promote": False}
try:
    _v11_receipt = run_v11_own_rad()
except Exception as _v11_err:
    traceback.print_exc()
    _v11_receipt = {
        "status": "RAD_ERROR_RESTORE_BEST",
        "promote": False,
        "error": f"{type(_v11_err).__name__}: {_v11_err}",
        "traceback": traceback.format_exc()[-4000:],
    }
    try:
        if _V11_V9.is_file():
            _v11_shutil.copy2(_V11_V9, _V11_PRIMARY)
        elif _V11_DINO.is_file():
            _v11_shutil.copy2(_V11_DINO, _V11_PRIMARY)
    except Exception:
        pass

_V11_AUDIT.write_text(
    _v11_json.dumps(_v11_receipt, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
_keys = (
    "status",
    "promote",
    "final_alpha",
    "own_soft_macro_auc",
    "nested_gold_macro",
    "final_gold_macro",
    "v9_parent_gold_macro",
    "error",
)
print(
    "V11 AUDIT",
    _v11_json.dumps({k: _v11_receipt.get(k) for k in _keys}, sort_keys=True, default=str),
)
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
    nb = json.loads(V1_NB.read_text(encoding="utf-8"))
    cells = list(nb["cells"])
    drop_idx = None
    for i in range(len(cells) - 1, -1, -1):
        src = "".join(cells[i].get("source", []))
        if cells[i].get("cell_type") == "code" and "main()" in src and "try:" in src:
            drop_idx = i
            break
    if drop_idx is None:
        raise RuntimeError("could not find V1 final main() cell")
    cells = cells[:drop_idx] + [code_cell(RAD_ARM)]
    out_nb = {
        "cells": cells,
        "metadata": nb.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(out_nb, indent=1) + "\n", encoding="utf-8")
    meta = {
        "id": "danielsolo1770/rsna-knee-v11-own-rad-gated",
        "title": "rsna-knee-v11-own-rad-gated",
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
            "danielsolo1770/rsna-knee-v3-labels-folds",
        ],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
        "machine_shape": "NvidiaTeslaT4",
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_NB}")
    print(f"wrote {OUT_META}")
    print(f"cells: {len(cells)}")


if __name__ == "__main__":
    main()
