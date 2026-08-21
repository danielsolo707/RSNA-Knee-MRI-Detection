#!/usr/bin/env python3
"""Build V10: V9 parent (DINO+Rad@0.20) + dual-gated ACL/Syn higher-α and e11 heads.

Parent fallback is always the verified V9 recipe (public 0.906).
Promotion only if nested gold OOF beats the V9 parent by >= 0.001.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_NB = ROOT / "kaggle" / "submission_v1" / "rsna-knee-v1-public-0899.ipynb"
OUT_DIR = ROOT / "kaggle" / "submission_v10_rad_cross"
OUT_NB = OUT_DIR / "rsna-knee-v10-rad-cross-gated.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

RAD_ARM = r'''
# === V10: V9 parent + dual-gated ACL/Syn Rad rank-crossing (+ e11 heads) =======
# 1) DINO package → submission_dino_v1.csv
# 2) RadImageNet (prefer e11 heads, else e9, else train) → V9-style α=0.20 parent
# 3) Target-wise higher α on ACL + Synovitis only, nested-gated vs V9 parent
# Any failure restores the best available parent (V9 ranks if built, else DINO).

import gc as _v10_gc
import hashlib as _v10_hashlib
import json as _v10_json
import shutil as _v10_shutil
from pathlib import Path as _V10Path

import numpy as _v10_np
import pandas as _v10_pd
import torch as _v10_torch
import torch.nn as _v10_nn
import torch.nn.functional as _v10_F

_V10_TARGETS = list(TARGETS)
_V10_UID = "StudyInstanceUID"
_V10_WORK = _V10Path("/kaggle/working")
_V10_PRIMARY = _V10_WORK / "submission.csv"
_V10_DINO = _V10_WORK / "submission_dino_v1.csv"
_V10_V9 = _V10_WORK / "submission_v9_parent.csv"
_V10_AUDIT = _V10_WORK / "v10_rad_audit.json"
_V10_OUT = _V10_WORK / "rsna_rad_v10"
_V10_TOKEN_DIM = 2048
_V10_HEAD_DIM = 512
_V10_RAD_SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
]
_V10_BASE_ALPHA = 0.20
_V10_CROSS_TARGETS = ("ACL", "Synovitis")
_V10_CROSS_GRID = _v10_np.array([0.20, 0.30, 0.40, 0.505], dtype=float)
_V10_GLOBAL_GRID = _v10_np.array([0.0, 0.05, 0.10, 0.15, 0.20], dtype=float)
_V10_EXPECTED_CKPT = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"


def _v10_log(msg):
    try:
        log(f"[V10] {msg}")
    except Exception:
        print(f"[V10] {msg}", flush=True)


def _v10_sha(path):
    h = _v10_hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _v10_find_file(name):
    root = _V10Path("/kaggle/input")
    if not root.is_dir():
        raise FileNotFoundError(name)
    for p in root.rglob(name):
        parts = set(p.parts)
        if "train_series" in parts or "test_series" in parts:
            continue
        if p.is_file():
            return p
    raise FileNotFoundError(name)


def _v10_rank(values):
    frame = _v10_pd.DataFrame(_v10_np.asarray(values, dtype=_v10_np.float64))
    return frame.rank(method="average", pct=True).to_numpy(_v10_np.float64)


def _v10_auc_1d(y, s):
    y = _v10_np.asarray(y, dtype=float)
    s = _v10_np.asarray(s, dtype=float)
    hard = (y >= 0.5).astype(float)
    pos = hard == 1
    neg = hard == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = _v10_np.argsort(s, kind="mergesort")
    ranks = _v10_np.empty(len(s), dtype=float)
    ranks[order] = _v10_np.arange(1, len(s) + 1, dtype=float)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _v10_macro_auc(y, pred):
    vals = []
    for j in range(y.shape[1]):
        a = _v10_auc_1d(y[:, j], pred[:, j])
        if a == a:
            vals.append(a)
    return float(_v10_np.mean(vals)) if vals else float("nan")


def _v10_validate_sub(frame, expected_ids):
    cols = [_V10_UID] + _V10_TARGETS
    if frame.columns.tolist() != cols:
        raise RuntimeError(f"schema drift: {frame.columns.tolist()}")
    ids = frame[_V10_UID].astype(str).tolist()
    exp = list(map(str, expected_ids))
    if ids != exp or len(ids) != len(set(ids)):
        raise RuntimeError("study identity/order drift")
    vals = frame[_V10_TARGETS].to_numpy(float)
    if not _v10_np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
        raise RuntimeError("invalid probability values")


class _V10FoundationQueryHead(_v10_nn.Module):
    def __init__(self, n_slot=3, n_slices=8, n_targets=12):
        super().__init__()
        self.n_slot = n_slot
        self.n_slices = n_slices
        self.project = _v10_nn.Sequential(
            _v10_nn.LayerNorm(_V10_TOKEN_DIM),
            _v10_nn.Linear(_V10_TOKEN_DIM, _V10_HEAD_DIM),
            _v10_nn.GELU(),
        )
        self.plane = _v10_nn.Parameter(_v10_torch.randn(n_slot, _V10_HEAD_DIM) * 0.01)
        self.position = _v10_nn.Parameter(_v10_torch.randn(n_slices, _V10_HEAD_DIM) * 0.01)
        self.query = _v10_nn.Parameter(_v10_torch.randn(n_targets, _V10_HEAD_DIM) * 0.02)
        self.attn = _v10_nn.MultiheadAttention(_V10_HEAD_DIM, 8, dropout=0.10, batch_first=True)
        self.fuse = _v10_nn.Sequential(
            _v10_nn.LayerNorm(_V10_HEAD_DIM * 4),
            _v10_nn.Linear(_V10_HEAD_DIM * 4, _V10_HEAD_DIM),
            _v10_nn.GELU(),
            _v10_nn.Dropout(0.15),
        )
        self.weight = _v10_nn.Parameter(_v10_torch.randn(n_targets, _V10_HEAD_DIM) * 0.02)
        self.bias = _v10_nn.Parameter(_v10_torch.zeros(n_targets))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), self.n_slot, self.n_slices, _V10_HEAD_DIM)
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
            _v10_torch.cat(
                [attended, mean, _v10_torch.abs(attended - mean), attended * mean], -1
            )
        )
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


def _v10_load_encoder(device):
    from torchvision.models import resnet50

    ckpt = _v10_find_file("ResNet50.pt")
    sha = _v10_sha(ckpt)
    if sha != _V10_EXPECTED_CKPT:
        _v10_log(f"RadImageNet sha mismatch (continuing): {sha}")

    class Enc(_v10_nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _v10_nn.Sequential(*list(resnet50(weights=None).children())[:-2])

        def forward(self, image):
            return self.backbone(image).mean(dim=(2, 3))

    model = Enc()
    state = _v10_torch.load(str(ckpt), map_location="cpu", weights_only=True)
    if state and all(str(k).startswith("backbone.") for k in state):
        model.load_state_dict(state, strict=True)
    else:
        cleaned = {}
        for k, v in state.items():
            k = str(k).replace("module.", "", 1)
            if k.startswith("backbone."):
                cleaned[k] = v
            elif k.startswith("fc."):
                continue
            else:
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
    _v10_log(f"RadImageNet encoder loaded sha={sha[:12]}")
    return model, sha


@_v10_torch.inference_mode()
def _v10_encode(cache, slot_mask, device):
    n, slots, slices, h, w = cache.shape
    features = _v10_np.zeros((n, slots * slices, _V10_TOKEN_DIM), _v10_np.float16)
    token_mask = _v10_np.repeat(slot_mask[:, :, None], slices, axis=2).reshape(n, -1)
    valid = _v10_np.flatnonzero(token_mask.reshape(-1) > 0)
    flat = cache.reshape(-1, h, w)
    model, sha = _v10_load_encoder(device)
    batch = 64 if device.type == "cuda" else 8
    for b0 in range(0, len(valid), batch):
        ix = valid[b0 : b0 + batch]
        x = _v10_torch.from_numpy(flat[ix]).to(device).float().div_(127.5).sub_(1.0)
        x = x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        with _v10_torch.autocast("cuda", enabled=device.type == "cuda"):
            feat = model(x)
        features.reshape(-1, _V10_TOKEN_DIM)[ix] = feat.float().cpu().numpy().astype(
            _v10_np.float16
        )
        if b0 % (batch * 50) == 0:
            _v10_log(f"encoded {b0}/{len(valid)} slices")
    del model
    if device.type == "cuda":
        _v10_torch.cuda.empty_cache()
    return features, token_mask.astype(_v10_np.float32), sha


@_v10_torch.inference_mode()
def _v10_predict(model, features, masks, indices, device, batch=64):
    model.eval()
    out = []
    for b0 in range(0, len(indices), batch):
        ix = indices[b0 : b0 + batch]
        x = _v10_torch.from_numpy(features[ix]).to(device)
        m = _v10_torch.from_numpy(masks[ix]).to(device)
        with _v10_torch.autocast("cuda", enabled=device.type == "cuda"):
            out.append(_v10_torch.sigmoid(model(x, m)).float().cpu())
    return _v10_torch.cat(out).numpy()


def _v10_make_targets(train):
    uid = _V10_UID
    sources = []
    for name in ("report_labels_v2.csv", "llm_labels_v2.csv", "labels_llm_gpt56sol.csv"):
        try:
            sources.append(_v10_pd.read_csv(_v10_find_file(name)))
            _v10_log(f"label source ok: {name}")
        except FileNotFoundError:
            _v10_log(f"label source missing: {name}")
    if not sources:
        raise RuntimeError("no report-label CSV attached")
    cube = []
    for frame in sources:
        frame = frame.copy()
        if frame[uid].duplicated().any():
            frame = frame.drop_duplicates(uid, keep="first")
        if not all(t in frame.columns for t in _V10_TARGETS):
            raise RuntimeError(f"label CSV missing targets: {frame.columns.tolist()[:20]}")
        aligned = train[[uid]].merge(frame[[uid] + _V10_TARGETS], on=uid, how="left")
        cube.append(aligned[_V10_TARGETS].to_numpy(float))
    cube = _v10_np.stack(cube)
    y = _v10_np.nanmean(cube, axis=0).astype(_v10_np.float32)
    with _v10_np.errstate(all="ignore"):
        disagreement = _v10_np.nanmean(_v10_np.abs(cube - y[None]), axis=0)
    agreement = _v10_np.clip(1.0 - 2.0 * disagreement, 0, 1)
    certainty = _v10_np.clip(2.0 * _v10_np.abs(y - 0.5), 0, 1)
    w = (0.15 + 0.85 * (0.65 * agreement + 0.35 * certainty)).astype(_v10_np.float32)
    gold = train[_V10_TARGETS].notna().all(axis=1).to_numpy()
    y[gold] = train.loc[gold, _V10_TARGETS].to_numpy(_v10_np.float32)
    w[gold] = 3.0
    nan = ~_v10_np.isfinite(y)
    y[nan] = 0.5
    w[nan] = 0.05
    return y, w, gold


def _v10_report_groups(train):
    report = (
        train.Report.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return _v10_np.array(
        [_v10_hashlib.sha256(x.encode()).hexdigest()[:24] for x in report]
    )


def _v10_train_fold(features, masks, y, weights, train_idx, val_idx, fold, device):
    from torch.utils.data import DataLoader, Dataset

    class Rows(Dataset):
        def __init__(self, indices):
            self.indices = _v10_np.asarray(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, k):
            i = int(self.indices[k])
            return features[i], masks[i], y[i], weights[i]

    model = _V10FoundationQueryHead().to(device)
    opt = _v10_torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=3e-3)
    gen = _v10_torch.Generator().manual_seed(int(SEED) + 200 + fold)
    loader = DataLoader(
        Rows(train_idx),
        batch_size=48,
        shuffle=True,
        generator=gen,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    best, best_auc, stale = None, -1.0, 0
    for epoch in range(24):
        model.train()
        for x, m, target, weight in loader:
            x, m = x.to(device), m.to(device)
            target, weight = target.to(device), weight.to(device)
            with _v10_torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x, m)
                raw = _v10_F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
                loss = (raw * weight).sum() / weight.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            _v10_nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pred = _v10_predict(model, features, masks, val_idx, device)
        score = _v10_macro_auc(y[val_idx], pred)
        _v10_log(f"fold {fold} ep {epoch}: weak-val AUC {score:.5f}")
        if score > best_auc + 2e-4:
            best_auc, stale = score, 0
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 5:
                break
        if time.time() - T0 > TIME_BUDGET - 0.5 * 3600:
            break
    return best, best_auc


def _v10_try_load_public_heads(expected_n_slot=3):
    """Prefer e9 (3-slot V9 recipe), then e11 only if plane dim matches.

    e11 heads historically use plane shape [4, 512] and are incompatible with the
    3-slot FS cache used for the verified 0.906 V9 parent. Skip shape mismatches.
    """
    candidates = [
        ("v52_radimagenet_heads.pt", "v52_oof.csv", "e9"),
        ("v52_e11_heads.pt", "v52_e11_oof.csv", "e11"),
    ]
    for head_name, oof_name, tag in candidates:
        try:
            path = _v10_find_file(head_name)
        except FileNotFoundError:
            continue
        payload = _v10_torch.load(str(path), map_location="cpu", weights_only=False)
        folds = payload.get("folds")
        if not isinstance(folds, list) or len(folds) != 5:
            continue
        states = []
        for rec in folds:
            st = rec.get("state_dict") if isinstance(rec, dict) else None
            if st is None:
                continue
            states.append(st)
        if len(states) != 5:
            continue
        # Reject heads whose slot embedding does not match our cache geometry.
        plane = states[0].get("plane")
        if plane is None:
            _v10_log(f"skip {tag}: missing plane param")
            continue
        n_slot = int(plane.shape[0])
        if n_slot != int(expected_n_slot):
            _v10_log(
                f"skip {tag}: plane slots={n_slot} != expected {expected_n_slot}"
            )
            continue
        try:
            oof_path = _v10_find_file(oof_name)
        except FileNotFoundError:
            oof_path = None
            _v10_log(f"skip {tag}: OOF CSV missing")
            continue
        _v10_log(f"public heads loaded ({tag}, slots={n_slot}): {path}")
        return states, oof_path, str(path), tag
    return None, None, None, None


def _v10_load_dino_oof():
    try:
        path = _v10_find_file("oof.npz")
    except FileNotFoundError:
        return None, None
    data = _v10_np.load(path, allow_pickle=True)
    ids = [str(x) for x in data["ids"].tolist()]
    if "pred" in data.files:
        pred = data["pred"].astype(_v10_np.float64)
    elif "ours" in data.files:
        pred = data["ours"].astype(_v10_np.float64)
    else:
        return None, None
    return ids, pred


def _v10_blend_ranks(base_rank, rad_rank, alphas):
    """alphas: array shape (12,) per-target Rad weight."""
    a = _v10_np.asarray(alphas, dtype=_v10_np.float64).reshape(1, -1)
    return (1.0 - a) * base_rank + a * rad_rank


def run_v10_rad_cross():
    audit = {
        "status": "DINO_OR_V9_PRESERVED",
        "promote": False,
        "evidence_boundary": (
            "Nested 58-label OOF is local diagnostic only. "
            "submission.csv overwritten only when promote=true vs V9 parent."
        ),
        "parent_recipe": "V9 DINO+Rad global alpha=0.20",
        "public_parent_score_reference": 0.906,
    }
    _V10_OUT.mkdir(parents=True, exist_ok=True)
    if not _V10_PRIMARY.is_file():
        raise FileNotFoundError("parent submission.csv missing before Rad arm")
    _v10_shutil.copy2(_V10_PRIMARY, _V10_DINO)
    audit["dino_sha256"] = _v10_sha(_V10_DINO)

    device = _v10_torch.device("cuda:0" if _v10_torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("RadImageNet arm requires CUDA")
    remaining = TIME_BUDGET - (time.time() - T0)
    audit["remaining_seconds_at_start"] = remaining
    if remaining < 1.5 * 3600:
        raise TimeoutError(f"only {remaining/60:.1f} min left for Rad arm")

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
        SLOTS = list(_V10_RAD_SLOTS)
        N_SLOT = 3
        IMG = CACHE_IMG = 224
        CACHE_SLICES = 8
        GROUP = 1
        N_GROUP = 8
        CROP_MM = 10_000.0
        SLICE_BAND = (0.12, 0.88)

        train = _v10_pd.read_csv(ROOT / "train.csv", dtype={_V10_UID: str})
        test = _v10_pd.read_csv(ROOT / "test.csv", dtype={_V10_UID: str})
        train_series = _v10_pd.read_csv(
            ROOT / "train_series.csv", dtype={_V10_UID: str, "SeriesInstanceUID": str}
        )
        test_series = _v10_pd.read_csv(
            ROOT / "test_series.csv", dtype={_V10_UID: str, "SeriesInstanceUID": str}
        )
        both = _v10_pd.concat([train_series, test_series], ignore_index=True)
        plane_map = dict(zip(both["SeriesInstanceUID"], both["Anatomical_Plane"]))

        public_states, public_oof_path, public_heads_path, head_tag = (
            _v10_try_load_public_heads(expected_n_slot=N_SLOT)
        )
        audit["public_heads"] = public_heads_path
        audit["head_tag"] = head_tag

        _v10_log("building test FS cache @224")
        hte = annotate(walk("test_series"))
        slots_te = pick_slots(hte, plane_map)
        st_te, Cte, Mte = build_cache(
            slots_te, plane_map, lat_of(hte, "v10-test "), "v10-test"
        )
        te_by = {str(u): i for i, u in enumerate(st_te)}
        te_order = _v10_np.array(
            [te_by[str(u)] for u in test[_V10_UID]], dtype=_v10_np.int64
        )
        Cte, Mte = Cte[te_order], Mte[te_order]
        st_te = [str(u) for u in test[_V10_UID]]
        test_features, test_mask, enc_sha = _v10_encode(Cte, Mte, device)
        del Cte, Mte, hte
        _v10_gc.collect()
        audit["encoder_sha256"] = enc_sha

        fold_states = None
        oof = None
        fold_id = None
        y = w = gold = None

        if public_states is not None and public_oof_path is not None:
            fold_states = public_states
            oof_df = _v10_pd.read_csv(public_oof_path, dtype={_V10_UID: str})
            oof_df = train[[_V10_UID]].merge(oof_df, on=_V10_UID, how="left")
            oof = oof_df[_V10_TARGETS].to_numpy(float)
            if "fold" in oof_df.columns:
                fold_id = oof_df["fold"].to_numpy(int)
            y, w, gold = _v10_make_targets(train)
            audit["head_source"] = f"public_{head_tag}"
        else:
            audit["head_source"] = "train_in_kernel"
            _v10_log("building train FS cache @224")
            htr = annotate(walk("train_series"))
            slots_tr = pick_slots(htr, plane_map)
            st_tr, Ctr, Mtr = build_cache(
                slots_tr, plane_map, lat_of(htr, "v10-train "), "v10-train"
            )
            tr_by = {str(u): i for i, u in enumerate(st_tr)}
            tr_order = _v10_np.array(
                [tr_by[str(u)] for u in train[_V10_UID]], dtype=_v10_np.int64
            )
            Ctr, Mtr = Ctr[tr_order], Mtr[tr_order]
            features, token_mask, _ = _v10_encode(Ctr, Mtr, device)
            del Ctr, Mtr, htr
            _v10_gc.collect()
            y, w, gold = _v10_make_targets(train)
            groups = _v10_report_groups(train)
            from sklearn.model_selection import GroupKFold

            splits = list(GroupKFold(5).split(features, groups=groups))
            fold_id = _v10_np.full(len(train), -1, dtype=_v10_np.int8)
            oof = _v10_np.zeros_like(y, dtype=_v10_np.float32)
            fold_states = []
            for fold, (tr, va) in enumerate(splits):
                fold_id[va] = fold
                state, score = _v10_train_fold(
                    features, token_mask, y, w, tr, va, fold, device
                )
                if state is None:
                    raise RuntimeError(f"fold {fold} produced no checkpoint")
                head = _V10FoundationQueryHead().to(device)
                head.load_state_dict(state, strict=True)
                oof[va] = _v10_predict(head, features, token_mask, va, device)
                fold_states.append(state)
                del head
                _v10_torch.cuda.empty_cache()
                audit.setdefault("fold_weak_auc", {})[str(fold)] = float(score)
            del features, token_mask
            _v10_gc.collect()

        if oof is None or fold_states is None:
            raise RuntimeError("Rad OOF / heads unavailable")
        if not _v10_np.isfinite(oof).all():
            raise RuntimeError("non-finite Rad OOF")

        weak_auc = _v10_macro_auc(y, oof)
        gold_auc = _v10_macro_auc(y[gold], oof[gold])
        audit["rad_soft_macro_auc"] = weak_auc
        audit["rad_gold_macro_auc"] = gold_auc
        _v10_log(f"Rad soft OOF {weak_auc:.5f} gold OOF {gold_auc:.5f}")

        dino_ids, dino_pred = _v10_load_dino_oof()
        if dino_ids is None:
            raise RuntimeError("DINO oof.npz not found for dual gate")
        dino_index = {u: i for i, u in enumerate(dino_ids)}
        train_index = {str(u): i for i, u in enumerate(train[_V10_UID])}
        gold_mask = gold
        gold_uids = train.loc[gold_mask, _V10_UID].astype(str).tolist()
        gold_y_all = train.loc[gold_mask, _V10_TARGETS].to_numpy(float)

        dino_rows, rad_rows, keep_gold_y, gold_folds = [], [], [], []
        for uid, gy in zip(gold_uids, gold_y_all):
            if uid not in dino_index or uid not in train_index:
                continue
            dino_rows.append(dino_index[uid])
            rad_rows.append(train_index[uid])
            keep_gold_y.append(gy)
            if fold_id is not None:
                gold_folds.append(int(fold_id[train_index[uid]]))
        if len(dino_rows) < 50:
            raise RuntimeError(f"too few aligned gold rows: {len(dino_rows)}")

        dino_g = dino_pred[_v10_np.array(dino_rows)]
        rad_g = oof[_v10_np.array(rad_rows)]
        gold_y = _v10_np.asarray(keep_gold_y, float)
        dino_rank = _v10_rank(dino_g)
        rad_rank = _v10_rank(rad_g)

        # V9 parent ranks on gold
        parent_alphas = _v10_np.full(len(_V10_TARGETS), _V10_BASE_ALPHA, dtype=float)
        parent_rank = _v10_blend_ranks(dino_rank, rad_rank, parent_alphas)
        parent_score = _v10_macro_auc(gold_y, parent_rank)
        dino_only = _v10_macro_auc(gold_y, dino_rank)
        rad_only = _v10_macro_auc(gold_y, rad_rank)
        audit["dino_gold_rank_macro"] = dino_only
        audit["rad_gold_rank_macro"] = rad_only
        audit["v9_parent_gold_macro"] = parent_score
        _v10_log(
            f"gold rank DINO={dino_only:.5f} Rad={rad_only:.5f} V9parent={parent_score:.5f}"
        )

        # Soft sanity (complementary family)
        soft_y = (y >= 0.5).astype(float)
        dino_soft_rows, rad_soft_rows = [], []
        for i, uid in enumerate(train[_V10_UID].astype(str)):
            if uid in dino_index:
                dino_soft_rows.append(dino_index[uid])
                rad_soft_rows.append(i)
        soft_dino = _v10_macro_auc(
            soft_y[_v10_np.array(rad_soft_rows)],
            dino_pred[_v10_np.array(dino_soft_rows)],
        )
        soft_rad = _v10_macro_auc(
            soft_y[_v10_np.array(rad_soft_rows)], oof[_v10_np.array(rad_soft_rows)]
        )
        audit["dino_soft_macro"] = soft_dino
        audit["rad_soft_macro"] = soft_rad
        soft_ok = soft_rad >= soft_dino - 0.05
        audit["soft_gate_ok"] = bool(soft_ok)
        if not soft_ok:
            audit["status"] = "SOFT_REJECT_RESTORE_DINO"
            _v10_shutil.copy2(_V10_DINO, _V10_PRIMARY)
            return audit

        # ---- Test inference for Rad (always; needed for V9 parent file) ----
        fold_preds = []
        all_te = _v10_np.arange(len(st_te), dtype=_v10_np.int64)
        for state in fold_states:
            head = _V10FoundationQueryHead().to(device)
            head.load_state_dict(state, strict=True)
            fold_preds.append(
                _v10_predict(head, test_features, test_mask, all_te, device)
            )
            del head
            _v10_torch.cuda.empty_cache()
        rad_test = _v10_np.mean(_v10_np.stack(fold_preds), axis=0)
        if not _v10_np.isfinite(rad_test).all():
            raise RuntimeError("non-finite Rad test preds")

        dino_test = _v10_pd.read_csv(_V10_DINO, dtype={_V10_UID: str})
        _v10_validate_sub(dino_test, test[_V10_UID])
        dino_test_rank = _v10_rank(dino_test[_V10_TARGETS].to_numpy(float))
        rad_test_rank = _v10_rank(rad_test)

        # Write V9 parent submission (global 0.20)
        v9_frame = dino_test.copy()
        v9_frame[_V10_TARGETS] = _v10_blend_ranks(
            dino_test_rank, rad_test_rank, parent_alphas
        )
        _v10_validate_sub(v9_frame, test[_V10_UID])
        v9_frame.to_csv(_V10_V9, index=False)
        v9_frame.to_csv(_V10_OUT / "submission_v9_parent.csv", index=False)
        audit["v9_parent_sha256"] = _v10_sha(_V10_V9)
        # Default primary = V9 parent until cross promotes
        _v10_shutil.copy2(_V10_V9, _V10_PRIMARY)

        rad_only_frame = _v10_pd.DataFrame(rad_test, columns=_V10_TARGETS)
        rad_only_frame.insert(0, _V10_UID, test[_V10_UID])
        rad_only_frame.to_csv(_V10_OUT / "submission_rad_only.csv", index=False)

        # ---- Nested gate for ACL/Syn higher alpha vs V9 parent ----
        if fold_id is None or len(gold_folds) != len(gold_y):
            gold_folds_arr = _v10_np.arange(len(gold_y)) % 5
        else:
            gold_folds_arr = _v10_np.asarray(gold_folds, int)

        cross_idx = [_V10_TARGETS.index(t) for t in _V10_CROSS_TARGETS]
        # Choose one shared cross-alpha via nested outer folds
        nested = _v10_np.zeros_like(parent_rank)
        choices = []
        fold_gains = []
        for outer in sorted(set(gold_folds_arr.tolist())):
            tr = gold_folds_arr != outer
            va = ~tr
            if int(va.sum()) == 0 or int(tr.sum()) < 8:
                continue
            scored = []
            for alpha in _V10_CROSS_GRID:
                alphas = parent_alphas.copy()
                for j in cross_idx:
                    alphas[j] = float(alpha)
                blend = _v10_blend_ranks(dino_rank[tr], rad_rank[tr], alphas)
                # light complexity penalty on alpha above base
                pen = 0.005 * float(max(0.0, alpha - _V10_BASE_ALPHA))
                scored.append(_v10_macro_auc(gold_y[tr], blend) - pen)
            best = int(_v10_np.argmax(scored))
            alpha = float(_V10_CROSS_GRID[best])
            choices.append(alpha)
            alphas = parent_alphas.copy()
            for j in cross_idx:
                alphas[j] = alpha
            nested[va] = _v10_blend_ranks(dino_rank[va], rad_rank[va], alphas)
            # held gain vs parent
            parent_va = _v10_macro_auc(gold_y[va], parent_rank[va])
            cand_va = _v10_macro_auc(gold_y[va], nested[va])
            fold_gains.append(float(cand_va - parent_va))

        nested_score = _v10_macro_auc(gold_y, nested)
        # median chosen alpha, but require > base to promote
        if choices:
            final_cross = float(_v10_np.median(_v10_np.asarray(choices)))
        else:
            final_cross = _V10_BASE_ALPHA
        final_alphas = parent_alphas.copy()
        for j in cross_idx:
            final_alphas[j] = final_cross
        final_oof = _v10_blend_ranks(dino_rank, rad_rank, final_alphas)
        final_score = _v10_macro_auc(gold_y, final_oof)
        positive_folds = int(sum(g > 0 for g in fold_gains))
        # also check per-target held gains for ACL/Syn
        per_target_pos = {}
        for t in _V10_CROSS_TARGETS:
            j = _V10_TARGETS.index(t)
            pos = 0
            for outer in sorted(set(gold_folds_arr.tolist())):
                va = gold_folds_arr == outer
                if int(va.sum()) == 0:
                    continue
                p = _v10_auc_1d(gold_y[va, j], parent_rank[va, j])
                c = _v10_auc_1d(gold_y[va, j], final_oof[va, j])
                if c == c and p == p and c >= p - 1e-9:
                    pos += 1
            per_target_pos[t] = pos

        supported = bool(
            soft_ok
            and final_cross > _V10_BASE_ALPHA + 1e-9
            and nested_score >= parent_score + 0.001
            and final_score >= parent_score + 0.001
            and positive_folds >= max(1, len(fold_gains) // 2)
            and all(per_target_pos[t] >= 3 for t in _V10_CROSS_TARGETS)
        )
        audit["cross_alpha_choices"] = choices
        audit["final_cross_alpha"] = final_cross
        audit["final_alphas"] = {
            t: float(final_alphas[i]) for i, t in enumerate(_V10_TARGETS)
        }
        audit["nested_gold_macro"] = nested_score
        audit["final_gold_macro"] = final_score
        audit["fold_gains_vs_v9"] = fold_gains
        audit["positive_outer"] = positive_folds
        audit["per_target_nonneg_folds"] = per_target_pos
        audit["selection_supported"] = supported
        _v10_log(
            f"cross gate parent={parent_score:.5f} nested={nested_score:.5f} "
            f"final={final_score:.5f} cross_a={final_cross} supported={supported} "
            f"gains={fold_gains}"
        )

        if not supported:
            audit["status"] = "CROSS_REJECTED_V9_PARENT"
            audit["promote"] = False
            # primary already V9 parent
            return audit

        # Promote test blend with target-wise alphas
        blended = dino_test.copy()
        blended[_V10_TARGETS] = _v10_blend_ranks(
            dino_test_rank, rad_test_rank, final_alphas
        )
        _v10_validate_sub(blended, test[_V10_UID])
        tag = int(round(1000 * final_cross))
        out_path = _V10_OUT / f"submission_v10_cross_{tag}.csv"
        blended.to_csv(out_path, index=False)
        _v10_shutil.copy2(out_path, _V10_PRIMARY)
        if _v10_sha(_V10_PRIMARY) != _v10_sha(out_path):
            raise RuntimeError("primary copy hash mismatch")
        # Confirm only ACL/Syn ranks differ from V9 parent on visible test
        v9_r = _v10_rank(v9_frame[_V10_TARGETS].to_numpy(float))
        b_r = _v10_rank(blended[_V10_TARGETS].to_numpy(float))
        changed = []
        for j, t in enumerate(_V10_TARGETS):
            if not _v10_np.array_equal(v9_r[:, j], b_r[:, j]):
                changed.append(t)
        audit["test_targets_rank_changed"] = changed
        audit["status"] = "CANDIDATE_SELECTED"
        audit["promote"] = True
        audit["selected_sha256"] = _v10_sha(_V10_PRIMARY)
        audit["selected_path"] = str(out_path)
        _v10_log(f"PROMOTED V10 cross alpha={final_cross} changed={changed}")
        return audit
    finally:
        for k, v in _saved.items():
            globals()[k] = v


# --- run parent DINO, then V10 Rad cross arm ---------------------------------
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

_v10_receipt = {"status": "RAD_NOT_RUN", "promote": False}
try:
    _v10_receipt = run_v10_rad_cross()
except Exception as _v10_err:
    traceback.print_exc()
    _v10_receipt = {
        "status": "RAD_ERROR_RESTORE_BEST",
        "promote": False,
        "error": f"{type(_v10_err).__name__}: {_v10_err}",
        "traceback": traceback.format_exc()[-4000:],
    }
    try:
        if _V10_V9.is_file():
            _v10_shutil.copy2(_V10_V9, _V10_PRIMARY)
        elif _V10_DINO.is_file():
            _v10_shutil.copy2(_V10_DINO, _V10_PRIMARY)
    except Exception:
        pass

_V10_AUDIT.write_text(
    _v10_json.dumps(_v10_receipt, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
_keys = (
    "status",
    "promote",
    "head_tag",
    "head_source",
    "final_cross_alpha",
    "nested_gold_macro",
    "final_gold_macro",
    "v9_parent_gold_macro",
    "error",
)
print(
    "V10 AUDIT",
    _v10_json.dumps({k: _v10_receipt.get(k) for k in _keys}, sort_keys=True, default=str),
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
        "id": "danielsolo1770/rsna-knee-v10-rad-cross-gated",
        "title": "rsna-knee-v10-rad-cross-gated",
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
            "stevenleehans/rsna-knee-llm-report-labels",
            "lixin73/rsna-knee-llm-report-labels-sol56",
            "marwanmath/resnet-50-radimagenet-marwan",
            "antoinegg1/rsna-knee-e9-radimagenet-heads-v15",
            "antoinegg1/rsna-knee-e11-diverse-heads-v20",
        ],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
        "machine_shape": "NvidiaTeslaT4",
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_NB}")
    print(f"wrote {OUT_META}")
    print(f"cells: {len(cells)} (was {len(nb['cells'])})")


if __name__ == "__main__":
    main()
