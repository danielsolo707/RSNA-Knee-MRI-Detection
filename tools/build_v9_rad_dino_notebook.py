#!/usr/bin/env python3
"""Build V9: V1 DINO baseline + dual-gated RadImageNet rank blend.

Forks the verified V1 public DINO notebook, then appends a Track-B arm that:
  1. Preserves submission.csv as the DINO fallback.
  2. Builds a 3-slot / 224px / 8-slice FS cache using V1 helpers.
  3. Encodes with frozen RadImageNet ResNet-50 (marwanmath ResNet50.pt).
  4. Loads public heads if present, else trains 5-fold query heads.
  5. Nested gold + soft dual gate vs DINO; overwrites only on promote.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_NB = ROOT / "kaggle" / "submission_v1" / "rsna-knee-v1-public-0899.ipynb"
OUT_DIR = ROOT / "kaggle" / "submission_v9_rad_dino"
OUT_NB = OUT_DIR / "rsna-knee-v9-rad-dino-gated.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

RAD_ARM = r'''
# === V9 Track B: dual-gated RadImageNet ResNet-50 rank blend ==================
# Parent submission.csv is the V1 DINO family. This arm may overwrite it only
# after nested gold OOF supports a positive Rad rank weight (verified public
# path to 0.906 uses ~20% Rad on DINO ranks). Any failure restores DINO.

import base64 as _v9_b64
import gc as _v9_gc
import hashlib as _v9_hashlib
import json as _v9_json
import shutil as _v9_shutil
import traceback as _v9_tb
from pathlib import Path as _V9Path

import numpy as _v9_np
import pandas as _v9_pd
import torch as _v9_torch
import torch.nn as _v9_nn
import torch.nn.functional as _v9_F

_V9_TARGETS = list(TARGETS)
_V9_UID = "StudyInstanceUID"
_V9_WORK = _V9Path("/kaggle/working")
_V9_PRIMARY = _V9_WORK / "submission.csv"
_V9_DINO_FALLBACK = _V9_WORK / "submission_dino_v1.csv"
_V9_AUDIT = _V9_WORK / "v9_rad_audit.json"
_V9_OUT = _V9_WORK / "rsna_rad_v9"
_V9_TOKEN_DIM = 2048
_V9_HEAD_DIM = 512
_V9_RAD_SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
]
_V9_ALPHA_GRID = _v9_np.array([0.0, 0.05, 0.10, 0.15, 0.20], dtype=float)
_V9_EXPECTED_CKPT = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"


def _v9_log(msg):
    try:
        log(f"[V9] {msg}")
    except Exception:
        print(f"[V9] {msg}", flush=True)


def _v9_sha(path):
    h = _v9_hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _v9_find_file(name):
    root = _V9Path("/kaggle/input")
    if not root.is_dir():
        raise FileNotFoundError(name)
    for p in root.rglob(name):
        # skip huge series trees
        parts = set(p.parts)
        if "train_series" in parts or "test_series" in parts:
            continue
        if p.is_file():
            return p
    raise FileNotFoundError(name)


def _v9_rank(values):
    frame = _v9_pd.DataFrame(_v9_np.asarray(values, dtype=_v9_np.float64))
    return frame.rank(method="average", pct=True).to_numpy(_v9_np.float64)


def _v9_auc_1d(y, s):
    y = _v9_np.asarray(y, dtype=float)
    s = _v9_np.asarray(s, dtype=float)
    hard = (y >= 0.5).astype(float)
    pos = hard == 1
    neg = hard == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = _v9_np.argsort(s, kind="mergesort")
    ranks = _v9_np.empty(len(s), dtype=float)
    ranks[order] = _v9_np.arange(1, len(s) + 1, dtype=float)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _v9_macro_auc(y, pred):
    vals = []
    for j in range(y.shape[1]):
        a = _v9_auc_1d(y[:, j], pred[:, j])
        if a == a:
            vals.append(a)
    if not vals:
        return float("nan")
    return float(_v9_np.mean(vals))


def _v9_validate_sub(frame, expected_ids):
    cols = [_V9_UID] + _V9_TARGETS
    if frame.columns.tolist() != cols:
        raise RuntimeError(f"schema drift: {frame.columns.tolist()}")
    ids = frame[_V9_UID].astype(str).tolist()
    exp = list(map(str, expected_ids))
    if ids != exp or len(ids) != len(set(ids)):
        raise RuntimeError("study identity/order drift")
    vals = frame[_V9_TARGETS].to_numpy(float)
    if not _v9_np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
        raise RuntimeError("invalid probability values")


class _V9FoundationQueryHead(_v9_nn.Module):
    def __init__(self, n_slot=3, n_slices=8, n_targets=12):
        super().__init__()
        self.n_slot = n_slot
        self.n_slices = n_slices
        self.project = _v9_nn.Sequential(
            _v9_nn.LayerNorm(_V9_TOKEN_DIM),
            _v9_nn.Linear(_V9_TOKEN_DIM, _V9_HEAD_DIM),
            _v9_nn.GELU(),
        )
        self.plane = _v9_nn.Parameter(_v9_torch.randn(n_slot, _V9_HEAD_DIM) * 0.01)
        self.position = _v9_nn.Parameter(_v9_torch.randn(n_slices, _V9_HEAD_DIM) * 0.01)
        self.query = _v9_nn.Parameter(_v9_torch.randn(n_targets, _V9_HEAD_DIM) * 0.02)
        self.attn = _v9_nn.MultiheadAttention(_V9_HEAD_DIM, 8, dropout=0.10, batch_first=True)
        self.fuse = _v9_nn.Sequential(
            _v9_nn.LayerNorm(_V9_HEAD_DIM * 4),
            _v9_nn.Linear(_V9_HEAD_DIM * 4, _V9_HEAD_DIM),
            _v9_nn.GELU(),
            _v9_nn.Dropout(0.15),
        )
        self.weight = _v9_nn.Parameter(_v9_torch.randn(n_targets, _V9_HEAD_DIM) * 0.02)
        self.bias = _v9_nn.Parameter(_v9_torch.zeros(n_targets))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), self.n_slot, self.n_slices, _V9_HEAD_DIM)
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
            _v9_torch.cat(
                [attended, mean, _v9_torch.abs(attended - mean), attended * mean], -1
            )
        )
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


def _v9_load_encoder(device):
    from torchvision.models import resnet50

    ckpt = _v9_find_file("ResNet50.pt")
    sha = _v9_sha(ckpt)
    if sha != _V9_EXPECTED_CKPT:
        _v9_log(f"RadImageNet sha mismatch (continuing): {sha}")
    class Enc(_v9_nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _v9_nn.Sequential(*list(resnet50(weights=None).children())[:-2])

        def forward(self, image):
            return self.backbone(image).mean(dim=(2, 3))

    model = Enc()
    state = _v9_torch.load(str(ckpt), map_location="cpu", weights_only=True)
    # accept backbone.* or raw resnet keys
    if state and all(str(k).startswith("backbone.") for k in state):
        model.load_state_dict(state, strict=True)
    else:
        # map torchvision-style into Sequential
        cleaned = {}
        for k, v in state.items():
            k = str(k).replace("module.", "", 1)
            if k.startswith("backbone."):
                cleaned[k] = v
            elif k.startswith("fc."):
                continue
            else:
                cleaned["backbone." + k] = v
        # rebuild expected keys from model
        try:
            model.load_state_dict(cleaned, strict=False)
        except Exception:
            # last resort: load into backbone children by filtering matching shapes
            msd = model.state_dict()
            filtered = {k: v for k, v in cleaned.items() if k in msd and msd[k].shape == v.shape}
            model.load_state_dict(filtered, strict=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters())
    _v9_log(f"RadImageNet encoder loaded params={n:,} sha={sha[:12]}")
    return model, sha


@_v9_torch.inference_mode()
def _v9_encode(cache, slot_mask, device):
    n, slots, slices, h, w = cache.shape
    features = _v9_np.zeros((n, slots * slices, _V9_TOKEN_DIM), _v9_np.float16)
    token_mask = _v9_np.repeat(slot_mask[:, :, None], slices, axis=2).reshape(n, -1)
    valid = _v9_np.flatnonzero(token_mask.reshape(-1) > 0)
    flat = cache.reshape(-1, h, w)
    model, sha = _v9_load_encoder(device)
    batch = 64 if device.type == "cuda" else 8
    for b0 in range(0, len(valid), batch):
        ix = valid[b0 : b0 + batch]
        x = _v9_torch.from_numpy(flat[ix]).to(device).float().div_(127.5).sub_(1.0)
        x = x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        with _v9_torch.autocast("cuda", enabled=device.type == "cuda"):
            feat = model(x)
        features.reshape(-1, _V9_TOKEN_DIM)[ix] = feat.float().cpu().numpy().astype(_v9_np.float16)
        if b0 % (batch * 50) == 0:
            _v9_log(f"encoded {b0}/{len(valid)} slices")
    del model
    if device.type == "cuda":
        _v9_torch.cuda.empty_cache()
    return features, token_mask.astype(_v9_np.float32), sha


@_v9_torch.inference_mode()
def _v9_predict(model, features, masks, indices, device, batch=64):
    model.eval()
    out = []
    for b0 in range(0, len(indices), batch):
        ix = indices[b0 : b0 + batch]
        x = _v9_torch.from_numpy(features[ix]).to(device)
        m = _v9_torch.from_numpy(masks[ix]).to(device)
        with _v9_torch.autocast("cuda", enabled=device.type == "cuda"):
            out.append(_v9_torch.sigmoid(model(x, m)).float().cpu())
    return _v9_torch.cat(out).numpy()


def _v9_make_targets(train):
    uid = _V9_UID
    sources = []
    for name in ("report_labels_v2.csv", "llm_labels_v2.csv", "labels_llm_gpt56sol.csv"):
        try:
            sources.append(_v9_pd.read_csv(_v9_find_file(name)))
            _v9_log(f"label source ok: {name}")
        except FileNotFoundError:
            _v9_log(f"label source missing: {name}")
    if not sources:
        raise RuntimeError("no report-label CSV attached")
    cube = []
    for frame in sources:
        frame = frame.copy()
        if frame[uid].duplicated().any():
            frame = frame.drop_duplicates(uid, keep="first")
        cols = [c for c in _V9_TARGETS if c in frame.columns]
        if len(cols) != len(_V9_TARGETS):
            raise RuntimeError(f"label CSV missing targets: {frame.columns.tolist()[:20]}")
        aligned = train[[uid]].merge(frame[[uid] + _V9_TARGETS], on=uid, how="left")
        cube.append(aligned[_V9_TARGETS].to_numpy(float))
    cube = _v9_np.stack(cube)
    y = _v9_np.nanmean(cube, axis=0).astype(_v9_np.float32)
    # disagreement-aware weights
    with _v9_np.errstate(all="ignore"):
        disagreement = _v9_np.nanmean(_v9_np.abs(cube - y[None]), axis=0)
    agreement = _v9_np.clip(1.0 - 2.0 * disagreement, 0, 1)
    certainty = _v9_np.clip(2.0 * _v9_np.abs(y - 0.5), 0, 1)
    w = (0.15 + 0.85 * (0.65 * agreement + 0.35 * certainty)).astype(_v9_np.float32)
    gold = train[_V9_TARGETS].notna().all(axis=1).to_numpy()
    y[gold] = train.loc[gold, _V9_TARGETS].to_numpy(_v9_np.float32)
    w[gold] = 3.0
    # fill remaining nan with 0.5 / low weight
    nan = ~_v9_np.isfinite(y)
    y[nan] = 0.5
    w[nan] = 0.05
    return y, w, gold


def _v9_report_groups(train):
    report = (
        train.Report.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return _v9_np.array(
        [_v9_hashlib.sha256(x.encode()).hexdigest()[:24] for x in report]
    )


def _v9_train_fold(features, masks, y, weights, train_idx, val_idx, fold, device):
    from torch.utils.data import DataLoader, Dataset

    class Rows(Dataset):
        def __init__(self, indices):
            self.indices = _v9_np.asarray(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, k):
            i = int(self.indices[k])
            return features[i], masks[i], y[i], weights[i]

    model = _V9FoundationQueryHead(
        n_slot=3, n_slices=8, n_targets=len(_V9_TARGETS)
    ).to(device)
    opt = _v9_torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=3e-3)
    gen = _v9_torch.Generator().manual_seed(int(SEED) + 100 + fold)
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
            with _v9_torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x, m)
                raw = _v9_F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                )
                loss = (raw * weight).sum() / weight.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            _v9_nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        pred = _v9_predict(model, features, masks, val_idx, device)
        score = _v9_macro_auc(y[val_idx], pred)
        _v9_log(f"fold {fold} ep {epoch}: weak-val AUC {score:.5f}")
        if score > best_auc + 2e-4:
            best_auc, stale = score, 0
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 5:
                break
        if time.time() - T0 > TIME_BUDGET - 0.5 * 3600:
            _v9_log("time budget near end during head train")
            break
    return best, best_auc


def _v9_try_load_public_heads(device):
    """Load Antoine-style v52 heads if attached."""
    for name in ("v52_radimagenet_heads.pt", "v52_e11_heads.pt"):
        try:
            path = _v9_find_file(name)
        except FileNotFoundError:
            continue
        payload = _v9_torch.load(str(path), map_location="cpu", weights_only=False)
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
        oof_path = None
        for oof_name in ("v52_oof.csv", "v52_e11_oof.csv"):
            try:
                oof_path = _v9_find_file(oof_name)
                break
            except FileNotFoundError:
                pass
        _v9_log(f"public heads loaded from {path}")
        return states, oof_path, str(path)
    return None, None, None


def _v9_load_dino_oof():
    """Return (ids, pred[n,12]) for public DINO family OOF if present."""
    try:
        path = _v9_find_file("oof.npz")
    except FileNotFoundError:
        return None, None
    data = _v9_np.load(path, allow_pickle=True)
    ids = [str(x) for x in data["ids"].tolist()]
    if "pred" in data.files:
        pred = data["pred"].astype(_v9_np.float64)
    elif "ours" in data.files:
        pred = data["ours"].astype(_v9_np.float64)
    else:
        return None, None
    return ids, pred


def run_v9_radimagenet():
    audit = {
        "status": "DINO_PRESERVED",
        "promote": False,
        "evidence_boundary": (
            "Nested 58-label OOF is a local diagnostic, not a Kaggle score. "
            "submission.csv is overwritten only when promote=true."
        ),
    }
    _V9_OUT.mkdir(parents=True, exist_ok=True)
    if not _V9_PRIMARY.is_file():
        raise FileNotFoundError("parent submission.csv missing before Rad arm")
    _v9_shutil.copy2(_V9_PRIMARY, _V9_DINO_FALLBACK)
    audit["dino_sha256"] = _v9_sha(_V9_DINO_FALLBACK)

    device = _v9_torch.device("cuda:0" if _v9_torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("RadImageNet arm requires CUDA")
    remaining = TIME_BUDGET - (time.time() - T0)
    audit["remaining_seconds_at_start"] = remaining
    if remaining < 1.5 * 3600:
        raise TimeoutError(f"only {remaining/60:.1f} min left for Rad arm")

    # --- geometry override for V52-style FS cache ---
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
        SLOTS = list(_V9_RAD_SLOTS)
        N_SLOT = 3
        IMG = CACHE_IMG = 224
        CACHE_SLICES = 8
        GROUP = 1
        N_GROUP = 8
        CROP_MM = 10_000.0
        SLICE_BAND = (0.12, 0.88)

        train = _v9_pd.read_csv(ROOT / "train.csv", dtype={_V9_UID: str})
        test = _v9_pd.read_csv(ROOT / "test.csv", dtype={_V9_UID: str})
        train_series = _v9_pd.read_csv(
            ROOT / "train_series.csv", dtype={_V9_UID: str, "SeriesInstanceUID": str}
        )
        test_series = _v9_pd.read_csv(
            ROOT / "test_series.csv", dtype={_V9_UID: str, "SeriesInstanceUID": str}
        )
        both = _v9_pd.concat([train_series, test_series], ignore_index=True)
        plane_map = dict(zip(both["SeriesInstanceUID"], both["Anatomical_Plane"]))

        public_states, public_oof_path, public_heads_path = _v9_try_load_public_heads(device)
        audit["public_heads"] = public_heads_path

        # Always need test features; train features if we train heads or lack public OOF
        _v9_log("building test FS cache @224")
        hte = annotate(walk("test_series"))
        slots_te = pick_slots(hte, plane_map)
        st_te, Cte, Mte = build_cache(slots_te, plane_map, lat_of(hte, "v9-test "), "v9-test")
        te_by = {str(u): i for i, u in enumerate(st_te)}
        te_order = _v9_np.array([te_by[str(u)] for u in test[_V9_UID]], dtype=_v9_np.int64)
        Cte, Mte = Cte[te_order], Mte[te_order]
        st_te = [str(u) for u in test[_V9_UID]]
        test_features, test_mask, enc_sha = _v9_encode(Cte, Mte, device)
        del Cte, Mte, hte
        _v9_gc.collect()
        audit["encoder_sha256"] = enc_sha

        fold_states = None
        oof = None
        fold_id = None
        y = w = gold = None

        if public_states is not None and public_oof_path is not None:
            fold_states = public_states
            oof_df = _v9_pd.read_csv(public_oof_path, dtype={_V9_UID: str})
            oof_df = train[[_V9_UID]].merge(oof_df, on=_V9_UID, how="left")
            oof = oof_df[_V9_TARGETS].to_numpy(float)
            if "fold" in oof_df.columns:
                fold_id = oof_df["fold"].to_numpy(int)
            y, w, gold = _v9_make_targets(train)
            audit["head_source"] = "public_attached"
        else:
            audit["head_source"] = "train_in_kernel"
            _v9_log("building train FS cache @224")
            htr = annotate(walk("train_series"))
            slots_tr = pick_slots(htr, plane_map)
            st_tr, Ctr, Mtr = build_cache(
                slots_tr, plane_map, lat_of(htr, "v9-train "), "v9-train"
            )
            tr_by = {str(u): i for i, u in enumerate(st_tr)}
            tr_order = _v9_np.array(
                [tr_by[str(u)] for u in train[_V9_UID]], dtype=_v9_np.int64
            )
            Ctr, Mtr = Ctr[tr_order], Mtr[tr_order]
            features, token_mask, _ = _v9_encode(Ctr, Mtr, device)
            del Ctr, Mtr, htr
            _v9_gc.collect()

            y, w, gold = _v9_make_targets(train)
            groups = _v9_report_groups(train)
            from sklearn.model_selection import GroupKFold

            splits = list(GroupKFold(5).split(features, groups=groups))
            fold_id = _v9_np.full(len(train), -1, dtype=_v9_np.int8)
            oof = _v9_np.zeros_like(y, dtype=_v9_np.float32)
            fold_states = []
            for fold, (tr, va) in enumerate(splits):
                fold_id[va] = fold
                state, score = _v9_train_fold(
                    features, token_mask, y, w, tr, va, fold, device
                )
                if state is None:
                    raise RuntimeError(f"fold {fold} produced no checkpoint")
                head = _V9FoundationQueryHead().to(device)
                head.load_state_dict(state, strict=True)
                oof[va] = _v9_predict(head, features, token_mask, va, device)
                fold_states.append(state)
                del head
                _v9_torch.cuda.empty_cache()
                audit.setdefault("fold_weak_auc", {})[str(fold)] = float(score)
            del features, token_mask
            _v9_gc.collect()
            _v9_torch.save(
                {"folds": [{"fold": i, "state_dict": s} for i, s in enumerate(fold_states)]},
                _V9_OUT / "v9_rad_heads.pt",
            )
            oof_frame = _v9_pd.DataFrame(oof, columns=_V9_TARGETS)
            oof_frame.insert(0, _V9_UID, train[_V9_UID])
            oof_frame["fold"] = fold_id
            oof_frame["is_gold"] = gold.astype(_v9_np.uint8)
            oof_frame.to_csv(_V9_OUT / "v9_rad_oof.csv", index=False)

        if oof is None or fold_states is None:
            raise RuntimeError("Rad OOF / heads unavailable")
        if not _v9_np.isfinite(oof).all():
            raise RuntimeError("non-finite Rad OOF")

        weak_auc = _v9_macro_auc(y, oof)
        gold_auc = _v9_macro_auc(y[gold], oof[gold])
        audit["rad_soft_macro_auc"] = weak_auc
        audit["rad_gold_macro_auc"] = gold_auc
        _v9_log(f"Rad soft OOF {weak_auc:.5f} gold OOF {gold_auc:.5f}")

        # DINO OOF align on gold rows
        dino_ids, dino_pred = _v9_load_dino_oof()
        gold_mask = gold if gold is not None else train[_V9_TARGETS].notna().all(axis=1).to_numpy()
        gold_uids = train.loc[gold_mask, _V9_UID].astype(str).tolist()
        gold_y = train.loc[gold_mask, _V9_TARGETS].to_numpy(float)
        if dino_ids is None:
            raise RuntimeError("DINO oof.npz not found for dual gate")
        dino_index = {u: i for i, u in enumerate(dino_ids)}
        train_index = {str(u): i for i, u in enumerate(train[_V9_UID])}
        dino_rows = []
        rad_rows = []
        keep_gold_y = []
        gold_folds = []
        for uid, gy in zip(gold_uids, gold_y):
            if uid not in dino_index or uid not in train_index:
                continue
            dino_rows.append(dino_index[uid])
            rad_rows.append(train_index[uid])
            keep_gold_y.append(gy)
            if fold_id is not None:
                gold_folds.append(int(fold_id[train_index[uid]]))
        if len(dino_rows) < 50:
            raise RuntimeError(f"too few aligned gold rows: {len(dino_rows)}")
        dino_g = dino_pred[_v9_np.array(dino_rows)]
        rad_g = oof[_v9_np.array(rad_rows)]
        gold_y = _v9_np.asarray(keep_gold_y, float)
        dino_rank = _v9_rank(dino_g)
        rad_rank = _v9_rank(rad_g)
        base_score = _v9_macro_auc(gold_y, dino_rank)
        rad_only = _v9_macro_auc(gold_y, rad_rank)
        audit["dino_gold_rank_macro"] = base_score
        audit["rad_gold_rank_macro"] = rad_only

        # soft diversity gate: rad soft should not be catastrophic vs dino soft
        soft_y = (y >= 0.5).astype(float)
        dino_soft_rows = []
        rad_soft_rows = []
        for i, uid in enumerate(train[_V9_UID].astype(str)):
            if uid in dino_index:
                dino_soft_rows.append(dino_index[uid])
                rad_soft_rows.append(i)
        dino_soft = dino_pred[_v9_np.array(dino_soft_rows)]
        rad_soft = oof[_v9_np.array(rad_soft_rows)]
        soft_dino_macro = _v9_macro_auc(soft_y[_v9_np.array(rad_soft_rows)], dino_soft)
        soft_rad_macro = _v9_macro_auc(soft_y[_v9_np.array(rad_soft_rows)], rad_soft)
        audit["dino_soft_macro"] = soft_dino_macro
        audit["rad_soft_macro_aligned"] = soft_rad_macro
        soft_ok = soft_rad_macro >= soft_dino_macro - 0.05  # looser: complementary family
        audit["soft_gate_ok"] = bool(soft_ok)

        # nested alpha on gold folds
        if fold_id is None or len(gold_folds) != len(gold_y):
            # fallback: single LOO-style grid without folds
            gold_folds = list(range(len(gold_y)))  # degenerate
            nested = _v9_np.zeros_like(dino_rank)
            choices = []
            for outer in range(len(gold_y)):
                tr = _v9_np.ones(len(gold_y), dtype=bool)
                tr[outer] = False
                va = ~tr
                scored = []
                for alpha in _V9_ALPHA_GRID:
                    blend = (1.0 - alpha) * dino_rank[tr] + alpha * rad_rank[tr]
                    scored.append(_v9_macro_auc(gold_y[tr], blend) - 0.01 * float(alpha))
                best = int(_v9_np.argmax(scored))
                alpha = float(_V9_ALPHA_GRID[best])
                choices.append(alpha)
                nested[va] = (1.0 - alpha) * dino_rank[va] + alpha * rad_rank[va]
        else:
            gold_folds = _v9_np.asarray(gold_folds, int)
            nested = _v9_np.zeros_like(dino_rank)
            choices = []
            for outer in sorted(set(gold_folds.tolist())):
                tr = gold_folds != outer
                va = ~tr
                if va.sum() == 0 or tr.sum() < 10:
                    continue
                scored = []
                for alpha in _V9_ALPHA_GRID:
                    blend = (1.0 - alpha) * dino_rank[tr] + alpha * rad_rank[tr]
                    scored.append(_v9_macro_auc(gold_y[tr], blend) - 0.01 * float(alpha))
                best = int(_v9_np.argmax(scored))
                alpha = float(_V9_ALPHA_GRID[best])
                choices.append(alpha)
                nested[va] = (1.0 - alpha) * dino_rank[va] + alpha * rad_rank[va]

        nested_score = _v9_macro_auc(gold_y, nested)
        final_alpha = float(min(0.20, _v9_np.median(_v9_np.asarray(choices)))) if choices else 0.0
        final_oof = (1.0 - final_alpha) * dino_rank + final_alpha * rad_rank
        final_score = _v9_macro_auc(gold_y, final_oof)
        positive_folds = int(sum(a > 0 for a in choices))
        supported = bool(
            soft_ok
            and final_alpha > 0
            and positive_folds >= max(1, len(choices) // 2)
            and nested_score >= base_score + 0.001
            and final_score >= base_score + 0.001
        )
        audit["alpha_choices"] = choices
        audit["final_alpha"] = final_alpha
        audit["nested_gold_macro"] = nested_score
        audit["final_gold_macro"] = final_score
        audit["positive_outer"] = positive_folds
        audit["selection_supported"] = supported
        _v9_log(
            f"gate base={base_score:.5f} nested={nested_score:.5f} "
            f"final={final_score:.5f} alpha={final_alpha} supported={supported}"
        )

        if not supported:
            audit["status"] = "OOF_REJECTED_DINO_PRESERVED"
            _v9_shutil.copy2(_V9_DINO_FALLBACK, _V9_PRIMARY)
            return audit

        # test inference
        fold_preds = []
        all_te = _v9_np.arange(len(st_te), dtype=_v9_np.int64)
        for state in fold_states:
            head = _V9FoundationQueryHead().to(device)
            head.load_state_dict(state, strict=True)
            fold_preds.append(_v9_predict(head, test_features, test_mask, all_te, device))
            del head
            _v9_torch.cuda.empty_cache()
        rad_test = _v9_np.mean(_v9_np.stack(fold_preds), axis=0)
        if not _v9_np.isfinite(rad_test).all():
            raise RuntimeError("non-finite Rad test preds")

        baseline = _v9_pd.read_csv(_V9_DINO_FALLBACK, dtype={_V9_UID: str})
        _v9_validate_sub(baseline, test[_V9_UID])
        rad_frame = _v9_pd.DataFrame(rad_test, columns=_V9_TARGETS)
        rad_frame.insert(0, _V9_UID, test[_V9_UID])
        _v9_validate_sub(rad_frame, test[_V9_UID])
        rad_frame.to_csv(_V9_OUT / "submission_rad_only.csv", index=False)

        base_rank = _v9_rank(baseline[_V9_TARGETS].to_numpy(float))
        rad_test_rank = _v9_rank(rad_test)
        blended = baseline.copy()
        blended[_V9_TARGETS] = (1.0 - final_alpha) * base_rank + final_alpha * rad_test_rank
        _v9_validate_sub(blended, test[_V9_UID])
        out_path = _V9_OUT / f"submission_v9_rad_{int(round(1000 * final_alpha)):03d}.csv"
        blended.to_csv(out_path, index=False)
        _v9_shutil.copy2(out_path, _V9_PRIMARY)
        if _v9_sha(_V9_PRIMARY) != _v9_sha(out_path):
            raise RuntimeError("primary copy hash mismatch")
        audit["status"] = "CANDIDATE_SELECTED"
        audit["promote"] = True
        audit["selected_sha256"] = _v9_sha(_V9_PRIMARY)
        audit["selected_path"] = str(out_path)
        _v9_log(f"PROMOTED Rad blend alpha={final_alpha}")
        return audit
    finally:
        # restore geometry so any later cells are not surprised
        for k, v in _saved.items():
            globals()[k] = v


# --- run parent DINO, then optional Rad arm ---------------------------------
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

_v9_receipt = {"status": "RAD_NOT_RUN"}
try:
    _v9_receipt = run_v9_radimagenet()
except Exception as _v9_err:
    traceback.print_exc()
    _v9_receipt = {
        "status": "RAD_ERROR_DINO_PRESERVED",
        "error": f"{type(_v9_err).__name__}: {_v9_err}",
        "traceback": traceback.format_exc()[-4000:],
    }
    try:
        if _V9_DINO_FALLBACK.is_file():
            _v9_shutil.copy2(_V9_DINO_FALLBACK, _V9_PRIMARY)
        elif Path("submission.csv").is_file():
            pass
    except Exception:
        pass

_V9_AUDIT.write_text(
    _v9_json.dumps(_v9_receipt, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
_keys = (
    "status", "promote", "final_alpha", "nested_gold_macro", "final_gold_macro",
    "rad_soft_macro_auc", "dino_soft_macro", "error", "head_source",
)
print("V9 AUDIT", _v9_json.dumps({k: _v9_receipt.get(k) for k in _keys}, sort_keys=True, default=str))
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
    # Drop V1's final try/main cell; replace with DINO+Rad runner
    # V1 final cell is the last code cell that calls main()
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
        "id": "danielsolo1770/rsna-knee-v9-rad-dino-gated",
        "title": "rsna-knee-v9-rad-dino-gated",
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
