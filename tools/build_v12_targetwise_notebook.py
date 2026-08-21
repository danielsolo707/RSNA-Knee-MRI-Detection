#!/usr/bin/env python3
"""Build V12: target-wise own-vs-e9 Rad family, dual-gated vs V9 parent.

Uses frozen V11 own heads/OOF dataset + public e9 OOF/heads.
No retrain. Promote only if nested gold >= V9 parent + 0.002.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_NB = ROOT / "kaggle" / "submission_v1" / "rsna-knee-v1-public-0899.ipynb"
OUT_DIR = ROOT / "kaggle" / "submission_v12_targetwise"
OUT_NB = OUT_DIR / "rsna-knee-v12-targetwise-gated.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

RAD_ARM = r'''
# === V12: target-wise own vs e9 Rad, dual-gated vs V9 parent ==================
# Parent public: V9 DINO+e9@0.20 = 0.906
# For each target pick own or e9 by soft OOF, then nest-gate DINO+rad alpha.

import gc as _v12_gc
import hashlib as _v12_hashlib
import json as _v12_json
import shutil as _v12_shutil
from pathlib import Path as _V12Path

import numpy as _v12_np
import pandas as _v12_pd
import torch as _v12_torch
import torch.nn as _v12_nn

_V12_TARGETS = list(TARGETS)
_V12_UID = "StudyInstanceUID"
_V12_WORK = _V12Path("/kaggle/working")
_V12_PRIMARY = _V12_WORK / "submission.csv"
_V12_DINO = _V12_WORK / "submission_dino_v1.csv"
_V12_V9 = _V12_WORK / "submission_v9_parent.csv"
_V12_AUDIT = _V12_WORK / "v12_rad_audit.json"
_V12_OUT = _V12_WORK / "rsna_rad_v12"
_V12_TOKEN_DIM = 2048
_V12_HEAD_DIM = 512
_V12_BASE_ALPHA = 0.20
_V12_MIN_GAIN = 0.002
_V12_ALPHA_GRID = _v12_np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30], dtype=float)
_V12_RAD_SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
]
_V12_EXPECTED_CKPT = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"


def _v12_log(msg):
    try:
        log(f"[V12] {msg}")
    except Exception:
        print(f"[V12] {msg}", flush=True)


def _v12_sha(path):
    h = _v12_hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _v12_find_file(name):
    root = _V12Path("/kaggle/input")
    for p in root.rglob(name):
        parts = set(p.parts)
        if "train_series" in parts or "test_series" in parts:
            continue
        if p.is_file():
            return p
    raise FileNotFoundError(name)


def _v12_rank(values):
    frame = _v12_pd.DataFrame(_v12_np.asarray(values, dtype=_v12_np.float64))
    return frame.rank(method="average", pct=True).to_numpy(_v12_np.float64)


def _v12_auc_1d(y, s):
    y = _v12_np.asarray(y, dtype=float)
    s = _v12_np.asarray(s, dtype=float)
    hard = (y >= 0.5).astype(float)
    pos = hard == 1
    neg = hard == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = _v12_np.argsort(s, kind="mergesort")
    ranks = _v12_np.empty(len(s), dtype=float)
    ranks[order] = _v12_np.arange(1, len(s) + 1, dtype=float)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _v12_macro_auc(y, pred):
    vals = []
    for j in range(y.shape[1]):
        a = _v12_auc_1d(y[:, j], pred[:, j])
        if a == a:
            vals.append(a)
    return float(_v12_np.mean(vals)) if vals else float("nan")


def _v12_validate_sub(frame, expected_ids):
    cols = [_V12_UID] + _V12_TARGETS
    if frame.columns.tolist() != cols:
        raise RuntimeError(f"schema drift: {frame.columns.tolist()}")
    ids = frame[_V12_UID].astype(str).tolist()
    exp = list(map(str, expected_ids))
    if ids != exp or len(ids) != len(set(ids)):
        raise RuntimeError("study identity/order drift")
    vals = frame[_V12_TARGETS].to_numpy(float)
    if not _v12_np.isfinite(vals).all() or vals.min() < 0 or vals.max() > 1:
        raise RuntimeError("invalid probability values")


def _v12_blend_ranks(base, rad, alpha):
    a = float(alpha)
    return (1.0 - a) * base + a * rad


def _v12_blend_ranks_per_target(base, rad, alphas):
    a = _v12_np.asarray(alphas, dtype=_v12_np.float64).reshape(1, -1)
    return (1.0 - a) * base + a * rad


class _V12Head(_v12_nn.Module):
    def __init__(self, n_slot=3, n_slices=8, n_targets=12):
        super().__init__()
        self.n_slot = n_slot
        self.n_slices = n_slices
        self.project = _v12_nn.Sequential(
            _v12_nn.LayerNorm(_V12_TOKEN_DIM),
            _v12_nn.Linear(_V12_TOKEN_DIM, _V12_HEAD_DIM),
            _v12_nn.GELU(),
        )
        self.plane = _v12_nn.Parameter(_v12_torch.randn(n_slot, _V12_HEAD_DIM) * 0.01)
        self.position = _v12_nn.Parameter(_v12_torch.randn(n_slices, _V12_HEAD_DIM) * 0.01)
        self.query = _v12_nn.Parameter(_v12_torch.randn(n_targets, _V12_HEAD_DIM) * 0.02)
        self.attn = _v12_nn.MultiheadAttention(_V12_HEAD_DIM, 8, dropout=0.10, batch_first=True)
        self.fuse = _v12_nn.Sequential(
            _v12_nn.LayerNorm(_V12_HEAD_DIM * 4),
            _v12_nn.Linear(_V12_HEAD_DIM * 4, _V12_HEAD_DIM),
            _v12_nn.GELU(),
            _v12_nn.Dropout(0.15),
        )
        self.weight = _v12_nn.Parameter(_v12_torch.randn(n_targets, _V12_HEAD_DIM) * 0.02)
        self.bias = _v12_nn.Parameter(_v12_torch.zeros(n_targets))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), self.n_slot, self.n_slices, _V12_HEAD_DIM)
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
            _v12_torch.cat(
                [attended, mean, _v12_torch.abs(attended - mean), attended * mean], -1
            )
        )
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


def _v12_load_encoder(device):
    from torchvision.models import resnet50

    ckpt = _v12_find_file("ResNet50.pt")
    sha = _v12_sha(ckpt)

    class Enc(_v12_nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _v12_nn.Sequential(*list(resnet50(weights=None).children())[:-2])

        def forward(self, image):
            return self.backbone(image).mean(dim=(2, 3))

    model = Enc()
    state = _v12_torch.load(str(ckpt), map_location="cpu", weights_only=True)
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
        model.load_state_dict(cleaned, strict=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    _v12_log(f"encoder sha={sha[:12]}")
    return model, sha


@_v12_torch.inference_mode()
def _v12_encode(cache, slot_mask, device):
    n, slots, slices, h, w = cache.shape
    features = _v12_np.zeros((n, slots * slices, _V12_TOKEN_DIM), _v12_np.float16)
    token_mask = _v12_np.repeat(slot_mask[:, :, None], slices, axis=2).reshape(n, -1)
    valid = _v12_np.flatnonzero(token_mask.reshape(-1) > 0)
    flat = cache.reshape(-1, h, w)
    model, sha = _v12_load_encoder(device)
    batch = 64 if device.type == "cuda" else 8
    for b0 in range(0, len(valid), batch):
        ix = valid[b0 : b0 + batch]
        x = _v12_torch.from_numpy(flat[ix]).to(device).float().div_(127.5).sub_(1.0)
        x = x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        with _v12_torch.autocast("cuda", enabled=device.type == "cuda"):
            feat = model(x)
        features.reshape(-1, _V12_TOKEN_DIM)[ix] = feat.float().cpu().numpy().astype(
            _v12_np.float16
        )
    del model
    if device.type == "cuda":
        _v12_torch.cuda.empty_cache()
    return features, token_mask.astype(_v12_np.float32), sha


@_v12_torch.inference_mode()
def _v12_predict(model, features, masks, indices, device, batch=64):
    model.eval()
    out = []
    for b0 in range(0, len(indices), batch):
        ix = indices[b0 : b0 + batch]
        x = _v12_torch.from_numpy(features[ix]).to(device)
        m = _v12_torch.from_numpy(masks[ix]).to(device)
        with _v12_torch.autocast("cuda", enabled=device.type == "cuda"):
            out.append(_v12_torch.sigmoid(model(x, m)).float().cpu())
    return _v12_torch.cat(out).numpy()


def _v12_load_heads(path, expected_slots=3):
    payload = _v12_torch.load(str(path), map_location="cpu", weights_only=False)
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError(f"bad heads payload {path}")
    states = []
    for rec in folds:
        st = rec.get("state_dict") if isinstance(rec, dict) else None
        if st is None:
            raise RuntimeError("missing state_dict")
        plane = st.get("plane")
        if plane is None or int(plane.shape[0]) != expected_slots:
            raise RuntimeError(f"slot mismatch in {path}")
        states.append(st)
    return states


def _v12_load_oof_csv(name, train):
    path = _v12_find_file(name)
    df = _v12_pd.read_csv(path, dtype={_V12_UID: str})
    df = train[[_V12_UID]].merge(df, on=_V12_UID, how="left")
    arr = df[_V12_TARGETS].apply(_v12_pd.to_numeric, errors="coerce").to_numpy(float)
    if not _v12_np.isfinite(arr).all():
        raise RuntimeError(f"non-finite OOF in {name}")
    fold = df["fold"].to_numpy(int) if "fold" in df.columns else None
    _v12_log(f"loaded OOF {name}")
    return arr, fold


def _v12_load_dino_oof():
    data = _v12_np.load(_v12_find_file("oof.npz"), allow_pickle=True)
    ids = [str(x) for x in data["ids"].tolist()]
    pred = data["pred"] if "pred" in data.files else data["ours"]
    return ids, pred.astype(_v12_np.float64)


def run_v12_targetwise():
    audit = {
        "status": "KEEP_V9_PARENT",
        "promote": False,
        "min_gain": _V12_MIN_GAIN,
        "parent_recipe": "V9 DINO+e9 alpha=0.20",
        "public_parent_score_reference": 0.906,
    }
    _V12_OUT.mkdir(parents=True, exist_ok=True)
    if not _V12_PRIMARY.is_file():
        raise FileNotFoundError("submission.csv missing")
    _v12_shutil.copy2(_V12_PRIMARY, _V12_DINO)
    audit["dino_sha256"] = _v12_sha(_V12_DINO)

    device = _v12_torch.device("cuda:0" if _v12_torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required")

    global SLOTS, N_SLOT, IMG, CACHE_IMG, CACHE_SLICES, CROP_MM, SLICE_BAND, GROUP, N_GROUP
    _saved = {
        "SLOTS": SLOTS, "N_SLOT": N_SLOT, "IMG": IMG, "CACHE_IMG": CACHE_IMG,
        "CACHE_SLICES": CACHE_SLICES, "CROP_MM": CROP_MM, "SLICE_BAND": SLICE_BAND,
        "GROUP": GROUP, "N_GROUP": N_GROUP,
    }
    try:
        SLOTS = list(_V12_RAD_SLOTS)
        N_SLOT = 3
        IMG = CACHE_IMG = 224
        CACHE_SLICES = 8
        GROUP = 1
        N_GROUP = 8
        CROP_MM = 10_000.0
        SLICE_BAND = (0.12, 0.88)

        train = _v12_pd.read_csv(ROOT / "train.csv", dtype={_V12_UID: str})
        test = _v12_pd.read_csv(ROOT / "test.csv", dtype={_V12_UID: str})
        train_series = _v12_pd.read_csv(
            ROOT / "train_series.csv", dtype={_V12_UID: str, "SeriesInstanceUID": str}
        )
        test_series = _v12_pd.read_csv(
            ROOT / "test_series.csv", dtype={_V12_UID: str, "SeriesInstanceUID": str}
        )
        both = _v12_pd.concat([train_series, test_series], ignore_index=True)
        plane_map = dict(zip(both["SeriesInstanceUID"], both["Anatomical_Plane"]))

        gold = train[_V12_TARGETS].notna().all(axis=1).to_numpy()
        # soft labels from three teachers average if available else labels_v3
        try:
            lab = _v12_pd.read_csv(_v12_find_file("labels_v3.csv"), dtype={_V12_UID: str})
            lab = train[[_V12_UID]].merge(lab, on=_V12_UID, how="left")
            y_soft = lab[_V12_TARGETS].apply(_v12_pd.to_numeric, errors="coerce").to_numpy(float)
        except FileNotFoundError:
            # mean of public report labels
            cube = []
            for name in ("report_labels_v2.csv", "llm_labels_v2.csv", "labels_llm_gpt56sol.csv"):
                try:
                    fr = _v12_pd.read_csv(_v12_find_file(name), dtype={_V12_UID: str})
                    fr = train[[_V12_UID]].merge(fr, on=_V12_UID, how="left")
                    cube.append(fr[_V12_TARGETS].apply(_v12_pd.to_numeric, errors="coerce").to_numpy(float))
                except FileNotFoundError:
                    pass
            y_soft = _v12_np.nanmean(_v12_np.stack(cube), axis=0) if cube else None
        if y_soft is None:
            raise RuntimeError("no soft labels")
        y_soft[~_v12_np.isfinite(y_soft)] = 0.5
        y_soft[gold] = train.loc[gold, _V12_TARGETS].to_numpy(float)

        e9_oof, e9_fold = _v12_load_oof_csv("v52_oof.csv", train)
        own_oof, own_fold = _v12_load_oof_csv("v11_own_oof.csv", train)
        fold_id = own_fold if own_fold is not None else e9_fold
        if fold_id is None:
            raise RuntimeError("need fold ids on OOF")

        # per-target soft pick own vs e9
        pick = []
        soft_e9 = {}
        soft_own = {}
        for j, t in enumerate(_V12_TARGETS):
            se = _v12_auc_1d(y_soft[:, j], e9_oof[:, j])
            so = _v12_auc_1d(y_soft[:, j], own_oof[:, j])
            soft_e9[t] = se
            soft_own[t] = so
            # require clear own advantage
            use_own = so == so and se == se and so >= se + 0.005
            pick.append("own" if use_own else "e9")
        audit["soft_e9"] = soft_e9
        audit["soft_own"] = soft_own
        audit["target_pick"] = {t: pick[i] for i, t in enumerate(_V12_TARGETS)}
        n_own = sum(1 for p in pick if p == "own")
        audit["n_targets_own"] = n_own
        _v12_log(f"target picks own={n_own}/12 -> {audit['target_pick']}")

        rad_oof = e9_oof.copy()
        for j, p in enumerate(pick):
            if p == "own":
                rad_oof[:, j] = own_oof[:, j]
        soft_rad = _v12_macro_auc(y_soft, rad_oof)
        soft_e9_m = _v12_macro_auc(y_soft, e9_oof)
        audit["soft_rad_macro"] = soft_rad
        audit["soft_e9_macro"] = soft_e9_m
        _v12_log(f"soft rad_family={soft_rad:.5f} e9={soft_e9_m:.5f}")

        # Gold nested vs V9
        dino_ids, dino_pred = _v12_load_dino_oof()
        dino_index = {u: i for i, u in enumerate(dino_ids)}
        train_index = {str(u): i for i, u in enumerate(train[_V12_UID])}
        gold_uids = train.loc[gold, _V12_UID].astype(str).tolist()
        gold_y_all = train.loc[gold, _V12_TARGETS].to_numpy(float)
        dino_rows, tr_rows, keep_y, gfolds = [], [], [], []
        for uid, gy in zip(gold_uids, gold_y_all):
            if uid not in dino_index or uid not in train_index:
                continue
            dino_rows.append(dino_index[uid])
            tr_rows.append(train_index[uid])
            keep_y.append(gy)
            gfolds.append(int(fold_id[train_index[uid]]))
        gold_y = _v12_np.asarray(keep_y, float)
        dino_rank = _v12_rank(dino_pred[_v12_np.array(dino_rows)])
        e9_rank = _v12_rank(e9_oof[_v12_np.array(tr_rows)])
        rad_rank = _v12_rank(rad_oof[_v12_np.array(tr_rows)])
        gfolds = _v12_np.asarray(gfolds, int)

        v9_parent = _v12_blend_ranks(dino_rank, e9_rank, _V12_BASE_ALPHA)
        v9_score = _v12_macro_auc(gold_y, v9_parent)
        rad_only = _v12_macro_auc(gold_y, rad_rank)
        audit["v9_parent_gold_macro"] = v9_score
        audit["rad_family_gold_macro"] = rad_only
        _v12_log(f"gold V9parent={v9_score:.5f} rad_family={rad_only:.5f}")

        # If no target picked own, rad_family == e9 -> candidate is pure V9 at alpha 0.2
        if n_own == 0:
            audit["status"] = "NO_OWN_TARGETS_KEEP_V9"
            # still write V9 parent for cleanliness below
        nested = _v12_np.zeros_like(dino_rank)
        choices = []
        fold_gains = []
        for outer in sorted(set(gfolds.tolist())):
            tr = gfolds != outer
            va = ~tr
            if int(va.sum()) == 0 or int(tr.sum()) < 8:
                continue
            scored = []
            for alpha in _V12_ALPHA_GRID:
                blend = _v12_blend_ranks(dino_rank[tr], rad_rank[tr], alpha)
                scored.append(_v12_macro_auc(gold_y[tr], blend) - 0.005 * float(alpha))
            best = int(_v12_np.argmax(scored))
            alpha = float(_V12_ALPHA_GRID[best])
            choices.append(alpha)
            nested[va] = _v12_blend_ranks(dino_rank[va], rad_rank[va], alpha)
            fold_gains.append(
                float(
                    _v12_macro_auc(gold_y[va], nested[va])
                    - _v12_macro_auc(gold_y[va], v9_parent[va])
                )
            )
        nested_score = _v12_macro_auc(gold_y, nested)
        final_alpha = float(_v12_np.median(_v12_np.asarray(choices))) if choices else 0.0
        final_oof = _v12_blend_ranks(dino_rank, rad_rank, final_alpha)
        final_score = _v12_macro_auc(gold_y, final_oof)
        positive = int(sum(g > 0 for g in fold_gains))
        supported = bool(
            n_own >= 1
            and final_alpha > 0
            and nested_score >= v9_score + _V12_MIN_GAIN
            and final_score >= v9_score + _V12_MIN_GAIN
            and positive >= max(2, len(fold_gains) // 2)
        )
        audit.update(
            {
                "alpha_choices": choices,
                "final_alpha": final_alpha,
                "nested_gold_macro": nested_score,
                "final_gold_macro": final_score,
                "fold_gains_vs_v9": fold_gains,
                "positive_outer": positive,
                "selection_supported": supported,
            }
        )
        _v12_log(
            f"gate V9={v9_score:.5f} nested={nested_score:.5f} final={final_score:.5f} "
            f"a={final_alpha} pos={positive} supported={supported}"
        )

        # Test encode once
        _v12_log("test FS cache + encode")
        hte = annotate(walk("test_series"))
        slots_te = pick_slots(hte, plane_map)
        st_te, Cte, Mte = build_cache(
            slots_te, plane_map, lat_of(hte, "v12-test "), "v12-test"
        )
        te_by = {str(u): i for i, u in enumerate(st_te)}
        te_order = _v12_np.array([te_by[str(u)] for u in test[_V12_UID]], dtype=_v12_np.int64)
        Cte, Mte = Cte[te_order], Mte[te_order]
        test_feat, test_mask, enc_sha = _v12_encode(Cte, Mte, device)
        del Cte, Mte, hte
        _v12_gc.collect()
        audit["encoder_sha256"] = enc_sha

        e9_states = _v12_load_heads(_v12_find_file("v52_radimagenet_heads.pt"), 3)
        own_states = _v12_load_heads(_v12_find_file("v11_own_heads.pt"), 3)
        all_te = _v12_np.arange(len(test), dtype=_v12_np.int64)

        def mean_predict(states):
            preds = []
            for st in states:
                head = _V12Head().to(device)
                head.load_state_dict(st, strict=True)
                preds.append(_v12_predict(head, test_feat, test_mask, all_te, device))
                del head
                _v12_torch.cuda.empty_cache()
            return _v12_np.mean(_v12_np.stack(preds), axis=0)

        e9_test = mean_predict(e9_states)
        own_test = mean_predict(own_states)
        rad_test = e9_test.copy()
        for j, p in enumerate(pick):
            if p == "own":
                rad_test[:, j] = own_test[:, j]

        dino_test = _v12_pd.read_csv(_V12_DINO, dtype={_V12_UID: str})
        _v12_validate_sub(dino_test, test[_V12_UID])
        dino_r = _v12_rank(dino_test[_V12_TARGETS].to_numpy(float))
        e9_r = _v12_rank(e9_test)
        rad_r = _v12_rank(rad_test)

        v9_frame = dino_test.copy()
        v9_frame[_V12_TARGETS] = _v12_blend_ranks(dino_r, e9_r, _V12_BASE_ALPHA)
        _v12_validate_sub(v9_frame, test[_V12_UID])
        v9_frame.to_csv(_V12_V9, index=False)
        v9_frame.to_csv(_V12_OUT / "submission_v9_parent.csv", index=False)
        _v12_shutil.copy2(_V12_V9, _V12_PRIMARY)
        audit["v9_parent_sha256"] = _v12_sha(_V12_V9)

        if not supported:
            audit["status"] = "OOF_REJECTED_KEEP_V9"
            return audit

        blended = dino_test.copy()
        blended[_V12_TARGETS] = _v12_blend_ranks(dino_r, rad_r, final_alpha)
        _v12_validate_sub(blended, test[_V12_UID])
        out_path = _V12_OUT / f"submission_v12_a{int(round(1000*final_alpha)):03d}.csv"
        blended.to_csv(out_path, index=False)
        _v12_shutil.copy2(out_path, _V12_PRIMARY)
        audit["status"] = "CANDIDATE_SELECTED"
        audit["promote"] = True
        audit["selected_sha256"] = _v12_sha(_V12_PRIMARY)
        audit["selected_path"] = str(out_path)
        _v12_log(f"PROMOTED V12 alpha={final_alpha} own_targets={n_own}")
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

_v12_receipt = {"status": "RAD_NOT_RUN", "promote": False}
try:
    _v12_receipt = run_v12_targetwise()
except Exception as _v12_err:
    traceback.print_exc()
    _v12_receipt = {
        "status": "RAD_ERROR_KEEP_BEST",
        "promote": False,
        "error": f"{type(_v12_err).__name__}: {_v12_err}",
        "traceback": traceback.format_exc()[-3000:],
    }
    try:
        if _V12_V9.is_file():
            _v12_shutil.copy2(_V12_V9, _V12_PRIMARY)
        elif _V12_DINO.is_file():
            _v12_shutil.copy2(_V12_DINO, _V12_PRIMARY)
    except Exception:
        pass

_V12_AUDIT.write_text(
    _v12_json.dumps(_v12_receipt, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
_keys = (
    "status", "promote", "final_alpha", "n_targets_own", "nested_gold_macro",
    "final_gold_macro", "v9_parent_gold_macro", "soft_rad_macro", "error",
)
print("V12 AUDIT", _v12_json.dumps({k: _v12_receipt.get(k) for k in _keys}, sort_keys=True, default=str))
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
        raise RuntimeError("V1 final cell not found")
    cells = cells[:drop_idx] + [code_cell(RAD_ARM)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": nb.get("metadata", {}),
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    meta = {
        "id": "danielsolo1770/rsna-knee-v12-targetwise-gated",
        "title": "rsna-knee-v12-targetwise-gated",
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
            "danielsolo1770/rsna-knee-v11-own-rad-artifacts",
        ],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
        "machine_shape": "NvidiaTeslaT4",
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_NB}")
    print(f"wrote {OUT_META}")


if __name__ == "__main__":
    main()
