#!/usr/bin/env python3
"""Build the V15 5-fold student OOF kernel (T4 GPU).

Fixed pre-registered protocol (docs/v15_student_5fold_plan.md):
- Bridge volumes from the private v15-bridge dataset (chunked npz, SHA-pinned manifest).
- Teacher targets = 24 cols [12 DINO OOF | 12 V52 rad OOF], SHA-pinned inputs.
- Folds = v52_oof.csv 'fold' column (pinned), gold = is_gold (58 rows).
- For fold f: fit only on rows with fold != f, gold rows excluded from fit
  (source-held-out for the fold that owns them); predict fold-f rows OOF.
- 2 seeds x 12 epochs, exact training_source.py hyperparameters.
- Gates (V14_INVESTIGATION.md §5): G1 full-OOF macro delta vs V10 >= 0,
  G2 >= 3/5 folds positive, G3 no target collapse > 0.02, G4 rank corr < 0.98.
- No competition submission in-kernel. Writes v15_oof.csv + v15_audit.json.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

UID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
PLANES = ["AXIAL", "CORONAL", "SAGITTAL"]
DEPTH_INDICES = np.asarray([0, 2, 4, 6, 9, 11, 13, 15], dtype=np.int64)
SEEDS = [20260873, 20260897]
EPOCHS = 12
BATCH = 8
EVAL_BATCH = 12

# SHA-pinned teacher inputs (local verified copies).
DINO_OOF_SHA = "62d47ba4c0c8347b5b24e7fd2aa517aae0fd6d4656fd5d829fd3a50f0159909c"
V52_OOF_SHA = "f5b1f02029af98740565bec5b6a76d8471b4b6a6039d09ea410368230cb1f03f"

# V10 parent recipe (from v10_rad_audit.json): rank blend DINO + rad,
# per-target rad weight: 0.20 all, ACL/Synovitis 0.505.
V10_FINAL_ALPHAS = {
    "ACL": 0.505, "MCL": 0.20, "Medial Meniscus": 0.20,
    "Lateral Meniscus": 0.20, "Medial OA": 0.20, "Lateral OA": 0.20,
    "PF OA": 0.20, "Effusion": 0.20, "Synovitis": 0.505, "Baker's": 0.20,
    "Contusion": 0.20, "Fracture": 0.20,
}

T0 = time.time()
ALIVE_LIMIT = 4.5 * 3600.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def deadline(where):
    elapsed = time.time() - T0
    if elapsed > ALIVE_LIMIT:
        raise TimeoutError(f"V15 deadline exceeded at {where}: {elapsed:.1f}s")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def locate_file(name, expected_sha=None, skip_dirs=("train_series", "test_series")):
    matches = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        if name in files:
            matches.append(Path(root) / name)
    if expected_sha is not None:
        matches = [p for p in matches if sha256_file(p) == expected_sha]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {name} hash={str(expected_sha)[:16]}, found={matches}"
        )
    return matches[0]


def rank_pct(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    result = np.zeros_like(matrix, dtype=float)
    for j in range(result.shape[1]):
        result[:, j] = pd.Series(matrix[:, j]).rank(method="average", pct=True).to_numpy()
    return result


def auc_1d(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    hard = (y >= 0.5).astype(float)
    pos = hard == 1
    neg = hard == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_auc(y, pred):
    vals = [auc_1d(y[:, j], pred[:, j]) for j in range(y.shape[1])]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def make_targets(train, locate):
    """Exact V10 recipe (_v10_make_targets): weighted mean of 3 LLM label CSVs,
    gold overwritten with expert labels, gold weight 3.0."""
    sources = []
    for name in ("report_labels_v2.csv", "llm_labels_v2.csv", "labels_llm_gpt56sol.csv"):
        try:
            sources.append(pd.read_csv(locate(name)))
        except (FileNotFoundError, RuntimeError):
            log(f"label source missing: {name}")
    if not sources:
        raise RuntimeError("no report-label CSV attached")
    cube = []
    for frame in sources:
        frame = frame.copy()
        if frame[UID].duplicated().any():
            frame = frame.drop_duplicates(UID, keep="first")
        if not all(t in frame.columns for t in TARGETS):
            raise RuntimeError(f"label CSV missing targets: {frame.columns.tolist()[:20]}")
        aligned = train[[UID]].merge(frame[[UID] + TARGETS], on=UID, how="left")
        cube.append(aligned[TARGETS].to_numpy(float))
    cube = np.stack(cube)
    y = np.nanmean(cube, axis=0).astype(np.float32)
    with np.errstate(all="ignore"):
        disagreement = np.nanmean(np.abs(cube - y[None]), axis=0)
    agreement = np.clip(1.0 - 2.0 * disagreement, 0, 1)
    certainty = np.clip(2.0 * np.abs(y - 0.5), 0, 1)
    w = (0.15 + 0.85 * (0.65 * agreement + 0.35 * certainty)).astype(np.float32)
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    y[gold] = train.loc[gold, TARGETS].to_numpy(np.float32)
    w[gold] = 3.0
    nan = ~np.isfinite(y)
    y[nan] = 0.5
    w[nan] = 0.05
    return y, w, gold


def is_validation(uid, fold):
    digest = hashlib.sha256(f"resnet18-distill-v1:f{fold}:{uid}".encode()).hexdigest()
    return int(digest[:8], 16) % 10 == 0


class BridgeDataset(Dataset):
    """Reads chunked bridge volumes via mmap; study = absolute row index 0..4407."""

    def __init__(self, rows, manifest, chunk_maps, presence, target, augment=False):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.augment = bool(augment)
        self.starts = [int(c["study_start"]) for c in manifest["chunks"]]
        self.stops = [int(c["study_stop"]) for c in manifest["chunks"]]
        self.maps = chunk_maps
        self.presence = presence
        self.target = target

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        study = int(self.rows[index])
        chunk = bisect.bisect_right(self.starts, study) - 1
        if chunk < 0 or study >= self.stops[chunk]:
            raise RuntimeError(f"study/chunk routing failure: {study}")
        volume = np.asarray(
            self.maps[chunk][study - self.starts[chunk]], dtype=np.uint8
        ).copy()
        if self.augment:
            if torch.rand(()) < 0.5:
                volume = volume[:, ::-1].copy()
            if torch.rand(()) < 0.5:
                volume = volume[:, :, :, ::-1].copy()
        return (
            torch.from_numpy(volume),
            torch.from_numpy(self.presence[study].astype(np.float32)),
            torch.from_numpy(self.target[study]),
            torch.tensor(study, dtype=torch.long),
        )


class R18Student(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        old_weight = base.conv1.weight.detach().mean(dim=1, keepdim=True)
        base.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            base.conv1.weight.copy_(old_weight)
        self.encoder = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.register_buffer("mean", torch.tensor([0.449]).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.226]).view(1, 1, 1, 1))
        descriptor = 3 * 3 * 512 + 3
        self.trunk = nn.Sequential(
            nn.Linear(descriptor, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(1536, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.public_head = nn.Linear(384, 12)
        self.rad_head = nn.Linear(384, 12)

    def forward(self, volume, presence):
        batch, planes, depth, height, width = volume.shape
        value = volume.float().mul_(1.0 / 255.0).reshape(
            batch * planes * depth, 1, height, width
        )
        value = F.interpolate(value, size=(160, 160), mode="bilinear", align_corners=False)
        value = (value - self.mean) / self.std
        value = self.encoder(value).mean((2, 3)).reshape(batch, planes, depth, 512)
        descriptor = torch.cat(
            [value.mean(2), value.amax(2), value.std(2, unbiased=False)], dim=2
        )
        descriptor = descriptor * presence[:, :, None]
        hidden = self.trunk(torch.cat([descriptor.reshape(batch, -1), presence], dim=1))
        return torch.cat([self.public_head(hidden), self.rad_head(hidden)], dim=1)


def predict_loader(model, loader, device):
    model.eval()
    out = np.zeros((len(loader.dataset), 24), dtype=np.float32)
    starts = np.zeros(len(loader.dataset), dtype=np.int64)
    pos = 0
    with torch.inference_mode():
        for volume, presence, y, study in loader:
            n = len(study)
            batch = torch.sigmoid(
                model(
                    volume.to(device, non_blocking=True),
                    presence.to(device, non_blocking=True),
                )
            ).float().cpu().numpy()
            out[pos:pos + n] = batch
            starts[pos:pos + n] = study.numpy()
            pos += n
    return out, starts


def train_fold(
    train_loader, valid_loader, device, seed, epochs, label
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = R18Student().to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 8.0e-5},
        {"params": model.trunk.parameters(), "lr": 3.0e-4},
        {"params": model.public_head.parameters(), "lr": 4.0e-4},
        {"params": model.rad_head.parameters(), "lr": 4.0e-4},
    ], weight_decay=0.025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=8.0e-6
    )
    best_score = -float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        losses = []
        for volume, presence, y, study in train_loader:
            volume = volume.to(device, non_blocking=True)
            presence = presence.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(volume, presence)
                probability = torch.sigmoid(logits)
                regression = F.smooth_l1_loss(probability, y, beta=0.06)
                perm = torch.randperm(len(y), device=device)
                delta = y - y[perm]
                mask = delta.abs() > 0.06
                pairwise = F.softplus(
                    -torch.sign(delta) * (logits - logits[perm])
                )
                pairwise = (pairwise * delta.abs() * mask).sum() / (
                    (delta.abs() * mask).sum() + 1.0e-6
                )
                loss = regression + 0.10 * pairwise
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()

        model.eval()
        valid_pred, valid_target = [], []
        with torch.inference_mode():
            for volume, presence, y, study in valid_loader:
                with torch.autocast("cuda", dtype=torch.float16):
                    batch_pred = torch.sigmoid(model(
                        volume.to(device, non_blocking=True),
                        presence.to(device, non_blocking=True),
                    )).float().cpu().numpy()
                valid_pred.append(batch_pred)
                valid_target.append(y.numpy())
        valid_pred = np.concatenate(valid_pred)
        valid_target = np.concatenate(valid_target)
        correlations = np.asarray([
            np.corrcoef(valid_pred[:, j], valid_target[:, j])[0, 1]
            for j in range(24)
        ], dtype=np.float64)
        mse = float(np.mean((valid_pred - valid_target) ** 2))
        score = float(np.nanmean(correlations) - 0.50 * mse)
        log(f"{label} seed={seed} epoch={epoch} loss={np.mean(losses):.4f} "
            f"score={score:.4f} val_public={np.nanmean(correlations[:12]):.4f}")
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"{label} seed {seed} produced no checkpoint")
    model.load_state_dict(best_state)
    return model, best_score


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all 4407")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--train-csv", type=Path, default=None)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("V15 requires CUDA")

    # --- inputs ---
    dino_path = locate_file("oof.npz", DINO_OOF_SHA)
    v52_path = locate_file("v52_oof.csv", V52_OOF_SHA)
    bridge_manifest = locate_file("v15_bridge_manifest.json")
    manifest = json.loads(bridge_manifest.read_text())
    chunk_dir = bridge_manifest.parent

    competition = Path("/kaggle/input/rsna-knee-abnormality-detection")
    if not (competition / "train.csv").is_file():
        competition = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
    train = pd.read_csv(
        args.train_csv if args.train_csv else (competition / "train.csv"),
        dtype={UID: str},
    )
    if args.train_csv is None and (len(train) != 4407 or not train[UID].is_unique):
        raise RuntimeError("train identity contract failed")
    if args.limit > 0:
        train = train.iloc[: args.limit].reset_index(drop=True)
        log(f"limit={args.limit} -> {len(train)} studies (local test mode)")
    n_total = len(train)

    if manifest.get("recipe") != "V55-bridge":
        raise RuntimeError(f"bridge recipe drift: {manifest.get('recipe')}")
    if int(manifest.get("studies", 0)) != n_total:
        raise RuntimeError(f"bridge studies drift: {manifest.get('studies')}")
    if int(manifest["chunks"][-1]["study_stop"]) != n_total:
        raise RuntimeError("bridge does not cover all studies")

    # --- teacher targets, aligned to train row order ---
    dino = np.load(dino_path, allow_pickle=False)
    dino_ids = [str(x) for x in dino["ids"]]
    dino_index = {u: i for i, u in enumerate(dino_ids)}
    v52 = pd.read_csv(v52_path, dtype={UID: str})
    v52_index = {u: i for i, u in enumerate(v52[UID].astype(str))}
    if len(v52) != len(train):
        raise RuntimeError(f"v52 row count drift: {len(v52)} vs {len(train)}")

    dino_pred = np.zeros((n_total, 12), dtype=np.float32)
    rad_oof = np.zeros((n_total, 12), dtype=np.float32)
    fold_id = np.full(n_total, -1, dtype=np.int8)
    is_gold = np.zeros(n_total, dtype=bool)
    for row, uid in enumerate(train[UID].astype(str)):
        if uid in dino_index:
            dino_pred[row] = dino["pred"][dino_index[uid]]
        if uid in v52_index:
            rad_oof[row] = v52.loc[v52_index[uid], TARGETS].to_numpy(np.float32)
            fold_id[row] = int(v52["fold"].iloc[v52_index[uid]])
            is_gold[row] = bool(int(v52["is_gold"].iloc[v52_index[uid]]))
    teacher = np.concatenate([dino_pred, rad_oof], axis=1).astype(np.float32)
    if teacher.shape != (n_total, 24):
        raise RuntimeError(f"teacher shape failed: {teacher.shape}")
    if (fold_id == -1).any():
        raise RuntimeError("unassigned fold rows")

    # V10-identical soft labels (weighted mean of 3 LLM sources + expert gold).
    soft_y_cont, soft_w, gold_from_labels = make_targets(train, locate_file)
    if not np.array_equal(gold_from_labels, is_gold):
        raise RuntimeError("gold mask drift vs v52 is_gold")
    soft_y = (soft_y_cont >= 0.5).astype(float)

    gold_y = train.loc[is_gold, TARGETS].to_numpy(np.float64)
    gold_rows = np.flatnonzero(is_gold)
    gold_folds = fold_id[gold_rows]
    log(f"teacher ok dino={DINO_OOF_SHA[:12]} v52={V52_OOF_SHA[:12]} "
        f"gold={len(gold_rows)} gold_per_fold="
        f"{np.bincount(gold_folds, minlength=5).tolist()}")

    # --- V10 parent OOF (exact recipe) ---
    alphas = np.asarray([V10_FINAL_ALPHAS[t] for t in TARGETS], dtype=np.float64)
    parent_oof = (1.0 - alphas) * rank_pct(dino_pred) + alphas * rank_pct(rad_oof)

    # --- bridge chunks (mmap) ---
    chunk_maps = []
    for row in manifest["chunks"]:
        volume_path = chunk_dir / row["volume_file"]
        observed = sha256_file(volume_path)
        if observed != row["volume_sha256"]:
            raise RuntimeError(
                f"bridge chunk hash drift {volume_path.name}: {observed[:16]}"
            )
        chunk_maps.append(np.load(volume_path, mmap_mode="r", allow_pickle=False))

    presence = np.zeros((n_total, 3), dtype=np.float32)
    for row in manifest["chunks"]:
        study_start = int(row["study_start"])
        study_stop = int(row["study_stop"])
        presence_path = chunk_dir / row["presence_file"]
        observed = sha256_file(presence_path)
        if observed != row["presence_sha256"]:
            raise RuntimeError(
                f"bridge presence hash drift {presence_path.name}: {observed[:16]}"
            )
        presence[study_start:study_stop] = np.load(presence_path, allow_pickle=False)

    # --- folds ---
    all_rows = np.arange(n_total)
    oof_student = np.zeros((n_total, 24), dtype=np.float32)
    fold_receipts = []
    for fold in range(5):
        deadline(f"fold{fold}_start")
        fold_mask = fold_id == fold
        # Source-held-out: gold rows are excluded from EVERY fit.
        fit_rows = all_rows[~fold_mask & ~is_gold]
        pred_rows = all_rows[fold_mask]
        valid_mask = np.asarray(
            [is_validation(UID_OF(train, row), fold) for row in fit_rows], dtype=bool
        )
        tr_rows = fit_rows[~valid_mask]
        va_rows = fit_rows[valid_mask]
        log(f"fold {fold}: fit={len(fit_rows)} train={len(tr_rows)} "
            f"valid={len(va_rows)} oof={len(pred_rows)}")

        train_loader = DataLoader(
            BridgeDataset(tr_rows, manifest, chunk_maps, presence, teacher, augment=True),
            batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
        )
        valid_loader = DataLoader(
            BridgeDataset(va_rows, manifest, chunk_maps, presence, teacher, augment=False),
            batch_size=EVAL_BATCH, shuffle=False, num_workers=0, pin_memory=True,
        )

        fold_preds = []
        fold_scores = []
        for seed in SEEDS:
            model, score = train_fold(
                train_loader, valid_loader, device, seed, EPOCHS,
                f"f{fold}",
            )
            pred_loader = DataLoader(
                BridgeDataset(pred_rows, manifest, chunk_maps, presence, teacher, augment=False),
                batch_size=EVAL_BATCH, shuffle=False, num_workers=0, pin_memory=True,
            )
            pred, starts = predict_loader(model, pred_loader, device)
            order = np.argsort(starts, kind="stable")
            fold_preds.append(pred[order])
            fold_scores.append(score)
            del model
            torch.cuda.empty_cache()
        oof_student[pred_rows] = np.mean(fold_preds, axis=0)
        fold_receipts.append({
            "fold": fold,
            "fit_rows": int(len(fit_rows)),
            "train_rows": int(len(tr_rows)),
            "valid_rows": int(len(va_rows)),
            "oof_rows": int(len(pred_rows)),
            "seed_scores": fold_scores,
        })
        log(f"fold {fold} complete scores={[f'{s:.4f}' for s in fold_scores]}")

    # --- deployment OOF: 0.80 public + 0.20 rad ranks ---
    public_rank = rank_pct(oof_student[:, :12])
    rad_rank = rank_pct(oof_student[:, 12:])
    student_oof = 0.80 * public_rank + 0.20 * rad_rank

    # --- gates ---
    parent_macro = macro_auc(soft_y, parent_oof)
    student_macro = macro_auc(soft_y, student_oof)
    delta = student_macro - parent_macro

    fold_deltas = []
    for fold in range(5):
        fold_mask = fold_id == fold
        p = macro_auc(soft_y[fold_mask], parent_oof[fold_mask])
        s = macro_auc(soft_y[fold_mask], student_oof[fold_mask])
        fold_deltas.append(float(s - p))
    positive_folds = int(sum(d > 0 for d in fold_deltas))

    per_target = {}
    collapse_targets = []
    for j, t in enumerate(TARGETS):
        p = auc_1d(soft_y[:, j], parent_oof[:, j])
        s = auc_1d(soft_y[:, j], student_oof[:, j])
        per_target[t] = {"parent_auc": p, "student_auc": s, "delta": s - p}
        if s == s and p == p and (p - s) > 0.02:
            collapse_targets.append(t)

    rank_corr = float(np.mean([
        np.corrcoef(
            rank_pct(student_oof[:, j]).ravel(), rank_pct(parent_oof[:, j]).ravel()
        )[0, 1]
        for j in range(12)
    ]))

    gold_student = macro_auc(gold_y, student_oof[gold_rows])
    gold_parent = macro_auc(gold_y, parent_oof[gold_rows])
    gold_blend = macro_auc(gold_y, 0.50 * rank_pct(student_oof[gold_rows]) + 0.50 * rank_pct(parent_oof[gold_rows]))

    gates = {
        "G1_full_oof_macro_delta_vs_v10": float(delta),
        "G1_pass": bool(delta >= 0.0),
        "G2_positive_folds": positive_folds,
        "G2_pass": bool(positive_folds >= 3),
        "G3_collapse_targets": collapse_targets,
        "G3_pass": len(collapse_targets) == 0,
        "G4_rank_corr_vs_v10": rank_corr,
        "G4_pass": bool(rank_corr < 0.98),
    }
    all_pass = bool(
        gates["G1_pass"] and gates["G2_pass"] and gates["G3_pass"] and gates["G4_pass"]
    )

    # --- write outputs ---
    oof_df = pd.DataFrame(student_oof, columns=TARGETS)
    oof_df.insert(0, UID, train[UID].astype(str).to_numpy())
    oof_path = out / "v15_oof.csv"
    if oof_path.exists():
        raise RuntimeError(f"refusing to replace {oof_path}")
    oof_df.to_csv(oof_path, index=False)

    oof24_df = pd.DataFrame(oof_student, columns=TARGETS + [t + "_rad" for t in TARGETS])
    oof24_df.insert(0, UID, train[UID].astype(str).to_numpy())
    oof24_path = out / "v15_oof_24.csv"
    oof24_df.to_csv(oof24_path, index=False)

    soft_df = pd.DataFrame(soft_y_cont, columns=TARGETS)
    soft_df.insert(0, UID, train[UID].astype(str).to_numpy())
    soft_df.insert(1, "w", soft_w.mean(axis=1).astype(np.float32))
    soft_df.insert(2, "is_gold", is_gold.astype(np.int8))
    soft_path = out / "v15_soft_labels.csv"
    soft_df.to_csv(soft_path, index=False)

    audit = {
        "status": "V15_5FOLD_OOF_COMPLETE",
        "candidate": "0.50 rank(V10) + 0.50 rank(student), student OOF from 5-fold source-held-out",
        "inputs": {
            "dino_oof_sha256": DINO_OOF_SHA,
            "v52_oof_sha256": V52_OOF_SHA,
            "bridge_manifest_sha256": sha256_file(bridge_manifest),
            "bridge_recipe": manifest["recipe"],
            "bridge_stored_depth": manifest.get("stored_depth"),
        },
        "protocol": {
            "folds": "v52_oof.csv fold column",
            "gold_rows": int(is_gold.sum()),
            "gold_per_fold": np.bincount(gold_folds, minlength=5).tolist(),
            "gold_excluded_from_every_fit": True,
            "seeds": SEEDS,
            "epochs": EPOCHS,
        },
        "gates": gates,
        "all_gates_pass": all_pass,
        "evidence": {
            "full_oof_soft_macro_parent_v10": parent_macro,
            "full_oof_soft_macro_student": student_macro,
            "gold58_macro_parent_v10": gold_parent,
            "gold58_macro_student": gold_student,
            "gold58_macro_50_50_blend": gold_blend,
            "fold_macro_deltas_vs_v10": fold_deltas,
            "per_target": per_target,
            "rank_corr_vs_v10": rank_corr,
        },
        "fold_receipts": fold_receipts,
        "oof_sha256": sha256_file(oof_path),
        "oof24_sha256": sha256_file(oof24_path),
        "soft_labels_sha256": sha256_file(soft_path),
        "submit_allowed": False,
        "elapsed_seconds": round(time.time() - T0, 3),
    }
    audit_path = out / "v15_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log("V15_5FOLD_OOF_COMPLETE " + json.dumps(
        {k: audit[k] for k in ("status", "gates", "all_gates_pass", "evidence", "oof_sha256")},
        sort_keys=True, default=str,
    ))
    return 0


def UID_OF(train, row):
    return str(train.iloc[row][UID])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:
        traceback.print_exc()
        Path("/kaggle/working/v15_audit.json").write_text(
            json.dumps({
                "status": "V15_ERROR_NO_OOF",
                "error": f"{type(err).__name__}: {err}",
                "traceback": traceback.format_exc()[-4000:],
                "submit_allowed": False,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise