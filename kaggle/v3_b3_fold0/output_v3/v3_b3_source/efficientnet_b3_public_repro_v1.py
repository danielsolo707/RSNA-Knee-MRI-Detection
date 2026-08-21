#!/usr/bin/env python3
"""Leakage-controlled EfficientNet-B3 MIL training for RSNA Knee."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TARGETS: Tuple[str, ...] = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)
PLANES: Tuple[str, ...] = ("Sagittal", "Coronal", "Axial")
UID = "StudyInstanceUID"


@dataclass(frozen=True)
class TrainConfig:
    image_size: int = 288
    max_series_slices: int = 64
    train_slices: int = 20
    valid_slices: int = 32
    folds: int = 5
    seed: int = 20260809
    batch_size: int = 2
    workers: int = 4
    backbone_chunk: int = 12
    frozen_epochs: int = 1
    unfrozen_epochs: int = 2
    frozen_lr: float = 8e-4
    unfrozen_lr: float = 8e-5
    weight_decay: float = 1e-4
    gold_weight: float = 3.0
    pseudo_weight_floor: float = 0.20
    backbone: str = "efficientnet_b3"


_DATE = re.compile(
    r"\b(?:19|20)\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b|"
    r"\b\d{1,2}[-/.]\d{1,2}[-/.](?:\d{2}|\d{4})\b"
)
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_SPACE = re.compile(r"\s+")


def normalize_report_for_group(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _DATE.sub(" <date> ", text)
    text = _NUMBER.sub(" <num> ", text)
    return _SPACE.sub(" ", text).strip()


def report_group_id(value: object) -> str:
    return hashlib.sha256(normalize_report_for_group(value).encode()).hexdigest()[:24]


def _indexed_labels(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {UID, *TARGETS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if frame[UID].duplicated().any():
        raise ValueError(f"{name} contains duplicate {UID} rows")
    return frame[[UID, *TARGETS]].copy().set_index(UID).apply(pd.to_numeric, errors="coerce")


def build_public_consensus(
    train: pd.DataFrame,
    pilkwang: pd.DataFrame,
    steven: pd.DataFrame,
    lixin: pd.DataFrame,
    pseudo_weight_floor: float = 0.20,
) -> pd.DataFrame:
    """Equal-source soft labels; weights come only from agreement/certainty."""
    if train[UID].duplicated().any():
        raise ValueError(f"train contains duplicate {UID} rows")
    base = train[[UID, "Report", *TARGETS]].copy().set_index(UID)
    sources = [
        _indexed_labels(pilkwang, "pilkwang"),
        _indexed_labels(steven, "steven"),
        _indexed_labels(lixin, "lixin"),
    ]
    out = base.reset_index()[[UID, "Report"]].copy()
    out["report_group"] = out["Report"].map(report_group_id)
    out = out.drop(columns="Report")
    cube = np.stack([source.reindex(base.index).to_numpy(float) for source in sources])
    if np.nanmin(cube) < 0.0 or np.nanmax(cube) > 1.0:
        raise ValueError("public labels must be in [0, 1]")
    available = np.isfinite(cube).sum(axis=0)
    if np.any(available < 2):
        row, target = np.argwhere(available < 2)[0]
        raise ValueError(f"fewer than two public labels at row={row}, target={TARGETS[target]}")
    consensus = np.nanmean(cube, axis=0)
    disagreement = np.nanmean(np.abs(cube - consensus[None]), axis=0)
    agreement = np.clip(1.0 - 2.0 * disagreement, 0.0, 1.0)
    certainty = np.clip(2.0 * np.abs(consensus - 0.5), 0.0, 1.0)
    weight = pseudo_weight_floor + (1.0 - pseudo_weight_floor) * (
        0.65 * agreement + 0.35 * certainty
    )
    out["is_expert"] = base[list(TARGETS)].notna().all(axis=1).astype(np.int8).to_numpy()
    for j, target in enumerate(TARGETS):
        out[f"pseudo::{target}"] = consensus[:, j].astype(np.float32)
        out[f"weight::{target}"] = weight[:, j].astype(np.float32)
        out[f"gold::{target}"] = pd.to_numeric(base[target], errors="coerce").to_numpy()
    return out


def _group_statistics(label_table: pd.DataFrame) -> pd.DataFrame:
    pseudo_cols = [f"pseudo::{target}" for target in TARGETS]
    gold_cols = [f"gold::{target}" for target in TARGETS]
    records = []
    for group_id, group in label_table.groupby("report_group", sort=True):
        gold = group[gold_cols].to_numpy(float)
        records.append({
            "report_group": group_id,
            "size": len(group),
            "expert_count": int(group["is_expert"].sum()),
            "pseudo_sum": group[pseudo_cols].to_numpy(float).sum(axis=0),
            "gold_positive": np.nansum(gold, axis=0),
            "gold_known": np.isfinite(gold).sum(axis=0).astype(float),
        })
    return pd.DataFrame(records)


def assign_group_balanced_folds(
    label_table: pd.DataFrame, n_splits: int = 5, seed: int = 20260809
) -> pd.Series:
    """Deterministic greedy balance with normalized-report groups kept atomic."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    groups = _group_statistics(label_table)
    target_size = float(groups["size"].sum()) / n_splits
    target_pseudo = np.sum(np.stack(groups["pseudo_sum"]), axis=0) / n_splits
    total_gold_pos = np.sum(np.stack(groups["gold_positive"]), axis=0)
    target_gold_pos = total_gold_pos / n_splits
    target_gold_known = np.sum(np.stack(groups["gold_known"]), axis=0) / n_splits
    rng = np.random.default_rng(seed)
    rarity = 1.0 / np.maximum(total_gold_pos, 1.0)
    priority = [
        100.0 * row["expert_count"]
        + 10.0 * float(np.dot(row["gold_positive"], rarity))
        + math.log1p(row["size"])
        for _, row in groups.iterrows()
    ]
    groups = groups.assign(_priority=priority, _jitter=rng.uniform(0, 1e-6, len(groups)))
    groups = groups.sort_values(["_priority", "_jitter", "report_group"], ascending=[False, False, True])
    fold_size = np.zeros(n_splits)
    fold_pseudo = np.zeros((n_splits, len(TARGETS)))
    fold_gold_pos = np.zeros_like(fold_pseudo)
    fold_gold_known = np.zeros_like(fold_pseudo)
    assignment: Dict[str, int] = {}
    for _, row in groups.iterrows():
        scores = []
        for fold in range(n_splits):
            candidate_size = fold_size.copy()
            candidate_pseudo = fold_pseudo.copy()
            candidate_gold_pos = fold_gold_pos.copy()
            candidate_gold_known = fold_gold_known.copy()
            candidate_size[fold] += row["size"]
            candidate_pseudo[fold] += row["pseudo_sum"]
            candidate_gold_pos[fold] += row["gold_positive"]
            candidate_gold_known[fold] += row["gold_known"]
            size_cost = np.mean(((candidate_size - target_size) / max(target_size, 1.0)) ** 2)
            pseudo_cost = np.mean(((candidate_pseudo - target_pseudo[None]) / np.maximum(target_pseudo[None], 1.0)) ** 2)
            gold_pos_cost = np.mean(((candidate_gold_pos - target_gold_pos[None]) / np.maximum(target_gold_pos[None], 1.0)) ** 2)
            gold_known_cost = np.mean(((candidate_gold_known - target_gold_known[None]) / np.maximum(target_gold_known[None], 1.0)) ** 2)
            scores.append(size_cost + 0.35 * pseudo_cost + 4.0 * gold_pos_cost + 2.0 * gold_known_cost)
        best = int(np.argmin(np.asarray(scores) + 1e-12 * np.arange(n_splits)))
        assignment[str(row["report_group"])] = best
        fold_size[best] += row["size"]
        fold_pseudo[best] += row["pseudo_sum"]
        fold_gold_pos[best] += row["gold_positive"]
        fold_gold_known[best] += row["gold_known"]
    result = label_table["report_group"].map(assignment)
    if result.isna().any():
        raise AssertionError("some report groups were not assigned")
    return result.astype(np.int8)


def pick_one_series_per_plane(series: pd.DataFrame) -> Dict[str, Optional[str]]:
    required = {"SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"}
    missing = required.difference(series.columns)
    if missing:
        raise ValueError(f"series table missing columns: {sorted(missing)}")
    chosen: Dict[str, Optional[str]] = {}
    for plane in PLANES:
        subset = series.loc[series["Anatomical_Plane"].eq(plane)].copy()
        if subset.empty:
            chosen[plane] = None
            continue
        for column in ("Fluid_Sensitive", "Fat_Suppression"):
            subset[column] = pd.to_numeric(subset[column], errors="coerce").fillna(0)
        subset = subset.sort_values(
            ["Fluid_Sensitive", "Fat_Suppression", "SeriesInstanceUID"],
            ascending=[False, False, True],
        )
        chosen[plane] = str(subset.iloc[0]["SeriesInstanceUID"])
    return chosen


def pick_series_slots(series: pd.DataFrame, count: int = 3) -> List[Optional[str]]:
    """Prefer one fluid-sensitive series per plane, then fill empty slots."""
    required = {"SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"}
    missing = required.difference(series.columns)
    if missing:
        raise ValueError(f"series table missing columns: {sorted(missing)}")
    frame = series.copy()
    for column in ("Fluid_Sensitive", "Fat_Suppression"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["_plane"] = frame["Anatomical_Plane"].map({p: i for i, p in enumerate(PLANES)}).fillna(3)
    preferred = frame.loc[frame["Fluid_Sensitive"].eq(1)].sort_values(
        ["_plane", "Fat_Suppression", "SeriesInstanceUID"],
        ascending=[True, False, True],
    )
    selected: List[str] = []
    seen_planes = set()
    for _, row in preferred.iterrows():
        plane = str(row["Anatomical_Plane"])
        if plane not in seen_planes:
            selected.append(str(row["SeriesInstanceUID"]))
            seen_planes.add(plane)
        if len(selected) == count:
            break
    ordered = frame.sort_values(
        ["Fluid_Sensitive", "Fat_Suppression", "_plane", "SeriesInstanceUID"],
        ascending=[False, False, True, True],
    )
    for series_uid in ordered["SeriesInstanceUID"].astype(str):
        if len(selected) == count:
            break
        if series_uid not in selected:
            selected.append(series_uid)
    return selected + [None] * (count - len(selected))


def uniform_slice_indices(length: int, count: int) -> np.ndarray:
    if length <= 0 or count <= 0:
        raise ValueError("length and count must be positive")
    if length <= count:
        return np.concatenate([np.arange(length), np.full(count - length, length - 1)]).astype(np.int64)
    return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)


def stochastic_slice_indices(length: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if length <= count:
        return uniform_slice_indices(length, count)
    edges = np.linspace(0, length, count + 1)
    values = []
    for left, right in zip(edges[:-1], edges[1:]):
        lo = int(math.floor(left))
        hi = max(lo + 1, int(math.ceil(right)))
        values.append(int(rng.integers(lo, min(hi, length))))
    return np.asarray(values, dtype=np.int64)


def pooled_macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, Dict[str, float]]:
    from sklearn.metrics import roc_auc_score
    per_target: Dict[str, float] = {}
    for j, target in enumerate(TARGETS):
        valid = np.isfinite(y_true[:, j]) & np.isfinite(y_score[:, j])
        per_target[target] = (
            float(roc_auc_score(y_true[valid, j], y_score[valid, j]))
            if valid.sum() and np.unique(y_true[valid, j]).size == 2
            else float("nan")
        )
    finite = [value for value in per_target.values() if np.isfinite(value)]
    return (float(np.mean(finite)) if finite else float("nan")), per_target


def prepare_fold_rows(
    label_table: pd.DataFrame, fold: int, config: TrainConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = label_table.loc[label_table["fold"].ne(fold)].copy()
    valid_rows = label_table.loc[label_table["fold"].eq(fold) & label_table["is_expert"].eq(1)].copy()
    if train_rows.empty or valid_rows.empty:
        raise ValueError(f"empty train/validation split for fold {fold}")
    for target in TARGETS:
        gold = pd.to_numeric(train_rows[f"gold::{target}"], errors="coerce")
        is_gold = train_rows["is_expert"].eq(1) & gold.notna()
        train_rows[f"target::{target}"] = train_rows[f"pseudo::{target}"].where(~is_gold, gold)
        train_rows[f"train_weight::{target}"] = train_rows[f"weight::{target}"].where(~is_gold, config.gold_weight)
        valid_rows[f"target::{target}"] = pd.to_numeric(valid_rows[f"gold::{target}"], errors="raise")
        valid_rows[f"train_weight::{target}"] = 1.0
    if set(train_rows["report_group"]).intersection(valid_rows["report_group"]):
        raise AssertionError("normalized report leakage across train and validation")
    return train_rows, valid_rows


def _training_imports():
    try:
        import cv2
        import pydicom
        import timm
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("training requires torch, timm, pydicom, and opencv-python") from exc
    return cv2, pydicom, timm, torch, nn, DataLoader, Dataset


def load_dicom_volume(series_dir: Path, image_size: int, max_slices: int = 64) -> np.ndarray:
    cv2, pydicom, _, _, _, _, _ = _training_imports()
    slices: List[Tuple[int, np.ndarray]] = []
    for path in sorted(series_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(path))
            image = ds.pixel_array.astype(np.float32)
            image = image * float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            image += float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
                image = image.max() - image
            slices.append((int(getattr(ds, "InstanceNumber", 0) or 0), image))
        except Exception:
            continue
    if not slices:
        raise RuntimeError(f"no decodable slices in {series_dir}")
    slices.sort(key=lambda item: item[0])
    volume = np.stack([item[1] for item in slices])
    lo, hi = np.percentile(volume, [1.0, 99.0])
    hi = max(float(hi), float(lo) + 1.0)
    volume = np.clip((volume - lo) / (hi - lo), 0.0, 1.0)
    if len(volume) > max_slices:
        start = (len(volume) - max_slices) // 2
        volume = volume[start:start + max_slices]
    resized = np.stack([
        cv2.resize(x, (image_size, image_size), interpolation=cv2.INTER_AREA)
        for x in volume
    ])
    # Match the public .903 pipeline's training-time uint8 ETL boundary.
    return (resized * 255.0).astype(np.uint8).astype(np.float32) / 255.0


def make_dataset_class():
    _, _, _, torch, _, _, Dataset = _training_imports()
    class KneeStudyDataset(Dataset):
        def __init__(self, rows, series, image_root, config, training, seed):
            self.rows = rows.reset_index(drop=True)
            self.series = {str(uid): group.copy() for uid, group in series.groupby(UID, sort=False)}
            self.image_root, self.config, self.training, self.seed = Path(image_root), config, training, seed

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row, config = self.rows.iloc[index], self.config
            uid = str(row[UID])
            count = config.train_slices if self.training else config.valid_slices
            # DataLoader seeds each worker deterministically.  Drawing once from
            # its advancing torch RNG gives a fresh augmentation on every visit
            # without sacrificing run-level reproducibility.
            visit_seed = int(torch.randint(0, 2**31 - 1, (1,)).item()) if self.training else 0
            rng = np.random.default_rng(self.seed + 1_000_003 * index + visit_seed)
            empty = pd.DataFrame(columns=["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"])
            chosen = pick_series_slots(self.series.get(uid, empty), len(PLANES))
            planes, masks = [], []
            for series_uid in chosen:
                if series_uid is None:
                    planes.append(np.zeros((count, config.image_size, config.image_size), np.float32))
                    masks.append(np.zeros(count, bool))
                    continue
                volume = load_dicom_volume(
                    self.image_root / uid / series_uid,
                    config.image_size,
                    config.max_series_slices,
                )
                idx = stochastic_slice_indices(len(volume), count, rng) if self.training else uniform_slice_indices(len(volume), count)
                sampled = volume[idx].astype(np.float32)
                if self.training:
                    sampled = np.clip(sampled * rng.uniform(0.85, 1.15), 0, 1) ** rng.uniform(0.85, 1.15)
                    if rng.random() < 0.5:
                        sampled = sampled[:, :, ::-1].copy()
                sampled = (sampled - 0.449) / 0.226
                planes.append(sampled)
                masks.append(np.ones(count, bool))
            return {
                "uid": uid,
                "images": torch.from_numpy(np.stack(planes)[:, :, None]),
                "mask": torch.from_numpy(np.stack(masks)),
                "target": torch.tensor([row[f"target::{x}"] for x in TARGETS], dtype=torch.float32),
                "weight": torch.tensor([row[f"train_weight::{x}"] for x in TARGETS], dtype=torch.float32),
            }
    return KneeStudyDataset


def make_model_class():
    _, _, timm, torch, nn, _, _ = _training_imports()
    class KneeMILModel(nn.Module):
        def __init__(self, backbone="efficientnet_b3", checkpoint=None, backbone_chunk=12):
            super().__init__()
            self.backbone_chunk = int(backbone_chunk)
            self.backbone = timm.create_model(backbone, pretrained=False, in_chans=1, num_classes=0, global_pool="avg")
            if checkpoint:
                state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
                if isinstance(state, Mapping) and "state_dict" in state:
                    state = state["state_dict"]
                state = {str(k).replace("module.", "", 1): v for k, v in state.items()}
                expected = self.backbone.state_dict()
                if "conv_stem.weight" in state and state["conv_stem.weight"].shape[1] == 3:
                    state["conv_stem.weight"] = state["conv_stem.weight"].sum(dim=1, keepdim=True)
                state = {key: value for key, value in state.items() if key in expected}
                shape_errors = {
                    key: (tuple(value.shape), tuple(expected[key].shape))
                    for key, value in state.items()
                    if value.shape != expected[key].shape
                }
                if shape_errors:
                    raise RuntimeError(f"backbone tensor shape mismatch: {shape_errors}")
                result = self.backbone.load_state_dict(state, strict=True)
                if result.missing_keys or result.unexpected_keys:
                    raise RuntimeError(f"backbone state mismatch: {result}")
            features, hidden = int(self.backbone.num_features), 384
            self.attn_v, self.attn_u = nn.Linear(features, hidden), nn.Linear(features, hidden)
            self.attn_out = nn.Linear(hidden, len(TARGETS))
            self.target_weight = nn.Parameter(torch.empty(len(TARGETS), features))
            self.target_bias = nn.Parameter(torch.zeros(len(TARGETS)))
            self.slice_head = nn.Linear(features, len(TARGETS))
            self.mix_logit = nn.Parameter(torch.tensor(-1.1))
            nn.init.xavier_uniform_(self.target_weight)

        def set_backbone_trainable(self, value):
            for parameter in self.backbone.parameters():
                parameter.requires_grad = value

        def forward(self, images, mask):
            batch, planes, slices, channels, height, width = images.shape
            flat = images.reshape(batch * planes * slices, channels, height, width)
            features = []
            train_backbone = self.training and any(p.requires_grad for p in self.backbone.parameters())
            for start in range(0, len(flat), self.backbone_chunk):
                chunk = flat[start:start + self.backbone_chunk]
                if train_backbone:
                    from torch.utils.checkpoint import checkpoint
                    features.append(checkpoint(self.backbone, chunk, use_reentrant=False))
                else:
                    features.append(self.backbone(chunk))
            features = torch.cat(features, dim=0)
            features = features.reshape(batch, planes * slices, -1)
            flat_mask = mask.reshape(batch, planes * slices)
            gated = torch.tanh(self.attn_v(features)) * torch.sigmoid(self.attn_u(features))
            attention = self.attn_out(gated).masked_fill(~flat_mask[:, :, None], -1e4).softmax(dim=1)
            pooled = torch.einsum("bnt,bnf->btf", attention, features)
            attention_logits = torch.einsum("btf,tf->bt", pooled, self.target_weight) + self.target_bias
            max_logits = self.slice_head(features).masked_fill(~flat_mask[:, :, None], -1e4).max(dim=1).values
            mix = torch.sigmoid(self.mix_logit)
            return (1.0 - mix) * attention_logits + mix * max_logits
    return KneeMILModel


def train_one_fold(label_table, series, image_root, output_dir, fold, backbone_checkpoint, config):
    _, _, _, torch, nn, DataLoader, _ = _training_imports()
    random.seed(config.seed + fold)
    np.random.seed(config.seed + fold)
    torch.manual_seed(config.seed + fold)
    Dataset, Model = make_dataset_class(), make_model_class()
    train_rows, valid_rows = prepare_fold_rows(label_table, fold, config)
    train_loader = DataLoader(Dataset(train_rows, series, image_root, config, True, config.seed + fold), batch_size=config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=True, persistent_workers=config.workers > 0)
    valid_loader = DataLoader(Dataset(valid_rows, series, image_root, config, False, config.seed + fold), batch_size=1, shuffle=False, num_workers=max(1, config.workers // 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU is required")
    core_model = Model(config.backbone, backbone_checkpoint, config.backbone_chunk).to(device)
    model = nn.DataParallel(core_model) if torch.cuda.device_count() > 1 else core_model
    loss_fn, scaler, history = nn.BCEWithLogitsLoss(reduction="none"), torch.amp.GradScaler("cuda"), []
    schedule = [(False, config.frozen_lr)] * config.frozen_epochs + [(True, config.unfrozen_lr)] * config.unfrozen_epochs
    optimizer, active_phase = None, None
    for epoch, (unfreeze, lr) in enumerate(schedule):
        if active_phase != unfreeze:
            core_model.set_backbone_trainable(unfreeze)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=config.weight_decay,
            )
            active_phase = unfreeze
        assert optimizer is not None
        model.train()
        for layer in core_model.backbone.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        losses = []
        for batch in train_loader:
            images, mask = batch["images"].to(device), batch["mask"].to(device)
            target, weight = batch["target"].to(device), batch["weight"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                raw = loss_fn(model(images, mask), target)
                loss = (raw * weight).sum() / weight.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "backbone_trainable": unfreeze,
            "learning_rate": lr,
        })
    model.eval()
    uids, truth, score = [], [], []
    with torch.no_grad():
        for batch in valid_loader:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(batch["images"].to(device), batch["mask"].to(device))
            uids.extend(batch["uid"])
            truth.append(batch["target"].numpy())
            score.append(torch.sigmoid(logits.float()).cpu().numpy())
    truth, score = np.concatenate(truth), np.concatenate(score)
    macro, per_target = pooled_macro_auc(truth, score)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame({UID: uids})
    for j, target in enumerate(TARGETS):
        oof[f"gold::{target}"], oof[f"pred::{target}"] = truth[:, j], score[:, j]
    oof.to_csv(output_dir / f"fold{fold}_expert_oof.csv", index=False)
    destination = output_dir / f"fold{fold}_final.pt"
    torch.save({"model": core_model.state_dict(), "backbone": config.backbone, "config": asdict(config), "fold": fold, "fold_diagnostic_macro_auc": macro, "fold_diagnostic_per_target_auc": per_target, "history": history, "visible_gpus": torch.cuda.device_count(), "backbone_checkpoint_sha256": hashlib.sha256(backbone_checkpoint.read_bytes()).hexdigest()}, destination)
    return destination


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train-csv", "series-csv", "pilkwang-labels", "steven-labels", "lixin-labels", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    config = TrainConfig()
    paths = [args.train_csv, args.series_csv, args.pilkwang_labels, args.steven_labels, args.lixin_labels]
    frames = [pd.read_csv(path) for path in paths]
    labels = build_public_consensus(frames[0], frames[2], frames[3], frames[4], config.pseudo_weight_floor)
    labels["fold"] = assign_group_balanced_folds(labels, config.folds, config.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output_dir / "fold_labels.csv", index=False)
    manifest = {
        "config": asdict(config), "rows": len(labels), "expert_rows": int(labels["is_expert"].sum()),
        "fold_counts": labels["fold"].value_counts().sort_index().to_dict(),
        "fold_expert_counts": labels.loc[labels["is_expert"].eq(1), "fold"].value_counts().sort_index().to_dict(),
        "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.prepare_only:
        return
    if args.fold is None or args.image_root is None or args.backbone_checkpoint is None:
        parser.error("training requires --fold, --image-root, and --backbone-checkpoint")
    if not 0 <= args.fold < config.folds:
        parser.error(f"--fold must be in [0, {config.folds - 1}]")
    print("saved", train_one_fold(labels, frames[1], args.image_root, args.output_dir, args.fold, args.backbone_checkpoint, config))


if __name__ == "__main__":
    main()
