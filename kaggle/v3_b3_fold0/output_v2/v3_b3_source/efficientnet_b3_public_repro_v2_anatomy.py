#!/usr/bin/env python3
"""Anatomy-aware EfficientNet-B3 MIL training for RSNA Knee.

This module deliberately reuses the audited v1 DICOM and label pipeline while
changing only three trainable assumptions that v1 could not express:

* each slice feature receives its acquisition-plane and normalized stack position;
* expert-labelled studies are sampled often enough to influence every epoch;
* fine-tuning is restricted to the final two EfficientNet stages.

The output checkpoint contract remains compatible with the v1 inference runner,
provided this v2 module is supplied to ``--module`` at inference time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _load_base_module():
    path = Path(__file__).with_name("efficientnet_b3_public_repro_v1.py")
    if not path.is_file():
        raise FileNotFoundError(f"audited v1 dependency is absent: {path}")
    name = "rsna_knee_effnet_b3_public_repro_v1_dependency"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import audited v1 dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()
TARGETS = BASE.TARGETS
PLANES = BASE.PLANES
UID = BASE.UID
TrainConfig = BASE.TrainConfig
load_dicom_volume = BASE.load_dicom_volume
uniform_slice_indices = BASE.uniform_slice_indices
stochastic_slice_indices = BASE.stochastic_slice_indices
pooled_macro_auc = BASE.pooled_macro_auc


@dataclass(frozen=True)
class TrainingPolicy:
    """Choices specific to v2 and therefore stored separately from TrainConfig."""

    expert_sample_fraction: float = 0.12
    unfrozen_backbone_blocks: int = 2
    gradient_clip_norm: float = 5.0
    model_family: str = "efficientnet_b3_anatomy_mil_v2"


def _training_imports():
    return BASE._training_imports()


def pick_series_slots(series: pd.DataFrame, count: int = 3):
    """Return one fixed slot per anatomical plane.

    The v1 selector intentionally backfills a missing plane with another useful
    series.  That is appropriate for an anatomy-agnostic model, but would attach
    the wrong plane embedding in v2.  Preserve empty slots instead so index zero,
    one, and two always mean sagittal, coronal, and axial respectively.
    """
    if int(count) != len(PLANES):
        raise ValueError(f"v2 requires exactly {len(PLANES)} anatomical plane slots")
    selected = BASE.pick_one_series_per_plane(series)
    return [selected[plane] for plane in PLANES]


def make_dataset_class():
    """Build the v1 study dataset with fixed anatomical plane slots."""
    _, _, _, torch, _, _, Dataset = _training_imports()

    class AnatomyAwareKneeStudyDataset(Dataset):
        def __init__(self, rows, series, image_root, config, training, seed):
            self.rows = rows.reset_index(drop=True)
            self.series = {
                str(uid): group.copy() for uid, group in series.groupby(UID, sort=False)
            }
            self.image_root = Path(image_root)
            self.config = config
            self.training = bool(training)
            self.seed = int(seed)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row, config = self.rows.iloc[index], self.config
            uid = str(row[UID])
            count = config.train_slices if self.training else config.valid_slices
            visit_seed = (
                int(torch.randint(0, 2**31 - 1, (1,)).item()) if self.training else 0
            )
            rng = np.random.default_rng(self.seed + 1_000_003 * index + visit_seed)
            empty = pd.DataFrame(
                columns=[
                    "SeriesInstanceUID",
                    "Anatomical_Plane",
                    "Fluid_Sensitive",
                    "Fat_Suppression",
                ]
            )
            chosen = pick_series_slots(self.series.get(uid, empty), len(PLANES))
            planes, masks = [], []
            for series_uid in chosen:
                if series_uid is None:
                    planes.append(
                        np.zeros((count, config.image_size, config.image_size), np.float32)
                    )
                    masks.append(np.zeros(count, bool))
                    continue
                volume = load_dicom_volume(
                    self.image_root / uid / series_uid,
                    config.image_size,
                    config.max_series_slices,
                )
                indices = (
                    stochastic_slice_indices(len(volume), count, rng)
                    if self.training
                    else uniform_slice_indices(len(volume), count)
                )
                sampled = volume[indices].astype(np.float32)
                if self.training:
                    sampled = np.clip(sampled * rng.uniform(0.85, 1.15), 0, 1)
                    sampled = sampled ** rng.uniform(0.85, 1.15)
                    if rng.random() < 0.5:
                        sampled = sampled[:, :, ::-1].copy()
                sampled = (sampled - 0.449) / 0.226
                planes.append(sampled)
                masks.append(np.ones(count, bool))
            return {
                "uid": uid,
                "images": torch.from_numpy(np.stack(planes)[:, :, None]),
                "mask": torch.from_numpy(np.stack(masks)),
                "target": torch.tensor(
                    [row[f"target::{target}"] for target in TARGETS], dtype=torch.float32
                ),
                "weight": torch.tensor(
                    [row[f"train_weight::{target}"] for target in TARGETS],
                    dtype=torch.float32,
                ),
            }

    return AnatomyAwareKneeStudyDataset


def make_model_class():
    """Build a target-attention MIL model with fixed anatomical coordinates."""
    _, _, timm, torch, nn, _, _ = _training_imports()

    class AnatomyAwareKneeMIL(nn.Module):
        def __init__(self, backbone="efficientnet_b3", checkpoint=None, backbone_chunk=12):
            super().__init__()
            self.backbone_chunk = int(backbone_chunk)
            self.backbone = timm.create_model(
                backbone,
                pretrained=False,
                in_chans=1,
                num_classes=0,
                global_pool="avg",
            )
            if checkpoint:
                state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
                if isinstance(state, Mapping) and "state_dict" in state:
                    state = state["state_dict"]
                state = {str(key).replace("module.", "", 1): value for key, value in state.items()}
                expected = self.backbone.state_dict()
                if "conv_stem.weight" in state and state["conv_stem.weight"].shape[1] == 3:
                    # Sum rather than mean preserves the response magnitude expected by
                    # the pretrained filters after the one-channel ImageNet normalization.
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
            self.plane_embedding = nn.Embedding(len(PLANES), features)
            self.position_projection = nn.Linear(4, features, bias=False)
            nn.init.normal_(self.plane_embedding.weight, std=0.01)
            nn.init.normal_(self.position_projection.weight, std=0.01)

            self.attn_v = nn.Linear(features, hidden)
            self.attn_u = nn.Linear(features, hidden)
            self.attn_out = nn.Linear(hidden, len(TARGETS))
            self.target_weight = nn.Parameter(torch.empty(len(TARGETS), features))
            self.target_bias = nn.Parameter(torch.zeros(len(TARGETS)))
            self.slice_head = nn.Linear(features, len(TARGETS))
            self.mix_logit = nn.Parameter(torch.tensor(-1.1))
            nn.init.xavier_uniform_(self.target_weight)

        def set_backbone_stage(self, final_blocks=0):
            """Freeze the backbone, optionally reopening only its final stages."""
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            final_blocks = int(final_blocks)
            if final_blocks <= 0:
                return
            blocks = list(self.backbone.blocks)
            if final_blocks > len(blocks):
                raise ValueError(
                    f"requested {final_blocks} trainable blocks from a {len(blocks)}-block backbone"
                )
            for block in blocks[-final_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for name in ("conv_head", "bn2"):
                layer = getattr(self.backbone, name, None)
                if layer is not None:
                    for parameter in layer.parameters():
                        parameter.requires_grad_(True)

        @staticmethod
        def _position_basis(slices, device, dtype):
            position = torch.linspace(-1.0, 1.0, slices, device=device, dtype=dtype)
            return torch.stack(
                (
                    position,
                    position.square(),
                    torch.sin(math.pi * position),
                    torch.cos(math.pi * position),
                ),
                dim=-1,
            )

        def forward(self, images, mask):
            batch, planes, slices, channels, height, width = images.shape
            if planes != len(PLANES):
                raise ValueError(f"expected {len(PLANES)} plane slots, got {planes}")
            flat = images.reshape(batch * planes * slices, channels, height, width)
            encoded = []
            train_backbone = self.training and any(
                parameter.requires_grad for parameter in self.backbone.parameters()
            )
            for start in range(0, len(flat), self.backbone_chunk):
                chunk = flat[start : start + self.backbone_chunk]
                if train_backbone:
                    from torch.utils.checkpoint import checkpoint

                    encoded.append(checkpoint(self.backbone, chunk, use_reentrant=False))
                else:
                    encoded.append(self.backbone(chunk))
            features = torch.cat(encoded, dim=0).reshape(batch, planes, slices, -1)

            plane_ids = torch.arange(planes, device=features.device)
            plane_context = self.plane_embedding(plane_ids)[None, :, None, :]
            position = self._position_basis(slices, features.device, features.dtype)
            position_context = self.position_projection(position)[None, None, :, :]
            features = features + plane_context + position_context
            features = features.reshape(batch, planes * slices, -1)
            flat_mask = mask.reshape(batch, planes * slices)

            gated = torch.tanh(self.attn_v(features)) * torch.sigmoid(self.attn_u(features))
            attention = self.attn_out(gated).masked_fill(
                ~flat_mask[:, :, None], -1e4
            ).softmax(dim=1)
            pooled = torch.einsum("bnt,bnf->btf", attention, features)
            attention_logits = (
                torch.einsum("btf,tf->bt", pooled, self.target_weight) + self.target_bias
            )
            max_logits = self.slice_head(features).masked_fill(
                ~flat_mask[:, :, None], -1e4
            ).max(dim=1).values
            mix = torch.sigmoid(self.mix_logit)
            return (1.0 - mix) * attention_logits + mix * max_logits

    return AnatomyAwareKneeMIL


def _make_expert_sampler(rows, fraction, seed, torch):
    """Sample a fixed expected expert fraction without discarding pseudo rows."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("expert_sample_fraction must be strictly between zero and one")
    expert = rows["is_expert"].to_numpy(np.int8).astype(bool)
    n_expert, n_pseudo = int(expert.sum()), int((~expert).sum())
    if n_expert == 0 or n_pseudo == 0:
        raise ValueError("expert-aware sampling requires both expert and pseudo-labelled rows")
    expert_multiplier = fraction * n_pseudo / ((1.0 - fraction) * n_expert)
    weights = np.where(expert, expert_multiplier, 1.0).astype(np.float64)
    generator = torch.Generator().manual_seed(int(seed))
    from torch.utils.data import WeightedRandomSampler

    sampler = WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=len(rows),
        replacement=True,
        generator=generator,
    )
    return sampler, {
        "expert_rows": n_expert,
        "pseudo_rows": n_pseudo,
        "expert_multiplier": float(expert_multiplier),
        "expected_expert_fraction": float(fraction),
    }


def train_one_fold(
    label_table,
    series,
    image_root,
    output_dir,
    fold,
    backbone_checkpoint,
    config,
    policy=TrainingPolicy(),
):
    _, _, _, torch, nn, DataLoader, _ = _training_imports()
    random.seed(config.seed + fold)
    np.random.seed(config.seed + fold)
    torch.manual_seed(config.seed + fold)
    Dataset, Model = make_dataset_class(), make_model_class()
    train_rows, valid_rows = BASE.prepare_fold_rows(label_table, fold, config)
    sampler, sampler_meta = _make_expert_sampler(
        train_rows,
        policy.expert_sample_fraction,
        config.seed + 10_007 * fold,
        torch,
    )
    train_loader = DataLoader(
        Dataset(train_rows, series, image_root, config, True, config.seed + fold),
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.workers,
        pin_memory=True,
        persistent_workers=config.workers > 0,
    )
    valid_loader = DataLoader(
        Dataset(valid_rows, series, image_root, config, False, config.seed + fold),
        batch_size=1,
        shuffle=False,
        num_workers=max(1, config.workers // 2),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU is required")
    core_model = Model(config.backbone, backbone_checkpoint, config.backbone_chunk).to(device)
    model = nn.DataParallel(core_model) if torch.cuda.device_count() > 1 else core_model
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    scaler = torch.amp.GradScaler("cuda")
    history = []
    schedule = (
        [(0, config.frozen_lr)] * config.frozen_epochs
        + [(policy.unfrozen_backbone_blocks, config.unfrozen_lr)] * config.unfrozen_epochs
    )
    optimizer, active_blocks = None, None
    for epoch, (trainable_blocks, learning_rate) in enumerate(schedule):
        if active_blocks != trainable_blocks:
            core_model.set_backbone_stage(trainable_blocks)
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=learning_rate,
                weight_decay=config.weight_decay,
            )
            active_blocks = trainable_blocks
        assert optimizer is not None
        model.train()
        for layer in core_model.backbone.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        losses = []
        sampled_expert = 0
        sampled_total = 0
        for batch in train_loader:
            images, mask = batch["images"].to(device), batch["mask"].to(device)
            target, weight = batch["target"].to(device), batch["weight"].to(device)
            # Expert rows have the same gold weight for all twelve targets.
            sampled_expert += int((weight.min(dim=1).values >= config.gold_weight).sum().item())
            sampled_total += int(len(weight))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                raw = loss_fn(model(images, mask), target)
                loss = (raw * weight).sum() / weight.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), policy.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "trainable_backbone_blocks": int(trainable_blocks),
                "learning_rate": float(learning_rate),
                "observed_expert_fraction": sampled_expert / max(sampled_total, 1),
            }
        )

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
    macro, per_target = BASE.pooled_macro_auc(truth, score)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame({UID: uids})
    for target_index, target in enumerate(TARGETS):
        oof[f"gold::{target}"] = truth[:, target_index]
        oof[f"pred::{target}"] = score[:, target_index]
    oof.to_csv(output_dir / f"fold{fold}_expert_oof.csv", index=False)
    destination = output_dir / f"fold{fold}_final.pt"
    torch.save(
        {
            "model": core_model.state_dict(),
            "model_family": policy.model_family,
            "backbone": config.backbone,
            "config": asdict(config),
            "training_policy": asdict(policy),
            "sampler": sampler_meta,
            "fold": fold,
            "fold_diagnostic_macro_auc": macro,
            "fold_diagnostic_per_target_auc": per_target,
            "history": history,
            "visible_gpus": torch.cuda.device_count(),
            "backbone_checkpoint_sha256": hashlib.sha256(
                backbone_checkpoint.read_bytes()
            ).hexdigest(),
        },
        destination,
    )
    return destination


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train-csv",
        "series-csv",
        "pilkwang-labels",
        "steven-labels",
        "lixin-labels",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)

    # Expert oversampling replaces some of the multiplicative gold weighting;
    # keeping both at their v1 values would overfit the 46-or-so training experts.
    config = replace(TrainConfig(), gold_weight=2.0)
    policy = TrainingPolicy()
    paths = [
        args.train_csv,
        args.series_csv,
        args.pilkwang_labels,
        args.steven_labels,
        args.lixin_labels,
    ]
    frames = [pd.read_csv(path) for path in paths]
    labels = BASE.build_public_consensus(
        frames[0], frames[2], frames[3], frames[4], config.pseudo_weight_floor
    )
    labels["fold"] = BASE.assign_group_balanced_folds(labels, config.folds, config.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output_dir / "fold_labels.csv", index=False)
    manifest = {
        "config": asdict(config),
        "training_policy": asdict(policy),
        "rows": len(labels),
        "expert_rows": int(labels["is_expert"].sum()),
        "fold_counts": labels["fold"].value_counts().sort_index().to_dict(),
        "fold_expert_counts": labels.loc[
            labels["is_expert"].eq(1), "fold"
        ].value_counts().sort_index().to_dict(),
        "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "base_module_sha256": hashlib.sha256(
            Path(BASE.__file__).read_bytes()
        ).hexdigest(),
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.prepare_only:
        return
    if args.fold is None or args.image_root is None or args.backbone_checkpoint is None:
        parser.error("training requires --fold, --image-root, and --backbone-checkpoint")
    if not 0 <= args.fold < config.folds:
        parser.error(f"--fold must be in [0, {config.folds - 1}]")
    print(
        "saved",
        train_one_fold(
            labels,
            frames[1],
            args.image_root,
            args.output_dir,
            args.fold,
            args.backbone_checkpoint,
            config,
            policy,
        ),
    )


if __name__ == "__main__":
    main()
