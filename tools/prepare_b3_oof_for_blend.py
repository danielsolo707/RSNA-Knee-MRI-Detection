#!/usr/bin/env python3
"""Build blend-ready CSVs from B3 fold OOF + DINO npz.

B3 fold OOF columns look like:
  StudyInstanceUID, fold, is_expert, target::ACL, pred::ACL, ...

This emits:
  artifacts/blend_inputs/b3_oof.csv   — StudyInstanceUID + 12 pred columns
  artifacts/blend_inputs/dino_oof.csv — from research/pilkwang_oof/oof.npz
  artifacts/blend_inputs/soft_labels.csv — soft targets from B3 OOF (target::*)

Usage:
  python tools/prepare_b3_oof_for_blend.py
  python tools/prepare_b3_oof_for_blend.py --folds 0 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
UID = "StudyInstanceUID"


def find_fold_oof(fold: int) -> Path | None:
    candidates = [
        ROOT / f"artifacts/b3_fold{fold}_ok/v3_b3_fold{fold}/fold{fold}_full_oof.csv",
        ROOT / f"artifacts/b3_fold{fold}_ok/fold{fold}_full_oof.csv",
    ]
    # Also search one level under artifacts/b3_foldN_ok
    base = ROOT / f"artifacts/b3_fold{fold}_ok"
    if base.is_dir():
        for p in base.rglob(f"fold{fold}_full_oof.csv"):
            candidates.append(p)
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def load_b3_oofs(folds: list[int]) -> pd.DataFrame:
    frames = []
    missing = []
    for fold in folds:
        path = find_fold_oof(fold)
        if path is None:
            missing.append(fold)
            continue
        df = pd.read_csv(path)
        pred_cols = {f"pred::{t}": t for t in TARGETS}
        if not set(pred_cols).issubset(df.columns):
            raise ValueError(f"{path} missing pred columns; has {list(df.columns)[:8]}...")
        out = df[[UID]].copy()
        for src, dst in pred_cols.items():
            out[dst] = pd.to_numeric(df[src], errors="coerce")
        # Keep soft targets from first available fold for label file
        for t in TARGETS:
            col = f"target::{t}"
            if col in df.columns:
                out[f"_soft::{t}"] = pd.to_numeric(df[col], errors="coerce")
        out["_fold"] = fold
        frames.append(out)
        print(f"loaded fold{fold}: {len(out)} rows from {path}")
    if missing:
        raise FileNotFoundError(f"missing OOF for folds: {missing}")
    return pd.concat(frames, ignore_index=True)


def export_dino(out_dir: Path) -> Path:
    npz_path = ROOT / "research/pilkwang_oof/oof.npz"
    data = np.load(npz_path, allow_pickle=True)
    ids = data["ids"].astype(str)
    pred = data["pred"].astype(float)
    targets = [str(t) for t in data["targets"].tolist()]
    # Align column order to TARGETS
    idx = [targets.index(t) for t in TARGETS]
    mat = pred[:, idx]
    df = pd.DataFrame({UID: ids})
    for i, t in enumerate(TARGETS):
        df[t] = mat[:, i]
    path = out_dir / "dino_oof.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} rows)")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "blend_inputs",
    )
    args = ap.parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    b3 = load_b3_oofs(args.folds)
    # One row per study (folds are disjoint held-out)
    soft_cols = [f"_soft::{t}" for t in TARGETS]
    has_soft = all(c in b3.columns for c in soft_cols)

    pred = b3[[UID, *TARGETS]].copy()
    # Drop duplicate UIDs if any (should not happen across folds)
    if pred[UID].duplicated().any():
        n = int(pred[UID].duplicated().sum())
        print(f"WARNING: {n} duplicate UIDs; keeping first")
        pred = pred.drop_duplicates(UID, keep="first")
    b3_path = out_dir / "b3_oof.csv"
    pred.to_csv(b3_path, index=False)
    print(f"wrote {b3_path} ({len(pred)} rows)")

    if has_soft:
        soft = b3[[UID]].copy()
        for t in TARGETS:
            soft[t] = b3[f"_soft::{t}"]
        soft = soft.drop_duplicates(UID, keep="first")
        soft_path = out_dir / "soft_labels.csv"
        soft.to_csv(soft_path, index=False)
        print(f"wrote {soft_path} ({len(soft)} rows)")

    export_dino(out_dir)
    print("done. Next:")
    print(
        "  python tools/oof_rank_blend.py "
        f"--pred dino {out_dir / 'dino_oof.csv'} "
        f"--pred b3 {out_dir / 'b3_oof.csv'} "
        f"--labels {out_dir / 'soft_labels.csv'} "
        f"--output-dir artifacts/blend_v5"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
