#!/usr/bin/env python3
"""V3.2 label post-process: Synovitis soft boost from Effusion + source mix.

Starts from V3.1 fold-cross labels (artifacts/v3_labels/labels_v3_fold*.csv) and:
  * for non-gold rows, blends Synovitis toward Effusion when effusion is high
    and synovitis is muted (public hypothesis that helped report-label analysis)
  * slightly upweights Baker's low scores (silence ≈ negative)
  * writes artifacts/v32_labels/ + audit JSON

Does NOT change fold assignments (keeps folds_v3.csv) so models stay comparable.
"""

from __future__ import annotations

import argparse
import json
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


def auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def process_fold(
    df: pd.DataFrame,
    train: pd.DataFrame,
    syn_alpha: float,
    baker_floor: float,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    gold_ids = set(
        train.loc[train[TARGETS].notna().all(axis=1), UID].astype(str).tolist()
    )
    is_gold = out[UID].astype(str).isin(gold_ids)
    syn = out["Synovitis"].astype(float).to_numpy()
    eff = out["Effusion"].astype(float).to_numpy()
    baker = out["Baker's"].astype(float).to_numpy()
    syn_new = syn.copy()
    # Soft-impute synovitis from effusion only on non-gold rows.
    mask = (~is_gold.to_numpy()) & (syn < 0.45) & (eff > 0.55)
    syn_new[mask] = (1.0 - syn_alpha) * syn[mask] + syn_alpha * eff[mask]
    # Mild always-on mix for non-gold (smaller alpha) — helps weak sources
    mask2 = (~is_gold.to_numpy()) & ~mask
    syn_new[mask2] = (1.0 - 0.15) * syn[mask2] + 0.15 * eff[mask2]
    syn_new = np.clip(syn_new, 0.01, 0.99)
    out["Synovitis"] = syn_new
    if "Synovitis__source" in out.columns:
        out.loc[mask, "Synovitis__source"] = (
            out.loc[mask, "Synovitis__source"].astype(str) + "+effusion_boost"
        )

    # Baker's: silence / very low → slightly more negative prior on non-gold
    baker_new = baker.copy()
    bmask = (~is_gold.to_numpy()) & (baker < 0.25)
    baker_new[bmask] = np.minimum(baker[bmask], baker_floor)
    out["Baker's"] = np.clip(baker_new, 0.01, 0.99)

    # Gold diagnostic on this fold bank using train gold labels
    gold = train.loc[train[TARGETS].notna().all(axis=1)].copy()
    soft = (
        out.set_index(UID)[["Synovitis", "Baker's", "Effusion"]]
        .apply(pd.to_numeric, errors="coerce")
        .add_suffix("_soft")
    )
    joined = gold.set_index(UID).join(soft, how="inner")
    y_syn = joined["Synovitis"].astype(float).to_numpy()
    soft_syn = joined["Synovitis_soft"].astype(float).to_numpy()
    y_bak = joined["Baker's"].astype(float).to_numpy()
    soft_bak = joined["Baker's_soft"].astype(float).to_numpy()
    stats = {
        "syn_boost_rows": int(mask.sum()),
        "baker_floor_rows": int(bmask.sum()),
        "gold_syn_auc": auc((y_syn >= 0.5).astype(float), soft_syn),
        "gold_baker_auc": auc((y_bak >= 0.5).astype(float), soft_bak),
        "n_gold_joined": int(len(joined)),
    }
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=ROOT / "artifacts" / "v3_labels")
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "raw" / "train.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "v32_labels")
    ap.add_argument("--syn-alpha", type=float, default=0.40)
    ap.add_argument("--baker-floor", type=float, default=0.08)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train)
    audit = {"syn_alpha": args.syn_alpha, "baker_floor": args.baker_floor, "folds": {}}
    # copy folds unchanged
    folds_src = args.src / "folds_v3.csv"
    if folds_src.is_file():
        (args.out / "folds_v3.csv").write_bytes(folds_src.read_bytes())
    for fold in range(5):
        src = args.src / f"labels_v3_fold{fold}.csv"
        df = pd.read_csv(src)
        out, stats = process_fold(df, train, args.syn_alpha, args.baker_floor)
        dest = args.out / f"labels_v3_fold{fold}.csv"
        out.to_csv(dest, index=False)
        audit["folds"][str(fold)] = stats
        print(f"fold{fold}", stats, "->", dest)
    # global labels_v3.csv if present
    base = args.src / "labels_v3.csv"
    if base.is_file():
        df = pd.read_csv(base)
        # labels_v3 may lack weight columns format of fold files; only process if Synovitis present
        if "Synovitis" in df.columns and "Effusion" in df.columns:
            # reuse process with fake weight cols
            out, stats = process_fold(df, train, args.syn_alpha, args.baker_floor)
            out.to_csv(args.out / "labels_v3.csv", index=False)
            audit["global"] = stats
    (args.out / "audit_v32.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    meta = {
        "title": "rsna-knee-v32-labels-folds",
        "id": "danielsolo1770/rsna-knee-v32-labels-folds",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (args.out / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", args.out / "audit_v32.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
