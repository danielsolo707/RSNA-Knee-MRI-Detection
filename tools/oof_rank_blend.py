#!/usr/bin/env python3
"""OOF-gated per-target rank blend of prediction matrices.

Inputs are CSV files with StudyInstanceUID + 12 target columns (probabilities).
Optional gold/soft label CSV with the same targets for OOF scoring.

Emits:
  - blend weights per target (JSON)
  - blended predictions CSV
  - per-target AUC table when labels are available
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Trapezoidal ROC-AUC without sklearn (works in bare envs)."""
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
    # average ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(y_score):
        j = i
        while j + 1 < len(y_score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j) + 1.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

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


def rank_pct(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # average ties
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j) + 1.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks / len(values)


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, dict[str, float]]:
    per = {}
    for i, name in enumerate(TARGETS):
        yt = y[:, i]
        if len(np.unique(yt)) < 2:
            per[name] = float("nan")
            continue
        per[name] = float(roc_auc_score(yt, p[:, i]))
    vals = [v for v in per.values() if v == v]
    return float(np.mean(vals)) if vals else float("nan"), per


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred",
        action="append",
        nargs=2,
        metavar=("NAME", "CSV"),
        required=True,
        help="model name and prediction CSV (repeatable)",
    )
    parser.add_argument("--labels", type=Path, help="soft/gold labels CSV for OOF")
    parser.add_argument(
        "--label-prefix",
        default="",
        help="optional column prefix for labels (e.g. gold:: or empty)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.0,
        help="only blend a model into a target if OOF AUC >= best single + min_gain",
    )
    args = parser.parse_args()

    frames = {}
    for name, path in args.pred:
        df = pd.read_csv(path)
        missing = {UID, *TARGETS}.difference(df.columns)
        if missing:
            raise ValueError(f"{path} missing {missing}")
        frames[name] = df.set_index(UID)[TARGETS].apply(pd.to_numeric, errors="coerce")

    shared = None
    for frame in frames.values():
        shared = set(frame.index) if shared is None else shared.intersection(frame.index)
    if not shared:
        raise RuntimeError("no shared study IDs across prediction files")
    ids = sorted(shared)
    mats = {name: frame.loc[ids].to_numpy(float) for name, frame in frames.items()}
    rank_mats = {
        name: np.stack([rank_pct(mat[:, t]) for t in range(len(TARGETS))], axis=1)
        for name, mat in mats.items()
    }

    labels = None
    if args.labels is not None:
        lab = pd.read_csv(args.labels).set_index(UID)
        cols = [f"{args.label_prefix}{t}" for t in TARGETS]
        if not set(cols).issubset(lab.columns):
            # try raw target names
            cols = TARGETS
        labels = lab.loc[ids, cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        # binarize soft labels at 0.5 for diagnostic AUC only
        labels_bin = (labels >= 0.5).astype(float)
    else:
        labels_bin = None

    names = list(mats.keys())
    weights = {t: {n: 0.0 for n in names} for t in TARGETS}
    chosen = {}
    for t_idx, target in enumerate(TARGETS):
        scores = {}
        if labels_bin is not None:
            for name in names:
                y = labels_bin[:, t_idx]
                if len(np.unique(y)) < 2:
                    scores[name] = float("nan")
                else:
                    scores[name] = float(roc_auc_score(y, mats[name][:, t_idx]))
            best_name = max(scores, key=lambda n: (-1 if scores[n] != scores[n] else scores[n]))
            best = scores[best_name]
            selected = [
                n
                for n, s in scores.items()
                if s == s and s + 1e-12 >= best + args.min_gain
            ]
            # Always keep the best; include others that are not worse than best+min_gain
            # (min_gain=0 means all non-worse). Prefer at least 2 if they are within 0.01.
            if len(selected) == 1:
                close = [
                    n
                    for n, s in scores.items()
                    if s == s and s >= best - 0.01 and n != best_name
                ]
                selected = list(dict.fromkeys([best_name] + close))
        else:
            selected = names
            scores = {n: float("nan") for n in names}
            best_name = names[0]

        mass = 1.0 / len(selected)
        for n in selected:
            weights[target][n] = mass
        chosen[target] = {
            "selected": selected,
            "scores": scores,
            "best": best_name,
        }

    blended_rank = np.zeros((len(ids), len(TARGETS)), dtype=np.float64)
    for t_idx, target in enumerate(TARGETS):
        for name in names:
            w = weights[target][name]
            if w:
                blended_rank[:, t_idx] += w * rank_mats[name][:, t_idx]

    # Map ranks back to a probability-like scale by averaging selected raw probs
    # weighted the same way (for submission compatibility). Ranking is what AUC
    # cares about; raw average of selected models is fine after rank selection.
    blended_prob = np.zeros_like(blended_rank)
    for t_idx, target in enumerate(TARGETS):
        for name in names:
            w = weights[target][name]
            if w:
                blended_prob[:, t_idx] += w * mats[name][:, t_idx]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({UID: ids})
    for t_idx, target in enumerate(TARGETS):
        out[target] = blended_prob[:, t_idx]
    out.to_csv(args.output_dir / "blend_predictions.csv", index=False)

    report = {"weights": weights, "per_target": chosen}
    if labels_bin is not None:
        macro, per = macro_auc(labels_bin, blended_prob)
        singles = {}
        for name in names:
            m, p = macro_auc(labels_bin, mats[name])
            singles[name] = {"macro": m, "per_target": p}
        report["blend_macro_auc"] = macro
        report["blend_per_target_auc"] = per
        report["singles"] = singles
    (args.output_dir / "blend_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in report if k != "weights"}, indent=2))
    print("wrote", args.output_dir / "blend_predictions.csv")


if __name__ == "__main__":
    main()
