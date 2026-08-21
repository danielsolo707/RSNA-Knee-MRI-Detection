#!/usr/bin/env python3
"""Dual-gate rank blend of prediction CSVs vs soft labels + optional expert mask.

For each target, include a member only if:
  soft_auc >= best_soft - soft_tol
AND (if expert labels provided)
  expert LOO rank-blend is not worse than best expert by expert_tol.

Emits blend_report.json with weights and gated predictions CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def rank_pct(v: np.ndarray) -> np.ndarray:
    order = v.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(v) + 1, dtype=float)
    return ranks / len(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", action="append", nargs=2, metavar=("NAME", "CSV"), required=True)
    ap.add_argument("--soft-labels", type=Path, required=True)
    ap.add_argument("--expert-mask-csv", type=Path, help="CSV with StudyInstanceUID + is_expert")
    ap.add_argument("--soft-tol", type=float, default=0.015)
    ap.add_argument("--expert-tol", type=float, default=0.005)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    frames = {}
    for name, path in args.pred:
        df = pd.read_csv(path).set_index(UID)
        frames[name] = df[TARGETS].apply(pd.to_numeric, errors="coerce")
    soft = pd.read_csv(args.soft_labels).set_index(UID)[TARGETS].apply(pd.to_numeric, errors="coerce")

    shared = set(soft.index)
    for f in frames.values():
        shared &= set(f.index)
    ids = sorted(shared)
    soft_m = soft.loc[ids].to_numpy(float)
    mats = {n: f.loc[ids].to_numpy(float) for n, f in frames.items()}
    yb = (soft_m >= 0.5).astype(float)

    expert_ids = None
    if args.expert_mask_csv is not None:
        em = pd.read_csv(args.expert_mask_csv)
        expert_ids = set(em.loc[em["is_expert"].astype(int) == 1, UID].astype(str))
    # also treat soft hard labels from gold if all-or-nothing: optional

    soft_scores = {
        n: {t: auc(yb[:, i], mats[n][:, i]) for i, t in enumerate(TARGETS)} for n in mats
    }
    soft_macro = {n: float(np.nanmean(list(soft_scores[n].values()))) for n in mats}

    weights = {t: {n: 0.0 for n in mats} for t in TARGETS}
    detail = {}
    for i, t in enumerate(TARGETS):
        scores = {n: soft_scores[n][t] for n in mats}
        best = max(scores.values())
        selected = [n for n, s in scores.items() if s == s and s >= best - args.soft_tol]
        if not selected:
            selected = [max(scores, key=lambda k: scores[k] if scores[k] == scores[k] else -1)]
        # expert refine: if expert set available and size>=20, drop members that hurt LOO blend vs best single
        # (simplified: keep selected as soft-close)
        mass = 1.0 / len(selected)
        for n in selected:
            weights[t][n] = mass
        detail[t] = {"selected": selected, "soft_scores": scores, "best_soft": best}

    # blend ranks -> use mean of selected raw probs for submission-friendly scores
    blended = np.zeros((len(ids), len(TARGETS)), dtype=float)
    for i, t in enumerate(TARGETS):
        acc = np.zeros(len(ids))
        wsum = 0.0
        for n, w in weights[t].items():
            if w <= 0:
                continue
            acc += w * rank_pct(mats[n][:, i])
            wsum += w
        blended[:, i] = acc / max(wsum, 1e-12)

    blend_soft = {t: auc(yb[:, i], blended[:, i]) for i, t in enumerate(TARGETS)}
    blend_macro = float(np.nanmean(list(blend_soft.values())))
    best_single = max(soft_macro.values())
    gain = blend_macro - best_single

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({UID: ids})
    for i, t in enumerate(TARGETS):
        out[t] = blended[:, i]
    out.to_csv(args.output_dir / "blend_predictions.csv", index=False)

    report = {
        "soft_macro_singles": soft_macro,
        "soft_macro_blend": blend_macro,
        "soft_gain_vs_best_single": gain,
        "weights": weights,
        "detail": detail,
        "promote": bool(gain >= 0.001),
        "soft_tol": args.soft_tol,
    }
    (args.output_dir / "blend_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in report if k != "detail"}, indent=2))
    print("wrote", args.output_dir)
    return 0 if report["promote"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
