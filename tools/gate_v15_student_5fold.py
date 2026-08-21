#!/usr/bin/env python3
"""Local verifier for the V15 5-fold student OOF kernel.

Independently recomputes the gates from the downloaded kernel outputs
(v15_oof.csv, v15_audit.json) plus the pinned local teacher inputs.

Usage:
  python tools/gate_v15_student_5fold.py \
      --oof research/v14_top20_cli/v15_kernel_output/v15_oof.csv \
      --soft research/v14_top20_cli/v15_kernel_output/v15_soft_labels.csv \
      --audit research/v14_top20_cli/v15_kernel_output/v15_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
UID = "StudyInstanceUID"

DINO_OOF_SHA = "62d47ba4c0c8347b5b24e7fd2aa517aae0fd6d4656fd5d829fd3a50f0159909c"
V52_OOF_SHA = "f5b1f02029af98740565bec5b6a76d8471b4b6a6039d09ea410368230cb1f03f"

V10_FINAL_ALPHAS = {
    "ACL": 0.505, "MCL": 0.20, "Medial Meniscus": 0.20,
    "Lateral Meniscus": 0.20, "Medial OA": 0.20, "Lateral OA": 0.20,
    "PF OA": 0.20, "Effusion": 0.20, "Synovitis": 0.505, "Baker's": 0.20,
    "Contusion": 0.20, "Fracture": 0.20,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    pos, neg = hard == 1, hard == 0
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", type=Path, required=True)
    ap.add_argument("--soft", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    args = ap.parse_args()

    audit = json.loads(args.audit.read_text())
    oof = pd.read_csv(args.oof, dtype={UID: str})
    soft = pd.read_csv(args.soft, dtype={UID: str})

    # row-order contract vs train.csv
    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv", dtype={UID: str})
    if oof[UID].tolist() != train[UID].astype(str).tolist():
        raise SystemExit("V15 gate: OOF row order does not match train.csv")
    if soft[UID].tolist() != train[UID].astype(str).tolist():
        raise SystemExit("V15 gate: soft labels row order does not match train.csv")

    if sha256_file(args.soft) != audit.get("soft_labels_sha256"):
        raise SystemExit("V15 gate: soft labels SHA does not match kernel audit")

    # teacher inputs from pinned local copies
    dino = np.load(ROOT / "research" / "pilkwang_oof" / "oof.npz", allow_pickle=False)
    if sha256_file(ROOT / "research" / "pilkwang_oof" / "oof.npz") != DINO_OOF_SHA:
        raise SystemExit("V15 gate: DINO OOF local hash mismatch")
    v52 = pd.read_csv(
        ROOT / "references" / "candidates" / "antoine_e9_files" / "v52_oof.csv",
        dtype={UID: str},
    )
    if sha256_file(
        ROOT / "references" / "candidates" / "antoine_e9_files" / "v52_oof.csv"
    ) != V52_OOF_SHA:
        raise SystemExit("V15 gate: V52 OOF local hash mismatch")

    dino_ids = [str(x) for x in dino["ids"]]
    dino_index = {u: i for i, u in enumerate(dino_ids)}
    v52_index = {u: i for i, u in enumerate(v52[UID].astype(str))}
    n = len(train)
    dino_pred = np.zeros((n, 12), dtype=np.float32)
    rad_oof = np.zeros((n, 12), dtype=np.float32)
    fold_id = np.full(n, -1, dtype=np.int8)
    is_gold = np.zeros(n, dtype=bool)
    for row, uid in enumerate(train[UID].astype(str)):
        if uid in dino_index:
            dino_pred[row] = dino["pred"][dino_index[uid]]
        if uid in v52_index:
            rad_oof[row] = v52.loc[v52_index[uid], TARGETS].to_numpy(np.float32)
            fold_id[row] = int(v52["fold"].iloc[v52_index[uid]])
            is_gold[row] = bool(int(v52["is_gold"].iloc[v52_index[uid]]))

    alphas = np.asarray([V10_FINAL_ALPHAS[t] for t in TARGETS], dtype=np.float64)
    parent_oof = (1.0 - alphas) * rank_pct(dino_pred) + alphas * rank_pct(rad_oof)
    student_oof = oof[TARGETS].to_numpy(np.float64)

    soft_cont = soft[TARGETS].to_numpy(np.float64)
    if np.isnan(soft_cont).any():
        raise SystemExit("V15 gate: NaN in soft labels")
    soft_y = (soft_cont >= 0.5).astype(float)
    if not np.array_equal(soft["is_gold"].to_numpy(np.int8).astype(bool), is_gold):
        raise SystemExit("V15 gate: soft is_gold does not match v52 gold mask")
    gold_y = train.loc[is_gold, TARGETS].to_numpy(np.float64)
    gold_rows = np.flatnonzero(is_gold)

    parent_macro = macro_auc(soft_y, parent_oof)
    student_macro = macro_auc(soft_y, student_oof)
    delta = student_macro - parent_macro

    fold_deltas = []
    for fold in range(5):
        mask = fold_id == fold
        fold_deltas.append(
            macro_auc(soft_y[mask], student_oof[mask])
            - macro_auc(soft_y[mask], parent_oof[mask])
        )
    positive_folds = int(sum(d > 0 for d in fold_deltas))

    collapse = []
    for j, t in enumerate(TARGETS):
        p = auc_1d(soft_y[:, j], parent_oof[:, j])
        s = auc_1d(soft_y[:, j], student_oof[:, j])
        if s == s and p == p and (p - s) > 0.02:
            collapse.append(t)

    rank_corr = float(np.mean([
        np.corrcoef(
            rank_pct(student_oof[:, j]).ravel(), rank_pct(parent_oof[:, j]).ravel()
        )[0, 1]
        for j in range(12)
    ]))

    gold_student = macro_auc(gold_y, student_oof[gold_rows])
    gold_parent = macro_auc(gold_y, parent_oof[gold_rows])
    gold_blend = macro_auc(
        gold_y, 0.50 * rank_pct(student_oof[gold_rows]) + 0.50 * rank_pct(parent_oof[gold_rows])
    )

    gates = {
        "G1_full_oof_macro_delta_vs_v10": float(delta),
        "G1_pass": bool(delta >= 0.0),
        "G2_positive_folds": positive_folds,
        "G2_pass": bool(positive_folds >= 3),
        "G3_collapse_targets": collapse,
        "G3_pass": len(collapse) == 0,
        "G4_rank_corr_vs_v10": rank_corr,
        "G4_pass": bool(rank_corr < 0.98),
    }
    all_pass = all(gates[k + "_pass"] for k in ("G1", "G2", "G3", "G4"))

    result = {
        "verifier": "tools/gate_v15_student_5fold.py",
        "status": "V15_GATE_PASS" if all_pass else "V15_GATE_BLOCKED",
        "submit_allowed": all_pass,
        "gates": gates,
        "evidence": {
            "full_oof_soft_macro_parent_v10": parent_macro,
            "full_oof_soft_macro_student": student_macro,
            "gold58_macro_parent_v10": gold_parent,
            "gold58_macro_student": gold_student,
            "gold58_macro_50_50_blend": gold_blend,
            "fold_macro_deltas_vs_v10": fold_deltas,
            "rank_corr_vs_v10": rank_corr,
        },
        "oof_sha256": sha256_file(args.oof),
        "audit_sha256": sha256_file(args.audit),
        "kernel_audit_status": audit.get("status"),
        "kernel_audit_oof_sha256": audit.get("oof_sha256"),
    }
    out = ROOT / "artifacts" / "v15_student_5fold_gate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "v15_gate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())