"""Score labels_v4_qwen7b.csv vs gold-58 + compare against current bank.
Usage: python tools/validate_labels_v4.py <path-to-csv>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
           "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]
CURRENT_BANK_MACRO = 0.8919


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    df = pd.read_csv(sys.argv[1], dtype={"StudyInstanceUID": str})
    print(f"labels_v4 rows: {len(df)}")
    missing = [t for t in TARGETS if t not in df.columns]
    if missing:
        print("MISSING TARGET COLUMNS:", missing)
        return
    nan_counts = {t: int(df[t].isna().sum()) for t in TARGETS}
    print("NaN per target:", nan_counts)

    train = pd.read_csv(REPO / "data/raw/train.csv", dtype={"StudyInstanceUID": str})
    gold = train.dropna(subset=TARGETS, how="all").set_index("StudyInstanceUID")
    dfi = df.set_index("StudyInstanceUID")

    print(f"\n{'target':<17s} {'AUC':>7s} {'pos_rate_4407':>14s} {'n_pos_gold':>11s}")
    aucs = []
    for t in TARGETS:
        pr = float((dfi[t] > 0.5).mean())
        gy = gold[t].dropna()
        common = gy.index.intersection(dfi.index)
        y = gy.loc[common]
        p = dfi.loc[common, t]
        m = y.notna() & p.notna()
        auc = roc_auc_score((y[m] > 0.5).astype(int), p[m])
        aucs.append(auc)
        print(f"{t:<17s} {auc:>7.3f} {pr:>14.3f} {int((gy>0.5).sum()):>11d}")

    macro = float(np.mean(aucs))
    verdict = "BETTER - adopt!" if macro > CURRENT_BANK_MACRO else ("TIE - inspect per-target" if abs(macro - CURRENT_BANK_MACRO) < 0.005 else "WORSE - do not adopt")
    print(f"\nMACRO v4: {macro:.4f} vs current bank {CURRENT_BANK_MACRO:.4f}")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
