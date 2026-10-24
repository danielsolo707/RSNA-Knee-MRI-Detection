"""Measure every local label bank (and fusions of them) against the 58 gold studies.

This is the cheapest decisive experiment in the project: it needs no GPU, no corpus,
no training. The public Raptor labels cover only the 4,349 weak-label studies (gold
held out), but our locally cached banks cover all 4,407 - including the 58 gold ones.
So we can rank label banks, and fusions of them, directly on the only honest yardstick
we have.

Usage:
    python tools/score_label_banks.py [--out artifacts/label_scores.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LAB = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]


def rank_pct(x: np.ndarray) -> np.ndarray:
    """Percentile rank along axis 0, column-wise for 2-D input."""
    return np.apply_along_axis(
        lambda v: pd.Series(v).rank(method="average", pct=True).to_numpy(),
        0,
        np.asarray(x, dtype=float),
    )


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based (Mann-Whitney U) AUC with tie handling; NaN if a class is missing."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    ranks = np.empty(len(s_sorted), dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    # ranks and y_sorted share the same (sorted) order - index both the same way.
    return float((ranks[y_sorted == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def load_banks() -> dict[str, pd.DataFrame]:
    banks: dict[str, pd.DataFrame] = {}
    for f in sorted(glob.glob(str(ROOT / "data" / "labels" / "*" / "*.csv"))):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        cols = set(df.columns)
        if not cols.issuperset(LAB):
            continue
        id_col = "StudyInstanceUID" if "StudyInstanceUID" in cols else None
        if id_col is None:
            continue
        df[id_col] = df[id_col].astype(str)
        df = df.drop_duplicates(subset=[id_col]).set_index(id_col)[LAB].apply(
            pd.to_numeric, errors="coerce"
        )
        name = Path(f).parent.name + "/" + Path(f).stem
        banks[name] = df
    return banks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/label_scores.json")
    args = ap.parse_args()

    gold = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    gold = gold.set_index("StudyInstanceUID")[LAB].apply(pd.to_numeric, errors="coerce")
    gold_ids = gold[gold.notna().any(axis=1)].index
    print(f"gold studies: {len(gold_ids)}")

    banks = load_banks()
    print(f"label banks found: {len(banks)}")

    report: dict = {"n_gold": len(gold_ids), "single": {}, "per_target": {}, "fusion": {}}

    # ---- single banks ----
    for name, df in banks.items():
        df = df.reindex(gold_ids)
        aucs, macro = {}, []
        for t in LAB:
            y = gold.loc[gold_ids, t].to_numpy(dtype=float)
            s = rank_pct(df[t].to_numpy(dtype=float))
            a = auc(y, s)
            aucs[t] = None if np.isnan(a) else round(a, 4)
            if not np.isnan(a):
                macro.append(a)
        report["single"][name] = round(float(np.mean(macro)), 4) if macro else None
        report["per_target"][name] = aucs
        print(f"  {name:52s} macro={report['single'][name]}")

    # ---- fusions (rank-average of every combination, up to size 4) ----
    names = list(banks)
    aligned = {n: rank_pct(banks[n].reindex(gold_ids)[LAB].to_numpy(dtype=float)) for n in names}
    best: list[tuple[float, list[str]]] = []
    for r in range(2, min(5, len(names)) + 1):
        for combo in combinations(names, r):
            fused = np.mean([aligned[n] for n in combo], axis=0)
            macro = []
            for ti, t in enumerate(LAB):
                y = gold.loc[gold_ids, t].to_numpy(dtype=float)
                a = auc(y, fused[:, ti])
                if not np.isnan(a):
                    macro.append(a)
            if not macro:
                continue
            m = float(np.mean(macro))
            key = " + ".join(Path(c).parent.name + ":" + Path(c).stem for c in combo)
            report["fusion"][key] = round(m, 4)
            best.append((m, list(combo)))
    best.sort(reverse=True)
    print("\nbest fusions:")
    for m, combo in best[:8]:
        print(f"  {m:.4f}  {[Path(c).stem for c in combo]}")

    report["best_fusion"] = (
        {"macro": round(best[0][0], 4), "banks": list(best[0][1])} if best else None
    )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
