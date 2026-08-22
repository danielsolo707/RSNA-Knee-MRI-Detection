"""Label audit: 6 sources x 12 targets vs the 58 organizer-gold rows.

Answers: which SOURCES are trustworthy per TARGET, where do sources disagree,
and does the current 3-source mean bank beat single sources?
Pure CPU/pandas - no training involved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
           "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]

SOURCES = {
    "report_v2": REPO / "data/labels/pilkwang/report_labels_v2.csv",
    "llm_v2": REPO / "data/labels/stevenleehans/llm_labels_v2.csv",
    "llm_full": REPO / "data/labels/stevenleehans/llm_labels_full.csv",
    "llm_v4blend": REPO / "data/labels/stevenleehans/llm_labels_v4_blend.csv",
    "gpt56sol_llm": REPO / "data/labels/lixin73/labels_llm_gpt56sol.csv",
    "gpt56sol_report": REPO / "data/labels/lixin73/report_labels_gpt56sol.csv",
}


def load_source(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        print(f"MISSING: {path}")
        return None
    df = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    keep = ["StudyInstanceUID"] + [t for t in TARGETS if t in df.columns]
    df = df[keep].copy()
    for t in TARGETS:
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce").clip(0, 1)
    return df.set_index("StudyInstanceUID")


def main() -> None:
    train = pd.read_csv(REPO / "data/raw/train.csv", dtype={"StudyInstanceUID": str})
    gold = train.dropna(subset=TARGETS, how="all").set_index("StudyInstanceUID")
    for t in TARGETS:
        gold[t] = pd.to_numeric(gold[t], errors="coerce")
    print(f"gold rows: {len(gold)}")

    banks = {}
    for name, path in SOURCES.items():
        df = load_source(path)
        if df is not None:
            banks[name] = df
            print(f"{name}: {len(df)} studies, cols={[c for c in df.columns if c != 'StudyInstanceUID'][:3]}...")

    # positive rates over the full corpus
    print("\n=== positive rate over full corpus (n~4407) ===")
    rows = []
    for name, df in banks.items():
        row = {"source": name}
        for t in TARGETS:
            if t in df.columns:
                row[t] = round(float((df[t] > 0.5).mean()), 3)
        rows.append(row)
    pr = pd.DataFrame(rows).set_index("source")
    print(pr[TARGETS].to_string())

    # per source x target: AUC + accuracy vs gold
    print("\n=== vs GOLD-58: AUC | acc(binarized) | n_pos_gold ===")
    header = f"{'target':<17s}" + "".join(f"{s:>16s}" for s in banks)
    print(header)
    auc_tab = {s: {} for s in banks}
    for t in TARGETS:
        gy = gold[t].dropna()
        if len(gy) < 5 or gy.nunique() < 2:
            print(f"{t:<17s}" + "".join(f"{'--':>16s}" for _ in banks))
            continue
        gpos = int((gy > 0.5).sum())
        line = f"{t:<17s}"
        for name, df in banks.items():
            common = gy.index.intersection(df.index)
            if len(common) < 5 or common.nunique() < 5:
                line += f"{'--':>16s}"
                continue
            y = gy.loc[common]
            p = df.loc[common, t]
            m = y.notna() & p.notna()
            y, p = y[m], p[m]
            if y.nunique() < 2:
                line += f"{'--':>16s}"
                continue
            try:
                auc = roc_auc_score((y > 0.5).astype(int), p)
                acc = float(((p > 0.5) == (y > 0.5)).mean())
                auc_tab[name][t] = round(auc, 3)
                line += f"{auc:>9.3f}/{acc:.2f}"
            except Exception:
                line += f"{'ERR':>16s}"
        line += f"  npos={gpos}"
        print(line)

    print("\n=== macro AUC over evaluable targets ===")
    for name, d in auc_tab.items():
        if d:
            vals = list(d.values())
            print(f"{name:>16s}: {np.mean(vals):.4f} (over {len(vals)} targets)")

    # pairwise inter-source disagreement per target (binarized, full corpus)
    print("\n=== pairwise mean disagreement rate (full corpus, binarized) ===")
    names = list(banks)
    print(f"{'':>16s}" + "".join(f"{n:>14s}" for n in names))
    for a in names:
        line = f"{a:>16s}"
        for b in names:
            if a == b:
                line += f"{'-':>14s}"
                continue
            da, db_ = banks[a], banks[b]
            common = da.index.intersection(db_.index)
            diffs = []
            for t in TARGETS:
                if t in da.columns and t in db_.columns:
                    xa = (da.loc[common, t] > 0.5).astype(int)
                    xb = (db_.loc[common, t] > 0.5).astype(int)
                    diffs.append(float((xa != xb).mean()))
            line += f"{np.mean(diffs):>13.3f}"
        print(line)

    # current production bank = mean of 3
    trio = [n for n in ("report_v2", "llm_v2", "gpt56sol_llm") if n in banks]
    if len(trio) >= 2:
        idx = banks[trio[0]].index
        for n in trio[1:]:
            idx = idx.union(banks[n].index)
        prod = pd.DataFrame(index=idx)
        for t in TARGETS:
            cols = [banks[n][t] for n in trio if t in banks[n].columns]
            prod[t] = pd.concat(cols, axis=1).mean(axis=1)
        print(f"\n=== CURRENT BANK (mean of {trio}) vs gold-58 ===")
        aucs = []
        for t in TARGETS:
            gy = gold[t].dropna()
            common = gy.index.intersection(prod.index)
            y = gy.loc[common]
            p = prod.loc[common, t]
            m = y.notna() & p.notna()
            if y[m].nunique() < 2:
                continue
            auc = roc_auc_score((y[m] > 0.5).astype(int), p[m])
            aucs.append(auc)
            print(f"{t:<17s} AUC={auc:.3f}")
        if aucs:
            print(f"{'MACRO':<17s} AUC={np.mean(aucs):.4f}")


if __name__ == "__main__":
    main()
