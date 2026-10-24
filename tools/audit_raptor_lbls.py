"""Audit Raptor LLM soft labels on the training corpus.

Raptor labels cover 4,349 non-gold rows (gold held out), so there is nothing to
join on the 58-gold set. The honest audit here is:
  * positive rate per target (pathology detection),
  * agreement vs each legacy label bank over the shared 4,3xx rows,
  * per-target rate shift that might signal a systematic labeling bias.

Writes artifacts/raptor_labels/raptor_label_audit.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
     "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
     "Baker's", "Contusion", "Fracture"]

RAPTOR = REPO / "artifacts" / "raptor_labels" / "labels_llm_soft.csv"
SOURCES = {
    "report_v2": REPO / "data/labels/pilkwang/report_labels_v2.csv",
    "llm_v2": REPO / "data/labels/stevenleehans/llm_labels_v2.csv",
    "llm_full": REPO / "data/labels/stevenleehans/llm_labels_full.csv",
    "llm_v4blend": REPO / "data/labels/stevenleehans/llm_labels_v4_blend.csv",
    "gpt56sol_llm": REPO / "data/labels/lixin73/labels_llm_gpt56sol.csv",
    "gpt56sol_report": REPO / "data/labels/lixin73/report_labels_gpt56sol.csv",
}
OUT = REPO / "artifacts" / "raptor_labels" / "raptor_label_audit.json"


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    for t in T:
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce").clip(0, 1)
    return df[[c for c in T if c in df.columns]]


def main() -> None:
    rap = load(RAPTOR)
    print(f"raptor rows: {len(rap)}", flush=True)

    banks = {n: load(p) for n, p in SOURCES.items() if p.is_file()}
    for n, df in banks.items():
        print(f"{n}: {len(df)} rows", flush=True)

    rows = []
    for t in T:
        rp = rap[t]
        entry = {"target": t, "n_raptor": int(rp.notna().sum()),
                 "raptor_pos_rate": round(float((rp > 0.5).mean()), 3),
                 "raptor_mean": round(float(rp.mean()), 3),
                 "raptor_soft": {"p20": round(float(rp.quantile(0.2)), 3),
                                 "p50": round(float(rp.quantile(0.5)), 3),
                                 "p80": round(float(rp.quantile(0.8)), 3)}}
        for n, df in banks.items():
            if t not in df.columns:
                continue
            common = rap.index.intersection(df.index)
            a = (rap.loc[common, t] > 0.5).astype(int)
            b = (df.loc[common, t] > 0.5).astype(int)
            entry[n + "_agree"] = round(float((a == b).mean()), 3)
            entry[n + "_pos"] = round(float((b > 0).mean()), 3)
        rows.append(entry)
        line = f"{t:17s} rap-pos={entry['raptor_pos_rate']:.2f} mean={entry['raptor_mean']:.2f}"
        for n in banks:
            if n + "_agree" in entry:
                line += f" | {n}: agree={entry[n+'_agree']:.2f}"
        print(line, flush=True)

    OUT.write_text(json.dumps({"per_target": rows, "src": str(RAPTOR),
                               "note": "Raptor labels hold the 58 gold rows out; agreement is over shared train rows only."},
                              indent=2), encoding="utf-8")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()