"""Transform folds_v3.csv -> the raptor trainer's fold_file JSON.

The raptor trainer (tools/train_raptor.py) reads --fold_file as a JSON with a
"folds" key mapping StudyInstanceUID -> fold int (0..folds-1). folds_v3.csv
already has that mapping (report_group-based GroupKFold 5). This just re-serializes
it in the format the owned-trainer (V35+) expects, so the 5-fold owned family can
reuse the established split. Pure CPU.

Writes artifacts/kaggle_dataset_v32_labels/folds_raptor.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "artifacts" / "kaggle_dataset_v32_labels" / "folds_v3.csv"
DST = REPO / "artifacts" / "kaggle_dataset_v32_labels" / "folds_raptor.json"


def main() -> None:
    df = pd.read_csv(SRC, dtype={"StudyInstanceUID": str, "fold": int})
    mapping = {u: int(f) for u, f in zip(df["StudyInstanceUID"], df["fold"])}
    n = len(mapping)
    counts = df["fold"].value_counts().sort_index()
    out = {"folds": mapping, "n": n, "nfolds": 5, "counts": {str(k): int(v) for k, v in counts.items()}}
    DST.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {DST} | {n} studies -> {dict(counts)}", flush=True)


if __name__ == "__main__":
    main()