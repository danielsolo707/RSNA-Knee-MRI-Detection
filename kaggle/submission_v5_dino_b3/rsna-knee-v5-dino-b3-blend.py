#!/usr/bin/env python3
"""V5 inference: rank-blend public DINO anchor with trained B3 folds when present.

Design:
  * Always produce a valid submission under the 9h budget.
  * If B3 checkpoints are missing/incomplete, fall back to DINO-only (0.899 path
    must be provided by attaching the same assets as submission_v1).
  * When B3 folds 0..4 are present, per-target rank-mean with DINO (equal weight
    by default; optional weight JSON freezes OOF-chosen weights).

This script is the competition-entry surface. Training happens offline in
v5_production fold kernels.
"""

from __future__ import annotations

import json
import os
import time
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


def rank_pct_matrix(pred: np.ndarray) -> np.ndarray:
    out = np.zeros_like(pred, dtype=np.float64)
    for t in range(pred.shape[1]):
        values = pred[:, t]
        order = values.argsort(kind="mergesort")
        ranks = np.empty(len(values), dtype=np.float64)
        ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
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
        out[:, t] = ranks / len(values)
    return out


def find_one(pattern: str) -> Path | None:
    hits = sorted(Path("/kaggle/input").glob(pattern)) if Path("/kaggle/input").exists() else []
    return hits[0] if hits else None


def load_weight_table(path: Path | None) -> dict[str, dict[str, float]] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("weights", payload)


def blend_rank(
    members: dict[str, np.ndarray],
    weights: dict[str, dict[str, float]] | None,
) -> np.ndarray:
    """members: name -> [N, 12] probabilities. Returns blended ranks [N, 12]."""
    ranked = {name: rank_pct_matrix(mat) for name, mat in members.items()}
    n = next(iter(members.values())).shape[0]
    out = np.zeros((n, len(TARGETS)), dtype=np.float64)
    for t, target in enumerate(TARGETS):
        if weights is None:
            # equal weight over available members
            for mat in ranked.values():
                out[:, t] += mat[:, t]
            out[:, t] /= max(len(ranked), 1)
        else:
            total = 0.0
            for name, mat in ranked.items():
                w = float(weights.get(target, {}).get(name, 0.0))
                if w:
                    out[:, t] += w * mat[:, t]
                    total += w
            if total <= 0:
                for mat in ranked.values():
                    out[:, t] += mat[:, t]
                out[:, t] /= max(len(ranked), 1)
            else:
                out[:, t] /= total
    return out


def main() -> None:
    t0 = time.time()
    # Placeholder: the full DINO + B3 decode path is wired when B3 checkpoints
    # land. Until then this module documents the blend contract and can merge
    # precomputed prediction CSVs attached as datasets (debug / dry-run).
    dino_csv = find_one("**/dino_submission.csv") or find_one("**/submission_dino.csv")
    b3_csv = find_one("**/b3_submission.csv") or find_one("**/submission_b3.csv")
    weight_json = find_one("**/blend_report.json") or find_one("**/blend_weights.json")

    test_csv = find_one("**/test.csv")
    if test_csv is None:
        raise FileNotFoundError("test.csv not found under /kaggle/input")
    test = pd.read_csv(test_csv)
    uids = test[UID].astype(str).tolist()

    members: dict[str, np.ndarray] = {}
    if dino_csv is not None:
        dino = pd.read_csv(dino_csv).set_index(UID).reindex(uids)[TARGETS].to_numpy(float)
        members["dino"] = dino
    if b3_csv is not None:
        b3 = pd.read_csv(b3_csv).set_index(UID).reindex(uids)[TARGETS].to_numpy(float)
        members["b3"] = b3

    if not members:
        # Fail closed with a clearly invalid constant only in local dry-run;
        # competition notebooks must attach at least the DINO path assets.
        raise RuntimeError(
            "No prediction sources found. Attach DINO submission assets and/or "
            "B3 fold predictions, or run the full decode path notebook cells."
        )

    weights = load_weight_table(weight_json)
    blended = blend_rank(members, weights)
    # Convert ranks back to (0,1) scores; rank order is preserved which is what AUC uses.
    out = pd.DataFrame({UID: uids})
    for i, target in enumerate(TARGETS):
        out[target] = blended[:, i]
    out_path = Path("/kaggle/working/submission.csv")
    out.to_csv(out_path, index=False)
    meta = {
        "members": list(members.keys()),
        "weights_source": str(weight_json) if weight_json else None,
        "rows": len(out),
        "elapsed_seconds": time.time() - t0,
    }
    Path("/kaggle/working/v5_blend_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
