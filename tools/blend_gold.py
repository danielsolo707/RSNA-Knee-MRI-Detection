"""Gold-58 per-arm blend search + report. Pure CPU.

Reads per-arm row-level prediction files (CSV: StudyInstanceUID + 12 targets,
or npz: pred/ids/truth) plus an optional label file (train.csv carries 58 gold
rows). On the shared labeled rows it:

  * evaluates each arm's macro-AUC vs gold (honest single shared eval),
  * per-target rank-average blends: (a) all arms, (b) greedy best-subset,
  * reports whether any blend beats the single best arm.

Reusable harness for ALL blend work — weak owned arms today, the
DINOsaur-transformer + Raptor-coatnet + maxspan/widedense arms once their
per-arm prediction files are preserved by the inference kernel.

Usage:
  python tools/blend_gold.py \\
      --arm v31=artifacts/v31_runs/gold_oof.csv \\
      --arm v34=artifacts/v34_remote/raptor_gold_v34f0.npz \\
      --gold data/raw/train.csv \\
      -o artifacts/blend_gold_out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

UID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    pos, neg = y == 1, y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = p.argsort(kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    pr = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and pr[j + 1] == pr[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def rankpct(x: np.ndarray) -> np.ndarray:
    o = x.argsort(kind="mergesort")
    r = np.empty_like(o, np.float64)
    r[o] = np.arange(1, len(x) + 1)
    sx = x[o]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r / len(x)


def per_target_macro(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    out = {}
    for i, t in enumerate(TARGETS):
        out[t] = roc_auc(y[:, i], p[:, i])
    vals = [v for v in out.values() if v == v]
    out["macro"] = float(np.mean(vals)) if vals else float("nan")
    return out


def load_arm(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".npz":
        d = np.load(path, allow_pickle=True)
        ids = d["ids"] if "ids" in d else d["StudyInstanceUID"]
        pred = d["pred"] if "pred" in d else d["preds"]
        df = pd.DataFrame(pred, columns=TARGETS)
        df.insert(0, UID, [str(x) for x in ids])
        return df
    return pd.read_csv(path, dtype={UID: str})
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="NAME=PATH (repeatable)")
    ap.add_argument("--gold", default=None, help="train.csv (uses its 58 gold rows)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    arms = {}
    for item in a.arm:
        name, _, p = item.partition("=")
        arms[name] = load_arm(Path(p))
    print("arms:", {n: df.shape[0] for n, df in arms.items()}, flush=True)

    # shared rows across all arms
    common_ids = sorted(set.intersection(*[set(df[UID]) for df in arms.values()]))
    print("rows shared across all arms:", len(common_ids), flush=True)

    label_df = None
    if a.gold:
        lab = pd.read_csv(a.gold, dtype={UID: str})
        lab = lab[lab[TARGETS].notna().all(axis=1)]
        label_df = lab.set_index(UID)
        print("gold rows with labels:", len(label_df), flush=True)

    eval_ids = [u for u in common_ids if u in label_df.index] if label_df is not None else common_ids
    print("eval rows (labeled/shared):", len(eval_ids), flush=True)

    mats = {n: df.set_index(UID).reindex(eval_ids)[TARGETS].astype(np.float64).to_numpy()
            for n, df in arms.items()}
    y = None
    if label_df is not None:
        y = (label_df.reindex(eval_ids)[TARGETS].to_numpy(np.float32) > 0.5).astype(np.float32)

    rep = {"n_eval": len(eval_ids), "arms": {n: p for n, _, p in [x.partition("=") for x in a.arm]}}
    scores = {}
    if y is not None:
        for name, m in mats.items():
            scores[name] = per_target_macro(y, m)
            print(f"{name:12s} macro={scores[name]['macro']:.4f}", flush=True)
        rep["arm_scores"] = scores

        rank_mats = {n: np.stack([rankpct(m[:, i]) for i in range(len(TARGETS))], 1)
                     for n, m in mats.items()}
        names = list(mats)

        # (a) rank-mean of ALL arms
        all_avg = np.mean([rank_mats[n] for n in names], axis=0)
        s_all = per_target_macro(y, all_avg)
        print(f"rank-mean(all) macro={s_all['macro']:.4f}", flush=True)

        # (b) greedy best-subset in rank space
        order = sorted(names, key=lambda n: -scores[n]["macro"])
        cur_avg, cur_n, chosen, best_m = None, 0, [], -1.0
        for n in order:
            trial = rank_mats[n] if cur_avg is None else (cur_avg * cur_n + rank_mats[n]) / (cur_n + 1)
            m = per_target_macro(y, trial)["macro"]
            if m >= best_m - 1e-12:
                best_m, cur_avg, cur_n = m, trial, cur_n + 1
                chosen.append(n)
            else:
                break
        print("greedy best-subset:", chosen, "macro:", best_m, flush=True)

        blend_df = pd.DataFrame({UID: eval_ids})
        if cur_avg is not None:
            for i, t in enumerate(TARGETS):
                blend_df[t] = cur_avg[:, i]
        Path(a.out).mkdir(parents=True, exist_ok=True)
        blend_df.to_csv(Path(a.out) / "blend_predictions.csv", index=False)
        rep.update({"allavg_macro": s_all["macro"], "greedy_subset": chosen,
                    "greedy_macro": best_m, "best_single_macro": max(s["macro"] for s in scores.values())})

    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "blend_report.json").write_text(json.dumps(rep, indent=2, default=str, sort_keys=True),
                                                   encoding="utf-8")
    print("wrote blend_predictions.csv + blend_report.json ->", a.out, flush=True)


if __name__ == "__main__":
    main()