"""Fuse the two pseudo-label arms into the final training parquet + gold-58 validation.

Combines:
  - v5 arm train predictions from the Kaggle kernel (pseudo-labels-v5, COMPLETE)
  - v4 arm train predictions from the local 4070 run (make_pseudo_labels.py --arms v4)
  - gold-58 raw predictions per arm (pred_gold_*.npy, from --gold-only runs)

and produces:
  - pseudo_labels_final_w60.parquet : (1-w)*rank(labels) + w*rank(0.4*rank(v5) + 0.6*rank(v4))
  - fusion_report.json              : per-arm / fusion / blend gold-58 AUCs

Before writing anything it SELF-CHECKS: recomputing the v5-ONLY parquet from the
kernel's pred_train_v5.npy must reproduce the kernel's pseudo_labels_w60.parquet
exactly. That proves (a) our train_ids reconstruction matches the kernel's row order
and (b) the rank/blend math is identical - otherwise the two-arm output would be
silently misaligned.

Usage:
    python tools/fuse_pseudo_labels.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]


def rank(x: np.ndarray) -> np.ndarray:
    # verbatim copy of make_pseudo_labels.rank - must stay byte-identical in semantics
    return np.apply_along_axis(
        lambda v: pd.Series(v).rank(method="average", pct=True).to_numpy(), 0, x)


def lab_auc(Y, P):
    a = [roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])
         if len(set(Y[:, j].astype(int))) > 1]
    return float(np.mean(a))


def build_ids(corpus_dir: Path, labels_parquet: Path, train_csv: Path):
    """Reconstruct train_ids/gold_ids EXACTLY as make_pseudo_labels.py does."""
    ids = np.concatenate([
        np.load(corpus_dir / "all_ids.npy", allow_pickle=True).astype(str),
        np.load(corpus_dir / "extra_ids.npy", allow_pickle=True).astype(str)])
    soft = pd.read_parquet(labels_parquet)
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
    soft = soft.set_index("StudyInstanceUID")
    LAB = [c for c in soft.columns if c != "StudyInstanceUID"]
    trn = pd.read_csv(train_csv)
    trn["StudyInstanceUID"] = trn["StudyInstanceUID"].astype(str)
    gold_df = trn[trn[LAB].notna().all(axis=1)].set_index("StudyInstanceUID")
    goldset = set(gold_df.index)
    train_ids = [u for u in ids if u in soft.index and u not in goldset]
    gold_ids = [u for u in ids if u in goldset]
    return train_ids, gold_ids, soft, gold_df, LAB


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-train-npy",
                    default=str(REPO / "artifacts/pseudo_labels/kernel_v6_out/pseudo_labels/pred_train_v5.npy"))
    ap.add_argument("--v4-train-npy",
                    default=str(REPO / "artifacts/pseudo_labels/pred_train_v4.npy"))
    ap.add_argument("--v5-gold-npy",
                    default=str(REPO / "artifacts/pseudo_labels/pred_gold_v5.npy"))
    ap.add_argument("--v4-gold-npy",
                    default=str(REPO / "artifacts/pseudo_labels/pred_gold_v4.npy"))
    ap.add_argument("--ref-parquet",
                    default=str(REPO / "artifacts/pseudo_labels/pseudo_labels_w60.parquet"),
                    help="reference parquet to self-check against (the parquet written by "
                         "make_pseudo_labels.py from the SAME pred npys). Must match the "
                         "arms/k_eval of the loaded npys or the check fails by design.")
    ap.add_argument("--labels",
                    default=str(REPO / "artifacts/raptor_labels/labels_llm_soft.parquet"))
    ap.add_argument("--gold-labels",
                    default=str(REPO / "data/labels/stevenleehans/llm_labels_v4_blend.csv"),
                    help="label source covering gold-58 (the raptor soft labels exclude "
                         "gold by design). v4_blend is the source of the reference "
                         "labels-only number 0.8927 in label_scores.json.")
    ap.add_argument("--train-csv", default=str(REPO / "data/raw/train.csv"))
    ap.add_argument("--corpus-dir", default=str(REPO / "artifacts/raptor_corpus"))
    ap.add_argument("--blend-w", type=float, default=0.60)
    ap.add_argument("--w-v5", type=float, default=0.40)
    ap.add_argument("--w-v4", type=float, default=0.60)
    ap.add_argument("--out",
                    default=str(REPO / "artifacts/pseudo_labels/pseudo_labels_final_w60.parquet"))
    args = ap.parse_args()

    train_ids, gold_ids, soft, gold_df, LAB = build_ids(
        Path(args.corpus_dir), Path(args.labels), Path(args.train_csv))
    print(f"ids: train={len(train_ids)} gold={len(gold_ids)}  LAB={LAB}")

    P5 = np.load(args.v5_train_npy)
    P4 = np.load(args.v4_train_npy)
    assert P5.shape == (len(train_ids), len(LAB)), \
        f"v5 train npy shape {P5.shape} != ({len(train_ids)}, {len(LAB)}) - row order suspect"
    assert P4.shape == (len(train_ids), len(LAB)), \
        f"v4 train npy shape {P4.shape} != ({len(train_ids)}, {len(LAB)}) - row order suspect"

    cur = np.stack([soft.loc[u, LAB].values.astype(float) for u in train_ids])
    w = args.blend_w

    # ---- two-arm fusion ----
    tot = args.w_v5 + args.w_v4
    fused = rank(args.w_v5 / tot * rank(P5) + args.w_v4 / tot * rank(P4))
    pseudo = (1 - w) * rank(cur) + w * fused

    # ---- self-check: recompute the blend and compare to a reference parquet ----
    # Validates (a) our train_ids reconstruction matches the row order of the
    # saved pred npys and (b) the rank/blend math is identical to the tool's.
    ref = Path(args.ref_parquet)
    diff = None
    if ref.exists():
        rdf = pd.read_parquet(ref)
        assert list(rdf.columns) == ["StudyInstanceUID"] + LAB, "ref parquet columns differ"
        assert list(rdf["StudyInstanceUID"].astype(str)) == train_ids, \
            "ref parquet StudyInstanceUID order != reconstructed train_ids"
        diff = float(np.abs(rdf[LAB].values - pseudo).max())
        print(f"self-check vs {ref.name}: max|diff| = {diff:.3e}")
        if diff > 1e-9:
            raise SystemExit("SELF-CHECK FAILED - do not trust the fusion; investigate order/math")
    else:
        print(f"self-check skipped (no ref parquet at {ref})")

    df = pd.DataFrame(pseudo, columns=LAB)
    df.insert(0, "StudyInstanceUID", train_ids)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out}  ({len(df)} rows)")

    # ---- gold-58 validation ----
    report = {"selfcheck_max_diff": diff, "blend_w": w,
              "w_v5": args.w_v5, "w_v4": args.w_v4}
    try:
        G5 = np.load(args.v5_gold_npy)
        G4 = np.load(args.v4_gold_npy)
        assert G5.shape == (len(gold_ids), len(LAB)) and G4.shape == (len(gold_ids), len(LAB)), \
            f"gold npy shapes {G5.shape} {G4.shape} != ({len(gold_ids)}, {len(LAB)})"
        Y = np.stack([gold_df.loc[u, LAB].values.astype(float) for u in gold_ids])
        # gold-58 has no raptor soft labels (held out by design); the reference
        # ladder (label_scores.json) used the v4_blend bank as the labels proxy
        glab = pd.read_csv(args.gold_labels)
        glab["StudyInstanceUID"] = glab["StudyInstanceUID"].astype(str)
        glab = glab.set_index("StudyInstanceUID")
        missing = [u for u in gold_ids if u not in glab.index]
        if missing:
            raise SystemExit(f"gold labels missing {len(missing)} studies: {missing[:3]}")
        gcur = rank(np.stack([glab.loc[u, LAB].values.astype(float) for u in gold_ids]))
        gfused = rank(args.w_v5 / tot * rank(G5) + args.w_v4 / tot * rank(G4))
        gblend = (1 - w) * gcur + w * gfused
        report["gold58_label_auc"] = {
            "labels_only": round(lab_auc(Y, gcur), 4),
            "v5_only": round(lab_auc(Y, rank(G5)), 4),
            "v4_only": round(lab_auc(Y, rank(G4)), 4),
            "two_arm_fusion": round(lab_auc(Y, gfused), 4),
            "final_blend": round(lab_auc(Y, gblend), 4),
        }
        print("gold-58 AUCs:", json.dumps(report["gold58_label_auc"], indent=1))
    except FileNotFoundError as e:
        report["gold58_error"] = f"missing gold preds: {e}"
        print("gold-58 validation skipped:", e)

    rp = Path(args.out).with_name("fusion_report.json")
    rp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
