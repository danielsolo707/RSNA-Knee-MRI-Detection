#!/usr/bin/env python3
"""Independent fixed-recipe gate for V13 DINO + E9 + E11.

The candidate is pre-committed from Antoine's E11 contract:
- ten targets: 0.30 DINO + 0.35 E9 Rad-FS + 0.35 E11 Rad-NOFS
- Baker's and Fracture: pure DINO

It is compared against the deployed V9 parent (0.80 DINO + 0.20 E9 on all
12 targets). No weights are fit by this script. It reports fixed grouped-fold
results and a paired study bootstrap over the 58 official image-labelled rows.
"""

from __future__ import annotations

import argparse
import hashlib
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
PRESERVED = {"Baker's", "Fracture"}
UID = "StudyInstanceUID"
EXPECTED_SHA256 = {
    "dino": "62d47ba4c0c8347b5b24e7fd2aa517aae0fd6d4656fd5d829fd3a50f0159909c",
    "e9": "f5b1f02029af98740565bec5b6a76d8471b4b6a6039d09ea410368230cb1f03f",
    "e11": "f5415485f57103fe86370e3d4bba420485a7497ef199d40545a20f6e53a0f0e2",
    "train": "8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33",
    "contract": "7f91e189ccc472b710791730bbbc2d26b3a1d04b45f60a0166445be158a093d3",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc_1d(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    positive = y >= 0.5
    negative = ~positive
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    return float(
        (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def macro_auc(y: np.ndarray, pred: np.ndarray) -> float:
    values = [auc_1d(y[:, j], pred[:, j]) for j in range(y.shape[1])]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def rank_matrix(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(np.asarray(values, dtype=float)).rank(
        method="average", pct=True
    ).to_numpy(float)


def align_oof(path: Path, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, dtype={UID: str})
    expected_columns = [UID, *TARGETS, "fold", "is_gold"]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(f"OOF target/schema order drift in {path}: {frame.columns.tolist()}")
    if frame[UID].duplicated().any():
        raise ValueError(f"duplicate IDs in {path}")
    aligned = pd.DataFrame({UID: ids}).merge(frame, on=UID, how="left", validate="1:1")
    pred = aligned[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(pred).all():
        raise ValueError(f"missing/non-finite predictions after aligning {path}")
    folds = aligned["fold"].to_numpy(int)
    return pred, folds


def build_predictions(
    dino: np.ndarray, e9: np.ndarray, e11: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dino_r = rank_matrix(dino)
    e9_r = rank_matrix(e9)
    e11_r = rank_matrix(e11)
    v9 = 0.80 * dino_r + 0.20 * e9_r
    candidate = 0.30 * dino_r + 0.35 * e9_r + 0.35 * e11_r
    for j, target in enumerate(TARGETS):
        if target in PRESERVED:
            candidate[:, j] = dino_r[:, j]
    return v9, candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13035)
    parser.add_argument("--min-gain", type=float, default=0.005)
    parser.add_argument("--min-positive-folds", type=int, default=3)
    parser.add_argument("--min-bootstrap-win", type=float, default=0.80)
    parser.add_argument("--max-soft-drop", type=float, default=0.005)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    paths = {
        "dino": root / "research" / "pilkwang_oof" / "oof.npz",
        "e9": root / "references" / "candidates" / "antoine_e9_files" / "v52_oof.csv",
        "e11": root / "references" / "candidates" / "antoine_e11_files" / "v52_e11_oof.csv",
        "train": root / "data" / "raw" / "train.csv",
        "contract": root / "references" / "candidates" / "antoine_e11_files" / "e11_contract.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    observed_sha = {name: sha256(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED_SHA256[name], "observed": observed_sha[name]}
        for name in paths
        if observed_sha[name] != EXPECTED_SHA256[name]
    }
    if mismatches:
        raise ValueError(f"artifact SHA-256 mismatch: {mismatches}")

    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    default_name = contract.get("default_configuration")
    default_weights = (contract.get("configurations", {}).get(default_name, {}).get("weights"))
    expected_voted = [target for target in TARGETS if target not in PRESERVED]
    if (
        default_name != "parent035_e11035"
        or default_weights != {"e11": 0.35, "e2": 0.30000000000000004, "parent": 0.35}
        or contract.get("targets") != TARGETS
        or contract.get("preserved_targets") != ["Baker's", "Fracture"]
        or contract.get("voted_targets") != expected_voted
        or contract.get("rank_basis") != "rank each arm within the scored population, independently per target"
    ):
        raise ValueError("external E11 contract does not match the fixed V13 recipe")

    with np.load(paths["dino"], allow_pickle=True) as dino_npz:
        ids = [str(value) for value in dino_npz["ids"].tolist()]
        dino = np.asarray(dino_npz["pred"], dtype=float)
        derived = np.asarray(dino_npz["y_derived"], dtype=float)
        gold_mask_npz = np.asarray(dino_npz["gold_mask"], dtype=bool)
        dino_targets = dino_npz["targets"].astype(str).tolist()
    if dino_targets != TARGETS:
        raise ValueError(f"DINO target order drift: {dino_targets}")
    if dino.shape != (len(ids), len(TARGETS)):
        raise ValueError(f"unexpected DINO shape {dino.shape}")

    e9, e9_folds = align_oof(paths["e9"], ids)
    e11, e11_folds = align_oof(paths["e11"], ids)
    if not np.array_equal(e9_folds, e11_folds):
        raise ValueError("E9 and E11 fold assignments differ")

    train = pd.read_csv(paths["train"], dtype={UID: str})
    train = pd.DataFrame({UID: ids}).merge(train, on=UID, how="left", validate="1:1")
    gold_mask = train[TARGETS].notna().all(axis=1).to_numpy()
    if not np.array_equal(gold_mask, gold_mask_npz):
        raise ValueError("train.csv gold mask differs from DINO OOF contract")
    y_gold = train.loc[gold_mask, TARGETS].to_numpy(float)

    # Rank within the population being scored. The external contract evaluates
    # the 58 official labels as their own scored population; the soft check uses
    # all 4,407 OOF rows. Ranking all rows and then subsetting would change the
    # convex rank blend even though each arm's standalone AUC is unchanged.
    v9_all, candidate_all = build_predictions(dino, e9, e11)
    v9_gold, candidate_gold = build_predictions(
        dino[gold_mask], e9[gold_mask], e11[gold_mask]
    )
    gold_folds = e9_folds[gold_mask]

    v9_macro = macro_auc(y_gold, v9_gold)
    candidate_macro = macro_auc(y_gold, candidate_gold)
    gain = candidate_macro - v9_macro
    per_target = {}
    for j, target in enumerate(TARGETS):
        parent_auc = auc_1d(y_gold[:, j], v9_gold[:, j])
        candidate_auc = auc_1d(y_gold[:, j], candidate_gold[:, j])
        per_target[target] = {
            "v9": parent_auc,
            "candidate": candidate_auc,
            "delta": candidate_auc - parent_auc,
        }

    fold_results = []
    for fold in sorted(np.unique(gold_folds).tolist()):
        mask = gold_folds == fold
        parent_score = macro_auc(y_gold[mask], v9_gold[mask])
        candidate_score = macro_auc(y_gold[mask], candidate_gold[mask])
        fold_results.append(
            {
                "fold": int(fold),
                "rows": int(mask.sum()),
                "v9": parent_score,
                "candidate": candidate_score,
                "gain": candidate_score - parent_score,
            }
        )
    positive_folds = sum(row["gain"] > 0 for row in fold_results)

    rng = np.random.default_rng(args.seed)
    bootstrap_delta = []
    for _ in range(args.bootstrap):
        sample = rng.integers(0, len(y_gold), size=len(y_gold))
        parent_score = macro_auc(y_gold[sample], v9_gold[sample])
        candidate_score = macro_auc(y_gold[sample], candidate_gold[sample])
        if np.isfinite(parent_score) and np.isfinite(candidate_score):
            bootstrap_delta.append(candidate_score - parent_score)
    bootstrap_delta = np.asarray(bootstrap_delta, dtype=float)
    bootstrap_win = float(np.mean(bootstrap_delta > 0))

    soft_v9 = macro_auc(derived, v9_all)
    soft_candidate = macro_auc(derived, candidate_all)
    promote = bool(
        gain >= args.min_gain
        and positive_folds >= args.min_positive_folds
        and bootstrap_win >= args.min_bootstrap_win
        and soft_candidate >= soft_v9 - args.max_soft_drop
    )

    receipt = {
        "status": "PROMOTE" if promote else "REJECT",
        "promote": promote,
        "recipe": {
            "voted_targets": [target for target in TARGETS if target not in PRESERVED],
            "preserved_targets": sorted(PRESERVED),
            "weights": {"dino": 0.30, "e9_rad_fs": 0.35, "e11_rad_nofs": 0.35},
            "selection": "fixed external E11 default; no local weight fitting",
        },
        "n_gold": int(gold_mask.sum()),
        "v9_gold_macro": v9_macro,
        "candidate_gold_macro": candidate_macro,
        "gold_gain": gain,
        "soft_v9_macro": soft_v9,
        "soft_candidate_macro": soft_candidate,
        "soft_delta": soft_candidate - soft_v9,
        "fold_results": fold_results,
        "positive_folds": int(positive_folds),
        "bootstrap": {
            "seed": args.seed,
            "requested_draws": args.bootstrap,
            "valid_draws": int(len(bootstrap_delta)),
            "p_candidate_wins": bootstrap_win,
            "mean_delta": float(np.mean(bootstrap_delta)),
            "q025": float(np.quantile(bootstrap_delta, 0.025)),
            "q500": float(np.quantile(bootstrap_delta, 0.5)),
            "q975": float(np.quantile(bootstrap_delta, 0.975)),
        },
        "thresholds": {
            "min_gain": args.min_gain,
            "min_positive_folds": args.min_positive_folds,
            "min_bootstrap_win": args.min_bootstrap_win,
            "max_soft_drop": args.max_soft_drop,
        },
        "per_target": per_target,
        "external_contract_default": contract.get("default_configuration"),
        "external_contract_nested": contract.get("nested_selection"),
        "artifacts": {name: {"path": str(path), "sha256": observed_sha[name]} for name, path in paths.items()},
        "evidence_boundary": "OOF diagnostics only; not a Kaggle leaderboard score.",
    }

    output = args.output or (root / "artifacts" / "v13_three_arm_gate.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "v9_gold_macro": v9_macro,
        "candidate_gold_macro": candidate_macro,
        "gold_gain": gain,
        "positive_folds": positive_folds,
        "bootstrap_win": bootstrap_win,
        "soft_delta": soft_candidate - soft_v9,
        "output": str(output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
