"""Build the leakage-aware V3 soft-label bank and grouped five-fold split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


UID = "StudyInstanceUID"
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
SOURCES = {
    "lixin": Path("data/labels/lixin73/report_labels_gpt56sol.csv"),
    "pilkwang": Path("data/labels/pilkwang/report_labels_v2.csv"),
    "steven2": Path("data/labels/stevenleehans/llm_labels_v2.csv"),
    "steven4": Path("data/labels/stevenleehans/llm_labels_v4_blend.csv"),
}
SOURCE_GAP_GATE = 0.015


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = (start + end + 1) / 2.0
        start = end
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def normalise_report(report: str) -> str:
    return re.sub(r"\s+", " ", str(report).strip().lower())


def report_group(report: str) -> str:
    return hashlib.sha1(normalise_report(report).encode("utf-8")).hexdigest()


def build_folds(
    train_rows: list[dict[str, str]],
    soft: np.ndarray,
    gold_mask: np.ndarray,
    folds: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """Greedily balance duplicate-report groups, gold positives and weak positives."""
    groups: dict[str, list[int]] = defaultdict(list)
    hashes: list[str] = []
    for index, row in enumerate(train_rows):
        key = report_group(row.get("Report", ""))
        hashes.append(key)
        groups[key].append(index)

    hard = (soft > 0.5).astype(np.float64)
    group_records = []
    for key, indices in groups.items():
        idx = np.asarray(indices, dtype=int)
        gold_pos = np.zeros(len(TARGETS), dtype=np.float64)
        for row_index in idx[gold_mask[idx]]:
            gold_pos += np.array(
                [float(train_rows[row_index][target]) for target in TARGETS],
                dtype=np.float64,
            )
        weak_pos = hard[idx].sum(axis=0)
        rarity = float((gold_pos * 30.0 + weak_pos).sum())
        group_records.append((key, indices, gold_pos, weak_pos, rarity))

    group_records.sort(
        key=lambda item: (
            -float(item[2].sum() > 0),
            -item[4],
            -len(item[1]),
            item[0],
        )
    )

    total_gold = sum((item[2] for item in group_records), np.zeros(len(TARGETS)))
    total_weak = sum((item[3] for item in group_records), np.zeros(len(TARGETS)))
    target_size = len(train_rows) / folds
    target_gold = total_gold / folds
    target_weak = total_weak / folds
    fold_size = np.zeros(folds, dtype=np.float64)
    fold_gold = np.zeros((folds, len(TARGETS)), dtype=np.float64)
    fold_weak = np.zeros((folds, len(TARGETS)), dtype=np.float64)
    assignments: dict[str, int] = {}

    gold_scale = np.maximum(target_gold, 1.0)
    weak_scale = np.maximum(target_weak, 1.0)
    for key, indices, gold_pos, weak_pos, _ in group_records:
        scores = []
        for fold in range(folds):
            size_before = ((fold_size[fold] - target_size) / target_size) ** 2
            size_after = ((fold_size[fold] + len(indices) - target_size) / target_size) ** 2
            size_cost = size_after - size_before
            gold_before = np.mean(((fold_gold[fold] - target_gold) / gold_scale) ** 2)
            gold_after = np.mean(
                ((fold_gold[fold] + gold_pos - target_gold) / gold_scale) ** 2
            )
            gold_cost = gold_after - gold_before
            weak_before = np.mean(((fold_weak[fold] - target_weak) / weak_scale) ** 2)
            weak_after = np.mean(
                ((fold_weak[fold] + weak_pos - target_weak) / weak_scale) ** 2
            )
            weak_cost = weak_after - weak_before
            scores.append(5.0 * gold_cost + weak_cost + 0.5 * size_cost)
        chosen = min(range(folds), key=lambda fold: (scores[fold], fold_size[fold], fold))
        assignments[key] = chosen
        fold_size[chosen] += len(indices)
        fold_gold[chosen] += gold_pos
        fold_weak[chosen] += weak_pos

    return np.asarray([assignments[key] for key in hashes], dtype=np.int8), hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v3_labels"))
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    if len(train_rows) != 4407:
        raise ValueError(f"expected 4,407 training rows, found {len(train_rows)}")
    train_ids = [row[UID] for row in train_rows]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("train.csv contains duplicate StudyInstanceUID values")
    train_set = set(train_ids)

    source_rows: dict[str, dict[str, dict[str, str]]] = {}
    input_hashes = {str(args.train): sha256(args.train)}
    for name, path in SOURCES.items():
        rows = read_rows(path)
        mapped = {row[UID]: row for row in rows}
        if len(mapped) != len(rows):
            raise ValueError(f"{name} contains duplicate StudyInstanceUID values")
        if not set(mapped).issubset(train_set):
            raise ValueError(f"{name} contains IDs outside train.csv")
        for row in rows:
            for target in TARGETS:
                value = float(row[target])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name}/{row[UID]}/{target}: invalid {value}")
        source_rows[name] = mapped
        input_hashes[str(path)] = sha256(path)

    gold_mask = np.asarray(
        [all(row.get(target, "") != "" for target in TARGETS) for row in train_rows],
        dtype=bool,
    )
    if int(gold_mask.sum()) != 58:
        raise ValueError(f"expected 58 gold rows, found {int(gold_mask.sum())}")
    gold_indices = np.flatnonzero(gold_mask)
    gold_y = np.asarray(
        [[float(train_rows[i][target]) for target in TARGETS] for i in gold_indices],
        dtype=np.float64,
    )

    source_auc: dict[str, dict[str, float]] = {name: {} for name in SOURCES}
    source_gold_pred: dict[str, np.ndarray] = {}
    for name, mapped in source_rows.items():
        pred = np.full((len(gold_indices), len(TARGETS)), 0.5, dtype=np.float64)
        for row_pos, row_index in enumerate(gold_indices):
            uid = train_ids[row_index]
            if uid in mapped:
                pred[row_pos] = [float(mapped[uid][target]) for target in TARGETS]
        source_gold_pred[name] = pred
        for target_index, target in enumerate(TARGETS):
            source_auc[name][target] = auc(gold_y[:, target_index], pred[:, target_index])

    # Target-wise selection is deliberately simpler than a 286-point weight search.
    # With only 58 gold rows, choosing one stable source per target is less likely to
    # fit noise than a dense set of target-specific convex weights.
    selected_source: dict[str, str] = {}
    for target in TARGETS:
        selected_source[target] = max(
            SOURCES, key=lambda name: source_auc[name][target]
        )

    # Leave-one-out source selection is the label gate. Each gold row is scored using
    # a source chosen without that row's label, even though final non-gold training
    # labels use the more stable source choice fitted on all 58 rows.
    loo_pred = np.zeros_like(gold_y)
    loo_choice_counts: dict[str, dict[str, int]] = {
        target: {name: 0 for name in SOURCES} for target in TARGETS
    }
    for target_index, target in enumerate(TARGETS):
        for holdout in range(len(gold_indices)):
            keep = np.arange(len(gold_indices)) != holdout
            best_name = max(
                SOURCES,
                key=lambda name: auc(
                    gold_y[keep, target_index],
                    source_gold_pred[name][keep, target_index],
                ),
            )
            loo_pred[holdout, target_index] = source_gold_pred[best_name][holdout, target_index]
            loo_choice_counts[target][best_name] += 1

    loo_auc = {
        target: auc(gold_y[:, index], loo_pred[:, index])
        for index, target in enumerate(TARGETS)
    }
    loo_macro = float(np.mean(list(loo_auc.values())))
    if loo_macro < 0.89:
        raise RuntimeError(f"V3 label gate failed: leave-one-out macro AUC={loo_macro:.6f}")

    soft = np.zeros((len(train_rows), len(TARGETS)), dtype=np.float64)
    weights = np.zeros_like(soft)
    chosen_per_cell: list[list[str]] = [["" for _ in TARGETS] for _ in train_rows]
    fallbacks: list[dict[str, str]] = []
    for row_index, uid in enumerate(train_ids):
        for target_index, target in enumerate(TARGETS):
            if gold_mask[row_index]:
                soft[row_index, target_index] = float(train_rows[row_index][target])
                weights[row_index, target_index] = 3.0
                chosen_per_cell[row_index][target_index] = "gold"
                continue

            name = selected_source[target]
            if uid not in source_rows[name]:
                fallback_order = sorted(
                    SOURCES,
                    key=lambda candidate: source_auc[candidate][target],
                    reverse=True,
                )
                name = next(candidate for candidate in fallback_order if uid in source_rows[candidate])
                fallbacks.append({UID: uid, "target": target, "source": name})
            row = source_rows[name][uid]
            probability = float(row[target])
            reliability = float(np.clip((source_auc[name][target] - 0.5) / 0.5, 0.0, 1.0))
            certainty = abs(probability - 0.5) * 2.0
            confidence = (0.20 + 0.80 * certainty) * (0.50 + 0.50 * reliability)
            if f"{target}__conf" in row and row[f"{target}__conf"] != "":
                reported = float(row[f"{target}__conf"])
                confidence = math.sqrt(max(confidence, 0.0) * np.clip(reported, 0.0, 1.0))
            soft[row_index, target_index] = np.clip(probability, 0.01, 0.99)
            weights[row_index, target_index] = np.clip(confidence, 0.15, 1.0)
            chosen_per_cell[row_index][target_index] = name

    folds, hashes = build_folds(train_rows, soft, gold_mask, folds=5)
    if any(len(set(folds[np.asarray(hashes) == key])) != 1 for key in set(hashes)):
        raise AssertionError("duplicate report group crossed folds")
    fold_counts = np.bincount(folds, minlength=5)
    if np.any(fold_counts < 0.75 * len(train_rows) / 5) or np.any(
        fold_counts > 1.25 * len(train_rows) / 5
    ):
        raise RuntimeError(f"fold-size balance gate failed: {fold_counts.tolist()}")
    gold_fold_counts = np.bincount(folds[gold_mask], minlength=5)
    if np.any(gold_fold_counts < 7):
        raise RuntimeError(f"gold fold coverage gate failed: {gold_fold_counts.tolist()}")

    # Build a separate label bank for every model fold. Source ranking for fold k
    # uses only gold rows outside k; its diagnostic prediction is then evaluated
    # only on the held-out gold rows in k. If the best source does not beat the
    # runner-up by SOURCE_GAP_GATE, use the top-two mean rather than overfit a
    # 46/47-row source comparison.
    gold_folds = folds[gold_indices]
    crossfit_pred = np.zeros_like(gold_y)
    crossfit_policy: dict[str, dict[str, object]] = {}
    fold_label_paths: list[Path] = []
    label_fields = [UID]
    for target in TARGETS:
        label_fields.extend([target, f"{target}__weight", f"{target}__source"])

    for heldout_fold in range(5):
        fit_gold = gold_folds != heldout_fold
        valid_gold = gold_folds == heldout_fold
        fold_policy: dict[str, object] = {}
        selected_names: dict[str, list[str]] = {}
        ranked_names: dict[str, list[str]] = {}
        fitted_auc: dict[str, dict[str, float]] = {}

        for target_index, target in enumerate(TARGETS):
            scores = {
                name: auc(
                    gold_y[fit_gold, target_index],
                    source_gold_pred[name][fit_gold, target_index],
                )
                for name in SOURCES
            }
            ranked = sorted(
                SOURCES,
                key=lambda name: (
                    -1.0 if not math.isfinite(scores[name]) else scores[name], name
                ),
                reverse=True,
            )
            best, runner_up = ranked[:2]
            gap = scores[best] - scores[runner_up]
            chosen = [best] if gap >= SOURCE_GAP_GATE else [best, runner_up]
            selected_names[target] = chosen
            ranked_names[target] = ranked
            fitted_auc[target] = scores
            fold_policy[target] = {
                "ranked_sources": ranked,
                "fitted_auc": scores,
                "best_minus_runner_up": gap,
                "gap_gate": SOURCE_GAP_GATE,
                "selected_sources": chosen,
            }
            crossfit_pred[valid_gold, target_index] = np.mean(
                [source_gold_pred[name][valid_gold, target_index] for name in chosen],
                axis=0,
            )

        fold_rows_out: list[dict[str, object]] = []
        for row_index, uid in enumerate(train_ids):
            row_out: dict[str, object] = {UID: uid}
            for target_index, target in enumerate(TARGETS):
                if gold_mask[row_index]:
                    probability = float(train_rows[row_index][target])
                    confidence = 3.0
                    source_name = "gold"
                else:
                    available = [
                        name for name in selected_names[target] if uid in source_rows[name]
                    ]
                    if not available:
                        available = [
                            name for name in ranked_names[target] if uid in source_rows[name]
                        ][:1]
                    if not available:
                        raise ValueError(f"no label source for {uid}/{target}/fold{heldout_fold}")
                    values = [float(source_rows[name][uid][target]) for name in available]
                    probability = float(np.mean(values))
                    reliability = float(
                        np.clip(
                            (np.mean([fitted_auc[target][name] for name in available]) - 0.5)
                            / 0.5,
                            0.0,
                            1.0,
                        )
                    )
                    certainty = abs(probability - 0.5) * 2.0
                    confidence = (0.20 + 0.80 * certainty) * (0.50 + 0.50 * reliability)
                    if len(available) > 1:
                        confidence *= 0.85
                    confidence = float(np.clip(confidence, 0.15, 1.0))
                    source_name = "+".join(available)
                row_out[target] = f"{np.clip(probability, 0.01, 0.99):.8f}"
                row_out[f"{target}__weight"] = f"{confidence:.8f}"
                row_out[f"{target}__source"] = source_name
            fold_rows_out.append(row_out)

        fold_label_path = args.output / f"labels_v3_fold{heldout_fold}.csv"
        write_rows(fold_label_path, label_fields, fold_rows_out)
        fold_label_paths.append(fold_label_path)
        crossfit_policy[str(heldout_fold)] = fold_policy

    crossfit_auc = {
        target: auc(gold_y[:, index], crossfit_pred[:, index])
        for index, target in enumerate(TARGETS)
    }
    crossfit_macro = float(np.mean(list(crossfit_auc.values())))
    if crossfit_macro < 0.89:
        raise RuntimeError(
            f"V3.1 fold-cross-fitted label gate failed: macro AUC={crossfit_macro:.6f}"
        )

    label_rows: list[dict[str, object]] = []
    for row_index, uid in enumerate(train_ids):
        out: dict[str, object] = {UID: uid}
        for target_index, target in enumerate(TARGETS):
            out[target] = f"{soft[row_index, target_index]:.8f}"
            out[f"{target}__weight"] = f"{weights[row_index, target_index]:.8f}"
            out[f"{target}__source"] = chosen_per_cell[row_index][target_index]
        label_rows.append(out)

    fold_rows = [
        {
            UID: uid,
            "fold": int(folds[index]),
            "report_group": hashes[index],
            "is_gold": int(gold_mask[index]),
        }
        for index, uid in enumerate(train_ids)
    ]
    label_path = args.output / "labels_v3.csv"
    fold_path = args.output / "folds_v3.csv"
    write_rows(label_path, label_fields, label_rows)
    write_rows(fold_path, [UID, "fold", "report_group", "is_gold"], fold_rows)

    chosen_auc = {
        target: source_auc[selected_source[target]][target] for target in TARGETS
    }
    fold_summary = {}
    for fold in range(5):
        mask = folds == fold
        gold_fold = mask & gold_mask
        fold_summary[str(fold)] = {
            "rows": int(mask.sum()),
            "gold_rows": int(gold_fold.sum()),
            "gold_positive_counts": {
                target: int(
                    sum(float(train_rows[i][target]) for i in np.flatnonzero(gold_fold))
                )
                for target in TARGETS
            },
        }

    audit = {
        "format": "rsna-knee-v3-label-bank-v1",
        "rows": len(train_rows),
        "gold_rows": int(gold_mask.sum()),
        "targets": TARGETS,
        "selected_source": selected_source,
        "source_gold_auc": source_auc,
        "selected_full_gold_auc": chosen_auc,
        "selected_full_gold_macro_auc": float(np.mean(list(chosen_auc.values()))),
        "leave_one_out_source_selection_auc": loo_auc,
        "leave_one_out_source_selection_macro_auc": loo_macro,
        "leave_one_out_choice_counts": loo_choice_counts,
        "fold_crossfit_source_gap_gate": SOURCE_GAP_GATE,
        "fold_crossfit_policy": crossfit_policy,
        "fold_crossfit_gold_auc": crossfit_auc,
        "fold_crossfit_gold_macro_auc": crossfit_macro,
        "fallback_cells": fallbacks,
        "fold_summary": fold_summary,
        "input_sha256": input_hashes,
        "output_sha256": {
            str(label_path): sha256(label_path),
            str(fold_path): sha256(fold_path),
            **{str(path): sha256(path) for path in fold_label_paths},
        },
    }
    audit_path = args.output / "audit_v3.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"labels: {label_path} ({len(label_rows)} rows)")
    print(f"folds:  {fold_path} ({len(fold_rows)} rows)")
    print(f"LOO source-selection macro AUC: {loo_macro:.6f}")
    print(f"full fixed-source macro AUC:    {audit['selected_full_gold_macro_auc']:.6f}")
    print(f"fold-crossfit gold macro AUC:   {crossfit_macro:.6f}")
    print("selected sources:")
    for target in TARGETS:
        print(f"  {target:18s} {selected_source[target]:10s} AUC={chosen_auc[target]:.4f}")
    print("fold sizes:", [fold_summary[str(fold)]["rows"] for fold in range(5)])
    print("gold per fold:", [fold_summary[str(fold)]["gold_rows"] for fold in range(5)])


if __name__ == "__main__":
    main()
