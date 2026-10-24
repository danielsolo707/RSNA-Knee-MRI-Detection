"""Read-only coverage and Gold58 leakage audit for the downloaded HPO label assets.

Print aggregate counts only; never print study IDs or report text. Run from the
repository root with: python tools/audit_hpo_labels.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HPO = ROOT / "artifacts/hpo_label_audit"
UID = "StudyInstanceUID"
BANKS = ("labels_v0raw", "labels_v1_blend", "labels_v1_gemlow", "labels_v1_perlabel")


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path, dtype={UID: str})
    else:
        df = pd.read_parquet(path)
    df[UID] = df[UID].astype(str)
    return df


def values(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def main() -> None:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    train = load(ROOT / "data/raw/train.csv")
    targets = [c for c in train.columns if c not in (UID, "Report")]
    train_ids = set(train[UID])
    gold_ids = set(train.loc[train[targets].notna().all(axis=1), UID])
    nongold_ids = train_ids - gold_ids
    check(len(train) == len(train_ids), "duplicate organizer study IDs")
    check(len(gold_ids) == 58, "unexpected number of fully labeled organizer studies")
    print(f"Organizer: {len(train_ids)} studies, {len(gold_ids)} Gold58, {len(nongold_ids)} non-gold")

    parts = [np.load(ROOT / f"artifacts/raptor_corpus/{part}_ids.npy", allow_pickle=False).astype(str).tolist()
             for part in ("all", "extra")]
    corpus = parts[0] + parts[1]
    corpus_ids = set(corpus)
    print(f"Corpus: {len(corpus)} rows, {len(corpus) - len(corpus_ids)} duplicates, "
          f"{len(train_ids - corpus_ids)} missing organizer, {len(corpus_ids - train_ids)} extra")
    check(len(corpus) == len(corpus_ids) and corpus_ids == train_ids, "corpus coverage/uniqueness")

    fmap = json.loads((HPO / "folds_v0.json").read_text(encoding="utf-8"))["folds"]
    counts = dict(sorted(Counter(fmap.values()).items()))
    gold_folds = dict(sorted(Counter(fmap.get(uid) for uid in gold_ids).items(), key=lambda x: str(x[0])))
    nongold_folds = Counter(fmap.get(uid) for uid in nongold_ids)
    print(f"Folds: {len(fmap)} entries, {len(train_ids - fmap.keys())} missing organizer, "
          f"{len(fmap.keys() - train_ids)} extra; counts={counts}; Gold58={gold_folds}")
    check(set(fmap) == train_ids, "fold map coverage")
    check({fmap.get(uid) for uid in gold_ids} == {-1}, "gold fold assignment must be -1")
    check(set(nongold_folds) == set(range(5)), "non-gold fold values must be 0..4")
    check(all(count > 0 for count in nongold_folds.values()), "empty non-gold fold")
    reports = train["Report"].fillna("").astype(str)
    gold_reports = set(reports[train[UID].isin(gold_ids)])
    nongold = train.loc[~train[UID].isin(gold_ids), [UID]].copy()
    nongold["report"] = reports.loc[nongold.index]
    nongold["fold"] = nongold[UID].map(fmap)
    report_groups = nongold.groupby("report").fold.agg(["size", "nunique"])
    repeats = int((report_groups["size"] > 1).sum())
    cross_fold = int((report_groups["nunique"] > 1).sum())
    gold_report_overlap = int(nongold["report"].isin(gold_reports).sum())
    print(f"Exact report duplicates: {repeats} non-gold groups, {cross_fold} crossing "
          f"non-gold folds, {gold_report_overlap} non-gold reports also in Gold58")
    check(cross_fold == 0, "identical reports split across non-gold folds")

    raptor = load(ROOT / "artifacts/raptor_labels/labels_llm_soft.parquet")
    sources = [("raptor", raptor)] + [(name, load(HPO / f"{name}.parquet")) for name in BANKS]
    for name, df in sources:
        ids = set(df[UID])
        a = values(df, targets)
        bad = int((~np.isfinite(a) | (a < 0) | (a > 1)).sum())
        duplicates = len(df) - len(ids)
        print(f"{name}: {len(df)} rows, {duplicates} duplicates, "
              f"{len(ids & gold_ids)} Gold58 overlap, {len(nongold_ids - ids)} missing non-gold, "
              f"{len(ids - nongold_ids)} extra, {bad} invalid target values")
        check(not duplicates and ids == nongold_ids and not bad, f"{name} coverage/target values")

    ycols = ["y_" + t.replace(" ", "_").replace("Baker's", "Bakers") for t in targets]
    mcols = [c.replace("y_", "m_", 1) for c in ycols]
    goldcols = [c.replace("y_", "gold_", 1) for c in ycols]
    organizer_gold = train.set_index(UID).loc[sorted(gold_ids), targets].to_numpy(dtype=float)
    banks = {name: df.set_index(UID) for name, df in sources[1:]}
    for name in ("train_labels_v0", "train_labels_v1"):
        df = load(HPO / f"{name}.parquet")
        ids = set(df[UID])
        duplicates = len(df) - len(ids)
        y = values(df, ycols)
        mask = values(df, mcols)
        bad_y = int((~np.isfinite(y) | (y < 0) | (y > 1)).sum())
        bad_mask = int((~np.isin(mask, [0, 1])).sum())
        gold_y = values(df.set_index(UID).loc[sorted(gold_ids)], ycols)
        exact_gold_y = int(np.isclose(gold_y, organizer_gold, rtol=0, atol=1e-8).sum())
        print(f"{name}: {len(df)} rows, {duplicates} duplicates, "
              f"{len(train_ids - ids)} missing organizer, {len(ids - train_ids)} extra, "
              f"{bad_y} invalid y, {bad_mask} invalid masks, "
              f"{int((mask == 0).sum())} masked cells; Gold58 y exactly matches "
              f"{exact_gold_y}/{gold_y.size} organizer targets")
        check(not duplicates and ids == train_ids and not bad_y and not bad_mask,
              f"{name} coverage/y/mask values")
        if name == "train_labels_v0":
            flagged = set(df.loc[df["is_gold"].astype(bool), UID])
            gold_table = values(df.set_index(UID).loc[sorted(gold_ids)], goldcols)
            nongold_table = values(df.loc[~df[UID].isin(gold_ids)], goldcols)
            matches = int(np.isclose(gold_table, organizer_gold, rtol=0, atol=1e-8).sum())
            print(f"  is_gold flags: {len(flagged)}, correctly flagged: {len(flagged & gold_ids)}; "
                  f"gold_* matches organizer: {matches}/{gold_table.size}; "
                  f"non-gold gold_* finite: {int(np.isfinite(nongold_table).sum())}")
            check(flagged == gold_ids and matches == gold_table.size and not np.isfinite(nongold_table).any(),
                  "v0 organizer gold columns/flags")
        else:
            # The v1 y targets are not organizer truth as a complete vector, but
            # this cannot establish their provenance; its Gold58 rows stay excluded.
            nongold = df.set_index(UID).loc[sorted(nongold_ids)]
            reference = banks["labels_v1_gemlow"].loc[sorted(nongold_ids)]
            matches = int(np.isclose(values(nongold, ycols), values(reference, targets),
                                     rtol=0, atol=1e-8).sum())
            print(f"  non-gold y matches labels_v1_gemlow: {matches}/{len(nongold_ids) * len(targets)}")
            check(matches == len(nongold_ids) * len(targets), "v1/gemlow non-gold label alignment")

    if failures:
        print("FAIL: " + "; ".join(failures))
        raise SystemExit(1)
    print("PASS: study coverage, numeric targets, masks, folds and explicit Gold58 flags verified.")
    print("CAVEAT: patient grouping/label provenance are unverified; Gold58 is not a blind test.")
    print("Never train on Gold58 rows, including those in train_labels_v1.")


if __name__ == "__main__":
    main()
