"""Score the V31 5-fold hard family on the 58 organizer-gold studies.

Uses the same GroupKFold as training so each gold row is predicted by the
fold that did not train on it (honest OOF). Also reports the 5-fold mean
over all gold rows (not OOF) as a secondary number.

Writes artifacts/v31_runs/gold_oof.json + gold_oof.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from v31probe import (  # noqa: E402
    TARGETS,
    N_FOLDS,
    FullFT,
    CacheDS,
    load_cache,
    report_group,
    set_seed,
)

CACHE = REPO / "artifacts" / "v31_cache" / "v31_cache"
RUNS = REPO / "artifacts" / "v31_runs"
OUT = RUNS / "gold_oof.json"
CSV = RUNS / "gold_oof.csv"
TRAIN_CSV = REPO / "data" / "raw" / "train.csv"
BATCH = 8
SEED = 20260821
DINO = "facebook/dinov2-small"


def fold_best(fold: int) -> Path:
    p = RUNS / f"fold{fold}_hard" / "best.pt"
    if p.is_file():
        return p
    alt = RUNS / f"fold{fold}_hard" / "fold4_out" / "best.pt"
    if alt.is_file():
        return alt
    raise FileNotFoundError(f"missing best.pt for fold {fold}")


def macro_and_per_target(y: np.ndarray, p: np.ndarray):
    per = {}
    sc = []
    for j, t in enumerate(TARGETS):
        yy = (y[:, j] > 0.5).astype(int)
        if len(np.unique(yy)) < 2:
            per[t] = None
            continue
        try:
            a = float(roc_auc_score(yy, p[:, j]))
        except Exception:
            a = None
        per[t] = a
        if a is not None:
            sc.append(a)
    return (float(np.mean(sc)) if sc else float("nan")), per


def predict_idx(net, ds, idx, dev):
    sub = CacheDS(ds.x, ds.mask, ds.y, ds.uids, idx, augment=False)
    net.eval()
    ps, uids = [], []
    loader = torch.utils.data.DataLoader(
        sub, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True
    )
    autocast = torch.amp.autocast("cuda")
    with torch.no_grad():
        for xb, mb, _tb, uid in loader:
            xb = xb.to(dev, non_blocking=True)
            mb = mb.to(dev, non_blocking=True)
            with autocast:
                pr = torch.sigmoid(net(xb, mb)[0])
                pr = 0.5 * (pr + torch.sigmoid(net(torch.flip(xb, dims=[-1]), mb)[0]))
            ps.append(pr.float().cpu().numpy())
            uids.extend(list(uid))
    return np.concatenate(ps, axis=0), np.array(uids)


def main():
    t0 = time.time()
    set_seed(SEED)
    assert CACHE.is_dir(), f"cache missing: {CACHE}"
    man, x, mask, y_cache, uids = load_cache(CACHE)
    uids = np.array(uids)
    print(f"cache n={len(uids)} recipe={man.get('recipe')}", flush=True)

    train = pd.read_csv(TRAIN_CSV, dtype={"StudyInstanceUID": str})
    gold_df = train.dropna(subset=TARGETS, how="all").set_index("StudyInstanceUID")
    for t in TARGETS:
        gold_df[t] = pd.to_numeric(gold_df[t], errors="coerce")
    gold_ids = set(gold_df.index)
    print(f"gold rows in train.csv: {len(gold_df)}", flush=True)

    uid_to_i = {u: i for i, u in enumerate(uids)}
    gold_pos = [uid_to_i[u] for u in gold_df.index if u in uid_to_i]
    missing = [u for u in gold_df.index if u not in uid_to_i]
    print(f"gold in cache: {len(gold_pos)} missing: {len(missing)}", flush=True)

    groups = np.array([report_group(u) for u in uids])
    folds = list(GroupKFold(N_FOLDS).split(np.arange(len(uids)), groups=groups))

    dummy_y = np.zeros((len(uids), len(TARGETS)), np.float32)
    ds = CacheDS(x, mask, dummy_y, uids, np.arange(len(uids)), augment=False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}", flush=True)

    oof_p = np.full((len(gold_pos), len(TARGETS)), np.nan, np.float32)
    oof_u = np.array([uids[i] for i in gold_pos])
    pos_set = {int(i): k for k, i in enumerate(gold_pos)}
    fold_gold_n = []

    mean_acc = np.zeros((len(gold_pos), len(TARGETS)), np.float64)
    mean_n = np.zeros(len(gold_pos), np.int32)

    for fold in range(N_FOLDS):
        ck_path = fold_best(fold)
        print(f"\n=== fold {fold} {ck_path} ===", flush=True)
        net = FullFT(DINO, dropout=0.3).to(dev)
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        net.eval()
        print(f"loaded auc={ck.get('auc')} epoch={ck.get('epoch')} ema={ck.get('used_ema')}", flush=True)

        _tr, va_idx = folds[fold]
        va_gold = [int(i) for i in va_idx if int(i) in pos_set]
        fold_gold_n.append(len(va_gold))
        print(f"val gold in fold: {len(va_gold)}", flush=True)

        all_gold_idx = np.array(gold_pos, dtype=np.int64)
        pred_all, pred_uids = predict_idx(net, ds, all_gold_idx, dev)
        # map back by uid
        u2p = {u: pred_all[j] for j, u in enumerate(pred_uids)}
        for k, u in enumerate(oof_u):
            mean_acc[k] += u2p[u]
            mean_n[k] += 1
            i_global = gold_pos[k]
            if i_global in set(va_gold):
                oof_p[k] = u2p[u]

        del net, ck
        torch.cuda.empty_cache()
        print(f"fold {fold} done {(time.time()-t0)/60:.1f} min", flush=True)

    y = gold_df.loc[list(oof_u), TARGETS].to_numpy(np.float32)
    oof_ok = np.isfinite(oof_p).all(axis=1)
    oof_macro, oof_per = macro_and_per_target(y[oof_ok], oof_p[oof_ok])
    mean_p = (mean_acc / np.maximum(mean_n[:, None], 1)).astype(np.float32)
    mean_macro, mean_per = macro_and_per_target(y, mean_p)

    receipt = {
        "n_gold_train_csv": int(len(gold_df)),
        "n_gold_in_cache": int(len(gold_pos)),
        "n_oof": int(oof_ok.sum()),
        "fold_gold_n": fold_gold_n,
        "oof_macro_auc": oof_macro,
        "fivefold_mean_macro_auc": mean_macro,
        "oof_per_target": oof_per,
        "fivefold_mean_per_target": mean_per,
        "missing_from_cache": missing,
        "seconds": round(time.time() - t0, 1),
        "recipe": man.get("recipe"),
        "note": "OOF = each gold study predicted by the fold that held it out. fivefold_mean = average of all 5 models on every gold row.",
    }
    OUT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    out_df = pd.DataFrame(oof_p, columns=TARGETS)
    out_df.insert(0, "StudyInstanceUID", oof_u)
    out_df.to_csv(CSV, index=False)
    print(json.dumps(receipt, indent=2), flush=True)
    print(f"wrote {OUT} and {CSV}", flush=True)


if __name__ == "__main__":
    main()
