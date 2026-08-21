#!/usr/bin/env python3
"""Offline dual-gate: DINO + target-wise e9/e11 vs V9 parent (DINO+e9@0.20)."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def auc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    hard = (y >= 0.5).astype(float)
    pos = hard == 1
    neg = hard == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def rank_mat(a):
    out = np.zeros_like(a, float)
    for j in range(a.shape[1]):
        order = np.argsort(a[:, j], kind="mergesort")
        r = np.empty(len(a))
        r[order] = np.arange(1, len(a) + 1)
        out[:, j] = r / len(a)
    return out


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_oof(path):
    rows = load_csv(path)
    ids = [r["StudyInstanceUID"] for r in rows]
    mat = np.array([[float(r[t]) for t in TARGETS] for r in rows], float)
    fold = np.array([int(r["fold"]) for r in rows], int) if "fold" in rows[0] else None
    return ids, mat, fold


dino = np.load("research/pilkwang_oof/oof.npz", allow_pickle=True)
dino_ids = [str(x) for x in dino["ids"]]
dino_pred = dino["pred"].astype(float)
dino_idx = {u: i for i, u in enumerate(dino_ids)}

e9_ids, e9, e9_fold = load_oof("references/candidates/antoine_e11_files/../antoine_e11_files/v52_e11_oof.csv")
# e9 is e11 path wrong - load e9 from kaggle path if we have v52_oof - use e11 for now after fix
# Actually download e9 oof from e9 dataset - try v52_oof from e9
e9_path = Path("references/candidates/antoine_e11_files/v52_e11_oof.csv")
# We need e9 separately - from V11 audit we know e9 gold. Load from pilkwang? not there.
# Check if e9 oof exists
for p in [
    Path("references/candidates/antoine_e11_files/v52_e11_oof.csv"),
    Path("artifacts/v11_rad_ok/rsna_rad_v11/v11_own_oof.csv"),
]:
    print("exists", p, p.exists())

# Download e9 oof if needed via listing - use e11 only vs dino first
e11_ids, e11, e11_fold = load_oof("references/candidates/antoine_e11_files/v52_e11_oof.csv")
print("e11 n", len(e11_ids), "fold", e11_fold is not None)

# soft labels
lab = {r["StudyInstanceUID"]: r for r in load_csv("artifacts/v3_labels/labels_v3.csv")}
train = {r["StudyInstanceUID"]: r for r in load_csv("data/raw/train.csv")}

# align e11 with dino
common = [u for u in e11_ids if u in dino_idx and u in lab]
print("common", len(common))
e11_by = {u: i for i, u in enumerate(e11_ids)}
dino_m = np.array([dino_pred[dino_idx[u]] for u in common])
e11_m = np.array([e11[e11_by[u]] for u in common])
soft = np.array([[float(lab[u][t]) for t in TARGETS] for u in common])
folds = np.array([e11_fold[e11_by[u]] for u in common]) if e11_fold is not None else None

print("soft dino", np.nanmean([auc(soft[:, j], dino_m[:, j]) for j in range(12)]))
print("soft e11", np.nanmean([auc(soft[:, j], e11_m[:, j]) for j in range(12)]))

# gold
g_uids = [u for u in common if all(train[u].get(t) not in (None, "") for t in TARGETS)]
print("gold aligned", len(g_uids))
gy = np.array([[float(train[u][t]) for t in TARGETS] for u in g_uids])
gd = rank_mat(np.array([dino_pred[dino_idx[u]] for u in g_uids]))
ge = rank_mat(np.array([e11[e11_by[u]] for u in g_uids]))
gf = np.array([e11_fold[e11_by[u]] for u in g_uids])

# V9 parent needs e9 - without e9 use pure dino as base and e11 as rad
# If we only have e11, V9 parent = dino+e9 was 0.8516; approximate parent as dino ranks only OR use e11@0.2 as alt
parent = 0.8 * gd + 0.2 * ge  # not V9 but DINO+e11@0.2
parent = rank_mat(parent)  # re-rank? actually rank blend then rank again is ok
# proper: ranks already percentile
parent = 0.8 * gd + 0.2 * ge
print("parent macro (dino+e11@0.2)", np.nanmean([auc(gy[:, j], parent[:, j]) for j in range(12)]))
print("dino gold", np.nanmean([auc(gy[:, j], gd[:, j]) for j in range(12)]))
print("e11 gold", np.nanmean([auc(gy[:, j], ge[:, j]) for j in range(12)]))

# target-wise soft pick
print("\nper-target soft")
for j, t in enumerate(TARGETS):
    print(f"  {t:18s} dino {auc(soft[:,j], dino_m[:,j]):.4f} e11 {auc(soft[:,j], e11_m[:,j]):.4f}")

# Try target-wise alpha on e11 vs pure dino nested
alpha_grid = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
nested = np.zeros_like(gd)
choices = []
for outer in sorted(set(gf.tolist())):
    tr = gf != outer
    va = ~tr
    scored = []
    for a in alpha_grid:
        blend = (1 - a) * gd[tr] + a * ge[tr]
        scored.append(np.nanmean([auc(gy[tr, j], blend[:, j]) for j in range(12)]) - 0.005 * a)
    bi = int(np.argmax(scored))
    a = alpha_grid[bi]
    choices.append(a)
    nested[va] = (1 - a) * gd[va] + a * ge[va]
print("choices", choices, "median", np.median(choices))
print("nested", np.nanmean([auc(gy[:, j], nested[:, j]) for j in range(12)]))
print("dino only", np.nanmean([auc(gy[:, j], gd[:, j]) for j in range(12)]))
