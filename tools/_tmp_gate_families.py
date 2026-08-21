#!/usr/bin/env python3
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


d = np.load("research/pilkwang_oof/oof.npz", allow_pickle=True)
dino_ids = [str(x) for x in d["ids"]]
dino = d["pred"].astype(float)
gold_mask = d["gold_mask"]
yder = d["y_derived"].astype(float)
print("dino", dino.shape, "gold", int(gold_mask.sum()))

# steven
st_path = Path("references/candidates/steven_exp025/output/oof_predictions.csv")
st_rows = load_csv(st_path)
print("steven n", len(st_rows), "keys", list(st_rows[0].keys())[:20])

# map steven
def mat_from_rows(rows, id_key="StudyInstanceUID"):
    idx = {str(r[id_key]): i for i, r in enumerate(rows)}
    # find target cols
    cols = []
    for t in TARGETS:
        for c in rows[0].keys():
            if c == t or c.endswith(t) or t in c:
                cols.append(c)
                break
        else:
            cols.append(None)
    print("mapped cols", cols)
    return idx, cols


# try common patterns
for label, path in [
    ("steven", st_path),
    ("dehai", Path("references/candidates/dehai_v14/output/submission.csv")),
]:
    if not path.exists():
        continue
    rows = load_csv(path)
    print(label, "n", len(rows), "sample keys", list(rows[0].keys())[:15])

# gold from train
train = {r["StudyInstanceUID"]: r for r in load_csv("data/raw/train.csv")}
gold_uids = [u for u, r in train.items() if all(r.get(t) not in (None, "") for t in TARGETS)]
print("gold uids from train", len(gold_uids))

# If steven has preds aligned
st_by = {str(r.get("StudyInstanceUID", r.get("study_id", ""))): r for r in st_rows}
common = [u for u in dino_ids if u in st_by]
print("common dino-steven", len(common))
if common and all(t in st_rows[0] for t in TARGETS):
    st_mat = np.array([[float(st_by[u][t]) for t in TARGETS] for u in dino_ids if u in st_by])
    dino_m = np.array([dino[dino_ids.index(u)] for u in dino_ids if u in st_by])
    soft = np.array([yder[dino_ids.index(u)] for u in dino_ids if u in st_by])
    print("soft dino", np.nanmean([auc(soft[:, j], dino_m[:, j]) for j in range(12)]))
    print("soft steven", np.nanmean([auc(soft[:, j], st_mat[:, j]) for j in range(12)]))
    # gold
    g_idx = [i for i, u in enumerate([u for u in dino_ids if u in st_by]) if gold_mask[dino_ids.index(u)]]
    if g_idx:
        # rebuild
        pass
