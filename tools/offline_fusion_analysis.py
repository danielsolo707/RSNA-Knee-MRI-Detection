"""Offline gold-58 analysis (no GPU): rank-fuse every arm we have gold preds for,
and gate the leave-one-out (LOO) distillation teacher.

Arms: public probe (v5_full_swa, v4_full, v4_full_swa), V38 run artifacts
(base/ep16/swa/epens), V31 gold-OOF (DINOv2 family).
Teacher gate: label-AUC on gold-58 of v4-only pseudo-labels blended with the
soft labels, vs the V38 teacher which included the student (v5) itself.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "analysis_offline_0830"
OUT.mkdir(parents=True, exist_ok=True)

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]


def macro_auc(truth, pred):
    return float(np.mean([roc_auc_score(truth[:, j], pred[:, j]) for j in range(truth.shape[1])]))


def per_target(truth, pred):
    return {LAB[j]: round(roc_auc_score(truth[:, j], pred[:, j]), 4) for j in range(12)}


def rank01(x):
    r = np.zeros_like(x, dtype=float)
    for j in range(x.shape[1]):
        r[:, j] = rankdata(x[:, j]) / len(x)
    return r


def fuse(arms, weights):
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    out = sum(wi * rank01(a) for wi, a in zip(w, arms))
    return out


# ---------------------------------------------------------------- load arms
z = np.load(ROOT / "artifacts/remote_arm-gold-probe/arm_gold_preds.npz")
ids_probe = list(z["ids"])
truth = z["truth"]
assert list(z["lab"]) == LAB

arms = {
    "v5_full_swa": z["pred_v5_full_swa"],
    "v4_full": z["pred_v4_full"],
    "v4_full_swa": z["pred_v4_full_swa"],
}

v38dir = ROOT / "artifacts/remote_rsna-knee-v38-pseudo-ft"
for name, fname in [("v38_base", "raptor_gold_v38pseudo_base.npz"),
                    ("v38_ep16", "raptor_gold_v38pseudo_ep16.npz"),
                    ("v38_swa", "raptor_gold_v38pseudo_swa.npz"),
                    ("v38_epens", "raptor_gold_v38pseudo_epens.npz"),
                    ("v38_best", "raptor_gold_v38pseudo.npz")]:
    zb = np.load(v38dir / fname)
    assert list(zb["ids"]) == ids_probe, f"{fname} id order mismatch"
    arms[name] = zb["pred"]

g = pd.read_csv(ROOT / "artifacts/v31_runs/gold_oof.csv").set_index("StudyInstanceUID")
assert list(g.index) == ids_probe, "gold_oof id order mismatch"
arms["v31_oof"] = g[LAB].to_numpy(dtype=float)

report = {"n_gold": len(ids_probe), "arms": {}, "fusions": [], "corrs": {}}

for name, a in arms.items():
    report["arms"][name] = {"macro": round(macro_auc(truth, a), 4),
                            "per_target": per_target(truth, a)}

# ------------------------------------------------------------ correlations
names = list(arms)
for a, b in combinations(names, 2):
    rs = [np.corrcoef(rankdata(arms[a][:, j]), rankdata(arms[b][:, j]))[0, 1]
          for j in range(12)]
    report["corrs"][f"{a}~{b}"] = round(float(np.mean(rs)), 4)

# ---------------------------------------------------------------- fusions
def add(label, arm_names, weights):
    pred = fuse([arms[n] for n in arm_names], weights)
    report["fusions"].append({"label": label, "arms": dict(zip(arm_names, weights)),
                              "macro": round(macro_auc(truth, pred), 4)})
    return pred


add("v5 solo", ["v5_full_swa"], [1])
add("v4 solo", ["v4_full"], [1])
add("v5+v4 w60 (V37 config)", ["v5_full_swa", "v4_full"], [0.4, 0.6])
add("v5+v4 equal", ["v5_full_swa", "v4_full"], [0.5, 0.5])

# did V38 fine-tuning create anything complementary?
for cand in ["v38_ep16", "v38_swa", "v38_epens", "v38_best"]:
    add(f"{cand} solo", [cand], [1])
    add(f"v5+{cand}", ["v5_full_swa", cand], [0.5, 0.5])
    add(f"v4+{cand}", ["v4_full", cand], [0.5, 0.5])
    add(f"v5+v4+{cand}", ["v5_full_swa", "v4_full", cand], [0.4, 0.6, 0.5])

# cross-family: V31 DINOv2 OOF with Raptor CoAtNet arms
for w31 in (0.1, 0.2, 0.3, 0.5):
    add(f"v5+v4+v31 (v31 w={w31})", ["v5_full_swa", "v4_full", "v31_oof"],
        [(1 - w31) * 0.4, (1 - w31) * 0.6, w31])
add("v5+v31 50/50", ["v5_full_swa", "v31_oof"], [0.5, 0.5])
add("v4+v31 50/50", ["v4_full", "v31_oof"], [0.5, 0.5])
add("all four equal", ["v5_full_swa", "v4_full", "v4_full_swa", "v31_oof"], [1, 1, 1, 1])

report["fusions"].sort(key=lambda r: -r["macro"])

# ------------------------------------------------- LOO teacher gate (labels)
gold_lbl = pd.read_csv(ROOT / "data/labels/stevenleehans/llm_labels_v4_blend.csv")
gold_lbl = gold_lbl.set_index("StudyInstanceUID").loc[ids_probe, LAB].to_numpy(dtype=float)

pg_v4 = np.load(ROOT / "artifacts/pseudo_labels/pred_gold_v4.npy")
pg_v5 = np.load(ROOT / "artifacts/pseudo_labels/pred_gold_v5.npy")
soft = gold_lbl  # soft labels on gold rows = the v4_blend bank

def label_auc(pseudo, soft_, w):
    mix = w * rank01(pseudo) + (1 - w) * rank01(soft_)
    return macro_auc(truth, mix)

loo = {"soft_only": macro_auc(truth, soft), "v4_only": macro_auc(truth, pg_v4),
       "v5_only": macro_auc(truth, pg_v5),
       "fused_teacher(v4+v5)": macro_auc(truth, 0.6 * rank01(pg_v4) + 0.4 * rank01(pg_v5)),
       "sweeps": []}
for w in (0.4, 0.5, 0.6, 0.7, 0.8):
    loo["sweeps"].append({
        "blend_w": w,
        "LOO_v4+soft": round(label_auc(pg_v4, soft, w), 4),
        "v5+soft (V38 teacher style)": round(label_auc(pg_v5, soft, w), 4),
        "fused+soft (V38 actual)": round(label_auc(
            0.6 * rank01(pg_v4) + 0.4 * rank01(pg_v5), soft, w), 4),
    })

report["loo_teacher_gate"] = loo

with open(OUT / "offline_fusion_report.json", "w") as f:
    json.dump(report, f, indent=1)

print("=== singles ===")
for n, r in report["arms"].items():
    print(f"  {n:16s} {r['macro']:.4f}")
print("=== top fusions ===")
for r in report["fusions"][:14]:
    print(f"  {r['macro']:.4f}  {r['label']}")
print("=== LOO teacher gate ===")
print(f"  soft-only labels      {loo['soft_only']:.4f}")
print(f"  v4-only preds         {loo['v4_only']:.4f}")
print(f"  fused teacher preds   {loo['fused_teacher(v4+v5)']:.4f}")
for s in loo["sweeps"]:
    print(f"  w={s['blend_w']:.1f}: LOO(v4+soft) {s['LOO_v4+soft']:.4f} | "
          f"v5+soft {s['v5+soft (V38 teacher style)']:.4f} | "
          f"fused+soft {s['fused+soft (V38 actual)']:.4f}")
