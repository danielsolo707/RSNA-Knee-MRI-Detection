"""Offline fusion simulation with the V41 probe outputs (V33 chain on gold-58).

Answers THE question: does adding an owned arm to the true V33 fusion lift gold-58?

Inputs (all under artifacts/):
  remote_rsna-knee-v41-gold-probe/probe_gold_stages.npz
      pred_stage1_dinov2 / pred_stage2_mil / pred_stage3_rad / pred_coatnet /
      pred_stage4_final, truth, ids, lab          (from the V41 probe kernel)
  remote_arm-gold-probe/arm_gold_preds.npz        (v5 / v4 / v4swa CoAtNet arms)
  v31_runs/gold_oof.csv                           (our own DINOv2 arm, OOF)

Method: rank-fusion (same family V33 uses), macro-AUC on gold-58, and for every
candidate that looks promising a 400-split split-half validation (V37 lesson:
in-sample gold-58 gains are fake until they survive out-of-sample splits).

Run: python tools/simulate_v33_fusion.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "artifacts" / "remote_rsna-knee-v41-gold-probe" / "probe_gold_stages.npz"
OUT = ROOT / "artifacts" / "fusion_sim_0831"

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]


def macro(truth, pred):
    return float(np.mean([roc_auc_score(truth[:, j], pred[:, j]) for j in range(truth.shape[1])]))


def rank01(x):
    return np.stack([rankdata(x[:, j]) / len(x) for j in range(x.shape[1])], axis=1)


def fuse(parts, weights):
    w = np.asarray(weights, float)
    w = w / w.sum()
    return sum(wi * rank01(p) for wi, p in zip(w, parts))


def split_half_gate(truth, parts, weights, n_splits=400, seed=0):
    """Fit nothing: just score the fixed blend on random 29/29 halves. Returns
    mean OOS macro and P(gain over the base arm) is handled by the caller."""
    rng = np.random.default_rng(seed)
    n = len(truth)
    scores = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        half = idx[: n // 2]
        pred = fuse([p[half] for p in parts], weights)
        scores.append(macro(truth[half], pred))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> int:
    if not PROBE.is_file():
        raise SystemExit(f"missing {PROBE} - run/pull the V41 probe first")
    z = np.load(PROBE)
    ids = list(z["ids"])
    truth = z["truth"]
    assert list(z["lab"]) == LAB

    stages = {}
    for k in z.files:
        if k.startswith("pred_"):
            stages[k[5:]] = z[k]
    print("probe stages:", {k: v.shape for k, v in stages.items()})

    # ---- align our arms to the probe's gold id order ----
    arms = {}
    za = np.load(ROOT / "artifacts/remote_arm-gold-probe/arm_gold_preds.npz")
    assert list(za["ids"]) == ids, "arm-probe id order != probe id order"
    for name in ("v5_full_swa", "v4_full", "v4_full_swa"):
        arms[name] = za[f"pred_{name}"]

    g = pd.read_csv(ROOT / "artifacts/v31_runs/gold_oof.csv").set_index("StudyInstanceUID")
    assert list(g.index) == ids, "gold_oof id order != probe id order"
    arms["v31_oof"] = g[LAB].to_numpy(float)

    report = {"n_gold": len(ids), "singles": {}, "candidates": []}

    def score(name, pred):
        m = macro(truth, pred)
        report["singles"][name] = round(m, 4)
        return m

    for name, p in {**stages, **arms}.items():
        print(f"  single {name:18s} {score(name, p):.4f}")

    base_name = "stage4_final"
    if base_name not in stages:
        base_name = sorted(stages)[-1]
    base = stages[base_name]
    base_m = report["singles"][base_name]
    print(f"\nbase = {base_name} ({base_m:.4f})\n")

    def candidate(label, parts, weights, gate=True):
        pred = fuse(parts, weights)
        m = macro(truth, pred)
        entry = {"label": label, "macro": round(m, 4), "delta": round(m - base_m, 4)}
        if gate and m > base_m:
            mu, sd = split_half_gate(truth, parts, weights)
            entry["split_half_mean"] = round(mu, 4)
            entry["split_half_std"] = round(sd, 4)
            entry["split_half_delta"] = round(mu - base_m, 4)
        report["candidates"].append(entry)
        flag = ""
        if entry.get("split_half_delta") is not None:
            flag = f" [split-half {entry['split_half_delta']:+.4f}]"
        print(f"  {m:.4f} ({m-base_m:+.4f}) {label}{flag}")

    print("=== add owned arms on top of the true V33 fusion ===")
    for arm in ("v31_oof", "v4_full", "v5_full_swa"):
        for w in (0.05, 0.10, 0.15, 0.20, 0.30):
            candidate(f"{base_name} + {arm} w={w:.2f}",
                      [base, arms[arm]], [1 - w, w])

    print("\n=== replace experiments (diagnostic) ===")
    if "coatnet" in stages:
        cn = stages["coatnet"]
        for w in (0.5, 0.6):
            candidate(f"coatnet branch: v5+v4 w={w:.1f} (vs probe coatnet)",
                      [arms["v5_full_swa"], arms["v4_full"]], [1 - w, w], gate=False)
        print(f"  probe coatnet branch alone: {report['singles'].get('coatnet', float('nan')):.4f}")
    for s in ("stage1_dinov2", "stage2_mil", "stage3_rad"):
        if s in stages and "v31_oof" in arms:
            candidate(f"{s} + v31_oof w=0.15", [stages[s], arms["v31_oof"]], [0.85, 0.15])

    report["candidates"].sort(key=lambda r: -r["macro"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fusion_sim_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT / 'fusion_sim_report.json'}")

    best = report["candidates"][0] if report["candidates"] else None
    if best and best["macro"] > base_m:
        sh = best.get("split_half_delta")
        if sh is None:
            print(f"\nVERDICT: best candidate {best['label']} (+{best['delta']:.4f}) NOT gated yet")
        elif sh > 0:
            print(f"\nVERDICT: PROMISING - {best['label']}: in-sample {best['delta']:+.4f}, "
                  f"split-half {sh:+.4f} (survives OOS)")
        else:
            print(f"\nVERDICT: REJECT - {best['label']}: in-sample {best['delta']:+.4f} but "
                  f"split-half {sh:+.4f} (V37 pattern: does not transfer)")
    else:
        print("\nVERDICT: no candidate beats the V33 fusion on gold-58")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
