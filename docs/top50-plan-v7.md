# Top-50 plan V7 (post B3 failure)

**Date:** 2026-08-11  
**Best score:** 0.899 (V1 DINO) — **keep selected**  
**Target:** public ≥ **0.900** (top 50)  
**Hard constraint:** 5 submits/day; code competition; no public-LB soup  

---

## What we already proved (do not re-run)

| Experiment | Result | Lesson |
|------------|--------|--------|
| V1 public DINOv2 ensemble | **0.899** | Anchor. Preserve. |
| V2 DINO+Tony OOF blend | 0.899 null | Weak import ⇒ null CSV |
| V3 Tony exact | **0.828** | Public claims ≠ reproducible |
| V5 gold-gated DINO+B3 | **0.893** | Gold LOO n=58 overfits; soft OOF was right |
| V6 B3 1+7 epochs 5-fold | soft **0.779** vs DINO **0.850** | More epochs ≠ enough; dual gate empty |

**Banned:** more EfficientNet-B3 ↔ DINO weight tweaks of the current V5/V6 family.

---

## Why +0.001 is hard but possible

- Macro AUC across 12 targets: one weak target can erase gains.
- V1 is already a strong multi-fold DINO rank ensemble.
- The only safe path is **new ranking diversity** that is positive under:
  1. **Soft OOF** (4,407 studies), and  
  2. **Expert LOO** (58 gold), and  
  3. **No catastrophic target** (no >0.02 collapse vs V1).

---

## Strategy stack (ordered by expected value / time)

### Track A — Public family import with dual OOF gate (this week, primary)

**Goal:** Attach 1–2 independent public inference families that are **not** our weak B3 and **not** Tony.

Candidates (local / to pull):

| Family | Why |
|--------|-----|
| Fracture TTA / pool probe (`renta0426/...`) | Small V1-adjacent tweak; may lift Fracture/Contusion |
| DINOv2 meniscus resolution (`wguesdon/...`) | Same backbone class, different geometry → rank diversity |
| Enhanced ensemble (`sakhawathossen/...`) | Multi-model public soup — only if OOF exists or can be rebuilt |
| Steven exp025 (already local OOF) | Soft loses to DINO everywhere; gold helps Lateral Meniscus only — **low priority** |

**Promotion rule (dual gate):** for each target, include a family only if:

- soft OOF AUC ≥ V1_soft − 0.015, **and**
- expert LOO rank-blend ≥ V1_expert − 0.005, **and**
- selected set improves soft macro by ≥ 0.001 **or** expert macro by ≥ 0.005

**Submit rule:** at most **one** code-comp submit per gate pass. Message: `V7 dual-gated <families>`.

**Fallback if all public families fail gate:** do not submit; go to Track B.

### Track B — Label upgrade + new CNN family (parallel / next GPU block)

**Goal:** A second architecture trained on **better labels** so soft OOF can approach ~0.84+ before any blend.

1. Land V3.2 (or better) labels on Kaggle (upload currently flaky — retry or embed in kernel).
2. Synovitis silence policy + multi-source fusion (steven4-strong targets).
3. Train **one** new family (prefer ConvNeXt-tiny / EffNet-B4 MIL or multi-series attention), **not** more V6 B3.
4. Same dual gate vs DINO before submit.

Budget: ~10–15 GPU h for 5-fold if T4 recipe holds.

### Track C — V1 surgical micro-edits (only with OOF)

Examples: fracture-aware TTA, pool probe, slice count at infer.  
**Never** ship without soft OOF on at least fold-level probe or documented public fingerprint match.

---

## Execution order (start now)

```
1. Write this plan (done)
2. Pull fracture-TTA + meniscus-DINO (+ enhanced ensemble) kernels
3. Extract: can we get OOF or only test preds?
   - If only test: build infer notebooks, score diversity vs V1 visible test (weak signal)
   - Prefer families with train OOF or retrainable scripts
4. Dual-gate any candidate with DINO OOF anchor
5. Build V7 notebook: V1 path + gated family(s) → submission.csv
6. Push → dry-run → one competition submit if gate passes
7. If fail: freeze Track A, start Track B label upload + new trainer scaffold
```

---

## Success metrics

| Milestone | Metric |
|-----------|--------|
| Gate pass | Soft macro ≥ DINO soft; or dual-gated blend ≥ DINO soft + 0.001 |
| Top 50 | Public ≥ 0.900 without dropping below V1 on select |
| Abort | Any gate fails → keep V1 selected; no submit |

---

## Explicit non-goals

- Re-blending V5/V6 B3 checkpoints with DINO  
- Tony-only or unreproduced “0.90x” notebooks  
- Gold-only weight fitting (V5 failure mode)  
- Horizontal flips on laterality targets  

---

## Live execution notes

- V7 meniscus kernel v1 **ERROR**: private author labels `wguesdon/rsna-knee-llm-report-labels-opus` not accessible.
- v2 **patched** → **COMPLETE** but **gate FAIL**:
  - Soft OOF macro vs extracted labels **0.776** (ridge/attn blend)
  - Gold OOF macro **0.771** (n=58, noisy)
  - Visible-test ranks **far** from V1 (mean |Δ| ≈ 0.26) — not a safe ensemble partner
- **Do not submit** meniscus-DINO as standalone or equal-blend with V1.
- Artifacts: `artifacts/v7_meniscus_ok/` (submission.csv sample + emb cache only; no full OOF CSV export).

## Next (Track A remainder / Track B)

1. Probe other public families with **known high public scores** or strong local OOF (DINOsaur V2, etc.) under dual gate.
2. If none pass: **Track B** — ship better soft labels + train a true second architecture with soft OOF ≥ 0.83 before blend.

### V8 Salem multi (2026-08-12)

- Kernel COMPLETE: https://www.kaggle.com/code/danielsolo1770/rsna-knee-v8-salem-multi
- **Effectively V1 clone:** final `submission.csv` identical to V1 on visible test.
- Extra B3/RadImageNet branches all **safe-failed** (wrong `/kaggle/input` paths / missing private fold datasets).
- **No submit** (would not move the score).

## Resume line

**V7/V8 public imports failed → Track A closed. See `docs/top50-plan-v9-trackb.md`.**

### V9 Track B launched (2026-08-12)

- Plan: `docs/top50-plan-v9-trackb.md`
- Kernel: https://www.kaggle.com/code/danielsolo1770/rsna-knee-v9-rad-dino-gated
- Recipe: V1 DINO fallback + dual-gated RadImageNet ResNet-50 rank blend (verified public 0.906 path)
- Builder: `tools/build_v9_rad_dino_notebook.py`
- **Do not competition-submit** until `v9_rad_audit.json` shows `promote=true`
- Keep V1 0.899 selected until gated public ≥ 0.900
