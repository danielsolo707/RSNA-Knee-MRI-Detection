# Top-50 plan V9 — Track B (post import-roulette)

**Date:** 2026-08-12 (updated after V9 LB)  
**Best selected:** **0.906** (V9 DINO+Rad α=0.20) — select this  
**Old anchor:** 0.899 (V1) demoted  
**Target:** public **≥ ~0.913** (live top 50; board moved from 0.900)  
**Current rank:** **~#86 / 1222**  
**Hard rules:** 5 code submits/day · dual soft+expert gate · no V5/V6 B3 reblend  

---

## Why Track A is closed

| Run | Result | Verdict |
|-----|--------|---------|
| V1 DINO public ensemble | **0.899** | Anchor |
| V5 gold-gated DINO+B3 | **0.893** | Gold LOO overfit |
| V6 B3 1+7 ep | soft **0.779** vs DINO **0.850** | Gate empty |
| V7 meniscus-DINO | soft **~0.776** | Gate fail |
| V8 Salem multi | **byte-identical V1** | Path fails; no-op |

**Banned:** more EfficientNet-B3 ↔ DINO weight soup; unreproduced “0.90x” notebooks; gold-only fitting.

---

## The one public recipe that already clears +0.001

[prvsiyan “read the report, then the knee”](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee)  
**verified public 0.906** via:

1. **Anchor:** audited public DINO family (~0.899)  
2. **Second family:** frozen **RadImageNet ResNet-50** encoder + 5-fold query heads  
3. **Combine:** **rank** blend, Rad weight ≈ **20%** (nested gold gate, not raw-prob mean)  

Public assets we can attach:

| Asset | Ref |
|-------|-----|
| RadImageNet ResNet-50 | `marwanmath/resnet-50-radimagenet-marwan` (`ResNet50.pt`) |
| Optional pre-trained heads+OOF | `antoinegg1/rsna-knee-e9-radimagenet-heads-v15` |
| DINO package + OOF | `pilkwang/rsna-knee-weights` (`oof.npz`) |
| Report teachers | pilkwang + stevenleehans + lixin73 label datasets |

Local download of Antoine/Marwan may 403/reset; **attach on Kaggle** instead.

---

## Track B strategy stack (ordered)

### B1 — Dual-gated DINO + RadImageNet (NOW / primary)

**Kernel family:** `rsna-knee-v9-rad-dino-gated`

1. Run **V1 DINO** first → always write `submission.csv` + `submission_dino_v1.csv`.  
2. Override geometry to V52 contract: **3 FS slots · 224px · 8 slices · CROP_MM=1e4 · band (0.12,0.88)**.  
3. Frozen RadImageNet encode train (+test).  
4. Prefer load public heads if present; else train 5-fold `FoundationQueryHead` on 3-teacher soft labels (gold rows weight 3).  
5. **Dual gate before overwrite:**
   - Soft OOF (report labels / derived): rad macro ≥ DINO soft − 0.015, **or** target-wise selection.  
   - Expert nested LOO (n=58): rank blend with α∈{0,0.05,0.10,0.15,0.20} must beat pure DINO by ≥ **0.001** nested, ≥3/5 outer folds choose α>0, final α ≤ 0.20.  
6. If gate fails → **keep V1**; log audit JSON.  
7. If gate passes → rank-blend test preds; **one** competition submit.

**Success:** public ≥ 0.900 without selecting below V1.

### B2 — Own Rad family with V3.2 labels (if B1 gate fails)

- Train query heads on our fold-cross V3.1/V3.2 labels (synovitis silence).  
- Soft OOF target ≥ **0.83** before any blend attempt.  
- Same dual gate vs DINO OOF.

### B3 — Second architecture only after B1/B2 (top-10 track)

- ConvNeXt / multi-series attention / specialist heads.  
- Not started until public ≥ 0.900.

---

## Dual-gate contract (frozen)

For candidate family **R** vs DINO **D**:

| Check | Rule |
|-------|------|
| Soft | per-target include R only if `auc_R ≥ best − 0.015` |
| Expert nested | outer-fold α selection; promote only if nested ≥ base + 0.001 |
| Catastrophe | no target drop > 0.02 vs DINO on soft when included |
| Submit | only if promote == true; message `V9 dual-gated rad α={alpha}` |

Tooling: `tools/dual_gate_blend.py` + in-kernel audit JSON.

---

## Budget

| Item | Estimate |
|------|----------|
| V1 DINO package infer | ~0.5–1.5 h T4 |
| Rad train encode 4407 × 24 slices | ~1–2 h |
| Head train 5× ≤24 ep | ~0.3–0.6 h |
| Test encode + infer | ~0.2–0.5 h |
| **Total V9** | **~3–5 h** one GPU notebook |
| Remaining GPU budget | ~20 h after prior folds |

---

## Explicit non-goals

- Re-blending V5/V6 B3 checkpoints  
- Submitting V8 / V7 / meniscus  
- Trusting public score claims without dual gate  
- Deselecting V1 until a gated public ≥ 0.900  

---

## Execution checklist

- [x] Close Track A (V7/V8 evidence)  
- [x] Lock B1 = RadImageNet rank blend recipe  
- [x] Build + push `rsna-knee-v9-rad-dino-gated`  
- [x] COMPLETE dry-run; read `v9_rad_audit.json` → **PROMOTE**  
- [x] Competition submit `55442177` (pending score)  
- [ ] If public ≥ 0.900: select V9; else keep V1  
- [ ] If reject on LB: B2 own heads on V3 labels  

### V9 dry-run gate + LB (2026-08-12)

| Metric | Value |
|--------|------:|
| status | **CANDIDATE_SELECTED** |
| promote | **true** |
| final_alpha | **0.20** (5/5 outer folds) |
| DINO gold rank macro | 0.8400 |
| nested blend gold | **0.8516** (+0.012) |
| Rad gold OOF | 0.8543 |
| **Public LB** | **0.906** (submit `55442177`) |
| Rank | **~#86 / 1222** |
| head_source | public_attached (antoine e9) |

Artifacts: `artifacts/v9_rad_ok/`  
Kernel: https://www.kaggle.com/code/danielsolo1770/rsna-knee-v9-rad-dino-gated  

### After V9 success — path to ~0.913

1. **V10a:** target-wise α (ACL/Syn higher, rest 0.20) if nested vs **V9 parent** supports.  
2. **V10b:** e11 diverse heads dual-gated vs V9.  
3. **V10c:** train own heads on V3 labels if public heads plateau.  
4. Parent for all gates is now **V9**, not V1.

## Resume line

**V9 = 0.906 public (rank ~86). Select it. Top 50 ≈ 0.913 — next is gated V10 for ~+0.007.**
