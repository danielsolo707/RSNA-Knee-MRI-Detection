# DEEP RESEARCH → 0.940 program (written 2026-08-21)

**Incumbent:** V27 `55580781` **0.914** — keep selected until an owned file beats it.
**Goal:** public **0.940**. Deadline 2026-10-22 (~9 weeks left).
**Sources:** Kaggle forum topics 735304, 733517, 735154, 735767, 733932 + own ledger.

---

## 1. Hard evidence from the board (what actually scores)

| Who | Rank | Score | Setup |
|---|---|---|---|
| Tucker Arrants | 8 | **0.934 → 0.942** | single 5-fold, **224 px**, standard small arch, TTA, ~25 min infer, *crude* labels |
| Scott Willis | 25 | **0.938** | single model, **single fold** |
| k256.dev | 71 | 0.926 | single 5-fold, **336 px**, LLM labels only |
| Tim Krige | 83 | 0.92 | single model, **OOF ≈ LB within noise** |
| Tom Aindow | 46 | 0.915 | DINOv2, **392 px**, 150 mm crop (**0.383 mm/px**), **bag of 32 random slices/study** during training |
| Our soup ceiling | — | 0.911–0.914 | V22…V27 all saturate here |

**Conclusion #1:** 0.940 is reachable by ONE well-trained 5-fold family, not by soups.
**Conclusion #2:** OOF ≈ LB (Tim). Trust OOF; never burn daily submits on probes.

## 2. What does NOT move the needle (measured nulls — do not reopen)

- Bigger encoder: DINOv2-S→B = +0.0011 vs noise floor 0.0020 (stevenleehans). Our V30 confirms.
- DICOM metadata shortcut: 0.65 random / 0.60 scanner-grouped (Zhukov). Dead.
- Public soup re-weighting: 4 submissions proved the 0.911–0.914 wall.
- RAD-DINO-style swaps without matching normalization constants (silent-bug warning).

## 3. Why OUR trained models stall at OOF 0.82–0.83 (gap analysis)

Top solos sit at OOF 0.92+. The +0.09 gap is NOT capacity. It is:

1. **Depth coverage.** We feed 9 fixed slices (3 anchors × 3 adjacent). Tom feeds a
   **bag of 32 randomly sampled slices per study** during training. Meniscus/ACL
   findings live anywhere along the joint; 9 fixed anchors miss them.
2. **Effective resolution.** Tim: "if you can't see mm-level detail, neither can the
   model." Our 224 px over a 160 mm crop = 0.71 mm/px. Tom uses 0.383 mm/px.
   Tucker wins at 224 px but likely with a much tighter physical crop.
3. **Unfreeze depth + training length.** Tim: strong correlation between number of
   unfrozen layers and score. Our fold-0 ran 4 epochs and was STILL climbing when
   the alive-limit killed it (0.739→0.757→0.786→0.784). We never once trained a
   model to convergence.
4. **Physical slice ordering.** Filenames are random SOP UIDs; filename order matches
   anatomy ~5% of the time. Must sort by ImagePositionPatient·normal(ImageOrientationPatient).
   (We already do this — keep it.)

## 4. THE LOCKED RECIPE — V31

One change-set, fold-0 gated, no submission until 5-fold OOF exists.

### Input pipeline (the big bets)
- Physically sorted slices (existing code — keep).
- **mm-aware crop:** center crop 140 mm (tighter than 160 mm; knee joint ≈ 120–150 mm).
- **Resolution 288 px** (0.49 mm/px). Fallback 256 if VRAM-bound on 8 GB.
- **Slice coverage: 24 slots per plane** — 8 anchors × 3 adjacent triplets, anchors
  linspace across the middle 80% of the stack. Train-time random anchor jitter ±1.
  (MIL head aggregates slots; this approximates Tom's bag-of-32 while staying cacheable.)
- 3 FS planes as separate slot groups (keep).
- uint8 disk cache, one atomic file per study, same as V28/V30 tooling.

### Model & training
- Backbone **dinov2-small** first (proven parity with base; cheap enough to converge),
  then one **dinov2-base** arm ONLY if small converges ≥ 0.88.
- **Unfreeze ALL blocks** with layer-wise LR decay (LLRD 0.65–0.75); head LR 3e-4,
  backbone LR 3e-5. If OOM on 8 GB: unfreeze last 8 blocks minimum.
- **Train to convergence: 20–30 epochs**, cosine schedule, warmup 1 epoch, EMA 0.999.
  This is the single biggest delta vs every run we have done so far.
- Loss: BCE + 0.25 aux (keep). Labels: existing 3-teacher soft bank (keep — Tucker
  hit 0.942 with cruder labels than ours; labels are not the blocker).
- GroupKFold(5) on uid hash (keep).
- Augmentation: light — flip laterality-consistent, small rotate ≤ 7°, intensity jitter;
  NO random crop that breaks mm scale.

### Gates (raised — old 0.85 gate was too low given OOF≈LB evidence)
| fold-0 soft val | action |
|---|---|
| < 0.84 | geometry bug — inspect cached images visually before touching hparams |
| 0.84–0.87 | iterate: tighter crop → more slots → longer schedule (one lever at a time) |
| 0.87–0.89 | clone folds 1–4, expect LB ≈ OOF ± 0.005 |
| ≥ 0.89 | clone folds 1–4 AND queue the base-backbone arm |

## 5. Compute plan (three engines, zero excuses)

| Engine | Budget | Role |
|---|---|---|
| **Local RTX 4070 Laptop 8 GB** | free, unlimited | PRIMARY trainer. 11 GiB cache on disk; fp16 + grad-accum; checkpoint every epoch; resumable. Tim trains 5 folds in ~1 day on a 3070 — we have equivalent silicon. |
| **Kaggle GPU** | 30 h/week, refresh Fri 2026-08-22 00:00 UTC | Cache builds are CPU (free). Use GPU quota for fold-0 probes + final inference kernels only. |
| **Colab T4 (one account)** | ~12 h/session | Overflow folds / overnight runs. |

Modal stays blocked until the `as-...` token secret arrives; ignore it.

Disk check before starting: need ~12 GB free for V31 cache + ~2 GB checkpoints.

## 6. Execution order

**Phase 0 — tonight (CPU only, no quota):**
1. Build `tools/build_v31_cache.py` from `build_v28_cache.py`: 288 px, 140 mm crop,
   24 slots/plane, anchor jitter metadata in manifest.
2. Push Kaggle CPU kernel `rsna-knee-v31-cache-288`. Poll to COMPLETE (~2 h expected).
3. Pull cache locally (or stream to Colab). Verify manifest n=4407 + visual spot-check
   of 10 studies × 3 planes (save contact-sheet PNGs into artifacts/).

**Phase 1 — tomorrow after GPU refresh (or tonight on Colab/local):**
4. Local fold-0 V31 probe: dinov2-small full-unfreeze LLRD, 24 epochs, EMA.
   Record epoch curve. Gate per §4 table.
5. One geometry iteration max before cloning folds.

**Phase 2 — this week:**
6. Folds 1–4 (local + Colab in parallel, checkpoint-resumable).
7. 5-fold OOF report: macro + per-target + rank-corr vs V27 members.

**Phase 3 — first owned submission (V32):**
8. Inference kernel: V27 ensemble members + V31 5-fold heads, fail-closed gates
   (same contract as V29 plan: no target drop > 0.02, rank-corr < 0.95, soft-OOF
   not worse than parent by > 0.002).
9. Submit ONCE. Expected 0.92–0.93 if OOF ≥ 0.88.

**Phase 4 — toward 0.940:**
10. Second family for diversity: dinov2-base @ 336 px tight-crop OR ConvNeXt-tiny
    2.5D. Blend by OOF-fitted nonnegative weights (rank space).
11. Optional: target-specialist heads for Lateral Meniscus / MCL (our weakest).

## 7. Discipline (unchanged)

- Never another public soup submit. Never gold-58 tuning. Trainer kernels never
  write submission.csv. Kaggle CLI only. Secrets off-repo. One Google account.
- Every experiment logs: config hash, epoch curve, OOF, receipt JSON — into
  `artifacts/v31/`.
