# PLAN — top 50 (updated 2026-09-29)

**Current decision: keep V48.** V48 (`56495458`) is COMPLETE at public AUC **0.943**, rank **362 / 4561**. Rank 50 scores **0.952**; target **≥0.953** to clear the 0.952 tie (not a guarantee). Private scores are unavailable. The V47-in-flight and V44-selected directions below are **historical, superseded**; see [`HANDOFF.md`](HANDOFF.md) §14 for the live evidence.

## Gate 1 — audit assets before using GPU (done 2026-09-29)

Run `python tools/audit_hpo_labels.py` to repeat this read-only, aggregate-only audit of `nartaa/rsna-knee-hpo-assets` against local competition data:

- All 4 alternative label banks and the existing Raptor bank cover exactly **4,349 non-gold studies**, no duplicate IDs, missing IDs, gold overlap, or nonfinite/out-of-range targets. The 336 px local corpus covers all **4,407** organizer studies.
- `folds_v0.json` covers all 4,407 IDs: Gold58 are all `-1`; the remaining counts are **870/870/870/870/869** for folds 0–4. Exact duplicate reports stay within non-gold folds (45 groups), but **one non-gold report is identical to a Gold58 report**. This is *not* proof of patient-level grouping; `train.csv` has no patient identifier.
- **Do not use the 4,407-row `train_labels_v0/v1` tables unfiltered as training labels.** V0 has `is_gold` for all 58 plus `gold_*` values matching all 696 organizer truths. V1 lacks `is_gold` yet contains all 58 gold rows: its gold `y_*` entries match organizer truths in 195/696 cells, so provenance cannot be ruled out. On non-gold rows its `y_*` columns are exactly `labels_v1_gemlow`. Filter gold by the organizer UID set regardless of flags.
- The HPO assets are **candidate labels, not a proven scoring arm**. The original HPO worker tunes on a fixed subset of fold 0 soft-label OOF; this fold and the Gold58-best checkpoint do not provide an independent validation gate.

## Gate 2 — controlled label-only image experiment (not launched)

1. Lock **one** candidate bank (for example `labels_v1_gemlow`) and a control (for example `labels_v0raw`) *before* comparing outcomes. Train from the same public initialization with identical pixels, model, seed, fold(s), optimizer, window geometry, and **predeclared fixed training schedule**. Exclude Gold58 by organizer UID even if using the 4,407-row tables. Verify training/held-out ID sets are disjoint and all non-gold IDs have assignments. Do not resume across arms.
2. Keep a non-gold model-selection split separate from the scored held-out fold(s), or use a fixed final epoch. **Do not use the downloaded trainer's default `best.pt` or its default `raptor_oof_*` as clean evidence**: `artifacts/hpo_label_audit/train_knee.py` selects `best.pt` on Gold58 every epoch, writes OOF predictions from that checkpoint, and stores each arm's own soft labels as OOF `truth`. The `--oof_every` route instead selects top-k epochs on the held-out OOF subset. A fixed-epoch prediction/export path is required for a multi-epoch fair test; a one-epoch run is only a feasibility check.
3. Save study-level, fixed-epoch OOF predictions for the **same** held-out studies. Evaluate both arms against one immutable arm-independent reference, with per-target AUC, mask/coverage, and uncertainty. Scores against either arm's own soft labels measure agreement, **not hidden leaderboard AUC**. Seek independent adjudicated non-gold labels if feasible; otherwise treat weak-label OOF as provisional. The HPO fold-0 subset was used for tuning, so avoid it as the sole selection gate. Patient-level isolation is unverified.
4. Only after choosing a model and epoch without Gold58, check Gold58 **once as an external sanity check**, not as an unbiased test: the incumbent lineage already tuned on these 58, and one report duplicates a non-gold report. A single-target replacement would need roughly **+0.12 target AUC** for +0.010 macro before blend dilution; look for repeatable multi-target gain instead. If evidence is insufficient, **do not spend a submit**.

## Gate 3 — inference and one-factor submission

Retain V48 as the rollback. Promote only a reproducible complementary checkpoint with a verified inference recipe; do not blindly reweight existing correlated arms. Time the full inference pipeline on a **~1,322-study proxy** under Kaggle-equivalent **2×T4** conditions, with margin below the **nine-hour hidden-rerun limit** (the push checks only three public studies; actual hidden runtime cannot be measured before a submission). Verify zero missing-arm fallbacks, then submit one change. Recheck the live cutoff before spending a daily submit. GPU was **30 h remaining** on 2026-09-29 (refresh 2026-10-03); no GPU training or new submission was started during this audit.

---

## Archived plan (2026-09-18; superseded)

Historical snapshot: **V44 = 0.937**, rank **~1104 / 3956**. Top-50 **0.948**. V47 was in flight; V44 was the selected incumbent. The following steps are retained for context, **not for execution**.

## Active method (historical)

Forum recipe that actually sits at 0.936–0.947: **small 224–288 CNN + better soft labels + gold held out**.
V43 DINOv2-S/280 + weak LOO labels died at gold **0.879**. Do not resume it.

**V45 COMPLETE — FAIL.** ResNet18@224 gold-58 **0.8385**. Closed as blend arm.

**V46 (RUNNING):** fine-tune public `raptor_ft_coatnet_v5_full_swa.pt` on tasmeemreza labels, gold-58 out, `eval_first`.
Kernel `danielsolo1770/rsna-knee-v46-coatnet-tasmeem`. Keep **V44 selected**.

| gold-58 | Action |
|---|---|
| < 0.90 | discard arm; do not blend |
| 0.90–0.91 | inspect per-target; no submit |
| ≥ 0.91 | 5-fold clone, then tiny rank-blend only where corr with V44 is not ~1 |

---

## What the 2026-09-05 scan found

### Public assets (usable now)

| Asset | Score | Use |
|---|---|---|
| V44 = H&S v38 4-Raptor + Medial Meniscus bag T30/R60/bag10 | **0.937** | **selected incumbent** |
| `renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid` | 0.937 | already forked |
| Bag `renta0426/rsna-knee-public0033-meniscus-bag-v1` | trained for **both** menisci | overlay currently Medial-only |
| Dread `raptor-knee-finespacing` | author: solo **0.932** | extra CoAtNet; gold search put weight → 0 |
| Dread widedense SWA @ 62 windows | Jack: **0.930** | correlated with V44 parent |
| Pranjal 3-arm Raptor gold 0.9288 | below us | do not submit |

No other public residual bag (Fracture / Synovitis / Lat-meniscus / OA) exists.

### Closed (measured)

- V37 complement reweight **0.935**
- V42 owned-arm blend **0.934**
- V43 gold-58 **0.879** — do not resume
- Extra Raptor arms (widefov/fullspan/finespacing) zeroed on gold by Pranjal
- 3-study Contusion/ACL “fixes” — public-test hacking, useless on hidden 1322

### Forum (not in notebooks)

- Labels ≫ architecture. Soft labels first.
- Small 224–288 CNN, 5-fold, own/public LLM labels → 0.936–0.947.
- Dread: same CoAtNet, more windows +0.003; new weights **maybe Sunday**; will not dump best.
- SpeedSci rank 117: v10 labels help DINOv2, v9 labels help ConvNeXt — label×arch interaction.

### Papers (Firecrawl research index)

Clinical multi-condition knee MRI nets (Radiology:AI 2021, Eur Radiol 2025) confirm **per-structure specialists + MIL**, not a bigger 12-head backbone. Does not give us a checkpoint.

### scientific-agent-skills (K-Dense-AI)

Usable as **research procedure**, not as a trainer. Do **not** install all 163 skills.

Install only if needed later:

- `pydicom` — DICOM hygiene (we already have working LUT/MONOCHROME1)
- `paper-lookup` / `literature-review` — already covered by Firecrawl research CLI
- `hugging-science` — find published MRI checkpoints
- `scientific-critical-thinking` — gate “blend one more arm”
- `experimental-design` — 1-factor submits

---

## Route

### Gate 0

Select **V44** (`56021967`). Leave V33 as backup. Do not submit V45.

### Track 1 — V45 small CNN — CLOSED

Path `kaggle/train_v45_smallcnn/`. Gold **0.8385**. Do not blend.

### Track 1b — V46 CoAtNet FT (GPU, in flight)

Path `kaggle/train_v46_coatnet_tasmeem/`. Must beat `eval_first` baseline. Gold still ≥ 0.90 to blend.

Lateral-meniscus overlay remains deferred. Do **not** add a 5th Raptor arm.

### Track 2 — Dread drop (parallel, 0 train)

Watch `dreaddevelopment` datasets + discussion 737696. If a **new** CoAtNet lands:

1. Byte-copy V44 parent cell, swap **one** arm to the new ckpt with **that** corpus recipe (span/slots/windows from the dataset card, not maxspan defaults).
2. Keep Medial overlay.
3. One dry-run, one submit.

Expected: 0.938–0.940 if the drop is the 0.936–0.938 solo he described. Still short of 0.944 unless he releases the unblended 0.938 + we keep the meniscus residual.

### Track 3 — next public residual / 0.938+ notebook

Poll daily: `kaggle kernels list` ScoreDescending + datasets `*bag* *residual*`. Fork exact, same V44 protocol. Do not 12-target-blend a sub-0.92 arm.

### Track 4 — only if V45 gold ≥ 0.91

5-fold on the same recipe, then one fusion submit. Not before.

---

## Do not

- Resume V43 / V31 / Qwen labeling
- Submit 3-study CSV hacks
- Blend V31/V43 into V44
- Spend GPU copying Raptor from ImageNet
- Install the full scientific-agent-skills tree

---

## This week’s calendar

| When | Action |
|---|---|
| Now | V45 train RUNNING on T4. Select V44. |
| On V45 COMPLETE | apply gold gate above |
| Sat–Sun | Watch Dread; fork within hours of a new dataset |
| Daily | public-kernel/dataset poll |
