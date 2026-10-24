# V15 — 5-Fold Expert-Label OOF for the ResNet18 Student

Status: **pre-registered** (2026-08-13) — no remote run yet
Predecessor: `docs/V14_INVESTIGATION.md` §5 (the only permitted next path)

---

## 1. Purpose

Produce a **direct expert-label OOF** for the ResNet18 student via a **5-fold protocol
on the 4,407 training studies**, evaluated on the **58 gold rows held out from fit
(source-held-out)**, not teacher labels. This is the single permitted route to
deploying the student as the V14 candidate (`0.50 rank(V10 parent) + 0.50 rank(student)`).

## 2. Fixed artifacts (all verified locally 2026-08-13)

| Artifact | Location | Shape / content | Row order |
|---|---|---|---|
| train.csv (competition) | `data/raw/train.csv` | 4,407 studies; 58 complete expert labels | StudyInstanceUID |
| DINO teacher OOF | `research/pilkwang_oof/oof.npz` | `pred` (4407, 12) + `gold_mask` (58) | same as train |
| Rad V52 teacher OOF | `references/candidates/antoine_e9_files/v52_oof.csv` | 12 cols + `fold` (0–4) + `is_gold` | same as train |
| Student checkpoints | `research/v14_top20_cli/resnet18_student/` | 2 seeds, SHA-pinned | — |
| V10 parent OOF | computed from DINO + V52 with V10 recipe | (4407, 12) | same |

**Fold split (pre-registered, pinned):** the `fold` column of `v52_oof.csv`:
{0: 882, 1: 882, 2: 881, 3: 881, 4: 881}; gold per fold {0: 10, 1: 15, 2: 16, 3: 9, 4: 8}.
No re-split is allowed after results are seen.

**Teacher targets (pre-registered):** 24 columns = [12 public DINO OOF | 12 rad V52 OOF]
(padded to row order of train.csv). Same `_teacher83` contract the student was trained on.

**Soft labels (pre-registered, V10-identical):** `_v10_make_targets(train)` from the V10
kernel — weighted mean of 3 LLM report-label CSVs
(`report_labels_v2.csv`, `llm_labels_v2.csv`, `labels_llm_gpt56sol.csv` from the same
pinned datasets V10 used), gold rows overwritten with expert labels (weight 3.0),
NaN rows → 0.5/0.05. Gate thresholds on soft labels therefore compare like-for-like
with V10's own soft gates (`soft_v9_macro` 0.85336 reference, not the local
dino+rad mean which yields a different scale).

## 3. Protocol

1. **Bridge volumes**: rebuild the V55 bridge `[3, 16, 256, 256]` uint8 for all 4,407
   training studies from DICOM (fluid-sensitive FS series per plane; physical order;
   central 16-slice window; resize 256; per-volume uint8 min/max) — the exact recipe in
   `research/v14_top20_cli/resnet18_student/inference_runtime.py::read_volume`.
   Built on Kaggle in a **CPU kernel** (same pattern as `build_v3_cache_notebook.py`),
   chunked npz + manifest + SHA256, published as a private dataset.
2. **5-fold training** (T4 kernel, ~10 trainings = 5 folds × 2 seeds × 12 epochs,
   same hyperparameters as the sealed student):
   - For fold `f`: train on the 4,407 minus fold-`f` rows, **gold rows excluded from
     fit for the fold that owns them** (source-held-out; gold rows are never trained on
     by the model that predicts them).
   - OOF predictions: fold-`f` model predicts fold-`f` rows → every study gets an OOF
     prediction from a model that never fit it. Gold OOF subset = the 58 gold rows.
3. **No in-kernel submission.** Kernel writes:
   - `v15_oof.csv` (4407 × 12 student OOF, train row order)
   - `v15_oof_24.csv` (4407 × 24, public | rad heads)
   - `v15_soft_labels.csv` (4407 × 12 V10-identical soft labels + `w` + `is_gold`)
   - `v15_audit.json` (hashes, fold AUCs, gates, receipt)
4. **Local gate tool** (`tools/gate_v15_student_5fold.py`) independently verifies
   receipt + hashes + gates using the kernel-written `v15_soft_labels.csv`
   (SHA-checked against the audit) so its numbers match the kernel's; only its
   `submit_allowed=true` may trigger a remote run.

## 4. Gates (from V14_INVESTIGATION.md §5, unchanged)

| Gate | Rule |
|---|---|
| G1 | non-negative full-OOF macro delta vs V10 parent OOF |
| G2 | ≥ 3/5 folds positive (fold-level macro delta vs V10 parent OOF) |
| G3 | no target collapse > 0.02 (student OOF per-target macro AUC vs teacher soft labels, any target) |
| G4 | diversity: rank corr (student OOF, V10 parent OOF) < 0.98 |

Additional evidence (not gates, reported only):
- gold-58 OOF macro AUC (student) vs gold-58 OOF macro AUC (V10 parent) — source-held-out
- 0.50/0.50 rank blend OOF macro on the 58 gold rows

## 5. Promotion

Only if **all gates pass** (G1–G4) and the local verifier approves:
1. One remote inference run (V10 parent + student at 0.50/0.50 rank, on test).
2. One competition submit.
3. Rollback = V10 parent materialized (SHA `29a1ae14b5610d5056b9ce06d19442d7378a3915e31c67377d615b99477e4e29`).

Any failure → keep V10; do NOT blend or fit on the 58 gold rows (barred by
`docs/v14_top50_plan.md` and `docs/V14_INVESTIGATION.md`).

## 6. Budget

- GPU quota remaining: 4.43 h (refresh 2026-08-15). Kernel B estimated ≤ 2.5 h on T4:
  bridge preload ~20 min + 10 trainings × ~8 min + OOF inference ~30 min.
- Kernel A (CPU, bridge build): est. ~2 h for 4,407 studies (V3.1 cache did 4,407 in
  3,436 s at 77/min; student bridge is larger).
- Bridge dataset upload: ~3.5 GB (4,407 × 3.1 MB) chunked.

## 7. Files

- `tools/build_v15_bridge_notebook.py` — kernel A builder (CPU)
- `tools/build_v15_student_5fold_notebook.py` — kernel B builder (T4)
- `tools/gate_v15_student_5fold.py` — local verifier
- `research/v14_top20_cli/resnet18_student/inference_runtime.py` — bridge recipe source