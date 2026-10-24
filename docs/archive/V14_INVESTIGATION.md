# V14 investigation outcome — RSNA Knee Abnormality Detection

**Date:** 2026-08-13  
**Account:** `danielsolo1770` — Kaggle CLI only  
**Decision:** **No new submission.** V9/V10 (public `0.906`) remains the incumbent.

---

## 1. What was actually built and run

### 1.1 V14 student gated kernel (remote, private)

- Kernel: `danielsolo1770/rsna-knee-v14-student-v10-gated` (version 2)
- Recipe: exact remote V10 parent + ResNet18 student, **0.50 parent + 0.50 student in per-target rank space**
  (fixed pre-registered weight; weights < 0.50 were explicitly vetoed because they preserve every parent ranking on a 3-row set)
- Runtime: Kaggle T4 (first attempt hit P100 and failed closed — no submission; second attempt pinned T4)
- Artifacts: `research/v14_top20_cli/v14_kernel_output_v2/`
  - `submission_v10_parent.csv` sha256 `29a1ae14…` — **exact match** to official V10 (`55455223`)
  - `submission_v14_student.csv` — schema/IDs/finite values verified
  - `v14_student_v10_audit.json`, `v10_rad_audit.json`, `resnet18_runtime_receipt.json`
- Local gate: `tools/gate_v14_student_v10.py` → receipt
  `V14_RUNTIME_VALID_BUT_PROMOTION_BLOCKED`, `submit_allowed=false`

### 1.2 Why promotion stayed blocked

- The student has **no direct expert-label OOF**: its validation is teacher-derived
  (`resnet18_student/training_receipt.json`: 3,931 strict non-gold train rows, 415 validation rows,
  `gold_rows_used_for_fit_or_selection = 0`), and its sealed 58-gold diagnostic is
  **diagnostic only**, exactly the gate that misled V13 (`+0.022` local → `-0.007` public).
- Per the sparse-label gating rule, a candidate without source-held-out expert-label
  evidence is not submit-ready.

---

## 2. Critical finding: the visible 3-row `test.csv` is NOT the scoring set

The competition ships `test.csv` with only 3 studies, but the public leaderboard
scores cannot be produced on 3 rows:

- Macro AUC over 12 targets with 3 rows and binary labels is a multiple of `1/48`.
- Official displayed scores: `0.899`, `0.906`, `0.893`, `0.828`.
  - `0.899 × 48 = 43.152` (not integer) → impossible under 3-row scoring
  - `0.906 × 48 = 43.488` (not integer)
  - No integer quarter-sum falls inside any rounding tolerance of these scores.
- A constraint DP over all per-target label states for V1/V3/V5/V9/V10/V13 returned
  **zero consistent solutions**.

Conclusion: public scores are computed on a hidden test set far larger than the
3 visible studies. Therefore **“ranking changes on the 3 visible rows” are not
evidence of score change** — this is the same trap that made V13 look good locally.

---

## 3. Candidate audits (all read-only, Kaggle CLI)

| Candidate | Finding | Verdict |
|---|---|---|
| Sakhawat enhanced ensemble | Same DINO family; no independent gain vs exact V10 | Rejected |
| V63 BiomedCLIP/ViTDet | OOF diagnostic above parent, but no rank change vs V10 on 3-row set; heads not publicly attachable (hash mismatch, `exact_public_family_oof.npz` missing for forks) | Blocked |
| Target-specialist student (2-target) | Only teacher validation + sealed-58 diagnostic; no direct image OOF | Blocked |
| RTA-HMIL (Blacklions) | Gold OOF macro `0.7214` — far below incumbent | Rejected |
| OrthoDiffusion probes | Diagnostic only | Rejected |
| Pilkwang/DINO 20-member | Blend with V9 never beat V9 OOF | Rejected |

---

## 4. Current verified state

- Best public: **0.906** (V9 `55442177`, V10 `55455223`)
- Rank: **99 / 1300**; Top-50 cutoff ≈ **0.915** (need +0.009)
- Submissions used: 7 lifetime; today: 0 (limit 5/day)
- Deadline: 2026-10-22

---

## 5. Next real step (no promises)

The only path that can *prove* an upgrade before spending a daily submission:

1. Train/run the student (or the 2-target specialist) so that **expert-label OOF is
   produced by a 5-fold protocol on the 4,407 training studies**, evaluated on the
   58 gold rows held out from fit (`source-held-out`), not teacher labels.
2. Gate: non-negative full-OOF macro delta vs V10, ≥3/5 folds positive,
   no target collapse >0.02, diversity (rank corr < 0.98).
3. Only then: one remote run + one competition submit; rollback = V10 parent materialized.

Any further attempt that starts from a “3-row ranking change” or a 58-gold bootstrap
alone is barred by this document.