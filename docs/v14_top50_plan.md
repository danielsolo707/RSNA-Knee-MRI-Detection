# V14 Top-50 Upgrade Plan — RSNA Knee Abnormality Detection

**Status:** implementation-ready; submission is blocked until every gate passes.
**Date:** 2026-08-13
**Incumbent:** V9/V10 RadImageNet family, public AUC `0.906`.
**Rejected predecessor:** V13 fixed `0.30 DINO + 0.35 E9 + 0.35 E11`, public AUC `0.899`.
**Target:** Top 50 proxy around `0.915`; this is a target, not a guarantee.

## 1. What the evidence says

### 1.1 Leaderboard and public research

- Kaggle CLI was used for leaderboard, submissions, kernels, datasets, outputs, and discussion retrieval.
- The live top 20 was checked. Only one top-20 account exposed directly relevant public kernels; those kernels are EDA/baseline, not a reproducible final `0.929` recipe.
- Public `20-member` artifacts were tested as an anchor. They are effectively the same DINO-family signal and did not improve the incumbent on full OOF; adding them at tested weights reduced the score. They are vetoed from V14.
- Discussion evidence says the task labels are image-derived and authoritative; report text is weak supervision, not ground truth.
- Exactly 58 studies have complete image-review labels, and they are not a random sample (for example, ACL prevalence is materially different from the corpus). No label threshold or blend weight may be fitted on those 58 and treated as generalization evidence.
- DICOM metadata has scanner/site shortcut signal under random folds but loses substantially under scanner-grouped folds. It is not a primary predictor for V14.
- Public discussion evidence also warns that DINOv3 can regress when the available fine-tuning/data regime is small; V13's regression is consistent with this risk.

### 1.2 Internal postmortem

- V13's 58-row bootstrap gain was large, but its full OOF delta was negative and its public score fell by `0.007`.
- The 58-row gate therefore failed as a proxy for public LB. It is retained only as a sealed diagnostic, never as a sufficient promotion criterion.
- The incumbent V9/V10 remains the rollback and deployment parent.
- No all-`0.5` fallback is permitted. Any failed new arm falls back to materialized V9.

## 2. Candidate hypothesis

### H1 — independent student signal

A strict non-gold ResNet18 student trained on the available weak/teacher targets and multi-series volumes may capture a different error pattern from the incumbent RadImageNet/DINO arms. The student is not assumed superior; it is valuable only if it is demonstrably diverse and stable.

### H2 — report-aware auxiliary signal

A deterministic multilingual report matcher can provide an auxiliary signal for findings that are linguistically robust, especially Baker's cyst and effusion. It must not overwrite image predictions or be fitted to the 58 labels. It is used only as a diagnostic/optional low-weight arm if it shows source-held-out utility and does not inject obvious label/report contradictions.

### Deployment rule

The first V14 candidate is **the exact verified remote V10 parent plus a fixed 50% rank contribution from the independently trained ResNet18 student**. Outer weights below 50% are explicitly vetoed for this competition's three-row visible test because they preserve every parent ranking and cannot change AUC there. The report matcher and the unavailable BiomedCLIP/ViTDet heads remain diagnostic-only. V13's DINO/E9/E11 recipe is not reused.

## 3. Implementation stages

1. Materialize and hash the V9 parent before any new model work.
2. Build a strict student inference arm using existing checkpoint/artifact sources; no gold row may be used for training, checkpoint selection, blend fitting, or thresholds.
3. Align all predictions by `StudyInstanceUID` and target name, with explicit target-order and fold contracts.
4. Build an optional report-only probe with language-aware lexicons and language-specific negation direction. This is an auxiliary audit, not a label replacement.
5. Generate candidates with pre-registered fixed weights only: exact remote V10 parent, exact remote V9 parent for regression comparison, and `0.50 remote-V10 + 0.50 student` in per-target rank space. Do not spend runs on 10–40% outer Student weights because they preserve the three-study parent ordering in the observed test. The report arm is diagnostic-only and is not deployed.
6. Do not search arbitrary blend weights on the 58 gold rows. Any choice is fixed before sealed-gold inspection.
7. Run local validation and then an isolated Kaggle kernel runtime audit. Submission is one-shot only after all gates pass.

## 4. Promotion gates

All gates are blocking.

### A. Artifact and contract gates

- exact 13-column schema: `StudyInstanceUID` plus the 12 competition targets;
- exact test row count;
- exact test ID set and order;
- no duplicate/missing IDs;
- finite predictions in `[0, 1]`;
- target order matches the canonical list at every load boundary;
- model/checkpoint/feature/source SHA-256 matches a manifest;
- no credential or secret in notebook, logs, outputs, or archives;
- V9 fallback is materialized before any new arm and is independently auditable.

### B. Fit firewall

- complete gold set size and mask are checked;
- gold rows are excluded from student training, validation/checkpoint selection, report vocabulary tuning, blend fitting, and threshold selection;
- all weights and transformations are recorded before sealed-gold diagnostic output;
- no test labels or leaderboard-derived tuning.

### C. OOF utility gates

Use full 4,407-study OOF where available, plus at least five fixed study-level folds and a source-held-out diagnostic where the source allows it.

A new arm must satisfy all of:

- non-negative full-OOF macro delta versus V9, with a safety margin preferred;
- positive delta in at least 3/5 fixed folds;
- no catastrophic per-target regression; in particular no target may drop by more than `0.02` without a compensating independent gain;
- bootstrap win probability computed on study-level macro scores, not only the 58 gold rows;
- prediction diversity from V9 sufficient to justify blending (rank correlation not effectively 1.0);
- subgroup stability across protocol/plane/language/scanner groups where labels permit;
- candidate does not rely on one tiny subgroup or one target.

Because local OOF-to-public correlation is imperfect, the practical promotion threshold is deliberately conservative: the candidate must beat V9 on the full OOF and remain non-negative on every pre-registered stability slice. If this cannot be demonstrated, V9 is kept and no submission is made.

### D. Remote runtime gates

- kernel completes within the 4.5-hour cap with reserve;
- all inputs are discovered unambiguously and hash-checked;
- new-arm failure causes a hard fail-closed fallback to V9, never neutral predictions;
- audit receipt is `V14_SELECTED` only after all gates pass;
- submission CSV is re-read and re-audited after kernel completion.

## 5. Stop and rollback rules

- If student full-OOF delta is negative: reject student; do not blend or submit.
- If the report arm disagrees with image-derived labels on its diagnostic slice or shows only gold-tuned gain: reject it.
- If any contract, hash, deadline, ID, schema, or finite-value check fails: reject the run and use V9 only.
- If remote output is incomplete or ambiguous: no submission.
- If a candidate is submitted and public AUC is below `0.906`, V9 remains the official deployment parent and the candidate is not repeated.
- Never repeat an existing submission reference.

## 6. Success definition

A successful V14 is not “a notebook ran.” It is a reproducible, independently audited candidate with evidence of generalization, a valid remote output, and a public score exceeding `0.906`. Reaching `0.915`/Top 50 is the objective but cannot be promised before the hidden leaderboard evaluates the submission.

## 7. Current decision

Proceed with the student arm as the only new primary candidate. Keep report-aware matching as a diagnostic/optional low-weight arm. Do not reuse V13's DINO/RadImageNet blend and do not spend a submission quota on an unqualified candidate.
