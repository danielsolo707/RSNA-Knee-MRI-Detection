# Submission 3: top-10 program

Decision date: 2026-08-10

## Result that changes the plan

V1 and V2 both scored **0.899**. V2 added per-target votes from the independent
Tony five-fold family using only targets that were non-worse on both the 4,407-row
derived-label OOF set and the 58 gold studies. That was sensible but produced no
displayed leaderboard gain. The live top-10 boundary is **0.924**, a 0.025 gap.

Therefore V3 will not be another inference-weight or TTA-pooling probe. The DINO
0.899 model remains an anchor, but the Tony family gets zero default weight. It may
return only if a stronger new model shows nested-OOF complementarity with it.

Top 10 is a target, not a promise: a 0.025 macro-AUC jump requires new supervision
and new image representations, not a cosmetic ensemble change.

Execution checkpoint (2026-08-10): the label-bank gate passed at `0.890649`
leave-one-out gold macro AUC (`0.900955` fixed full-gold audit), with all 4,407
studies covered. The private dataset is `danielsolo1770/rsna-knee-v3-labels-folds`.
The corrected 288px fold-0 B3 notebook is packaged; folds 1-4 are not authorized
until fold 0 passes the OOF and runtime gates below.

Peer-review hold: Grok Build reviewed the packaged trainer read-only and correctly
flagged that its current output is expert-only OOF and its selected report-label
source is fixed using all 58 gold rows. Consequently fold-0 version 2 is a runtime
smoke test only, not promotion evidence. Before folds 1-4, V3.1 must select label
sources using only gold rows outside each held-out fold, write predictions for the
entire held-out fold, and either enforce or explicitly revise the per-target
source-gap gate. No competition submission may use the smoke checkpoint.

Fold-0 smoke result: version 2 ended in `ERROR`. The training subprocess was
terminated by `SIGKILL` at notebook time 2,664.8 seconds, approximately 32.6
minutes after the trainer launched. Kaggle emitted no explicit OOM or other
lower-level cause, so only the resource/process kill is confirmed. Downloaded
artifacts verified the expected input hashes, 4,407 unique UIDs, balanced fold
counts and 58 expert rows; `fold0_final.pt` and `fold0_expert_oof.csv` were not
created. This run supplies no model or promotion evidence and will not be
resubmitted unchanged.

## V3 model package

### 1. A new cross-fitted label bank

Create one private training table with, for every report and target:

- probability, positive/negative/uncertain/not-addressed state;
- evidence span, negation scope, anatomy/compartment and chronicity;
- language/script and translation status;
- agreement across the three existing public LLM-label sources plus one new
  independent structured extraction pass.

Fit target-specific fusion weights using only cross-fitted predictions on the 58
gold studies. Keep missingness policies target-specific: synovitis silence remains
unknown, while Baker's silence can receive a stronger negative prior. Gold studies
must never tune a labeler and evaluate that same labeler in the same fold.

Promotion gate:

- gold macro AUC at least 0.890 under bootstrap/cross-fitting;
- no target more than 0.02 below the best input label source;
- all 4,407 studies covered with explicit uncertainty rather than blanket negatives.

### 2. High-resolution EfficientNet-B3 five-fold family

Reproduce the strongest visible 0.903 recipe as an independently trained model:

- ImageNet EfficientNet-B3, single-channel MRI;
- one fluid-sensitive series per sagittal/coronal/axial plane;
- 12 ordered slices per series in training, up to 32 at inference;
- slope/intercept, MONOCHROME1 correction, 1st/99th percentile windowing,
  square padding and 288-pixel inputs;
- max slice-embedding pooling, mean series logits;
- report-grouped and scanner-aware five folds, no horizontal flip;
- BCE on fused soft labels, eight epochs, AdamW/cosine schedule.

Run one fold per Kaggle training notebook so every job stays below nine hours.
Publish the five checkpoints and preprocessing manifest to a private Kaggle dataset,
including hashes, fold assignments, OOF predictions and the exact label-table hash.

Promotion gate:

- exactly one OOF prediction for every training study and every gold study;
- derived-label macro OOF at least 0.865 and gold macro OOF at least 0.870;
- bootstrap lower bound improves on the 0.8497 DINO OOF anchor;
- no DICOM decode-error cluster by scanner/protocol.

### 3. A complementary long-context specialist family

Train a DINOv2-base or video/3D MIL model on multiple eligible series rather than
one hard-selected slot. Use slice attention, then series attention, with four clinical
heads: ligament/meniscus, OA, fluid/synovium/cyst, and trauma. Standardize physical
field of view and use both full-knee and joint-centered crops.

This family is retained only if it adds at least 0.004 nested-OOF macro AUC when
blended with B3, with gains present in at least six targets. A single-target gain
cannot justify its runtime.

## Honest blending and final promotion

Build an OOF matrix containing the existing 20-member DINO anchor, the new B3
folds and the long-context family. Learn nonnegative target-specific rank weights
inside nested folds, shrink them toward equal weights, then evaluate once on the
held-out outer fold. Report random/report-grouped and scanner-grouped results.

V3 may be submitted only when all gates pass:

1. Nested macro OOF is at least 0.875 and improves at least 0.015 over the DINO
   OOF anchor.
2. Gold macro OOF is at least 0.880 with no target collapse larger than 0.03.
3. The ensemble beats its strongest parent by at least 0.005 in nested OOF.
4. Prediction correlation is below 0.98 for at least six targets, proving useful
   diversity rather than duplicated rankings.
5. A full inference rehearsal projects below 7.5 hours on the assigned Kaggle GPU,
   leaves at least 20% timing margin and produces a strictly validated CSV.
6. Every model, external label table and pretrained weight is public-rule compliant
   and hash-pinned; the final notebook has internet disabled.

## Concrete Kaggle artifacts

1. `labels-v3-build`: creates the cross-fitted soft-label table and validation report.
2. `b3-fold-0` through `b3-fold-4`: train and export one fold each.
3. `long-context-folds`: trains the complementary family only after B3 passes.
4. `rsna-knee-v3-infer`: loads pinned private checkpoint datasets, runs B3 + DINO
   + approved specialist inference, applies frozen target weights, validates and
   writes only `submission.csv`.

The submission slot is spent only after the final inference notebook completes a
public dry run and all validation gates above are recorded. The live comparison
targets are top 50 at 0.899, top 10 at 0.924 and leader at 0.940.

## Evidence and references

- V1 submission `55389600`: 0.899.
- V2 submission `55391636`: 0.899; target-specific Tony blend not promoted.
- V3 submission `55400073`: 0.828. The released Tony four-fold notebook was
  reproduced byte-for-byte and its visible-test output hash matched, but its
  hidden rerun did not reproduce Tony's displayed 0.905 team score. Reject the
  standalone Tony family and do not spend slots on parameter-only variants.
- Live context after V3: our best is 0.899 at rank 96; top 50 is 0.900, top 10
  is 0.925, and the leader is 0.942.
- V3.1 fold-cross-fitted label audit: `0.893737` held-out-gold macro AUC. For
  each model fold, source ranking excludes that fold's gold rows; a best-source
  advantage below `0.015` triggers a top-two average. Dataset version 2 contains
  `labels_v3_fold0.csv` through `labels_v3_fold4.csv` and their SHA-256 audit.
- First V3.1 smoke: kernel version 3 at
  `danielsolo1770/rsna-knee-v3-1-b3-fold-0-smoke`; 224px, 8 slices, one frozen
  epoch, batch 2, workers 0, temporary uint8 cache, epoch checkpoint, and full
  882-row fold-0 OOF. Do not launch folds 1-4 or submit unless its artifact,
  UID, runtime and OOF gates pass.
- V3.1 smoke result: `ERROR` from `SIGKILL` approximately 32.6 minutes after
  the training subprocess began. With workers set to zero it cached 471 studies
  (567,439,134 bytes) and produced neither epoch checkpoint nor OOF. This rules
  out worker fan-out as the repair and confirms that on-demand DICOM decoding
  cannot reach an epoch checkpoint in the available execution window. The next
  authorized job is CPU-only preprocessing of every study into a persistent,
  manifest-verified uint8 cache; only then may the same fold-0 smoke be rerun.
- CPU cache kernel version 1 is running at
  `danielsolo1770/rsna-knee-v3-1-uint8-cache`. Required promotion artifacts are
  4,407 unique study files, fixed `[3,12,224,224]` uint8 shapes, a COMPLETE
  manifest, competition-input and module hashes, aggregate cache SHA-256, and a
  runtime within Kaggle's CPU notebook limit. It consumes no GPU or submission.
- `research/pilkwang_oof/oof.npz`: DINO OOF anchor.
- `research/pilkwang_oof/merge_gain.npz`: DINO/Tony OOF comparison.
- [Public 0.899 DINO notebook](https://www.kaggle.com/code/aadigupta7686/0-899-let-me-cook)
- [Visible B3 design analysis](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee)
- [Organizer clinical definitions](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)
- [Organizer LLM-label clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)
