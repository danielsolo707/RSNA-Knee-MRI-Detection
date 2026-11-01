# Competition strategy (historical, 2026-08-10)

> **Superseded as a live plan.** Final result and next steps are in
> `docs/HANDOFF.md`. This file is the original research synthesis; do not
> treat its leaderboard cutoffs or submission-1 plan as current.

Last updated: 2026-08-10. Leaderboard thresholds move; the observed public
leaderboard was about **0.900 for top 50** and **0.924 for top 10** when this plan
was written. Those are planning gates, not promises.

## What the task actually demands

The task predicts ACL, MCL, medial/lateral meniscus, medial/lateral/PF
osteoarthritis, effusion, synovitis, Baker's cyst, contusion, and fracture from
multi-series knee MRI. The score is the mean of 12 ROC AUCs, so a weak rare
target hurts just as much as a common one. The hidden test set is roughly 1,300
studies and notebooks have a nine-hour CPU/GPU limit with internet disabled.

Only 58 of 4,407 training studies have organizer gold labels; the other 4,349
have reports but blank target columns. Positive counts in the gold set range
from 9 (MCL) to 35 (effusion), making ordinary random cross-validation highly
unstable. Report-derived weak labels are therefore necessary, but their missing
mentions must not be interpreted uniformly as negatives.

## Baseline chosen for submission 1

The verified public 0.899 notebook uses DINOv2-small features on physically
ordered three-slice inputs, sequence/plane slots, laterality normalization,
anatomy-aware heads, multiple folds/checkpoints, target-specific pooling, and
rank-mean ensembling. It validates the weight manifest and preprocessing
fingerprint before inference. This is a better first measurement than a new,
unvalidated rewrite.

The baseline's likely ceiling comes from correlated members, fixed hard routing
of one series per slot, 2D/2.5D context, and label noise inherited from report
extraction. The public score also cannot prove private-LB generalization.

## The route to top 50

1. **Reproduce and lock the baseline.** Preserve the exact first notebook,
   record its score/runtime, and fail closed on missing weights or malformed
   output.
2. **Build honest OOF.** Use multilabel-stratified folds and report both random
   and scanner/fingerprint-grouped validation. Bootstrap study-level AUC for
   per-target confidence intervals. Never tune solely on 58 gold cases.
3. **Improve weak labels first.** Extract structured probabilities, negation,
   uncertainty, chronicity, anatomy, and “not addressed” from every report.
   Audit language/script and translation errors. The public analysis found LLM
   labels near 0.878 macro AUC on gold versus 0.814 for regex; targeted synovitis
   inference from effusion improved the estimate, while blanket imputation
   regressed. Treat this as a hypothesis to reproduce, not a leaderboard fact.
4. **Add one diverse image model.** Train a ConvNeXt/EfficientNet-style
   hierarchical MIL model with the same physical geometry but learned
   slice/series attention. Blend only when grouped OOF gains and prediction
   diversity agree.
5. **Per-target rank blending.** Fit nonnegative, regularized OOF blend weights;
   use equal/rank averaging when uncertainty is too high. Public-LB probing is
   not a substitute for OOF.

The immediate promotion gate is reproducible grouped-OOF improvement plus a
public score above 0.900 without a material runtime increase.

## The route from top 50 toward top 10

### Label science

- Run two independent structured labelers and adjudicate disagreements using a
  third prompt/model or conservative soft targets.
- Calibrate report probabilities on the gold 58 with leave-one-out or bootstrap
  estimates; upweight gold without repeatedly oversampling the same patients.
- Use target-specific silence policies: Baker's silence can be close to
  negative, whereas synovitis silence is often unknown. Preserve uncertainty as
  a training weight.
- Add clinically implied OA language (cartilage loss, osteophytes, subchondral
  changes, joint-space loss) and distinguish borderline findings, which the
  organizer says should be negative.

### Image models

- Robust DICOM ingestion: transfer-syntax coverage, physical slice order,
  orientation and left/right normalization, spacing-aware resize, intensity
  windowing, duplicate/localizer rejection, and deterministic caching.
- Multi-instance hierarchy: slice encoder → series attention → plane attention
  → target-specific heads. Feed multiple eligible series rather than selecting
  only one.
- Finding specialists: trauma (fracture/contusion), ligament/meniscus,
  compartment-specific OA, and fluid/synovium/cyst. Each receives the planes and
  contrasts that carry the finding.
- ROI route: learn knee/joint localization or segmentation, then train on both
  full-field and standardized physical crops. KAMRNet is a useful architectural
  reference for segmentation-guided multi-plane classification.
- Diversity: combine DINO/ViT, CNN, and a longer-context 3D or video backbone;
  distill only after the ensemble proves its value.

### Generalization controls

Scanner metadata alone reportedly drops from about 0.652 random CV to 0.598
scanner-grouped CV, so it is a weak residual feature at most. Measure gains
under unseen-scanner folds and exclude any feature that exploits study identity
or acquisition artifacts. Track model correlation per target, not only the
macro score.

### Runtime and submission discipline

- Cache decoded/cropped tensors in training; inference must remain a single
  offline Kaggle notebook under nine hours.
- Keep one production branch, pinned assets, hashes, deterministic preprocessing,
  and a fallback only for diagnostics—not a silent constant-prediction entry.
- Reserve daily submissions for ablations with a stated OOF hypothesis. Select
  the two final entries by robustness/diversity, not public score alone.

## Relevant prior art and repositories

- [MRNet project](https://stanfordmlgroup.github.io/projects/mrnet/) and
  [competition](https://stanfordmlgroup.github.io/competitions/mrnet/): classic
  per-slice features plus study-level aggregation across knee MRI planes.
- [MRNet competition implementation](https://github.com/yashbhalgat/MRNet-Competition):
  reproducible multi-plane reference.
- [KAMRNet](https://github.com/zqiuak/KAMRNet): segmentation-guided,
  multi-plane knee MRI classification.
- [RSNA 2024 Lumbar Spine MRI](https://github.com/nshen7/rsna-2024): related
  multi-series MRI preprocessing/training structure.
- [RSNA Breast Cancer 1st place](https://github.com/dangnh0611/kaggle_rsna_breast_cancer):
  strong precedent for ROI detection, fast DICOM preprocessing, fold ensembles,
  and optimized inference.
- [SKM-TEA](https://github.com/StanfordMIMI/skm-tea) and
  [SKID](https://github.com/sadimanna/skid): useful knee MRI representation and
  anatomy resources, subject to competition-rule and license review before use.

External medical datasets must be checked for public availability, license,
non-commercial restrictions, and competition compatibility before training.

## Competition-specific evidence

- [Competition overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)
- [Organizer clinical overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)
- [Organizer clarification permitting hosted LLM report extraction](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)
- [Missing/not-addressed label analysis](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733932)
- [Scanner fingerprint validation analysis](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733517)

