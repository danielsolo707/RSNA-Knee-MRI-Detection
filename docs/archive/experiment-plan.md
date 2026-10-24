# Ordered experiment plan

Every experiment needs fixed folds, per-target OOF AUC, macro AUC, bootstrap
confidence intervals, runtime, and prediction correlations. Promote a change
only when its gain is plausible under grouped validation and no critical target
collapses.

| ID | Experiment | Hypothesis | Promotion gate |
|---|---|---|---|
| E00 | Exact public 0.899 rerun | Establish account/runtime/score baseline | Valid CSV, completed score, <9 h |
| E01 | Recreate public OOF and folds | Detect train/inference or CV mismatch | Reproducible within stated variance |
| E02 | Structured multi-LLM weak labels | Better supervision than one label source | Gold bootstrap macro gain; no >0.03 target loss |
| E03 | Target-specific missingness | Synovitis/effusion and Baker's policies reduce false negatives | Grouped OOF +0.003 macro or clear target gain |
| E04 | Multi-series learned attention | More signal than hard one-series slot selection | Grouped OOF +0.003 and acceptable runtime |
| E05 | CNN hierarchical MIL | Backbone diversity adds complementary ranking | Blend gain +0.002; mean target correlation <0.98 |
| E06 | Trauma specialist | Fracture/contusion need distinct sampling and context | Mean of two target AUCs +0.01, macro nonnegative |
| E07 | Meniscus/ligament specialist | Sagittal/coronal routing improves small structures | Four-target mean +0.007, macro nonnegative |
| E08 | OA ROI/segmentation crop | Standardized joint crops improve compartment scores | Three OA targets +0.007, grouped CV stable |
| E09 | Fluid specialist | Axial/fluid-sensitive series improve effusion/synovitis/cyst | Three-target mean +0.007 |
| E10 | Regularized per-target blend | OOF-calibrated diversity beats global average | Nested OOF +0.003 vs rank mean |
| E11 | Long-context 3D/video model | Through-plane continuity adds complementary signal | Blend gain +0.002 within runtime budget |
| E12 | Distill/prune/compile | Preserve score with safer inference | AUC loss <0.001; runtime/VRAM ≥25% better |

Milestones:

- **Top-50 candidate:** E00 plus one independently validated label/model gain;
  target public score ≥0.900.
- **Top-20 candidate:** at least three diverse, grouped-OOF-positive components;
  target ≥0.915.
- **Top-10 candidate:** label ensemble + specialists + diverse long-context model
  and a stable nested-OOF blend; observed leaderboard target ≥0.924.

