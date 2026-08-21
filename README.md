# RSNA Knee Abnormality Detection

Working repository for the Kaggle code competition. The first submission is an
exact, private rerun of the public `0.899 - Let Me Cook` DINOv2 ensemble, kept
unchanged so its score is a trustworthy baseline before experiments begin.

## Current verified state

- **Incumbent:** V9 (`55442177`) at **0.906** public AUC; V10 ties at `0.906`.
- **Current rank:** **99 / 1300**; live Top-50 cutoff: **0.915**.
- **V13 result:** ref `55466431`, **0.899** public AUC, `-0.007` vs incumbent.
  V13 is rejected even though its 58-study gold gate predicted `+0.02221`; that
  small-gold transfer signal did not generalize to the public leaderboard.
- Keep V9 selected. Do not retune/re-submit the same DINO/E9/E11 three-arm family.
- Use **Kaggle CLI only** for all Kaggle status, leaderboard and submission work.

## Original baseline

- Kaggle kernel: [RSNA Knee V1 verified public 0.899 baseline](https://www.kaggle.com/code/danielsolo1770/rsna-knee-v1-verified-public-0-899-baseline)
- Source: `kaggle/submission_v1/rsna-knee-v1-public-0899.ipynb`
- Runtime: Kaggle T4 GPU, internet disabled, competition data plus the public
  label/weight datasets declared in `kernel-metadata.json`
- Public reference: [0.899 - Let Me Cook](https://www.kaggle.com/code/aadigupta7686/0-899-let-me-cook)
- Metric: unweighted macro ROC AUC across 12 abnormalities
- **Submission 1 result:** `0.899` public AUC (Kaggle submission `55389600`,
  completed 2026-08-09). At the time recorded, this was approximately display
  rank 98 within a large `0.899` tie; the top-50 boundary was `0.899` and the
  top-10 boundary was `0.924`.
- **Submission 2 result:** `0.899` public AUC (Kaggle submission `55391636`,
  completed 2026-08-10). It added OOF-gated target-specific rank votes from the
  independent Tony five-fold family but produced no displayed improvement over V1.
  At the time recorded, the approximate display rank was 71 inside the large
  `0.899` tie; top 50 was also `0.899`, top 10 was `0.924`, and the leader was
  `0.940`. The V2 blend is therefore not the foundation for V3.

## Historical V5 execution status

> **Resume handoff:** read **`docs/SESSION_HANDOFF.md` first** if the agent or
> chat dies — full history, paths, Kaggle slugs, and ordered next commands.

- Historical at that point: best score was **0.899** (V1). The current verified
  incumbent is V9 at **0.906**; see the current-state section above.
- **B3 path closed:** V5 submit 0.893; V6 soft OOF 0.779 ≪ DINO 0.850.
- **Top-50 plan V7:** `docs/top50-plan-v7.md` — dual-gate public second family (meniscus-res DINO) then labels+new arch.
- Tools: `tools/dual_gate_blend.py`. Meniscus run: `kaggle/submission_v7_meniscus_dino/`.
- Full resume: `docs/SESSION_HANDOFF.md`.

## V3 execution status

- The audited V3 label bank contains 4,407 studies and uses a fixed per-target
  source chosen without evaluating a source on its own held-out gold row.
- Leave-one-out gold macro AUC is `0.890649`; the fixed full-gold source audit is
  `0.900955`. The five report-grouped folds contain 882/882/881/881/881 studies
  and 12/11/12/11/12 gold cases.
- Private label dataset: [rsna-knee-v3-labels-folds](https://www.kaggle.com/datasets/danielsolo1770/rsna-knee-v3-labels-folds).
- Fold-0 trainer: [RSNA Knee V3.1 B3 Fold 0 Smoke](https://www.kaggle.com/code/danielsolo1770/rsna-knee-v3-1-b3-fold-0-smoke). Version 3 was a one-epoch, 224px/8-slice runtime and OOF-contract probe using fold-cross-fitted labels, zero data-loader workers, a temporary uint8 disk cache, epoch checkpointing and all 882 fold-0 rows in OOF. It still received `SIGKILL` about 32.6 minutes after training began, after caching only 471 studies (567,439,134 bytes); no checkpoint or OOF was written. This reproduces the earlier failure with workers disabled, so on-demand DICOM preprocessing is rejected. A separate CPU-only cache job is required before another fold-0 GPU smoke. It is not submission evidence. The earlier full-recipe run;
  version 2 ended in `ERROR` after the trainer received `SIGKILL` about 32.6
  minutes after training launched (44.4 minutes total notebook time). The label
  manifest and all 4,407 unique fold rows were recovered, but no checkpoint or
  OOF file was written. Kaggle did not report a more specific cause, so OOM is
  not asserted. Later folds remain blocked until V3.1 adds fold-cross-fitted
  label selection, full held-out-fold OOF and resource-bounded preprocessing.
- V3.1 label audit: fold-specific source choice uses only out-of-fold gold, with
  a `0.015` best-versus-runner-up AUC gap gate (otherwise the top two sources
  are averaged). Held-out-gold macro AUC is `0.893737`; five hash-pinned label
  banks are published in private dataset version 2.
- CPU cache builder: [RSNA Knee V3.1 uint8 cache](https://www.kaggle.com/code/danielsolo1770/rsna-knee-v3-1-uint8-cache), version 1 running. It writes one compressed, atomic cache file per study with fixed `[3,12,224,224]` uint8 tensors, then verifies 4,407 UIDs and records input/module/aggregate SHA-256 hashes. No GPU or competition submission is used.
- **Submission 3 result:** `0.828` public AUC (Kaggle submission `55400073`,
  completed 2026-08-10), an exact standalone reproduction of Tony Li's public
  four-fold DINOv2-small inference notebook. This is a decisive failed external
  validation: Tony's team had displayed `0.905`, but that score did not reproduce
  from the released notebook and weights under our hidden rerun.
  The private notebook source hash matches the public source byte-for-byte and
  its validated visible-test CSV SHA256 is
  `c0b1dada1d85bcaca341e45aff7a5e3bb71f992738b42527637ad1396d866e5e`.
  No DINO-anchor blend or target gating from V2 was applied. Standalone Tony and
  further speculative Tony-weight tweaks are rejected. Our best score remains
  `0.899`; at the result audit the team was rank 96, the top-50 boundary was
  `0.900`, top 10 was `0.925`, and the leader was `0.942`.

## Repository map

- `docs/strategy.md` — research synthesis and top-50/top-10 strategy
- `docs/experiment-plan.md` — ordered experiments and promotion gates
- `tools/validate_submission.py` — strict output/schema/range validator
- `kaggle/submission_v1/` — immutable first-submission notebook and metadata
- `data/raw/` — local competition metadata only; ignored to prevent redistribution
- `references/` — locally pulled public Kaggle sources; ignored

## Validate a downloaded prediction file

```powershell
python tools/validate_submission.py kaggle/submission_v1/output/submission.csv data/raw/test.csv
```

Competition data is governed by the competition's MIRA data-use terms and must
not be committed or redistributed.
