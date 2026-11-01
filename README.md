# RSNA 2026 Knee MRI Abnormality Detection (Top 23% — 0.942 Private AUC)

End-to-end competition run for the Kaggle
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
code competition: multi-label prediction of **12 knee MRI findings** — ACL, MCL,
Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion,
Synovitis, Baker's, Contusion, Fracture — from a single study, scored by
unweighted macro ROC-AUC.

The project covers the whole stack a competitive medical-imaging submission needs:
label-bank engineering and audits, CPU-side slice caching, serverless GPU training
(Modal Labs L4/T4), 2.5D ConvNeXt / CoAtNet / ViT inference kernels, flip-TTA
ensembling, per-class blend weights, and receipt-first submission tooling.
**Final result: rank 1,450 / 6,425 teams (top 23%), 0.942 private macro ROC-AUC**
(0.958 public, submission V51 `57516897`).

## Highlights & Final Results

| | |
|---|---|
| **Final rank** | **1,450 / 6,425 teams (top 23%)** |
| **Private macro ROC-AUC** | **0.942** |
| Public macro ROC-AUC | 0.958 (submission V51, ref `57516897`) |
| Metric | unweighted macro ROC-AUC over 12 findings, hidden rerun |
| Format | kernels-only code competition, closed 2026-10-22 |
| Final recipe | Raptor 384+224 + ConvNeXt R6 flip-TTA + 2.5D ConvNeXt MIL, per-class blend weights |
| Best owned arm | 0.879 macro AUC on the honest 58-study gold holdout (V43) — below the promotion gate |

The private board was brutal: the leader scored 0.960, top-50 was 0.951, and
**1,322 teams finished on exactly 0.942** — the largest tie on the board. The
final submission was a verified, per-class-weighted blend of the strongest public
2.5D ConvNeXt/Raptor pipelines; every owned model had already been measured-closed
before the deadline (see the post-mortem below).

## How the pipeline fits together

```
competition DICOM studies (multi-sequence knee MRI)
      │
      ├─ CPU cache builders  (kaggle/cache_v28_224, cache_v30_224, cache_v31_280, tools/build_*_cache.py)
      │     └─► cached slice tensors, 3 slots × 3 slices @ 224/280 px — no DICOM decode on GPU
      │
      ├─ serverless training  (modal_v28/)
      │     └─► DINOv2-small partial fine-tune + slot-attention MIL head
      │          L4/T4-only workers · $22 in-code GPU spend cap · resumable checkpoints
      │          GroupKFold(5) by report group · per-fold receipts + OOF predictions
      │
      └─ inference & ensembling  (kaggle/, 60+ kernels from V1 → V51)
            └─► Raptor 384+224 · CoAtNet-R6 · 2.5D ConvNeXt MIL · flip-TTA
                  └─► per-class weighted blend ──► submission.csv ──► tools/validate_submission.py
```

## Engineering highlights

### Serverless GPU training — `modal_v28/`

- Modal Labs app that runs training **off the notebook quota**: only `L4`/`T4`
  workers are allowed (`H100`/`A100` are hard-refused in-code), with a **hard `$22`
  GPU spend cap** enforced against a persisted spend ledger.
- `facebook/dinov2-small` **partial fine-tune**: encoder frozen except the last 4
  transformer blocks, separate learning rates (2e-5 backbone / 2e-4 head), and a
  slot-attention MIL head over 3 slots × 3 cached slices.
- **Resumable checkpoints** on a Modal Volume: `last.pt` carries epoch, optimizer,
  AMP scaler and history, so a preempted run continues instead of restarting; a
  fold-0 AUC gate decides whether folds 1–4 are cloned at all.
- **GroupKFold(5)** split by hashed report group keeps a study/report from spanning
  folds; per-fold receipts and out-of-fold predictions are kept in `artifacts/v28_oof/`
  (5-fold blend: **0.818 OOF macro AUC**).

### Submission automation — `tools/`

- **Kernel builders** generate Kaggle notebooks from templates (`build_*.py`,
  e.g. `build_v48_speedy_raptors.py`), and verification scripts check the generated
  kernels before a push (`verify_v35_resume.py`).
- **Label-bank tooling**: provenance audits and agreement scoring
  (`audit_raptor_lbls.py`, `audit_hpo_labels.py`, `score_label_banks.py`,
  `fuse_pseudo_labels.py`, `build_v3_labels.py`, `build_v32_synovitis_labels.py`).
- **Gold holdout scorer + offline blend engine** (`score_v31_gold.py`, `blend_gold.py`,
  `oof_rank_blend.py`, `dual_gate_blend.py`) so blend decisions were tested on CPU
  against the honest gold set before spending GPU or a submission.
- **Validators** (`validate_submission.py`): exact column set/order, UID coverage,
  finite `[0, 1]` probabilities, row-count match — run on every submission.
- **Submit / watch ops** (`submit_v48.py`, `submit_v50.py`, `submit_v51.py`,
  `watch_submission.py`, `watch_kernel.py`, `pull_kernel_output.py`) record
  per-arm receipts — fallback counts, GPU names, timings, worker exit codes — so
  silent `try/except` degradation is caught before a submit is spent.

### Inference & ensembling — `kaggle/`

- 2.5D ConvNeXt MIL + CoAtNet-R6 + Raptor pipelines, each with **flip-TTA** and
  multi-GPU workers (data-parallel across the two T4s).
- **Per-class blend weights** for the final submission: Raptor/CoAtNet blend
  0.5/0.5 with overrides for Synovitis 0.8, PF OA 0.6, Medial OA 0.5, Lateral OA 0.2,
  Baker's 0.35 — receipt in `artifacts/v51_out/blend_receipt.json`.
- Kernel sources for every experiment from **V1 → V51** are kept in `kaggle/`
  (cache builders, fold trainers, probes, submission notebooks), so each result
  in the timeline below is traceable to code.

## Result timeline

| Submission | Public | Private | Recipe |
|---|---|---|---|
| V33 (`55771082`) | 0.936 | 0.919 | fork of `romantamrazov/rsna-knee-dinosaur-v4` |
| V44 (`56021967`) | 0.937 | 0.919 | fork of public meniscus-0.937 + Lateral-Meniscus overlay |
| V47 (`56336119`) | 0.940 | 0.923 | fork of `mattiaangeli/bend-the-knee-to-the-dinosaurs` |
| V48 (`56495458`) | 0.943 | 0.927 | fork of `mattiaangeli/speedy-raptors` v34 |
| **V51 (`57516897`)** | **0.958** | **0.942** | **fork of `goodpjw2008` CoAtNet-R6 2.5D MIL + per-class blend — final** |

Owned-model arms (V31 / V38 / V39 / V42 / V43 / V45 / V46) never passed their
promotion gates; every closure is documented with a number and a receipt in
[`docs/HANDOFF.md`](docs/HANDOFF.md) §3–§14.

## What didn't work / honest post-mortem

- **Owned models never got within striking distance.** The best honest owned run
  was V43 at 0.879 macro AUC on the 58-study gold holdout, versus a public ceiling
  of 0.943+ (0.958 by the final week). V31 topped out at 0.830 gold-OOF; V38/V39
  self-distillation collapsed; V42's fusion lost 0.002; V45 reached 0.838; V46
  errored. Root causes: label banks were the real bottleneck, a single local
  RTX 4070 + 30 h/week Kaggle T4 budget, and a gold holdout too small/contaminated
  to tune against.
- **Fork-the-moving-ceiling was the winning strategy.** V33 0.936 → V44 0.937 →
  V47 0.940 → V48 0.943 → V51 0.958 public. Each fork was verified by receipts
  (arm status, zero fallbacks) *before* spending a submit.
- **Final-day selection paid off.** With hours left, V51 (public 0.958) was chosen
  over V48 (0.943) on public/private drop-risk. The private shake-up proved it right:
  V51 landed at 0.942, V48 at 0.927 — about 530 ranks saved.
- **Compute scarcity was the wrong long-term assumption.** Quota stopped binding
  after 2026-09-23 — recipe quality was the constraint, not GPU hours.
- **Lessons for next time.** Track each candidate lineage's public/private drop
  early and select finals by shake-up risk; audit label provenance before training;
  if the ceiling is public forks, spend the saved GPU on validation depth
  (honest OOF references with per-target uncertainty) rather than owned models that
  cannot close a 0.06 gap; enter the final week with a pre-built fork pipeline and
  pre-committed selection criteria.

Full final-week log and post-mortem: [`docs/HANDOFF.md`](docs/HANDOFF.md) §15–§17.

## Repo map

| Path | Role |
|---|---|
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | Source of truth: full session log, final result, post-mortem |
| [`docs/PLAN_0940.md`](docs/PLAN_0940.md), [`docs/PLAN_TOP50.md`](docs/PLAN_TOP50.md) | Forward plans at the time (superseded, kept for context) |
| [`docs/strategy.md`](docs/strategy.md), [`docs/research-similar-competitions.md`](docs/research-similar-competitions.md) | Early research synthesis |
| [`docs/archive/`](docs/archive/) | Superseded session handoffs and earlier plans |
| [`modal_v28/`](modal_v28/) | Modal Labs serverless training: `app.py` (L4/T4 workers, $22 cap, fold orchestration) + `train_core.py` (DINOv2-small partial FT, GroupKFold, checkpoints) |
| [`kaggle/`](kaggle/) | 60+ kernel sources (71 notebooks): cache builders, fold trainers, V1→V51 submission notebooks, probes |
| [`tools/`](tools/) | 95 scripts: kernel builders, label-bank tooling, gold scorer/blend engine, validators, submit/watch scripts |
| [`artifacts/lb_scan/`](artifacts/lb_scan/) | Leaderboard snapshots incl. the full private board (`lb_private.json`, 6,425 teams) |
| [`artifacts/v51_out/`](artifacts/v51_out/), [`artifacts/v50_live/`](artifacts/v50_live/) | Final submission outputs + blend receipts |
| [`artifacts/lastday_scan/`](artifacts/lastday_scan/) | The five public kernels pulled on final day (full notebooks) |
| [`artifacts/v28_oof/`](artifacts/v28_oof/), [`artifacts/v28_remote/`](artifacts/v28_remote/) | DINOv2 5-fold OOF predictions and per-fold receipts |
| `artifacts/*_labels/`, [`artifacts/pseudo_labels/`](artifacts/pseudo_labels/) | Label banks used across experiments |
| [`research/`](research/) | Public leaderboard dumps, pulled notebooks, forum-topic JSON |

## Data use

Competition data is governed by the competition's MIRA data-use terms and is
**not** committed or redistributed here: the repo contains no DICOM files, no
pixel arrays and no trained weights (~42 GB of corpus, caches and checkpoints
were deleted when the competition closed). What remains is source code, receipts,
manifests, label banks and small per-study prediction files. Kaggle competition
data and any public kernel sources remain subject to their original authors'
terms; the blend logic and tooling in this repo are the work of the repository
owner.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Daniel Soleimany.
