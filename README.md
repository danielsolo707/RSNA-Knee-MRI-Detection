# RSNA Knee Abnormality Detection — CLOSED (archive)

Working repo for the Kaggle code competition
[`rsna-knee-abnormality-detection`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
(kernels-only). **The competition closed 2026-10-22.** This repo is now an archive —
weights and caches were deleted, evidence and code are kept.

**Final result: rank 1450 / 6425 (top ~23%), private AUC 0.942** via V51
(`57516897`) — a private fork of `goodpjw2008`'s public 0.958 kernel (Raptor 384+224 +
ConvNeXt R6 flip-TTA + 2.5D ConvNeXt MIL) with per-class blend weights.
Full story, final-week log, and post-mortem: [`HANDOFF.md`](HANDOFF.md) §15–§17.

## Result timeline

| Submission | Public | Private | Recipe |
|---|---|---|---|
| V33 (`55771082`) | 0.936 | 0.919 | fork of `romantamrazov/rsna-knee-dinosaur-v4` |
| V44 (`56021967`) | 0.937 | 0.919 | fork of public meniscus-0.937 + Lateral-Meniscus overlay |
| V47 (`56336119`) | 0.940 | 0.923 | fork of `mattiaangeli/bend-the-knee-to-the-dinosaurs` |
| V48 (`56495458`) | 0.943 | 0.927 | fork of `mattiaangeli/speedy-raptors` v34 |
| **V51 (`57516897`)** | **0.958** | **0.942** | **fork of `goodpjw2008` CoAtNet-R6 2.5D MIL + per-class blend — final** |

Owned-model arms (V31/V38/V39/V42/V43/V45/V46) never passed their gates; every closure
is documented with receipts in `HANDOFF.md` §3–§14.

## Repo map

| Path | Role |
|---|---|
| `HANDOFF.md` | Source of truth: full session log + final result + post-mortem |
| `kaggle/` | All submission/train kernel sources (V33 → V51, fold trainers, probes) |
| `tools/` | Trainers, gold scorer, validators, kernel builders, submit/watch scripts |
| `artifacts/lb_scan/` | Leaderboard snapshots incl. full private board (`lb_private.json`) |
| `artifacts/lastday_scan/` | The five public kernels pulled on final day (full notebooks) |
| `artifacts/v51_out/`, `artifacts/v50_live/` | Final submission outputs + blend receipts |
| `artifacts/*_labels/`, `artifacts/pseudo_labels/` | Label banks used across experiments |
| `references/spy/` | Pulled public kernel sources (Raptor train/infer, DINOsaur) |
| `docs/archive/` | Superseded plans |

Competition data is governed by the competition MIRA data-use terms and must not be
committed or redistributed.
