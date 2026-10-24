# HANDOFF — COMPETITION CLOSED (archive)

> **The competition ended 2026-10-22. Final result and post-mortem: §15–§17 below.**
> Everything from §1 on is the historical session log — read it for lessons, not for next steps.
> Old forward plans ([`PLAN_TOP50.md`](PLAN_TOP50.md), [`PLAN_0940.md`](PLAN_0940.md)) are void.

**Updated:** 2026-10-25 (final wrap-up) · this file wins over `docs/archive/*`, `README.md`, and every older SESSION/TARGET/DEEP_RESEARCH note.
**Competition:** rsna-knee-abnormality-detection · closed 2026-10-22 · kernels-only
**Owner:** danielsolo1770 · team Daniel solo (`16697840`) · private GitHub mirror `rsna-knee-workspace`

---

## 0) FINAL STATE AT A GLANCE

| Item | Value |
|---|---|
| **Final rank** | **1450 / 6425** (top ~23%), private **0.942** — final submission **V51** (`57516897`, 2026-10-22 08:25). |
| V51 recipe | Private fork of `goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958` v27: Raptor 384+224 + ConvNeXt R6 flip-TTA + 2.5D ConvNeXt MIL (public 0.958). Kernel: `kaggle/submission_v51_public_0958/`. |
| Shake-up | V51 0.958→**0.942**; V48 0.943→0.927; V47 0.940→0.923; V44 0.937→0.919; V33 0.936→0.919. Choosing V51 (smallest drop) saved ~530 ranks. |
| Tie structure | **1322 teams at exactly 0.942** (board's biggest tie). 0.943 = 305 teams above; 0.944 = 211. |
| Cutoffs (private) | Top-50 **0.951**, top-10 **0.957**, leader **0.960**, median 0.925. Goal ≥0.953 public-equivalent missed by **~0.009**. |
| Repo state | Archived 2026-10-25: all weights/caches deleted (~42 GB freed), receipts+evidence kept, private LB snapshot in `artifacts/lb_scan/lb_private.json`. |

---

## 1) WHAT THIS SESSION DID (2026-08-25)

1. Re-read live LB + forum + public kernels. Public ceiling is **0.936**, not 0.924.
2. Confirmed fold-4 `best.pt` is valid (AUC 0.8261, EMA, epoch 15). Flattened it to `artifacts/v31_runs/fold4_hard/best.pt`. Deleted 0-byte `last.pt`. Wrote `receipt.json`.
3. **V31 gold-58 honest OOF = 0.8305.** 5-fold mean on gold = 0.9327 is **leaky** (gold rows were in the train folds). V31 is **not** a 0.930 submit arm. Receipt: `artifacts/v31_runs/gold_oof.json`.
4. Built, pushed, dry-ran, and submitted V33: byte-copy of `romantamrazov/rsna-knee-dinosaur-v4` v `344688577`. Hidden rerun **COMPLETE ~4 h**. Ref **`55771082` public `0.936`**. Competition submit **requires** `-f submission.csv` (without it: HTTP 400).
5. Downloaded Raptor LLM labels: `artifacts/raptor_labels/` (4,349 rows, gold held out). 16 GB corpus download **failed** (GCS SSL EOF) — retry with the signed-URL chunked pattern, not `kaggle datasets download`.
6. Archived 14 stale session/plan docs → `docs/archive/`. Deleted `_tmp_*` extracts. README now points here.

---

## 2) THE PATH (do not reopen the old soup / V31-as-primary track)

### Stage 1 — 0.924 → 0.930 — DONE

V33 scored **0.936** (`55771082`). Do not retune it. Do not add V31. Do not spend submits copying the same public fork.

On the Kaggle submission page, **select V33** if it is not already the selected file.

### Stage 2 — owned model toward 0.937–0.940

V31 280 px / 21 FS-only slots is **closed as a primary recipe** (gold OOF 0.830 vs Raptor gold 0.917). Next owned train uses **Raptor pixels + gold held out**.

Public assets already in hand or published:

| Asset | Where |
|---|---|
| Train script | `references/spy/highscore/raptor_train/knee-mri-training-the-twelve-finding-model.py` |
| Labels (4,349 soft, gold out) | `artifacts/raptor_labels/labels_llm_soft.parquet` |
| Corpus (~16 GB) | On Kaggle as attached datasets. Local `all_vols.npy` **stuck at 7.55/15.90 GB (47.5%)**: GCS `403 Access denied — not available in your location`. Resume needs a network path GCS will serve (VPN / different egress). Extra part not started. |
| Weights | `raptor-knee-widedense` (our 0.924) and `raptor-knee-maxspan` (V33) |

Fold-0 v3 hit the 9h cap after **2/12 epochs, gold 0.760** (rising). Weights: `artifacts/v34_remote/raptor_ft_v34f0.pt`. T4 pace is ~4 h/epoch at bs=2 — 12 epochs will not fit one Kaggle session.

**Do not push a resume kernel until Friday 2026-08-29 GPU refresh.** Leftover **1.66 h** is less than one epoch.

Resume is wired in `tools/train_raptor.py` (auto-finds `raptor_ft_v34f0.pt` under `/kaggle/input`). GCS blocks dataset uploads from this network, so Friday **does not** use a checkpoint dataset. It sources V34 kernel output instead.

Friday (after GPU refresh):

```powershell
$py = "C:\Users\ASUS\AppData\Local\Programs\Python\Python312\python.exe"
& $py -m kaggle kernels push -p kaggle\train_v35_raptor_f0_resume -t 32000
```

Kernel `danielsolo1770/rsna-knee-v35-raptor-f0-resume` attaches `danielsolo1770/rsna-knee-v34-raptor-f0` as `kernel_sources` (the cancelled v3 still has `raptor_ft_v34f0.pt` in output). Never write `submission.csv`.

| Fold-0 gold-58 | Action |
|---|---|
| < 0.88 | inspect windows/labels; do not clone folds |
| 0.88–0.91 | clone 5-fold; expect public ~0.92–0.93 |
| ≥ 0.91 | real challenger; 5-fold then one owned submit |

Raptor trainer never trains on gold. Copy that. Our V31 trained on gold and the leaky 0.933 must not be quoted as a result.

### Stage 3 — 0.940 (September, only after Stage 2 5-fold exists)

Owned 5-fold ± light rank-blend with public Raptor if target rank-corr is not ~1. Second **small** family (224 px class), not a bigger DINO. Label cleanup on synovitis / lateral meniscus only.

---

## 3) V31 FAMILY (kept on disk, not a submit)

Recipe: V31 cache 280 px / 21 slots / DINOv2-S full unfreeze LLRD / hard labels / batch 16.

| Fold | Weak-label val | Gold-OOF n | `best.pt` |
|---|---|---|---|
| 0 | 0.8185 | 13 | `artifacts/v31_runs/fold0_hard` |
| 1 | 0.8299 | 10 | `artifacts/v31_runs/fold1_hard` |
| 2 | 0.8361 | 14 | `artifacts/v31_runs/fold2_hard` |
| 3 | 0.8180 | 13 | `artifacts/v31_runs/fold3_hard` |
| 4 | 0.8261 | 8 | `artifacts/v31_runs/fold4_hard` |

Honest gold-58 OOF **0.8305**. Weakest: Synovitis 0.729, Lat. meniscus 0.764, Med. meniscus 0.772. Do not blend into V33.

Cache: `artifacts/v31_cache/v31_cache` (4407 complete).

---

## 4) CLOSED PATHS (do NOT reopen)

- Qwen2.5-7B labeling (0.57 then 0.71 vs gold; bank was 0.89).
- Soft-label training **on V31 geometry** (hard won ~+1 pt there). Soft is still correct **on Raptor geometry** (their trainer expects it).
- V31 280 px / 21-slot geometry as the main model.
- Public DINO/E9/E11/DINOv3 soups **without** Raptor (measured ceiling 0.912–0.914).
- Gold-58 blend-weight search (Raptor: +0.010 gold, +0.001 LB).
- Encoder-size-as-lever, DICOM metadata features, multi-account anything.
- Quoting V31 5-fold-mean gold 0.933 as if it were OOF.

---

## 5) KEY PATHS / COMMANDS

```
HANDOFF.md                                   # this file
kaggle/submission_v33_dino_raptor_0936/      # Stage 1 kernel
kaggle/raptor-single-model/                  # 0.924 incumbent
references/spy/highscore/dinosaur_v4/        # upstream 0.936 notebook
references/spy/highscore/raptor_train/       # public train script
artifacts/raptor_labels/                     # 4349 soft labels
artifacts/v31_runs/gold_oof.json             # V31 gold receipt
tools/score_v31_gold.py                      # re-run gold OOF
tools/train_fold.py / tools/v31probe.py      # V31 trainer (legacy)
tools/validate_submission.py
```

```powershell
$py = "C:\Users\ASUS\AppData\Local\Programs\Python\Python312\python.exe"
& $py -m kaggle competitions submissions -c rsna-knee-abnormality-detection --csv
& $py -m kaggle quota
& $py -m kaggle kernels status danielsolo1770/rsna-knee-v33-dino-raptor-0936
& $py -m kaggle kernels status danielsolo1770/rsna-knee-raptor-single-model
```

WMI launch for any local trainer (tool wrapper kills `Start-Process` children):

```powershell
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = "`"$py`" `"$wd\tools\train_fold.py`" <N> 24"; CurrentDirectory = $wd }
```

---

## 6) INFRA LESSONS (still true)

- WMI `Win32_Process.Create` survives shell job-kill; `Start-Process` does not.
- Single-writer `.train.lock` in `train_fold.py`.
- `transformers==4.44.2` on Kaggle images (newer breaks Dinov2 LayerScale).
- kaggle kernel-output downloads have no resume; use signed-URL chunked pattern (`tools/download_v31_cache.py`).
- Never unset `KAGGLE_API_TOKEN`. Trainer kernels never write `submission.csv`.
- **Code competition:** blends must live in a hidden-rerun notebook. CSV upload is not a submit.

---

## 8) THIS SESSION (2026-08-27) — deep scan + unblocked tooling

Work from a full live scan (LB, quota, kernels, trainer, labels, corpus, V34/V35):
all in `tools/` + `artifacts/`; nothing submitted (GPU empty anyway).

### Leaderboard (live, 2026-08-27)
- Incumbent V33 **0.936**, rank **209 / 2519**, tied with **260 teams** at 0.936.
- Top-50 = **0.937** (85 teams ≥ 0.937). Top-10 = 0.945. Leader 0.952.
- We are a **public-fork tie**; the 260-way pile means 0.936 is NOT a unique rank.
  Every score above it (0.937→0.952) is an **owned model**, not a public fork.

### Label audit (free, CPU) — findings
- Built `tools/audit_raptor_lbls.py` → `artifacts/raptor_labels/raptor_label_audit.json`.
- Raptor LLM labels (the 4,349-row set V34/V35 train on) are **high-agreement with the
  legacy banks on most targets (0.85–0.97)**. The outlier targets are:
  - **Effusion agree 0.65** (all banks disagree worst here)
  - **Synovitis agree 0.65–0.79** — but Raptor prevalence (0.24, mean 0.45) matches the
    gold rate (0.47) better than the legacy banks (0.13). Raptor seems *more correct* on Synovitis.
- Takeaway: **labels are strong; do NOT spend the next run on wholesale label rework.**
  The earlier "Synovitis/Lat-meniscus are the bottleneck" is only partially reproduced —
  Raptor labels already fixed Synovitis prevalence. Focus budget on model/geometry, not labels.

### Compute reality (the key decision-shaping finding)
- V34 T4 log: **2/12 epochs in ~9h → ~4.5 h/epoch** at CoAtNet-384 / bs 2 / grad_ckpt.
- **12 epochs = ~54 h for ONE fold; 5 folds = ~270 h** — far beyond the 30 h/week quota
  and a single 9 h kernel. **Finishing V4-384 as a 5-fold is not realistic on T4.**
- Local RTX 4070 Laptop (8 GB) is **free/unlimited**; V31 280px DINOv2-S cache is complete
  locally but that recipe is closed (gold OOF 0.830).
- **Recommendation (pending your OK): do NOT spend Friday's refresh on V34-384 the naive way.**
  Use Friday's GPU on a **smaller/cheaper variant that fits 9 h** (see options below), or
  split by running fold-0 to e.g. the best gold, then evaluate before committing folds.

### Hardening added (verified, compiles)
- `tools/train_raptor.py`: added `--require_resume` fail-closed guard (abort instead of
  silently retraining from scratch on a paid run).
- `tools/build_v35_resume.py` + `tools/verify_v35_resume.py`: rebuild + verify the V35
  resume notebook cleanly (the incremental patch had corrupted it). `verify_v35_resume.py`
  reports **COHERENT** (argparse flag + guard + argv + auto-discovery all present, compiles).
- `tools/build_folds_raptor_json.py` → `artifacts/kaggle_dataset_v32_labels/folds_raptor.json`
  (reuse the established report_group 5-fold split for the owned family).

### Plan options for Friday (pick one before spending GPU)
| Option | What | GPU use | Fit (≤9h) |
|---|---|---|---|
| A | V34-384 resume, fold-0 to ~12 ep (current plan) | ~54 h across recalls | ❌ per-session |
| B | **Downscale owned to 336→224 w/ cheap backbone** on Raptor corpus, fold-0 fast | ~×3 cheaper | ✅ |
| C | V31 local DINO-S 280 (free GPU, offline) | 0 quota | ✅ but capped 0.830 |
| D | Blend-first offline (DINOsaur+maxspan+widedense) to find 0.937- before training | 0 | ✅ |


### Option D executed (2026-08-27) — offline blend harness + findings

**Built and validated `tools/blend_gold.py`** (pure-CPU gold-58 blend engine):
- Reproduces known numbers exactly: v31 gold-OOF **0.8305** (= gold_oof.json ✓),
  v34 fold-0 **0.7596** (= raptor_ft_v34f0.json ✓) → the AUC/rank/greedy logic is trustworthy.
- Greedy best-subset in rank space correctly **rejects** the weak v34 arm and keeps v31 solo
  (blending both = 0.8208 < 0.8305). Emits `blend_predictions.csv` + `blend_report.json`.
- Reusable for ANY per-arm row-level predictions (CSV or npz) — including the strong arms
  once their per-arm files are preserved.

**Strategic finding (why naive re-weighting won't break the tie):**
- V33's 0.936 = rank-fusion of transformer (0.920) + coatnet v5_swa (0.924) with per-target
  weights. **260 teams run the same fusion** → same predictions → the tie.
- The repo's own closed-path ledger already measured: gold-58 blend-weight search on these
  two arms = **+0.010 gold but only +0.001 LB**. Re-weighting the same two arms is capped.
- **The one genuinely new combination available offline:** add the **maxspan** checkpoint as a
  THIRD Raptor arm. It is already attached to V33 (`dreaddevelopment/raptor-knee-maxspan`)
  but **not loaded** — `ARMS` has only v5_full_swa, `LEGACY_ARM` has v4_full. The kernel's own
  comment says maxspan's 58-study gains concentrate on ACL / Med-Meniscus / Lat-Meniscus /
  Lat-OA / Fracture — exactly the targets the fusion overrides. A 3-arm Raptor fusion is a
  prediction vector the public 0.936 does not contain.

**CORRECTION (2026-08-29):** the maxspan/complement arm **is already loaded and fused**.
The V33 dry-run log shows `[DINOsaur V4.2] complement raptor_ft_coatnet_v4_full.pt on cuda:1`,
i.e. the second checkpoint runs on Kaggle's 2nd T4 and is rank-fused per-target. What is
*not* done is giving it **meaningful** weight: the complement table starts at 0.10–0.16 but
the `corr > 0.992 → ×0.50` damping cut it to **0.04–0.08** on the 3-study dry run.
So Option D is not "add a 3rd arm" — it is "**let the existing 3rd arm actually vote**":
raise the complement table and/or relax the damping, then verify on real 1322-row predictions.
Caveat: `artifacts/v33_remote/*.log` is the **3-study dry run only**; the hidden-rerun log was
never downloaded, so the true hidden-run complement weights are unknown.

**Blocker found:** the maxspan checkpoint's `.pt` filename inside that dataset is not
documented anywhere in the repo, and listing the dataset needs OAuth (current token lacks it).
One line inside the Friday kernel resolves it: `print(os.listdir('/kaggle/input'))` walk.

**Friday runbook (Option D completion — ~30 min GPU, 1 dry-run, no submit):**
1. Push `kaggle/submission_v36_maxspan_arm/` (to be built: copy of V33 with (a) a dataset
   listing printout, (b) maxspan added to the coatnet rank-average as a 3rd arm,
   (c) per-arm CSVs **preserved** — remove the V18 hygiene unlink of
   `submission_coatnet.csv` / `submission_transformer_0920.csv`).
2. Dry-run on the 3-study public test → confirms wiring + prints the maxspan filename.
3. Pull per-arm outputs → run `tools/blend_gold.py` on the 58-gold eval variant if built,
   or at minimum verify the 3-arm fusion is finite/valid.
4. Only if the 3-arm fusion shows a real gain (gold direction + LB dry-run) → submit as V36.
   Gate: fail-closed, keep V33 selected otherwise.

**My pick on current evidence: D now (free, zero-risk signal) then B Friday.** D may already
reach 0.937+ by combining our own arms; B is the only owned path that fits the quota.

**Incumbent = V33 0.936; V34 gold 0.760 after 2/12 epochs; V35 resume kernel is ready (`kaggle/train_v35_raptor_f0_resume`) — push Friday after GPU refresh. GCS geo-blocks both corpus download and dataset upload from this network.**

---

## 9) THIS SESSION (2026-08-29) — full audit, no GPU spent, no submit

Re-measured everything live. Three HANDOFF claims were wrong and are fixed above:

| Claim | Reality (2026-08-29) |
|---|---|
| GPU left 1.66 h | **28.73 h**, refresh 2026-09-05 |
| Top-50 = 0.937, we are 143/2387, 103-way tie | **Top-50 = 0.940**, we are **111/2631**, **326-way tie** |
| Maxspan arm attached but not loaded | **Loaded and fused**, just at 0.04–0.08 effective weight |

### New tooling
- `tools/lb_scan.py` — **replaces the broken `tools/_tmp_lb_scan.py`**. That file imports
  `from kagglesdk import KaggleApi`, which does not exist in kagglesdk 0.1.35. Correct call
  path: `KaggleClient().competitions.competition_api_client.get_leaderboard(
  ApiGetLeaderboardRequest(competition_name=..., override_public=True, page_token=...))`.
  Writes `artifacts/lb_scan/lb.json` (2631 rows).
- `kaggle competitions leaderboard --download` returns **404** for this competition (code
  competition). Use `tools/lb_scan.py`, not the CLI.

### V35 feasibility (the hard constraint)
V35 = CoAtNet-384 / bs 2 / grad_ckpt / 12 epochs, resuming from V34's epoch-1 checkpoint.
V34 measured **2 epochs in ~9 h ⇒ ~4.5 h/epoch on T4**. The 10 remaining epochs cost
**≈ 45 h** against a **9 h per-session cap** and a **28.73 h weekly quota**.
**Finishing fold-0 is not possible this quota week**; 5 folds would be ≈ 270 h.
Any owned-train plan must cut per-epoch cost (resolution / k / bs / frozen blocks), not
just resume harder.

### Corpus download is COMPLETE + VERIFIED
Both parts finished 2026-08-30 and were integrity-checked (`tools/verify_corpus_integrity.py`):
`all_vols.npy` (3200,44,336,336) uint8 + `extra_vols.npy` (1207,44,336,336) = **4407 studies**
= 4349 train + 58 gold, byte-exact sizes. Local inference on the 4070 reproduces the Kaggle
kernel to 0.0001 AUC. Local 4070 throughput ≈ 3–4.8 studies/s (vs 1.08/s on T4) — full-arm
k24 inference over 4,349 studies runs in ~40 min locally, **free**.

### Rank arithmetic to plan against
- 0.937 → leaves the 326-way pile (rank ≈ 87).
- 0.940 → top-50 (56 teams are ≥ 0.940).
- 0.946 → top-10.
- The pile grew 103 → 260 → 326 in four days and top-50 moved 0.937 → 0.940 in two.
  **The target is a moving bar; a static 0.936 loses ~100 ranks per week of standing still.**

---

## 10) THIS SESSION (2026-08-30 evening) — V37 closed, V38 failed its gate, V39 built

### V37 result: 0.935 — WORSE. Blend re-weighting is CLOSED.
`55871971` (V33 + complement weight 0.60 flat) scored public **0.935** vs V33's **0.936**.
The +0.0041 out-of-sample gold-58 gain did **not** transfer. **Track A is closed as
measured-negative.** Keep V33 selected. Corollary: gold-58 (n=58) can rank options, not
size them; V33's fusion weights were hand-tuned by the author against the real hidden set.

### V38 result: gate FAILED — self-distillation collapse.
V38 fine-tuned `v5_full_swa` on the fused pseudo-labels (teacher = 0.6·rank(v4+v5) +
0.4·rank(soft), which **includes the student's own** in-sample predictions). Gold-58 went
0.9131 (eval_first baseline) → **0.9012** (ep16) → 0.9026 (ep17); SWA 0.9034, epoch-ensemble
0.9043. Best checkpoint = the baseline. The student was pushed toward a sharpened copy of its
own overfit predictions = confirmation bias. **Do NOT chain epochs on the V38 recipe.**
Receipts: `artifacts/remote_rsna-knee-v38-pseudo-ft/`, `artifacts/v38_out/`.

### The fix: leave-one-out teacher (validated offline, zero GPU).
Rebuilt the pseudo-labels with the student removed: `pseudo = 0.4·rank(soft) + 0.6·rank(v4)`.
Gold-58 label AUC **0.9269** (vs soft-only 0.8927, vs the self-inclusive 0.9292). The teacher
is now independent of the student. Deliverable: `artifacts/pseudo_labels/pseudo_labels_LOO_v4_w60.parquet`
(report `fusion_report_LOO_v4.json`).

Offline fusion sweep (`tools/offline_fusion_analysis.py`, report
`artifacts/analysis_offline_0830/offline_fusion_report.json`): best CoAtNet pair v5+v4 =
**0.9172**; adding our own V31 DINO OOF arm (w=0.2) lifts it to **0.9196** — the V38 variants
add nothing. V31 (our model, gold OOF 0.830) is a genuinely decorrelated third vote if it can
be shipped inside a 9 h kernel.

### V39 = V38 with the single variable changed (the teacher). PUSHED, queued.
`tools/build_v39_loo_ft.py` → `kaggle/ft_v39_loo_ft/`. Same recipe as V38 (resume v5_full_swa,
LR 1e-5/1e-4, 2 epochs, k=10, k_eval=24, eval_first, require_resume, best-by-gold) but the
labels are the LOO parquet. Gate: gold-58 must beat the eval_first baseline (~0.9131) or the
run is a no-submit (best ckpt falls back to the baseline).
**Pushed as version 3** (`kernelId 132566931`) after the user freed the GPU slots; watcher
`tools/watch_and_pull.py` polls it and pulls outputs on terminal state.
**New account limit discovered:** pushing a GPU kernel while 2 GPU sessions are live/queued fails
with an *empty-body* SaveKernel error — the real message is `Maximum batch GPU session count of 2
reached`. Cancel the running sessions first, then re-push. The CLI/SDK swallows this as a JSON
decode error; inspect the raw response body to see it.

### V39 RESULT: gate FAILED again — then CANCELLED; pivot to local sweeps
ep16 gold = **0.9024** (baseline 0.9131) — nearly identical to V38's 0.9012 despite the
student-free teacher. **Diagnosis update: the collapse is fine-tune dynamics (overshoot on a
converged model), not teacher composition.** The run was cancelled mid-ep17 (~3 h quota saved);
log pulled to `artifacts/remote_rsna-knee-v39-loo-ft/rsna-knee-v39-loo-ft.log`.

**Local sweep program (RTX 4070 Laptop, free, 5.5x T4 step rate, ~40 min/epoch):**
- `tools/train_raptor.py` + new flags `--eval_every_steps N` (mid-epoch gold eval, saves best
  immediately — the sweet spot may be INSIDE an epoch) and `--max_steps` (step cap for sweeps).
- `tools/train_local_raptor.py` — launcher patching Kaggle-only discovery onto local paths
  (corpus / ids / train.csv); verified end-to-end: local `eval_first` reproduces the Kaggle
  baseline **0.9129 vs 0.9131** (numerics only).
- First experiment **L1g** (bb 3e-6 / head 3e-5, eval every 100 steps): gold flat at
  0.906–0.911 across all 1,700 steps, never beat 0.9129. **With V38 (self-inclusive teacher)
  and V39 (LOO teacher) also failing, fine-tuning the converged v5 checkpoint on pseudo-labels
  is CLOSED in all three variants tested.** The labels are good predictors (0.9269 label AUC)
  but training a converged model on in-sample arm predictions pulls it toward memorized train
  predictions, not generalizable signal. Receipts: `artifacts/local_runs/L1g/`.
- Next owned-arm candidate: retrain our own DINOv2 arm on the new labels (v31 cache is complete
  locally), or add the existing v31 OOF arm to the V33 fusion.

**V41 gold probe (pushed 2026-08-30 evening, version 1, RUNNING):** `tools/build_v41_gold_probe.py`
→ `kaggle/probe_v41_gold/`. Runs the ENTIRE V33 chain (DINOv2 ensemble → 3D-MIL → rad/calibration
→ CoAtNet dual-ckpt fusion) on the **58 gold TRAIN studies** via a read_csv interceptor + 5
asserted source patches; snapshots every stage and scores them. Outputs:
`probe_stage{1..4}*.csv`, `probe_coatnet.csv`, `probe_gold_stages.npz`, `probe_gold_report.json`.
Watcher pulls them on completion to `artifacts/remote_rsna-knee-v41-gold-probe/`.
**Then:** `tools/simulate_v33_fusion.py` simulates adding owned arms on top of the true V33
fusion, with mandatory split-half validation (V37 lesson) before any submit.

### V41 RESULT — probe worked; gold-58 turns out to be CONTAMINATED for V33
Ran in ~11 min, all outputs pulled to `artifacts/remote_rsna-knee-v41-gold-probe/`.
Per-stage gold-58 macro AUC:

| stage | gold-58 |
|---|---|
| stage1_dinov2 (20-member DINOv2) | **0.9958** |
| stage3_rad (calibrated) | 0.9741 |
| stage2_mil | 0.9594 |
| **stage4_final (the 0.936 fusion)** | **0.9580** |
| coatnet branch (v5+v4) | 0.9202 |

**The critical finding: these gold numbers are IN-SAMPLE, not generalization.** The V33
notebook's own comments reference tuning "on the 45-study gold set (phase2/blend_panel.py)"
and a "58-study gate" — the author tuned this fusion ON gold-58. A genuinely out-of-sample
model does not hit 0.9958 macro across all 12 targets on 58 studies. So gold-58 overstates
V33 by ~0.02 (gold 0.9580 vs hidden-test 0.936), exactly the same trap as V31's leaky 0.933.

**Consequences:**
- gold-58 CANNOT validate fusion changes to V33 — the metric is already exploited by V33's
  own tuning. This also retro-explains V37: our gold-guided re-weighting didn't transfer
  because gold was already mined by V33.
- Clean/OOS arms on gold-58 remain: v5 (0.9131), v4 (0.9139), v4_full_swa (0.9120),
  v31_oof (0.8305) — raptor excludes gold and v31 is OOF.
- `simulate_v33_fusion.py`: adding ANY clean arm to stage4_final lowers gold-58 (all
  negative). The script's top hit (stage1_dinov2 + v31_oof, +0.031 split-half) is a
  LEAKAGE ARTIFACT of stage1's contaminated 0.9958 — do not act on it.
- Reaching 0.940 by gold-guided fusion of V33 is CLOSED. Remaining routes: (a) submit a
  V33+owned-arm blend and measure on hidden test directly (submits are quota-free), or
  (b) build a genuinely new owned model (retrain v31-style DINOv2 on the LOO labels).

### Route B LAUNCHED (2026-08-31 ~00:45, local 4070, free)
`tools/train_local_dino_loo.py` → v31probe with new **gold_holdout** mode: trains ONLY on
the 4,349 pseudo-labelled studies (`pseudo_labels_LOO_v4_w60.csv`), validates every epoch on
the 58 gold studies with their TRUE labels — gold never enters training, so the number is
honest for this arm (unlike every V33 branch). DINOv2-S full-unfreeze LLRD, EMA, batch 8×accum 2,
20 epochs, patience 6, ~12–13 min/epoch locally (1.4 it/s at bs4). Output dir
`artifacts/dino_loo_runs/loo1/` (console.log, best.pt by gold macro, receipt.json).
**Gate:** gold-58 macro ≳ 0.87 → then fuse with V33 and submit. v31 hard-label reference:
honest gold OOF 0.8305; weak-label val per fold 0.818–0.836.
**Status: PAUSED by user at epoch ~2** to focus Kaggle-side. Smoke verified (1.4 it/s, ep1
gold 0.68, eval loop works). Resumable from `artifacts/dino_loo_runs/loo1/last.pt`.

### V42 = V33 + our v31 owned arm (route a) — RESULT: hidden 0.934 vs V33 0.936 (−0.002, REJECTED)
- Submitted 2026-08-31 (ref 55904718). Hidden rerun scored **0.934** — blending the
  0.8305-OOF v31 arm at w=0.10 *diluted* V33. Route-(a) premise falsified on the only
  clean measurement surface: a weak owned arm must reach ~0.92 before blending helps.
- Build (kept for reference): 5 × fold best.pt (88.8 MB each) as private dataset
  `danielsolo1770/rsna-knee-v31-weights`; `tools/build_v42_v31_fusion.py` →
  `kaggle/submission_v42_v31_fusion/`: 4 V33 cells untouched + snapshot + v31 arm
  (280px/21-slot TEST cache from DICOM, 5-fold DINOv2-S inference with TTA flip,
  fail-soft) + rank fusion `submission.csv = 0.9*rank(V33) + 0.1*rank(v31)` (flat global
  weight — V37 lesson; gold is contaminated for V33 so no gold-tuning). Per-arm CSVs +
  npz preserved for offline analysis. Upload lesson: `kaggle datasets create -p
  <ABSOLUTE path>` — relative/slash paths break the CLI resume-tracking silently.

### V43 = owned arm from scratch (DONE 2026-08-31 — honest gold-58 = 0.8792, blend DEFERRED)
V42 showed the existing v31 arm (gold OOF 0.8305) dilutes V33 at blend weight 0.10. Gate for
route-(a): the owned arm must reach ~0.92+ before blending helps. V43 trains a fresh DINOv2-S
on the LOO pseudo-labels with gold-58 held out for honest validation, so the gold macro AUC is
an unbiased gate number (unlike V33's, which is contaminated by gold-tuning).

**Build:** `tools/build_v43_owned_train.py` → `kaggle/train_owned_v43/`. Embeds `tools/v31probe.py`
(both `if __name__ == "__main__": main()` variants stripped — Kaggle cells set `__name__` to
`"__main__"`), plus the proven transformers==4.44.2 pip pin (newer breaks Dinov2 LayerScale).
Labels attached as private dataset `danielsolo1770/rsna-knee-v43-labels` (uploaded this session,
CC0-1.0, contains `pseudo_labels_LOO_v4_w60.csv` only). Cache reused from
`danielsolo1770/rsna-knee-v31-cache-280` (fold4 provenance). Competition data attached for
`train.csv` (the gold-58 ground truth).

**v31probe.py hardening for V43:**
- New `Cfg` fields: `max_wall_h` (Kaggle wall budget, default 0=off) and `ckpt_every_min`
  (periodic resumable checkpoint, default 0=off).
- `out_of_time(reserve_s)` helper; don't-start-epoch guard (30-min reserve); mid-epoch hard
  stop (8-min reserve); `save_resume(note)` writes `last.pt` + `history.json` + `receipt.json`.
- `evaluate()` now returns `(auc, Y, P)` — on best gold-holdout epoch, persists
  `gold_preds.npz` (pred, truth, ids) for offline analysis.
- `receipt.json` carries `status: "complete" | "wall-stopped"` and `epochs_done`.

**Kaggle config (9h T4 session, ~12 min/epoch at batch 16):**
```
epochs=26, batch=16, accum=1, lr_head=3e-4, lr_bb_top=5e-5, llrd=0.7,
patience=6, warmup=2, ema_decay=0.999, num_workers=2,
max_wall_h=8.3, ckpt_every_min=20
```
Patience 6 means early-stop if gold macro stops improving for 6 epochs; 26 epochs is the
ceiling (~5.2 h at 12 min/epoch) well inside 8.3 h wall.

**Local smoke (2 epochs, batch 4×accum 4, RTX 4070 Laptop, free):** loss 0.8425 → 0.8192,
gold-58 val AUC 0.6775 → 0.6881 (both epochs `save_resume("running")` fired, `gold_preds.npz`
written, `receipt.json` `status="complete"`, `epochs_done=2`). Wiring is clean; Kaggle path
exercises the same code with wall budget enabled.

**Push status — v1 FAILED on a P100, v2 re-pushed with the T4 pin:**
- **v1 (2026-08-31 ~14:55) ERROR at 0.26 h:** `CUDA error: no kernel image is available`.
  Root cause: the metadata was **missing `"machine_shape"`**, so Kaggle scheduled it on a
  **Tesla P100 (sm_60)**, and the base-image torch requires sm_70+ (min capability 7.0).
  Every earlier working kernel (V33, V42) pins `"machine_shape": "NvidiaTeslaT4"` — that
  field was silently load-bearing.
- **v2 (re-pushed ~15:35):** added `"machine_shape": "NvidiaTeslaT4"` to the metadata AND
  a startup GPU-capability guard in the trainer cell (`assert torch.cuda.get_device_capability()
  [0] >= 7`) that fails in seconds, before the ~10-min cache load, if scheduling ever picks
  an incompatible GPU again. Status QUEUED → waiting for a T4 session.
- Runner-up note: pull-log on Windows needs `PYTHONUTF8=1` (the CLI's default cp1252 codec
  chokes on the log's non-ASCII).

Version 2 URL: `https://www.kaggle.com/code/danielsolo1770/rsna-knee-v43-owned-train`.
Expected wall time ~5-7 h. Receipt to `artifacts/remote_rsna-knee-v43-owned-train/` via
`tools/watch_and_pull.py` on terminal state.

**Decision gate on completion:**
| gold-58 macro AUC | Action |
|---|---|
| < 0.87 | arm too weak to blend; close route-(a) |
| 0.87–0.92 | promising; build V44 = V33 + V43 arm (rank fusion, flat global weight) |
| ≥ 0.92 | real challenger; submit as V44 and/or 5-fold clone |

**RESULT (2026-08-31 ~21:39): honest gold-58 macro = 0.8792 (best epoch 19). Blend DEFERRED.**
- Ran all 26 epochs (~6.1 h, ~13.7 min/epoch), stopped by `patience=6` after ep19 never
  improved; log ends cleanly `early stop best=0.8792 / DONE best=0.8792`. Pulled
  `best.pt` (88.8 MB), `gold_preds.npz`, `history.json` to
  `artifacts/remote_rsna-knee-v43-owned-train/v43_out/`. Per-target gold AUC: weak on
  Fracture 0.761, Lateral meniscus 0.780, Synovitis 0.781, Lateral OA 0.804, Medial OA
  0.813; strong on Baker's 0.978, MCL 0.968, Effusion 0.963, Medial Meniscus 0.935.
- Output-commit anomaly: `receipt.json` and `last.pt` never landed in the Kaggle output
  (last.pt is ~355 MB with optimizer state — likely dropped by the output commit). For a
  COMPLETED run this is cosmetic (best.pt is the usable checkpoint); fix only matters if
  a future run needs mid-epoch resume.
- 0.8792 clears the 0.87 low gate but sits in the marginal band. Offline decorrelation
  check (spearman rank-corr of V43 gold preds, n=58): vs **V33 stage4_final = 0.87**
  (high — little independent vote), vs our old v31-OOF arm = 0.66. Combined with the V42
  lesson (a weaker owned arm diluted), blending now would very likely dilute again.
- **Decision (Option A, user-confirmed): do NOT blend V43 into V33.** Spend Friday's
  (2026-09-05) GPU refresh on a STRONGER owned arm instead, then reconsider fusion.
- Friday plan: resume/extend the V43 recipe toward ≥0.90-0.92 — it was still slowly
  improving at ep26 (val plateauing, EMA fluctuating), so more epochs / a second seed or
  fold, or a label/hyperparam improvement. `best.pt` is preserved to init from. GPU quota
  was 4.99 h at completion (this run spent ~6.1 h).

### Infra lessons added this session
- **GPU kernels MUST pin `"machine_shape": "NvidiaTeslaT4"` in kernel-metadata.json.**
  Without it, Kaggle may schedule on a P100 (sm_60); the current base-image torch needs
  sm_70+ and dies with `CUDA error: no kernel image` (cost V43 v1 a session). V33/V42
  had it all along — copy it into every GPU kernel. Add a startup
  `torch.cuda.get_device_capability()` assert so a bad assignment fails in seconds, not
  after loading the cache.
- Long background watchers MUST use file credentials (`~/.kaggle`): the env
  `KAGGLE_API_TOKEN` on this machine is short-lived and expires mid-run (~40 min);
  `tools/watch_and_pull.py` now strips it to force file auth. BUT file-auth long watches
  still went into a persistent "Generate New Token" error loop (~9-15 min in) during V43;
  a fresh process always re-authenticates fine. `watch_and_pull.py` now uses **exponential
  backoff** on repeated status errors (min(interval*2^n, 900s)) instead of hammering every
  poll — this let the V43 watch ride through and catch COMPLETE at 4.5 h.
- Never leave a background training run behind `| head`/`| tail` on Windows — when the
  pipe consumer exits the writer dies (cost one smoke run). Redirect to a file instead.
- Kaggle push errors arrive as an EMPTY-body 200; patch/inspect the raw response to see
  messages like `Maximum batch GPU session count of 2 reached`.

---

## 11) THIS SESSION (2026-09-05) — public scan + V44 fork pushed

Full notebook + discussion pass before spending the Friday GPU refresh.

### Live board
- 0.936 pile **326 → 546**. Rank **111 → 696**. Top-50 **0.940 → 0.943**. Leader 0.953.
- GPU quota refreshed: **30 h / 0 used**.

### New public artifact that actually moves the needle
`renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid` (26 votes, 2026-09-04) + dataset `renta0426/rsna-knee-public0033-meniscus-bag-v1`.
Not a 12-target blend. Parent = H&S v38 (4-view Raptor: maxspan + native384 + native384dense). Overlay = DINOv2-**Base** fullfit bag, **Medial Meniscus only**, rank mix Transformer 0.30 + Raptor 0.60 + bag 0.10. Other 11 targets byte-copied. That is why V42 failed and this works.

### Forum (no new 0.940 notebook)
- tennogh 0.942/0.943 TTA @ 288; Tucker 0.943 5-fold @ 224; Scott 0.938→0.947 5-fold small model; Yann 0.936 1-fold ResNet 224. Labels > architecture. Soft first.
- Dread: more Raptor weights maybe Sunday; published ckpts same labels, different geometry; SWA slightly worse than last epoch. Jack: widedense SWA solo **0.930** (our Raptor solo was 0.924 — different infer).
- `tasmeemreza/rsna-knee-refined-llm-labels` v2: CoAtNet 0.919→**0.926** on public LB. Best new label bank.
- native384 / native384dense / finespacing exist; 0.937 kernel already uses the first two.

### What we did
Exact private fork → `kaggle/submission_v44_meniscus_0937/`, pushed v1, **RUNNING**.
Configs copied to `references/spy/public0033_meniscus/`.
**Do not submit until COMPLETE.** Keep V33 selected if V44 ≠ 0.937.

### After V44 scores
V44 **HIT 0.937** (`56021967`). Do **not** resume V43.

**V45 COMPLETE 2026-09-05 — FAIL GATE.** Honest gold-58 **0.8385** (20 ep, 1.08 h). Worse than V43 0.879. Do **not** blend. Receipt: `artifacts/remote_rsna-knee-v45-smallcnn-224/raptor_ft_v45.json`. Weak: Synovitis 0.747, PF OA 0.773, Lat meniscus 0.784.

**V46 started:** public CoAtNet-384 SWA fine-tune on tasmeemreza (author 0.919→0.926). Kernel https://www.kaggle.com/code/danielsolo1770/rsna-knee-v46-coatnet-tasmeem
`eval_first` keeps public baseline if FT hurts. Gate: must beat eval_first **and** gold ≥ 0.90 before any blend. Keep V44 selected.

---

## 12) THIS SESSION (2026-09-18) — public 0.941 fork (V47)

Public ceiling moved 0.937 → **0.941** while we sat on V44.

### Live board
- V44 still **0.937**, rank **~1104 / 3956**, 77-way pile.
- Public 0.941 pile: **534 teams**. Top-50 **0.948**. Leader **0.957**.
- GPU **30 h** unused at start of session. V46 ERROR (eval_first gold 0.9126, 0 epochs).

### Notebook forked
Exact private copy of `mattiaangeli/bend-the-knee-to-the-dinosaurs` (canonical public **0.941**):
4-view Raptor + residual-gated CoAtNet e4/e6/e8 (α=0.40) + per-target outer weights
(ACL 0.75, Medial Meniscus 0.80, **Lateral Meniscus 1.00**, Lateral OA 0.75, Fracture 0.75, else 0.60).

Kernel: `kaggle/submission_v47_public_0941/` → `danielsolo1770/rsna-knee-v47-public-0941` v1 **COMPLETE** (3-study dry-run, coat arm 0 failures).
Submitted **`56336119`** PENDING. Expected public **0.941**. Keep V44 selected if V47 ≠ 0.941.

### V47 RESULT: **0.940** — took the incumbent slot
`56336119` scored public **0.940** (not 0.941). Rank moved ~1104 → **1069 / 4226**.
Smaller than the fork's parent, but a real +0.003 over V44. V47 is now the floor.

---

## 13) THIS SESSION (2026-09-23) — live audit, V48 built, the plateau confirmed

### Live board (re-measured, not inherited)
- V47 **0.940**, rank **1069 / 4226**, **978 teams strictly above us**.
- Top-50 **0.950** · top-10 **0.955** · leader **0.958**. Gap to top-50 **+0.010**.
- Pile shape: 0.941 n=399, 0.942 n=228, 0.943 n=194, then a **cliff** — 0.944 n=34.
  Standing still at 0.940 costs ~100+ ranks/week.
- GPU **29.83 h of 30 h remaining** (refresh 2026-09-26) — but see the correction below:
  a push costs ~8 min, so **quota is not the binding constraint**. The binding constraints are
  recipe quality and the 5 submits/day.

### The public ceiling moved — and we know exactly to what
`mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` (**2026-09-22**, 148 votes) is the
successor to the Dinosaurs notebook our V47 forked. Per `jiweiliu/rsna-knee-fast-2xt4-inference`,
**v34 of that notebook scored 0.943 publicly** and is the current anchor:
- four CoAt readers at **0.25 each** — residual-gated top3, **D4 Depth-Zone SWA3**, **Global96 top3**,
  **Repair-v1 top3** (the last two are new since our V47 fork)
- CoAt family fusion changed from rank-of-mean-ranks to **rank-of-mean-probabilities**
- CoAt/Raptor stays **0.40 / 0.60**

Verified by inspection: our fork contains the `repairv1` wiring (18 hits) and the
`_coat_member_probability` probability-mean reduction, so the pulled default **is** the v34 recipe.

### The plateau is documented, not guessed
`evgendvorkin/rsna-baseline` (221 votes) is an **audit + negative-result record** of our exact
Dinosaurs lineage. It ran **five controlled variations** off the 0.940 reference and reports
completed submission-API scores: three tied at **0.940**, two regressed to **0.922**.
- Regressions both came from **intensity-gain TTA integration** (gains 0.9/1.0/1.1). Rejected.
- E11/E13 head residuals and alternate Rad-head residuals: **no displayed gain** (all 0.940).
- Its conclusion: the three-decimal public score cannot resolve these differences, and the
  Dinosaurs recipe is a **plateau**, not a slope.
- It independently rediscovers **our own V41 finding**: Gold58 is a *selection set* for the
  residual-CoAt release (its trainer uses Gold58 to pick published epochs), and evaluating all
  five fold heads on a training row is **contaminated**. Do not gold-tune fusion on it.

Cross-reference: `docs/archive/` + §10 already closed gold-guided fusion of V33 for the same reason.

### V48 built (fork of the 0.943 anchor)
`tools/build_v48_speedy_raptors.py` emits two variants from the pulled source:
- `kaggle/submission_v48_speedy_raptors/` — FULL run
- `kaggle/submission_v48_dry/` — first-3-studies variant (later found identical in cohort to the ordinary push, since public test has only three studies). Every author gate intact.

Metadata carries all 14 dataset sources, both kernel sources, the private-BYOD docker image, and
the load-bearing `"machine_shape": "NvidiaTeslaT4"` pin.

**Why a verification run first:** the CoAt readers are wrapped in `try/except` and degrade
**silently** — a missing arm just drops a member instead of failing the run. Only the receipts show it.

### CORRECTION — the public test is 3 studies, so a push is minutes, not hours
`data/raw/test.csv` is **4 lines = header + 3 studies**. That reframes the whole execution model:

| stage | what runs | cost |
|---|---|---|
| `kernels push` → notebook run | the **3 public studies** | **~15 min** |
| `competition_submit_code` → hidden rerun | the **1322 hidden studies** | hours, server-side, **cannot be dry-run** |

So the notebook's "runs comfortably in less than 9 hrs" describes the **hidden rerun**, not our
push. Our `[:3]` cohort patch was therefore a **no-op** — both "dry" and "full" ran the same 3
studies. The real verification is reading the receipts from the ordinary push. (V47's "3-study
dry-run" language in §12 was the same misunderstanding.)

**Measured consequence — compute is NOT the constraint.** After both V48 runs, quota read
**0.17 h used of 30 h**. One full iteration costs ~8 minutes of GPU, not a 9 h session.
Every "we cannot afford this experiment" argument in §9–§10 rested on a 4.5–9 h per-run
assumption, and that assumption was **wrong for the push stage**. The genuinely scarce
resources are a **better recipe** and the **5 submits/day**. Stop gating experiments on quota;
gate them on whether the recipe can plausibly beat 0.943.

### V48 verification: GREEN
Every author gate passed and **all four CoAt readers loaded** on 2×T4 with **zero fallbacks**:

| arm | receipt status | models | fallback |
|---|---|---|---|
| `resgated_top3` | `VALID_COAT_RESGATED_EP10_TOP3_RANK_SUBMISSION` | 3 | 0 |
| `global96_top3` | `VALID_COATNET_GLOBAL96_TOP3_T4X2_SUBMISSION` | 3 | 0 |
| `repairv1_top3` | `VALID_COATNET_REPAIRV1_TOP3_PROBABILITY_RANK_SUBMISSION` | 3 | 0 |
| `d4_swa3` | present in family members | 3 | 0 |

`_coat_raptor_blend_receipt.json` confirms the contract is the **0.943 v34** one, not an older recipe:

```json
{"contract":"public_raptor_private_residual_coat_v32_no_inner_rerank_v1",
 "family_reduction":"rank_of_member_probability_mean",
 "members":["resgated_top3","global96_top3","d4_swa3","repairv1_top3"],
 "private_alpha":0.4,"public_raptor_alpha":0.6,
 "within_coat":{"resgated_top3":0.25,"global96_top3":0.25,"d4_swa3":0.25,"repairv1_top3":0.25}}
```

### V48 SUBMITTED — ref **`56495458`** (v1 COMPLETE, 0.25 h)
Accepted by the API, which also proves the kernel wrote a valid `submission.csv`.
Hidden rerun scoring now. **Expected ~0.943.** Keep V47 0.940 as the floor.

### Frontier intel: 0.943 is the anchor and nobody has publicly beaten it
- `romantamrazov/rsna-knee-dinosaur-v5` states it uses the **0.943 pipeline as the immutable
  primary inference graph** and only *adds* an independently trained ConvNeXt afterwards, with
  Gold58 OOF gates (≥0.72 whole-model, ≥0.70 per target) and a 25% cap. It does **not** claim a
  higher score than the anchor.
- `jiweiliu` v9/v10 and `haideptry` both restate 0.943 as the current recipe.
- Conclusion: the public frontier is **0.943**, and the only remaining public direction is
  "guarded residual on top of 0.943".

### Honest read on top-50
Forking public work tops out at **0.943**. Top-50 is **0.950**. No public asset closes a
+0.007 gap that the public ceiling itself does not contain, and our owned track is closed
across every variant measured (V43 0.879, V45 0.838, V46 ERROR, V38/V39 collapse, V42 −0.002).
Realistic plan: bank 0.943 now, then keep polling for the next drop. Top-50 stays a stretch goal.

### Network lesson (new)
`api.kaggle.com` and `www.kaggleusercontent.com` throw **intermittent
`SSLEOFError: UNEXPECTED_EOF_WHILE_READING`** on this machine. Stripping the proxy env vars
(`tools/pull_kernel_output.py` does this) is **not sufficient** — the failure is transient, not
proxy-only. **Retry loops are required**; `tools/submit_v48.py` wraps every API call in
30 attempts with a 20 s delay and rides straight through.

- **Never pull this kernel's output unfiltered** — the CoAt readers vendor whole `cv2`/`timm`
  trees, so it is thousands of files. Use `--file-pattern "(receipt|_complete\.json|submission)"`.
- `kaggle competitions submit` (CLI) can still work when the Python SDK is flaky; the Python API
  gives real error text, so prefer it when you need to know *why* a submit failed.

---

## 14) LIVE AUDIT (2026-09-29) — V48 scored, top-50 target moved

- `python tools/watch_v48_score.py --once` and the Kaggle submission list confirm V48
  (`56495458`) **COMPLETE, public 0.943**. V47 (`56336119`) remains the 0.940 backup.
  This is a public-LB result; private scores are still blank.
- `python tools/lb_scan.py` fetched **4561 teams**: our rank **362**, 252 strictly
  above; rank 50 **0.952**, rank 10 0.957. There are 10 teams at 0.952 and 11 at
  0.953: aim for **at least 0.953**, preferably more as the cutoff moves. The
  0.943 tie has **548 teams**; another identical public fork is not a top-50 route.
- `python -m kaggle quota`: 30.00 GPU hours remaining, refresh 2026-10-03.
  `python tools/submit_v48.py --check`: 5 submissions allowed now. Neither budget
  is a reason to submit an unvalidated blend.
- `python tools/verify_corpus_integrity.py` passed: both local 336 px Raptor
  volume parts are present (4407 studies). Local 4349-row soft labels and two
  public Raptor checkpoints are present. `data/raw/` contains competition CSVs.
- Public frontier check: scored 0.945 notebooks
  [`pjmathematician/rsna-knee-d4-lite`](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-lite)
  and [`aastikrajan15/knee-s75-w50`](https://www.kaggle.com/code/aastikrajan15/knee-s75-w50)
  rely on private assets in their scored versions; neither is a reproducible
  +0.009 solution. [`nartaa/rsna-knee-hpo-assets`](https://www.kaggle.com/datasets/nartaa/rsna-knee-hpo-assets)
  offers a *candidate* label experiment, not a scored checkpoint. Confirm its
  labels, gold holdout, preprocessing, provenance, and permissions before use.
- **HPO label audit completed (CPU only):** `python tools/audit_hpo_labels.py`
  verifies 4 alternative banks and the Raptor bank each cover exactly 4,349
  non-gold IDs with finite 0–1 targets and zero gold overlap. All 4,407
  corpus IDs and fold entries match organizer studies; folds are Gold58=`-1`
  and five non-gold folds of 870/870/870/870/869. The 4,407-row
  `train_labels_v0` includes `gold_*` columns matching **all 696** organizer
  truths and an accurate `is_gold` flag; `train_labels_v1` has no flag but
  still includes all 58 gold rows. Its non-gold `y_*` exactly matches
  `labels_v1_gemlow`; 195/696 gold `y_*` values exactly match organizer
  targets, so gold-row provenance is not established. Exclude Gold58 by
  organizer UID, **not** by the presence of an `is_gold` column. Forty-five
  exact duplicate-report groups stay inside a single non-gold fold, but one
  non-gold report is identical to a Gold58 report. Patient-level group
  isolation is unverified (no patient ID in `train.csv`).
- **Downloaded trainer caveat:** `artifacts/hpo_label_audit/train_knee.py`
  also picks `best.pt` by Gold58 AUC and exports its OOF from that selected
  checkpoint; OOF `truth` is the arm's own soft label, so arm-to-arm scores
  are not directly comparable. `--oof_every` picks top-k epochs on an OOF
  subset; HPO's fold 0 subset was tuned by the original worker. Use fixed
  epochs or a separate selection split and a common reference before gating.
- **Validation correction:** `tools/train_raptor.py` excludes gold IDs from
  training, but evaluates on Gold58 repeatedly and selects the best epoch by
  Gold58 AUC (`gold_auc`). That introduces checkpoint-selection bias; Gold58 is
  not an independent final validation score after repeated epoch/model/blend
  searches. V33/V48 lineage also tuned on Gold58 (§10, §13). Do not infer a
  hidden-LB gain from gold-based weight/epoch optimization.

**Next experiment (not launched):** with label coverage audited, train one
12-target image-model control and candidate with identical checked study folds
(patient grouping only if a patient ID is available) and a predeclared fixed
epoch or separate non-gold model-selection split. Save study-level OOF
predictions; compare on one common arm-independent reference with per-target
AUC and uncertainty, then use Gold58 only once as a non-blind external sanity
check. Promote only a reproducible, multi-target improvement and test it as a
single change to V48 on a 2×T4 inference proxy with zero missing-arm
fallbacks before using one hidden rerun. See the dated plan in `PLAN_TOP50.md`.

---

## 15) FINAL WEEK (2026-10-14 → 10-22) — public ceiling leap, last-day forks

> Reconstructed from `artifacts/` receipts (this stretch had no HANDOFF entry until now).

- Public 0.958 kernels appeared in the final week (`goodpjw2008` CoAtNet-R6 2.5D MIL,
  `heliosli` 0.9586 dataset version). The public ceiling jumped 0.943 → 0.9586 —
  far past the old top-50 cutoff math in §14.
- `artifacts/r6_check*/manifest.json` — integrity checks of the pulled R6 asset
  bundle; `artifacts/remote_rsna-knee-v48-speedy-raptors/` — V48 receipts.
- **V49** (`kaggle/submission_v49_dinosaur_v5/`) — fork of `romantamrazov/rsna-knee-dinosaur-v5`
  (0.943 pipeline + guarded ConvNeXt residual). Kept as a hedge; did not beat V50/V51 publicly.
- **V50** (`kaggle/submission_v50_public_09586/`, live outputs in `artifacts/v50_live/`) —
  fork of the `heliosli` 0.9586 pipeline. Two-variant run: Raptor-only and Raptor+224 blend
  (`raptor_sub.csv`, `raptor224_sub.csv`).
- **V51** (`kaggle/submission_v51_public_0958/`, outputs in `artifacts/v51_out/`) —
  fallback fork of `goodpjw2008` v27 with per-class blend weights
  (blend receipt: raptor 0.5 / ours 0.5, Synovitis 0.8, PF OA 0.6, Medial OA 0.5,
  Lateral OA 0.2, Baker's 0.35; 3-study public cohort, zero failed arms).
- **Decision that mattered:** with hours left, V51 (public 0.958, smaller public/private
  drop risk) was selected over V48 (0.943, own fork lineage). The private LB proved
  this right: the 0.958-class forks fell ~0.016 while the 0.943-class fell ~0.016 too,
  and V51's blend hedges per class — final 0.942 vs V48's 0.927 (**+0.015, ~530 ranks**).
- `artifacts/lastday_scan/` holds the five candidate public kernels pulled on 2026-10-22
  (evgendvorkin, goodpjw2008, heliosli, lavinwins, prvsiyan) with full notebooks.

## 16) FINAL RESULT (private LB, 2026-10-22)

- **Rank 1450 / 6425 (top 22.6%), private AUC 0.942** — `Daniel solo` (team `16697840`),
  submission V51 `57516897` 2026-10-22 08:25 (source: `artifacts/lb_scan/lb_private.json`,
  full 6425-team board).
- Board: leader 0.960 · top-10 0.957 · top-50 0.951 · median 0.925.
  Our 0.942 sits in the board's largest tie (**1322 teams**); 0.943 would have meant
  rank ~1148 (305 teams above), 0.944 → ~939 (211 above).
- Top-50 goal missed by ~0.009 private AUC. All owned-model routes had been
  measured-closed before the deadline (§10–§14): V31 0.830 honest OOF, V38/V39
  distillation collapse, V42 −0.002, V43 0.879, V45 0.838, V46 ERROR,
  gold-guided fusion contaminated (Gold58 is a selection set, §10/§13).
- What the final submission actually was: a **public-kernel fork with per-class blend
  weights** — the honest ceiling of this repo's compute (single local 4070 + 30 h/wk
  Kaggle T4 quota) against teams training multi-model ensembles on rented GPUs.

## 17) POST-MORTEM — what worked, what didn't, what we'd do differently

**Worked**
- Fork-the-moving-ceiling strategy: V33 0.936 → V44 0.937 → V47 0.940 → V48 0.943 →
  V51 0.958 public. Each fork was verified by receipts (arm status, zero fallbacks)
  *before* spending a submit.
- Receipt-first verification discipline (`tools/build_*` + `watch_*`): every silent
  try/except degradation in public kernels was caught by receipts, never by scores.
- Measured-closure hygiene: every dead route (V31 primary, blend re-weighting, self-distil,
  LOO-distil, gold-guided fusion) is closed with a number and a receipt in this file —
  no zombie experiments came back.
- Final-day selection: picking V51 over V48 on shake-up-risk logic gained ~530 ranks.

**Didn't work**
- Owned models never got within striking distance (0.830–0.879 vs public 0.943+).
  Root causes: label banks were the bottleneck (§1 item 3), single-GPU training budget,
  and Gold58 too small/contaminated to tune against (§10, §14 correction).
- Top-50 arithmetic was computed against the *public* cutoff for months; the private
  cutoff (0.951) and the 0.958 forks made the plan obsolete in the last week.
- The compute-is-scarce assumption (§9–§10) was wrong for the push stage (§13
  correction): quota was never binding after 2026-09-23; recipe quality was.

**Do differently next time**
- Track the *private/public drop* of each candidate lineage early; select finals by
  shake-up risk, not just public score (this paid off here — keep it).
- If the ceiling is public forks, spend the saved GPU on *validation depth*
  (honest OOF references, per-target AUC with uncertainty) instead of owned models
  that cannot close a 0.06 gap.
- Audit label provenance before any training (the HPO audit in §14 took CPU-only time
  and killed a whole route before GPU was wasted).
- Enter the final week with a pre-built fork pipeline (we had one — it delivered
  V49→V51 in a day) and pre-committed selection criteria.

## 18) REPO ARCHIVE STATE (2026-10-25)

Competition closed → this repo is an archive. Cleanup performed 2026-10-25:

- **Deleted (~42 GB):** `artifacts/raptor_corpus/` (21 GB corpus), `artifacts/v31_cache/`
  (14 GB pixel cache), all `*.pt/*.pth/*.ckpt` weights everywhere (v31_runs,
  local_runs, dino_loo_runs, public_arms, v34_remote, remote_* train dirs,
  references/candidates), dataset staging dirs, dry-run/probe/error scratch dirs,
  `__pycache__`, `.firecrawl/`, `.workbuddy-ai/`.
- **Kept:** all receipts/manifests/logs-of-record, `lb_scan/` (incl. full private LB),
  `lastday_scan/` (5 final-week public kernels, full notebooks), `v50_live/` + `v51_out/`
  (final submission outputs + blend receipts), label banks (`v3/v32/v43_labels`,
  `pseudo_labels`, `raptor_labels` on disk), all `kaggle/` kernel sources, all `tools/`,
  `references/spy/` (public kernel source pulls).
- Everything above is committed and pushed to the private mirror
  `github.com/danielsolo707/rsna-knee-workspace`.
