# HANDOFF — start here (next AI / next session)

**MAJOR UPDATE: 2026-08-23 evening**

| Item | Value |
|---|---|
| NEW INCUMBENT | **Raptor solo 55722146 = 0.924** (ref 55722146), rank **169/2246** |
| Old soup | V27 = 0.914 (backup) |
| Fold-0 HARD | **0.8185** done locally (artifacts/v31_runs/fold0_hard) |
| Fold-1 HARD | running locally pid via WMI, best 0.8288 @ep14 of 24 |
| KEY DISCOVERY | dreaddevelopment/knee-mri-twelve-findings-from-a-single-model = public 0.924 solo CoAtNet-384 recipe + PUBLIC WEIGHTS dataset raptor-knee-widedense; our fork kernel rsna-knee-raptor-single-model |
| Recipe intel | 64 slices/study in 5 slots (SAG_FS18/SAG14/COR_FS12/COR8/AX12), coverage 6-94 percent (outer slices matter for MCL/LatMeniscus), 140mm crop, 336px, CoAtNet384, per-target attention head |
| Label verdict | hard labels beat soft by ~1pt (0.8185 vs 0.8088); Qwen7B labels FAILED twice (0.57 then 0.71 macro vs gold) - do not retry 7B; bank stays 0.8919 |
| Infra lessons | WMI Win32_Process.Create launches survive shell job-kills (Start-Process children get killed); single-writer .train.lock; left-padding required for batched Qwen generate; kaggle CLI output has no resume -> tools/download_v31_cache.py pattern |
| Kaggle quota | ~21h GPU remaining (refresh Fri) |
| Next | folds 2-3 local queue, fold-4 Kaggle, blend raptor+V27 members, adopt 6-94pct coverage into own cache v32 |

Older sections below are history.

# HANDOFF — start here (next AI / next session)

**Updated:** 2026-08-21 (evening) · supersedes all older sections below it where they conflict
**Competition:** `rsna-knee-abnormality-detection` (code competition, hidden test, 9 h notebook)
**Owner:** `danielsolo1770` · team **Daniel solo**
**Goal:** public **0.940**. Deadline **2026-10-22**.
**Strategy doc:** `docs/DEEP_RESEARCH_0940.md` (read together with this file).

---

## 0. Hard rules (unchanged)

- **Never push this project to GitHub.**
- Kaggle operations: **CLI only**.
- Do **not** unset `KAGGLE_API_TOKEN`. Token file `~\.kaggle\access_token`.
  - ⚠ NEW (kaggle>=2.2): if you set env var `KAGGLE_API_TOKEN`, its VALUE must be the
    token itself, NOT a path. `KaggleApi().authenticate()` finds the file by default —
    prefer that (this bug cost us one failed download run).
- Hermes `kaggle.exe` SSL-fails. Always use:
  `C:\Users\ASUS\AppData\Local\Programs\Python\Python312\python.exe -m kaggle ...`
- No daily submits on soups. No gold-58 tuning. Trainer kernels never write submission.csv.
- Secrets stay off-repo/off-chat. The Drive copy of `access_token` is the sanctioned
  way to bootstrap new Colab sessions.

## 1. Live state (2026-08-21 evening)

| Item | Value |
|---|---|
| Selected / incumbent | **V27 `55580781` → 0.914 public** |
| Rank | ~526 / 2097 (leader ~0.952; gold zone est. 0.975 public by end) |
| GPU quota Kaggle | exhausted → refreshes Fri 2026-08-22 00:00 UTC (then 30 h) |
| **V30 fold-0 full-FT probe** | **DONE: best 0.8201** (EMA @ep16 of cosine-30, LLRD 0.7, full unfreeze; early-stop ep23). Receipt+ckpt: Drive `MyDrive/rsna-knee-v31/fold0_fullft/` |
| **Scientific conclusion** | converged training on V30 geometry caps ≈ **0.82 soft-val** → remaining gap to 0.92 is **input geometry / labels**, not recipe or encoder size |
| **V31 cache (280px, 140mm crop, 21 slots)** | BUILT & COMPLETE on Kaggle (`rsna-knee-v31-cache-280`, n=4407/4407 ok, manifest verified). x_f16 = 14,511,369,600 B |
| V31 download LOCAL | resumable downloader running: `tools/download_v31_cache.py --status`; dest `artifacts/v31_cache/v31_cache/` |
| V31 on Colab | self-driving watcher installed: finishes dl → verifies → AUTO-launches fold-0 (`/content/watcher.log`, ckpt→Drive `fold0_v31/`) |

## 2. Lessons from the 2026-08-21 session (APPLY TO ALL FUTURE RUNS)

1. **Verify before train/download-trust**: byte-compare every artifact vs manifest
   (`x_f16.dat == prod(x_shape)*2`). This caught a missing `y.npy` AND a nested-dir
   path crash within minutes instead of hours.
2. **Everything background + status-file polling**: long jobs run as detached processes
   writing a status/log file; monitor by reading files, never by holding the cell.
3. **kaggle CLI output download has NO resume** and dies mid-file on big outputs.
   Use signed URLs via `list_kernel_session_output` + ranged chunk GETs
   (`tools/download_v31_cache.py` is the reference implementation).
4. **Checkpoints stream to persistent storage every epoch** (Drive locally-mounted dir);
   any crash costs ≤1 epoch.
5. **EMA + cosine-to-convergence works**: raw peaks then decays; EMA holds the peak.
   Expect peak around 55–70% of schedule; patience 6–7 early-stops cleanly.
6. One lever per experiment; log config hash + epoch curve + receipt into artifacts.

## 3. NEXT: local V31 fold-0 training (the decisive experiment)

When local download hits 100 % and verify says ALL GREEN:

```powershell
$py = "C:\Users\ASUS\AppData\Local\Programs\Python\Python312\python.exe"
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # if not yet
& $py tools\train_local_v31.py    # wrapper: sets cfg for local paths, launches bg, logs to artifacts\
```

Local trainer contract (from lessons):
- `cache_dir=artifacts/v31_cache/v31_cache`, `out_dir=artifacts/v31_runs/fold0_v31`
- **batch 4 / accum 4** (8 GB VRAM; V31 = 21 slots × 280px ≈ 2.2× activation of V30)
- If OOM: enable grad-checkpointing on encoder before shrinking further
- `num_workers=8` (i7-13650HX, 20 threads), `persistent_workers=True`
- env `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- epochs 24, warmup 1, patience 6, EMA 0.999, TTA on val — same as probe otherwise
- Gate (unchanged): fold-0 soft-val **≥ 0.85 → clone folds 1–4**; 0.83–0.85 → iterate
  geometry once more; < 0.83 → labels are the binding constraint, pivot to label work.

## 4. Scored ledger (unchanged today — no submissions spent)

| Ref | Name | Public | Verdict |
|---|---|---:|---|
| `55580781` | V27 blend | **0.914** | INCUMBENT |
| `55567630` | V25 | 0.913 | backup |
| `55575435` | V24 dinosaur v2 | 0.913 | 2nd family |
| older rows | … | ≤0.912 | see git history of this file |

Soup ceiling re-confirmed 0.911–0.914. **Never another soup submit.**

## 5. Closed paths (do not reopen)

V41/E10 family · DINOv3 w>0.20 · EfficientNet-B3 · ResNet18 students · Tony standalone ·
gold-58 promotion · DICOM-metadata features · bigger-encoder-as-lever (proven null) ·
undertrained probes as evidence · multi-account Colab · submitting from off-Kaggle.

## 6. Code map (new items)

| Path | Role |
|---|---|
| `docs/DEEP_RESEARCH_0940.md` | forum evidence + strategy + gates |
| `tools/build_v31_cache.py` | generates the V31 CPU cache kernel |
| `kaggle/cache_v31_280/` | pushed kernel source (COMPLETE run on Kaggle) |
| `tools/download_v31_cache.py` | **resumable** signed-URL downloader (--status flag) |
| `modal_v28/train_core.py` | old partial-FT trainer (superseded by v31probe) |
| Colab `/content/v31probe.py` (+Drive copy) | full-unfreeze LLRD+EMA trainer used for both probes |

## 7. Resume commands

```powershell
cd "C:\Users\ASUS\Desktop\DanielEmpire\ML\RSNA Knee Abnormality Detection"
$py = "C:\Users\ASUS\AppData\Local\Programs\Python\Python312\python.exe"
& $py tools\download_v31_cache.py --status        # download progress
& $py -m kaggle quota                              # GPU quota
& $py -m kaggle competitions submissions -c rsna-knee-abnormality-detection --csv
& $py -m kaggle kernels status danielsolo1770/rsna-knee-v31-cache-280
```

Colab (if tab still open): check `/content/watcher.log` and
`MyDrive/rsna-knee-v31/fold0_v31/receipt.json`.

## 8. One-sentence resume

**Keep V27 0.914 selected. V30-converged=0.8201 proves recipe isn't the blocker;
V31 (280px/140mm/21-slot) is downloading locally (~53%) — when green, train local
fold-0 (batch4/accum4); gate ≥0.85 → 5-fold; else pivot to labels. End of every
session: push artifacts to private dataset `rsna-knee-v31-workspace`.**

---

*(Historical notes from 2026-08-17..19 — V28/V30 thin-mid caches, Modal blockage,
Colab bootstrap details — remain valid as history; see `git`-less copies in docs/
(`HANDOFF_V28.md`, `CONTINUE_NEW_COLAB.md`, `TARGET_0940.md`) and the strategy doc.*
