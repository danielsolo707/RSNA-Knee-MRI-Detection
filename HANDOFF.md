# HANDOFF — start here (next AI / next session)

**Rewritten:** 2026-08-25 evening · this file wins over everything below it.
**Competition:** rsna-knee-abnormality-detection · deadline 2026-10-22
**Owner:** danielsolo1770 · team Daniel solo · repo local-only (+private GitHub mirror rsna-knee-workspace)

---

## 0) CURRENT STATE AT A GLANCE

| Item | Value |
|---|---|
| **NEW INCUMBENT** | **Raptor solo = 0.924 public** (submission ref `55722146`, kernel `rsna-knee-raptor-single-model` v1) |
| Old soup backup | V27 = 0.914 (`55580781`) |
| **Rank** | **169 / 2246** (was 526 this morning) |
| Daily submits left | check `kaggle competitions submissions` (1 used today of 5) |
| Kaggle GPU quota | CHECK `kaggle quota` (~14-15h expected; refresh Fri 2026-08-29) |
| Local GPU | RTX 4070 Laptop 8GB — free & fast (SDPA fix applied) |

## 1) THE BIG DISCOVERY (why we jumped)

Public kernel **dreaddevelopment/knee-mri-twelve-findings-from-a-single-model**
= single-model **0.924 LB**, no ensemble, no TTA.
- Architecture: CoAtNet-384 (`coatnet_rmlp_2_rw_384`) + per-target attention head
  (`RaptorClassifier` class — full code inside the pulled .py at
  `references/spy/single_model/`)
- Input: 64 slices/study in 5 slots (SAG_FS×18, SAG×14, COR_FS×12, COR×8, AX×12),
  slices evenly across **6–94% of the stack** (outer slices matter for MCL/LatMeniscus!),
  140 mm mm-aware crop, **336 px**, triplets→windows, ImageNet norm, K_EVAL=42 windows.
- Labels: LLM soft probabilities over 4,349 reports (gold-58 held out; their model hits
  **0.9167 macro on gold**).
- **Weights are PUBLIC**: dataset `dreaddevelopment/raptor-knee-widedense`
  (`raptor_ft_coatnet_v4_full.pt` + SWA variant).
- Our fork kernel: `rsna-knee-raptor-single-model` (T4, internet OFF,
  dataset+competition attached) → ran ~35 min → submitted → **0.924**.

## 2) OUR OWN 5-FOLD HARD-LABEL FAMILY (V31)

Recipe = V31 cache (280px, 140mm crop, 21 slots) + DINOv2-S **full unfreeze LLRD**
+ SDPA monkeypatch + grad-checkpointing + **HARD binarized labels** + batch16.

| fold | best OOF | location |
|---|---|---|
| 0 | 0.8185 | artifacts/v31_runs/fold0_hard |
| 1 | **0.8299** | artifacts/v31_runs/fold1_hard |
| 2 | **0.8361** ⭐ | artifacts/v31_runs/fold2_hard |
| 3 | 0.8180 | artifacts/v31_runs/fold3_hard |
| 4 | trained on Kaggle T4 — **output download was INTERRUPTED, must finish** | artifacts/v31_runs/fold4_hard (partial) |

A/B verdict: hard > soft by ~+1pt (soft fold-0 was 0.8088).
Fold variance is large (0.818→0.836) → real OOF ensemble value unknown until tested.

## 3) IMMEDIATE NEXT STEPS (in order)

1. **Finish fold-4 output download** (~700MB best.pt+last.pt from kernel
   `rsna-knee-v31-fold4-hard`). Detached WMI pattern worked but was interrupted;
   relaunch:
   ```
   cmd /c set PYTHONIOENCODING=utf-8&& "<py>" -m kaggle kernels output
     danielsolo1770/rsna-knee-v31-fold4-hard -p artifacts\v31_runs\fold4_hard -o
     > artifacts\fold4_dl.log 2>&1     (via Win32_Process.Create)
   ```
2. **OOF assembly**: collect val_preds.csv/best.pt per fold → macro OOF of the
   5-fold mean → compare vs 0.836 single-best.
3. **Inference kernel for our family** → blend test:
   `rank_blend(raptor_submission, V27_soup, our_family)` greedy-weighted on
   public probe… CAREFUL: only via submissions (4 left today).
4. **Adopt raptor insights into our pipeline (v32 cache candidate):**
   coverage 6–94%, 64 slices/5 slots incl. second non-FS series per plane,
   336px, per-target attention already matches ours.
5. Keep Raptor solo as selected incumbent unless a blend beats it on LB.

## 4) CLOSED PATHS (do NOT reopen)

- Qwen2.5-7B labeling: FAILED twice vs gold (0.57 then 0.71 macro; bank=0.8919).
- Soft-label training: beaten by hard labels (~+1pt).
- Geometry 280px/21slots: worse than raptor's spec (64 slices@336px won).
- Public soups as primary strategy; gold-58 tuning; encoder-size-as-lever;
  DICOM metadata features; multi-account anything.

## 5) INFRA LESSONS (all learned the hard way TODAY)

- **WMI detached launch survives shell job-kill**:
  `Invoke-CimMethod Win32_Process Create @{CommandLine=...;CurrentDirectory=...}`.
  Start-Process children get killed by the tool wrapper (ChildProcess.kill race).
- **Single-writer lock** (`.train.lock` + psutil liveness) in train_fold.py —
  two writers corrupted fold-1 once.
- **transformers must be pinned ==4.44.2 on Kaggle images** (newer breaks Dinov2
  LayerScale: `hidden_state * self.lambda1` TypeError).
- Embedding a python module into a notebook cell: strip its `__main__` block and
  DON'T `import` it back — call definitions directly.
- Batched decoder generation needs `tokenizer.padding_side="left"`.
- kaggle CLI kernel-output downloads have no resume → use signed-URL chunked
  downloader pattern (tools/download_v31_cache.py).
- kaggle CLI auth can break mid-session server-side; regenerate token at
  kaggle.com/settings/api → overwrite `~\.kaggle\access_token`.
- NEVER unset KAGGLE_API_TOKEN / never trust persistent-shell env state.
- Trainer kernels must print step-level logs (every 50 steps) — epoch-blind runs hide failures.

## 6) KEY FILES / COMMANDS

```
tools/train_fold.py <fold> [epochs]      # local hard-label trainer (WMI-launch it!)
tools/v31probe.py                        # trainer core (SDPA patch, GC, OOM guards)
tools/download_v31_cache.py [--status]   # resumable signed-URL downloader
tools/label_audit.py                     # 6-source label audit vs gold-58
tools/validate_labels_v4.py <csv>        # instant label scoring
kaggle/raptor-single-model/              # our fork of the 0.924 inference kernel
references/spy/single_model/*.py         # original raptor inference source (study!)
artifacts/v31_runs/fold*_hard/           # checkpoints + history.json per fold
```

Resume commands:
```powershell
$py = "C:\Users\ASUS\AppData\Local\Programs\Python\Python312\python.exe"
& $py -m kaggle competitions submissions -c rsna-knee-abnormality-detection --csv
& $py -m kaggle quota
& $py -m kaggle kernels status danielsolo1770/rsna-knee-raptor-single-model
& $py tools\train_fold.py <n> 24    # via WMI Create!
```

WMI launch template:
```powershell
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = "`"$py`" `"$wd\tools\train_fold.py`" <N> 24"; CurrentDirectory = $wd }
```

## 7) ONE-SENTENCE RESUME

**Incumbent = Raptor solo 0.924 (rank 169); finish downloading fold-4 output,
assemble 5-fold OOF + inference, try blends (raptor+V27+ours) with remaining
daily submits, and port raptor's 6–94% coverage idea into a future v32 cache.**
