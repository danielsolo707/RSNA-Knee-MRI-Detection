# Handoff — RSNA Knee, target 0.930 / 0.940 (for the next AI)

> **STALE (2026-08-17).** Current start-here is **`docs/HANDOFF.md`** (2026-10-25). Incumbent is V27 **0.914**. Do not use the live table below.

**Written:** 2026-08-17  
**Why this file exists:** weekly GPU may run out (`refresh 2026-08-22T00:00:00Z`). Do **not** restart from V1. Read **`docs/HANDOFF.md` first**, then `docs/archive/TARGET_0940.md`, then `docs/archive/SESSION_HANDOFF.md` only for history.

**Owner Kaggle:** `danielsolo1770`  
**Team:** Daniel solo (`16697840`)  
**Competition:** `rsna-knee-abnormality-detection`  
**Local repo:** `C:\Users\ASUS\Desktop\DanielEmpire\ML\RSNA Knee Abnormality Detection`  
**Hard rules:** Kaggle **CLI only**. **Never push this project to GitHub.** Do not use Kaggle MCP if CLI works. Do not unset `KAGGLE_API_TOKEN` if CLI then fails auth (token file is `~/.kaggle/access_token`).

---

## 1. Live state (update this table when you score something)

| Item | Value as of 2026-08-17 ~15:00 UTC |
|---|---|
| **Selected / incumbent** | **V27 0.914** — ref `55580781` (V25 0.913 is backup) |
| Rank / teams | **274 / 1790** (moves daily) |
| Top-50 cutoff | **0.927** |
| Top-10 cutoff | **~0.940** |
| Leader | **0.951** |
| Deadline | **2026-10-22** |
| Daily submits | **4 used / 1 remaining** (resets midnight UTC) |
| GPU | **20.45 h used / 9.55 h left** — refresh **2026-08-22T00:00:00Z** |
| V27 submit | **`55580781` COMPLETE 0.914** — new incumbent |
| V28 trainer | **RUNNING** `danielsolo1770/rsna-knee-v28-dino-ft-fold0` v1 — **no competition submit** |

### Scored ledger (do not repeat)

| Ref | Name | Public | Verdict |
|---|---|---:|---|
| `55567630` | **V25** V16 + Mattia DINOv3 w=0.20 | **0.913** | **INCUMBENT — keep selected** |
| `55575435` | V24 DINOsaur v2 | 0.913 | Tie; diverse 2nd family; do not replace V25 |
| `55580781` | V27 0.80 V25 + 0.20 V24 | PENDING | Wait; if ≤0.913 keep V25; if >0.913 new incumbent |
| `55570567` | V26 DINOv3 w=0.30 | 0.912 | Reject; **never raise DINOv3 above 0.20** |
| `55564553` | V23 V41+E10 0.70+E11 | 0.912 | V41 family **CLOSED** |
| `55564382` / `55562219` | V22 V41+E10 0.60 | 0.911 | CLOSED |
| `55491586` | V16 V10+E11 w=0.20 | 0.912 | Old floor under V25 |
| `55466431` | V13 3-arm | 0.899 | Gold +0.022 **lied**; never gold-tune |
| `55416721` | V5 DINO+B3 | 0.893 | B3 **CLOSED** |
| `55400073` | Tony exact | 0.828 | Public 0.905 did **not** reproduce |

---

## 2. Research answer: can 0.930 happen? (independent of V27)

**Yes. Chance from research, not from our pending file: 5 / 10.**

Evidence (finished contests + this board’s own forum, not our PENDING submit):

- **This contest, other teams:** Tucker Arrants **single model 0.934 @ 224px, 5-fold**. Tim Krige **~0.92** single model, OOF ≈ LB. Tom Aindow **0.915** single DINOv2. So **0.930 is already on this leaderboard** as a *trained* 5-fold, not as a public soup.
- **Stanford MRNet (knee MRI, done):** plane-wise 2.5D + fuse → ACL **0.96**, meniscus **0.85**, later papers **~0.93** val. Knee MRI *can* sit at 0.93 if you train.
- **RSNA Kaggle gold medals (lumbar 2024, abdomen 2023, breast 2022, c-spine 2022):** winners **train 5-fold**, localize then classify, attention MIL + aux, small/medium backbones. They do **not** win by remixing 20 public checkpoints.
- **Our measured soup ceiling:** V22–V26 = **0.911–0.913**. That path **cannot** add +0.017.

So: **research says 0.930 is possible. Our current *file* does not get us there. Only an owned trained 5-fold (V28→V29) can.**

If GPU dies this week: do **not** spend the last daily submit on another V41/E10 fork. Wait for quota 2026-08-22 and continue V28.

---

## 3. What is running / what to do next

### A. When V27 `55580781` scores

```bash
kaggle competitions submissions -c rsna-knee-abnormality-detection --csv
```

- Score **> 0.913** → new incumbent. Update this table. Keep V25 as backup selected if Kaggle allows 2.
- Score **= 0.913** → tie; **keep V25**.
- Score **< 0.913** → reject; **keep V25**.
- **Do not retune** the 0.80/0.20 weights on the LB.

### B. When V28 fold-0 finishes

```bash
kaggle kernels status danielsolo1770/rsna-knee-v28-dino-ft-fold0
kaggle kernels output danielsolo1770/rsna-knee-v28-dino-ft-fold0 -p artifacts/v28_remote/fold0 -o
```

Read `v28_fold0/receipt.json` (`best_val_auc`) and `v28_fold0/best.pt`.

| `best_val_auc` (soft val, not gold-58) | Next |
|---|---|
| **NaN / crash / no `best.pt`** | Read the `.log`. Fix decode/DINOv2 path. Do **not** submit. |
| **< 0.75** | Pipeline/pixels/labels wrong. Debug images. Do **not** train folds 1–4. |
| **0.75–0.83** | Weak but maybe salvageable. One more fold-0 tweak (lr / unfreeze 6 / more epochs), then decide. |
| **≥ 0.83** | Clone trainer to folds 1–4 (`FOLD = 1..4`). Same recipe. No LB submit until 5-fold OOF exists. |

**V28 must never write `submission.csv`.** If it does, delete it. This kernel is **train only**.

### C. After 5-fold OOF exists (V29 infer)

Build a **new** kernel = V25 (or V27 if it won) in-kernel **plus** 5-fold V28 heads. Gate vs parent:

- Soft OOF not worse than parent by > 0.002, or clearly stronger on most targets
- Mean rank-corr vs parent **< 0.95**
- No target drop > 0.02
- Fail-closed to V25/V27 file

Do **not** promote on 58-gold gain alone (V13 lesson).

---

## 4. Code map (do not reinvent)

| Path | Role |
|---|---|
| `docs/HANDOFF_V28.md` | **This file — start here** |
| `docs/TARGET_0940.md` | Locked 0.940 program |
| `docs/SESSION_HANDOFF.md` | Long history (V1–V26); stale in places |
| `kaggle/submission_v25_v16_dinov3/` | Incumbent 0.913 recipe |
| `kaggle/submission_v24_dinosaur_v3/` | 0.913 dinosaur (ROOT patched v2) |
| `kaggle/submission_v27_v25_v24_blend/` | V27 blender; dry-run `V27_BLENDED` |
| `kaggle/train_v28_dino_ft/` | Fold-0 trainer + `kernel-metadata.json` |
| `tools/build_v27.py` | Rebuild V27 |
| `tools/build_v28_train.py` | Rebuild V28 (source of truth for trainer) |
| `research/prior_rsna_writeups/` | Pulled RSNA gold threads + `REPORT.md` |
| `artifacts/v27_remote/kernel_output/v27_blend_audit.json` | `status: V27_BLENDED` |

### V28 recipe (frozen unless fold-0 proves it broken)

- DINOv2-**small** from `metaresearch/dinov2/PyTorch/small/1`
- Unfreeze **last 4** blocks
- **224px**, crop **160 mm**, **3 adjacent** slices as RGB, **3 FS planes**
- Labels: mean of `report_labels_v2.csv` + `llm_labels_v2.csv` + `labels_llm_gpt56sol.csv`; official gold **overwrites** those rows but is **not** used to pick hparams
- GroupKFold(5) on `md5(uid)[:8]` groups
- Loss: BCE + **0.25 aux** (RSNA 2024 Avengers)
- Head: **per-target attention MIL** over slots
- `ALIVE = 3.8h`; `EPOCHS = 6`; `FOLD = 0`; `SMOKE = False`
- Push cap used: `-t 15000`

If quota is almost gone, **do not push another GPU kernel**. Leave V28 running or wait until 2026-08-22.

---

## 5. Closed paths (do not reopen)

- V41 / E10 0.50–0.70 / “0.93” public notebooks  
- DINOv3 blend weight **> 0.20**  
- EfficientNet-B3 family  
- ResNet18 students / V15  
- Tony standalone  
- Gold-58 promotion / V13-style 3-arm retune  
- Importing **other RSNA contests’ patient DICOMs** (wrong body, MIRA/license)  
- Real cheating (test leak, probing hidden labels)

Legal “cheat code” we already stole from gold writeups: **train 5-fold + MIL + aux + 224px**, then blend **diverse** families.

---

## 6. Resume commands

```bash
cd "C:/Users/ASUS/Desktop/DanielEmpire/ML/RSNA Knee Abnormality Detection"

kaggle competitions submissions -c rsna-knee-abnormality-detection --csv
kaggle competitions submission-limits -c rsna-knee-abnormality-detection
kaggle quota
kaggle kernels status danielsolo1770/rsna-knee-v28-dino-ft-fold0
kaggle kernels status danielsolo1770/rsna-knee-v27-v25-v24-blend
```

If V28 is still RUNNING and GPU remaining < 1 h, **do not push**. Let it finish or die; download logs either way.

---

## 7. One-sentence resume

**Keep V25 0.913 selected until 55580781 scores; V28 fold-0 is the only path to 0.930; if weekly GPU is empty, wait until 2026-08-22, download `receipt.json`, and only then clone folds 1–4 — never another public soup.**
