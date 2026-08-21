# Session handoff — RSNA Knee Abnormality Detection

**Last updated:** 2026-08-17 evening — **Next AI: read `docs/HANDOFF_V28.md` first.** V25 0.913 incumbent. V27 `55580781` PENDING. V28 fold-0 RUNNING.
**Owner Kaggle:** `danielsolo1770`  
**Team:** Daniel solo (`16697840`)  
**Competition:** `rsna-knee-abnormality-detection`  
**Local-only repo:** `C:\Users\ASUS\Desktop\DanielEmpire\ML\RSNA Knee Abnormality Detection`  
**Standing rule:** Kaggle/competition work stays local. **Never push this project to GitHub.**

---

## 1. Resume here — current live state

**New incumbent: V25 at 0.913** (ref `55567630`) — V16 parent + DINOv3 w=0.20.
Rank **218 / 1746**. Top-50 still **0.926** (gap **-0.013**). Leader 0.950.

**Overnight 2026-08-16/17:** private fork of public V41
(`DINOv2 20 + Tony 4 + DINOv3 ViT-S/16 + E10 uniform_060`) is the active
challenger. Use **Kaggle CLI only**.

| Item | Current value |
|---|---|
| Verified best public score | **0.913** — V25 (ref `55567630`) |
| Current public rank | **274 / 1790** |
| Live Top-50 cutoff | **0.927** |
| Gap from best | **-0.014** to Top-50 |
| Live Top-10 cutoff | **0.939** |
| Leader | **0.950** (Brandon Low) |
| Active challenger | **V22 v2** `danielsolo1770/rsna-knee-v22-dinov3-e10-060` version 2 |
| V22 v1 submit | ref `55562219` PENDING — 3600s cap, likely too short; ignore |
| V22 v2 submit | ref **`55564382`** PENDING — 32000s cap, dry-run `CANDIDATE_SELECTED` |
| V23 dry-run | **COMPLETE / ready** — E10 `uniform_070` + **E11_APPLIED** α=0.20; do not submit until V22 v2 scores |
| Daily submissions | 3 used / 2 remaining |
| V22 v1 score | **0.911** — reject; below V16 0.912 |
| V22 v2 score | **0.911** — same recipe confirmed; V22 family CLOSED |
| V23 score | **0.912** — tie with V16; do not prefer over incumbent; V41/E10/E11 family CLOSED |
| V25 score | **0.913** NEW INCUMBENT (ref `55567630`) |
| V26 score | **0.912** — reject; w=0.30 too much DINOv3; stop weight search |
| V24 score | **0.913** — tie with V25; do not prefer over V25; useful as a diverse second 0.913 family |
| V25 dry-run | **V25_BLENDED** (V16 + DINOv3 w=0.20) |
| Daily submissions | reset; 1 used today on V25, 4 remaining |
| GPU remaining | **13.13 h** (refresh 2026-08-22) |
| Deadline | 2026-10-22 |

V22 dry-run v1 already passed: `CANDIDATE_SELECTED` / `VALID_DINOV3_E10_HYBRID_ALPHA060`.
Next staged kernels (do not push until V22 v2 dry-run is done):
- V23: `kaggle/submission_v23_dinov3_e10_e11` (public V43b, E10 0.70 + E11)
- V24: `kaggle/submission_v24_dinosaur_v3` (DINOsaur V3)

> V13 did not improve the incumbent: `0.899` vs `0.906` (`-0.007`). Top 50 has
> not been achieved.

### Monitor completed

`tools/monitor_v13_submit.py` resumed submission `55466431` without a duplicate,
read the completed score through Kaggle CLI, pulled the leaderboard and wrote
the final receipt to `artifacts/v13_remote/monitor_state.json`:

- Stage: `complete`
- V13 public score: `0.899`
- Account best: `0.912` (V16 v3, 2026-08-13)
- Team rank: `99 / 1300`
- Top-50 cutoff: `0.915`

The two UI entries labelled `2 Subagents` are historical completed results. Both
sub-agents finished successfully; there are no live child workers.

---

## 2. Immediate resume commands

Hermes terminal runs Bash/MSYS on Windows. Use Bash syntax and native Windows
paths for Windows executables.

```bash
cd 'C:/Users/ASUS/Desktop/DanielEmpire/ML/RSNA Knee Abnormality Detection'
unset PYTHONPATH KAGGLE_API_TOKEN

# Current V13 score/status
kaggle competitions submissions \
  -c rsna-knee-abnormality-detection --csv | sed -n '1,5p'

# Read monitor receipt
python -c "import json; print(json.dumps(json.load(open('artifacts/v13_remote/monitor_state.json')), indent=2))"
```

Kaggle operations and retrieval must use **Kaggle CLI only**. Do not use a
Kaggle MCP or browser/CDP for Kaggle data. The current CLI uses the recommended
API access token in `~/.kaggle/access_token`; do not force the token into
`kaggle.json`, which is the separate legacy username/key format.

The monitor is complete. It can be rerun idempotently to reproduce the receipt,
but it must not create another submission:

```bash
unset PYTHONPATH
export V13_KERNEL_VERSION=2
C:/Users/ASUS/AppData/Local/Programs/Python/Python312/python.exe \
  tools/monitor_v13_submit.py
```

The monitor resumes submission ref `55466431` from the state file when the
saved version/message match, so rerunning it does **not** create a duplicate.

---

## 3. V13 candidate — frozen recipe

### Kernel and submission

- Kernel: `danielsolo1770/rsna-knee-v13-three-arm-gated`
- Kaggle URL: https://www.kaggle.com/code/danielsolo1770/rsna-knee-v13-three-arm-gated
- Successful dry-run kernel version: **2**
- Competition submission ref: **`55466431`**
- Description: `V13 gated DINO+E9 Rad-FS+E11 Rad-NOFS fixed 0.30/0.35/0.35`

### Fixed three-arm recipe

For ten targets:

```text
0.30 × DINO rank + 0.35 × E9 RadImageNet-FS rank + 0.35 × E11 RadImageNet-NOFS rank
```

Preserve pure DINO rank for:

- `Baker's`
- `Fracture`

This recipe is pre-committed from the E11 external contract. Do not tune weights
again on the same 58 gold studies after seeing the V13 leaderboard result.

### Diversity

- **DINO parent:** verified 20-member public baseline family.
- **E9:** RadImageNet ResNet-50 encoder, 5-fold query heads, three fat-suppressed
  slots (`SAG_FS`, `COR_FS`, `AX_FS`).
- **E11:** independently trained 5-fold query heads, three non-fat-suppressed
  planes plus a sagittal FS anchor (`SAG_NOFS`, `COR_NOFS`, `AX_NOFS`, `SAG_FS`),
  physical crop 130 mm.

E11 adds real pixel/contrast diversity, not only another head on identical
features.

---

## 4. V13 evidence and gates

### Independently reproduced local OOF

All IDs, target order, folds and hashes were independently checked by a second
worker.

| Metric | V9 parent | V13 | Delta |
|---|---:|---:|---:|
| Gold macro ROC-AUC (58 studies) | 0.8515822091 | **0.8737902881** | **+0.0222080789** |
| Soft OOF macro | 0.8533634561 | 0.8523522425 | -0.0010112136 |

Grouped-fold V13 gains:

| Fold | Rows | Gain |
|---:|---:|---:|
| 0 | 10 | +0.0739616402 |
| 1 | 15 | +0.0400132275 |
| 2 | 16 | +0.0125476665 |
| 3 | 9 | +0.0224867725 |
| 4 | 8 | +0.0112103175 |

- Positive folds: **5/5**
- Local paired bootstrap, 10,000 draws: **P(V13 > V9) = 0.9983**
- Mean bootstrap delta: `+0.0221272`
- 95% bootstrap interval: `[+0.0077183, +0.0366543]`
- Soft catastrophe floor: passed (`-0.00101 > -0.005`)

### Successful Kaggle dry-run receipt (kernel v2)

- Status: **`V13_SELECTED`**
- Promote: `true`
- Runtime: `643.70 s` (~10.7 min for the visible 3-study test)
- Remote bootstrap, 5,000 draws: `0.9994`
- Gold gain: `+0.0222080789`
- Positive folds: `5/5`
- Selected submission SHA-256:
  `56e2a02a906004d94ab73f6a2d7e1d7c6039d397e9c7b75a9c4cacc12d70dbfc`

### Evidence boundary

The naive transfer forecast was:

```text
0.906 + 0.0222 ≈ 0.9282
```

A conservative expectation was centered near `0.925`, but the real score was
**`0.899`**. This falsified the transfer forecast: the 58-study gold improvement
did not generalize to the public leaderboard.

---

## 5. Safety and runtime fixes completed

V13 version 1 was a dry-run only. It safely failed before submission because a
small-test slot-fill guard required 90%, while the 3-study visible test had 8/9
usable E9 slots = 88.9%. No competition submission was sent from version 1.

Version 2 includes these fixes and safeguards:

1. E9 visible-test fill threshold changed from 90% to 80%.
2. Every study must still have at least one usable slot.
3. V9 is materialized before the diagnostic V13 gate, so later failures retain
   the scored incumbent.
4. Removed the dangerous emergency all-`0.5` submission. If the DINO parent
   fails, the notebook refuses to emit a chance submission.
5. Added SHA-256 authentication for DINO/E9/E11 OOFs, train CSV and E11 contract.
6. Added target-order/schema validation for DINO, E9 and E11 artifacts.
7. Added explicit validation that the E11 contract matches the frozen recipe.
8. Added the soft-OOF catastrophe guard to the independent gate.
9. Added deadline checks inside encoder batches, every head fold, head batches,
   after inference and immediately before final V13 selection/write.
10. End-to-end runtime cap reduced from 4.8 h to **4.5 h** to preserve GPU quota
    headroom after two dry-runs.
11. On V13 failure after V9 exists, restore V9; do not submit a speculative file.
12. Tampered-artifact rejection, deadline failure path, state-dict load/forward,
    Python compile and nbformat validation all passed locally.

---

## 6. Artifact authentication

Expected source SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| DINO OOF `oof.npz` | `62d47ba4c0c8347b5b24e7fd2aa517aae0fd6d4656fd5d829fd3a50f0159909c` |
| E9 OOF `v52_oof.csv` | `f5b1f02029af98740565bec5b6a76d8471b4b6a6039d09ea410368230cb1f03f` |
| E11 OOF `v52_e11_oof.csv` | `f5415485f57103fe86370e3d4bba420485a7497ef199d40545a20f6e53a0f0e2` |
| Train CSV | `8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33` |
| E11 contract | `7f91e189ccc472b710791730bbbc2d26b3a1d04b45f60a0166445be158a093d3` |
| E9 heads | `0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa` |
| E11 heads | `df98518d0a8f7cc506b4b806da03c3fca18c8f8f743e813c5401759b45e0f1fe` |
| RadImageNet encoder | `08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734` |

---

## 7. Submission history and incumbent policy

| Ref | Candidate | Public | Decision |
|---|---|---:|---|
| **`55466431`** | **V13 DINO + E9 + E11** | **0.899** | Reject; `-0.007` vs incumbent |
| `55455223` | V10 ACL/Syn Rad cross | 0.906 | Tie; do not prefer over V9 |
| **`55442177`** | **V9 DINO + E9 α=0.20** | **0.906** | Verified incumbent |
| `55416721` | V5 DINO + B3 | 0.893 | Reject |
| `55391636` | V2 DINO + Tony | 0.899 | Reject |
| `55389600` | V1 DINO baseline | 0.899 | Old anchor |
| `55400073` | V3 Tony exact | 0.828 | Reject |

Current incumbent policy:

- Keep V9 (`55442177`) as the verified incumbent at `0.906`.
- Top 50 currently requires `0.915`; the gap is `+0.009`.
- Never select a score below 0.906.
- Do not spend another submission on a trivial reweight of the same three OOFs.

---

## 8. Key local files

```text
docs/SESSION_HANDOFF.md
    This file.

tools/build_v13_three_arm_notebook.py
    Generates the private V13 notebook and kernel metadata.

tools/gate_v13_three_arm.py
    Independent deterministic OOF/hash/contract gate.

tools/monitor_v13_submit.py
    Downloads and verifies dry-run outputs, submits only on V13_SELECTED,
    then monitors the competition score and leaderboard.

kaggle/submission_v13_three_arm/
    rsna-knee-v13-three-arm-gated.ipynb
    kernel-metadata.json

artifacts/v13_three_arm_gate.json
    Local 10,000-bootstrap receipt.

artifacts/v13_remote/monitor_state.json
    Current monitor/submission state.

artifacts/v13_remote/monitor.log
    Append-only monitor log.

artifacts/v13_remote/kernel_output/
    Downloaded remote dry-run outputs and audit.

artifacts/v13_remote/kernel_output/v13_three_arm_audit.json
    Verified Kaggle v2 dry-run receipt.

references/candidates/antoine_e9_files/
    E9 heads and OOF.

references/candidates/antoine_e11_files/
    E11 heads, OOF and external contract.

research/pilkwang_oof/oof.npz
    DINO OOF bundle.
```

---

## 9. Closed paths — do not repeat

- V2 DINO + Tony blend: no improvement.
- V3 Tony exact: large regression.
- V5/V6 B3 family: regression / weak soft OOF.
- V7 meniscus branch: gate failure.
- V8 Salem/import path: clone/path problems.
- V10 tiny targetwise cross: same 0.906 as V9.
- V11 own Rad heads: soft OOF too weak; rejected without submit.
- V12 targetwise own/e9 mix: only +0.0014 nested; rejected without submit.
- V13 three-arm DINO/E9/E11: gold gate predicted improvement but public score
  regressed to `0.899`; close this family and do not retune it on the same gold.
- Do not lower gate thresholds merely to create another submission.
- Do not fit and report weights on the same 58 gold rows.
- Do not push competition artifacts or code to GitHub.

---

## 10. Next step after the V13 regression

1. Preserve V9 at `0.906` as the incumbent.
2. Treat the 58-study gold gate as overfit/low-coverage evidence for this family.
3. Do not retune V13 weights or submit another blend of the same three arms.
4. Before using another submission, require a genuinely new family plus stronger
   transfer evidence than the same 58 gold studies.
5. Retrieve all Kaggle status, leaderboard and quota information through Kaggle
   CLI only.

---

## 11. Resume in one sentence

**V13 ref `55466431` completed at 0.899, regressing `-0.007` despite its +0.02221 gold gate; reject V13, preserve V9 at 0.906, current rank is 99/1300 with Top-50 cutoff 0.915, and use Kaggle CLI only for all further Kaggle retrieval and operations.**

---

## 12. V15 status (2026-08-13, current session)

Protocol pre-registered in `docs/v15_student_5fold_plan.md`. Two kernels built, both tested end-to-end locally:

**Kernel A (CPU, bridge build)** � `tools/v15_bridge_runtime.py` + `tools/build_v15_bridge_notebook.py`:
- Rebuilds V55 bridge `[3, 16, 256, 256]` uint8 from DICOM for all 4,407 training studies (fluid-sensitive FS series per plane, physical order, central 16-slice, resize 256, per-volume uint8 min/max; recipe = `research/v14_top20_cli/resnet18_student/inference_runtime.py::read_volume`).
- Writes depth-selected `[3, 8, 256, 256]` chunks as raw `.npy` (`chunk_XXXX_volume.npy` + `chunk_XXXX_presence.npy`) + `v15_bridge_manifest.json` (per-file SHA256, study_start/study_stop, recipe "V55-bridge") + `receipt.json` + per-study receipt. ~6.9 GB raw.
- Tested locally on synthetic DICOM: full pipeline, manifest contract OK (3 studies, 3 planes, presence = [1,1,1]).

**Kernel B (T4, 5-fold OOF)** � `tools/v15_student_5fold_runtime.py` + `tools/build_v15_student_5fold_notebook.py`:
- Folds/gold from `v52_oof.csv` (fold column, is_gold 58). Gold rows excluded from every fit (source-held-out).
- Teacher targets: 24 cols [12 DINO OOF (`pilkwang/rsna-knee-weights` oof.npz, SHA-pinned) | 12 V52 rad OOF (SHA-pinned)].
- Soft labels: **V10-identical `_v10_make_targets`** (3 LLM CSVs: report_labels_v2 / llm_labels_v2 / labels_llm_gpt56sol; gold weight 3.0) � NOT the local dino+rad mean (that changes the scale; V10 reference soft_v9_macro = 0.85336).
- Student: ResNet18, 2 seeds x 12 epochs, exact training_source.py hyperparameters, hash-based 10% validation for checkpoint selection.
- OOF: fold-f model predicts fold-f rows. Deployment OOF = 0.80 rank(public head) + 0.20 rank(rad head).
- Gates G1-G4 computed in-kernel AND independently by `tools/gate_v15_student_5fold.py` (uses kernel-written `v15_soft_labels.csv`, SHA-checked vs audit).
- Writes `v15_oof.csv`, `v15_oof_24.csv`, `v15_soft_labels.csv`, `v15_audit.json`; refuses to overwrite existing `v15_oof.csv`; `ALIVE_LIMIT` 4.5 h; on error writes `V15_ERROR_NO_OOF` audit; **no in-kernel submission**.
- Tested locally end-to-end on 60 synthetic studies (CPU, 1 epoch): full 5-fold loop, gates computed, all 4 outputs written.

**Dataset sources for kernel B** (bridge must be created/published first): `danielsolo1770/rsna-knee-v15-bridge` (private, pending), `pilkwang/pilkwang-public-dataset-for-notebooks-figures`, `pilkwang/rsna-knee-weights`, `antoinegg1/rsna-knee-e9-radimagenet-heads-v15`, `pilkwang/rsna-knee-llm-labels`, `stevenleehans/rsna-knee-llm-report-labels`, `lixin73/rsna-knee-llm-report-labels-sol56`.

**Current SHAs**: bridge runtime `f18646fa�`, bridge notebook `2d6a04a8�`, 5fold runtime `5c49b832�`, 5fold notebook `be20c9bc�`.

**Next steps** (in order): (1) user go-ahead to spend GPU quota ? push + run kernel A (CPU) ? download bridge output; (2) publish as private dataset `danielsolo1770/rsna-knee-v15-bridge`; (3) push + run kernel B (T4, est <= 2.5 h) ? download outputs; (4) run gate verifier ? submit only if `submit_allowed`; rollback = V10 SHA `29a1ae14b5610d5056b9ce06d19442d7378a3915e31c67377d615b99477e4e29`. Budget: 4.43 h GPU quota left (refresh 2026-08-15).

---

## 13. V15 result (2026-08-13 evening, GATE BLOCKED - do not submit)

Full pipeline executed remotely as planned:

1. **Kernel A** (`danielsolo1770/rsna-knee-v15-bridge`, CPU, v2): built bridge for all **4,407** studies -> **69 chunks**, **13,221/13,221 plane volumes present (100%)**, 56 min. Fix: mount is nested one level deeper; added `_resolve_root` two-level scan (same as V14). manifest sha `d5ef8562b69e0923f8f51e0b5daeb6adc8b3065cbae4ef7e244f4acea76a4e05`.
2. **Kernel B** (`danielsolo1770/rsna-knee-v15-student-5fold`, T4, v1): 2 h 20 m, status `V15_5FOLD_OOF_COMPLETE`, outputs downloaded: `v15_oof.csv` (sha `46a39319�`), `v15_oof_24.csv` (`8ea60120�`), `v15_soft_labels.csv` (`362b873a�`), `v15_audit.json`.
   - Bridge mounted as **kernel_sources** (not dataset) - avoids slow 6.9 GB re-upload; kernel A output lives under `/kaggle/input/rsna-knee-v15-bridge/v15_bridge/`, found by recursive locate_file.
   - DINO oof.npz source identified: **`pilkwang/rsna-knee-weights`** (contains oof.npz + merge_gain.npz + m_*.pt); added to dataset_sources; sha verified `62d47ba4�` matches local.
3. **Gate (in-kernel and local, identical)**: `V15_GATE_BLOCKED`, `submit_allowed: false`:
   - G1 full-OOF soft macro: student 0.84343 vs parent 0.85713 (**-0.0137**) - FAIL
   - G2 positive folds 0/5 (all fold deltas -0.011 to -0.016) - FAIL
   - G3 collapse targets: MCL, Lateral Meniscus, Contusion - FAIL
   - G4 rank corr 0.907 - PASS
   - gold58: student 0.87013 > parent 0.85445, blend 0.87082 (gold-only, not trusted for decisions)

**Decision: no submission. Keep V10 (best 0.906).** Per protocol: do not lower thresholds, do not retune same family on gold.

Artifacts: `artifacts/v15_student_5fold_gate/` (kernel_v15_audit.json, v15_gate.json, v15_oof.csv, v15_oof_24.csv, v15_soft_labels.csv). Local per-target AUC re-derivation matches kernel evidence (student 0.8430 macro).

**Remaining**: GPU quota spent (2h20m of 4.43h used). Next family needs genuinely new transfer evidence; V13-style gold gains are not sufficient.

---

## 14. V16 status (2026-08-13/14, e11 blend onto V10 parent - v3 REBUILT and pushed)

### RESULT (2026-08-14 early UTC): SUBMITTED 55491586 -> PUBLIC 0.912, RANK 79/1410

**V16 v3 scored 0.912 (+0.006 over V10/V9 incumbent 0.906) - NEW ACCOUNT BEST.**
Rank improved 99 -> **79 / 1410**. Top-50 cutoff (as of 2026-08-14 00:39 UTC):
**0.919**, gap -0.007. Leaderboard saved: `artifacts/v16_remote/leaderboard_2026-08-14.csv`.

Key facts of the winning submission:
- ref **55491586**, submitted 2026-08-13 21:34:52 UTC, kernel version 3 of
  `danielsolo1770/rsna-knee-v16-e11-blend`, status COMPLETE, publicScore 0.912.
- The submission-mode re-run completed WITHOUT consuming the weekly GPU quota
  (quota stayed ~92.8 min the whole time) - the ~4.5h runtime estimate was
  wrong; submission re-runs run fast/on separate accounting. If this holds,
  submitting again after the quota refresh is not even necessary.
- V16 v3 was submitted ~21:34 UTC after the user approved using the remaining
  daily submissions (v1/v2/v3 = 3 used that day; v1 55489524 and v2 55490131
  both errored - format, totalBytes 0 - because they only produced 3-row files).

### Root cause of the failed submissions (55489524 v1, 55490131 v2)

**Submission mode re-runs the notebook against the FULL hidden test (~1,322 studies;
V10's log: "sizing for 4407 train + 1322 test studies"). Successful submissions are
~380 KB (V10 55455223 = 379,667 B / 0.906; V13 55466431 = 379,736 B / 0.899).** The
3-row public `test.csv` is only a sample. v1 (mounted V10 notebook as kernel_source)
and v2 (3-row V10 CSV embedded as base64) could never produce a file covering the
hidden test: `_v10_validate_sub` against the full test fails -> fallback writes 3 rows
-> "incorrect format", totalBytes 0.

### v3 design (BUILT, PUSHED as version 3, VERIFIED 2026-08-14 ~00:00)

Same shape as V13 (which proved an in-kernel full pipeline fits): **V16 v3 notebook =
V10 cells 0-28 + FULL untrimmed V10 cell 29 (runs `main()` + `run_v10_rad_cross()`
in-kernel) + new appended cell** (`temp\opencode\v16_appended_cell_v3.py`). The V16
cell reads the **in-kernel V10 parent** (prefer `rsna_rad_v10/submission_v10_cross_505.csv`,
else `submission_v9_parent.csv`, else primary `submission.csv`), gates w=0.20 of the
E11 4-slot 130 mm arm on OOF + gold rows (G1 soft +0.00396, G2 5/5 gold folds,
G3 worst -0.00102, G4 rank corr 0.7346 - all PASS), decodes the full test with the E11
recipe, blends ranks, validates against full `test.csv` UIDs, overwrites
`submission.csv`; on any error the in-kernel V10 parent submission is preserved.
`kernel_sources` removed (parent is in-kernel now); 5 h total TIME_BUDGET cap injected
before the run block (V13 precedent: main + E9 + E11 fit in 4.5 h on the full test).

v3 normal-mode run (3 public studies) COMPLETE, ~36 min, both audits clean:
- `v10_rad_audit.json`: status CANDIDATE_SELECTED, alpha 0.505, gold 0.854985 -
  deterministic reproduction of the original V10 run.
- `v16_e11_audit.json`: status V16_PROMOTED, gate passes, parent rows 3 / test rows 3,
  e11 encoder sha matches expected RadImageNet checkpoint.
- Artifacts: `temp\opencode\v16_out_v3\` (submission.csv = 725 B blended, both audits,
  `rsna_rad_v10/` + `rsna_rad_v16/` dirs, log ends "[ 1800.0s] done").
- Builder: `temp\opencode\build_v16_notebook_v3.py` (asserts no embedded payload, no
  kernel_sources, full cell 29 present); gate verifier `verify_v16_gate.py` updated to
  read the v3 cell - ALL MATCH.

### Next step - do NOT submit more variants

- Protocol: no LB-hacking / no re-tuning on the same 58 gold rows / no lowering gate
  thresholds. V16 v3 was the pre-registered final candidate and it won. STOP here.
- If any future candidate is built (new family only, e.g. fresh heads or a new arm),
  re-submit script ready: `temp\opencode\submit_v16_v3.py` (kagglesdk, correct
  private-attr pattern; `--force` skips the quota guard; usage documented in-file).
- GPU quota refresh was 2026-08-15T00:00:00Z; submission re-runs apparently do not
  charge the weekly GPU quota, so a future submit does not depend on it.
- Never push this project to GitHub.

## 15. V17 exploration (2026-08-14, after V16 v3 scored 0.912) - ALL CANDIDATES GATE-BLOCKED

Surveyed every new public package released 2026-08-12..14 for a genuinely new
arm. Result: **no valid candidate - no submission made. V16 v3 (0.912) stays
the incumbent.** Daily submissions (5) remain unused.

| Package | Family | OOF shipped? | Independent gate result | Verdict |
|---|---|---|---|---|
| surasan092/rsna-knee-dinov3-s-v13-1-gpu-util-full-ft-model | DINOv3-S+ 336px 6-slot full-FT | YES (per-fold val_preds + indices + folds.csv) | soft macro **0.82897** vs parent 0.85713 (**-0.0282**); gold macro 0.820 vs 0.85445 | **BLOCKED** |
| prvsiyan/rsna-knee-b3-v47-public-deployment | EfficientNet-B3 224px | NO (audit only on 58 gold rows; oof_predictions: false) | cannot gate (only forbidden gold-tuned diagnostics) | BLOCKED |
| bishalroy999/rsna-cache336-v2 | dinov2s 336px cache+ckpt | NO (only fold0 metrics json) | cannot gate | BLOCKED |
| mattiaangeli/rsna-knee-radimagenet-foldsv1-heads | same RadImageNet family | NO | same-family arm; protocol | BLOCKED |
| prvsiyan/rsna-knee-resnet18-student-v1 | student distillation | NO | V15 lesson (-0.0137 blocked) | BLOCKED |
| dragonseal/rsna-v4-weights, sadamtorres/rsna-ft-b, etc. | misc | NO | 0-1 downloads, no evidence | BLOCKED |

Independent verification was run locally with the exact V16 label/score pipeline
(reproduces parent 0.85713 / gold 0.85445 / gold=58 exactly - see
	emp\opencode\v16_soft_sanity.py): DINOv3 focal-family OOF (concatenated
per-fold val predictions mapped via folds.csv) scores soft macro 0.82897,
gold macro 0.82000 - both far below the parent. Per the V15 precedent
(-0.0137 blocked), the arm is rejected.

Note: the DINOv3 package's own member_results.csv (derived_auc mean ~0.84) is
consistent with our independent number (~0.829 on our soft labels; label table
differences explain the small offset). The verdict does not depend on which
label table is used.

### What a future valid candidate needs (monitor checklist)

- Released package with **per-fold validation predictions + indices + fold
  table** (like surasan092's layout), OR weights we can run OOF inference on
  within our GPU budget (92 min refresh 2026-08-15T00:00Z; push+verify ~36 min).
- Gate: OOF soft macro >= parent 0.85713 (tolerance ~0.002), gold macro >=
  0.85445, no target below -0.01 - computed with
  	emp\opencode\research\dinov3_oof_gate.py pattern (label table: 3-source
  LLM mean + official labels on the 58 gold rows, rank-based macro AUC).
- Genuinely different backbone/architecture or label source (not same-family
  RadImageNet heads, not another student).
- Re-submit path ready: build_v16_notebook_v3 pattern (in-kernel parent +
  appended cell) + submit_v16_v3.py.

### V17 in-kernel arms (2026-08-14 evening) - BUILT, awaiting quota

Per docs/TOP50_PLAN_V17.md (pre-registered), the V17 notebook is BUILT and
statically verified:

- `kaggle\submission_v17_dbase_e11blend\rsna-knee-v17-dbase-e11blend.ipynb`
  (33 cells) + `kernel-metadata.json` (id danielsolo1770/rsna-knee-v17-dbase-e11blend,
  model_sources += metaresearch/dinov2/PyTorch/base/1, enable_internet false, T4).
- Cells 0..30 == V16 v3 except cell 29 cap line 5h -> 8h (verified line-by-line).
- Cell 31 = Arm A (DINOv2-B/14 from-scratch ViT-B/14 arch, from-scratch because
  internet is disabled; HF key mapping + safetensors + torch.hub-format fallback;
  e11 4-slot geometry @224; encode chunks 900/700; GroupKFold(5); gates A1-A4;
  strict +0.0005; w_A=0.10 blend onto V16 candidate; checkpoint
  submission_after_e11.csv; arm_a_oof.npz; arm_a_audit.json).
- Cell 32 = Arm B (RadImageNet ResNet50 `_v10_load_encoder`, 6-slot @336
  SAG_FLUID_FS/COR_FLUID_FS/AX_FLUID_FS/SAG_FLUID_NOFS/COR_T1/SAG_T1,
  `_V10FoundationQueryHead` 2048-dim, seed_off=400; incumbent = after-A via
  arm_a_oof.npz else V16 candidate; w_B=0.10; arm_b_audit.json).
- Verified by `verify_v17_build.py` (structure + symbol audit + hygiene).
- Build files: `%LOCALAPPDATA%\Temp\opencode\build_v17_notebook.py`,
  `verify_v17_build.py`, `v17_arm_a_cell.py`, `v17_arm_b_cell.py`.

NEXT: wait for GPU quota refresh 2026-08-15T00:00:00Z (need ~2h for
normal-mode verify), then push + verify + check audits + submit (1 of 5 daily).
Outcome rules (plan, immutable): score > 0.912 -> new incumbent; else V16 v3
stays, document, stop.

UPDATE 2026-08-14 12:30 Tehran: kernel v1 PUSHED
(https://www.kaggle.com/code/danielsolo1770/rsna-knee-v17-dbase-e11blend).
Normal-mode verify needs ~2h quota (arms train on the full train split);
quota still 92.8 min (refresh 2026-08-15T00:00:00Z = 03:30 Tehran). Scripts:
- %LOCALAPPDATA%\Temp\opencode\run_v17_verify.py (run session, poll, download,
  validate audits: needs ARM_A_PROMOTED + ARM_B_PROMOTED, 3-row submission.csv)
- %LOCALAPPDATA%\Temp\opencode\submit_v17.py (guards >=3h quota; submission
  re-run does not consume quota per v16 precedent)
After refresh: run_v17_verify.py -> inspect report -> submit_v17.py -> poll
submission -> docs update.

DEVIATION (user decision 2026-08-14 12:40 Tehran): the user authorized using up
to 3 of today's 5 daily submissions as verify runs BEFORE the quota refresh,
because submission-mode re-runs consume NO GPU quota (v16 precedent). Cells
0..30 are byte-identical to the proven v3 kernel; only cells 31/32 are new.
Risk accepted: a crash in cells 31/32 loses one submission (failed submission,
no score). Evidence from a successful submission run: the final score itself
(==0.912 => gates blocked, >0.912 => promoted); audits may or may not be
downloadable from submission runs (try kaggle kernels output after the run).
Outcome rules unchanged: score > 0.912 -> new incumbent; <= 0.912 -> V16 v3
stays, document, stop. If the run fails, fix and resubmit (max 3 used today;
2 remain as backup).

## 16. V17 v5 run + submission 55503652 (2026-08-14 evening) - ARMS FAILED, SUBMISSION STUCK PENDING

### Root cause of the v3/v4 builder mystery (RESOLVED)
`build_v17_notebook.py` originally emitted list-source cells (v3 cells 29/30)
as `src.split("\n")` parts WITHOUT trailing newlines; Jupyter joins list parts
with `''` -> each cell collapsed to ONE line. Cell 29 starts with `#` -> the
entire 32,439-char cell became one giant comment -> silent no-op -> `_V10Path`
and all parent definitions never created. Explains: v3 unlock = COMPLETE with
no output files (cells 29/30 no-op), v4 full = ERROR at In[23]
(NameError: name '_V10Path' is not defined). FIXED: all cell sources are now
plain strings with real newlines + `id` fields (uuid4); rebuild verified
(33 cells, all parse, cap 8h once in cell 29).

### v5 (full fixed) normal-mode run - COMPLETE, no promotion
- Pushed ~06:50 UTC with `-t 3900` (65-min cap); COMPLETE ~07:55 UTC.
- Arm A: **ARM_A_ERROR** - pos_embed size mismatch when loading base weights:
  checkpoint `[1, 1370, 768]` (DINOv2 base has register tokens + 518px
  interpolation) vs plain ViT-B/14 `[1, 197, 768]` -> RuntimeError ->
  no candidate.
- Arm B: **ARM_B_GATE_BLOCKED** - ran to completion (~62.8 min); gates
  a1/a2/a3 pass, **a4 fails (rank corr 0.9977 >= 0.95)**, fold 1 gold gain
  negative (-0.00011) -> strict_improve_ok false. soft 0.86151 vs incumbent
  0.86108, gold 0.86051 vs 0.85922 (OOF-local diagnostics only).
- Result: submission.csv = V16 v3 incumbent blend unchanged
  (sha256 6043087642e2d642813c6119abb775d8b9e5d0d245271138d8b587ebfec5bc64
  == the exact file that scored 0.912). Verified byte-identical to
  out_v5/rsna_rad_v16/submission_v16.csv and out_v5/rsna_rad_v10/
  submission_v10_cross_505.csv (parent 29a1ae14...).
- Logs/artifacts: %LOCALAPPDATA%\Temp\opencode\out_v5\ (log + audits + csvs).

### Submission 55503652 - STUCK PENDING (root cause: no GPU quota)
- Created 10:49:50 UTC with VERSION=5 via submit_v17.py --force (description
  documents the arm failure + file sha).
- The submission-mode re-run ("v6") shows COMPLETE but did NOT execute:
  quota unchanged (29.87099611h used / 30h, 7.7 min left) and v6 log+output
  are byte-identical to v5's (same sha 1ffc81fc...). Only 0.13h quota existed
  at submission time; a 63-min GPU notebook cannot run. Submission stuck
  PENDING with totalBytes=0; REST download 404s; no error_description.
- The v16 precedent ("submission-mode consumes no quota") is FALSE for
  v17/quota-exhausted conditions - the run simply never starts.

### Next (immutable outcome rules)
1. After quota refresh 2026-08-15T00:00:00Z (=03:30 Tehran, ~10h): if
   55503652 is still PENDING, re-run submit_v17.py (VERSION=5, verify #2 of
   authorized 3) - the submission-mode run will actually execute (~63 min)
   and score ~0.912 (deterministic file).
2. Score > 0.912 -> new incumbent; <= 0.912 -> V16 v3 stays, document, stop.
3. If Arm A is ever retried (not planned - V17 is the final attempt per
   plan): fix `_v17_load_base` to handle register tokens / interpolate
   pos_embed, or skip base-weights load for the from-scratch arch.

### RESULT 2026-08-14 evening: submission 55503652 SCORED 0.912 - V17 CLOSED
Submission completed on its own (the scorer accepted the v6 inherited output
without a fresh execution) with **publicScore = 0.912** - exactly the V16 v3
incumbent score, as expected from the byte-identical file
(sha 6043087642e2d642813c6119abb775d8b9e5d0d245271138d8b587ebfec5bc64).
Per the immutable outcome rules (<= 0.912): **V16 v3 stays the incumbent,
document, stop.** No further submissions planned. Daily submissions used: 1 of
5 on 2026-08-14 (verify #1 of authorized 3).

## 17. V17 CONTINUES (user decision 2026-08-14 evening) - Arm A FIXED, rerun queued

The user confirmed the top-50 goal stands and the plan continues. V17 is NOT
closed: Arm A's error was a plain bug, now fixed and verified locally:

- `Net.pos_embed` `(1,197,768)` -> `(1,257,768)` (224px/14 = 16x16 patches +
  cls = 257; old size was for 196px and would have failed forward anyway).
- `_v17_load_base` interpolates the DINOv2 base pos_embed (1370 = 1 cls +
  37x37 grid @518px) to 257 (1 cls + 16x16 grid) via `_v17_interp_pos`
  (bicubic, align_corners=False, cls preserved).
- Verified locally: synthetic DINOv2 checkpoint -> all 198 params load,
  forward 224px -> (1,768) finite (`arm_a_load_sim.py` PASS).
- Notebook rebuilt (33 cells, `build_v17_notebook.py` OK, `verify_v17_build.py`
  OK); run_v17_verify.py updated (version arg, MIN_QUOTA 2.5h, submission.csv
  col = StudyInstanceUID); submit_v17.py VERSION=7 + corrected quota note
  (submission-mode re-runs DO need available GPU quota - v6 stall proved it).

RERUN PLAN (docs/TOP50_PLAN_V17.md "Update 2026-08-14 evening"):
1. After quota refresh 2026-08-15T00:00:00Z (=03:30 Tehran): push fixed build
   with `-t 7500` (125-min cap; next version = 7).
2. `run_v17_verify.py 7` -> audits: want ARM_A_PROMOTED / ARM_B_PROMOTED +
   3-row submission.csv.
3. If either arm promotes: submit (verify #2 of authorized 3). If neither:
   V16 v3 stays, no submission.
4. Outcome rules unchanged: > 0.912 -> new incumbent; <= 0.912 -> V16 v3
   stays, document, stop.

Current GPU quota: 7.7 min left; refresh 2026-08-15T00:00:00Z.

