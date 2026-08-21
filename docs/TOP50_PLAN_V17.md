# TOP-50 PLAN V17 — two in-kernel trained arms onto the V16 incumbent (0.912)

Status: PRE-REGISTERED 2026-08-14. No gate threshold or weight in this doc may be
changed after seeing any leaderboard result. Build only after user approval.

## Goal and constraints

- Incumbent: V16 v3 = in-kernel V10 parent (DINO small ensemble + RadImageNet e9
  cross, alpha 0.20 / ACL+Syn 0.505) + e11 diverse 4-slot 130mm arm at w=0.20.
  Public 0.912, rank 79/1410. Top-50 cutoff 0.919 (gap -0.007).
- Submission-mode re-runs do not charge the weekly GPU quota (observed with
  ref 55491586). Normal-mode verify runs DO charge it (92 min remaining now,
  refresh 2026-08-15T00:00:00Z).
- Survey of all public packages released 2026-08-12..14 (DINOv3-S+, B3, pulse-v6,
  knee-mri-fold-weights, cache336, radimagenet-foldsv1, students, ...): **none
  ships gateable OOF except DINOv3-S+ which independently fails its gate
  (soft 0.82897 vs parent 0.85713, see SESSION_HANDOFF.md sec 15).** Therefore
  the only valid path is NEW ARMS TRAINED IN-KERNEL with their own CV OOF.

## V17 design (all pre-registered)

Kernel: new notebook `rsna-knee-v17-dbase-e11blend` (kernel version 1) = V16 v3
cells 0..30 (cells 29/30 unchanged: in-kernel V10 parent + e11 gate/blend) + two
NEW appended cells (31, 32). TIME_BUDGET cap raised to 8*3600 s.

### Arm A (cell 31) — DINOv2-base heads, e11 geometry

- Encoder: **DINOv2 ViT-B/14** (`metaresearch/dinov2/pytorch/base/1` mounted as
  model source; load via torch.hub `dinov2_vitb14` with pretrained=False and the
  mounted pytorch_model.bin mapped onto the hub architecture; if the mount or
  mapping fails, fall back to hub download; if both fail -> arm skipped, audit
  records "ARM_A_LOAD_FAIL", blend continues without it).
- Geometry: e11 slots (SAG_NOFS, COR_NOFS, AX_NOFS, SAG_FS), 130mm crop, 224px,
  8 slices, N_GROUP=8 windows (identical to `_v16_e11_test`).
- Training: chunked encode of train (4,407 studies) -> frozen features; labels =
  `_v10_make_targets` (3-source soft, gold rows fixed); GroupKFold(5) on
  `_v10_report_groups`; heads = `_V10FoundationQueryHead` trained by
  `_v10_train_fold` (lr 2e-4, 24 ep, early stop); OOF per fold concatenated.
- Test: chunked encode of full test -> mean of 5 fold heads.
- Gate A (vs parent, all pre-registered):
  - A1: soft_macro(A_rank) >= soft_macro(parent_rank) - 0.002
  - A2: gold_macro(A_rank) >= gold_macro(parent_rank) - 0.002
  - A3: worst per-target soft delta >= -0.01
  - A4: mean rank corr(parent, A) < 0.95
  - PASS => blend w_A = 0.10 rank: cand = 0.90*incumbent + 0.10*A_rank; write
    checkpoint submission.csv + rsna_rad_v17/submission_v17_a.csv + audit.
  - FAIL => keep incumbent file, audit "ARM_A_GATE_BLOCKED".

### Arm B (cell 32) — RadImageNet 6-slot heads at 336px

- Encoder: same RadImageNet ResNet50 as V10 (`_v10_load_encoder`).
- Geometry: 6 slots per DINOv3 recipe (SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS,
  SAG_FLUID_NOFS, COR_T1, SAG_T1), 130mm crop, **336px**, 8 slices, N_GROUP=8.
- Training: chunked encode (memory: 6 slots x 336px x 8 slices per study ~5.4 MB
  -> encode in study chunks, keep only features); same labels, GroupKFold(5),
  `_v10_train_fold`; OOF per fold.
- Test: chunked encode of full test -> mean of 5 heads.
- Gate B (vs parent, same thresholds as A1-A4).
  - PASS => blend w_B = 0.10 rank onto incumbent-after-A; write checkpoint +
    submission_v17_b.csv + audit.
  - FAIL => keep incumbent file, audit "ARM_B_GATE_BLOCKED".

### Joint decision rule (pre-registered)

- Incumbent after e11 (cell 30) = V16 v3 candidate = guaranteed valid file
  (checkpointed before arm A starts).
- Candidate after A passes = incumbent + A (w 0.10). Candidate after B passes =
  incumbent-after-A + B (w 0.10).
- Final promote ONLY if cand_soft >= incumbent_soft + 0.0005 AND cand_gold >=
  incumbent_gold + 0.0005 at each stage (strict improvement, computed in-kernel
  on OOF/gold rows, recorded in audit). Otherwise keep the previous file.
- Any arm failure (load/encode/timeout/exception) degrades gracefully: audit
  records it, blend continues with remaining arms. The last written
  submission.csv always covers the full test (validated per stage).

## Runbook

1. Build + static checks (this repo, CPU only): `temp\opencode\build_v17_notebook.py`
   (asserts: no embedded payload, cells 0..30 byte-identical to V16 v3, new
   cells present, model_sources present, cap 8h).
2. After quota refresh (2026-08-15T00:00:00Z): push kernel (version 1).
3. Normal-mode verify run (~2h: parent 30 min + A ~65 min + B ~90 min, 3-row
   test). Check audits: v10_rad_audit.json CANDIDATE_SELECTED, v16_e11_audit.json,
   v17_arm_a_audit.json, v17_arm_b_audit.json; submission.csv rows = 3.
4. Submit (1 of 5 daily). Submission re-run ~7-8h (free quota). Poll every
   >=10 min (403s at higher frequency).
5. Outcome rules: score > 0.912 -> new incumbent, update SESSION_HANDOFF sec 1.
   Score <= 0.912 or reject -> V16 v3 stays incumbent; document; stop.

## Time/memory guards (pre-registered)

- Arm A needs >= 2.0h remaining else skip (audit ARM_A_SKIPPED_TIME).
- Arm B needs >= 2.0h remaining after A else skip.
- Encode chunks sized so peak RAM < 9 GB (chunk ~800 studies @336 for arm B).
- TIME_BUDGET cap 8h; each arm writes its own audit + checkpoint before next arm.

## Update 2026-08-14 evening — v5 run executed, arms failed, PLAN CONTINUES

User decision: the top-50 goal stands; V17 continues (not "final attempt").
v5 (version 5, full fixed build) ran to COMPLETE in normal mode (cap 3900s):

- Arm A: **ARM_A_ERROR** — `_v17_load_base` copied the DINOv2 base pos_embed
  `[1,1370,768]` (518px 37x37 grid) into a model whose pos_embed was
  `[1,197,768]` (sized for 196px) -> size mismatch RuntimeError at
  load_state_dict; no candidate, no OOF, no blend.
- Arm B: **ARM_B_GATE_BLOCKED** — ran fully; gates a1/a2/a3 pass, **a4 fails**
  (mean rank corr vs incumbent 0.9977 >= 0.95) and gold fold 1 gain
  -0.00011 -> strict_improve_ok false. OOF-local diagnostics only.
- Result: submission.csv stayed the V16 v3 incumbent blend
  (sha 6043087642e2d642813c6119abb775d8b9e5d0d245271138d8b587ebfec5bc64).

### Arm A fix (applied 2026-08-14 evening, verified locally)

- `Net.pos_embed` corrected `(1,197,768)` -> `(1,257,768)` (224px/14 = 16x16
  patches + cls = 257 tokens; the old 197 was a 196px size).
- `_v17_load_base` now routes `embeddings.position_embeddings` / `pos_embed`
  through `_v17_interp_pos`: cls token kept, 37x37 grid bicubic-interpolated
  to 16x16, reattached (standard DINOv2 pos-embed interpolation).
- Locally verified: synthetic DINOv2 base checkpoint (1370-token pos_embed,
  HF naming) loads all 198 params; forward on 224px -> (1,768) finite
  (`temp\opencode\arm_a_load_sim.py`).
- Notebook rebuilt (33 cells) + `verify_v17_build.py` passes.

### Rerun protocol (after quota refresh 2026-08-15T00:00:00Z)

1. Push the fixed build: `kaggle kernels push -p kaggle\submission_v17_dbase_e11blend -t 7500`
   (next version = 7; cap 125 min: cells 0..30 ~35 min + Arm A ~25-35 min +
   Arm B ~55 min).
2. `python run_v17_verify.py 7` (MIN_QUOTA now 2.5h; submission.csv col check
   fixed to StudyInstanceUID).
3. If audits show ARM_A_PROMOTED / ARM_B_PROMOTED: set submit_v17.py
   VERSION=7 + DESCRIPTION from audits, submit (verify #2 of authorized 3).
   If neither promotes: V16 v3 stays incumbent (no new file to submit).
4. Outcome rules unchanged: score > 0.912 -> new incumbent; <= 0.912 -> V16 v3
   stays, document, stop.
5. Time guard note: both arms need >= 2.0h of TIME_BUDGET remaining; the
   125-min cap still lets Arm B run (~55 min) after Arm A (~35 min).