# Target 0.940 — locked execution

**Incumbent:** V25 `55567630` **0.913** (do not deselect below this)  
**Goal:** public **0.940**  
**Deadline:** 2026-10-22  
**Rule:** Kaggle CLI only. Never push this repo to GitHub.

## Why 0.940 is not another public fork

Forum evidence (2026-08-15..17):

- Tucker Arrants: **single model 0.934 @ 224px, 5-fold**
- Tim Krige: single model **~0.92**, OOF ≈ LB; unfreeze more DINO layers; LLM labels are often wrong
- Tom Aindow: single DINOv2 **0.915**; high scores on the efficiency board are not 20-model soups

Public DINO+E9+E11+Mattia soups saturate at **0.913** (measured: V22/V23/V24/V25/V26).

## Live program

1. **V27** (in flight): fail-closed `0.80*rank(V25)+0.20*rank(V24)`. Submit only if dry-run `V27_BLENDED`. Expected +0.000–0.003. Not the 0.940 path.
2. **V28** (owned train): DINOv2-small, last 4 blocks unfrozen, 224px, 160 mm crop, 3 adjacent slices, 3 FS slots, 3-teacher soft labels, GroupKFold 5. Fold 0 first, no submission from the trainer.
3. **After fold-0 val AUC is real** (≥ ~0.80 on soft val): run folds 1–4, then a gated infer kernel that blends the owned OOF family onto V25/V27 only if rank-corr < 0.95 and no target collapse.
4. **Labels v2** in parallel on CPU: target-specific silence, do not fit on the 58 gold rows.
5. **Stop:** V41 clones, B3, students, DINOv3 w>0.20, gold-58 promotion.

## Success bars

| Public of owned 5-fold alone | Action |
|---|---|
| ≥ 0.930 | new incumbent; ensemble toward 0.940 |
| 0.920–0.929 | keep; blend with V25/V24 |
| 0.905–0.919 | third family only if diverse |
| < 0.900 | debug pixels/labels; do not blend |
