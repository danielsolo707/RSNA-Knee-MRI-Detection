# Kaggle GPU budget — 28 hours (wave-1)

Last updated: 2026-08-10. Full narrative: `docs/SESSION_HANDOFF.md`.

## Hard rules

1. **CPU only:** cache, labels, OOF blend math, dataset packaging.
2. **GPU only:** B3 fold training, final DINO+B3 inference.
3. **Never GPU:** Tony, curiosity public kernels, broken smokes, on-demand DICOM train.
4. **One fold at a time** (sequential).
5. **No submit** until OOF gates pass.

## Working production recipe (fold0 v4 proven)

| Knob | Value |
|------|--------|
| image_size | 288 |
| train/valid slices | **8** (from 12-slice cache) |
| batch | **1** |
| workers | **0** |
| backbone_chunk | **6** |
| epochs | **1 frozen + 3 unfrozen** |
| require-cache | on |
| session cache | `/kaggle/working/v3_1_session_cache` only |

## Accounting

| When | Job | GPU h | Outcome |
|------|-----|-------|---------|
| 2026-08-10 | fold0 v1–v2 | ~0.25 | ERROR: mkdir on `/kaggle/input` |
| 2026-08-10 | fold0 v3 | ~0.8 | SIGKILL OOM (batch2/12s/w2) |
| 2026-08-10 | fold0 v4 | **0.91** | **COMPLETE** — 882 OOF, final.pt |
| 2026-08-10 | fold1 | ~0.9 | RUNNING / check status |
| pending | fold2–4 | ~2.7 | not pushed until fold1 COMPLETE |
| pending | dino+b3 infer | 2–3 | after all folds |
| **5-fold train projection** | | **~4.5** | under 20h cap |

## After top 50 (residual hours)

1. Stronger B3 (+epochs/seed) only if OOF undertrained  
2. Else one diverse family (not DINO clone)  
3. Else specialists (Synovitis / trauma)  
4. Final 2 private-robust submissions  
