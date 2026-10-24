# Transferable lessons from similar competitions (historical)

> Live state is `HANDOFF.md`. This note was compiled 2026-08-10.

Compiled 2026-08-10 for RSNA Knee Abnormality Detection.

## Scoreboard reality (this competition)

| Gate | Public AUC (live ~975 teams) |
|------|------------------------------|
| Top 50 | **0.900** |
| Top 10 | **0.925** |
| Leader | **0.942** |
| Our best | **0.899** (#97) |

Gap to top 50 is **+0.001**. Gap to top 10 is **+0.026**. Top 50 is an
execution problem (new model family + blend). Top 10 is a research program
(labels + specialists + diversity + clean OOF blending).

## RSNA Breast Cancer Detection (1st place patterns)

Source: dangnh0611 / discussion 392449 and public repo summary.

| Tactic | Transfer to knee MRI |
|--------|----------------------|
| **ROI detector → classifier** (YOLOX then ConvNeXt) | Joint/meniscus crop or FOV normalize before classify; full-field + crop dual path |
| Multi-fold strong single backbone | Our B3 5-fold family |
| External data carefully licensed | Check MIRA terms before any external knee set |
| TensorRT / optimized infer | After score locks; efficiency track exists here too |
| Checkpoint selection on CV, not public LB | Nested OOF only |

## RSNA 2023 Abdominal Trauma (Team Oxygen 1st)

| Tactic | Transfer |
|--------|----------|
| Multi-organ / multi-head specialists | Trauma / fluid / ligament / OA heads |
| CoAt / EfficientNet diversity | DINO (transformer) + B3 (CNN) is the minimum diversity |
| 4-fold + full-data models | After 5-fold OOF, optional full-data finetune of one seed |
| Post-process ensemble per target | Nonnegative per-target rank weights |

## RSNA 2024 Lumbar Spine Degenerative Classification

| Tactic | Transfer |
|--------|----------|
| Study-level + series-level models | Series attention over multiple fluid-sensitive series, not one hard slot |
| Multi-crop aggregation | Physical mm crop (130 mm public recipe) + full FOV |
| Sagittal T2 + axial T2 jointly | Keep all three planes; do not drop axial for “speed” |
| Small model ensembles beat one giant model | Prefer many medium models inside 9 h |

## MRNet (Stanford knee MRI)

| Tactic | Transfer |
|--------|----------|
| Per-plane slice features → study aggregation | Our MIL attention head |
| Multi-plane fusion | Sagittal + coronal + axial slots with plane embeddings |
| Transfer learning + careful aug | No horizontal flip (laterality); intensity only |
| Fixed slice sampling | Uniform at val; stochastic at train |

## What actually moves **this** leaderboard

1. **Report → soft labels** (macro gold ~0.89+). Weakest targets: Synovitis,
   Lateral OA, Lateral Meniscus, Contusion.
2. **A second architecture** (CNN MIL) blended with public DINO ranks.
3. **Physical geometry** (order, laterality, square pad, mm crop, fluid slots).
4. **OOF-gated rank blends** — probability averages and unreproduced public
   weights are traps (our Tony 0.828 vs claimed 0.905).

## Explicit non-goals for the next week

- More Tony-only or unreproduced “0.90x” notebooks.
- Public-LB ensemble soup without nested OOF.
- Horizontal flips on laterality-sensitive targets.
- Submitting smoke / partial checkpoints.

## Execution map to top 10

```
Top 50 (0.900–0.905)
  └─ V5: parallel 288 cache → B3 5-fold → OOF blend with DINO 0.899 anchor

Top 20 (~0.913+)
  └─ Fix Synovitis/Lateral labels
  └─ Second diverse family (ConvNeXt / longer-context MIL)
  └─ Target specialists (trauma + fluid)

Top 10 (≥0.925)
  └─ Nested multi-family blend + ROI crop path
  └─ Multi-seed full-data distillation for inference budget
  └─ Strict private-LB robustness (scanner-grouped CV)
```
