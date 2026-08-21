# Prior RSNA top solutions — what we take

Pulled via Kaggle CLI from finished contests (lumbar 2024, abdomen 2023, breast 2022–23). Original post bodies are not in `topics show`; methods below are from the long author comments + public repos named in those threads.

## Cheat-code (legal)

Winners did **not** win by forking public soups. They:

1. **Localize, then classify** (breast 1st YOLOX→ConvNeXt; lumbar 2nd crop-around-disc then classify).
2. **Train a 5-fold of a medium backbone** (lumbar 7th “single stage wins”; lumbar 5th: convnext_base lost to smaller nets).
3. **Attention MIL + auxiliary loss** (lumbar 1st Avengers: MIL helped LB more than CV; aux head helped 4/5 folds).
4. **Pseudo-labels / mixed supervision** (lumbar 2nd: `0.5*label + 0.5*pseudo`).
5. **Per-organ / per-condition heads** (abdomen Oxygen; breast 6th multi-view multi-stage).

## What we installed into *this* contest

| Tactic | Where |
|---|---|
| Two diverse 0.913 families, rank blend 0.20 | **V27** kernel (running) |
| DINOv2-small, unfreeze last 4, 224px, 3-slice 2.5D | **V28** trainer |
| Attention MIL + 0.25 aux loss (Avengers) | **V28** `PartialFT` |
| Do not scale to huge backbones | V28 stays ViT-S |

## What we will not steal

Other RSNA **patient pixels** (MIRA / license). Their **weights** are for spine/breast/CT, not knees. Only the **recipe**.
