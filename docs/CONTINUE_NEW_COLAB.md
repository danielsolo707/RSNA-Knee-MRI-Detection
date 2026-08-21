# Continue on a new Colab / Gmail

Same Kaggle account: `danielsolo1770`. New Google account is fine. Do not farm a second Colab for quota stacking in parallel with the old one.

## Already saved off the old Drive

- Private Kaggle dataset `danielsolo1770/rsna-knee-v28-heads`  
  5× small + 5× base `best.pt`, fold0_slice9, OOF CSVs, receipts, HF configs, `train_core.py`
- Local copy: `artifacts/v28_oof/oof_receipt.json`
- Colab snapshot: `artifacts/colab_v28_scratchpad.ipynb`
- Trainer: `modal_v28/train_core.py` (has `expand_slices`)
- V29 fail-closed kernel source: `kaggle/submission_v29_v27_v28/` — **do not submit**
- V30 cache kernel: `danielsolo1770/rsna-knee-v30-cache-224` (CPU, no submit)

Do **not** download the 4 GB V28 cache locally. Pull it on the new Colab from Kaggle.

## New Colab bootstrap

```python
# Runtime -> T4. Mount the NEW Drive.
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")

# Kaggle token: write ~/.kaggle/access_token then
# export KAGGLE_API_TOKEN from that file. Do not paste token into chat.

!kaggle kernels output danielsolo1770/rsna-knee-v28-cache-224 -p /content/v28_kaggle_out -o
!kaggle datasets download danielsolo1770/rsna-knee-v28-heads -p /content/v28_heads -o --unzip
!kaggle kernels status danielsolo1770/rsna-knee-v30-cache-224
```

When V30 is COMPLETE:

```python
!kaggle kernels output danielsolo1770/rsna-knee-v30-cache-224 -p /content/v30_kaggle_out -o
```

Copy `modal_v28/train_core.py` to `/content/train_core.py`.

## State

| Item | Value |
|---|---|
| Incumbent | V27 `55580781` **0.914** — keep selected |
| V28 OOF blend | **0.818** vs parent ~0.857 — gate FAIL |
| slice9 fold-0 | **0.826** vs RGB-small 0.816 — do not 5-fold |
| Goal | public **0.930** via V30 depth cache, not another soup |

## Next

1. Poll V30 cache until `status=complete` and `n=4407`.
2. Fold-0 on V30 cache, DINOv2-small, unfreeze 6, TTA. Need **≥ 0.85** soft val.
3. Only then 5-fold. Infer = V27 + owned heads, fail-closed.
4. Keep V27 until a new file is public **> 0.914**.
