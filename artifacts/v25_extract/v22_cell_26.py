_A5_SAVED = dict(globals())
import gc, os, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')
cv2.setNumThreads(1)
CROP_MM = 130.0
SIZE = 336
SLICE_BAND = (0.12, 0.88)
N_SLICE = 16
INTENSITY = 'slice'
SLOTS = [('Sagittal', 1), ('Sagittal', 0), ('Coronal', 1), ('Coronal', 0), ('Axial', 1), ('Axial', 0)]
N_SLOT = len(SLOTS)
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

def _find_dir(*names):
    root = Path('/kaggle/input')
    cand = []
    for n in names:
        cand += [root / n, root / 'competitions' / n, root / 'datasets' / n]
        for parent in (root / 'datasets', root / 'competitions', root):
            if parent.is_dir():
                try:
                    cand += [d / n for d in parent.iterdir() if d.is_dir()]
                except OSError:
                    pass
    for p in cand:
        if p.is_dir():
            return p
    return None
COMP = _find_dir('rsna-knee-abnormality-detection')
CKPT = _find_dir('knee-mri-fold-weights')
assert COMP is not None, 'competition data not attached'
assert CKPT is not None, 'fold weights not attached'
assert (COMP / 'sample_submission.csv').exists(), f'no competition data at {COMP}'
assert list(CKPT.glob('*_f*.pt')), f'no checkpoints at {CKPT}'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'competition : {COMP}')
print(f'checkpoints : {CKPT}')
print(f'device      : {DEV}')
for i in range(torch.cuda.device_count() if DEV == 'cuda' else 0):
    cc = torch.cuda.get_device_capability(i)
    print(f'  gpu{i}       : {torch.cuda.get_device_name(i)} sm_{cc[0]}{cc[1]}, {torch.cuda.get_device_properties(i).total_memory / 2 ** 30:.0f} GiB, native bf16={cc >= (8, 0)}')