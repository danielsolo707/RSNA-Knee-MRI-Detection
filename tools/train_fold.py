"""Generic local hard-label trainer: python tools/train_fold.py <fold> [epochs]"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


fold = int(sys.argv[1]) if len(sys.argv) > 1 else 0
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 24

out_dir = REPO / "artifacts" / "v31_runs" / f"fold{fold}_hard"
out_dir.mkdir(parents=True, exist_ok=True)
_log = open(out_dir / "console.log", "a", buffering=1, encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log)
sys.stderr = Tee(sys.__stderr__, _log)

from v31probe import Cfg, main  # noqa: E402

cfg = Cfg(
    cache_dir=str(REPO / "artifacts" / "v31_cache" / "v31_cache"),
    out_dir=str(out_dir),
    labels_csv="",
    hard_labels=True,
    fold=fold,
    epochs=epochs,
    batch=16,
    accum=1,
    lr_head=3e-4,
    lr_bb_top=5e-5,
    llrd=0.7,
    patience=99,
    warmup=1,
    ema_decay=0.999,
    num_workers=0,
)
main(cfg)
