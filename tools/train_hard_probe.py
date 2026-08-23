"""Arm Y probe: V31 recipe with HARD binarized labels, 10 epochs, local GPU.
Compare against soft-label control curve (same fold/seed): ep5=0.7851 ep9=0.8031.
"""
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


out_dir = REPO / "artifacts" / "v31_runs" / "fold0_hard"
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
    fold=0,
    epochs=24,
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
