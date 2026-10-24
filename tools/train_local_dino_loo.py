"""Route B: retrain our own DINOv2-S arm (v31 geometry) on the LOO pseudo-labels.

Clean-by-construction: trains ONLY on the 4,349 pseudo-labelled studies and validates
on the 58 gold studies with their TRUE labels (gold never enters training), so the
gold-58 number is an honest measure for this arm.

Usage:  python tools/train_local_dino_loo.py [--smoke]
Ctrl+C is safe - resumes from last.pt on next run.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from v31probe import Cfg, main  # noqa: E402

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    out_dir = REPO / "artifacts" / "dino_loo_runs" / ("smoke" if smoke else "loo1")
    out_dir.mkdir(parents=True, exist_ok=True)

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

    _log = open(out_dir / "console.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, _log)
    sys.stderr = Tee(sys.__stderr__, _log)

    cfg = Cfg(
        cache_dir=str(REPO / "artifacts" / "v31_cache" / "v31_cache"),
        out_dir=str(out_dir),
        labels_csv=str(REPO / "artifacts" / "pseudo_labels" / "pseudo_labels_LOO_v4_w60.csv"),
        gold_holdout=True,
        gold_csv=str(REPO / "data" / "raw" / "train.csv"),
        fold=0,
        epochs=2 if smoke else 20,
        batch=4 if smoke else 8,
        accum=4 if smoke else 2,      # effective batch 16
        lr_head=3e-4,
        lr_bb_top=5e-5,
        llrd=0.7,
        patience=99 if smoke else 6,
        warmup=1 if smoke else 2,
        ema_decay=0.999,
        # Windows spawn of DataLoader workers hits MemoryError (14.5 GB cache paged in).
        num_workers=0,
    )
    main(cfg)
