"""Launch local fold-0 V31 training on the RTX 4070 Laptop (8 GB).

Usage:  python tools/train_local_v31.py [--smoke]
Run inside any terminal; Ctrl+C is safe (resumes from last.pt on next run).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from v31probe import Cfg, main  # noqa: E402

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    cfg = Cfg(
        cache_dir=str(REPO / "artifacts" / "v31_cache" / "v31_cache"),
        out_dir=str(REPO / "artifacts" / "v31_runs" / ("fold0_smoke" if smoke else "fold0_v31")),
        fold=0,
        epochs=2 if smoke else 24,
        batch=3 if smoke else 4,
        accum=4,
        lr_head=3e-4,
        lr_bb_top=5e-5,
        llrd=0.7,
        patience=6 if not smoke else 99,
        warmup=1,
        ema_decay=0.999,
        num_workers=8,
    )
    main(cfg)
