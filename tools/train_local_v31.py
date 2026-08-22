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
    out_dir = REPO / "artifacts" / "v31_runs" / ("fold0_smoke" if smoke else "fold0_v31")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout/stderr -> console + log file (crash forensics + live monitoring)
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

    from v31probe import Cfg, main

    cfg = Cfg(
        cache_dir=str(REPO / "artifacts" / "v31_cache" / "v31_cache"),
        out_dir=str(out_dir),
        fold=0,
        epochs=2 if smoke else 24,
        batch=2 if smoke else 16,
        accum=4 if smoke else 1,
        lr_head=3e-4,
        lr_bb_top=5e-5,
        llrd=0.7,
        patience=6 if not smoke else 99,
        warmup=1,
        ema_decay=0.999,
        # Windows spawn of DataLoader workers hit MemoryError (16 GB RAM with the
        # 14.5 GB cache paged in). Load batches in-process instead.
        num_workers=0,
    )
    main(cfg)
