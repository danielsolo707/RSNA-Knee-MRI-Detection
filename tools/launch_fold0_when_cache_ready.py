#!/usr/bin/env python3
"""When V5 288 cache is COMPLETE, push fold-0 GPU train (wave-1 budget).

Usage:
  python tools/launch_fold0_when_cache_ready.py          # poll until ready, then push
  python tools/launch_fold0_when_cache_ready.py --once   # check once
  python tools/launch_fold0_when_cache_ready.py --force  # push fold0 even if cache not complete
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
CACHE_SLUG = "danielsolo1770/rsna-knee-v5-parallel-uint8-cache-288"
FOLD0_DIR = Path(__file__).resolve().parents[1] / "kaggle" / "v5_production" / "b3_fold0"


def env() -> dict:
    e = os.environ.copy()
    e["KAGGLE_API_TOKEN"] = TOKEN_PATH.read_text(encoding="utf-8").strip()
    return e


def kernel_status(slug: str) -> str:
    p = subprocess.run(
        ["kaggle", "kernels", "status", slug],
        capture_output=True,
        text=True,
        env=env(),
    )
    return ((p.stdout or "") + (p.stderr or "")).strip()


def push_fold0() -> int:
    print(f"Pushing fold0 from {FOLD0_DIR}", flush=True)
    p = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(FOLD0_DIR)],
        env=env(),
    )
    return int(p.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--interval", type=int, default=120)
    args = parser.parse_args()

    while True:
        status = kernel_status(CACHE_SLUG)
        print(time.strftime("%H:%M:%S"), status, flush=True)
        upper = status.upper()
        if args.force or "COMPLETE" in upper:
            if "COMPLETE" not in upper and args.force:
                print("FORCE push without COMPLETE cache — require-cache may fail", flush=True)
            code = push_fold0()
            if code == 0:
                print(
                    "Fold0 pushed. GPU clock is running — watch "
                    "danielsolo1770/rsna-knee-v5-b3-fold0",
                    flush=True,
                )
                print(
                    "After COMPLETE: read fold0_timing.json; if projected_five_fold_hours>20, "
                    "cut unfrozen epochs before folds 1-4.",
                    flush=True,
                )
            return code
        if "ERROR" in upper or "CANCEL" in upper:
            print("Cache failed; not launching GPU fold0.", flush=True)
            return 2
        if args.once:
            print("Cache not ready yet.", flush=True)
            return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
