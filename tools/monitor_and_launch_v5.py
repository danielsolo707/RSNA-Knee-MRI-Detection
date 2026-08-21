#!/usr/bin/env python3
"""Poll Kaggle cache kernels; when V5 288 cache completes, print next launch steps.

This is a local helper. It does not auto-submit competition entries.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
CACHE_SLUG = "danielsolo1770/rsna-knee-v5-parallel-uint8-cache-288"
OLD_CACHE_SLUG = "danielsolo1770/rsna-knee-v3-1-uint8-cache"
FOLD0_DIR = Path(__file__).resolve().parents[1] / "kaggle" / "v5_production" / "b3_fold0"


def env_with_token() -> dict:
    env = os.environ.copy()
    token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    env["KAGGLE_API_TOKEN"] = token
    return env


def status(slug: str) -> str:
    proc = subprocess.run(
        ["kaggle", "kernels", "status", slug],
        capture_output=True,
        text=True,
        env=env_with_token(),
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    return text.strip()


def main() -> int:
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    print(f"monitoring {CACHE_SLUG} every {interval}s", flush=True)
    while True:
        new_s = status(CACHE_SLUG)
        old_s = status(OLD_CACHE_SLUG)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "NEW:", new_s, flush=True)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "OLD:", old_s, flush=True)
        if "COMPLETE" in new_s.upper() or "COMPLETE" in new_s:
            print("CACHE COMPLETE — next commands:", flush=True)
            print(
                f'  kaggle kernels push -p "{FOLD0_DIR}"',
                flush=True,
            )
            print(
                "  # after fold0 COMPLETE with fold0_final.pt + fold0_full_oof.csv,",
                flush=True,
            )
            print("  # launch folds 1-4 the same way, then OOF blend + submit.", flush=True)
            return 0
        if "ERROR" in new_s.upper() or "CANCEL" in new_s.upper():
            print("CACHE FAILED — inspect Kaggle logs and re-push.", flush=True)
            return 2
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
