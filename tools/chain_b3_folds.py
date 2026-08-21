#!/usr/bin/env python3
"""Sequentially push B3 folds 1-4 when the previous fold COMPLETES.

Usage:
  python tools/chain_b3_folds.py --start 1 --end 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

TOKEN = Path.home() / ".kaggle" / "access_token"
ROOT = Path(__file__).resolve().parents[1] / "kaggle" / "v5_production"


def env() -> dict:
    e = os.environ.copy()
    if not TOKEN.is_file():
        raise FileNotFoundError(f"missing Kaggle token: {TOKEN}")
    e["KAGGLE_API_TOKEN"] = TOKEN.read_text(encoding="utf-8").strip()
    # Also support legacy kaggle.json users without clobbering if present
    return e


def status(fold: int) -> str:
    slug = f"danielsolo1770/rsna-knee-v5-b3-fold{fold}"
    p = subprocess.run(
        ["kaggle", "kernels", "status", slug],
        capture_output=True,
        text=True,
        env=env(),
    )
    text = ((p.stdout or "") + (p.stderr or "")).strip()
    # Transient network / auth glitches must not look like kernel ERROR
    if "Authentication required" in text or "SSL" in text or "Max retries" in text:
        return f"KernelWorkerStatus.TRANSIENT {text[:120]}"
    return text


def push(fold: int) -> None:
    path = ROOT / f"b3_fold{fold}"
    print(f"PUSH fold{fold} from {path}", flush=True)
    # Re-read token every push in case shell env expired
    subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(path)],
        check=True,
        env=env(),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=4)
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()

    # Assume start fold is already pushed; wait for it, then push start+1..end
    for fold in range(args.start, args.end + 1):
        print(f"Waiting for fold{fold} COMPLETE...", flush=True)
        while True:
            try:
                s = status(fold)
            except Exception as exc:
                print("status error", exc, flush=True)
                time.sleep(args.interval)
                continue
            print(time.strftime("%H:%M:%S"), s[:200], flush=True)
            u = s.upper()
            # Only trust explicit Kaggle worker statuses (not Python ConnectionResetError).
            if "KERNELWORKERSTATUS.COMPLETE" in u:
                break
            if "KERNELWORKERSTATUS.ERROR" in u or "KERNELWORKERSTATUS.CANCEL" in u:
                print(f"fold{fold} failed; stop chain", flush=True)
                return 2
            # RUNNING / TRANSIENT / network noise: keep waiting
            time.sleep(args.interval)
        nxt = fold + 1
        if nxt <= args.end:
            push(nxt)
    print("All requested folds finished COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
