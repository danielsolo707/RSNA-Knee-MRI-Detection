#!/usr/bin/env python3
"""Sequentially push V6 B3 folds when the previous fold COMPLETES."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

TOKEN = Path.home() / ".kaggle" / "access_token"
ROOT = Path(__file__).resolve().parents[1] / "kaggle" / "v6_production"


def env() -> dict:
    e = os.environ.copy()
    if not TOKEN.is_file():
        raise FileNotFoundError(f"missing Kaggle token: {TOKEN}")
    e["KAGGLE_API_TOKEN"] = TOKEN.read_text(encoding="utf-8").strip()
    return e


def status(fold: int) -> str:
    slug = f"danielsolo1770/rsna-knee-v6-b3-fold{fold}"
    p = subprocess.run(
        ["kaggle", "kernels", "status", slug],
        capture_output=True,
        text=True,
        env=env(),
    )
    text = ((p.stdout or "") + (p.stderr or "")).strip()
    if "Authentication required" in text or "SSL" in text or "Max retries" in text:
        return f"KernelWorkerStatus.TRANSIENT {text[:120]}"
    return text


def push(fold: int) -> None:
    path = ROOT / f"b3_fold{fold}"
    print(f"PUSH fold{fold} from {path}", flush=True)
    subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(path)],
        check=True,
        env=env(),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=4)
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument(
        "--push-start",
        action="store_true",
        help="push --start fold immediately, then wait and chain",
    )
    args = ap.parse_args()

    if args.push_start:
        push(args.start)

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
    print("All requested V6 folds finished COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
