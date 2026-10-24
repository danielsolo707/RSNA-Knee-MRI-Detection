"""Submit V50 (public 0.9586 fork) to the RSNA knee code competition.

Mirrors tools/submit_v48.py: every API call is retried because this host has
intermittent SSLEOFError on api.kaggle.com.

Usage:
    python tools/submit_v50.py --check
    python tools/submit_v50.py --submit
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

COMP = "rsna-knee-abnormality-detection"
KERNEL = "danielsolo1770/rsna-knee-v50-public-0-9586-fork"
VERSION = 2
MESSAGE = (
    "V50 private fork of public heliosli 0.9586: Raptor 384+224 + R6 ConvNeXt "
    "flip-TTA + 2.5D ConvNeXt MIL, per-finding rank blend; final-day submission"
)

TOKEN = Path.home() / ".kaggle" / "access_token"


def with_retries(fn, attempts: int = 30, delay: float = 20.0):
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - network is flaky by nature
            last = exc
            print(f"  attempt {i}/{attempts} failed: {type(exc).__name__}", flush=True)
            if i < attempts:
                time.sleep(delay)
    raise SystemExit(f"all {attempts} attempts failed; last error: {last}")


def api():
    if TOKEN.exists():
        os.environ["KAGGLE_API_TOKEN"] = TOKEN.read_text().strip()
    from kaggle.api.kaggle_api_extended import KaggleApi

    a = KaggleApi()
    a.authenticate()
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    if not (args.check or args.submit):
        ap.error("pass --check or --submit")

    a = with_retries(api)
    print("authenticated")

    limits = with_retries(lambda: a.competition_get_submission_limits(COMP))
    print("limits:", limits)

    if not args.submit:
        return

    allowed = limits.get("_num_allowed_now") if isinstance(limits, dict) else None
    if allowed is not None and int(allowed) <= 0:
        raise SystemExit("no submissions allowed right now - stopping")

    res = with_retries(
        lambda: a.competition_submit_code(
            file_name="submission.csv",
            message=MESSAGE,
            competition=COMP,
            kernel=KERNEL,
            kernel_version=VERSION,
        )
    )
    print("SUBMITTED ref:", getattr(res, "ref", res))


if __name__ == "__main__":
    sys.exit(main())
