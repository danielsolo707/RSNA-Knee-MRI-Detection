"""Poll the competition submissions until V48 (ref 56495458) scores.

Shells out to the `kaggle` CLI rather than the Python SDK: the SDK's
`competitions_submissions_list` signature differs across versions (AttributeError)
and this host throws intermittent SSLEOFError on api.kaggle.com. The CLI call is
proven to work here, so wrap it in retries.

Usage:
    python tools/watch_v48_score.py                # watch until scored
    python tools/watch_v48_score.py --once         # single poll, print table
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
COMP = "rsna-knee-abnormality-detection"
WATCH_REF = "56495458"      # V48
FLOOR = 0.940               # V47, the incumbent to beat


def clean_env() -> dict:
    """Kaggle's API breaks behind the sandbox proxy - drop it."""
    env = dict(os.environ)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        env.pop(k, None)
    env["PYTHONUTF8"] = "1"
    return env


def fetch_rows() -> list[dict]:
    """One CLI call. Raises on failure so the caller can retry."""
    proc = subprocess.run(
        [PY, "-m", "kaggle", "competitions", "submissions", "-c", COMP, "--csv"],
        cwd=REPO, capture_output=True, text=True, env=clean_env(), timeout=120,
    )
    if proc.returncode != 0 or "ref,fileName" not in proc.stdout:
        raise RuntimeError((proc.stderr or proc.stdout or "no output").strip()[:300])
    return list(csv.DictReader(io.StringIO(proc.stdout)))


def with_retries(attempts: int = 40, delay: float = 20.0) -> list[dict]:
    for i in range(1, attempts + 1):
        try:
            return fetch_rows()
        except Exception as exc:  # noqa: BLE001 - network is flaky by nature
            print(f"  poll {i}/{attempts} failed: {type(exc).__name__}", flush=True)
            if i < attempts:
                time.sleep(delay)
    raise SystemExit(f"all {attempts} attempts failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=900)
    args = ap.parse_args()

    while True:
        rows = with_retries()
        stamp = time.strftime("%H:%M:%S")
        me = next((r for r in rows if r["ref"] == WATCH_REF), None)
        top = rows[0] if rows else {}

        if me:
            status = me["status"].replace("SubmissionStatus.", "")
            print(
                f"[{stamp}] V48 ref={me['ref']} status={status} score={me['publicScore'] or '-'}"
                f"  | newest ref={top.get('ref')} score={top.get('publicScore') or '-'}",
                flush=True,
            )
            if me["publicScore"] and status == "COMPLETE":
                score = float(me["publicScore"])
                verdict = "BEAT FLOOR" if score > FLOOR else "did NOT beat the 0.940 floor"
                print(f"[{stamp}] V48 SCORED {score} -- {verdict}", flush=True)
                return
        else:
            print(f"[{stamp}] V48 ref not listed yet; newest={top.get('ref')}", flush=True)

        if args.once:
            print(f"\n{'ref':<10} {'score':<8} {'status':<30} description")
            for r in rows[:6]:
                print(
                    f"{r['ref']:<10} {(r['publicScore'] or '-'):<8} "
                    f"{r['status'].replace('SubmissionStatus.', ''):<30} {r['description'][:60]}"
                )
            return

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
