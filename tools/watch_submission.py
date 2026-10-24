"""Poll a competition SUBMISSION (ref) until it is scored.

Why this exists: in a Kaggle **code** competition the hidden rerun is NOT the kernel's own
session. `kaggle kernels status <slug>` stays COMPLETE from the earlier dry run, so
`tools/watch_kernel.py` / `watch_and_pull.py` return instantly and tell you nothing.

The thing to poll is the submission row: `kaggle competitions submissions --csv`, matched
by ref. It goes PENDING -> COMPLETE, and only then does `publicScore` get filled in.

Usage
    python tools/watch_submission.py 55871971
    python tools/watch_submission.py 55871971 --interval 300 --timeout-h 9

Also prints a comparison against the currently-selected submission so you can see at a
glance whether the new one beat it.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import time

PY = sys.executable
COMP = "rsna-knee-abnormality-detection"


def env() -> dict:
    """Strip the sandbox proxy - it breaks some Kaggle API endpoints."""
    e = dict(os.environ)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        e.pop(k, None)
        e.pop(k.lower(), None)
    return e


def fetch() -> list[dict]:
    p = subprocess.run(
        [PY, "-m", "kaggle", "competitions", "submissions", "-c", COMP, "--csv"],
        env=env(), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or f"exit {p.returncode}").strip()[-300:])
    rows = list(csv.DictReader(io.StringIO(p.stdout)))
    return [r for r in rows if r.get("ref")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--timeout-h", type=float, default=9.5)
    args = ap.parse_args()

    t0 = time.time()
    deadline = t0 + args.timeout_h * 3600
    print(f"watching submission {args.ref} (poll {args.interval}s, timeout {args.timeout_h}h)",
          flush=True)

    seen_status = ""
    while True:
        try:
            rows = fetch()
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] fetch error: {e}", flush=True)
            time.sleep(args.interval)
            continue

        row = next((r for r in rows if r["ref"] == args.ref), None)
        if row is None:
            print(f"[{time.strftime('%H:%M:%S')}] ref {args.ref} not found yet", flush=True)
        else:
            st = (row.get("status") or "").replace("SubmissionStatus.", "")
            sc = row.get("publicScore") or ""
            tag = f"{st}" + (f"  publicScore={sc}" if sc else "")
            if tag != seen_status:
                print(f"[{time.strftime('%H:%M:%S')}] elapsed {(time.time()-t0)/3600:.2f}h | {tag}",
                      flush=True)
                seen_status = tag
            if st not in ("PENDING", "RUNNING", "SUBMITTED", ""):
                print(f"\n=== submission {args.ref} -> {st} ===", flush=True)
                print(f"  publicScore : {sc or '(none)'}"
                      f"   private: {row.get('privateScore') or '(none)'}", flush=True)
                print(f"  description: {row.get('description','')[:160]}", flush=True)

                # compare against every scored submission we have
                scored = []
                for r in rows:
                    s = r.get("publicScore")
                    if s:
                        try:
                            scored.append((float(s), r["ref"], r.get("description", "")[:60]))
                        except ValueError:
                            pass
                scored.sort(reverse=True)
                print("\n  leaderboard among our submissions:", flush=True)
                for s, ref, desc in scored[:5]:
                    mark = " <-- this run" if ref == args.ref else ""
                    print(f"    {s:.4f}  {ref}  {desc}{mark}", flush=True)
                return 0

        if time.time() > deadline:
            print("timeout reached; submission may still be running", flush=True)
            return 2
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
