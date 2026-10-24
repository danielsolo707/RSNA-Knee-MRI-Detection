"""Watch a Kaggle kernel, then pull and validate its outputs when it stops.

Designed to be left running in the background for the multi-hour inference/training
kernels this project depends on. When the kernel reaches a terminal state it:

  1. pulls the log (always - works even for ERROR/CANCELLED runs)
  2. pulls the rest of the outputs if the run COMPLETED
  3. validates submission.csv (row count, study order, finite values) if present
  4. prints a short summary

Notes learned the hard way:
  * The sandbox HTTP proxy breaks `ListKernelSessionOutput`; every Kaggle call here
    runs with the proxy env vars stripped.
  * Kernel outputs are committed however the run ends - including CANCELLED
    (verified on V34), so the log pull is worth doing even on failure.

Usage
    python tools/watch_and_pull.py danielsolo1770/rsna-knee-v37-complement
    python tools/watch_and_pull.py <slug> --interval 180 --timeout-h 9 --no-pull
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

TERMINAL_OK = ("COMPLETE",)
TERMINAL_BAD = ("ERROR", "CANCEL", "FAILED", "TIMEOUT")


def env() -> dict:
    """Strip the sandbox proxy - it breaks some Kaggle API endpoints.

    Also strip KAGGLE_API_TOKEN: on this machine the env token is short-lived
    and expires mid-run (a 12 h watcher died after ~40 min with a credentials
    error while the kernel ran on). The ~/.kaggle file credentials keep working,
    so forcing the CLI onto file auth makes long watches self-sufficient.
    """
    e = dict(os.environ)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "KAGGLE_API_TOKEN"):
        e.pop(k, None)
        e.pop(k.lower(), None)
    return e


def run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, env=env(), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def status(slug: str) -> str:
    p = run([PY, "-m", "kaggle", "kernels", "status", slug])
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or f"exit {p.returncode}").strip()[-300:])
    return (p.stdout or "").strip()


def pull(slug: str, out: Path, pattern: str | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    argv = [PY, "-m", "kaggle", "kernels", "output", slug, "-p", str(out), "-o"]
    if pattern:
        argv += ["--file-pattern", pattern]
    p = run(argv)
    sys.stdout.write(p.stdout or "")
    if p.returncode:
        sys.stdout.write(p.stderr or "")


def validate_sub(sub: Path) -> str:
    p = run([PY, str(REPO / "tools" / "validate_submission.py"),
             str(sub), str(REPO / "data" / "raw" / "test.csv")])
    return (p.stdout or p.stderr or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--interval", type=int, default=180, help="seconds between polls")
    ap.add_argument("--timeout-h", type=float, default=9.5)
    ap.add_argument("--no-pull", action="store_true")
    args = ap.parse_args()

    slug = args.slug
    name = slug.rstrip("/").split("/")[-1]
    out = REPO / "artifacts" / f"remote_{name}"
    deadline = time.time() + args.timeout_h * 3600
    t0 = time.time()
    last = ""
    errs = 0

    print(f"watching {slug} (poll {args.interval}s, timeout {args.timeout_h}h)", flush=True)

    while True:
        try:
            s = status(slug)
        except Exception as e:
            errs += 1
            # Back off on repeated errors: a fixed 3-min hammer kept the V43 watch
            # stuck on an auth/rate-limit failure; longer gaps let it recover.
            wait = min(args.interval * (2 ** min(errs, 4)), 900)
            print(f"[{time.strftime('%H:%M:%S')}] status error x{errs} "
                  f"(retry in {wait}s): {e}", flush=True)
            if time.time() + wait > deadline:
                print("watch timeout reached during backoff; giving up", flush=True)
                return 2
            time.sleep(wait)
            continue
        errs = 0

        if s != last:
            print(f"[{time.strftime('%H:%M:%S')}] elapsed {(time.time()-t0)/3600:.2f}h | {s}",
                  flush=True)
            last = s

        u = s.upper()
        if any(k in u for k in TERMINAL_OK):
            state = "COMPLETE"
            break
        if any(k in u for k in TERMINAL_BAD):
            state = "FAILED"
            break

        if time.time() > deadline:
            print("watch timeout reached; giving up (kernel may still be running)", flush=True)
            return 2
        time.sleep(args.interval)

    print(f"\n=== {slug} -> {state} after {(time.time()-t0)/3600:.2f}h ===", flush=True)

    if args.no_pull:
        return 0 if state == "COMPLETE" else 1

    # Log first - it is cheap and available even on failure.
    print("\n--- pulling log ---", flush=True)
    pull(slug, out, r"\.(log)$")

    if state != "COMPLETE":
        print("run did not complete; log is the useful artifact", flush=True)
        _tail_log(out / f"{name}.log")
        return 1

    print("\n--- pulling outputs ---", flush=True)
    pull(slug, out, None)

    files = sorted(p for p in out.iterdir() if p.is_file())
    print(f"\n{len(files)} file(s) in {out}:", flush=True)
    for p in files:
        print(f"  {p.stat().st_size/1e6:10.2f} MB  {p.name}", flush=True)

    sub = out / "submission.csv"
    if sub.is_file():
        print("\n--- validating submission.csv ---", flush=True)
        print(validate_sub(sub), flush=True)
    else:
        print("\nno submission.csv (expected for a training-only kernel)", flush=True)

    prog = [p for p in files if p.name.startswith("progress_")]
    for p in prog:
        print(f"\n--- {p.name} ---", flush=True)
        print(p.read_text(encoding="utf-8"), flush=True)

    _tail_log(out / f"{name}.log")
    return 0


def _tail_log(p: Path, n: int = 25) -> None:
    if not p.is_file():
        return
    lines = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip().rstrip(",")
        if not ln:
            continue
        try:
            lines.append(json.loads(ln).get("data", "").rstrip())
        except Exception:
            lines.append(ln)
    lines = [l for l in lines if l]
    print(f"\n--- last {n} log lines ---", flush=True)
    for l in lines[-n:]:
        print("  " + l[:180], flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
