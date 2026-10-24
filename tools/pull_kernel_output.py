"""Pull outputs from a Kaggle kernel - including CANCELLED, errored or timed-out ones.

Why this exists
---------------
Kaggle commits `/kaggle/working` when a notebook stops, however it stops. Verified on
`danielsolo1770/rsna-knee-v34-raptor-f0`: status `CANCEL_ACKNOWLEDGED`, yet its
`raptor_ft_v34f0.pt` (293 MB), `.json`, `.npz` and full log were all downloaded
afterwards. That is what makes checkpoint-and-resume a viable strategy.

Two important behaviours discovered while writing this:
  * The sandbox sets HTTP(S)_PROXY to a local port that breaks some Kaggle endpoints
    (`ListKernelSessionOutput` fails with ProxyError/RemoteDisconnected). Unsetting the
    proxy env vars fixes it. This script does that automatically.
  * Use `--file-pattern` to grab only small files (progress json, receipts) when you
    just want to check on a run without paying for a 293 MB download.

Usage
    python tools/pull_kernel_output.py <owner/slug>
    python tools/pull_kernel_output.py <owner/slug> --small-only
    python tools/pull_kernel_output.py <owner/slug> --out artifacts/x --pattern '\\.pt$'
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

# Small, cheap-to-download files: progress + receipts + the log.
SMALL_PATTERN = r"\.(json|npz|csv|log)$"


def clean_env() -> dict:
    """Kaggle's API breaks behind the sandbox proxy for some endpoints - drop it."""
    env = dict(os.environ)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        env.pop(k, None)
        env.pop(k.lower(), None)
    return env


def run(argv: list[str]) -> int:
    r = subprocess.run(argv, env=clean_env(), capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode:
        sys.stdout.write(r.stderr)
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", help="owner/slug, e.g. danielsolo1770/rsna-knee-v36-public-init")
    ap.add_argument("--out", default=None,
                    help="destination dir (default artifacts/remote_<slug>)")
    ap.add_argument("--pattern", default=None, help="regex of files to fetch")
    ap.add_argument("--small-only", action="store_true",
                    help=f"only json/npz/csv/log (progress + receipts). cheap.")
    args = ap.parse_args()

    slug = args.kernel.rstrip("/").split("/")[-1]
    out = Path(args.out) if args.out else REPO / "artifacts" / f"remote_{slug}"
    out.mkdir(parents=True, exist_ok=True)

    pattern = args.pattern or (SMALL_PATTERN if args.small_only else None)
    argv = [PY, "-m", "kaggle", "kernels", "output", args.kernel, "-p", str(out), "-o"]
    if pattern:
        argv += ["--file-pattern", pattern]

    print(f"pulling {args.kernel} -> {out}" + (f"  [pattern {pattern}]" if pattern else ""))
    rc = run(argv)
    if rc:
        return rc

    files = sorted(p for p in out.iterdir() if p.is_file())
    print(f"\n{len(files)} file(s):")
    for p in files:
        print(f"  {p.stat().st_size/1e6:10.2f} MB  {p.name}")

    prog = [p for p in files if p.name.startswith("progress_")]
    for p in prog:
        print(f"\n--- {p.name} ---")
        print(p.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
