"""Poll a kernel until it leaves RUNNING. Prints DONE or FAILED."""
from __future__ import annotations

import subprocess
import sys
import time

PY = sys.executable
SLUG = sys.argv[1] if len(sys.argv) > 1 else "danielsolo1770/rsna-knee-v34-raptor-f0"


def status() -> str:
    proc = subprocess.run(
        [PY, "-m", "kaggle", "kernels", "status", SLUG],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or f"exit {proc.returncode}").strip())
    return (proc.stdout or "").strip()


def main() -> int:
    while True:
        try:
            s = status()
        except Exception:
            time.sleep(90)
            continue
        u = s.upper()
        if "COMPLETE" in u:
            print(f"DONE {SLUG}", flush=True)
            return 0
        if any(k in u for k in ("ERROR", "CANCEL", "FAILED")):
            print(f"FAILED {SLUG} {s}", flush=True)
            return 1
        time.sleep(90)


if __name__ == "__main__":
    raise SystemExit(main())
