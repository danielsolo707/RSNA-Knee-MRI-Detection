"""Poll one competition submission until it leaves PENDING. Prints DONE or FAILED."""
from __future__ import annotations

import csv
import io
import subprocess
import sys
import time

PY = sys.executable
REF = sys.argv[1] if len(sys.argv) > 1 else "55771082"
COMP = "rsna-knee-abnormality-detection"


def rows():
    proc = subprocess.run(
        [PY, "-m", "kaggle", "competitions", "submissions", "-c", COMP, "--csv"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"kaggle exit {proc.returncode}")
    return list(csv.DictReader(io.StringIO(proc.stdout)))


def main():
    while True:
        try:
            recs = rows()
        except Exception:
            time.sleep(60)
            continue
        hit = next((r for r in recs if str(r.get("ref")) == str(REF)), None)
        if hit is None:
            time.sleep(60)
            continue
        status = (hit.get("status") or "").upper()
        score = hit.get("publicScore") or ""
        if "PENDING" in status:
            time.sleep(90)
            continue
        if "COMPLETE" in status and score:
            print(f"DONE ref={REF} score={score}", flush=True)
            return 0
        print(f"FAILED ref={REF} status={status} score={score}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
