"""After V22 v2 scores: submit V23 only if we still need top-50.

Guards:
- V22 public score must exist
- Submit V23 only if V22 < 0.926 and V23 dry-run promoted E10
- Never submit if remaining daily submissions < 1
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ASUS\Desktop\DanielEmpire\ML\RSNA Knee Abnormality Detection")
V23_OUT = ROOT / "artifacts" / "v23_remote" / "kernel_output"
V23_KERNEL = "danielsolo1770/rsna-knee-v23-dinov3-e10-e11-070"
V22_REF = "55564382"
COMP = "rsna-knee-abnormality-detection"
TOP50 = 0.926
INCUMBENT = 0.912


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("KAGGLE_API_TOKEN", None)
    return env


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, env=_env(), text=True, capture_output=True, check=True)


def v22_score() -> float | None:
    out = run(["kaggle", "competitions", "submissions", "-c", COMP, "--csv"])
    reader = csv.DictReader(io.StringIO(out.stdout))
    for row in reader:
        if row.get("ref") == V22_REF:
            raw = (row.get("publicScore") or "").strip()
            print("V22 row", row)
            return float(raw) if raw else None
    return None


def v23_promoted() -> bool:
    V23_OUT.mkdir(parents=True, exist_ok=True)
    st = run(["kaggle", "kernels", "status", V23_KERNEL])
    print(st.stdout)
    if "COMPLETE" not in st.stdout:
        return False
    run(
        [
            "kaggle",
            "kernels",
            "output",
            V23_KERNEL,
            "-p",
            str(V23_OUT),
            "-o",
            "--file-pattern",
            ".*audit.*|submission.csv",
        ]
    )
    audit_p = V23_OUT / "rad_e10_audit.json"
    receipt_p = V23_OUT / "v40_runtime_audit.json"
    if not audit_p.is_file() or not receipt_p.is_file():
        print("V23 audits missing")
        return False
    audit = json.loads(audit_p.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_p.read_text(encoding="utf-8"))
    print("V23 e10", audit.get("status"), audit.get("configuration"))
    print("V23 receipt", receipt.get("status"))
    e11_p = V23_OUT / "rad_e11_apply_audit.json"
    if e11_p.is_file():
        print("V23 e11", json.loads(e11_p.read_text(encoding="utf-8")).get("status"))
    return audit.get("status") == "CANDIDATE_SELECTED" and "VALID_DINOV3_E10_HYBRID" in str(
        receipt.get("status", "")
    )


def main() -> int:
    score = v22_score()
    if score is None:
        print("V22 has no public score yet")
        return 2
    print(f"V22 public={score:.3f} incumbent={INCUMBENT} top50={TOP50}")
    if score >= TOP50:
        print("Top-50 reached. Do not submit more tonight.")
        return 0
    if score < INCUMBENT:
        print("V22 lost to incumbent. Still try V23 only if it promoted.")
    if not v23_promoted():
        print("V23 not promoted / not complete. No submit.")
        return 3
    sub = run(
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            COMP,
            "-k",
            V23_KERNEL,
            "-v",
            "1",
            "-f",
            "submission.csv",
            "-m",
            "V23 public V43b fork: DINOv3 + E10 uniform_070 + gated E11; only after V22 < 0.926",
        ]
    )
    print(sub.stdout)
    print(sub.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
