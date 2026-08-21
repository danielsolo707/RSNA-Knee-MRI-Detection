"""Overnight helper for V22 submit + V23 dry-run.

Use Kaggle CLI only. Never submits V23 here — that waits for a V22 score.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ASUS\Desktop\DanielEmpire\ML\RSNA Knee Abnormality Detection")
V22_OUT = ROOT / "artifacts" / "v22_remote" / "kernel_output_v2"
V22_KERNEL = "danielsolo1770/rsna-knee-v22-dinov3-e10-060"
V23_DIR = ROOT / "kaggle" / "submission_v23_dinov3_e10_e11"
COMP = "rsna-knee-abnormality-detection"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("KAGGLE_API_TOKEN", None)
    return env


def run(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, env=_env(), text=True, capture_output=True, check=check)


def main() -> int:
    V22_OUT.mkdir(parents=True, exist_ok=True)
    st = run(["kaggle", "kernels", "status", V22_KERNEL])
    print(st.stdout)
    if "COMPLETE" not in st.stdout:
        print("V22 kernel is not COMPLETE; refusing to submit.", file=sys.stderr)
        return 2

    out = run(
        [
            "kaggle",
            "kernels",
            "output",
            V22_KERNEL,
            "-p",
            str(V22_OUT),
            "-o",
            "--file-pattern",
            ".*audit.*|submission.csv",
        ]
    )
    print(out.stdout)
    if out.stderr:
        print(out.stderr)

    audit_path = V22_OUT / "rad_e10_audit.json"
    receipt_path = V22_OUT / "v40_runtime_audit.json"
    if not audit_path.is_file() or not receipt_path.is_file():
        print("missing audits", audit_path.exists(), receipt_path.exists(), file=sys.stderr)
        return 3
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    print("e10", audit.get("status"), audit.get("configuration"))
    print("receipt", receipt.get("status"))
    if audit.get("status") != "CANDIDATE_SELECTED" or audit.get("configuration") != "uniform_060":
        print("E10 did not promote uniform_060; not submitting.", file=sys.stderr)
        return 4
    if "VALID_DINOV3_E10_HYBRID" not in str(receipt.get("status", "")):
        print("runtime receipt invalid; not submitting.", file=sys.stderr)
        return 5

    sub = run(
        [
            "kaggle",
            "competitions",
            "submit",
            "-c",
            COMP,
            "-k",
            V22_KERNEL,
            "-v",
            "2",
            "-f",
            "submission.csv",
            "-m",
            "V22 v2: DINOv2+Tony+DINOv3 + E10 uniform_060, 32000s cap, dry-run CANDIDATE_SELECTED",
        ]
    )
    print(sub.stdout)
    print(sub.stderr)

    push = run(
        [
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(V23_DIR),
            "-t",
            "32000",
        ]
    )
    print(push.stdout)
    print(push.stderr)
    print("V22 v2 submitted and V23 dry-run pushed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
