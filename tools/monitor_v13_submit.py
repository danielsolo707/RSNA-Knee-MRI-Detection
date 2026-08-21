#!/usr/bin/env python3
"""Monitor V13 dry-run, submit only on verified promotion, record LB result."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "danielsolo1770/rsna-knee-v13-three-arm-gated"
VERSION = os.environ.get("V13_KERNEL_VERSION", "2")
COMP = "rsna-knee-abnormality-detection"
MESSAGE = "V13 gated DINO+E9 Rad-FS+E11 Rad-NOFS fixed 0.30/0.35/0.35"
OUT = ROOT / "artifacts" / "v13_remote"
STATE = OUT / "monitor_state.json"
LOG = OUT / "monitor.log"
TOKEN_FILE = Path.home() / ".kaggle" / "access_token"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    line = f"[{timestamp()}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_state() -> dict:
    if not STATE.is_file():
        return {}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(**updates) -> dict:
    state = read_state()
    state.update(updates)
    if updates.get("stage") == "monitor_started":
        state.pop("error", None)
    state["updated_at"] = timestamp()
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state


def run(args: list[str], timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Prefer the Kaggle CLI's cached OAuth session.  Do not inject the legacy
    # access_token file here: a revoked/expired KGAT value in the environment
    # overrides a valid OAuth login and breaks long-running monitors.
    env.pop("KAGGLE_API_TOKEN", None)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed rc={proc.returncode}: {' '.join(args)}\n"
            f"stdout={proc.stdout[-3000:]}\nstderr={proc.stderr[-3000:]}"
        )
    return proc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kernel_status() -> str:
    proc = run(["kaggle", "kernels", "status", f"{SLUG}/{VERSION}"], timeout=120)
    text = proc.stdout + proc.stderr
    for status in ("COMPLETE", "ERROR", "CANCELLED", "RUNNING", "QUEUED"):
        if status in text.upper():
            return status
    return text.strip() or "UNKNOWN"


def wait_for_kernel(max_seconds: int = 6 * 3600) -> str:
    started = time.time()
    previous = None
    while time.time() - started < max_seconds:
        status = kernel_status()
        if status != previous:
            log(f"kernel v{VERSION} status={status}")
            save_state(stage="kernel_wait", kernel_status=status)
            previous = status
        if status in {"COMPLETE", "ERROR", "CANCELLED"}:
            return status
        time.sleep(90)
    raise TimeoutError("kernel did not finish within six hours")


def find_unique(name: str) -> Path:
    matches = [path for path in OUT.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}, found {[str(path) for path in matches]}")
    return matches[0]


def validate_submission(path: Path, test_path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    with test_path.open("r", newline="", encoding="utf-8-sig") as handle:
        test_rows = list(csv.DictReader(handle))
    expected_columns = ["StudyInstanceUID", *TARGETS]
    if columns != expected_columns:
        raise RuntimeError(f"submission columns mismatch: {columns}")
    expected_ids = [row["StudyInstanceUID"] for row in test_rows]
    ids = [row["StudyInstanceUID"] for row in rows]
    if ids != expected_ids or len(ids) != len(set(ids)):
        raise RuntimeError("submission ID order/uniqueness mismatch")
    for row in rows:
        for target in TARGETS:
            value = float(row[target])
            if not (0.0 <= value <= 1.0):
                raise RuntimeError(f"invalid value for {target}: {value}")


def download_and_verify() -> tuple[Path, dict]:
    downloads = OUT / "kernel_output"
    if downloads.exists():
        shutil.rmtree(downloads)
    downloads.mkdir(parents=True)
    proc = run(
        ["kaggle", "kernels", "output", f"{SLUG}/{VERSION}", "-p", str(downloads), "-o"],
        timeout=900,
    )
    log(f"kernel outputs downloaded: {(proc.stdout + proc.stderr).strip()[-1000:]}")
    audit_path = find_unique("v13_three_arm_audit.json")
    submission = find_unique("submission.csv")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    gate = audit.get("gate") or {}
    failures = []
    if audit.get("status") != "V13_SELECTED":
        failures.append(f"status={audit.get('status')}")
    if audit.get("promote") is not True:
        failures.append("promote is not true")
    if gate.get("gate_pass") is not True:
        failures.append("gate_pass is not true")
    if int(gate.get("positive_folds", -1)) != 5:
        failures.append(f"positive_folds={gate.get('positive_folds')}")
    if float(gate.get("gold_gain", -1)) < 0.005:
        failures.append(f"gold_gain={gate.get('gold_gain')}")
    if float(gate.get("bootstrap_win", -1)) < 0.80:
        failures.append(f"bootstrap_win={gate.get('bootstrap_win')}")
    if float(gate.get("soft_delta", -99)) < -0.005:
        failures.append(f"soft_delta={gate.get('soft_delta')}")
    observed_hash = sha256(submission)
    if audit.get("selected_sha256") != observed_hash:
        failures.append(
            f"submission hash mismatch audit={audit.get('selected_sha256')} observed={observed_hash}"
        )
    validate_submission(submission, ROOT / "data" / "raw" / "test.csv")
    if failures:
        raise RuntimeError("V13 dry-run verification failed: " + "; ".join(failures))
    save_state(
        stage="dry_run_verified",
        audit=str(audit_path),
        submission=str(submission),
        submission_sha256=observed_hash,
        dry_run_status=audit.get("status"),
        gold_gain=gate.get("gold_gain"),
        bootstrap_win=gate.get("bootstrap_win"),
        positive_folds=gate.get("positive_folds"),
        elapsed_seconds=audit.get("elapsed_seconds"),
    )
    log(
        f"dry-run verified: status={audit['status']} gain={gate.get('gold_gain'):.6f} "
        f"bootstrap={gate.get('bootstrap_win'):.4f} hash={observed_hash[:12]}"
    )
    return submission, audit


def submissions() -> list[dict[str, str]]:
    proc = run(["kaggle", "competitions", "submissions", "-c", COMP, "--csv"], timeout=180)
    return list(csv.DictReader(proc.stdout.splitlines()))


def submit() -> str:
    before = {row.get("ref", "") for row in submissions()}
    proc = run(
        [
            "kaggle", "competitions", "submit", "-c", COMP,
            "-k", SLUG, "-f", "submission.csv", "-v", VERSION, "-m", MESSAGE,
        ],
        timeout=300,
    )
    log(f"submit response: {(proc.stdout + proc.stderr).strip()}")
    for _ in range(20):
        rows = submissions()
        fresh = [row for row in rows if row.get("ref", "") not in before]
        matching = [row for row in fresh if row.get("description") == MESSAGE]
        candidate = matching[0] if matching else (fresh[0] if fresh else None)
        if candidate:
            ref = candidate.get("ref", "")
            save_state(stage="submitted", submission_ref=ref, submission_row=candidate)
            log(f"competition submission created: ref={ref}")
            return ref
        time.sleep(30)
    raise RuntimeError("submission command returned but no new submission row appeared")


def wait_for_submission(ref: str, max_seconds: int = 8 * 3600) -> dict[str, str]:
    started = time.time()
    previous = None
    while time.time() - started < max_seconds:
        rows = submissions()
        row = next((item for item in rows if item.get("ref") == ref), None)
        if row is None:
            time.sleep(60)
            continue
        status = row.get("status", "")
        if status != previous:
            log(f"submission {ref} status={status} public={row.get('publicScore', '')}")
            save_state(stage="submission_wait", submission_ref=ref, submission_row=row)
            previous = status
        upper = status.upper()
        if "COMPLETE" in upper or "ERROR" in upper or "CANCEL" in upper:
            return row
        time.sleep(90)
    raise TimeoutError("submission did not finish within eight hours")


def live_leaderboard() -> dict:
    lb_dir = OUT / "leaderboard"
    if lb_dir.exists():
        shutil.rmtree(lb_dir)
    lb_dir.mkdir(parents=True)
    run(["kaggle", "competitions", "leaderboard", "-c", COMP, "-d", "-p", str(lb_dir)], timeout=300)
    archives = list(lb_dir.glob("*.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"expected leaderboard zip, got {archives}")
    with zipfile.ZipFile(archives[0]) as archive:
        member = archive.namelist()[0]
        text = archive.read(member).decode("utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    rows.sort(key=lambda row: float(row.get("Score") or -1), reverse=True)
    daniel = next(
        (dict(row, computed_rank=index) for index, row in enumerate(rows, 1)
         if row.get("TeamId") == "16697840" or row.get("TeamName") == "Daniel solo"),
        None,
    )
    cutoff50 = rows[49].get("Score") if len(rows) >= 50 else None
    leader = rows[0].get("Score") if rows else None
    return {"team": daniel, "top50_cutoff": cutoff50, "leader": leader, "team_count": len(rows)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = read_state()
    resume_ref = None
    if (
        prior.get("version") == VERSION
        and prior.get("message") == MESSAGE
        and prior.get("submission_ref")
    ):
        resume_ref = str(prior["submission_ref"])
    save_state(
        stage="monitor_started",
        slug=SLUG,
        version=VERSION,
        message=MESSAGE,
        submission_ref=resume_ref,
        submission_row=prior.get("submission_row") if resume_ref else None,
        leaderboard=prior.get("leaderboard") if resume_ref else None,
    )
    log("V13 autonomous monitor started")
    if resume_ref:
        ref = resume_ref
        log(f"resuming existing competition submission ref={ref}; no duplicate submit")
    else:
        status = wait_for_kernel()
        if status != "COMPLETE":
            save_state(stage="kernel_failed", kernel_status=status)
            log(f"STOP: dry-run kernel ended with {status}; no submission sent")
            return 2
        try:
            download_and_verify()
        except Exception as error:
            save_state(stage="verification_failed", error=f"{type(error).__name__}: {error}")
            log(f"STOP: {type(error).__name__}: {error}; no submission sent")
            return 3
        ref = submit()
    row = wait_for_submission(ref)
    public = row.get("publicScore", "")
    try:
        board = live_leaderboard() if public else None
    except Exception as error:
        board = {"error": f"{type(error).__name__}: {error}"}
    final_stage = "complete" if "COMPLETE" in row.get("status", "").upper() else "submission_failed"
    save_state(stage=final_stage, submission_ref=ref, submission_row=row, leaderboard=board)
    log(f"FINAL: ref={ref} status={row.get('status')} public={public} leaderboard={board}")
    return 0 if final_stage == "complete" else 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        OUT.mkdir(parents=True, exist_ok=True)
        save_state(stage="monitor_exception", error=f"{type(error).__name__}: {error}")
        log(f"FATAL: {type(error).__name__}: {error}")
        raise
