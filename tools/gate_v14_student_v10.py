from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
UID = "StudyInstanceUID"
COLUMNS = [UID, *TARGETS]
PARENT_SHA256 = "29a1ae14b5610d5056b9ce06d19442d7378a3915e31c67377d615b99477e4e29"
STUDENT_RUNTIME_SHA256 = "6c7c6f45d567173646a5e55169ca0cb7685204d32c0a0d7f124fbbe0a89f52f8"
STUDENT_MODEL_SHA256 = {
    "student_seed_20260873.pt": "9ad2edfbde91f31673104a01fb06d97d5424790aa2febbdc960011771946f69a",
    "student_seed_20260897.pt": "d55c767cfcdc1204497027e53212ded38c7a9dff866dd33ca014b656ed4ae0a5",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    out = np.zeros_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        out[:, j] = pd.Series(matrix[:, j]).rank(method="average", pct=True).to_numpy()
    return out


def validate(frame: pd.DataFrame, expected_ids: list[str], label: str) -> None:
    if frame.columns.tolist() != COLUMNS:
        raise RuntimeError(f"{label}: schema drift: {frame.columns.tolist()}")
    ids = frame[UID].astype(str).tolist()
    if ids != expected_ids or len(ids) != len(set(ids)):
        raise RuntimeError(f"{label}: ID set/order/uniqueness drift")
    values = frame[TARGETS].to_numpy(float)
    if values.shape != (len(expected_ids), len(TARGETS)):
        raise RuntimeError(f"{label}: shape drift {values.shape}")
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise RuntimeError(f"{label}: non-finite/out-of-range values")


def main() -> int:
    parent_path = ROOT / "research/v14_top20_cli/own_kernel_outputs/v10/submission.csv"
    student_path = ROOT / "research/v14_top20_cli/prvsiyan_knees_output/submission.csv"
    student_receipt_path = ROOT / "research/v14_top20_cli/prvsiyan_knees_output/resnet18_runtime_receipt.json"
    runtime_path = ROOT / "research/v14_top20_cli/resnet18_student/inference_runtime.py"
    test_path = ROOT / "data/raw/test.csv"
    out_dir = ROOT / "artifacts/v14_student_v10_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "V14_GATE_NOT_PROMOTED",
        "candidate": "0.50 rank(remote_v10) + 0.50 rank(student)",
        "parent_public_score": 0.906,
        "student_direct_expert_oof_available": False,
        "same_58_diagnostic_allowed_for_promotion": False,
        "submit_allowed": False,
        "blocking_reason": "No direct expert-label OOF exists for the proposed Student blend; remote runtime is execution evidence only.",
    }
    try:
        for p, expected, label in [
            (parent_path, PARENT_SHA256, "parent"),
            (runtime_path, STUDENT_RUNTIME_SHA256, "student_runtime"),
        ]:
            if not p.is_file():
                raise FileNotFoundError(p)
            observed = sha256(p)
            if observed != expected:
                raise RuntimeError(f"{label} SHA mismatch: {observed} != {expected}")
        receipt["parent_sha256"] = sha256(parent_path)
        receipt["student_runtime_sha256"] = sha256(runtime_path)
        test = pd.read_csv(test_path, dtype={UID: str})
        expected_ids = test[UID].astype(str).tolist()
        parent = pd.read_csv(parent_path, dtype={UID: str})
        student = pd.read_csv(student_path, dtype={UID: str})
        validate(parent, expected_ids, "parent")
        validate(student, expected_ids, "student")
        receipt["test_rows"] = len(expected_ids)
        receipt["parent_rows"] = len(parent)
        receipt["student_rows"] = len(student)
        runtime_receipt = json.loads(student_receipt_path.read_text())
        receipt["student_runtime_status"] = runtime_receipt.get("status")
        receipt["student_runtime_models"] = runtime_receipt.get("models")
        if runtime_receipt.get("status") != "R18_CPU_INFERENCE_COMPLETE":
            raise RuntimeError(f"student runtime status is not complete: {runtime_receipt.get('status')}")
        for record in runtime_receipt.get("models", []):
            name = Path(str(record.get("path", ""))).name
            if name in STUDENT_MODEL_SHA256 and record.get("sha256") != STUDENT_MODEL_SHA256[name]:
                raise RuntimeError(f"student model receipt hash mismatch for {name}")
        parent_rank = rank(parent[TARGETS].to_numpy(float))
        student_rank = rank(student[TARGETS].to_numpy(float))
        final = 0.50 * parent_rank + 0.50 * student_rank
        candidate = pd.DataFrame(final, columns=TARGETS)
        candidate.insert(0, UID, expected_ids)
        validate(candidate, expected_ids, "candidate")
        candidate_path = out_dir / "submission_v14_student_v10.csv"
        candidate.to_csv(candidate_path, index=False)
        receipt["candidate_sha256"] = sha256(candidate_path)
        receipt["candidate_path"] = str(candidate_path)
        receipt["rank_changed_targets"] = [
            target for j, target in enumerate(TARGETS)
            if tuple(np.argsort(-final[:, j], kind="stable"))
            != tuple(np.argsort(-parent_rank[:, j], kind="stable"))
        ]
        receipt["candidate_values"] = final.tolist()
        receipt["parent_values"] = parent[TARGETS].to_numpy(float).tolist()
        receipt["student_values"] = student[TARGETS].to_numpy(float).tolist()
        # This is deliberately a dry-run receipt, not a promotion receipt.
        receipt["status"] = "V14_RUNTIME_VALID_BUT_PROMOTION_BLOCKED"
    except Exception as exc:
        receipt["status"] = "V14_GATE_ERROR"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    out = out_dir / "v14_student_v10_gate.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] != "V14_GATE_ERROR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
