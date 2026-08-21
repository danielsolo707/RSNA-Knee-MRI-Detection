from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle" / "submission_v14_student_v10"
OUT_NB = OUT_DIR / "rsna-knee-v14-student-v10-gated.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
UID = "StudyInstanceUID"
EXPECTED_COLUMNS = [UID, *TARGETS]
PARENT_SHA256 = "29a1ae14b5610d5056b9ce06d19442d7378a3915e31c67377d615b99477e4e29"
STUDENT_SHA256 = {
    "student_seed_20260873.pt": "9ad2edfbde91f31673104a01fb06d97d5424790aa2febbdc960011771946f69a",
    "student_seed_20260897.pt": "d55c767cfcdc1204497027e53212ded38c7a9dff866dd33ca014b656ed4ae0a5",
}

RUNTIME = r'''
# === V14: exact remote V10 parent + independent ResNet18 student ===============
# Fixed pre-registered recipe: 0.50 rank(V10) + 0.50 rank(Student).
# The parent is materialized before the challenger. Any challenger failure restores it.
# This is an experiment gate; no competition submission is created in-kernel.
import gc as _v14_gc
import hashlib as _v14_hashlib
import json as _v14_json
import os as _v14_os
import shutil as _v14_shutil
import time as _v14_time
import traceback as _v14_traceback
from pathlib import Path as _V14Path

import numpy as _v14_np
import pandas as _v14_pd
import torch as _v14_torch

_V14_TARGETS = __V14_TARGETS__
_V14_UID = "StudyInstanceUID"
_V14_EXPECTED_COLUMNS = [_V14_UID, *_V14_TARGETS]
_V14_PARENT_SHA256 = __V14_PARENT_SHA256__
_V14_STUDENT_SHA256 = __V14_STUDENT_SHA256__
_V14_PARENT_NAME = "submission_v10_parent.csv"
_V14_PRIMARY = _V14Path("/kaggle/working/submission.csv")
_V14_PARENT = _V14Path("/kaggle/working") / _V14_PARENT_NAME
_V14_STUDENT = _V14Path("/kaggle/working/submission_v14_student.csv")
_V14_AUDIT = _V14Path("/kaggle/working/v14_student_v10_audit.json")
_V14_T0 = float(globals().get("T0", _v14_time.time()))
_V14_ALIVE_LIMIT = 4.5 * 3600.0
# Define these before the guarded body so an early parent/runtime failure can
# always produce a valid audit instead of masking the root cause with NameError.
_V14_TEST_ROOT = None
_v14_test = None
_v14_test_ids = None
_V14_AUDIT_PAYLOAD = {
    "status": "V14_NOT_STARTED",
    "candidate": "0.50 rank(remote_v10) + 0.50 rank(resnet18_student)",
    "parent_reference": "danielsolo1770/rsna-knee-v10-rad-cross-gated",
    "parent_public_score": 0.906,
    "evidence_boundary": "strict non-gold student firewall; no direct expert-label OOF for student; experiment only",
}


def _v14_sha(path):
    digest = _v14_hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _v14_log(msg):
    try:
        log("[V14] " + str(msg))
    except Exception:
        print("[V14] " + str(msg), flush=True)


def _v14_deadline(where):
    elapsed = max(0.0, _v14_time.time() - _V14_T0)
    if elapsed > _V14_ALIVE_LIMIT:
        raise TimeoutError(f"V14 deadline exceeded at {where}: {elapsed:.1f}s")


def _v14_validate(frame, expected_ids):
    if frame.columns.tolist() != _V14_EXPECTED_COLUMNS:
        raise RuntimeError(f"V14 schema drift: {frame.columns.tolist()}")
    ids = frame[_V14_UID].astype(str).tolist()
    expected = list(map(str, expected_ids))
    if ids != expected or len(ids) != len(set(ids)):
        raise RuntimeError("V14 ID set/order/uniqueness drift")
    values = frame[_V14_TARGETS].to_numpy(_v14_np.float64)
    if values.shape != (len(expected), len(_V14_TARGETS)):
        raise RuntimeError(f"V14 shape drift: {values.shape}")
    if not _v14_np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
        raise RuntimeError("V14 non-finite/out-of-range values")


def _v14_rank(matrix):
    matrix = _v14_np.asarray(matrix, dtype=float)
    result = _v14_np.zeros_like(matrix, dtype=float)
    for j in range(result.shape[1]):
        result[:, j] = _v14_pd.Series(matrix[:, j]).rank(method="average", pct=True).to_numpy()
    return result


def _v14_locate_file(name, expected_sha=None):
    matches = []
    for root, dirs, files in _v14_os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in {"train_series", "test_series"}]
        if name in files:
            matches.append(_V14Path(root) / name)
    if expected_sha is not None:
        matches = [p for p in matches if _v14_sha(p) == expected_sha]
    if len(matches) != 1:
        raise RuntimeError(f"V14 expected one {name} hash={expected_sha}, found={matches}")
    return matches[0]


def _v14_load_parent(test_ids):
    # V14 is built as a fork of the verified V10 notebook.  V10 must first
    # materialize its own parent in /kaggle/working; do not trust an arbitrary
    # mounted CSV or a same-shaped local artifact.
    source = _V14Path("/kaggle/working/submission.csv")
    if not source.is_file():
        raise RuntimeError("V10 did not materialize /kaggle/working/submission.csv")
    observed = _v14_sha(source)
    if observed != _V14_PARENT_SHA256:
        raise RuntimeError(
            f"V10 parent hash drift: observed={observed} expected={_V14_PARENT_SHA256}"
        )
    parent = _v14_pd.read_csv(source, dtype={_V14_UID: str})
    _v14_validate(parent, test_ids)
    _v14_shutil.copy2(source, _V14_PARENT)
    if _v14_sha(_V14_PARENT) != _V14_PARENT_SHA256:
        raise RuntimeError("V14 parent copy hash mismatch")
    _v14_log(f"parent materialized from V10: sha={_V14_PARENT_SHA256[:16]}")
    return parent


# The student runtime source is embedded from the audited public reproduction.
_V14_STUDENT_SOURCE_B64 = __V14_STUDENT_SOURCE_B64__
_V14_STUDENT_SOURCE_SHA256 = __V14_STUDENT_SOURCE_SHA256__


def _v14_run_student(test_ids):
    source = __import__("base64").b64decode(_V14_STUDENT_SOURCE_B64).decode()
    if _v14_hashlib.sha256(source.encode()).hexdigest() != _V14_STUDENT_SOURCE_SHA256:
        raise RuntimeError("V14 embedded student source hash mismatch")
    namespace = {"__name__": "__v14_student_runtime__", "__file__": "<v14-student-runtime>"}
    _v14_previous_output = _v14_os.environ.get("RSNA_OUTPUT")
    _v14_os.environ["RSNA_OUTPUT"] = str(_V14_STUDENT)
    compiled = compile(source, "<v14-student-runtime>", "exec")
    try:
        exec(compiled, namespace, namespace)
    finally:
        if _v14_previous_output is None:
            _v14_os.environ.pop("RSNA_OUTPUT", None)
        else:
            _v14_os.environ["RSNA_OUTPUT"] = _v14_previous_output
    if not _V14_STUDENT.is_file():
        raise RuntimeError("student runtime did not emit its output")
    student = _v14_pd.read_csv(_V14_STUDENT, dtype={_V14_UID: str})
    _v14_validate(student, test_ids)
    _V14_AUDIT_PAYLOAD["student_sha256"] = {
        name: _V14_STUDENT_SHA256[name] for name in sorted(_V14_STUDENT_SHA256)
    }
    _V14_AUDIT_PAYLOAD["student_submission_sha256"] = _v14_sha(_V14_STUDENT)
    return student


try:
    _v14_deadline("start")
    _V14_TEST_ROOT = None
    for candidate in [
        _V14Path("/kaggle/input/rsna-knee-abnormality-detection"),
        _V14Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    ]:
        if (candidate / "test.csv").is_file():
            _V14_TEST_ROOT = candidate
            break
    if _V14_TEST_ROOT is None:
        raise FileNotFoundError("V14 competition test root not found")
    _v14_test = _v14_pd.read_csv(_V14_TEST_ROOT / "test.csv", dtype={_V14_UID: str})
    if _V14_UID not in _v14_test or _v14_test[_V14_UID].isna().any() or not _v14_test[_V14_UID].is_unique:
        raise RuntimeError("V14 test identity contract failed")
    _v14_test_ids = _v14_test[_V14_UID].astype(str).tolist()
    _v14_parent = _v14_load_parent(_v14_test_ids)
    _V14_AUDIT_PAYLOAD.update({
        "test_rows": len(_v14_test),
        "parent_sha256": _v14_sha(_V14_PARENT),
        "parent_materialized": True,
    })
    _v14_deadline("after_parent")
    try:
        _v14_student = _v14_run_student(_v14_test_ids)
        _v14_deadline("after_student")
        _v14_parent_rank = _v14_rank(_v14_parent[_V14_TARGETS].to_numpy(float))
        _v14_student_rank = _v14_rank(_v14_student[_V14_TARGETS].to_numpy(float))
        _v14_final = 0.50 * _v14_parent_rank + 0.50 * _v14_student_rank
        _v14_candidate = _v14_pd.DataFrame(_v14_final, columns=_V14_TARGETS)
        _v14_candidate.insert(0, _V14_UID, _v14_test_ids)
        _v14_validate(_v14_candidate, _v14_test_ids)
        _v14_candidate.to_csv(_V14_PRIMARY, index=False)
        _V14_AUDIT_PAYLOAD.update({
            "status": "V14_EXPERIMENT_SELECTED",
            "submit_allowed": False,
            "selected_sha256": _v14_sha(_V14_PRIMARY),
            "student_rank_changed_targets": [
                target for j, target in enumerate(_V14_TARGETS)
                if tuple(_v14_np.argsort(-_v14_final[:, j], kind="stable"))
                != tuple(_v14_np.argsort(-_v14_parent_rank[:, j], kind="stable"))
            ],
        })
    except Exception as _v14_student_error:
        _V14_AUDIT_PAYLOAD.update({
            "status": "V14_STUDENT_ERROR_V10_PRESERVED",
            "submit_allowed": False,
            "student_error": f"{type(_v14_student_error).__name__}: {_v14_student_error}",
            "student_traceback": _v14_traceback.format_exc(),
        })
        _v14_shutil.copy2(_V14_PARENT, _V14_PRIMARY)
    _v14_validate(_v14_pd.read_csv(_V14_PRIMARY, dtype={_V14_UID: str}), _v14_test_ids)
except Exception as _v14_error:
    _V14_AUDIT_PAYLOAD.update({
        "status": "V14_PARENT_ERROR_NO_SUBMISSION",
        "submit_allowed": False,
        "error": f"{type(_v14_error).__name__}: {_v14_error}",
        "traceback": _v14_traceback.format_exc(),
    })
    if _V14_PARENT.is_file() and _v14_test_ids is not None:
        _v14_shutil.copy2(_V14_PARENT, _V14_PRIMARY)
    elif _V14_PRIMARY.exists():
        # V10 may have left a DINO emergency artifact; it is not a V14 parent.
        _V14_PRIMARY.unlink()
finally:
    if _V14_PRIMARY.is_file():
        _V14_AUDIT_PAYLOAD["primary_sha256"] = _v14_sha(_V14_PRIMARY)
    _V14_AUDIT.write_text(_v14_json.dumps(_V14_AUDIT_PAYLOAD, indent=2, sort_keys=True) + "\n")
    _v14_log(f"audit={_V14_AUDIT_PAYLOAD}")
'''

# Use the audited Student runtime source as an embedded, hash-pinned payload.
student_source_path = ROOT / "research" / "v14_top20_cli" / "resnet18_student" / "inference_runtime.py"
student_source = student_source_path.read_text(encoding="utf-8")
student_sha = hashlib.sha256(student_source.encode()).hexdigest()
import base64
runtime = RUNTIME.replace("__V14_TARGETS__", repr(TARGETS))
runtime = runtime.replace("__V14_PARENT_SHA256__", repr(PARENT_SHA256))
runtime = runtime.replace("__V14_STUDENT_SHA256__", repr(STUDENT_SHA256))
runtime = runtime.replace("__V14_STUDENT_SOURCE_B64__", repr(base64.b64encode(student_source.encode()).decode()))
runtime = runtime.replace("__V14_STUDENT_SOURCE_SHA256__", repr(student_sha))

# The public source uses environment variables for its output and is safe to
# execute after the exact V10 parent has been copied aside.
base_nb_path = ROOT / "kaggle" / "submission_v10_rad_cross" / "rsna-knee-v10-rad-cross-gated.ipynb"
base_nb = nbformat.read(base_nb_path, as_version=4)
nb = base_nb
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
nb.cells.append(nbformat.v4.new_code_cell(
    """# V14 student + exact V10 parent
import time
T0 = time.time()
print('V14 gated experiment appended after verified V10 runtime; no competition submit in kernel')
"""
))
nb.cells.append(nbformat.v4.new_code_cell(runtime))
OUT_DIR.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT_NB)
meta = {
    "id": "danielsolo1770/rsna-knee-v14-student-v10-gated",
    "title": "rsna-knee-v14-student-v10-gated",
    "code_file": OUT_NB.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "dataset_sources": [
        "prvsiyan/rsna-knee-resnet18-student-v1",
        "pilkwang/pilkwang-public-dataset-for-notebooks-figures",
        "pilkwang/rsna-knee-llm-labels",
        "stevenleehans/rsna-knee-llm-report-labels",
        "antoinegg1/rsna-knee-e9-radimagenet-heads-v15",
        "antoinegg1/rsna-knee-e11-diverse-heads-v20",
        "pilkwang/rsna-knee-weights",
        "marwanmath/resnet-50-radimagenet-marwan",
    ],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
    "kernel_sources": [],
    "keywords": ["gpu"],
    "machine_shape": "NvidiaTeslaT4",
}
OUT_META.write_text(json.dumps(meta, indent=2) + "\n")
print(OUT_NB)
print(OUT_META)
print("student_source_sha256", student_sha)
print("parent_sha256", PARENT_SHA256)
print("notebook_sha256", hashlib.sha256(OUT_NB.read_bytes()).hexdigest())
