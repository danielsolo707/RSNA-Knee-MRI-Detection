#!/usr/bin/env python3
"""Build the V15 5-fold student OOF notebook (T4) from the pinned runtime.

Usage: python tools/build_v15_student_5fold_notebook.py
Outputs: kaggle/submission_v15_student_5fold/rsna-knee-v15-student-5fold.ipynb
         kaggle/submission_v15_student_5fold/kernel-metadata.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle" / "submission_v15_student_5fold"
OUT_NB = OUT_DIR / "rsna-knee-v15-student-5fold.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

# The V15 5-fold OOF kernel is standalone: no submission in-kernel, no parent
# kernel source. Bridge volumes come from the v15-bridge kernel output, mounted
# as a kernel source (dataset publish would require a slow 6.9 GB re-upload).
DATASET_SOURCES = [
    "pilkwang/pilkwang-public-dataset-for-notebooks-figures",
    "pilkwang/rsna-knee-weights",
    "antoinegg1/rsna-knee-e9-radimagenet-heads-v15",
    "pilkwang/rsna-knee-llm-labels",
    "stevenleehans/rsna-knee-llm-report-labels",
    "lixin73/rsna-knee-llm-report-labels-sol56",
]
COMPETITION_SOURCES = ["rsna-knee-abnormality-detection"]
KERNEL_SOURCES = ["danielsolo1770/rsna-knee-v15-bridge"]

runtime_path = ROOT / "tools" / "v15_student_5fold_runtime.py"
runtime = runtime_path.read_text(encoding="utf-8")
runtime_sha = hashlib.sha256(runtime.encode()).hexdigest()

cells = [
    nbformat.v4.new_markdown_cell(
        "# V15: 5-fold expert-label OOF for the ResNet18 student\n"
        "Pre-registered protocol (docs/v15_student_5fold_plan.md). No competition "
        "submission is created in-kernel; outputs are v15_oof.csv + v15_audit.json."
    ),
    nbformat.v4.new_code_cell(
        f"%%writefile /kaggle/working/v15_student_5fold_runtime.py\n{runtime}"
    ),
    nbformat.v4.new_code_cell(
        "import subprocess, sys\n"
        "result = subprocess.run(\n"
        "    [sys.executable, '/kaggle/working/v15_student_5fold_runtime.py',\n"
        "     '--out-dir', '/kaggle/working'],\n"
        "    capture_output=True, text=True)\n"
        "print(result.stdout)\n"
        "print(result.stderr)\n"
        "if result.returncode != 0:\n"
        "    raise SystemExit(result.returncode)"
    ),
]
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
OUT_DIR.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT_NB)

meta = {
    "id": "danielsolo1770/rsna-knee-v15-student-5fold",
    "title": "rsna-knee-v15-student-5fold",
    "code_file": OUT_NB.name,
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "dataset_sources": DATASET_SOURCES,
    "competition_sources": COMPETITION_SOURCES,
    "model_sources": [],
    "kernel_sources": KERNEL_SOURCES,
    "keywords": ["gpu"],
    "machine_shape": "NvidiaTeslaT4",
}
OUT_META.write_text(json.dumps(meta, indent=2) + "\n")
print(OUT_NB)
print(OUT_META)
print("runtime_sha256", runtime_sha)
print("notebook_sha256", hashlib.sha256(OUT_NB.read_bytes()).hexdigest())
