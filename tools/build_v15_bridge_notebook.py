#!/usr/bin/env python3
"""Package the V15 bridge builder into a self-contained CPU Kaggle notebook.

Usage: python tools/build_v15_bridge_notebook.py
Output: kaggle/submission_v15_bridge/rsna-knee-v15-bridge.ipynb
        kaggle/submission_v15_bridge/kernel-metadata.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle" / "submission_v15_bridge"
OUT_NB = OUT_DIR / "rsna-knee-v15-bridge.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": hashlib.sha1(source.encode()).hexdigest()[:12],
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "tools" / "v15_bridge_runtime.py")
    parser.add_argument("--output", type=Path, default=OUT_NB)
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    driver = '''import subprocess
import sys

command = [
    sys.executable,
    "/kaggle/working/build_v15_bridge.py",
    "--root", "/kaggle/input/rsna-knee-abnormality-detection",
    "--out-dir", "/kaggle/working/v15_bridge",
]
print("Launching:", " ".join(command), flush=True)
subprocess.run(command, check=True)
'''
    notebook = {
        "cells": [
            code_cell(f"%%writefile /kaggle/working/build_v15_bridge.py\n{source}"),
            code_cell(driver),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    meta = {
        "id": "danielsolo1770/rsna-knee-v15-bridge",
        "title": "rsna-knee-v15-bridge",
        "code_file": OUT_NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "dataset_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "kernel_sources": [],
        "keywords": [],
        "machine_shape": "CPU",
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(OUT_NB)
    print(OUT_META)
    print("runtime_sha256", source_sha)
    print("notebook_sha256", hashlib.sha256(OUT_NB.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()