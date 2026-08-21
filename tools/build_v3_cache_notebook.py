#!/usr/bin/env python3
"""Package the V3.1 cache builder into a self-contained Kaggle notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    driver = '''import subprocess
import sys
from pathlib import Path

competition = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
if not competition.is_dir():
    matches = list(Path("/kaggle/input").glob("**/train_series.csv"))
    if not matches:
        raise FileNotFoundError("competition input is unavailable")
    competition = matches[0].parent
command = [
    sys.executable,
    "/kaggle/working/build_cache.py",
    "--train-csv", str(competition / "train.csv"),
    "--series-csv", str(competition / "train_series.csv"),
    "--image-root", str(competition / "train_series"),
    "--output-dir", "/kaggle/working/v3_1_cache",
]
print("Launching:", " ".join(command), flush=True)
subprocess.run(command, check=True)
'''
    notebook = {
        "cells": [
            code_cell(f"%%writefile /kaggle/working/build_cache.py\n{source}"),
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
