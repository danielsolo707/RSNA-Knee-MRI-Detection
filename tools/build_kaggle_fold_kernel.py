"""Build Kaggle GPU kernel: train V31 fold-4 with HARD labels using the cached corpus."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DST = REPO / "kaggle" / "v31_fold4_hard"
_trainer = (REPO / "tools" / "v31probe.py").read_text(encoding="utf-8")
# strip the __main__ block: inside a notebook cell __name__ IS "__main__" and
# main() would fire with default Colab paths before our launcher runs.
TRAINER_SRC = _trainer.replace('if __name__=="__main__":\n    main()', "").replace(
    'if __name__ == "__main__":\n    main()', "")

LAUNCH = '''
import glob, os
print("inputs:", os.listdir("/kaggle/input"), flush=True)
for d in os.listdir("/kaggle/input"):
    p = f"/kaggle/input/{d}"
    try:
        print(" ", d, "->", os.listdir(p)[:8], flush=True)
    except Exception as e:
        print(" ", d, "ERR", e, flush=True)
hits = glob.glob("/kaggle/input/**/x_f16.dat", recursive=True)
assert hits, "V31 cache (x_f16.dat) not found anywhere under /kaggle/input"
cache_dir = os.path.dirname(hits[0])
print("cache at", cache_dir, flush=True)
cfg = Cfg(
    cache_dir=cache_dir,
    out_dir="/kaggle/working/fold4_out",
    labels_csv="",
    hard_labels=True,
    fold=4,
    epochs=24,
    batch=16,
    accum=1,
    lr_head=3e-4,
    lr_bb_top=5e-5,
    llrd=0.7,
    patience=99,
    warmup=1,
    ema_decay=0.999,
    num_workers=2,
)
main(cfg)
'''

CELL = ("import subprocess, sys\n"
        "subprocess.run([sys.executable,'-m','pip','install','-q',"
        "'transformers==4.44.2','accelerate<1.0'], check=True)\n"
        "print('pinned transformers 4.44.2')\n"
        "\n" + TRAINER_SRC + "\n\n" + LAUNCH)


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    compile(CELL, "fold4", "exec")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            {"cell_type": "code", "metadata": {}, "source": CELL.splitlines(keepends=True),
             "outputs": [], "execution_count": None},
        ],
    }
    (DST / "rsna-knee-v31-fold4-hard.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    meta = {
        "id": "danielsolo1770/rsna-knee-v31-fold4-hard",
        "title": "rsna-knee-v31-fold4-hard",
        "code_file": "rsna-knee-v31-fold4-hard.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "kernel_sources": ["danielsolo1770/rsna-knee-v31-cache-280"],
        "dataset_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("fold4 kernel ready:", DST)


if __name__ == "__main__":
    main()
