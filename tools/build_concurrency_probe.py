"""Build N tiny GPU probe kernels to measure Kaggle's concurrent-session limit.

Purpose
-------
The user reports that a *running* Kaggle notebook is not killed when the GPU quota is
exhausted. If true, the binding constraints become (a) the per-session runtime cap and
(b) **how many sessions run concurrently** - not the 30 h/week quota.

Knowing the concurrency limit changes the plan completely: with C concurrent sessions we
can run a whole wave of experiments in the wall-clock time of one. So measure it, cheaply.

Each probe only prints device info, sleeps, and exits - ~4 min of GPU each.

Usage
    python tools/build_concurrency_probe.py --n 3
    python tools/build_concurrency_probe.py --n 3 --push     # push all of them
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CODE = r'''
import os, socket, time, subprocess, torch

t0 = time.time()
print("PROBE START  ", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "UTC", flush=True)
print("hostname     :", socket.gethostname(), flush=True)
print("torch        :", torch.__version__, flush=True)
print("cuda avail   :", torch.cuda.is_available(), flush=True)
print("device count :", torch.cuda.device_count(), flush=True)
for i in range(torch.cuda.device_count()):
    print(f"  gpu{i}     :", torch.cuda.get_device_name(i),
          f"| {torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB", flush=True)
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True,
                         timeout=60).stdout.strip(), flush=True)
except Exception as e:
    print("nvidia-smi failed:", e, flush=True)

# Do a tiny matmul so we know the GPU is actually usable, not just attached.
if torch.cuda.is_available():
    a = torch.randn(4096, 4096, device="cuda")
    for _ in range(20):
        a = a @ a / a.shape[-1]
    torch.cuda.synchronize()
    print("matmul ok, elapsed %.1fs" % (time.time() - t0), flush=True)

SLEEP = __SLEEP__
print(f"sleeping {SLEEP}s to hold the session open...", flush=True)
time.sleep(SLEEP)
print("PROBE END    ", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "UTC", flush=True)
print("total elapsed %.1fs" % (time.time() - t0), flush=True)
'''


def build(i: int, sleep_s: int) -> Path:
    d = REPO / "kaggle" / f"probe_concurrency_{i}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}},
        "cells": [
            {"cell_type": "code", "metadata": {},
             "source": CODE.replace("__SLEEP__", str(sleep_s)).splitlines(keepends=True),
             "outputs": [], "execution_count": None},
        ],
    }
    (d / f"probe-{i}.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    meta = {
        "id": f"danielsolo1770/probe-concurrency-{i}",
        "title": f"probe-concurrency-{i}",
        "code_file": f"probe-{i}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n",
                                            encoding="utf-8")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--sleep", type=int, default=240)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    dirs = [build(i, args.sleep) for i in range(1, args.n + 1)]
    print("built:", *[d.name for d in dirs], sep="\n  ")
    print(f"\nGPU cost if all run: ~{args.n * (args.sleep + 90) / 60:.1f} min")

    if args.push:
        import subprocess
        import sys
        for d in dirs:
            r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push",
                                "-p", str(d)], capture_output=True, text=True)
            print(f"push {d.name}: rc={r.returncode}")
            if r.returncode:
                print(r.stdout[-800:], r.stderr[-800:])
            time.sleep(2)
        print("\nall pushed. watch with:")
        for d in dirs:
            print(f"  kaggle kernels status danielsolo1770/{d.name.split('_')[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
