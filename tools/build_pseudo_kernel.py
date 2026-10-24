"""Build the Kaggle kernel that generates pseudo-labels for the 4,349 weak-label studies.

Why run this on Kaggle at all (we also have a local copy of the script)
----------------------------------------------------------------------
Kaggle already has the corpus and both checkpoints ATTACHED, so it can start immediately.
The local route is blocked until the corpus re-fetch finishes (~4 h after the downloader
truncation bug destroyed 7.65 GB). Running here removes that dependency from the critical
path and costs quota rather than wall-clock.

Cost control
------------
Inference is ~14.5 s/study/arm at k_eval=24. For 4,349 studies:
    k_eval=24, both arms : ~35 h  (too much - 4 sessions)
    k_eval=12, both arms : ~17.5h (2 sessions)
    k_eval=12, one arm   : ~8.8h  (1 session)  <-- chosen
Default is therefore `--arms v5 --k-eval 12`, which fits a single 9 h session.

Because a session can be killed at the 9 h cap, the script checkpoints partial
predictions periodically; kernel output stays downloadable after a kill (verified on V34).

Usage
    python tools/build_pseudo_kernel.py
    python tools/build_pseudo_kernel.py --arms v5,v4 --k-eval 12   # 2 sessions
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DST = REPO / "kaggle" / "pseudo_labels_v5"
SCRIPT = (REPO / "tools" / "make_pseudo_labels.py").read_text(encoding="utf-8")
LABELS = REPO / "artifacts" / "raptor_labels" / "labels_llm_soft.parquet"

LAUNCH = r'''
# ---- Kaggle wiring: point the script at the attached assets and run it ----
import os, sys, base64, glob
from pathlib import Path

_LABEL_B64 = """__LABEL_B64__"""
work = Path("/kaggle/working")
lab = work / "labels_llm_soft.parquet"
if not lab.is_file():
    lab.write_bytes(base64.b64decode(_LABEL_B64))
print("labels:", lab, lab.stat().st_size, "bytes", flush=True)

# Kaggle mounts datasets at /kaggle/input/datasets/<owner>/<slug>/ or /kaggle/input/<slug>/.
# Resolve by glob, never by a hardcoded path (that cost us a probe run already).
print("checkpoints visible:",
      glob.glob("/kaggle/input/**/raptor_ft_coatnet_*.pt", recursive=True), flush=True)

# The two corpus datasets mount into SEPARATE directories, so resolve each part by glob
# and pass them separately. Pointing both at one directory is what raised SystemExit.
p1 = sorted(glob.glob("/kaggle/input/**/all_vols.npy", recursive=True))
p2 = sorted(glob.glob("/kaggle/input/**/extra_vols.npy", recursive=True))
print("part1:", p1, flush=True)
print("part2:", p2, flush=True)
if not p1 or not p2:
    raise SystemExit("FATAL: corpus incomplete - attach BOTH knee-raptor-corpus and "
                     "knee-raptor-corpus-ext.")
CORPUS_DIR = str(Path(p1[0]).parent)
EXTRA_DIR = str(Path(p2[0]).parent)
print("CORPUS_DIR:", CORPUS_DIR, "| EXTRA_DIR:", EXTRA_DIR, flush=True)

# The trainer's _open_corpus() would find part 1 only; the script patches it to
# assemble BOTH parts, so no special handling is needed here.
sys.argv = [
    "make_pseudo_labels.py",
    "--corpus-dir", CORPUS_DIR,
    "--corpus-extra-dir", EXTRA_DIR,
    "--ckpt-dir", "/kaggle/input",
    "--labels", str(lab),
    "--out", str(work / "pseudo_labels"),
    "--arms", "__ARMS__",
    "--k-eval", "__KEVAL__",
    "--blend-w", "0.60",
    # bs=4 x k=12 = 48 windows/batch, the SAME peak memory the arm probe ran safely
    # (bs=2 x k=24), but with better GPU utilisation. That drops the estimate from
    # ~8.8 h (uncomfortably near the 9 h session cap) to ~6 h.
    "--bs", "4",
    "--workers", "0",
]
print("argv:", sys.argv, flush=True)
main()
'''


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    arms = "v5"
    keval = "12"
    if "--arms" in __import__("sys").argv:
        arms = __import__("sys").argv[__import__("sys").argv.index("--arms") + 1]
    if "--k-eval" in __import__("sys").argv:
        keval = __import__("sys").argv[__import__("sys").argv.index("--k-eval") + 1]

    b64 = base64.b64encode(LABELS.read_bytes()).decode("ascii")
    # train_raptor.py must be EMBEDDED, not merely shipped in the folder: `kaggle kernels
    # push` only runs the notebook, and non-notebook files are not extracted at runtime
    # (this cost a run with ModuleNotFoundError). Base64 it and write it out at runtime,
    # the same way the labels parquet is handled.
    trainer_b64 = base64.b64encode(
        (REPO / "tools" / "train_raptor.py").read_bytes()).decode("ascii")
    # `from __future__ import annotations` must be the first statement in a compiled
    # unit, and we prepend a preamble - so drop it (nothing here needs it).
    # New local name on purpose: assigning to the module-level SCRIPT inside main()
    # would rebind it as a local and break the read (UnboundLocalError).
    script_src = SCRIPT.replace("from __future__ import annotations\n", "", 1)
    # Strip the __main__ guard. In a notebook cell __name__ == "__main__", so the block
    # would run the moment the cell is defined - BEFORE the launch cell sets sys.argv.
    # argparse then parses papermill's own args (-f /tmp/xxx.json) and exits 2.
    for probe in ('if __name__ == "__main__":\n    raise SystemExit(main())\n',
                  'if __name__ == "__main__":\n    main()\n'):
        if probe in script_src:
            script_src = script_src.replace(probe, "")
            break
    else:
        raise SystemExit("could not find the __main__ guard to strip - check the script")
    preamble = (
        "# Materialise train_raptor.py at runtime (embedded; shipped files are not extracted)\n"
        "import os, sys, base64\n"
        "from pathlib import Path\n"
        "_TRAINER_B64 = \"\"\"__TRAINER_B64__\"\"\"\n"
        "_tp = Path('/kaggle/working/train_raptor.py')\n"
        "if not _tp.is_file():\n"
        "    _tp.write_bytes(base64.b64decode(_TRAINER_B64))\n"
        "sys.path.insert(0, '/kaggle/working')\n"
        "sys.path.insert(0, str(Path.cwd()))\n"
        "print('train_raptor:', _tp, _tp.stat().st_size, 'bytes', flush=True)\n"
    ).replace("__TRAINER_B64__", trainer_b64)
    # NOTE: must be script_src (future-import stripped), not SCRIPT, or prepending the
    # preamble puts `from __future__ import annotations` after real statements.
    cell = preamble + "\n" + script_src + "\n\n" + (LAUNCH.replace("__LABEL_B64__", b64)
                                                    .replace("__ARMS__", arms)
                                                    .replace("__KEVAL__", keval))
    compile(cell, "pseudo", "exec")  # syntax gate

    md = [
        "# Pseudo-labels for the 4,349 weak-label studies\n",
        "\n",
        "Why: measured out-of-sample on gold-58 (the public checkpoints held it out):\n",
        "\n",
        "| label source | AUC vs gold-58 |\n",
        "|---|---|\n",
        "| current LLM soft labels | 0.8927 |\n",
        "| dual-CoAtNet fusion pseudo-labels | 0.9172 |\n",
        "| **blend 40% current + 60% pseudo** | **0.9291 (+0.0364)** |\n",
        "\n",
        "+0.036 is ~9x the blend-weight effect that FAILED to transfer in V37, so it is\n",
        "large enough to survive n=58 noise.\n",
        "\n",
        "Caveat: on the TRAINING studies the public checkpoints' predictions are in-sample,\n",
        "so pseudo-labels there are optimistic. Handled by blending back with the original\n",
        "soft labels at a single GLOBAL ratio -- never per-target (V37 lesson).\n",
        "\n",
        f"Config: arms={arms}, k_eval={keval}, partial predictions checkpointed so a 9 h\n",
        "session kill does not lose the run.\n",
    ]
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": md},
            {"cell_type": "code", "metadata": {}, "source": cell.splitlines(keepends=True),
             "outputs": [], "execution_count": None},
        ],
    }
    out = DST / "pseudo-labels-v5.ipynb"
    out.write_text(json.dumps(nb), encoding="utf-8")

    meta = {
        "id": "danielsolo1770/pseudo-labels-v5",
        "title": "pseudo-labels-v5",
        "code_file": "pseudo-labels-v5.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "dreaddevelopment/knee-raptor-corpus",
            "dreaddevelopment/knee-raptor-corpus-ext",
            "dreaddevelopment/raptor-knee-maxspan",
            "dreaddevelopment/raptor-knee-widedense",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n",
                                              encoding="utf-8")
    print(f"pseudo-label kernel ready: {out}")
    print(f"  arms={arms}  k_eval={keval}  (est ~8.8 h for v5 at k=12)")


if __name__ == "__main__":
    main()
