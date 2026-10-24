"""Local fine-tune launcher: run tools/train_raptor.py on the RTX 4070 Laptop (free GPU).

train_raptor.py discovers its inputs Kaggle-style (walking /kaggle/input), which does not
exist locally. This launcher patches the three discovery functions onto the local layout,
then calls main() with a fixed base config plus any extra args you pass after the script
name. Run from the repo root.

Base config (fixed): resume the public v5_full_swa checkpoint, LOO pseudo-labels, 384 px,
k=10/k_eval=24, bs=2, grad_ckpt, eval_first, require_resume, workers=0.

Variable per experiment (pass as extra args): --tag, --bb_lr, --head_lr, --ft_epochs,
--eval_every_steps, --max_steps, --smoke.

Examples:
    python tools/train_local_raptor.py --smoke                      # wiring check
    python tools/train_local_raptor.py --tag L1 --bb_lr 3e-6 --head_lr 3e-5 \
        --ft_epochs 1 --eval_every_steps 100 --max_steps 700        # gentle-LR sweep

Outputs land in artifacts/local_runs/<tag>/ (console.log tee'd there).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
# NOTE: do not set PYTORCH_CUDA_ALLOC_CONF=expandable_segments here - it is not
# supported on Windows and raises a CUDA warning on first allocation.

import train_raptor as tr  # noqa: E402

CORPUS = REPO / "artifacts" / "raptor_corpus"
LABELS = REPO / "artifacts" / "pseudo_labels" / "pseudo_labels_LOO_v4_w60.parquet"
RESUME = REPO / "artifacts" / "public_arms" / "ckpt" / "raptor_ft_coatnet_v5_full_swa.pt"

_LOCAL_DIRS = [CORPUS, REPO / "data" / "raw", REPO / "artifacts" / "raptor_labels"]


def _local_corpus():
    vols = tr._TwoPartVols(np.load(CORPUS / "all_vols.npy", mmap_mode="r"),
                           np.load(CORPUS / "extra_vols.npy", mmap_mode="r"))
    masks = np.concatenate([np.load(CORPUS / "all_masks.npy"),
                            np.load(CORPUS / "extra_masks.npy")], axis=0)
    return vols, masks


def _local_ids():
    return np.concatenate([
        np.load(CORPUS / "all_ids.npy", allow_pickle=True).astype(str),
        np.load(CORPUS / "extra_ids.npy", allow_pickle=True).astype(str)])


def _local_find(*names, root="/kaggle/input"):
    for d in _LOCAL_DIRS:
        if not d.is_dir():
            continue
        for n in names:
            p = d / n
            if p.exists():
                return str(p)
    return None


def _tag_from(argv: list[str]) -> str:
    for i, x in enumerate(argv):
        if x == "--tag" and i + 1 < len(argv):
            return argv[i + 1]
    return "local"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def main() -> int:
    extra = sys.argv[1:]
    smoke = "--smoke" in extra
    tag = _tag_from(extra)
    out_dir = REPO / "artifacts" / "local_runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, p in (("labels", LABELS), ("resume ckpt", RESUME),
                    ("all_vols", CORPUS / "all_vols.npy"),
                    ("extra_vols", CORPUS / "extra_vols.npy")):
        if not Path(p).exists():
            raise SystemExit(f"missing {name}: {p}")

    # Patch Kaggle-only discovery onto the local layout BEFORE main() reads them.
    tr._open_corpus = _local_corpus
    tr._open_ids = _local_ids
    tr._find = _local_find

    base = [
        "train_raptor.py",
        "--arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",  # overridden by ckpt
        "--res", "384",                                        # overridden by ckpt
        "--bs", "2", "--k", "10", "--k_eval", "24",
        "--grad_ckpt", "--ckpt", "timm", "--norm", "imagenet",
        "--labels", str(LABELS),
        "--resume", str(RESUME),
        "--require_resume", "--eval_first",
        "--workers", "0",
        "--out_dir", str(out_dir),
        "--max_wall_h", "20",
        "--ckpt_every_min", "15",
        "--tag", tag,
    ]
    if smoke:
        base += ["--smoke", "--ft_epochs", "1", "--max_steps", "3",
                 "--eval_every_steps", "2"]
    elif "--ft_epochs" not in extra and "--epochs" not in extra:
        base += ["--ft_epochs", "1"]
    # Drop a duplicate --tag from extra (we already set it from the extracted value).
    passthrough = []
    skip = False
    for x in extra:
        if skip:
            skip = False
            continue
        if x == "--tag":
            skip = True
            continue
        passthrough.append(x)
    sys.argv = base + passthrough
    print("argv", sys.argv, flush=True)

    _log = open(out_dir / "console.log", "a", buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, _log)
    sys.stderr = Tee(sys.__stderr__, _log)

    tr.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
