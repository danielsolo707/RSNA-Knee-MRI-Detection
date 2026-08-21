"""Modal V28 trainer. Cheap GPU only. Resumable from /vol/checkpoints.

Run (after Modal token_id + token_secret are in ~/.modal.toml):

  python -m modal run modal_v28.app --action ping
  python -m modal run modal_v28.app --action smoke
  python -m modal run modal_v28.app --action train --fold 0
  python -m modal run modal_v28.app --action train-all

Do not pass H100/A100-80. $30 credit cap is enforced in-code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

VOL_NAME = "rsna-knee-v28-vol"
APP_NAME = "rsna-knee-v28"
# Starter credit is $30. Keep a hard dollar ceiling so we cannot burn the month
# on a stuck job. L4 is $0.000222/s ≈ $0.80/h.
MAX_GPU_USD = 22.0
L4_USD_PER_SEC = 0.000222
T4_USD_PER_SEC = 0.000164
SAFE_GPUS = ["L4", "T4"]
FOLD0_PROMOTE = 0.83

ROOT = Path(__file__).resolve().parent


def _download_dino():
    from transformers import AutoModel

    AutoModel.from_pretrained("facebook/dinov2-small")


image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install(
        "transformers==4.46.3",
        "safetensors",
        "accelerate",
        "pandas==2.2.3",
        "scikit-learn==1.5.2",
        "tqdm",
    )
    .run_function(_download_dino)
    .add_local_file(str(ROOT / "train_core.py"), remote_path="/root/train_core.py")
)

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


def _gpu_rate(name: str) -> float:
    n = (name or "").upper()
    if "L4" in n:
        return L4_USD_PER_SEC
    return T4_USD_PER_SEC


def _spend_guard(out_root: Path, gpu_name: str) -> None:
    spent_p = out_root / "spend.json"
    spent = 0.0
    if spent_p.is_file():
        try:
            spent = float(json.loads(spent_p.read_text())["usd"])
        except Exception:
            spent = 0.0
    if spent >= MAX_GPU_USD:
        raise RuntimeError(f"credit guard: already spent ~${spent:.2f} >= ${MAX_GPU_USD:.2f}")
    (out_root / "_gpu_rate.json").write_text(
        json.dumps({"gpu": gpu_name, "usd_per_sec": _gpu_rate(gpu_name), "spent_usd": spent}) + "\n"
    )


def _add_spend(out_root: Path, seconds: float, gpu_name: str) -> float:
    spent_p = out_root / "spend.json"
    spent = 0.0
    hist = []
    if spent_p.is_file():
        prev = json.loads(spent_p.read_text())
        spent = float(prev.get("usd", 0.0))
        hist = list(prev.get("jobs", []))
    add = float(seconds) * _gpu_rate(gpu_name)
    spent += add
    hist.append({"seconds": seconds, "gpu": gpu_name, "usd": add})
    spent_p.write_text(json.dumps({"usd": spent, "jobs": hist}, indent=2) + "\n")
    return spent


@app.function(timeout=120, cpu=0.5, memory=1024)
def ping() -> dict:
    import torch

    return {
        "ok": True,
        "cuda": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
    }


@app.function(gpu=["T4"], timeout=180, cpu=2, memory=4096)
def gpu_ping() -> dict:
    import torch

    ok = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if ok else "cpu"
    if any(bad in name.upper() for bad in ("H100", "H200", "B200", "A100")):
        raise RuntimeError(f"refusing expensive GPU: {name}")
    return {"ok": ok, "gpu": name, "torch": torch.__version__}


@app.function(
    gpu=SAFE_GPUS,
    timeout=60 * 60,
    cpu=4,
    memory=16384,
    volumes={"/vol": vol},
)
def smoke() -> dict:
    import sys
    import time

    sys.path.insert(0, "/root")
    from train_core import TrainCfg, train_one_fold

    t0 = time.time()
    gpu = __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else "cpu"
    out_root = Path("/vol")
    _spend_guard(out_root, gpu)
    cfg = TrainCfg(
        cache_dir="/vol/cache",
        out_dir="/vol/checkpoints/fold0_smoke",
        fold=0,
        epochs=1,
        batch=8,
        resume=True,
        smoke=True,
        num_workers=0,
    )
    rec = train_one_fold(cfg, commit=vol.commit)
    rec["spend_usd"] = _add_spend(out_root, time.time() - t0, gpu)
    vol.commit()
    return rec


@app.function(
    gpu=SAFE_GPUS,
    timeout=4 * 60 * 60,
    cpu=4,
    memory=16384,
    volumes={"/vol": vol},
)
def train_fold(fold: int = 0, epochs: int = 12, resume: bool = True) -> dict:
    import sys
    import time

    sys.path.insert(0, "/root")
    from train_core import TrainCfg, train_one_fold

    t0 = time.time()
    torch = __import__("torch")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if any(bad in gpu.upper() for bad in ("H100", "H200", "B200", "A100")):
        raise RuntimeError(f"refusing expensive GPU: {gpu}")
    out_root = Path("/vol")
    _spend_guard(out_root, gpu)
    cfg = TrainCfg(
        cache_dir="/vol/cache",
        out_dir=f"/vol/checkpoints/fold{int(fold)}",
        fold=int(fold),
        epochs=int(epochs),
        batch=16,
        resume=bool(resume),
        smoke=False,
        num_workers=2,
    )
    rec = train_one_fold(cfg, commit=vol.commit)
    rec["spend_usd"] = _add_spend(out_root, time.time() - t0, gpu)
    rec["gpu"] = gpu
    vol.commit()
    return rec


@app.function(timeout=120, cpu=0.5, memory=1024, volumes={"/vol": vol})
def fold0_ok_to_clone() -> dict:
    p = Path("/vol/checkpoints/fold0/receipt.json")
    if not p.is_file():
        return {"ok": False, "reason": "no fold0 receipt"}
    rec = json.loads(p.read_text())
    auc = float(rec.get("best_val_auc") or -1)
    return {"ok": auc >= FOLD0_PROMOTE, "best_val_auc": auc, "need": FOLD0_PROMOTE, "receipt": rec}


@app.local_entrypoint()
def main(action: str = "ping", fold: int = 0, epochs: int = 12, resume: bool = True):
    """Actions: ping | smoke | train | train-all | status."""
    if action == "ping":
        print(ping.remote())
        return
    if action == "gpu-ping":
        print(gpu_ping.remote())
        return
    if action == "smoke":
        print(smoke.remote())
        return
    if action == "status":
        print(fold0_ok_to_clone.remote())
        return
    if action == "train":
        print(train_fold.remote(fold=fold, epochs=epochs, resume=resume))
        return
    if action == "train-all":
        r0 = train_fold.remote(fold=0, epochs=epochs, resume=resume)
        print("fold0", r0)
        gate = fold0_ok_to_clone.remote()
        print("gate", gate)
        if not gate.get("ok"):
            print("stop: fold0 val below", FOLD0_PROMOTE, "- not cloning 1-4")
            return
        for f in range(1, 5):
            print(f"fold{f}", train_fold.remote(fold=f, epochs=epochs, resume=resume))
        return
    raise SystemExit(f"unknown action {action}")
