"""Resumable V28 DINOv2-small partial-FT. Cache-only (no DICOM on GPU)."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
N_SLOT = 3
N_SLICE = 3
IMG = 224
SEED = 20260817
UNFREEZE_LAST = 4
AUX_W = 0.25
N_FOLDS = 5


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def report_group(uid: str) -> str:
    return hashlib.md5(uid.encode()).hexdigest()[:8]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainCfg:
    cache_dir: str
    out_dir: str
    fold: int = 0
    epochs: int = 12
    batch: int = 16
    lr_head: float = 2e-4
    lr_backbone: float = 2e-5
    weight_decay: float = 1e-4
    unfreeze_last: int = UNFREEZE_LAST
    patience: int = 4
    resume: bool = True
    smoke: bool = False
    num_workers: int = 2
    seed: int = SEED
    dino_id: str = "facebook/dinov2-small"
    augment: bool = True
    mixup: float = 0.2
    dropout: float = 0.35
    tta: bool = True
    expand_slices: bool = False


class CacheDS(Dataset):
    def __init__(self, x, mask, y, uids, idx, augment=False, expand_slices=False):
        self.x = x
        self.mask = mask
        self.y = y
        self.uids = uids
        self.idx = np.asarray(idx, dtype=np.int64)
        self.augment = bool(augment)
        self.expand_slices = bool(expand_slices)

    def __len__(self):
        return int(len(self.idx))

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = np.asarray(self.x[j], dtype=np.float32)
        m = np.asarray(self.mask[j], dtype=np.float32)
        y = np.asarray(self.y[j], dtype=np.float32)
        if self.expand_slices and x.ndim == 4:
            s, c, h, w = x.shape
            x = np.repeat(x.reshape(s * c, 1, h, w), 3, axis=1)
            m = np.repeat(m, c)
        elif x.ndim == 4 and x.shape[1] == 1:
            x = np.repeat(x, 3, axis=1)
        if self.augment:
            if random.random() < 0.5:
                x = np.ascontiguousarray(x[..., :, ::-1])
            if random.random() < 0.7:
                x = np.clip(x * np.float32(random.uniform(0.8, 1.2)), 0.0, 1.0)
            if random.random() < 0.4:
                x = np.clip(x + np.float32(random.uniform(-0.08, 0.08)), 0.0, 1.0)
        return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(y), self.uids[j]


def load_cache(cache_dir: Path):
    man_p = cache_dir / "manifest.json"
    if not man_p.is_file():
        raise FileNotFoundError(f"cache missing at {cache_dir}")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    n = int(man["n"])
    shape = tuple(man["x_shape"])
    x = np.memmap(cache_dir / "x_f16.dat", dtype=np.float16, mode="r", shape=shape)
    mask = np.memmap(cache_dir / "mask.dat", dtype=np.uint8, mode="r", shape=(n, N_SLOT))
    y = np.load(cache_dir / "y.npy")
    uids = (cache_dir / "uids.txt").read_text(encoding="utf-8").splitlines()
    if len(uids) != n or y.shape[0] != n:
        raise ValueError(f"cache length mismatch n={n} uids={len(uids)} y={y.shape}")
    return man, x, mask, y, uids


class PartialFT(nn.Module):
    def __init__(self, dino_id: str, unfreeze_last: int, dropout: float = 0.35):
        super().__init__()
        from transformers import AutoModel

        self.enc = AutoModel.from_pretrained(dino_id)
        dim = int(self.enc.config.hidden_size)
        for p in self.enc.parameters():
            p.requires_grad = False
        layers = self.enc.encoder.layer
        for layer in layers[-unfreeze_last:]:
            for p in layer.parameters():
                p.requires_grad = True
        self.slot_proj = nn.Linear(dim, 256)
        self.query = nn.Parameter(torch.randn(len(TARGETS), 256) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.aux = nn.Linear(256, len(TARGETS))
        self.dim = dim
        self.n_blocks = len(layers)

    def forward(self, x, m):
        b, s, c, h, w = x.shape
        inp = x.reshape(b * s, c, h, w)
        rgb = inp if c == 3 else inp.mean(1, keepdim=True).repeat(1, 3, 1, 1)
        rgb = (rgb - 0.45) / 0.22
        out = self.enc(pixel_values=rgb)
        tok = self.slot_proj(out.last_hidden_state[:, 0]).reshape(b, s, -1)
        tok = tok * m.unsqueeze(-1)
        attn = torch.einsum("bsd,td->bst", tok, self.query) / 16.0
        attn = attn.masked_fill(m.unsqueeze(-1) == 0, -1e4)
        wgt = torch.softmax(attn, dim=1)
        pooled = torch.einsum("bst,bsd->btd", wgt, tok)
        main = self.head(pooled).squeeze(-1)
        aux = self.aux(tok.mean(1))
        return main, aux


def macro_auc(y, p) -> float:
    scores = []
    for j in range(y.shape[1]):
        if len(np.unique((y[:, j] > 0.5).astype(int))) < 2:
            continue
        try:
            scores.append(roc_auc_score((y[:, j] > 0.5).astype(int), p[:, j]))
        except Exception:
            continue
    return float(np.mean(scores)) if scores else float("nan")


def _amp():
    cuda = torch.cuda.is_available()
    try:
        autocast = lambda: torch.amp.autocast("cuda", enabled=cuda)
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    except Exception:
        autocast = lambda: torch.cuda.amp.autocast(enabled=cuda)
        scaler = torch.cuda.amp.GradScaler(enabled=cuda)
    return autocast, scaler


def train_one_fold(cfg: TrainCfg, commit=None) -> dict:
    t0 = time.time()
    set_seed(cfg.seed + cfg.fold)
    cache_dir = Path(cfg.cache_dir)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man, x, mask, y, uids = load_cache(cache_dir)
    log(f"cache n={len(uids)} shape={tuple(x.shape)} recipe={man.get('recipe')}", t0)

    groups = np.array([report_group(u) for u in uids])
    gkf = GroupKFold(N_FOLDS)
    folds = list(gkf.split(np.arange(len(uids)), groups=groups))
    tr_idx, va_idx = folds[int(cfg.fold)]
    if cfg.smoke:
        tr_idx = tr_idx[:80]
        va_idx = va_idx[:40]
        log("SMOKE subset on", t0)

    ds_tr = CacheDS(x, mask, y, uids, tr_idx, augment=bool(cfg.augment) and not cfg.smoke, expand_slices=bool(getattr(cfg, 'expand_slices', False)))
    ds_va = CacheDS(x, mask, y, uids, va_idx, augment=False, expand_slices=bool(getattr(cfg, 'expand_slices', False)))
    log(f"fold{cfg.fold} train={len(ds_tr)} val={len(ds_va)}", t0)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={dev} cuda={torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'}", t0)
    net = PartialFT(cfg.dino_id, cfg.unfreeze_last, dropout=cfg.dropout).to(dev)
    log(f"dinov2 dim={net.dim} blocks={net.n_blocks} unfrozen_last={cfg.unfreeze_last} mixup={cfg.mixup} drop={cfg.dropout}", t0)
    params = [
        {"params": [p for n, p in net.named_parameters() if p.requires_grad and n.startswith("enc.")], "lr": cfg.lr_backbone},
        {"params": [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith("enc.")], "lr": cfg.lr_head},
    ]
    opt = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    autocast, scaler = _amp()

    last_p = out / "last.pt"
    best_p = out / "best.pt"
    hist_p = out / "history.json"
    start_epoch = 0
    best = -1.0
    bad = 0
    history = []
    if cfg.resume and last_p.is_file():
        ckpt = torch.load(last_p, map_location=dev, weights_only=False)
        net.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        start_epoch = int(ckpt["epoch"]) + 1
        best = float(ckpt.get("best", -1.0))
        bad = int(ckpt.get("bad", 0))
        history = list(ckpt.get("history", []))
        log(f"resume epoch={start_epoch} best={best:.4f} bad={bad}", t0)
    else:
        log("fresh start (no last.pt)", t0)

    def persist(epoch: int, is_best: bool, extra: dict) -> None:
        payload = {
            "model": net.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best": best, "bad": bad, "fold": cfg.fold,
            "history": history, "cfg": asdict(cfg),
        }
        torch.save(payload, last_p)
        if is_best:
            torch.save({"model": net.state_dict(), "auc": best, "epoch": epoch, "fold": cfg.fold}, best_p)
        (out / "receipt.json").write_text(json.dumps(extra, indent=2) + "\n", encoding="utf-8")
        hist_p.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if commit is not None:
            commit()

    if start_epoch >= cfg.epochs:
        receipt = {"fold": cfg.fold, "best_val_auc": best, "status": "already_done", "epochs_done": start_epoch, "seconds": time.time() - t0}
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        log(json.dumps(receipt), t0)
        return receipt

    nw = 0 if cfg.num_workers <= 0 else cfg.num_workers
    for ep in range(start_epoch, cfg.epochs):
        net.train()
        losses = []
        loader = DataLoader(ds_tr, batch_size=cfg.batch, shuffle=True, num_workers=nw, pin_memory=dev.type == "cuda", drop_last=False)
        for x_b, m_b, t_b, _uid in loader:
            x_b, m_b, t_b = x_b.to(dev, non_blocking=True), m_b.to(dev, non_blocking=True), t_b.to(dev, non_blocking=True)
            if cfg.mixup > 0:
                lam = float(np.random.beta(cfg.mixup, cfg.mixup))
                perm = torch.randperm(x_b.size(0), device=x_b.device)
                x_b = lam * x_b + (1.0 - lam) * x_b[perm]
                t_b = lam * t_b + (1.0 - lam) * t_b[perm]
            opt.zero_grad(set_to_none=True)
            with autocast():
                logit, aux = net(x_b, m_b)
                loss = F.binary_cross_entropy_with_logits(logit, t_b)
                loss = loss + AUX_W * F.binary_cross_entropy_with_logits(aux, t_b)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        net.eval()
        ys, ps, vuids = [], [], []
        with torch.no_grad():
            for x_b, m_b, t_b, uid_b in DataLoader(ds_va, batch_size=cfg.batch, shuffle=False, num_workers=nw):
                x_b, m_b = x_b.to(dev, non_blocking=True), m_b.to(dev, non_blocking=True)
                with autocast():
                    pred = torch.sigmoid(net(x_b, m_b)[0])
                    if getattr(cfg, "tta", False):
                        pred = 0.5 * (pred + torch.sigmoid(net(torch.flip(x_b, dims=[-1]), m_b)[0]))
                    pred = pred.float().cpu().numpy()
                ys.append(t_b.numpy())
                ps.append(pred)
                vuids.extend(list(uid_b))
        yv = np.concatenate(ys)
        pv = np.concatenate(ps)
        auc = macro_auc(yv, pv)
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        row = {"epoch": ep, "loss": mean_loss, "val_auc": auc, "seconds": time.time() - t0}
        history.append(row)
        log(f"ep{ep} loss={mean_loss:.4f} val_auc={auc:.4f}", t0)
        is_best = bool(math.isfinite(auc) and auc > best)
        if is_best:
            best = float(auc)
            bad = 0
            pd.DataFrame(pv, columns=TARGETS).assign(StudyInstanceUID=vuids).to_csv(out / "val_preds.csv", index=False)
        else:
            bad += 1
        receipt = {
            "fold": cfg.fold, "best_val_auc": best, "last_epoch": ep, "last_val_auc": auc,
            "patience_bad": bad, "smoke": cfg.smoke, "img": IMG, "unfreeze_last": cfg.unfreeze_last,
            "epochs_target": cfg.epochs, "n_train": len(ds_tr), "n_val": len(ds_va), "batch": cfg.batch,
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "resumed_from": start_epoch, "seconds": time.time() - t0, "status": "running",
            "mixup": cfg.mixup, "dropout": cfg.dropout, "augment": cfg.augment,
        }
        persist(ep, is_best, receipt)
        if bad >= cfg.patience:
            log(f"early stop patience={cfg.patience} best={best:.4f}", t0)
            receipt["status"] = "early_stop"
            (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            if commit is not None:
                commit()
            break
    else:
        receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
        receipt["status"] = "complete"
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        if commit is not None:
            commit()

    log(json.dumps(receipt), t0)
    return receipt
