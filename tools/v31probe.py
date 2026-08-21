"""V31-probe: DINOv2-small FULL unfreeze + LLRD + EMA + cosine. Cache-only.

Local/Colab-agnostic: everything is driven by Cfg paths.
Proven on Colab T4 (2026-08-21): V30 fold0 -> best 0.8201 (EMA@ep16, early stop ep23).
"""
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

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SEED = 20260821
AUX_W = 0.25
N_FOLDS = 5


def log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def report_group(uid):
    return hashlib.md5(uid.encode()).hexdigest()[:8]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Cfg:
    cache_dir: str = "/content/v30_cache"
    out_dir: str = "/content/drive/MyDrive/rsna-knee-v31/fold0_fullft"
    fold: int = 0
    epochs: int = 30
    batch: int = 8
    accum: int = 2
    lr_head: float = 3e-4
    lr_bb_top: float = 5e-5
    llrd: float = 0.7
    weight_decay: float = 1e-4
    patience: int = 7
    warmup: int = 2
    dropout: float = 0.3
    mixup: float = 0.2
    ema_decay: float = 0.999
    tta: bool = True
    num_workers: int = 4
    seed: int = SEED
    dino_id: str = "facebook/dinov2-small"


class CacheDS(Dataset):
    def __init__(self, x, mask, y, uids, idx, augment=False):
        self.x, self.mask, self.y, self.uids = x, mask, y, uids
        self.idx = np.asarray(idx, dtype=np.int64)
        self.augment = bool(augment)

    def __len__(self):
        return int(len(self.idx))

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = np.asarray(self.x[j], dtype=np.float32)          # (S,1,H,W)
        m = np.asarray(self.mask[j], dtype=np.float32)
        y = np.asarray(self.y[j], dtype=np.float32)
        if x.ndim == 4 and x.shape[1] == 1:
            x = np.repeat(x, 3, axis=1)                       # -> (S,3,H,W)
        if self.augment:
            if random.random() < 0.5:
                x = np.ascontiguousarray(x[..., :, ::-1])
            if random.random() < 0.7:
                x = np.clip(x * np.float32(random.uniform(0.85, 1.15)), 0, 1)
            if random.random() < 0.4:
                x = np.clip(x + np.float32(random.uniform(-0.06, 0.06)), 0, 1)
        return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(y), self.uids[j]


def load_cache(cache_dir: Path):
    man = json.loads((cache_dir / "manifest.json").read_text())
    n = int(man["n"])
    shape = tuple(man["x_shape"])
    x = np.memmap(cache_dir / "x_f16.dat", dtype=np.float16, mode="r", shape=shape)
    n_slot = int(shape[1])
    mask = np.memmap(cache_dir / "mask.dat", dtype=np.uint8, mode="r", shape=(n, n_slot))
    y = np.load(cache_dir / "y.npy")
    uids = (cache_dir / "uids.txt").read_text().splitlines()
    assert len(uids) == n and y.shape[0] == n
    return man, x, mask, y, uids


class FullFT(nn.Module):
    """DINOv2 full unfreeze; attention-MIL over slot CLS tokens (same head family as V28)."""

    def __init__(self, dino_id, dropout=0.3):
        super().__init__()
        from transformers import AutoModel

        self.enc = AutoModel.from_pretrained(dino_id)
        dim = int(self.enc.config.hidden_size)
        self.n_blocks = len(self.enc.encoder.layer)
        self.slot_proj = nn.Linear(dim, 256)
        self.query = nn.Parameter(torch.randn(len(TARGETS), 256) * 0.02)
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(128, 1))
        self.aux = nn.Linear(256, len(TARGETS))
        self.dim = dim

    def forward(self, x, m):
        b, s, c, h, w = x.shape
        rgb = x.reshape(b * s, c, h, w)
        rgb = (rgb - 0.45) / 0.22
        out = self.enc(pixel_values=rgb)
        tok = self.slot_proj(out.last_hidden_state[:, 0]).reshape(b, s, -1)
        tok = tok * m.unsqueeze(-1)
        attn = torch.einsum("bsd,td->bst", tok, self.query) / 16.0
        attn = attn.masked_fill(m.unsqueeze(-1) == 0, -1e4)
        wgt = torch.softmax(attn, dim=1)
        pooled = torch.einsum("bst,bsd->btd", wgt, tok)
        return self.head(pooled).squeeze(-1), self.aux(tok.mean(1))


def build_llrd_opt(net, cfg):
    groups = []
    enc_blocks = net.enc.encoder.layer
    nB = len(enc_blocks)
    emb_lr = cfg.lr_bb_top * (cfg.llrd ** nB)
    groups.append({"params": list(net.enc.embeddings.parameters()), "lr": emb_lr})
    for i, blk in enumerate(enc_blocks):
        groups.append({"params": list(blk.parameters()),
                       "lr": cfg.lr_bb_top * (cfg.llrd ** (nB - 1 - i))})
    if getattr(net.enc, "layernorm", None) is not None:
        groups.append({"params": list(net.enc.layernorm.parameters()), "lr": cfg.lr_bb_top})
    heads = [p for n, p in net.named_parameters() if not n.startswith("enc.")]
    groups.append({"params": heads, "lr": cfg.lr_head})
    for g in groups:
        g["weight_decay"] = cfg.weight_decay
    return torch.optim.AdamW(groups)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        sd = model.state_dict()
        self.shadow = {k: v.detach().clone().float() for k, v in sd.items()
                       if v.dtype.is_floating_point}
        self.buffers = {k: v.detach().clone() for k, v in sd.items()
                        if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.buffers[k] = v.detach().clone()

    def copy_to(self, model):
        sd = model.state_dict()
        new = {}
        for k, v in sd.items():
            if k in self.shadow:
                new[k] = self.shadow[k].to(v.dtype)
            else:
                new[k] = self.buffers.get(k, v)
        model.load_state_dict(new)


def macro_auc(y, p):
    sc = []
    for j in range(y.shape[1]):
        yy = (y[:, j] > 0.5).astype(int)
        if len(np.unique(yy)) < 2:
            continue
        try:
            sc.append(roc_auc_score(yy, p[:, j]))
        except Exception:
            pass
    return float(np.mean(sc)) if sc else float("nan")


def cosine_lr(opt, ep, total, warmup, base_lrs):
    if ep < warmup:
        f = (ep + 1) / max(1, warmup)
    else:
        t = (ep - warmup) / max(1, total - warmup)
        f = 0.5 * (1 + math.cos(math.pi * t))
    for g, base in zip(opt.param_groups, base_lrs):
        g["lr"] = base * f


def main(cfg=Cfg()):
    t0 = time.time()
    set_seed(cfg.seed + cfg.fold)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man, x, mask, y, uids = load_cache(Path(cfg.cache_dir))
    log(f"cache n={len(uids)} shape={tuple(x.shape)} recipe={man.get('recipe')}", t0)
    groups = np.array([report_group(u) for u in uids])
    folds = list(GroupKFold(N_FOLDS).split(np.arange(len(uids)), groups=groups))
    tr_idx, va_idx = folds[cfg.fold]
    ds_tr = CacheDS(x, mask, y, uids, tr_idx, augment=True)
    ds_va = CacheDS(x, mask, y, uids, va_idx, augment=False)
    log(f"fold{cfg.fold} train={len(ds_tr)} val={len(ds_va)}", t0)
    dev = torch.device("cuda")
    net = FullFT(cfg.dino_id, cfg.dropout).to(dev)
    for p in net.enc.parameters():
        p.requires_grad = True
    log(f"dino dim={net.dim} blocks={net.n_blocks} UNFROZEN=ALL(LLRD {cfg.llrd})", t0)
    opt = build_llrd_opt(net, cfg)
    base_lrs = [g["lr"] for g in opt.param_groups]
    autocast = lambda: torch.amp.autocast("cuda")
    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(net, cfg.ema_decay)
    ema_net = FullFT(cfg.dino_id, cfg.dropout).to(dev)
    ema_net.eval()
    last_p, best_p = out / "last.pt", out / "best.pt"
    start_ep, best, bad, history = 0, -1.0, 0, []
    if last_p.is_file():
        ck = torch.load(last_p, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_ep = int(ck["epoch"]) + 1
        best = float(ck.get("best", -1))
        bad = int(ck.get("bad", 0))
        history = list(ck.get("history", []))
        if "ema" in ck:
            ema.shadow = {k: v.to(dev) for k, v in ck["ema"].items()}
        log(f"resume ep={start_ep} best={best:.4f}", t0)
    nw = cfg.num_workers
    for ep in range(start_ep, cfg.epochs):
        cosine_lr(opt, ep, cfg.epochs, cfg.warmup, base_lrs)
        net.train()
        losses = []
        t_ep = time.time()
        loader = DataLoader(ds_tr, batch_size=cfg.batch, shuffle=True, num_workers=nw,
                            pin_memory=True, drop_last=False, persistent_workers=nw > 0)
        opt.zero_grad(set_to_none=True)
        for it, (xb, mb, tb, _u) in enumerate(loader):
            xb = xb.to(dev, non_blocking=True)
            mb = mb.to(dev, non_blocking=True)
            tb = tb.to(dev, non_blocking=True)
            if cfg.mixup > 0:
                lam = float(np.random.beta(cfg.mixup, cfg.mixup))
                perm = torch.randperm(xb.size(0), device=dev)
                xb = lam * xb + (1 - lam) * xb[perm]
                tb = lam * tb + (1 - lam) * tb[perm]
            with autocast():
                lg, aux = net(xb, mb)
                loss = (F.binary_cross_entropy_with_logits(lg, tb)
                        + AUX_W * F.binary_cross_entropy_with_logits(aux, tb)) / cfg.accum
            scaler.scale(loss).backward()
            if (it + 1) % cfg.accum == 0 or (it + 1) == len(loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(net)
            losses.append(float(loss.detach()) * cfg.accum)

        def evaluate(model):
            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for xb, mb, tb, _u in DataLoader(ds_va, batch_size=cfg.batch, shuffle=False,
                                                 num_workers=nw, pin_memory=True):
                    xb = xb.to(dev)
                    mb = mb.to(dev)
                    with autocast():
                        pr = torch.sigmoid(model(xb, mb)[0])
                        if cfg.tta:
                            pr = 0.5 * (pr + torch.sigmoid(model(torch.flip(xb, dims=[-1]), mb)[0]))
                        ps.append(pr.float().cpu().numpy())
                    ys.append(tb.numpy())
            return macro_auc(np.concatenate(ys), np.concatenate(ps))

        auc_raw = evaluate(net)
        ema.copy_to(ema_net)
        auc_ema = evaluate(ema_net)
        auc = max(auc_raw, auc_ema)
        use_ema = auc_ema >= auc_raw
        row = {"epoch": ep, "loss": round(float(np.mean(losses)), 4),
               "val_auc_raw": round(auc_raw, 4), "val_auc_ema": round(auc_ema, 4),
               "val_auc": round(auc, 4), "ep_min": round((time.time() - t_ep) / 60, 1),
               "used_ema": use_ema}
        history.append(row)
        log(json.dumps(row), t0)
        is_best = bool(math.isfinite(auc) and auc > best)
        if is_best:
            best = float(auc)
            bad = 0
            src = ema_net if use_ema else net
            torch.save({"model": src.state_dict(), "auc": best, "epoch": ep,
                        "fold": cfg.fold, "used_ema": use_ema}, best_p)
        else:
            bad += 1
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                    "best": best, "bad": bad, "history": history,
                    "ema": {k: v.cpu() for k, v in ema.shadow.items()},
                    "cfg": asdict(cfg)}, last_p)
        (out / "history.json").write_text(json.dumps(history, indent=1))
        (out / "receipt.json").write_text(json.dumps(
            {"fold": cfg.fold, "best_val_auc": best, "last_epoch": ep,
             "status": "running", "cfg": asdict(cfg)}, indent=1))
        if bad >= cfg.patience:
            log(f"early stop best={best:.4f}", t0)
            break
    (out / "receipt.json").write_text(json.dumps(
        {"fold": cfg.fold, "best_val_auc": best, "status": "complete",
         "epochs_done": len(history), "cfg": asdict(cfg)}, indent=1))
    log(f"DONE best={best:.4f}", t0)


if __name__ == "__main__":
    main()
