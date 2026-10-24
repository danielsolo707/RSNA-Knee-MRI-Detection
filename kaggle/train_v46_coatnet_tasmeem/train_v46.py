# V46: fine-tune public CoAtNet-384 SWA on tasmeemreza labels. Gold-58 out.
# eval_first keeps the public baseline if FT hurts. Never writes submission.csv.

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

LAB = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
TAG = "v46"
SEED = 20260905
K_TRAIN = 8
K_EVAL = 24
BS = 4
EPOCHS = 6
BB_LR = 8e-6
HEAD_LR = 2e-4
WD = 0.02
WORKERS = 2
MAX_WALL_H = 8.3
CKPT_EVERY_MIN = 20.0
CKPT_NAME = "raptor_ft_coatnet_v5_full_swa.pt"
OUT = Path("/kaggle/working")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_name(*names, root="/kaggle/input"):
    root = Path(root)
    for p in root.rglob("*"):
        if p.is_file() and p.name in names:
            return p
    return None


class TwoPartVols:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]


def open_corpus():
    av, bv = find_name("all_vols.npy"), find_name("extra_vols.npy")
    am, bm = find_name("all_masks.npy"), find_name("extra_masks.npy")
    if not all((av, bv, am, bm)):
        raise SystemExit("attach both raptor corpus parts")
    vols = TwoPartVols(np.load(av, mmap_mode="r"), np.load(bv, mmap_mode="r"))
    masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
    return vols, masks


def open_ids():
    a, b = find_name("all_ids.npy"), find_name("extra_ids.npy")
    if not (a and b):
        raise SystemExit("missing id npy")
    return np.concatenate([
        np.load(a, allow_pickle=True).astype(str),
        np.load(b, allow_pickle=True).astype(str),
    ])


class StudyWindows(Dataset):
    def __init__(self, ids, id2row, labels, k, train, res):
        self.ids = ids
        self.id2row = id2row
        self.labels = labels
        self.k = k
        self.train = train
        self.res = res
        self.vols = None
        self.masks = None
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.ids)

    def _ensure(self):
        if self.vols is None:
            self.vols, self.masks = open_corpus()

    def _centers(self, valid, d, count):
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs:
            cs = [max(1, min((lo + hi) // 2, d - 2))]
        if self.train:
            pool = (cs * (count // len(cs) + 1))
            random.shuffle(pool)
            return pool[:count]
        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
        return [cs[i] for i in idx]

    def __getitem__(self, i):
        self._ensure()
        uid = self.ids[i]
        row = self.id2row[uid]
        d = self.vols.shape[1]
        m = self.masks[row]
        valid = np.where(m > 0)[0]
        if len(valid) < 3:
            valid = np.arange(min(3, d))
        cs = self._centers(valid, d, self.k)
        vol = self.vols[row]
        tiles = []
        for c in cs:
            c = max(1, min(c, d - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
            t = torch.from_numpy(np.ascontiguousarray(tri))
            if t.shape[-1] != self.res or t.shape[-2] != self.res:
                t = F.interpolate(t[None], size=(self.res, self.res), mode="bilinear", align_corners=False)[0]
            tiles.append(t)
        x = torch.stack(tiles, 0)
        if self.train:
            g = 1.0 + (random.random() - 0.5) * 0.20
            x = (x * g).clamp(0, 1)
        x = (x - self.mean) / self.std
        y = torch.from_numpy(self.labels[uid])
        return x, y


def collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])


class AttnHead(nn.Module):
    def __init__(self, backbone, f_dim, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(f_dim)
        self.att = nn.Sequential(
            nn.Linear(f_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n),
        )
        self.clsW = nn.Parameter(torch.zeros(n, f_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def forward(self, x):
        b, k = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(b, k, -1)
        h = self.norm(f)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


def core(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def main():
    sub = OUT / "submission.csv"
    if sub.exists():
        sub.unlink()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    assert torch.cuda.is_available(), "need GPU"
    cap = torch.cuda.get_device_capability(0)
    assert cap[0] >= 7, f"need sm_70+, got {cap}"
    n_gpu = torch.cuda.device_count()
    log(f"gpu0={torch.cuda.get_device_name(0)} n_gpu={n_gpu} cap={cap}")

    ckpt_path = find_name(CKPT_NAME)
    assert ckpt_path is not None, f"missing {CKPT_NAME}"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    res = int(ck.get("res", 384))
    log(f"init {ckpt_path} arch={arch} res={res} gold_auc={ck.get('gold_auc')}")

    ids = open_ids()
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)

    tcsv = find_name("train.csv")
    assert tcsv is not None
    tr = pd.read_csv(tcsv)
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
    gold_ids = [u for u in gold_df.index if u in idset]
    assert len(gold_ids) == 58, f"expected 58 gold, got {len(gold_ids)}"

    labp = find_name("train_labels_corrected_final.csv")
    assert labp is not None
    soft = pd.read_csv(labp)
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
    soft = soft.set_index("StudyInstanceUID")
    goldset = set(gold_ids)
    train_ids = [u for u in ids if u in soft.index and u not in goldset]
    labels = {u: np.clip(soft.loc[u, LAB].to_numpy(np.float32), 0, 1) for u in train_ids}
    for u in gold_ids:
        labels[u] = gold_df.loc[u, LAB].to_numpy(np.float32)
    log(f"labels={labp} train={len(train_ids)} gold={len(gold_ids)}")

    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device="cuda")

    tds = StudyWindows(train_ids, id2row, labels, K_TRAIN, True, res)
    vds = StudyWindows(gold_ids, id2row, labels, K_EVAL, False, res)
    tl = DataLoader(
        tds, batch_size=BS, shuffle=True, num_workers=WORKERS, drop_last=True,
        collate_fn=collate, pin_memory=True, persistent_workers=WORKERS > 0,
    )
    vl = DataLoader(
        vds, batch_size=max(1, BS // 2), shuffle=False, num_workers=WORKERS,
        collate_fn=collate, persistent_workers=WORKERS > 0,
    )

    bb = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3, global_pool="avg")
    try:
        bb.set_grad_checkpointing(True)
        log("grad checkpointing ON")
    except Exception as e:
        log(f"grad_ckpt skip {e}")
    model = AttnHead(bb, f_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model = model.cuda()
    if n_gpu > 1:
        model = nn.DataParallel(model)
        log(f"DataParallel x{n_gpu}")

    head_params = [p for n, p in core(model).named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    bb_params = [p for n, p in core(model).named_parameters() if n.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": bb_params, "lr": BB_LR}, {"params": head_params, "lr": HEAD_LR}],
        weight_decay=WD,
    )
    steps = max(len(tl) * EPOCHS, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[BB_LR, HEAD_LR], total_steps=steps, pct_start=0.15,
    )
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    scaler = torch.amp.GradScaler("cuda")
    amp_dtype = torch.float16

    best = -1.0
    best_state = None
    best_pack = None
    hist = []
    baseline = None
    patience = 3
    bad = 0

    @torch.no_grad()
    def evaluate():
        model.eval()
        ps, ys = [], []
        for x, y in vl:
            with torch.autocast("cuda", dtype=amp_dtype):
                o = torch.sigmoid(model(x.cuda(non_blocking=True)).float())
            ps.append(o.cpu().numpy())
            ys.append(y.numpy())
        p = np.concatenate(ps)
        y = np.concatenate(ys)
        aucs = {}
        for j, name in enumerate(LAB):
            yb = (y[:, j] >= 0.5).astype(int)
            if len(set(yb)) > 1:
                aucs[name] = float(roc_auc_score(yb, p[:, j]))
        macro = float(np.mean(list(aucs.values()))) if aucs else float("nan")
        return macro, aucs, p, y

    def out_of_time(reserve_s=0.0):
        return (time.time() - T0) + reserve_s > MAX_WALL_H * 3600.0

    def dump_state(path, extra):
        st = {
            "model": {k: v.detach().cpu() for k, v in core(model).state_dict().items()},
            "arch": arch,
            "res": res,
            "lab": LAB,
            "tag": TAG,
            **extra,
        }
        tmp = str(path) + ".tmp"
        torch.save(st, tmp)
        os.replace(tmp, path)
        return st

    def write_receipt(status, **kw):
        rec = {
            "tag": TAG,
            "status": status,
            "arch": arch,
            "res": res,
            "k_train": K_TRAIN,
            "k_eval": K_EVAL,
            "bs": BS,
            "n_train": len(train_ids),
            "n_gold": len(gold_ids),
            "labels": "tasmeemreza/train_labels_corrected_final.csv",
            "init": str(ckpt_path),
            "gold_held_out": True,
            "baseline_gold_auc": baseline,
            "best_gold_auc": best,
            "elapsed_h": round((time.time() - T0) / 3600.0, 3),
            "hist": hist,
            **kw,
        }
        (OUT / "receipt.json").write_text(json.dumps(rec, indent=2) + "\n")
        (OUT / f"raptor_ft_{TAG}.json").write_text(json.dumps(rec, indent=2) + "\n")

    au0, aucs0, p0, y0 = evaluate()
    baseline = au0
    best = au0
    best_pack = (p0, y0)
    best_state = dump_state(
        OUT / f"raptor_ft_{TAG}.pt",
        {"gold_auc": au0, "aucs": aucs0, "epoch": -1, "note": "eval_first"},
    )
    np.savez(OUT / f"raptor_gold_{TAG}.npz", pred=p0, truth=y0, ids=np.array(gold_ids))
    log(f"eval_first BASELINE gold={au0:.4f}")
    log("per-target " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(aucs0.items(), key=lambda kv: kv[1])))
    write_receipt("eval_first", epochs_done=0, aucs=aucs0)

    last_ckpt_t = time.time()
    stopped = False
    for ep in range(EPOCHS):
        if ep > 0 and out_of_time(1800):
            log(f"wall stop before ep{ep}")
            stopped = True
            break
        model.train()
        tot = 0.0
        t_ep = time.time()
        for it, (x, y) in enumerate(tl):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = lossf(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            if (it + 1) % 20 == 0 or (it + 1) == len(tl):
                el = time.time() - t_ep
                eta = el / (it + 1) * (len(tl) - it - 1)
                log(f"ep{ep} {it+1}/{len(tl)} loss={tot/(it+1):.4f} eta={eta/60:.1f}min")
            if time.time() - last_ckpt_t > CKPT_EVERY_MIN * 60:
                dump_state(OUT / f"raptor_ft_{TAG}_mid.pt", {"gold_auc": best, "epoch": ep, "step": it + 1})
                last_ckpt_t = time.time()
                log("mid ckpt")
            if out_of_time(480):
                dump_state(OUT / f"raptor_ft_{TAG}_mid.pt", {"gold_auc": best, "epoch": ep, "step": it + 1})
                log("wall mid-epoch")
                stopped = True
                break
        if stopped:
            break
        au, aucs, p, yv = evaluate()
        hist.append({"ep": ep, "loss": tot / max(len(tl), 1), "gold_auc": au, "aucs": aucs})
        improved = au > best
        if improved:
            best = au
            bad = 0
            best_state = dump_state(
                OUT / f"raptor_ft_{TAG}.pt",
                {"gold_auc": au, "aucs": aucs, "epoch": ep},
            )
            best_pack = (p, yv)
            np.savez(OUT / f"raptor_gold_{TAG}.npz", pred=p, truth=yv, ids=np.array(gold_ids))
            write_receipt("running", epochs_done=ep + 1, aucs=aucs)
        else:
            bad += 1
        log(f"ep{ep} GOLD {au:.4f} best={best:.4f} base={baseline:.4f} {'SAVED' if improved else f'pat={bad}'}")
        if bad >= patience:
            log(f"early stop patience={patience}")
            break

    status = "complete" if not stopped else "wall-stopped"
    if best_state is not None:
        torch.save(best_state, OUT / f"raptor_ft_{TAG}.pt")
        if best_pack is not None:
            np.savez(OUT / f"raptor_gold_{TAG}.npz", pred=best_pack[0], truth=best_pack[1], ids=np.array(gold_ids))
    write_receipt(status, epochs_done=len(hist), aucs=(best_state or {}).get("aucs", {}))
    log(f"DONE status={status} best_gold={best:.4f} baseline={baseline:.4f} delta={best - (baseline or 0):+.4f}")
    if best_state and "aucs" in best_state:
        for k, v in best_state["aucs"].items():
            log(f"  {k:18s} {v:.3f}")
    if sub.exists():
        sub.unlink()


if __name__ == "__main__":
    main()
