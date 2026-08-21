"""Build V29: V27 0.914 in-kernel + 5-fold V28 heads, fail-closed."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"C:\Users\ASUS\Desktop\DanielEmpire\ML\RSNA Knee Abnormality Detection")
V27_DIR = ROOT / "kaggle" / "submission_v27_v25_v24_blend"
DST = ROOT / "kaggle" / "submission_v29_v27_v28"

CHECKPOINT = r'''# V29: freeze V27 0.914 before owned V28 can touch it.
from pathlib import Path as _V29P
import shutil as _V29sh
_V29_WORK = _V29P("/kaggle/working")
_V29_PRIMARY = _V29_WORK / "submission.csv"
_V29_V27 = _V29_WORK / "submission_v27_parent.csv"
if not _V29_PRIMARY.is_file():
    raise RuntimeError("V27 parent submission.csv missing before V28 arm")
_V29sh.copy2(_V29_PRIMARY, _V29_V27)
print("[V29] V27 parent checkpointed", _V29_V27, "bytes", _V29_V27.stat().st_size)
'''

INFER = r'''# V29 owned 5-fold V28 infer + rank-blend onto V27. Fail-closed.
from __future__ import annotations
import json as _j
import shutil as _sh
from pathlib import Path as _P

import cv2
import numpy as _np
import pandas as _pd
import pydicom
import torch
import torch.nn as _nn
import torch.nn.functional as _F

cv2.setNumThreads(1)
_V29_W = 0.20
_TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
_IMG, _CROP, _NS, _NSlice = 224, 160.0, 3, 3
_SLOTS = [("Sagittal", True), ("Coronal", True), ("Axial", True)]
_WORK = _P("/kaggle/working")
_PRI = _WORK / "submission.csv"
_V27 = _WORK / "submission_v27_parent.csv"
_AUDIT = _WORK / "v29_blend_audit.json"
_rec = {"status": "V27_PRESERVED", "w": _V29_W, "error": None}

def _rank(df, cols):
    return df[cols].rank(method="average", pct=True)

def _find_dir(*names):
    root = _P("/kaggle/input")
    if not root.is_dir():
        return None
    hits = [p for p in root.rglob("*") if p.is_dir() and p.name in names]
    return hits[0] if hits else None

def _find_file(name):
    root = _P("/kaggle/input")
    if not root.is_dir():
        return None
    for p in root.rglob(name):
        if p.is_file():
            return p
    return None

class _Head(_nn.Module):
    def __init__(self, cfg_dir, dropout):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(str(cfg_dir), local_files_only=True)
        self.enc = AutoModel.from_config(cfg)
        dim = int(self.enc.config.hidden_size)
        self.slot_proj = _nn.Linear(dim, 256)
        self.query = _nn.Parameter(torch.zeros(len(_TARGETS), 256))
        self.head = _nn.Sequential(_nn.LayerNorm(256), _nn.Linear(256, 128), _nn.GELU(), _nn.Dropout(dropout), _nn.Linear(128, 1))
        self.aux = _nn.Linear(256, len(_TARGETS))

    def forward(self, x, m):
        b, s, c, h, w = x.shape
        inp = x.reshape(b * s, c, h, w)
        rgb = inp if c == 3 else inp.mean(1, keepdim=True).repeat(1, 3, 1, 1)
        rgb = (rgb - 0.45) / 0.22
        tok = self.slot_proj(self.enc(pixel_values=rgb).last_hidden_state[:, 0]).reshape(b, s, -1)
        tok = tok * m.unsqueeze(-1)
        attn = torch.einsum("bsd,td->bst", tok, self.query) / 16.0
        attn = attn.masked_fill(m.unsqueeze(-1) == 0, -1e4)
        wgt = torch.softmax(attn, dim=1)
        pooled = torch.einsum("bst,bsd->btd", wgt, tok)
        return self.head(pooled).squeeze(-1)

def _phys(sdir):
    keyed = []
    for f in sdir.glob("*.dcm"):
        ds = None
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            iop = _np.array(ds.ImageOrientationPatient, dtype=_np.float64).reshape(2, 3)
            key = float(_np.dot(_np.array(ds.ImagePositionPatient, dtype=_np.float64), _np.cross(iop[0], iop[1])))
        except Exception:
            key = float(getattr(ds, "InstanceNumber", 0) or 0) if ds is not None else 0.0
        keyed.append((key, f))
    keyed.sort()
    return [f for _, f in keyed]

def _crop(path):
    try:
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(_np.float32)
        arr *= float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        arr += float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    except Exception:
        return None
    try:
        ps = float(ds.PixelSpacing[0])
    except Exception:
        ps = _CROP / max(arr.shape)
    half = int(round(_CROP / max(ps, 1e-3) / 2))
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    crop = arr[max(0, cy - half):min(arr.shape[0], cy + half), max(0, cx - half):min(arr.shape[1], cx + half)]
    if crop.size == 0:
        return None
    lo, hi = _np.percentile(crop, [1, 99])
    crop = _np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.resize(crop, (_IMG, _IMG), interpolation=cv2.INTER_AREA)

def _pick(rows, plane):
    sub = rows[rows.Anatomical_Plane == plane]
    if sub.empty:
        return None
    scored = sub.copy()
    col = "Fat_Suppression" if "Fat_Suppression" in scored.columns else ("Fluid_Sensitive" if "Fluid_Sensitive" in scored.columns else None)
    scored["_fs"] = _pd.to_numeric(scored[col], errors="coerce").fillna(0) if col else 0
    return scored.sort_values("_fs", ascending=False).iloc[0]

def _study(uid, recs, root):
    x = _np.zeros((_NS, _NSlice, _IMG, _IMG), _np.float32)
    m = _np.zeros(_NS, _np.uint8)
    rows = _pd.DataFrame(recs)
    for i, (plane, _) in enumerate(_SLOTS):
        rec = _pick(rows, plane)
        if rec is None:
            continue
        sdir = root / uid / str(rec.SeriesInstanceUID)
        if not sdir.is_dir():
            continue
        files = _phys(sdir)
        if not files:
            continue
        mid = (int(round(0.20 * (len(files) - 1))) + int(round(0.80 * (len(files) - 1)))) // 2
        picks = [max(0, mid - 1), mid, min(len(files) - 1, mid + 1)]
        for j, p in enumerate(picks[:_NSlice]):
            img = _crop(files[p])
            if img is not None:
                x[i, j] = img
                m[i] = 1
    return x, m

try:
    if not _V27.is_file():
        raise RuntimeError("missing V27 checkpoint")
    heads = _find_dir("rsna-knee-v28-heads", "v28_heads")
    if heads is None:
        for p in _P("/kaggle/input").iterdir() if _P("/kaggle/input").is_dir() else []:
            if (p / "fold0_u6aug" / "best.pt").is_file() or (p / "small" / "config.json").is_file():
                heads = p
                break
    if heads is None:
        raise RuntimeError("v28 heads dataset missing")
    comp = _find_dir("rsna-knee-abnormality-detection")
    if comp is None:
        raise RuntimeError("competition data missing")
    test_csv = _pd.read_csv(comp / "test.csv", dtype={"StudyInstanceUID": str})
    series_csv = None
    for name in ("test_series.csv", "train_series.csv"):
        p = comp / name
        if p.is_file():
            series_csv = _pd.read_csv(p, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
            break
    if series_csv is None:
        raise RuntimeError("no series csv")
    series_root = None
    for name in ("test_series", "train_series", "test_images", "train_images"):
        d = comp / name
        if d.is_dir():
            series_root = d
            break
    if series_root is None:
        raise RuntimeError("no image root")
    recs = {s: g.to_dict("records") for s, g in series_csv.groupby("StudyInstanceUID")}
    uids = test_csv.StudyInstanceUID.astype(str).tolist()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fams = [
        ("u6aug", heads / "small", 0.35),
        ("base", heads / "base", 0.3),
    ]
    pred_fams = []
    for tag, cfg_dir, drop in fams:
        nets = []
        for k in range(5):
            ck = heads / f"fold{k}_{tag}" / "best.pt"
            if not ck.is_file():
                raise RuntimeError(f"missing {ck}")
            net = _Head(cfg_dir, drop).to(dev)
            payload = torch.load(ck, map_location=dev, weights_only=False)
            net.load_state_dict(payload["model"], strict=True)
            net.eval()
            nets.append(net)
        rows = []
        with torch.no_grad():
            for uid in uids:
                xx, mm = _study(uid, recs.get(uid, []), series_root)
                xt = torch.from_numpy(xx).unsqueeze(0).to(dev)
                mt = torch.from_numpy(mm.astype(_np.float32)).unsqueeze(0).to(dev)
                acc = None
                for net in nets:
                    p = torch.sigmoid(net(xt, mt))
                    pf = torch.sigmoid(net(torch.flip(xt, dims=[-1]), mt))
                    cur = 0.5 * (p + pf)
                    acc = cur if acc is None else acc + cur
                acc = (acc / len(nets)).float().cpu().numpy()[0]
                rows.append(acc)
        pred_fams.append(_np.stack(rows, 0))
        print("[V29] family", tag, "rows", len(rows))
    s = _pd.DataFrame(pred_fams[0], columns=_TARGETS)
    b = _pd.DataFrame(pred_fams[1], columns=_TARGETS)
    own = 0.5 * (s.rank(pct=True) + b.rank(pct=True))
    own.insert(0, "StudyInstanceUID", uids)
    v27 = _pd.read_csv(_V27, dtype={"StudyInstanceUID": str})
    cols = [c for c in v27.columns if c != "StudyInstanceUID"]
    own = own.set_index("StudyInstanceUID").reindex(v27.StudyInstanceUID).reset_index()
    if own[cols].isna().any().any():
        raise RuntimeError("V28 missing studies vs V27")
    out = v27.copy()
    out[cols] = (1.0 - _V29_W) * _rank(v27, cols) + _V29_W * _rank(own, cols)
    if not _np.isfinite(out[cols].to_numpy()).all():
        raise RuntimeError("non-finite V29 blend")
    out.to_csv(_PRI, index=False)
    _rec.update(status="V29_BLENDED", rows=int(len(out)), n_test=int(len(uids)))
    print("[V29] blended 0.80*rank(V27)+0.20*rank(V28) rows", len(out))
except Exception as e:
    _rec["status"] = "V29_ERROR_V27_PRESERVED"
    _rec["error"] = f"{type(e).__name__}: {e}"
    print("[V29] failed; restoring V27:", _rec["error"])
    if _V27.is_file():
        _sh.copy2(_V27, _PRI)
_AUDIT.write_text(_j.dumps(_rec, indent=2, sort_keys=True) + "\n")
print(_j.dumps(_rec, indent=2, sort_keys=True))
'''


def main() -> None:
    v27 = json.loads((V27_DIR / "rsna-knee-v27-v25-v24-blend.ipynb").read_text(encoding="utf-8"))
    compile(CHECKPOINT, "v29_ckpt", "exec")
    compile(INFER, "v29_infer", "exec")
    cells = list(v27["cells"])
    for src in (CHECKPOINT, INFER):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "source": [src],
                "outputs": [],
                "execution_count": None,
            }
        )
    nb = dict(v27)
    nb["cells"] = cells
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "rsna-knee-v29-v27-v28.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    meta = json.loads((V27_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    ds = list(meta.get("dataset_sources") or [])
    if "danielsolo1770/rsna-knee-v28-heads" not in ds:
        ds.append("danielsolo1770/rsna-knee-v28-heads")
    ms = list(meta.get("model_sources") or [])
    if "metaresearch/dinov2/PyTorch/base/1" not in ms:
        ms.append("metaresearch/dinov2/PyTorch/base/1")
    meta.update(
        {
            "id": "danielsolo1770/rsna-knee-v29-v27-v28",
            "title": "rsna-knee-v29-v27-v28",
            "code_file": "rsna-knee-v29-v27-v28.ipynb",
            "dataset_sources": ds,
            "model_sources": ms,
        }
    )
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("v29 cells", len(cells), "dst", DST)


if __name__ == "__main__":
    main()
