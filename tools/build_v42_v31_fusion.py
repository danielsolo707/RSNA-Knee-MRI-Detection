"""Build V42: V33 (the 0.936 submission kernel) + our own v31 5-fold DINOv2 arm.

Route (a) from the V41 verdict: gold-58 cannot validate fusion changes to V33
(V33 was tuned on gold), so the only clean measurement is the hidden test itself.
V42 adds a genuinely new prediction vector: our 5-fold DINOv2-S arm (280 px,
21-slot geometry, trained on the hard bank labels - honest gold OOF 0.8305,
decorrelated from every V33 branch by architecture AND training labels).

Structure (V33 cells untouched; three cells appended):
  snapshot:  submission.csv -> submission_v33_final.csv
  v31 arm:   build the 280px/21-slot TEST cache from DICOM, run the 5 folds
             (TTA flip), write submission_v31.csv        (fail-soft)
  fusion:    submission.csv = (1-w)*rank(V33) + w*rank(v31), w = 0.10 flat
             (V37 lesson: global weights only, never per-target fits)

If the v31 arm fails for any reason, submission.csv stays exactly V33.

Run: python tools/build_v42_v31_fusion.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "kaggle" / "submission_v33_dino_raptor_0936"
DST = REPO / "kaggle" / "submission_v42_v31_fusion"
V31_W = 0.10

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]

SNAPSHOT = '''# ---- V42: snapshot the pure V33 output before blending ----
import shutil as _v42_sh
_v42_sh.copy('/kaggle/working/submission.csv', '/kaggle/working/submission_v33_final.csv')
print('[v42] snapshotted V33 output -> submission_v33_final.csv', flush=True)
'''

V31_ARM = '''# ---- V42: our own v31 arm (5-fold DINOv2-S, 280px/21-slot) ----
# Fail-soft: any error leaves submission.csv = pure V33.
try:
    import gc, time
    from pathlib import Path as _v42_Path
    import cv2 as _v42_cv2, numpy as _v42_np, pandas as _v42_pd, pydicom as _v42_pydicom
    import torch as _v42_torch, torch.nn as _v42_nn
    _v42_cv2.setNumThreads(1)
    _T0 = time.time()

    def _log(m):
        print(f"[v31 {time.time()-_T0:6.0f}s] {m}", flush=True)

    IMG = 280
    CROP_MM = 140.0
    N_ANCHOR = 7
    N_SLOT = 3 * N_ANCHOR
    SLOTS = [("Sagittal", True), ("Coronal", True), ("Axial", True)]
    TARGETS = ''' + json.dumps(TARGETS) + '''

    def _find_dir(*names):
        root = _v42_Path("/kaggle/input")
        hits = []
        if root.is_dir():
            for p in root.rglob("*"):
                if p.is_dir() and p.name in names:
                    hits.append(p)
        return hits[0] if hits else None

    COMP = _find_dir("rsna-knee-abnormality-detection")
    assert COMP is not None, "competition data missing"
    SERIES_ROOT = COMP / "test_series" if (COMP / "test_series").is_dir() else COMP / "train_series"
    test = _v42_pd.read_csv(COMP / "test.csv")
    uids = test["StudyInstanceUID"].astype(str).tolist()
    ser = _v42_pd.read_csv(SERIES_ROOT.parent / (SERIES_ROOT.name + ".csv"),
                           dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    by = {u: g.to_dict("records") for u, g in
          ser[ser.StudyInstanceUID.isin(set(uids))].groupby("StudyInstanceUID")}
    _log(f"test studies {len(uids)} | with series metadata {len(by)}")

    def physical_order(sdir):
        keyed = []
        for f in sdir.glob("*.dcm"):
            ds = None
            try:
                ds = _v42_pydicom.dcmread(str(f), stop_before_pixels=True)
                iop = _v42_np.array(ds.ImageOrientationPatient, dtype=_v42_np.float64).reshape(2, 3)
                normal = _v42_np.cross(iop[0], iop[1])
                ipp = _v42_np.array(ds.ImagePositionPatient, dtype=_v42_np.float64)
                key = float(_v42_np.dot(ipp, normal))
            except Exception:
                try:
                    key = float(getattr(ds, "InstanceNumber", 0) or 0) if ds is not None else 0.0
                except Exception:
                    key = 0.0
            keyed.append((key, f))
        keyed.sort()
        return [f for _, f in keyed]

    def read_crop(path):
        try:
            ds = _v42_pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(_v42_np.float32)
            arr *= float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            arr += float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        except Exception:
            return None
        try:
            ps = float(ds.PixelSpacing[0])
        except Exception:
            ps = CROP_MM / max(arr.shape)
        half = int(round(CROP_MM / max(ps, 1e-3) / 2))
        cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
        y0, y1 = max(0, cy - half), min(arr.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(arr.shape[1], cx + half)
        crop = arr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        lo, hi = _v42_np.percentile(crop, [1, 99])
        crop = _v42_np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1)
        return _v42_cv2.resize(crop, (IMG, IMG), interpolation=_v42_cv2.INTER_AREA)

    def pick_fs(series_df, plane):
        sub = series_df[series_df.Anatomical_Plane == plane]
        if sub.empty:
            return None
        scored = sub.copy()
        if "Fat_Suppression" in scored.columns:
            scored["_fs"] = _v42_pd.to_numeric(scored.Fat_Suppression, errors="coerce").fillna(0)
        elif "Fluid_Sensitive" in scored.columns:
            scored["_fs"] = _v42_pd.to_numeric(scored.Fluid_Sensitive, errors="coerce").fillna(0)
        else:
            scored["_fs"] = 0
        return scored.sort_values("_fs", ascending=False).iloc[0]

    def build_study(uid, recs):
        x = _v42_np.zeros((N_SLOT, 1, IMG, IMG), _v42_np.float16)
        m = _v42_np.zeros(N_SLOT, _v42_np.uint8)
        rows = _v42_pd.DataFrame(recs)
        for pi, (plane, _fs) in enumerate(SLOTS):
            rec = pick_fs(rows, plane)
            if rec is None:
                continue
            sdir = SERIES_ROOT / uid / str(rec.SeriesInstanceUID)
            if not sdir.is_dir():
                continue
            files = physical_order(sdir)
            if not files:
                continue
            lo, hi = 0.10, 0.90
            i0 = int(round(lo * (len(files) - 1)))
            i1 = int(round(hi * (len(files) - 1)))
            span = max(i1 - i0, 1)
            anchors = [i0 + int(round(a * span / (N_ANCHOR - 1))) for a in range(N_ANCHOR)]
            ok = False
            for ai, idx in enumerate(anchors):
                img = read_crop(files[idx])
                if img is not None:
                    x[pi * N_ANCHOR + ai, 0] = img.astype(_v42_np.float16)
                    m[pi * N_ANCHOR + ai] = 1
                    ok = True
            if not ok:
                continue
        return x, m

    n = len(uids)
    X = _v42_np.zeros((n, N_SLOT, 1, IMG, IMG), _v42_np.float16)
    M = _v42_np.zeros((n, N_SLOT), _v42_np.uint8)
    for i, u in enumerate(uids):
        X[i], M[i] = build_study(u, by.get(u, []))
        if (i + 1) % 100 == 0 or (i + 1) == n:
            _log(f"cache {i+1}/{n} | fill {M[:i+1].mean():.3f}")

    class FullFT(_v42_nn.Module):
        """Inference-only copy of the v31 trainer head (no grad ckpt, no SDPA patch)."""
        def __init__(self, dino_path, dropout=0.3):
            super().__init__()
            from transformers import AutoModel
            self.enc = AutoModel.from_pretrained(dino_path)
            dim = int(self.enc.config.hidden_size)
            self.slot_proj = _v42_nn.Linear(dim, 256)
            self.query = _v42_nn.Parameter(_v42_torch.randn(len(TARGETS), 256) * 0.02)
            self.head = _v42_nn.Sequential(_v42_nn.LayerNorm(256), _v42_nn.Linear(256, 128),
                                           _v42_nn.GELU(), _v42_nn.Dropout(dropout),
                                           _v42_nn.Linear(128, 1))
            self.aux = _v42_nn.Linear(256, len(TARGETS))

        def forward(self, x, m):
            b, s, c, h, w = x.shape
            rgb = x.reshape(b * s, c, h, w)
            rgb = (rgb - 0.45) / 0.22
            out = self.enc(pixel_values=rgb)
            tok = self.slot_proj(out.last_hidden_state[:, 0]).reshape(b, s, -1)
            tok = tok * m.unsqueeze(-1)
            attn = _v42_torch.einsum("bsd,td->bst", tok, self.query) / 16.0
            attn = attn.masked_fill(m.unsqueeze(-1) == 0, -1e4)
            wgt = _v42_torch.softmax(attn, dim=1)
            pooled = _v42_torch.einsum("bst,bsd->btd", wgt, tok)
            return self.head(pooled).squeeze(-1)

    DINO = _v42_Path("/kaggle/input/models/metaresearch/dinov2/pytorch/small/1")
    assert (DINO / "config.json").is_file(), "dinov2-small model dataset not attached"
    W = _find_dir("rsna-knee-v31-weights")
    assert W is not None, "rsna-knee-v31-weights dataset not attached"
    dev = _v42_torch.device("cuda" if _v42_torch.cuda.is_available() else "cpu")
    autocast = _v42_torch.amp.autocast("cuda")
    BATCH = 8
    P_all = _v42_np.zeros((n, len(TARGETS)), _v42_np.float64)

    def _predict(net, i0, i1):
        xb = _v42_torch.from_numpy(X[i0:i1].astype(_v42_np.float32)).to(dev)
        xb = xb.repeat(1, 1, 3, 1, 1)
        mb = _v42_torch.from_numpy(M[i0:i1].astype(_v42_np.float32)).to(dev)
        with autocast:
            pr = _v42_torch.sigmoid(net(xb, mb))
            pr = 0.5 * (pr + _v42_torch.sigmoid(net(_v42_torch.flip(xb, dims=[-1]), mb)))
        return pr.float().cpu().numpy()

    for f in range(5):
        ck = _v42_torch.load(W / f"fold{f}_best.pt", map_location="cpu", weights_only=False)
        net = FullFT(str(DINO)).to(dev).eval()
        net.load_state_dict(ck["model"], strict=True)
        ps = []
        with _v42_torch.no_grad():
            for i0 in range(0, n, BATCH):
                try:
                    ps.append(_predict(net, i0, min(i0 + BATCH, n)))
                except RuntimeError:
                    _v42_torch.cuda.empty_cache()
                    for j in range(i0, min(i0 + BATCH, n)):
                        ps.append(_predict(net, j, j + 1))
        P_all += _v42_np.concatenate(ps)
        del net, ck, ps
        gc.collect()
        if dev.type == "cuda":
            _v42_torch.cuda.empty_cache()
        _log(f"fold {f} done")

    sub = _v42_pd.DataFrame({"StudyInstanceUID": uids})
    sub[TARGETS] = P_all / 5.0
    sub.to_csv("/kaggle/working/submission_v31.csv", index=False)
    _v42_np.savez("/kaggle/working/v31_arm_preds.npz", pred=P_all / 5.0,
                  ids=_v42_np.array(uids), lab=_v42_np.array(TARGETS))
    _log("wrote submission_v31.csv + v31_arm_preds.npz")
except Exception as _v31_err:
    import traceback as _v31_tb
    print(f"[v42] V31 ARM FAILED - keeping pure V33 submission: "
          f"{type(_v31_err).__name__}: {_v31_err}", flush=True)
    _v31_tb.print_exc()
'''

FUSION = '''# ---- V42 fusion: (1-w)*rank(V33) + w*rank(v31), flat global weight ----
import numpy as _fu_np
import pandas as _fu_pd

V31_W = ''' + str(V31_W) + '''
LAB = ''' + json.dumps(TARGETS) + '''

_v31_path = '/kaggle/working/submission_v31.csv'
import os as _fu_os
if _fu_os.path.isfile(_v31_path):
    v33 = _fu_pd.read_csv('/kaggle/working/submission_v33_final.csv',
                          dtype={'StudyInstanceUID': str})
    v31 = _fu_pd.read_csv(_v31_path, dtype={'StudyInstanceUID': str})
    v31 = v31.set_index('StudyInstanceUID').loc[v33['StudyInstanceUID'], :]
    r33 = v33[LAB].rank(method='average', pct=True)
    r31 = v31[LAB].rank(method='average', pct=True)
    corr = {t: float(_fu_np.corrcoef(r33[t], r31[t])[0, 1]) for t in LAB}
    out = v33.copy()
    out[LAB] = (1 - V31_W) * r33.to_numpy() + V31_W * r31.to_numpy()
    if not _fu_np.isfinite(out[LAB].to_numpy()).all():
        raise RuntimeError('non-finite blend values - refusing to write')
    out.to_csv('/kaggle/working/submission.csv', index=False)
    print('[v42] v33~v31 rank corr per target:',
          {k: round(v, 3) for k, v in corr.items()}, flush=True)
    print(f'[v42] final submission.csv = {1-V31_W:.2f}*rank(V33) + {V31_W:.2f}*rank(v31)', flush=True)
else:
    print('[v42] no v31 predictions - submission.csv remains pure V33', flush=True)
'''


def main() -> None:
    nb = json.loads((SRC / "rsna-knee-v33-dino-raptor-0936.ipynb").read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 4, f"expected 4 V33 code cells, got {len(code_cells)}"

    def cell(src: str, tag: str) -> dict:
        compile(src, tag, "exec")
        return {"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True),
                "outputs": [], "execution_count": None}

    new_cells = [cell(SNAPSHOT, "v42_snap"), cell(V31_ARM, "v42_arm"), cell(FUSION, "v42_fuse")]
    md = {"cell_type": "markdown", "metadata": {}, "source": [
        "# V42: V33 (0.936) + our own v31 5-fold DINOv2 arm\n",
        "\n",
        "V33 runs untouched; then our owned 5-fold DINOv2-S arm (280px/21-slot, trained on the\n",
        "hard bank labels, honest gold OOF 0.8305) is predicted on the test studies and blended\n",
        f"into the final submission at a flat rank weight w={V31_W}. Fail-soft: if the v31 arm\n",
        "fails, submission.csv stays pure V33. Per-arm predictions are preserved for offline\n",
        "analysis (submission_v33_final.csv, submission_v31.csv, v31_arm_preds.npz).\n",
    ]}
    nb["cells"] = nb["cells"] + [md] + new_cells
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "rsna-knee-v42-v31-fusion.ipynb").write_text(json.dumps(nb), encoding="utf-8")

    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = "danielsolo1770/rsna-knee-v42-v31-fusion"
    meta["title"] = "rsna-knee-v42-v31-fusion"
    meta["code_file"] = "rsna-knee-v42-v31-fusion.ipynb"
    meta["enable_internet"] = False
    ds = meta.setdefault("dataset_sources", [])
    if "danielsolo1770/rsna-knee-v31-weights" not in ds:
        ds.append("danielsolo1770/rsna-knee-v31-weights")
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("V42 fusion kernel ready:", DST)
    print(f"  blend weight w={V31_W} (flat, global - V37 lesson)")
    print("  datasets:", ds)
    print("  models:", meta.get("model_sources"))


if __name__ == "__main__":
    main()
