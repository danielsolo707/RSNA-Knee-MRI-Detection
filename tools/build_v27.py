"""Build V27: V25 incumbent (0.913) + DINOsaur V24 rank-blend w=0.20, fail-closed."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"C:\Users\ASUS\Desktop\DanielEmpire\ML\RSNA Knee Abnormality Detection")
V25_DIR = ROOT / "kaggle" / "submission_v25_v16_dinov3"
V24_DIR = ROOT / "kaggle" / "submission_v24_dinosaur_v3"
DST = ROOT / "kaggle" / "submission_v27_v25_v24_blend"

CHECKPOINT = r'''# V27 checkpoint: freeze the V25 0.913 file before DINOsaur can touch it.
from pathlib import Path as _V27Path
import shutil as _V27sh
_V27_WORK = _V27Path("/kaggle/working")
_V27_PRIMARY = _V27_WORK / "submission.csv"
_V27_V25 = _V27_WORK / "submission_v25_parent.csv"
if not _V27_PRIMARY.is_file():
    raise RuntimeError("V25 parent submission.csv missing before dinosaur arm")
_V27sh.copy2(_V27_PRIMARY, _V27_V25)
print("[V27] V25 parent checkpointed", _V27_V25, "bytes", _V27_V25.stat().st_size)
'''

BLEND = r'''# V27: rank-blend DINOsaur onto the V25 parent at pre-registered w=0.20.
# Fail-closed: any error restores the V25 0.913 file.
import json as _v27_json
import shutil as _v27_sh
from pathlib import Path as _V27P

import numpy as _v27_np
import pandas as _v27_pd

_V27_W = 0.20
_V27_WORK = _V27P("/kaggle/working")
_V27_PRIMARY = _V27_WORK / "submission.csv"
_V27_V25 = _V27_WORK / "submission_v25_parent.csv"
_V27_V24 = _V27_WORK / "submission_v24_dinosaur.csv"
_V27_AUDIT = _V27_WORK / "v27_blend_audit.json"
_v27_rec = {"status": "V25_PRESERVED", "w": _V27_W, "error": None}

def _v27_rank(df, cols):
    return df[cols].rank(method="average", pct=True)

try:
    if not globals().get("_V27_DINO_OK"):
        raise RuntimeError("dinosaur arm did not finish")
    if not _V27_V25.is_file():
        raise RuntimeError("missing V25 checkpoint")
    if _V27_PRIMARY.is_file():
        _v27_sh.copy2(_V27_PRIMARY, _V27_V24)
    v25 = _v27_pd.read_csv(_V27_V25, dtype={"StudyInstanceUID": str})
    v24 = _v27_pd.read_csv(_V27_PRIMARY, dtype={"StudyInstanceUID": str})
    cols = [c for c in v25.columns if c != "StudyInstanceUID"]
    if list(v25.columns) != list(v24.columns):
        raise RuntimeError("V24/V25 schema mismatch")
    if v25.StudyInstanceUID.tolist() != v24.StudyInstanceUID.tolist():
        v24 = v24.set_index("StudyInstanceUID").reindex(v25.StudyInstanceUID).reset_index()
        if v24[cols].isna().any().any():
            raise RuntimeError("V24 missing studies vs V25")
    blended = v25.copy()
    blended[cols] = (1.0 - _V27_W) * _v27_rank(v25, cols) + _V27_W * _v27_rank(v24, cols)
    if not _v27_np.isfinite(blended[cols].to_numpy()).all():
        raise RuntimeError("non-finite blend")
    blended.to_csv(_V27_PRIMARY, index=False)
    _v27_rec.update(
        status="V27_BLENDED",
        rows=int(len(blended)),
        v25_rows=int(len(v25)),
        v24_rows=int(len(v24)),
    )
    print("[V27] blended 0.80*rank(V25)+0.20*rank(V24) rows", len(blended))
except Exception as _v27_e:
    _v27_rec["status"] = "V27_ERROR_V25_PRESERVED"
    _v27_rec["error"] = f"{type(_v27_e).__name__}: {_v27_e}"
    print("[V27] blend failed; restoring V25:", _v27_rec["error"])
    if _V27_V25.is_file():
        _v27_sh.copy2(_V27_V25, _V27_PRIMARY)
_V27_AUDIT.write_text(_v27_json.dumps(_v27_rec, indent=2, sort_keys=True) + "\n")
print(_v27_json.dumps(_v27_rec, indent=2, sort_keys=True))
'''


def main() -> None:
    v25 = json.loads((V25_DIR / "rsna-knee-v25-v16-dinov3.ipynb").read_text(encoding="utf-8"))
    v24 = json.loads((V24_DIR / "rsna-knee-v24-dinosaur-v3.ipynb").read_text(encoding="utf-8"))
    cells = list(v25["cells"])
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "source": [CHECKPOINT],
            "outputs": [],
            "execution_count": None,
        }
    )
    # One isolated try so a half-written dinosaur 0.5 file cannot be blended.
    dino_parts = ["".join(c.get("source", [])) for c in v24["cells"][1:]]
    dino_body = "\n\n".join(dino_parts)
    future_lines = [
        ln for ln in dino_body.splitlines() if ln.startswith("from __future__")
    ]
    dino_body = "\n".join(
        ln for ln in dino_body.splitlines() if not ln.startswith("from __future__")
    )
    header = "\n".join(future_lines + ["_V27_DINO_OK = False"]) + "\n"
    wrapped = (
        header
        + "try:\n"
        + "\n".join("    " + ln if ln.strip() else "" for ln in dino_body.splitlines())
        + "\n    _V27_DINO_OK = True\n"
        + "    print('[V27] dinosaur arm finished')\n"
        + "except Exception as _v27_dino_err:\n"
        + "    _V27_DINO_OK = False\n"
        + "    print('[V27] dinosaur arm failed; V25 checkpoint kept:', "
        + "type(_v27_dino_err).__name__, _v27_dino_err)\n"
        + "    from pathlib import Path as _V27Fix\n"
        + "    import shutil as _V27FixSh\n"
        + "    _p25 = _V27Fix('/kaggle/working/submission_v25_parent.csv')\n"
        + "    _p1 = _V27Fix('/kaggle/working/submission.csv')\n"
        + "    if _p25.is_file():\n"
        + "        _V27FixSh.copy2(_p25, _p1)\n"
    )
    compile(wrapped, "v27_dino", "exec")
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "source": [wrapped],
            "outputs": [],
            "execution_count": None,
        }
    )
    compile(BLEND, "v27_blend", "exec")
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "source": [BLEND],
            "outputs": [],
            "execution_count": None,
        }
    )
    nb = dict(v25)
    nb["cells"] = cells
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "rsna-knee-v27-v25-v24-blend.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    meta25 = json.loads((V25_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta24 = json.loads((V24_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
    ds = list(dict.fromkeys(meta25.get("dataset_sources", []) + meta24.get("dataset_sources", [])))
    ms = list(dict.fromkeys(meta25.get("model_sources", []) + meta24.get("model_sources", [])))
    ks = list(dict.fromkeys((meta25.get("kernel_sources") or []) + (meta24.get("kernel_sources") or [])))
    meta = {
        "id": "danielsolo1770/rsna-knee-v27-v25-v24-blend",
        "title": "rsna-knee-v27-v25-v24-blend",
        "code_file": "rsna-knee-v27-v25-v24-blend.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": ds,
        "kernel_sources": ks,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ms,
        "machine_shape": "NvidiaTeslaT4",
    }
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("v27 cells", len(cells), "datasets", len(ds), "models", ms, "kernels", ks)


if __name__ == "__main__":
    main()
