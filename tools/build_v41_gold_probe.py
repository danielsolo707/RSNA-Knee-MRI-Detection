"""Build V41: run the ENTIRE V33 (DINOsaur V4) inference chain on the 58 GOLD TRAIN studies.

Why
---
We have gold-58 prediction vectors for our own arms (v5, v4, v31 OOF, any future
fine-tune) but NOT for V33's branches (20-member DINOv2 ensemble, 3D-MIL, rad arm,
CoAtNet pair, and the final fusion). Without them every fusion change is a blind
submit. This probe produces, in one ~1 h GPU kernel:

    probe_stage1_dinov2.csv   after the DINOv2 ensemble cell
    probe_stage2_mil.csv      after the 3D-MIL cell
    probe_stage3_rad.csv      after the rad/calibration cell
    probe_stage4_final.csv    the final V33 fusion (= what scored 0.936)
    probe_coatnet.csv         the CoAtNet dual-checkpoint branch alone
    probe_gold_stages.npz     all of the above as arrays + gold truth
    probe_gold_report.json    per-stage gold-58 macro + per-target AUCs

`tools/blend_gold.py` / `tools/offline_fusion_analysis.py` can then simulate
V33 + any owned arm offline, split-half validated, before spending a submit.

How (surgical patches to the V33 notebook, each asserted to hit exactly once)
------------------------------------------------------------------------------
1. Preamble cell: build the gold-58 set from train.csv and intercept
   pandas.read_csv so every test.csv / test_series.csv / sample_submission.csv
   read in ANY cell returns the gold subset of the TRAIN data.
2. walk() gains a gold-study filter (train_series has 4,407 studies; we scan 58).
3. The two `walk('test_series')` calls -> `walk('train_series')`.
4. Cell-3 SERIES_ROOT forced to train_series.
5. Cell-5 tsdir forced to train_series.
6. The V18 hygiene unlink of submission_coatnet.csv is disabled so we keep it.
Snapshot cells are inserted after each of the 4 code cells; a final cell scores
every stage against the gold labels and writes the artifacts.

Train-only; never writes a competition submission; NOT submitted itself.

Run: python tools/build_v41_gold_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "kaggle" / "submission_v33_dino_raptor_0936"
DST = REPO / "kaggle" / "probe_v41_gold"

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]

PREAMBLE = '''# ---- V41 GOLD PROBE preamble: point the whole V33 chain at the 58 gold TRAIN studies ----
import os as _p_os
import numpy as _p_np
import pandas as _p_pd
from pathlib import Path as _p_Path

_COMP = _p_Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
_LAB12 = ''' + json.dumps(LAB) + '''
_train_full = _p_pd.read_csv(_COMP / 'train.csv')
_train_full['StudyInstanceUID'] = _train_full['StudyInstanceUID'].astype(str)
_gold = _train_full[_train_full[_LAB12].notna().all(axis=1)].copy()
GOLD_IDS = sorted(_gold['StudyInstanceUID'].tolist())
_GOLD_SET = set(GOLD_IDS)
print(f"[probe] gold studies in train.csv: {len(GOLD_IDS)}", flush=True)

_ts_full = _p_pd.read_csv(_COMP / 'train_series.csv')
_ts_full['StudyInstanceUID'] = _ts_full['StudyInstanceUID'].astype(str)
_gold_series = _ts_full[_ts_full['StudyInstanceUID'].isin(_GOLD_SET)].copy()
_gold_sub = _gold[['StudyInstanceUID'] + _LAB12].copy()

_orig_read_csv = _p_pd.read_csv
def _probe_read_csv(path, *a, **k):
    name = str(path).replace('\\\\', '/').rsplit('/', 1)[-1].lower()
    if name == 'test.csv':
        df = _gold.copy()
    elif name == 'test_series.csv':
        df = _gold_series.copy()
    elif name == 'sample_submission.csv':
        df = _gold_sub.copy()
    else:
        return _orig_read_csv(path, *a, **k)
    dt = k.get('dtype')
    if dt:
        for c, t in dt.items():
            if c in df.columns:
                df[c] = df[c].astype(t)
    return df
_p_pd.read_csv = _probe_read_csv
print("[probe] read_csv interceptor installed (test.csv/test_series.csv/sample_submission.csv -> gold train subset)", flush=True)
'''

SNAPSHOT = '''# ---- V41 probe snapshot: {label} ----
import shutil as _sn_sh
import os as _sn_os
for _src, _dst in {copies}:
    if _sn_os.path.isfile(_src):
        _sn_sh.copy(_src, '/kaggle/working/' + _dst)
        print('[probe] snapshot ' + _dst + ' (' + str(_sn_os.path.getsize(_src)) + ' bytes)', flush=True)
    else:
        print('[probe] MISSING ' + _src, flush=True)
'''

SCORING = '''# ---- V41 probe: score every stage on gold-58 and dump artifacts ----
import json as _sc_json
import numpy as _sc_np
import pandas as _sc_pd
from sklearn.metrics import roc_auc_score as _sc_auc

_LAB12 = ''' + json.dumps(LAB) + '''
_COMP = '/kaggle/input/competitions/rsna-knee-abnormality-detection'
_tr = _sc_pd.read_csv(_COMP + '/train.csv')
_tr['StudyInstanceUID'] = _tr['StudyInstanceUID'].astype(str)
_gold = _tr[_tr[_LAB12].notna().all(axis=1)].set_index('StudyInstanceUID')
ids = sorted(_gold.index)
Y = _gold.loc[ids, _LAB12].to_numpy(float)

STAGES = [
    ('stage1_dinov2', 'probe_stage1_dinov2.csv'),
    ('stage2_mil', 'probe_stage2_mil.csv'),
    ('stage3_rad', 'probe_stage3_rad.csv'),
    ('coatnet', 'probe_coatnet.csv'),
    ('stage4_final', 'probe_stage4_final.csv'),
]
preds, report = {}, {}
for key, fname in STAGES:
    try:
        df = _sc_pd.read_csv('/kaggle/working/' + fname)
    except FileNotFoundError:
        report[key] = {'error': 'missing ' + fname}
        print('[probe-score] MISSING', fname, flush=True)
        continue
    df['StudyInstanceUID'] = df['StudyInstanceUID'].astype(str)
    df = df.set_index('StudyInstanceUID')
    cols = [c for c in _LAB12 if c in df.columns]
    P = df.loc[ids, cols].to_numpy(float)
    if not _sc_np.isfinite(P).all():
        P = _sc_np.where(_sc_np.isfinite(P), P, 0.5)
    aucs = {}
    for j, t in enumerate(cols):
        if len(set(Y[:, _LAB12.index(t)].astype(int))) > 1:
            aucs[t] = float(_sc_auc(Y[:, _LAB12.index(t)], P[:, j]))
    macro = float(_sc_np.mean(list(aucs.values()))) if aucs else float('nan')
    preds[key] = P
    report[key] = {'macro': round(macro, 4), 'per_target': {k: round(v, 4) for k, v in aucs.items()}}
    print(f'[probe-score] {key:14s} macro={macro:.4f}', flush=True)

if preds:
    _sc_np.savez('/kaggle/working/probe_gold_stages.npz',
                 **{('pred_' + k): v for k, v in preds.items()},
                 truth=Y, ids=_sc_np.array(ids), lab=_sc_np.array(_LAB12))
_sc_json.dump({'n_gold': len(ids), 'stages': report},
              open('/kaggle/working/probe_gold_report.json', 'w'), indent=1)
print('[probe-score] saved probe_gold_stages.npz / probe_gold_report.json', flush=True)
'''

PATCHES = {
    # walk(): skip non-gold studies when scanning train_series (4,407 -> 58)
    "walk_filter": (
        "    for study in os.scandir(base):\n"
        "        if study.is_dir():\n"
        "            for series in os.scandir(study.path):\n",
        "    for study in os.scandir(base):\n"
        "        if study.is_dir():\n"
        "            if globals().get('_GOLD_SET') and study.name not in _GOLD_SET:\n"
        "                continue\n"
        "            for series in os.scandir(study.path):\n",
    ),
    "walk_call_cell2": ("annotate(walk('test_series'))", "annotate(walk('train_series'))"),
    "series_root_cell3": ("SERIES_ROOT = COMP / 'test_series'", "SERIES_ROOT = COMP / 'train_series'"),
    "tsdir_cell5": ('tsdir = ROOT + "/test_series"', 'tsdir = ROOT + "/train_series"'),
    # keep the per-arm CoAtNet CSV (V18 hygiene would delete it)
    "hygiene_disable": (
        "    for _v18_temp in (\n"
        "        _blend_work / 'submission_coatnet.csv',\n"
        "        _blend_work / 'submission_transformer_0920.csv',\n"
        "    ):\n",
        "    for _v18_temp in ():  # probe: hygiene disabled, keep per-arm CSVs\n",
    ),
}

STAGE_SNAPSHOTS = [
    ("after DINOv2 ensemble cell",
     [("/kaggle/working/submission.csv", "probe_stage1_dinov2.csv")]),
    ("after 3D-MIL cell",
     [("/kaggle/working/submission.csv", "probe_stage2_mil.csv")]),
    ("after rad/calibration cell",
     [("/kaggle/working/submission.csv", "probe_stage3_rad.csv")]),
    ("after CoAtNet fusion cell",
     [("/kaggle/working/submission.csv", "probe_stage4_final.csv"),
      ("/kaggle/working/submission_coatnet.csv", "probe_coatnet.csv")]),
]


def main() -> None:
    nb = json.loads((SRC / "rsna-knee-v33-dino-raptor-0936.ipynb").read_text(encoding="utf-8"))
    cells = nb["cells"]
    code_idx = [i for i, c in enumerate(cells) if c["cell_type"] == "code"]
    assert len(code_idx) == 4, f"expected 4 code cells, got {len(code_idx)}"

    sources = ["".join(cells[i]["source"]) for i in code_idx]

    # walk() definition lives in the first code cell; the second walk call in cell 4.
    EXPECTED_HITS = {"walk_filter": 1, "walk_call_cell2": 2, "series_root_cell3": 1,
                     "tsdir_cell5": 1, "hygiene_disable": 1}
    for name, (old, new) in PATCHES.items():
        hits = sum(s.count(old) for s in sources)
        assert hits == EXPECTED_HITS[name], \
            f"patch {name}: expected exactly {EXPECTED_HITS[name]} hit(s), got {hits}"
    sources[0] = sources[0].replace(*PATCHES["walk_filter"]).replace(*PATCHES["walk_call_cell2"])
    sources[1] = sources[1].replace(*PATCHES["series_root_cell3"])
    sources[2] = sources[2].replace(*PATCHES["walk_call_cell2"])
    sources[3] = (sources[3].replace(*PATCHES["tsdir_cell5"])
                            .replace(*PATCHES["hygiene_disable"]))

    # gate: every patched cell must compile on its own (a cell may legally start
    # with `from __future__ import` - never join cells into one string to compile)
    for j, s in enumerate(sources):
        compile(s, f"v41_cell{j}", "exec")
    compile(PREAMBLE, "v41_preamble", "exec")
    compile(SCORING, "v41_scoring", "exec")

    def code_cell(src: str) -> dict:
        return {"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True),
                "outputs": [], "execution_count": None}

    out = []
    for c in cells[:code_idx[0]]:
        out.append(c)
    out.append(code_cell(PREAMBLE))
    for j, ci in enumerate(code_idx):
        out.append(code_cell(sources[j]))
        label, copies = STAGE_SNAPSHOTS[j]
        snap = SNAPSHOT.format(label=label, copies=repr(copies))
        compile(snap, f"v41_snap{j}", "exec")
        out.append(code_cell(snap))
    out.append(code_cell(SCORING))

    nb["cells"] = out
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "rsna-knee-v41-gold-probe.ipynb").write_text(json.dumps(nb), encoding="utf-8")

    meta = json.loads((SRC / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["id"] = "danielsolo1770/rsna-knee-v41-gold-probe"
    meta["title"] = "rsna-knee-v41-gold-probe"
    meta["code_file"] = "rsna-knee-v41-gold-probe.ipynb"
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print("V41 gold probe ready:", DST)
    print("  runs the FULL V33 chain (DINOv2 + MIL + rad + CoAtNet fusion) on gold-58 TRAIN studies")
    print("  outputs: probe_stage{1..4}*.csv, probe_coatnet.csv, probe_gold_stages.npz, probe_gold_report.json")
    print("  datasets/models/kernel sources carried over from V33:", meta["dataset_sources"])


if __name__ == "__main__":
    main()
