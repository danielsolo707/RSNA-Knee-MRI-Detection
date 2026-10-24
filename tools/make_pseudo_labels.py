"""Generate pseudo-labels for the 4,349 weak-label studies, locally on the 4070.

Why
---
Measured out-of-sample on gold-58 (the public checkpoints held gold-58 out of training,
so this is a clean measurement):

    current LLM soft labels                     0.8927
    dual-CoAtNet fusion (0.40*v5 + 0.60*v4)     0.9172   (+0.0244)
    blend 40% current + 60% pseudo              0.9291   (+0.0364)

+0.036 is ~9x the blend-weight effect that FAILED to transfer in V37, i.e. large enough
to survive n=58 noise. Training on better labels is the main route to 0.940.

Run it locally because the job is ~12 h on a 4070 (free) versus ~35 h of Kaggle quota for
the same work.

Caveat: on the 4,349 TRAINING studies the public checkpoints' predictions are in-sample,
so pseudo-labels there are optimistic. That is why we blend back with the original soft
labels at a single global ratio, and never fit per-target ratios (V37 lesson).

Usage
    python tools/make_pseudo_labels.py                       # both arms, k_eval=12
    python tools/make_pseudo_labels.py --k-eval 24 --arms v5 # slower, one arm
    python tools/make_pseudo_labels.py --limit 200 --smoke   # quick wiring check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    # Running as a notebook cell: __file__ is undefined. Paths are supplied via
    # --corpus-dir/--ckpt-dir/--labels/--out, so the repo root is only used to locate
    # train_raptor.py, which in a kernel sits next to the notebook.
    REPO = Path.cwd()
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(Path.cwd()))
import train_raptor as tr  # noqa: E402

# Defaults are the local layout; all overridable so the same script runs as a Kaggle
# kernel, where the corpus and checkpoints are already attached (and are NOT corrupted,
# unlike the local copy that the downloader bug damaged).
CORPUS = REPO / "artifacts" / "raptor_corpus"
CKPT = REPO / "artifacts" / "public_arms" / "ckpt"
LABELS = REPO / "artifacts" / "raptor_labels" / "labels_llm_soft.parquet"
OUT = REPO / "artifacts" / "pseudo_labels"

ARMS = {
    "v5": ("raptor_ft_coatnet_v5_full_swa.pt", 0.40),
    "v4": ("raptor_ft_coatnet_v4_full.pt", 0.60),
}


def find_ckpt(fname: str) -> Path:
    hits = sorted(CKPT.glob(f"**/{fname}"))
    if not hits:
        raise SystemExit(f"{fname} not found under {CKPT}. Run the checkpoint download first.")
    return hits[0]


def load_model(path: Path, dev: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    res = int(ck.get("res", 384))
    bb = tr.build_backbone(arch, pretrained=False)
    model = tr.RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(dev)
    return model, res, arch


@torch.no_grad()
def predict(model, ids, id2row, labels, res, k_eval, dev, amp_dtype, bs=2, workers=2,
            tag="", every=250, partial_path=None):
    ds = tr.StudyWindows(str(CORPUS), ids, id2row, labels, res, k_eval, train=False,
                         norm="imagenet", roi=False, roi_mode="compartment",
                         roi_pad=0.06, roi_overlap=0.12, roi_sbox=None, roi_cen=None)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                    collate_fn=tr.collate)
    P, t0 = [], time.time()
    for i, (x, _y) in enumerate(dl):
        with torch.autocast(dev, dtype=amp_dtype, enabled=(dev == "cuda")):
            o = torch.sigmoid(model(x.to(dev)).float())
        P.append(o.cpu().numpy())
        if (i + 1) % every == 0:
            done = (i + 1) * bs
            el = time.time() - t0
            eta = el / (i + 1) * (len(dl) - i - 1)
            print(f"  [{tag}] {done}/{len(ids)}  {done/el:.2f} studies/s  ETA {eta/3600:.2f}h",
                  flush=True)
            # Checkpoint partial predictions. On Kaggle a 9 h session cap can kill the
            # run; without this the whole ~9 h would be lost. Kernel output is still
            # downloadable after a kill (verified on V34).
            if partial_path is not None:
                np.save(partial_path, np.concatenate(P))
    return np.concatenate(P)


def rank(x: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(
        lambda v: pd.Series(v).rank(method="average", pct=True).to_numpy(), 0, x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-eval", type=int, default=12)
    ap.add_argument("--arms", default="v5,v4")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 is required on Windows: workers use spawn, which would lose "
                         "the patched _open_corpus")
    ap.add_argument("--blend-w", type=float, default=0.60,
                    help="weight on pseudo-labels in the final blend (global, not per-target)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gold-only", action="store_true",
                    help="predict ONLY the gold-58 studies per arm (saves "
                         "pred_gold_<arm>.npy + per-arm/fusion AUCs). Skips the "
                         "4349-study train pass and the parquet write. Used to "
                         "score any arm combination before spending hours on the "
                         "full train predictions.")
    ap.add_argument("--corpus-dir", default=str(CORPUS))
    ap.add_argument("--corpus-extra-dir", default="",
                    help="directory holding extra_vols.npy / extra_masks.npy / extra_ids.npy. "
                         "Defaults to --corpus-dir. On Kaggle the two corpus datasets mount "
                         "into SEPARATE directories, so this must be set explicitly.")
    ap.add_argument("--ckpt-dir", default=str(CKPT))
    ap.add_argument("--labels", default=str(LABELS))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    # Assign through globals(): plain assignment here would rebind these as locals and
    # break the `default=str(CORPUS)` reads above (UnboundLocalError).
    g = globals()
    g["CORPUS"] = Path(args.corpus_dir)
    g["EXTRA"] = Path(args.corpus_extra_dir) if args.corpus_extra_dir else Path(args.corpus_dir)
    g["CKPT"] = Path(args.ckpt_dir)
    g["LABELS"] = Path(args.labels)
    g["OUT"] = Path(args.out)
    if args.smoke:
        args.limit = args.limit or 120
        args.k_eval = 6

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (dev == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"device={dev} amp={amp_dtype} k_eval={args.k_eval} arms={args.arms}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- corpus (explicit two-part assembly; train_raptor._open_corpus would pick up
    #      part 1 only because a local all_vols.npy exists) ----
    # Part 1 lives in CORPUS, part 2 in EXTRA. These are the SAME directory locally but
    # DIFFERENT mounted directories on Kaggle (two datasets), so resolve them separately.
    need = [(CORPUS, f) for f in ("all_vols.npy", "all_masks.npy", "all_ids.npy")]
    need += [(EXTRA, f) for f in ("extra_vols.npy", "extra_masks.npy", "extra_ids.npy")]
    for d, f in need:
        if not (d / f).exists():
            raise SystemExit(f"missing {d/f} - corpus not complete yet")
    vols = tr._TwoPartVols(np.load(CORPUS / "all_vols.npy", mmap_mode="r"),
                           np.load(EXTRA / "extra_vols.npy", mmap_mode="r"))
    masks = np.concatenate([np.load(CORPUS / "all_masks.npy"),
                            np.load(EXTRA / "extra_masks.npy")], axis=0)
    ids = np.concatenate([np.load(CORPUS / "all_ids.npy", allow_pickle=True).astype(str),
                          np.load(EXTRA / "extra_ids.npy", allow_pickle=True).astype(str)])

    # StudyWindows._ensure() calls the MODULE-LEVEL _open_corpus(), so patch that.
    # Two traps avoided here:
    #   * the stock _open_corpus() returns part 1 only when a local all_vols.npy exists
    #     (it checks that path first and never looks for extra_vols.npy) - we must
    #     assemble both parts explicitly;
    #   * Windows DataLoader workers use spawn, so a monkeypatch would NOT survive into
    #     them - hence workers=0 by default.
    tr._open_corpus = lambda: (vols, masks)
    print(f"corpus: {len(ids)} studies, masks {masks.shape}", flush=True)
    id2row = {u: i for i, u in enumerate(ids)}

    # ---- labels + gold ----
    soft = pd.read_parquet(LABELS)
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
    soft = soft.set_index("StudyInstanceUID")
    # Locally this is <repo>/data/raw/train.csv, but inside a Kaggle kernel REPO is
    # /kaggle/working and the competition data mounts elsewhere - resolve by fallback.
    _cand = [REPO / "data" / "raw" / "train.csv",
             Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv"),
             Path("/kaggle/input/rsna-knee-abnormality-detection/train.csv")]
    trn_path = next((q for q in _cand if q.exists()), None)
    if trn_path is None:
        raise SystemExit("train.csv not found - attach the competition data.")
    print(f"train.csv: {trn_path}", flush=True)
    trn = pd.read_csv(trn_path)
    trn["StudyInstanceUID"] = trn["StudyInstanceUID"].astype(str)
    gold_df = trn[trn[tr.LAB].notna().all(axis=1)].set_index("StudyInstanceUID")
    goldset = set(gold_df.index)
    train_ids = [u for u in ids if u in soft.index and u not in goldset]
    gold_ids = [u for u in ids if u in goldset]
    if args.limit:
        train_ids = train_ids[:args.limit]
    print(f"train_ids={len(train_ids)}  gold={len(gold_ids)}", flush=True)

    labels = {u: soft.loc[u, tr.LAB].values.astype(np.float32) for u in train_ids}
    for u in gold_ids:
        labels[u] = gold_df.loc[u, tr.LAB].values.astype(np.float32)

    if args.gold_only:
        # Gold-58 only: cheap (~15 min/arm at k_eval=12) and answers "is this arm
        # combination worth the full 6 h train pass" before committing to it.
        from sklearn.metrics import roc_auc_score
        Y = np.stack([gold_df.loc[u, tr.LAB].values.astype(float) for u in gold_ids])

        def lab_auc(P):
            a = [roc_auc_score(Y[:, j], P[:, j]) for j in range(len(tr.LAB))
                 if len(set(Y[:, j].astype(int))) > 1]
            return float(np.mean(a))

        gp, rep = {}, {}
        for key in args.arms.split(","):
            key = key.strip()
            if key not in ARMS:
                raise SystemExit(f"unknown arm {key}; choose from {list(ARMS)}")
            fname, w = ARMS[key]
            model, res, arch = load_model(find_ckpt(fname), dev)
            print(f"[{key}-gold] {fname} res={res} arch={arch}", flush=True)
            t = time.time()
            gp[key] = predict(model, gold_ids, id2row, labels, res, args.k_eval,
                              dev, amp_dtype, bs=args.bs, workers=args.workers,
                              tag=f"{key}-gold")
            np.save(OUT / f"pred_gold_{key}.npy", gp[key])
            rep[key] = round(lab_auc(gp[key]), 4)
            print(f"[{key}-gold] AUC={rep[key]:.4f}  ({(time.time()-t)/60:.1f} min)",
                  flush=True)
            del model
            torch.cuda.empty_cache()
        if len(gp) > 1:
            tot = sum(ARMS[k][1] for k in gp)
            gf = rank(sum(ARMS[k][1] / tot * rank(gp[k]) for k in gp))
            rep["fusion"] = round(lab_auc(gf), 4)
            print(f"[fusion] gold-58 AUC={rep['fusion']:.4f}", flush=True)
        (OUT / "gold_only_report.json").write_text(json.dumps(rep, indent=1),
                                                   encoding="utf-8")
        print(f"wrote {OUT/'gold_only_report.json'}", flush=True)
        return 0

    # ---- run arms ----
    preds, meta = {}, {}
    for key in args.arms.split(","):
        key = key.strip()
        if key not in ARMS:
            raise SystemExit(f"unknown arm {key}; choose from {list(ARMS)}")
        fname, w = ARMS[key]
        path = find_ckpt(fname)
        model, res, arch = load_model(path, dev)
        print(f"[{key}] {path.name} res={res} arch={arch}", flush=True)
        t = time.time()
        # Crash resume for the ~6 h train pass: predict() checkpoints its
        # accumulated output every `every` batches; on restart we skip the studies
        # already covered. Valid because shuffle=False and workers=0 keep the
        # DataLoader order deterministic, so partial rows map to train_ids[:start].
        partial = OUT / f"partial_train_{key}.npy"
        seg = OUT / f"partial_train_{key}_seg.npy"
        tail = None
        if partial.exists():
            try:
                cand = np.load(partial)
                if 0 < len(cand) < len(train_ids) and cand.shape[1:] == (len(tr.LAB),):
                    tail = cand
            except Exception:
                pass
        if tail is not None and seg.exists():
            # fold a leftover segment from an interrupted resume into the tail
            try:
                s = np.load(seg)
                if 0 < len(s) and len(tail) + len(s) <= len(train_ids):
                    tail = np.concatenate([tail, s])
            except Exception:
                pass
        start = len(tail) if tail is not None else 0
        if start:
            print(f"[{key}] resuming after {start} studies from partial files", flush=True)
        ids_run = train_ids[start:]
        if len(ids_run):
            P_new = predict(model, ids_run, id2row, labels, res, args.k_eval,
                            dev, amp_dtype, bs=args.bs, workers=args.workers, tag=key,
                            partial_path=seg, every=250)
        else:
            P_new = np.zeros((0, len(tr.LAB)), dtype=np.float32)
        preds[key] = np.concatenate([tail, P_new]) if (tail is not None and len(P_new)) \
            else (P_new if tail is None else tail)
        np.save(partial, preds[key])
        seg.unlink(missing_ok=True)
        meta[key] = {"file": path.name, "weight": w, "arch": arch, "res": res,
                     "seconds": round(time.time() - t, 1)}
        print(f"[{key}] done in {(time.time()-t)/3600:.2f}h", flush=True)
        np.save(OUT / f"pred_train_{key}.npy", preds[key])
        del model
        torch.cuda.empty_cache()

    # ---- fuse (global weights, never per-target) ----
    tot = sum(ARMS[k][1] for k in preds)
    fused = sum(ARMS[k][1] / tot * rank(preds[k]) for k in preds)
    fused = rank(fused)
    np.save(OUT / "pred_train_fused.npy", fused)

    # ---- blend with original soft labels ----
    cur = np.stack([soft.loc[u, tr.LAB].values.astype(float) for u in train_ids])
    pseudo = (1 - args.blend_w) * rank(cur) + args.blend_w * fused
    df = pd.DataFrame(pseudo, columns=tr.LAB)
    df.insert(0, "StudyInstanceUID", train_ids)
    out_parquet = OUT / f"pseudo_labels_w{int(args.blend_w*100)}.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"\nwrote {out_parquet}  ({len(df)} rows)", flush=True)

    # ---- out-of-sample check on gold-58 ----
    report = {"arms": meta, "k_eval": args.k_eval, "blend_w": args.blend_w,
              "n_train": len(train_ids)}
    try:
        from sklearn.metrics import roc_auc_score
        gp, meta_g = {}, {}
        for key in preds:
            fname, w = ARMS[key]
            model, res, arch = load_model(find_ckpt(fname), dev)
            gp[key] = predict(model, gold_ids, id2row, labels, res, args.k_eval,
                              dev, amp_dtype, bs=args.bs, workers=args.workers, tag=f"{key}-gold")
            np.save(OUT / f"pred_gold_{key}.npy", gp[key])
            del model
            torch.cuda.empty_cache()
        Y = np.stack([gold_df.loc[u, tr.LAB].values.astype(float) for u in gold_ids])

        def lab_auc(P):
            a = [roc_auc_score(Y[:, j], P[:, j]) for j in range(len(tr.LAB))
                 if len(set(Y[:, j].astype(int))) > 1]
            return float(np.mean(a))

        tot = sum(ARMS[k][1] for k in gp)
        gfused = rank(sum(ARMS[k][1] / tot * rank(gp[k]) for k in gp))
        gcur = rank(np.stack([soft.loc[u, tr.LAB].values.astype(float)
                              for u in gold_ids if u in soft.index])) \
            if all(u in soft.index for u in gold_ids) else None
        report["gold58_label_auc"] = {"fusion": round(lab_auc(gfused), 4)}
        print(f"gold-58 out-of-sample: fusion pseudo-labels AUC={lab_auc(gfused):.4f}", flush=True)
    except Exception as e:  # noqa: BLE001
        report["gold58_error"] = str(e)
        print("gold-58 check skipped:", e, flush=True)

    (OUT / "pseudo_labels_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {OUT/'pseudo_labels_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
