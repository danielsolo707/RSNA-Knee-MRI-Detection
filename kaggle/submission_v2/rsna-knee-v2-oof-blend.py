# %% [notebook cell 2]
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import gc
import hashlib
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

# The label extractor is defined in the cells above when this runs as a notebook. As a
# plain script it is imported from the package source, so the two paths share one
# definition rather than keeping a copy each.

T0 = time.time()
SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]


# The centre crop has to be smaller than the smallest field of view in the corpus or it
# silently does nothing. Measured over every training series, the acquired field of view
# (Rows x PixelSpacing) has median 160 mm and runs from 70 to 320: a 160 mm crop is
# larger than the image in 60% of series and is skipped for all of them, which leaves
# their physical scale unnormalised. 130 mm is below the field of view of 99.6% of
# series and still contains the joint.
CROP_MM = 130.0

# Cache resolution. Everything downstream may downsample from this, so it is set by the
# most demanding configuration rather than by the default one.
CACHE_IMG = 336
GROUP = 3                  # slices per encoder input, stacked as the three channels
N_GROUP_MAX = 1
CACHE_FRACTION = 0.45      # share of free memory the pixel cache may take
CACHE_BUDGET_MAX_GB = 24.0 # hard ceiling regardless of what the machine reports
CACHE_BUDGET_GB = 12.0     # only the fallback, for a machine with no /proc/meminfo
TEST_SHARE = 0.30          # floor on the test corpus relative to the training one, since
                           # the visible test split is a stub and the scored one is not
HDR_THREADS = 16
PIX_THREADS = 12
ORDER_THREADS = 32         # slice-ordering is latency-bound on the mount, not CPU-bound
# Ceiling for the ordering pass. It has to be a ceiling because the pass is hundreds of
# thousands of small reads over a network mount, so its duration is a property of the
# mount rather than of the work, and varies between runs that do the same reading. It must not be a tight one: giving up leaves
# those series in file order, which is uncorrelated with anatomy, and that degradation is
# silent. So the ceiling sits well above what the pass ordinarily needs: its purpose is
# to stop the pass consuming the whole run on a slow mount, not to trim the ordinary
# case, and a ceiling tight enough to bind on a normal day would trade a silent
# degradation for a saving the run does not need.
ORDER_BUDGET_S = 5400

# Resolution is the axis under test. A feature of width d mm survives resampling only if
# the pixel pitch is at most d/2, and the pitch here is set by the crop above rather than
# by the acquired field of view: CROP_MM / P. At 224 px that is 0.58 mm, above the 0.5 mm
# a 1 mm tear needs; at 336 px it is 0.39 mm and clears it. Both configurations read the
# same cache, so the comparison isolates the resize.
RUNS = [
    {"name": "r224", "img": 224},
    {"name": "r336", "img": 336},
]

EPOCHS = 10
BATCH_STUDIES = 8          # a study is a bag of up to N_SLOT slot images
AUG_ROT_DEG = 8.0          # rigid jitter; see augment() for why neither flip is used
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.10
LAT_MIN_OFFSET_MM = 20.0   # inside this the side is not readable from geometry; see
                           # side_from_geometry()
SLICE_BAND = (0.20, 0.80)  # fraction of the ordered stack read_slot samples across

# --- What a slice IS, as opposed to how many of them there are --------------- #
#
# A member is a function of the pixels it was fitted on, and img/crop_mm/slices/band do
# not determine those pixels by themselves. Four further decisions do, none of them
# visible in any shape:
#
#   order          which slice is the next one along the stack
#   lat            which knees are mirrored, and on what evidence
#   slot_fallback  whether a T1 slot may be filled from a series that is not T1
#   decode_fill    what stands in for a slice that would not decode
#
# `native` is the reading derived in the sections below. `legacy` is the reading an
# imported member was fitted under. A member read under the wrong one loads with every
# shape matching, runs, and writes a plausible submission computed from the wrong image -
# so the choice travels with the member and is part of the key that decides which members
# can share a decode. The legacy rules are reproduced rather than corrected: correcting
# them would hand that member pixels its weights never saw.
RULES_NATIVE = {"order": "normal", "lat": "centre",
                "slot_fallback": False, "decode_fill": "nearest"}
RULES_LEGACY = {"order": "dominant_axis", "lat": "corner_x",
                "slot_fallback": True, "decode_fill": "zero"}
RULES = dict(RULES_NATIVE)
LEGACY_LAT_OFFSET_MM = 5.0   # the dead zone the legacy laterality rule was fitted with

LR_HEAD = 1e-3
LR_BACKBONE = 8e-6         # the encoder is adapted, not retrained
UNFREEZE_LAST = 6          # trainable transformer blocks, from the output end
WEIGHT_DECAY = 0.02
EVAL_BATCH = 8
TIME_BUDGET = 8.0 * 3600

# Submission V2 is a single, pre-registered OOF ablation.  Jitter was not part of the
# exported OOF predictions used to choose the blend, so it stays off at inference.
ENABLE_JITTER_TTA = False

# Six slots: three planes crossed with the acquisition axes. The fat-suppressed
# fluid-sensitive series exist for nearly every study; the T1 and the non-suppressed
# fluid-sensitive series are scarcer, which is what the presence mask is for.
SLOTS_RECOVERED = [
    ("SAG_FLUID_FS", "Sagittal", True, True),
    ("COR_FLUID_FS", "Coronal", True, True),
    ("AX_FLUID_FS", "Axial", True, True),
    ("SAG_FLUID_NOFS", "Sagittal", True, False),
    ("COR_T1", "Coronal", False, False),
    ("SAG_T1", "Sagittal", False, False),
]

# The alternative: plane x the single axis the delivered flags carry, ignoring the
# recovered weighting. Kept as
# a switch so the choice of slot definition can be varied while everything else is held
# fixed. Under this scheme a `Struct` slot mixes T1 series with non-fat-suppressed PD/T2
# series, which carry very different tissue contrast.
SLOTS_PUBLIC = [
    ("SAG_FLUID", "Sagittal", None, True),
    ("COR_FLUID", "Coronal", None, True),
    ("AX_FLUID", "Axial", None, True),
    ("SAG_STRUCT", "Sagittal", None, False),
    ("COR_STRUCT", "Coronal", None, False),
    ("AX_STRUCT", "Axial", None, False),
]

SLOT_SCHEME = os.environ.get("SLOT_SCHEME", "recovered")
SLOTS = SLOTS_PUBLIC if SLOT_SCHEME == "public" else SLOTS_RECOVERED
N_SLOT = len(SLOTS)

# How many 384-wide parts the per-slot feature is built from. The encoder emits one
# vector per token; a slot feature is a fixed summary of that grid, and the summary an
# imported member was fitted with carries a third part.
POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}

# Which slots an imported member's attention is tilted toward, per diagnosis. Indices are
# into SLOTS. This is a fixed table rather than a learned parameter, so it is part of that
# member's definition and has to be reproduced exactly for its weights to mean anything.
SLOT_PRIOR_TABLE = {
    "ACL": (0, 3, 5), "MCL": (1, 4),
    "Medial Meniscus": (0, 1, 3, 4), "Lateral Meniscus": (0, 1, 3, 4),
    "Medial OA": (1, 4, 5), "Lateral OA": (1, 4, 5),
    "PF OA": (0, 2, 5), "Effusion": (0, 2), "Synovitis": (0, 2),
    "Baker's": (0,), "Contusion": (0, 1, 2), "Fracture": (0, 1, 2, 4, 5),
}
SLOT_PRIOR_STRENGTH = 0.55

FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}
_SEP = re.compile(r"[_\-.]")
_FATSAT_RX = re.compile(r"\bfs\b|fatsat|fat sat|\bstir\b|\bspair\b|\bspir\b|\bwe\b|"
                        r"water excit|\btirm\b|\bsting\b|\bfatsup\b")
_T1_RX = re.compile(r"\bt1\b|\bt1w\b")
_T2_RX = re.compile(r"\bt2\b|\bt2w\b")
_PD_RX = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")


import threading

# ---- dual-GPU roster (the executive change vs the reference notebook) ------------------
# The reference ran members one after another on a single device and, when its time
# budget bound, surrendered TTA windows and then whole members (public 0.847 vs the
# package's 0.891). Members are independent functions of the same cache, so they
# parallelise across devices with no change to any member's pixels or weights.
if torch.cuda.is_available():
    DEVS = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
else:
    DEVS = [torch.device("cpu")]
print(f"devices: {[str(d) for d in DEVS]}")


# %% [notebook cell 4]
def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def find_root():
    for c in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/rsna-knee-abnormality-detection"),
              Path("data"), Path(".")]:
        if (c / "test.csv").is_file() and (c / "test_series").is_dir():
            return c
    # last resort: two-level scan, because the mount is nested one deeper than usual
    base = Path("/kaggle/input")
    if base.is_dir():
        for depth1 in sorted(p for p in base.iterdir() if p.is_dir()):
            for cand in [depth1] + sorted(p for p in depth1.iterdir() if p.is_dir()):
                if (cand / "test.csv").is_file():
                    return cand
    raise FileNotFoundError(
        f"competition mount not found (cwd {Path.cwd()}); expected a directory holding "
        f"test.csv and test_series/")


def find_dinov2(variant="small"):
    """Locate a mounted DINOv2 checkpoint directory by variant name."""
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    hits = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "config.json" in files and "dinov2" in root.lower():
            hits.append(Path(root))
    for h in hits:
        if variant in str(h).lower():
            return h
    return hits[0] if hits else None


ROOT = find_root()
log(f"input root: {ROOT}")


IMG = CACHE_IMG            # kept as the name the pixel reader and cache use


def available_gb():
    """Memory this machine will actually lend, read rather than assumed.

    A hardcoded ceiling is a guess about a machine the author is not sitting at, and a
    guess that is too low costs coverage silently while a guess that is too high ends the
    run. The machine will say, so it is asked.
    """
    try:
        with open("/proc/meminfo") as fh:
            info = {k.strip(): v for k, v in
                    (l.split(":", 1) for l in fh if ":" in l)}
        return int(info["MemAvailable"].split()[0]) / 1024 ** 2
    except Exception:
        return CACHE_BUDGET_GB / CACHE_FRACTION      # fall back to the old constant


def plan_cache(n_study, n_test=0):
    """Choose how many slices per slot the memory the machine has will allow.

    The cache is n_study x n_slot x slices x IMG^2 bytes. Coverage is the cheap axis -
    linear - and resolution the expensive one, so when the budget binds it is the slice
    count that gives way rather than the pixel grid. Deciding once, from the training
    corpus size, keeps train and test caches on the same group layout.

    Only a fraction of what is free is taken. The rest is not slack: the encoder, its
    activations, the pinned batches and the frames all come out of the same pool, and the
    cache is the one allocation big enough that overshooting it kills the run outright.
    """
    avail = available_gb()
    budget = min(avail * CACHE_FRACTION, CACHE_BUDGET_MAX_GB)
    # Both caches are held at once, and the test half is what the visible run cannot
    # show: here it is a handful of studies, and at scoring it is the whole hidden set.
    # Sizing against the training corpus alone therefore passes every run that can be
    # watched and overruns the one that counts.
    n_total = n_study + max(n_test, int(TEST_SHARE * n_study))
    per_slice = n_total * N_SLOT * IMG * IMG
    afford = int(budget * 1024 ** 3 // max(per_slice, 1))
    groups = max(1, min(N_GROUP_MAX, afford // GROUP))
    log(f"memory: {avail:.1f} GB available, {budget:.1f} GB to the cache; "
        f"sizing for {n_study} train + {n_total - n_study} test studies "
        f"-> {groups} group(s) of {GROUP} = {groups * GROUP} slices per slot"
        + (f" (wanted {N_GROUP_MAX})" if groups < N_GROUP_MAX else ""))
    return groups


N_GROUP = plan_cache(len(pd.read_csv(ROOT / "train.csv")),
                     len(pd.read_csv(ROOT / "test.csv")))
CACHE_SLICES = GROUP * N_GROUP
log(f"cache layout: {N_GROUP} groups x {GROUP} slices = {CACHE_SLICES} per slot")


# %% [notebook cell 6]
HDR_TAGS = ["SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
            "RepetitionTime", "EchoTime", "Laterality", "PixelSpacing", "Rows",
            "Columns", "RescaleSlope", "RescaleIntercept",
            # Position and orientation are read from the same header probe() already
            # opens, so they cost nothing, and they are what recovers the side when the
            # Laterality tag is absent - which it is for half the studies here.
            "ImagePositionPatient", "ImageOrientationPatient"]


def _hdr_vec(s, n):
    """Parse a DICOM multi-value string as stored by probe(): floats joined by `|`."""
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split("|")]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None


def side_from_geometry(h):
    """Study -> 'L' / 'R' / None, from where the image sits in the patient.

    `Laterality` (0020,0060) is Type 2C and may legitimately be absent; in this corpus it
    is missing on exactly half the studies, and the vendors it is missing from are whole
    vendors rather than scattered series. A study with no tag is not a left knee, but the
    normalisation upstream treats it as one, so half the corpus was never normalised and
    the five side-defined targets - the two menisci, the two tibiofemoral compartments
    and the medial collateral ligament - saw that axis reversed on a large minority of it.

    The patient coordinate system fixes this without the tag: +x is the patient's left, so
    the centre of a right knee sits at negative x. The centre is used rather than
    `ImagePositionPatient` itself because that is the corner of the image, which is offset
    by half a field of view - enough to change the sign on a knee near the midline.

    The median over a study's series is what is thresholded, not a single series: probe()
    reads one arbitrary slice per series, which on a sagittal stack can sit anywhere
    across the joint. Studies whose centre falls near the midline are left unresolved
    rather than guessed - measured against the tagged half, the rule is right 97% of the
    time overall and no better than chance inside 20 mm.
    """
    cx = {}
    for r in h.itertuples(index=False):
        ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
        iop = _hdr_vec(getattr(r, "ImageOrientationPatient", None), 6)
        ps = _hdr_vec(getattr(r, "PixelSpacing", None), 2)
        rows, cols = getattr(r, "Rows", None), getattr(r, "Columns", None)
        if ipp is None or iop is None or ps is None or not rows or not cols:
            continue
        try:
            c = ipp[:3] + iop[:3] * ps[1] * float(cols) / 2 + iop[3:6] * ps[0] * float(rows) / 2
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    out = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        out[st] = None if abs(m) < LAT_MIN_OFFSET_MM else ("R" if m < 0 else "L")
    return out


def side_from_corner_x(h):
    """The laterality an imported member was fitted under.

    It thresholds the median raw `ImagePositionPatient` x over a study's series. That is
    the x of the image *corner*, not of its centre, so it differs from the rule above by
    up to half a field of view - which is enough to reverse the sign on a knee scanned
    near the midline. The dead zone is 5 mm rather than 20 mm, so it also commits on
    studies the rule above leaves unresolved.

    Neither difference changes a shape. Each one decides whether a study is mirrored, and
    a study mirrored one way at training and the other at inference presents the five
    side-defined targets with their axis reversed.
    """
    out = {}
    for st, g in h.groupby("StudyInstanceUID"):
        xs = []
        for r in g.itertuples(index=False):
            ipp = _hdr_vec(getattr(r, "ImagePositionPatient", None), 3)
            if ipp is not None and np.isfinite(ipp).all():
                xs.append(float(ipp[0]))
        if not xs:
            out[st] = None
            continue
        x = float(np.median(xs))
        # DICOM patient coordinates are LPS: +x is the patient's left.
        out[st] = None if abs(x) < LEGACY_LAT_OFFSET_MM else ("R" if x < 0 else "L")
    return out


def lat_of(h, tag=""):
    """Study -> 'L' / 'R' / None: the tag where it exists, geometry where it does not.

    The tag is present on exactly half the studies here and is sometimes an empty
    string rather than absent, which is not the same as NaN. Treating the other half
    as left-sided is what `normalise_laterality` did by omission, so the geometry
    fallback is not a refinement - it is the difference between normalising half the
    corpus and normalising all of it.
    """
    geo = side_from_corner_x(h) if RULES["lat"] == "corner_x" else side_from_geometry(h)
    d, n_tag, n_geo, n_none, n_disagree = {}, 0, 0, 0, 0
    for st, g in h.groupby("StudyInstanceUID"):
        v = [str(x).strip().upper() for x in g["Laterality"].dropna()]
        if RULES["lat"] == "corner_x" and "ImageLaterality" in g.columns:
            # The legacy rule reads the second tag too, so a study tagged only there is
            # resolved from the tag rather than from geometry.
            v += [str(x).strip().upper() for x in g["ImageLaterality"].dropna()]
        v = [x[0] for x in v if x and x[0] in ("L", "R")]
        side = v[0] if v else None
        if side is not None:
            n_tag += 1
            if geo.get(st) is not None and geo[st] != side:
                n_disagree += 1
        else:
            side = geo.get(st)
            n_geo += side is not None
            n_none += side is None
        d[st] = side
    log(f"{tag}laterality: {n_tag} from the tag, {n_geo} from geometry, "
        f"{n_none} unresolved; tag and geometry disagree on {n_disagree} "
        f"({n_disagree / max(n_tag, 1):.1%} of the tagged)")
    return d



def probe(item):
    split, study, series, path = item
    row = {"split": split, "StudyInstanceUID": study, "SeriesInstanceUID": series,
           "dir": path}
    try:
        files = sorted(e.name for e in os.scandir(path) if e.name.endswith(".dcm"))
        row["files"] = files
        row["n_slices"] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]),
                             stop_before_pixels=True, force=True)
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
                row[t] = "|".join(str(x) for x in v)
            else:
                row[t] = str(v)
    except Exception as exc:
        row["err"] = str(exc)[:120]
    return row


def walk(split):
    """Every series directory of a split, with one header read per series.

    An absent split returns an empty frame *with the columns annotate expects*. Returning
    a bare DataFrame looks like the same thing and is not: the next call indexes
    `SeriesDescription` and raises KeyError, so the branch that exists to survive a
    missing split is what turns it into a crash.
    """
    base = ROOT / split
    items = []
    if not base.is_dir():
        return pd.DataFrame(columns=["split", "StudyInstanceUID", "SeriesInstanceUID",
                                     "dir", "files", "n_slices"] + HDR_TAGS)
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)


def annotate(df):
    """Recover fat suppression and pulse-sequence weighting from the header."""
    desc = (df["SeriesDescription"].fillna("") + " " + df["SequenceName"].fillna(""))
    desc = desc.str.lower().str.replace(_SEP, " ", regex=True)

    opts = df["ScanOptions"].fillna("").str.upper().str.split("|")
    # GE writes SAT_GEMS for spatial saturation, so ScanOptions must be matched as
    # exact tokens; a substring test on "SAT" fires on non-fat-sat series.
    opts_fs = opts.apply(lambda ts: any(t.strip() in FATSAT_OPTS for t in ts))
    df["fatsat"] = desc.str.contains(_FATSAT_RX) | opts_fs

    tr = pd.to_numeric(df["RepetitionTime"], errors="coerce")
    te = pd.to_numeric(df["EchoTime"], errors="coerce")
    gre = df["ScanningSequence"].fillna("").str.upper().str.contains("GR")
    t1, t2, pdw = desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX)

    df["weight"] = np.where(t1 & ~t2 & ~pdw, "T1",
                     np.where(t2 & ~pdw, "T2",
                       np.where(pdw, "PD",
                         np.where(gre, "GRE",
                           np.where(tr < 800, "T1",
                             np.where(te > 60, "T2",
                               np.where(tr >= 800, "PD", "UNK")))))))
    df["fluid"] = np.isin(df["weight"], ["PD", "T2"])
    df["px"] = pd.to_numeric(
        df["PixelSpacing"].fillna("").str.split("|").str[0].replace("", np.nan),
        errors="coerce")
    return df


# %% [notebook cell 8]
def pick_slots(series_df, plane_map):
    """One series per slot per study.

    Ties are broken toward the stack with the most slices: a thicker stack samples the
    joint more densely, and the three-slice sampler below benefits from the margin.
    """
    series_df = series_df.copy()
    series_df["plane"] = series_df["SeriesInstanceUID"].map(plane_map)
    out = {}
    for study, g in series_df.groupby("StudyInstanceUID"):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g["plane"] == plane) & (g["fatsat"] == fs)
            # fluid=None means "do not condition on weighting" - the public scheme,
            # where the single provided flag stands in for both axes at once.
            if fluid is not None:
                sel &= (g["fluid"] == fluid)
            cand = g[sel]
            # A slot with no series matching its predicate stays empty, and no substitute
            # is admitted from a neighbouring predicate. Relaxing the weighting to fill a
            # T1 slot would draw from the pool `SAG_FLUID_NOFS` selects from, since that
            # pool is what remains once the weighting is dropped: over the training corpus
            # it would put one series in two slots for 2383 of 4407 studies and leave 56%
            # of the T1 slot holding PD or T2. The presence mask would then assert a
            # sequence that was never acquired, and the per-diagnosis softmax of §6 would
            # divide its attention across two identical slots, giving one acquisition
            # about twice the weight it carries in a study that holds both. The mask is
            # there to say a slot is absent, which is what an absent slot is.
            if len(cand) == 0 and RULES["slot_fallback"] and fluid is False:
                # The relaxation the paragraph above rejects, reproduced because an
                # imported member was fitted with its T1 slots filled this way: over half
                # of that member's training studies had a T1 slot holding a series that
                # is not T1. Leaving those slots empty would present it with a presence
                # mask it never saw.
                cand = g[(g["plane"] == plane) & (~g["fatsat"])]
            if len(cand):
                chosen[name] = cand.sort_values("n_slices", ascending=False).iloc[0]
        out[study] = chosen
    return out


# %% [notebook cell 10]
ORDER_TAGS = [(0x0020, 0x0032), (0x0020, 0x0037), (0x0020, 0x0013)]

# Series in which at least one sampled slice would not decode. A list rather than a
# counter because appending is atomic under the reader threads, and reported rather than
# swallowed: unreported, a decode failure is indistinguishable from a black knee.
DECODE_FAILED = []


def cache_tag(rules=None):
    """The name a decoded cache is stored under.

    It has to name everything that decides the pixels, not only their dimensions. Two
    configurations that agree on resolution, slice count, crop and band but disagree on
    how a slice is chosen produce different arrays of identical shape - so a tag built
    from the dimensions alone lets the second attach to the first one's file and train
    against pixels it never asked for, with nothing anywhere reporting a mismatch.

    A native reading keeps the plain name, so caches decoded before the rules existed
    stay valid; anything else earns a suffix.
    """
    r = dict(RULES if rules is None else rules)
    t = (f"{CACHE_IMG}px_{CACHE_SLICES}sl_{int(CROP_MM)}mm_"
         f"{SLICE_BAND[0]:.2f}-{SLICE_BAND[1]:.2f}")
    if {k: r.get(k, v) for k, v in RULES_NATIVE.items()} != RULES_NATIVE:
        t += "_" + hashlib.md5(json.dumps(r, sort_keys=True).encode()).hexdigest()[:6]
    return t


def _natural_key(name):
    return tuple(int(x) if x.isdigit() else x.lower()
                 for x in re.split(r"(\d+)", str(name)))


def _order_dominant_axis(rec):
    """The slice order an imported member was fitted under.

    It sorts on the raw patient coordinate along whichever axis varies most across the
    stack, rather than on the projection onto the slice normal. The two differ by a sign,
    not by a formula: measured over this corpus every sagittal series has a slice normal
    with n_x in [-1.00, -0.98], so p.n is the negative of the raw x this sorts on and the
    two stacks come out exactly reversed. Because the band sampler truncates rather than
    rounds, its nine indices are not symmetric about the middle, so nine slices drawn from
    a twenty-six slice stack under one order share two with the other.

    Missing geometry falls back to `InstanceNumber` and then to a natural sort of the file
    name, both at the same 80% threshold the imported pipeline used.
    """
    files, d = rec["files"], rec["dir"]
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True,
                                 specific_tags=["ImagePositionPatient", "InstanceNumber"])
            raw = getattr(ds, "ImagePositionPatient", None)
            if raw is not None and len(raw) >= 3:
                c = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(c).all():
                    ipp = c
            n = getattr(ds, "InstanceNumber", None)
            if n is not None:
                inst = float(n)
        except Exception:
            pass
        rows.append((f, ipp, inst, pos))

    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(key=lambda r: (float(r[1][axis]) if r[1] is not None else spare,
                                 r[2] if r[2] is not None else float("inf"), r[3]))
    elif sum(r[2] is not None for r in rows) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float("inf"), r[3]))
    else:
        rows.sort(key=lambda r: _natural_key(r[0]))
    return [r[0] for r in rows], True


def order_slices(rec):
    """Return the series' files sorted along the through-plane axis.

    A DICOM file name here is a SOP Instance UID, which is assigned arbitrarily. Sorting
    by it therefore produces an order uncorrelated with anatomy - measured over one
    series, Spearman between file-name rank and physical position is 0.009, i.e. none.
    Anything that assumes the file order means something is then operating on noise: the
    three channels of a "2.5D" input are three unrelated views rather than neighbouring
    slices, "the middle of the stack" is a random subset, and reversing slice order to
    normalise laterality reverses nothing meaningful.

    The physical order is recoverable exactly. Each slice carries its position in patient
    coordinates and the in-plane axes; projecting the position onto the slice normal
    gives a signed through-plane coordinate, monotonic along the stack:

        n = r_x  x  r_y ,      k = p . n

    `InstanceNumber` is the fallback. It usually tracks the projection up to sign, but
    interleaved and multi-echo acquisitions need not number slices in the order they
    occupy in space - but the projection is signed in patient
    coordinates, which is what laterality normalisation needs.
    """
    if RULES["order"] == "dominant_axis":
        return _order_dominant_axis(rec)
    files, d = rec["files"], rec["dir"]
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True,
                                 specific_tags=ORDER_TAGS)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        keyed.append((k, f))
    if any(k is None for k, _ in keyed):
        # A series with no usable geometry keeps its arbitrary order; that is worse than
        # sorting but better than dropping the series, and it is logged as a count.
        return files, False
    return [f for _, f in sorted(keyed, key=lambda t: t[0])], True


def read_slot(rec, n_slice=None, out_size=None):
    """`n_slice` physically spread slices from one series, at `out_size` pixels.

    Returns uint8 [n_slice, out, out] normalised per-series to its 1st-99th
    percentile. Percentiles rather than min/max because MR intensity has no absolute
    scale and a single bright vessel would otherwise compress the whole dynamic range.

    Reading is the expensive half of this pipeline, so the caller reads once at the
    largest configuration it needs and derives the smaller ones from the returned buffer
    rather than re-reading.
    """
    n_slice = GROUP if n_slice is None else n_slice
    out_size = IMG if out_size is None else out_size
    files, d, px = rec.get("ordered") or rec["files"], rec["dir"], rec["px"]
    n = len(files)
    if n == 0:
        return None
    # Spread the samples over a central band of the stack: the outermost slices of a knee
    # series are mostly soft tissue outside the joint. The band is a constant rather than
    # a literal because how much of the stack is worth reading depends on how many slices
    # are being taken - at three the middle is all that fits, while at sixteen the ends
    # are worth having, and a Baker cyst sits at the posteromedial end of a sagittal one.
    lo, hi = int(SLICE_BAND[0] * (n - 1)), int(SLICE_BAND[1] * (n - 1))
    idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])

    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, "RescaleSlope", 1) or 1)
            ic = float(getattr(ds, "RescaleIntercept", 0) or 0)
            a = a * sl + ic
        except Exception:
            a = None                      # no shape is known here; see below
        planes.append(a)

    # A slice that would not decode has no shape of its own, and inventing one is how a
    # single unreadable file erases a whole series: a substitute allocated at the resize
    # target while the decoded slices are still native makes the shape check below take
    # the substitute as the authority and zero the good slices with it, leaving a black
    # slot that the presence mask still reports as acquired.
    #
    # A failure is instead filled from the nearest slice that did decode - the same
    # convention the sampler already uses when the band holds fewer distinct slices than
    # were asked for - and a series where nothing decodes is reported absent, which the
    # mask can express, rather than black, which it cannot.
    got = [k for k, p in enumerate(planes) if p is not None]
    if RULES["decode_fill"] == "zero":
        # What an imported member was fitted with: a failure becomes a zero plane at the
        # resize target, which the shape check below then propagates to the whole slot.
        # It is the behaviour the paragraph above describes and rejects, kept here only
        # because that member's weights were learned against slots blacked out this way.
        if not got:
            DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        planes = [np.zeros((out_size, out_size), np.float32) if p is None else p
                  for p in planes]
        got = list(range(len(planes)))
    if not got:
        DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        return None
    if len(got) < len(planes):
        DECODE_FAILED.append(rec.get("SeriesInstanceUID", d))
        for k, p in enumerate(planes):
            if p is None:
                planes[k] = planes[min(got, key=lambda j: abs(j - k))]

    # Slices of one series can still differ in matrix size - multi-echo and some
    # reformats do - and those are genuinely not stackable.
    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)

    # constant physical extent, then resize: PixelSpacing varies 3.4x across the corpus
    if px and np.isfinite(px) and px > 0:
        want = int(round(CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = h // 2, w // 2
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]

    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-6), 0, 1)

    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    # uint8, not float32. These buffers queue up between the reader threads and the
    # encoder, and at this size a float32 slot-series is several megabytes. Intensity is
    # already normalised into [0, 1] here, so eight bits cost nothing that a bilinear
    # resize has not already cost, and the queue is a quarter the size.
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)


# %% [notebook cell 12]
def normalise_laterality(img, plane, lat):
    """Map every knee onto a left-knee convention.

    Coronal and axial views mirror under a horizontal flip. Sagittal stacks are not
    mirror images of each other - the slice order runs medial-to-lateral in opposite
    directions - so the channel order is reversed instead.
    """
    if lat != "R":
        return img
    if plane in ("Coronal", "Axial"):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])


# %% [notebook cell 14]
# Where the geometric slice order may be remembered between runs. Unset on the platform,
# because each run gets a fresh machine and there is nothing to remember; set off it,
# where the same corpus is cached again at every resolution and slice count and the order
# is a function of neither. It is opt-in so that the scored run's behaviour is decided by
# the code rather than by whether a file happens to be lying about.
ORDER_CACHE = os.environ.get("RSNA_ORDER_CACHE") or None


def build_cache(slot_map, plane_map, lat_map, tag):
    """Decode every (study, slot) once into an in-memory uint8 array.

    Fine-tuning revisits the same pixels every epoch. Reading them from the mount each
    time would make the epoch count a function of I/O rather than of learning, so they
    are decoded once and held as bytes: intensity has already been normalised into
    [0, 1], and eight bits cost nothing a bilinear resize has not already cost.

    CACHE_SLICES positions are kept per slot, which the training loop reads as N_GROUP
    groups of GROUP consecutive channels.
    """
    studies = sorted(slot_map)
    sidx = {s: i for i, s in enumerate(studies)}
    cache = np.zeros((len(studies), N_SLOT, CACHE_SLICES, IMG, IMG), np.uint8)
    mask = np.zeros((len(studies), N_SLOT), np.float32)
    log(f"{tag}: cache {cache.shape} = {cache.nbytes / 1024 ** 3:.1f} GB")

    jobs = [(st, k, plane, slot_map[st][name])
            for st in studies
            for k, (name, plane, _, _) in enumerate(SLOTS)
            if name in slot_map[st]]
    n_job = len(jobs)

    # Ordering first, and as its own pass. It reads one header per slice of every chosen
    # series - far more file opens than the decode that follows - and on a network mount
    # that is latency, not work, so it gets its own wider pool.
    t_ord = time.time()
    n_slice_total = sum(len(j[3]["files"]) for j in jobs)
    log(f"{tag}: ordering {len(jobs)} slot-series ({n_slice_total} slice headers)")
    ok = done = 0
    CHUNK_O = 1024

    # A remembered order, when one is offered. The projection depends on the DICOM
    # geometry alone, so it is the same at every resolution and every slice count, and
    # it costs one header read per slice - the largest single cost in this pass. An entry
    # is validated by the number of files present, so a tree that has changed under it is
    # recomputed rather than trusted: order is derived data, and a stale entry would be
    # invisible in the way that matters most.
    seen = {}
    if ORDER_CACHE and Path(ORDER_CACHE).is_file():
        try:
            import json as _json
            seen = _json.loads(Path(ORDER_CACHE).read_text())
        except (OSError, ValueError):
            seen = {}
        hit = 0
        for _, _, _, rec in jobs:
            e = seen.get(rec["SeriesInstanceUID"])
            if e and len(e["files"]) == len(rec["files"]):
                rec["ordered"] = e["files"]
                ok += int(e["good"])
                hit += 1
        jobs = [j for j in jobs if "ordered" not in j[3]]
        log(f"{tag}: {hit} slot-series ordered from {ORDER_CACHE}, {len(jobs)} to read")

    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK_O):
            block = jobs[c0:c0 + CHUNK_O]
            for (_, _, _, rec), (files, good) in zip(
                    block, pool.map(lambda j: order_slices(j[3]), block)):
                rec["ordered"] = files
                ok += int(good)
                done += 1
                if ORDER_CACHE:
                    seen[rec["SeriesInstanceUID"]] = {"files": files, "good": bool(good)}
            # The ceiling is whichever comes first: the pass's own budget, or the share
            # of what is left of the run that it may take. The second is what makes the
            # first safe to set generously - a mount slow enough to matter cannot spend
            # the training time, because the budget shrinks as the run does.
            budget = min(ORDER_BUDGET_S, max(60.0, (TIME_BUDGET - (time.time() - T0)) * 0.35))
            if time.time() - t_ord > budget:
                log(f"{tag}: ordering budget spent at {done}/{len(jobs)}; "
                    f"the rest keep file order")
                break
    if ORDER_CACHE and done:
        import json as _json
        _t = Path(ORDER_CACHE).with_suffix(".tmp")
        _t.write_text(_json.dumps(seen))
        _t.replace(Path(ORDER_CACHE))
    log(f"{tag}: ordered {ok}/{n_job} by geometry "
        f"({n_job - ok} kept arbitrary) in {time.time() - t_ord:.0f}s")

    jobs = [(st, k, plane, slot_map[st][name])
            for st in studies
            for k, (name, plane, _, _) in enumerate(SLOTS)
            if name in slot_map[st]]
    log(f"{tag}: decoding {len(jobs)} slot-series")
    n_failed_before = len(DECODE_FAILED)

    CHUNK = 512
    done = 0
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK):
            block = jobs[c0:c0 + CHUNK]
            for (st, k, plane, _), img in zip(
                    block, pool.map(lambda j: read_slot(j[3], CACHE_SLICES, IMG), block)):
                done += 1
                if img is None:
                    continue
                cache[sidx[st], k] = normalise_laterality(img, plane,
                                                          lat_map.get(st)).numpy()
                mask[sidx[st], k] = 1.0
            if done % 4096 < CHUNK:
                log(f"  {tag} {done}/{len(jobs)}")
            if time.time() - T0 > TIME_BUDGET:
                log(f"  {tag}: time budget reached during decode")
                break
    n_failed = len(DECODE_FAILED) - n_failed_before
    log(f"{tag}: {int(mask.sum())}/{len(jobs)} slots filled"
        + (f"; {n_failed} series had a slice that would not decode" if n_failed else ""))
    gc.collect()
    return studies, cache, mask


# %% [notebook cell 16]
class SlotHead(nn.Module):
    """Per-diagnosis attention over the slot embeddings of one study.

    Each finding is read on particular sequences - cruciates sagittally, collateral
    ligaments and the meniscal body coronally, patellar cartilage axially - so pooling
    the slots identically would dilute the one that carries the evidence with the rest.

    The aggregation is deliberately this simple. With a study-level label there is no
    signal telling the model which part of a study matters, so extra attention
    parameters below the slot level would have nothing to learn from and would spend
    their capacity fitting noise.
    """

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        # An imported member carries a fixed per-(diagnosis, slot) tilt on the attention
        # logits, set from the anatomy table below rather than learned. It is a buffer, so
        # it travels in the state dict and must exist for that member to load; exp(0.55)
        # gives a preferred slot about 1.73x the weight of an unpreferred one, which
        # biases the softmax without ever excluding a slot.
        p_ = torch.zeros(n_out, n_slot)
        if prior and n_slot == len(SLOTS) and n_out == len(TARGETS):
            for t, slots in SLOT_PRIOR_TABLE.items():
                if t in TARGETS:
                    p_[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
        self.prior = prior
        if prior:
            self.register_buffer("slot_prior", p_)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class Model(nn.Module):
    """Encoder plus head, trained end to end.

    A study arrives as a bag of slot images. The bag is flattened for the encoder and
    folded back before the head, so the encoder never sees the study structure and the
    head never sees pixels.
    """

    def __init__(self, backbone, dim, pool="cls_mean", prior=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, img_size=None):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            # The cache is held at the highest resolution any configuration needs; the
            # rest downsample from it, so every configuration sees the same pixels
            # through a different sampling grid rather than a different crop.
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                              align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == "cls_mean_focal":
            # The upper tail of each channel over the patch grid, taken per channel
            # rather than by selecting whole patches: a finding occupies a small part of
            # the field, so a plain mean over 256 patches dilutes it by two orders of
            # magnitude, and this keeps the top eighth of each channel's responses.
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        feat = torch.cat(parts, dim=1).reshape(B, S, -1)
        return self.head(feat, mask)


def build_model(unfreeze_last, source=None, variant="small", pool="cls_mean",
                prior=False):
    """Load the encoder and open the last `unfreeze_last` blocks for training.

    The early blocks of a self-supervised transformer are generic edge and texture
    filters; the late blocks carry semantics. Opening only the late ones is the cautious
    choice - there may not be enough supervision here to improve the early ones and there
    is certainly enough to damage them - but how far the line should sit is a question
    the corpus has to answer rather than the intuition.

    `source` names where the weights come from. Left unset it is the attached model
    directory, which is the only thing available here. It is a parameter so that a run
    off the platform builds the same object from the same code rather than from a second
    definition that has to be kept in step by hand.
    """
    from transformers import AutoModel
    p = source if source is not None else find_dinov2(variant)
    if p is None:
        raise FileNotFoundError("DINOv2 weights not attached")
    bb = AutoModel.from_pretrained(str(p))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size
    trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    log(f"backbone: {n_layer} blocks, last {unfreeze_last} trainable "
        f"({trainable / 1e6:.1f}M params), feature dim {dim * POOL_PARTS[pool]}")
    return Model(bb, dim, pool=pool, prior=prior)


# %% [notebook cell 18]
def take_group(cache_rows, g):
    """Slice GROUP consecutive channels out of the cached slices."""
    return cache_rows[:, :, g * GROUP:(g + 1) * GROUP]


def augment(imgs):
    """A small rigid jitter and an intensity scale, applied to a whole bag at once.

    Neither flip is available here, and for different reasons. A horizontal flip would
    reintroduce the nuisance axis that the laterality normalisation removed - it would
    undo, once per batch, what the header pass was run to establish.

    A vertical flip is not a nuisance axis at all. A knee is acquired in a canonical
    orientation, and no study in this corpus looks like its own vertical mirror. An
    augmentation is meant to cover directions along which the label does not change; this
    one moves the input off the distribution the encoder will be asked about, which is a
    different thing. Where a finding sits in the frame is also information rather than
    noise - a Baker cyst is identified by lying in the popliteal fossa, not by its
    appearance alone.

    What is left is jitter that no label depends on: a few degrees of rotation, a few
    per cent of scale and translation. That still prevents memorising the exact framing,
    which is what an augmentation is for, while leaving the anatomy where it was.
    """
    # A bag arrives as [study, slot, GROUP, IMG, IMG]: five axes, not four. The warp is
    # a 2-D operation, so the two leading axes are folded together and restored after -
    # every slot image is an independent acquisition and gets its own jitter.
    lead = imgs.shape[:-3]
    x = imgs.reshape(-1, *imgs.shape[-3:]).float()
    n, dev = x.shape[0], x.device

    rot = (torch.rand(n, device=dev) - 0.5) * 2 * (AUG_ROT_DEG * np.pi / 180)
    # Zoom in only. `border` padding repeats the edge row outward, and the edge of this
    # crop is where the popliteal fossa sits; zooming out would fabricate tissue exactly
    # where a Baker cyst is looked for.
    sc = 1.0 + torch.rand(n, device=dev) * AUG_SCALE
    tx = (torch.rand(n, device=dev) - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(n, device=dev) - 0.5) * 2 * AUG_SHIFT
    cos, sin = torch.cos(rot) / sc, torch.sin(rot) / sc
    theta = torch.zeros(n, 2, 3, device=dev, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    scale = 1.0 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 2 * AUG_INTENSITY
    x = (x * scale).clamp(0, 255)
    return x.reshape(*lead, *x.shape[-3:]).to(imgs.dtype)


# %% [notebook cell 20]
FINGERPRINT_TOL = 2e-3


def fingerprint(model, dev, img_size, n_slot=None, group=None, seed=None):
    """The model's output on a fixed synthetic bag, as a portable identity.

    Weights that are loaded but read through the wrong preprocessing produce predictions,
    not errors. The submission is well formed, the log says nothing, and the difference is
    a number no output of the run reveals. Scaling that never happens, or happens twice,
    is enough on its own and changes no shape anywhere.

    So a set of weights carries the answer it gave to a question with no data in it. The
    input is generated from a seed rather than read, so it is the same on any machine, and
    it is pushed through the whole forward path - the byte scaling, the ImageNet
    normalisation, the resize, the encoder, the slot attention. Any of those differing
    moves the output by order one. Numerics differing between two GPUs moves it by about
    1e-5, which is why the tolerance sits between them rather than at zero.

    This checks that the model computes what it computed when it was fitted. It cannot
    check that the pixels reaching it are the right pixels; `read_slot` and the header
    pass answer to their own tests.
    """
    n_slot = N_SLOT if n_slot is None else n_slot
    group = GROUP if group is None else group
    seed = SEED if seed is None else seed
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randint(0, 256, (2, n_slot, group, img_size, img_size),
                         generator=g, dtype=torch.uint8).to(dev)
    mask = torch.ones(2, n_slot, device=dev)
    mask[1, -1] = 0.0                       # exercise the masked branch of the softmax
    was_training = model.training
    model.eval()
    with torch.no_grad():
        # float32 throughout: autocast would make the value depend on which device
        # happened to run it, and the point of the number is that it does not.
        out = model(imgs, mask, img_size).float().cpu().numpy()
    if was_training:
        model.train()
    return out


def check_fingerprint(model, dev, img_size, expected, tol=FINGERPRINT_TOL, tag=""):
    """Compare against a stored fingerprint; raise when the model is not the same map."""
    got = fingerprint(model, dev, img_size)
    exp = np.asarray(expected, np.float32)
    if got.shape != exp.shape:
        raise WeightsError(f"{tag}fingerprint shape {got.shape} != stored {exp.shape}: "
                           f"the architecture is not the one these weights were fitted to")
    d = float(np.abs(got - exp).max())
    if d > tol:
        raise WeightsError(
            f"{tag}fingerprint differs by {d:.4g} (tolerance {tol:g}). The weights load "
            f"but do not compute what they computed when fitted - preprocessing, "
            f"resolution or architecture has moved between the two runs.")
    log(f"{tag}fingerprint matches within {d:.2g}")
    return d


class WeightsError(RuntimeError):
    """Raised when attached weights cannot be trusted to be the ones that were fitted.

    Deliberately fatal for the same reason as LabelSourceError: a run that predicts from
    a mismatched model completes, writes a plausible submission, and differs from a
    correct one only in a number no output of the run reveals.
    """


def find_weights(name="manifest.json"):
    """Locate a mounted weights package, or return None if none is attached.

    Same shape as `find_label_table`: the notebook must keep working for a reader who
    attaches nothing, so absence is a path rather than an error. What must not be silent
    is a package that is attached and unusable, and that is what `load_weights` refuses.
    """
    import json
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name not in files:
            continue
        # The manifest decides, not the filenames beside it. Testing for a naming
        # convention makes the search agree with whatever the packager happened to call
        # its files last, which is a second definition of what a package is.
        try:
            man = json.loads((Path(root) / name).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(man.get("members"), list) and man["members"]:
            missing = [m["file"] for m in man["members"]
                       if not (Path(root) / m["file"]).is_file()]
            if missing:
                raise WeightsError(
                    f"{root} holds a manifest listing {len(man['members'])} members but "
                    f"{len(missing)} of their files are absent (first {missing[0]!r})")
            return Path(root)
    return None


# How a member is read at inference. Overlapping windows over the slices the cache
# already holds cost forward passes and no extra decoding, which is the cheap direction
# to spend; and averaging probabilities rather than logits is an arithmetic mean of risk
# rather than a geometric mean of odds, which orders studies differently. Both were
# chosen by measuring them on the folds each member held out rather than by argument.
TTA_OVERLAP = True
TTA_POOL = "prob"

# Per-target pooling over TTA windows. A fracture, a meniscal tear or a Baker's cyst
# lives on a few slices: when one window sees it and nine do not, the mean dilutes the
# detection, so the focal findings take the max (or top-k mean) over windows. Diffuse
# findings -- the three OA compartments, effusion, synovitis -- spread across the joint
# and keep the plain mean. Labels not listed keep the original probability mean exactly.
TTA_TARGET_POOL = {
    "Fracture": "max",
    "Contusion": "max",
    "Medial Meniscus": "max",
    "Lateral Meniscus": "max",
    "ACL": "top3",
    "MCL": "top3",
    "Baker's": "max",
}


def window_starts(n_slice, group, overlap=None):
    """Where each TTA window begins."""
    overlap = TTA_OVERLAP if overlap is None else overlap
    if overlap and n_slice >= group:
        return list(range(n_slice - group + 1))
    return [g * group for g in range(max(n_slice // group, 1))]


@torch.no_grad()
def predict_member(model, cache, mask, idx, dev, img_size, group=None, pool=None,
                   starts=None, jitter=False):
    """One member's predictions, pooled over its TTA windows.

    `jitter` adds one extra view per window drawn from the same rigid-jitter family the
    members were trained under; jittered views are averaged WITHIN their window first,
    so jitter smooths a window's estimate without blunting the across-window max that
    the focal findings rely on.
    """
    group = GROUP if group is None else group
    pool = TTA_POOL if pool is None else pool
    starts = window_starts(cache.shape[2], group) if starts is None else list(starts)
    if not starts:
        raise ValueError("predict_member was given no windows to average over")
    target_idx = {t: j for j, t in enumerate(TARGETS)}
    unknown = set(TTA_TARGET_POOL) - set(target_idx)
    if unknown:
        raise ValueError(f"unknown target(s) in TTA_TARGET_POOL: {unknown}")
    model.eval()
    out = []
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[sel]).to(dev)
        acc, n_pass, win_probs = None, 0, []
        for st in starts:
            rows = torch.from_numpy(
                np.ascontiguousarray(cache[sel, :, st:st + group])).to(dev)
            views = [rows] + ([augment(rows)] if jitter else [])
            wacc = None
            for view in views:
                with torch.autocast("cuda", enabled=dev.type == "cuda"):
                    z = model(view, m, img_size).float()
                p = torch.sigmoid(z)
                v = z if pool == "logit" else p
                acc = v if acc is None else acc + v
                n_pass += 1
                wacc = p if wacc is None else wacc + p
            win_probs.append(wacc / len(views))
        v = acc / n_pass
        final = torch.sigmoid(v) if pool == "logit" else v
        if TTA_TARGET_POOL:
            probs = torch.stack(win_probs, dim=0)      # [window, batch, target]
            for target, mode in TTA_TARGET_POOL.items():
                j = target_idx[target]
                col = probs[:, :, j]
                if mode == "mean":
                    final[:, j] = col.mean(dim=0)
                elif mode == "logit_mean":
                    final[:, j] = torch.sigmoid(torch.logit(
                        col.clamp(1e-6, 1 - 1e-6)).mean(dim=0))
                elif mode == "max":
                    final[:, j] = col.max(dim=0).values
                elif mode in ("top2", "top3"):
                    k = min(int(mode[3:]), col.shape[0])
                    final[:, j] = col.topk(k, dim=0).values.mean(dim=0)
                else:
                    raise ValueError(f"unknown TTA pool mode {mode!r}")
        out.append(final.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)



# HF from_pretrained mutates process-global state and is not guaranteed thread-safe, so
# model construction, weight loading and the fingerprint check are serialised; only
# inference -- the expensive part -- runs on both devices at once.
BUILD_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()

# An independently trained five-fold bundle joins the vote at reduced weight: a second
# training run (public 0.836 on its own) adds decorrelated errors, which is the only thing an
# inference-only run can add that the 20-member package does not already have. 0.5 per
# fold puts the bundle at ~11% of the total vote -- roughly its quality gap.
LEGACY_BUNDLE_FILE = "rsna_20260807_v1.pt"
# Per-target legacy vote selected before leaderboard submission.  `merge_gain.npz`
# contains OOF predictions for the 20-member package (ours) and this independently
# trained five-fold bundle (imported).  A target is blended only when it is non-worse
# on all 4,407 derived-label OOF rows and on the disjoint 58 gold rows.  The selected
# alpha is the legacy family's share of the final rank vote for that target.
LEGACY_TARGET_ALPHA = {
    "ACL": 0.00,
    "MCL": 0.10,
    "Medial Meniscus": 0.00,
    "Lateral Meniscus": 0.00,
    "Medial OA": 0.00,
    "Lateral OA": 0.20,
    "PF OA": 0.20,
    "Effusion": 0.35,
    "Synovitis": 0.00,
    "Baker's": 0.05,
    "Contusion": 0.00,
    "Fracture": 0.10,
}
N_PACKAGE_MEMBERS = 20
N_LEGACY_FOLDS = 5
LEGACY_TARGET_WEIGHT = np.array([
    (N_PACKAGE_MEMBERS * LEGACY_TARGET_ALPHA[t]) /
    (N_LEGACY_FOLDS * (1.0 - LEGACY_TARGET_ALPHA[t]))
    if LEGACY_TARGET_ALPHA[t] > 0 else 0.0
    for t in TARGETS
], dtype=np.float64)


def find_legacy_bundle():
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if LEGACY_BUNDLE_FILE in files:
            return Path(root) / LEGACY_BUNDLE_FILE
    return None


def legacy_group_members():
    """The five-fold bundle as extra, lower-weight members under RULES_LEGACY.

    The package's legacy pixel rules exist precisely to reproduce what this bundle was
    fitted on (dominant-axis slice order, corner-x laterality at 5 mm, T1 slot fallback,
    zero decode fill, 160 mm crop, central 60% band). The bundle predates fingerprints,
    which is accepted loudly and priced into its reduced weight; a fold whose state dict
    does not load, or whose predictions are degenerate, is dropped and costs its own
    vote only.
    """
    p = find_legacy_bundle()
    if p is None:
        log("no legacy bundle attached; blending skipped")
        return {}
    try:
        b = torch.load(p, map_location="cpu", weights_only=False)
        folds = b.get("fold_states") or []
        b_slots = [tuple(s)[0] for s in b.get("slots", SLOTS)]
        if list(b.get("targets", TARGETS)) != TARGETS or b_slots != [s[0] for s in SLOTS]:
            log(f"legacy bundle {p.name}: target/slot contract differs; blending skipped")
            return {}
        gr, n_gr = int(b.get("group", 3)), int(b.get("n_group", 3))
        variant = str(b.get("model_variant", "dinov2-small")).split("-")[-1]
        key = json.dumps({"img": int(b.get("img", 224)), "group": gr,
                          "slices": gr * n_gr, "crop_mm": 160.0, "band": [0.20, 0.80],
                          "rules": RULES_LEGACY, "slots": [s[0] for s in SLOTS]},
                         sort_keys=True)
        ms = [{"id": f"legacy-f{f.get('fold', k)}", "fold": f.get("fold", k),
               "state": f["state_dict"], "holdout": None,
               "weight": LEGACY_TARGET_WEIGHT.copy(),
               "pixel_group": key,
               "config": {"unfreeze_last": 6,
                          "variant": "base" if variant == "base" else "small",
                          "pool": "cls_mean_focal", "prior": True}}
              for k, f in enumerate(folds)]
        if ms:
            active = {t: LEGACY_TARGET_ALPHA[t] for t in TARGETS
                      if LEGACY_TARGET_ALPHA[t] > 0}
            log(f"legacy bundle {p.name}: {len(ms)} fold(s) join with "
                f"target-specific final shares {active}")
        return {key: ms} if ms else {}
    except Exception as exc:
        log(f"legacy bundle unusable ({type(exc).__name__}: {exc}); blending skipped")
        return {}


def _run_member(path, m, dev, Cte, Mte, idx, starts, jitter):
    """Load, verify and predict one member on one device. Returns (pred, timings)."""
    t0 = time.time()
    with BUILD_LOCK:
        if "state" in m:
            state, fp = m["state"], None
        else:
            ck = torch.load(Path(path) / m["file"], map_location="cpu",
                            weights_only=False)
            state, fp = ck["model"], ck.get("fingerprint")
        model = build_model(int(m["config"]["unfreeze_last"]),
                            variant=m["config"]["variant"],
                            pool=m["config"].get("pool", "cls_mean"),
                            prior=bool(m["config"].get("prior", False))).to(dev)
        model.load_state_dict(state)
        if fp is not None:
            check_fingerprint(model, dev, IMG, fp, tag=f"{m['id']}: ")
        else:
            log(f"  {m['id']}: no stored fingerprint (legacy bundle) -- "
                f"accepted at reduced weight")
    t_ready = time.time()
    p = predict_member(model, Cte, Mte, idx, dev, IMG, starts=starts, jitter=jitter)
    t_done = time.time()
    del model, state
    gc.collect()
    if dev.type == "cuda":
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    passes = len(starts) * (2 if jitter else 1)
    return p, (t_ready - t0, (t_done - t_ready) / max(passes, 1))


def _combine(per_member):
    """Weight-aware mean of per-member percentile ranks.

    Rank rather than probability, because the metric reads order and two members
    calibrated differently would otherwise have unequal say. Package members carry
    weight 1.0 -- their combiner is the one the 0.891 was measured under -- and legacy
    folds a reduced weight rather than a holdout-fitted one, which would import a few
    hundred training studies' sampling noise.
    """
    all_ids = sorted({s for m in per_member for s in m["ids"]})
    pos = {s: i for i, s in enumerate(all_ids)}
    acc = np.zeros((len(all_ids), len(TARGETS)), np.float64)
    tot = np.zeros(len(TARGETS), np.float64)
    for m in per_member:
        w = np.asarray(m.get("weight", 1.0), dtype=np.float64)
        if w.ndim == 0:
            w = np.full(len(TARGETS), float(w), dtype=np.float64)
        if w.shape != (len(TARGETS),):
            raise WeightsError(f"member {m['id']} has invalid target-weight shape {w.shape}")
        r = pd.DataFrame(m["pred"]).rank(pct=True).to_numpy()
        acc[[pos[s] for s in m["ids"]]] += r * w[None, :]
        tot += w
    if np.any(tot <= 0):
        raise WeightsError(f"ensemble has non-positive target weight(s): {tot}")
    return all_ids, acc / tot[None, :]


def infer_from_package(path):
    """Predict the test split from an attached package of trained members.

    Identical pixel contract to the reference implementation -- same caches, same
    windows, same fingerprints, same rank transform -- with executive changes only:

    1. A work queue over the devices: each GPU pops the next member when free (the
       reference ran one device and surrendered TTA windows, then members: 0.847).
    2. submission.csv is rewritten after every banked member, so a run killed at any
       point still submits the best partial ensemble instead of the 0.5 benchmark.
    3. A member that fails on one device is retried once on the other, then dropped;
       a dropped member costs one vote, never the run.
    4. An independently trained legacy bundle joins as five reduced-weight members.
    5. When the time estimate says the whole remaining ensemble fits with room to
       spare, each window gains one jittered TTA view (same family as training aug).
    """
    man = json.loads((Path(path) / "manifest.json").read_text())
    members = man["members"]
    log(f"weights package: {len(members)} member(s) from {path}; "
        f"{len(DEVS)} device(s)")

    test_df = pd.read_csv(ROOT / "test.csv")
    test_series = pd.read_csv(ROOT / "test_series.csv")
    plane_map = dict(zip(test_series["SeriesInstanceUID"],
                         test_series["Anatomical_Plane"]))
    hte = annotate(walk("test_series"))
    log(f"test header pass: {len(hte)} series")

    groups = {}
    for m in members:
        groups.setdefault(m["pixel_group"], []).append(m)
    # Legacy votes come last: if time binds after all, the weakest votes are the ones
    # surrendered, not the package's.
    groups.update(legacy_group_members())

    per_member = []
    est = {"fixed": None, "win": None}

    def bank(m, ids, pred, starts, jitter):
        if float(np.std(pred)) < 1e-9:
            log(f"  {m['id']}: degenerate predictions; not banked")
            return
        with STATE_LOCK:
            per_member.append({"id": m["id"], "ids": ids, "pred": pred,
                               "weight": m.get("weight", 1.0),
                               "holdout": m.get("holdout")})
            all_ids, acc = _combine(per_member)
            write_submission(acc, all_ids, test_df, "submission.csv")
            log(f"  banked {m['id']} fold {m.get('fold', '?')} "
                f"({len(starts)} window(s){', jitter' if jitter else ''}); "
                f"submission.csv = weighted rank mean of {len(per_member)} member(s)")

    for gi, (key, gm) in enumerate(groups.items(), 1):
        cfg = json.loads(key)
        adopt_config_globals(cfg)
        log(f"decode group {gi}/{len(groups)}: {cfg['img']}px x {cfg['slices']} slices, "
            f"crop {cfg['crop_mm']} mm -> {len(gm)} member(s)")
        st_te, Cte, Mte = build_cache(pick_slots(hte, plane_map), plane_map,
                                      lat_of(hte, "test "), f"test g{gi}")
        idx = np.arange(len(st_te))

        starts_full = window_starts(Cte.shape[2], GROUP)
        pending = sorted(gm, key=lambda m: -(m.get("holdout") or 0))
        left_after = sum(len(g) for j, (_, g) in enumerate(groups.items(), 1) if j > gi)

        def pop_next():
            """Next member, its window count, and whether jitter TTA is affordable.

            Windows are surrendered before members; jitter is granted only when the
            estimate says the whole remaining ensemble fits at double passes inside
            60% of the room. The question asked is whether ONE more member fits.
            """
            with STATE_LOCK:
                if not pending:
                    return None, None, False
                left = TIME_BUDGET - (time.time() - T0)
                remaining = len(pending) + left_after
                slots_left = -(-remaining // len(DEVS))       # ceil: concurrent slots
                starts, jit = starts_full, False
                if est["fixed"] is not None and est["win"] is not None:
                    afford = max(left * 0.9, 0.0)
                    room = afford / max(slots_left, 1)
                    if est["fixed"] + est["win"] > room:
                        log(f"  {left / 60:.0f} min left: surrendering "
                            f"{len(pending)} member(s); not one more fits")
                        pending.clear()
                        return None, None, False
                    jit = (ENABLE_JITTER_TTA and
                           est["fixed"] + 2 * len(starts_full) * est["win"]
                           <= room * 0.6)
                    per_win = est["win"] * (2 if jit else 1)
                    n_win = (int((room - est["fixed"]) / per_win)
                             if per_win > 0 else len(starts_full))
                    n_win = max(1, min(len(starts_full), n_win))
                    if n_win < len(starts_full):
                        mid = (len(starts_full) - n_win) // 2
                        starts = starts_full[mid:mid + n_win]
                return pending.pop(0), starts, jit

        def worker(dev):
            others = [d for d in DEVS if d is not dev]
            while True:
                m, starts, jit = pop_next()
                if m is None:
                    return
                for attempt, d in enumerate([dev] + others[:1]):
                    try:
                        p, (fs, ws) = _run_member(path, m, d, Cte, Mte, idx,
                                                  starts, jit)
                        with STATE_LOCK:
                            est["fixed"], est["win"] = fs, ws
                        bank(m, st_te, p, starts, jit)
                        break
                    except Exception as exc:
                        log(f"  MEMBER {m['id']} failed on {d} "
                            f"({type(exc).__name__}: {exc}); "
                            + ("retrying on peer device" if attempt == 0 and others
                               else "dropped -- costs one vote, not the run"))
                        if d.type == "cuda":
                            with torch.cuda.device(d):
                                torch.cuda.empty_cache()

        threads = [threading.Thread(target=worker, args=(d,)) for d in DEVS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        del Cte, Mte
        gc.collect()

    if not per_member:
        raise WeightsError("no member produced predictions; submission stays at 0.5")

    all_ids, acc = _combine(per_member)
    sub = write_submission(acc, all_ids, test_df, "submission.csv")
    log(f"final submission.csv = weighted rank mean of {len(per_member)} member(s); "
        f"{sub.shape}; nulls {int(sub[TARGETS].isna().sum().sum())}")
    return sub


def adopt_config_globals(cfg):
    """Point the pixel path at what one group of members was fitted on."""
    global IMG, CACHE_IMG, GROUP, CACHE_SLICES, N_GROUP, CROP_MM, SLICE_BAND, RULES
    CACHE_IMG = IMG = int(cfg["img"])
    GROUP = int(cfg["group"])
    CACHE_SLICES = int(cfg["slices"])
    N_GROUP = max(CACHE_SLICES // GROUP, 1)
    CROP_MM = float(cfg["crop_mm"])
    SLICE_BAND = tuple(float(x) for x in cfg["band"])
    # The four decisions that change what a slice is. A member fitted under one reading
    # and decoded under another gets pixels its weights never saw, with every shape
    # still agreeing, so an unrecognised name is refused rather than defaulted.
    rules = cfg.get("rules") or RULES_NATIVE
    unknown = {k: v for k, v in rules.items()
               if k not in RULES_NATIVE
               or v not in (RULES_NATIVE[k], RULES_LEGACY[k])}
    if unknown:
        raise WeightsError(f"the members record pixel rules this pipeline cannot "
                           f"reproduce: {unknown}")
    RULES = {**RULES_NATIVE, **rules}
    if [s[0] for s in SLOTS] != list(cfg["slots"]):
        raise WeightsError(
            f"the members were fitted on slots {cfg['slots']} and this pipeline defines "
            f"{[s[0] for s in SLOTS]}; a weight would be read against the wrong slot")


# %% [notebook cell 22]
def write_submission(pred, studies, test_df, path):
    """Write one submission file from a prediction matrix.

    Predictions are converted to per-column ranks first: the metric reads only order, so
    ranks discard nothing, and they make files from different configurations directly
    comparable and safe to average.
    """
    sub = pd.DataFrame(pd.DataFrame(pred).rank(pct=True).values, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", studies)
    sub = test_df[["StudyInstanceUID"]].merge(sub, on="StudyInstanceUID", how="left")
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv(path, index=False)
    return sub


def write_benchmark_submission():
    """Write the 0.5 benchmark file immediately.

    A submission that never writes scores nothing at all, which is strictly worse than
    scoring badly. The try/except around main() covers exceptions, but a kill for memory
    is a SIGKILL and never reaches it. So a valid file exists from the first second and
    is overwritten only once real predictions are ready.
    """
    t = pd.read_csv(ROOT / "test.csv")
    for c in TARGETS:
        t[c] = 0.5
    t.to_csv("submission.csv", index=False)



def main():
    write_benchmark_submission()
    pkg = find_weights()
    if pkg is None:
        raise WeightsError(
            "no weights package attached. This notebook is inference-only: attach the "
            "trained member package (the dataset holding manifest.json) and re-run.")
    infer_from_package(pkg)
    log("done")


try:
    main()
except Exception:
    traceback.print_exc()
    print("submission.csv holds the last banked state (or the 0.5 benchmark if "
          "nothing was banked)")
    raise

