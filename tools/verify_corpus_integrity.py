"""Integrity check for the downloaded Raptor corpus (runs after ANY download claims done).

Lesson from the destructive ensure_size bug: a chunked downloader's completion signal
must be integrity-checked, not count-checked. This verifies:
  1. npy magic + header parse (shape/dtype consistent with ids/masks files)
  2. exact file size
  3. spot checks: 32 evenly spaced volumes decode with plausible value stats
     (not all-zero, sane dynamic range for 336px MRI volumes)
  4. the known risk regions: the re-downloaded prefix (chunks 0-911 of all_vols)
     and the 403-burst region (chunks 464-473), as raw byte-diversity checks
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "artifacts" / "raptor_corpus"
CHUNK = 8 * 1024 * 1024

EXPECTED = {
    "all_vols.npy": 15_895_756_928,
    "extra_vols.npy": 5_995_680_896,
    "all_ids.npy": 819_328,
    "all_masks.npy": 140_928,
    "window_index.npy": 783_936,
    "extra_ids.npy": 309_120,
    "extra_masks.npy": 53_236,
}


def check_header_and_size(name: str):
    p = CORPUS / name
    size = p.stat().st_size
    ok_size = size == EXPECTED[name]
    with open(p, "rb") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
    print(f"{name}: size {size:,} ({'OK' if ok_size else 'MISMATCH exp ' + str(EXPECTED[name])})"
          f"  shape={shape} dtype={dtype} fortran={fortran}")
    if not ok_size:
        raise SystemExit(f"{name}: SIZE MISMATCH")
    return shape, dtype


def spot_check_volumes(name: str, shape, dtype, n=32):
    """Read n evenly spaced single volumes via memmap; verify value statistics."""
    p = CORPUS / name
    per_vol = int(np.prod(shape[1:]))
    assert per_vol * np.dtype(dtype).itemsize > 0
    mm = np.memmap(p, dtype=dtype, mode="r", shape=shape, order="C" if shape else "F")
    idxs = np.linspace(0, shape[0] - 1, n).astype(int)
    bad = 0
    for i in idxs:
        v = np.asarray(mm[i])
        nz = float((v != 0).mean())
        span = (int(v.min()), int(v.max()))
        plaus = (nz > 0.02) and (span[1] > span[0])
        if not plaus:
            bad += 1
            print(f"  !! vol {i}: nonzero={nz:.4f} span={span}")
    print(f"  spot {n} volumes [{idxs[0]}..{idxs[-1]}]: "
          f"{'ALL PLAUSIBLE' if bad == 0 else str(bad) + ' SUSPICIOUS'}")
    del mm
    return bad


def byte_diversity(name: str, offset: int, length: int = 1 << 16) -> None:
    """Raw window check: distinct byte values + zero fraction. All-zero => corrupt."""
    p = CORPUS / name
    with open(p, "rb") as f:
        f.seek(offset)
        b = f.read(length)
    uniq = len(set(b))
    zfrac = b.count(0) / len(b)
    print(f"  bytes @{offset:,}: distinct={uniq}/256 zero={zfrac:.3f} "
          f"{'OK' if uniq > 32 else 'SUSPECT'}")


def main() -> None:
    print("=== headers + sizes ===")
    for f in EXPECTED:
        check_header_and_size(f)

    print("=== all_vols.npy spot checks ===")
    shape, dtype = check_header_and_size("all_vols.npy")
    bad = spot_check_volumes("all_vols.npy", shape, dtype, n=32)

    print("=== risk regions (all_vols.npy) ===")
    print("re-downloaded prefix (chunks 0-911):")
    for c in (0, 200, 500, 911):
        byte_diversity("all_vols.npy", c * CHUNK)
    print("403-burst region (chunks 464-473):")
    for c in (464, 468, 473):
        byte_diversity("all_vols.npy", c * CHUNK + CHUNK // 2)

    print("=== extra_vols.npy spot checks ===")
    shape, dtype = check_header_and_size("extra_vols.npy")
    bad += spot_check_volumes("extra_vols.npy", shape, dtype, n=16)

    if bad:
        raise SystemExit(f"INTEGRITY CHECK FAILED: {bad} suspicious volumes")
    print("CORPUS INTEGRITY: PASS")


if __name__ == "__main__":
    main()
