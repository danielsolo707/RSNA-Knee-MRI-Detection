"""Parallel, resumable downloader for the public Raptor 336px corpus.

Why this exists: the single-stream `download_raptor_corpus.py` sustains ~0.22 MB/s on a
restricted route, while 8 parallel HTTP Range streams sustain ~0.49 MB/s (measured; 24
workers gives no further gain). That is a 2x wall-clock saving on a ~14 GB transfer.

It also fixes the progress readout. The old helper fetched 32 MB in memory and wrote it
in one `f.write()`, so its 1-second-delta bar showed 0.0 MB/s most seconds and a fake
33.5 MB/s on the flush second. That number is a write burst, NOT throughput, and the
ETAs derived from it were meaningless. This one reports a rolling average.

Design
  - File is preallocated to the full remote size; workers write at exact byte offsets.
  - A JSON manifest records completed chunk indices, so a kill/restart resumes cleanly.
  - Chunk size is 8 MB (better retry granularity than 32 MB).

Usage
    python tools/download_raptor_corpus_par.py            # both parts
    python tools/download_raptor_corpus_par.py --workers 8
    python tools/download_raptor_corpus_par.py --status
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "artifacts" / "raptor_corpus"
LOCK = DEST / ".download_par.lock"
TOKEN = Path.home() / ".kaggle" / "access_token"

CHUNK = 8 * 1024 * 1024
SMALL = 256 * 1024 * 1024

PARTS = [
    ("dreaddevelopment/knee-raptor-corpus",
     {"all_ids.npy": 819_328, "all_masks.npy": 140_928,
      "window_index.npy": 783_936, "all_vols.npy": 15_895_756_928}),
    ("dreaddevelopment/knee-raptor-corpus-ext",
     {"extra_ids.npy": 309_120, "extra_masks.npy": 53_236,
      "extra_meta.json": 85, "extra_vols.npy": 5_995_680_896}),
]

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auth_header() -> dict:
    tok = TOKEN.read_text(encoding="utf-8").strip() if TOKEN.exists() else ""
    if not tok:
        tok = os.environ.get("KAGGLE_API_TOKEN", "")
    return {"Authorization": f"Bearer {tok}"}


def signed_url(owner_slug: str, fname: str) -> str:
    import urllib.parse
    import urllib.request

    owner, slug = owner_slug.split("/")
    api = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{slug}/{fname}"
    req = urllib.request.Request(api, headers=auth_header())
    with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
        return urllib.parse.unquote(r.geturl())


def ensure_size(dst: Path, total: int) -> None:
    """Grow the file to `total` bytes WITHOUT destroying existing content.

    BUG THIS REPLACES: the old version did `open(dst, "wb")` which TRUNCATES to zero
    before `f.truncate(total)` extends it. On a partially-downloaded file that silently
    wiped every byte already fetched - it destroyed 7.65 GB of a 15.9 GB corpus, and the
    run then reported "100% complete" over a file that was half zeros.

    `open(dst, "r+b")` does not truncate, so previously written chunks survive.
    """
    if not dst.exists():
        with open(dst, "wb") as f:
            f.truncate(total)
        return
    cur = dst.stat().st_size
    if cur < total:
        with open(dst, "r+b") as f:      # r+b: preserves existing bytes
            f.truncate(total)
    # cur == total -> nothing to do; cur > total would be a real error, leave it alone.


def manifest_path(name: str) -> Path:
    return DEST / f".manifest_{name}.json"


def load_done(name: str) -> set:
    p = manifest_path(name)
    if p.exists():
        try:
            return set(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def save_done(name: str, done: set) -> None:
    manifest_path(name).write_text(json.dumps(sorted(done)), encoding="utf-8")


def fetch_whole(name: str, owner_slug: str, dst: Path, total: int) -> None:
    if dst.exists() and dst.stat().st_size == total and total > 0:
        log(f"{name}: present")
        return
    url = signed_url(owner_slug, name)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    dst.write_bytes(r.content)
    log(f"{name}: {len(r.content):,} B")


def fetch_parallel(name: str, owner_slug: str, dst: Path, total: int, workers: int) -> None:
    nchunks = (total + CHUNK - 1) // CHUNK
    done = load_done(name)
    pending = [i for i in range(nchunks) if i not in done]
    log(f"{name}: {total:,} B, {nchunks} chunks, {len(done)} already done, "
        f"{len(pending)} pending, {workers} workers")
    if not pending:
        log(f"{name}: complete")
        return

    ensure_size(dst, total)
    url = signed_url(owner_slug, name)

    state = {"bytes": 0, "t0": time.time(), "last_save": time.time()}
    state_lock = threading.Lock()
    stop = threading.Event()

    def progress() -> None:
        while not stop.is_set():
            time.sleep(10)
            with state_lock:
                elapsed = time.time() - state["t0"]
                # len(done)*CHUNK already IS the byte count - adding state["bytes"]
                # double-counts and pushes the readout past 100% (seen at 103%).
                # Completion is driven by len(done) == nchunks, so this was cosmetic,
                # but the ETA derived from it was wrong too.
                got = min(len(done) * CHUNK, total)
                rate = got / elapsed if elapsed > 0 else 0.0
                remain = (total - got) / rate if rate > 0 else float("inf")
                m, s = divmod(int(remain) if remain < 1e8 else 0, 60)
                h, m = divmod(m, 60)
                log(f"{name}: {100.0*got/total:5.1f}%  {got/1e9:6.2f}/{total/1e9:.2f} GB  "
                    f"avg {rate/1e6:5.2f} MB/s  ETA {h}h{m:02d}m{s:02d}s")

    tick = threading.Thread(target=progress, daemon=True)
    tick.start()

    def worker(i: int) -> None:
        fh = open(dst, "r+b")
        try:
            while True:
                with state_lock:
                    if not pending:
                        return
                    idx = pending.pop()
                start = idx * CHUNK
                end = min(start + CHUNK, total) - 1
                last_err = None
                for _ in range(6):
                    try:
                        r = requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                         timeout=300)
                        if r.status_code not in (200, 206):
                            raise RuntimeError(f"HTTP {r.status_code}")
                        data = r.content
                        if len(data) != end - start + 1:
                            raise RuntimeError(f"short read {len(data)}")
                        fh.seek(start)
                        fh.write(data)
                        fh.flush()
                        with state_lock:
                            done.add(idx)
                            state["bytes"] += len(data)
                            if time.time() - state["last_save"] > 30:
                                save_done(name, done)
                                state["last_save"] = time.time()
                        last_err = None
                        break
                    except Exception as e:  # noqa: BLE001
                        last_err = e
                        time.sleep(5)
                if last_err is not None:
                    with state_lock:
                        pending.append(idx)
                    log(f"{name}: chunk {idx} failed: {last_err}")
        finally:
            fh.close()

    try:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(worker, range(workers)))
    finally:
        stop.set()
        save_done(name, done)

    if len(done) != nchunks:
        raise SystemExit(f"{name}: incomplete ({len(done)}/{nchunks}) - rerun to resume")
    log(f"{name}: DONE")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    workers = 8
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    status_only = "--status" in sys.argv

    if status_only:
        for _, files in PARTS:
            for name, total in files.items():
                dst = DEST / name
                got = dst.stat().st_size if dst.exists() else 0
                done = len(load_done(name))
                nchunks = (total + CHUNK - 1) // CHUNK
                print(f"{name:20s} {got/1e9:8.3f}/{total/1e9:.3f} GB  "
                      f"chunks {done}/{nchunks}")
        return 0

    if LOCK.exists():
        pid = LOCK.read_text(encoding="utf-8").strip()
        try:
            import psutil
            if psutil.pid_exists(int(pid)):
                raise SystemExit(f"downloader pid {pid} already running")
        except ImportError:
            pass
    LOCK.write_text(str(os.getpid()), encoding="utf-8")

    try:
        for owner_slug, files in PARTS:
            for name, total in files.items():
                dst = DEST / name
                # File size is NOT proof of completeness: the file is preallocated to full
                # size up front, so a half-zeroed file still reports the right size. This
                # exact check declared a file "already complete" while 912 of its 1895
                # chunks were zeros. Completion must be judged by the CHUNK MANIFEST.
                nchunks = (total + CHUNK - 1) // CHUNK
                ndone = len(load_done(name))
                if (dst.exists() and dst.stat().st_size == total and total > 0
                        and ndone == nchunks):
                    log(f"{name}: already complete")
                    continue
                if dst.exists() and dst.stat().st_size == total and ndone and ndone < nchunks:
                    log(f"{name}: size ok but only {ndone}/{nchunks} chunks - resuming")
                if total <= SMALL:
                    fetch_whole(name, owner_slug, dst, total)
                else:
                    fetch_parallel(name, owner_slug, dst, total, workers)
    finally:
        try:
            LOCK.unlink()
        except OSError:
            pass
    log("ALL PARTS COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
