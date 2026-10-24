"""Parallel, resumable downloader for the public Raptor checkpoint arms.

Downloads the two public CoAtNet checkpoints as INDIVIDUAL files (not the dataset zip):
the kaggle CLI's zip route is single-stream (~0.1-0.5 MB/s here), while 8 parallel HTTP
Range streams sustain 2-8 MB/s on the same route (measured on the corpus fetch).

The 90 MB `raptor-knee-maxspan.zip` partial left by the kaggle CLI is NOT reusable:
zip bytes != raw .pt bytes, so this tool starts the .pt files fresh and the zip is
deleted once both checkpoints verify.

Reuses the design proven in download_raptor_corpus_par.py:
  - non-destructive ensure_size ("r+b", never "wb" on an existing file)
  - per-file JSON manifest of completed chunk indices for kill/restart resume
  - 8 MB chunks for fine retry granularity
plus one new behaviour: on HTTP 401/403 the signed URL is re-acquired inside the retry
loop (during the corpus fetch a ~2.5 min burst of 403s on chunks 464-473 was survived
only because the URL happened not to expire; refreshing it makes that deterministic).

Usage:
    python tools/download_pub_ckpt_par.py [--workers 8]
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
DEST = REPO / "artifacts" / "public_arms" / "ckpt"
TOKEN = Path.home() / ".kaggle" / "access_token"

CHUNK = 8 * 1024 * 1024

# (dataset slug, filename, exact remote size in bytes)
PARTS = [
    ("dreaddevelopment/raptor-knee-maxspan",
     "raptor_ft_coatnet_v5_full_swa.pt", 292_831_426),
    ("dreaddevelopment/raptor-knee-widedense",
     "raptor_ft_coatnet_v4_full.pt", 292_829_402),
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

    Never open an existing file in "wb" (that truncates to zero - the bug that
    destroyed 7.65 GB of corpus once). "r+b" preserves existing bytes.
    """
    if not dst.exists():
        with open(dst, "wb") as f:
            f.truncate(total)
        return
    cur = dst.stat().st_size
    if cur < total:
        with open(dst, "r+b") as f:
            f.truncate(total)


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
    url_holder = {"url": signed_url(owner_slug, name)}

    state = {"t0": time.time(), "last_save": time.time()}
    state_lock = threading.Lock()
    stop = threading.Event()

    def progress() -> None:
        while not stop.is_set():
            time.sleep(10)
            with state_lock:
                elapsed = time.time() - state["t0"]
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
                for attempt in range(8):
                    try:
                        r = requests.get(url_holder["url"],
                                         headers={"Range": f"bytes={start}-{end}"},
                                         timeout=300)
                        if r.status_code in (401, 403):
                            # Signed URL expired or was revoked: get a fresh one and
                            # retry this chunk immediately (do not count as a failure).
                            url_holder["url"] = signed_url(owner_slug, name)
                            raise RuntimeError(f"HTTP {r.status_code} (url refreshed)")
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
                            if time.time() - state["last_save"] > 30:
                                save_done(name, done)
                                state["last_save"] = time.time()
                        last_err = None
                        break
                    except Exception as e:  # noqa: BLE001
                        last_err = e
                        time.sleep(min(5 * (attempt + 1), 30))
                if last_err is not None:
                    with state_lock:
                        pending.append(idx)
                        log(f"{name}: chunk {idx} kept failing: {last_err}")
        finally:
            fh.close()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(worker, range(workers)))

    stop.set()
    save_done(name, done)
    if len(done) < nchunks:
        raise SystemExit(f"{name}: INCOMPLETE - {len(done)}/{nchunks} chunks. Re-run to resume.")
    log(f"{name}: DONE")


def verify(dst: Path, total: int) -> None:
    """A .pt file saved by torch>=1.6 is a zip container: PK magic + sane size."""
    size = dst.stat().st_size
    with open(dst, "rb") as f:
        magic = f.read(4)
    ok_magic = magic[:2] == b"PK"
    ok_size = size == total
    log(f"verify {dst.name}: size {size:,}/{total:,} "
        f"{'OK' if ok_size else 'MISMATCH'}; magic {magic!r} "
        f"{'OK' if ok_magic else 'BAD'}")
    if not (ok_magic and ok_size):
        raise SystemExit(f"{dst.name}: verification FAILED")


def main() -> None:
    workers = 8
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    if "--status" in sys.argv:
        for _slug, name, total in PARTS:
            dst = DEST / name
            d = load_done(name)
            have = dst.stat().st_size if dst.exists() else 0
            print(f"{name}: {len(d)}/{(total + CHUNK - 1)//CHUNK} chunks, "
                  f"file {have:,}/{total:,} B")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    for slug, name, total in PARTS:
        dst = DEST / name
        if dst.exists() and dst.stat().st_size == total and len(load_done(name)) == \
                (total + CHUNK - 1) // CHUNK:
            log(f"{name}: present")
            continue
        fetch_parallel(name, slug, dst, total, workers)
        verify(dst, total)
    log("ALL CHECKPOINTS COMPLETE")


if __name__ == "__main__":
    main()
