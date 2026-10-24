"""Resumable download of the public Raptor 336px corpus (both dataset parts).

Uses Kaggle API Bearer -> GCS signed URL + HTTP Range, so a dropped SSL
connection resumes from the current byte. Safe to kill and rerun.

    python tools/download_raptor_corpus.py
    python tools/download_raptor_corpus.py --status
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "artifacts" / "raptor_corpus"
LOCK = DEST / ".download.lock"
TOKEN = Path.home() / ".kaggle" / "access_token"
CHUNK = 32 * 1024 * 1024
SMALL = 256 * 1024 * 1024

PARTS = [
    ("dreaddevelopment/knee-raptor-corpus", [
        "all_ids.npy", "all_masks.npy", "window_index.npy", "all_vols.npy",
    ]),
    ("dreaddevelopment/knee-raptor-corpus-ext", [
        "extra_ids.npy", "extra_masks.npy", "extra_meta.json", "extra_vols.npy",
    ]),
]
KNOWN_SIZE = {
    "all_vols.npy": 15_895_756_928,
    "extra_vols.npy": 5_995_680_896,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auth_header() -> dict[str, str]:
    tok = TOKEN.read_text(encoding="utf-8").strip()
    if not tok:
        raise SystemExit(f"empty Kaggle token at {TOKEN}")
    return {"Authorization": f"Bearer {tok}"}


def signed_url(owner_slug: str, fname: str) -> str:
    owner, slug = owner_slug.split("/")
    api = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{slug}/{fname}"
    r = requests.get(api, headers=auth_header(), allow_redirects=False, timeout=60)
    loc = r.headers.get("Location")
    if r.status_code not in (301, 302, 303, 307, 308) or not loc:
        raise RuntimeError(f"no redirect for {owner_slug}/{fname}: HTTP {r.status_code}")
    return loc


def probe_total(url: str) -> tuple[int, bool]:
    last = None
    for attempt in range(12):
        try:
            r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
            if r.status_code == 206:
                return int(r.headers.get("Content-Range", "/0").split("/")[-1]), True
            if r.status_code == 200:
                return int(r.headers.get("Content-Length", 0)), False
            last = RuntimeError(f"probe HTTP {r.status_code}")
        except Exception as e:
            last = e
        time.sleep(min(30, 3 * (attempt + 1)))
    raise RuntimeError(last)


def fetch_whole(name: str, owner_slug: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(20):
        try:
            url = signed_url(owner_slug, name)
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            tmp.write_bytes(r.content)
            tmp.replace(dst)
            log(f"{name}: done ({len(r.content):,} B)")
            return
        except Exception as e:
            wait = min(60, 5 * (attempt + 1))
            log(f"{name}: attempt {attempt+1} failed ({e}); retry {wait}s")
            time.sleep(wait)
    raise SystemExit(f"{name}: giving up")


def fetch_resumable(name: str, owner_slug: str, dst: Path, total: int) -> None:
    import threading

    stop = threading.Event()

    def bar() -> None:
        last_b = dst.stat().st_size if dst.exists() else 0
        last_t = time.time()
        while not stop.is_set():
            time.sleep(1.0)
            try:
                cur = dst.stat().st_size
            except OSError:
                continue
            now = time.time()
            dt = max(now - last_t, 1e-6)
            speed = (cur - last_b) / dt
            last_b, last_t = cur, now
            pct = 100.0 * cur / total
            filled = min(40, int(pct // 2.5))
            eta = (total - cur) / speed if speed > 0 else float("inf")
            m, s = divmod(int(eta) if eta < 1e8 else 0, 60)
            print(f"\r{name}: [{'#'*filled}{'-'*(40-filled)}] {pct:5.1f}% "
                  f"{cur/1e9:6.2f}/{total/1e9:.2f} GB  {speed/1e6:5.1f} MB/s  ETA {m}m{s:02d}s   ",
                  end="", flush=True)

    tick = threading.Thread(target=bar, daemon=True)
    tick.start()
    try:
        while True:
            pos = dst.stat().st_size if dst.exists() else 0
            if pos >= total:
                break
            try:
                url = signed_url(owner_slug, name)
                end = min(pos + CHUNK, total) - 1
                r = requests.get(url, headers={"Range": f"bytes={pos}-{end}"}, timeout=300)
                if r.status_code == 206:
                    data = r.content
                    if len(data) != end - pos + 1:
                        raise RuntimeError(f"short read {len(data)}")
                    with open(dst, "ab") as f:
                        f.write(data)
                elif r.status_code == 200 and pos == 0:
                    with open(dst, "wb") as f:
                        f.write(r.content)
                    break
                else:
                    raise RuntimeError(f"HTTP {r.status_code} at pos={pos}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                stop.set()
                tick.join(timeout=2)
                print()
                pos_now = dst.stat().st_size if dst.exists() else 0
                log(f"{name}: {e}; pos={pos_now:,}; backoff 10s")
                time.sleep(10)
                stop = threading.Event()
                tick = threading.Thread(target=bar, daemon=True)
                tick.start()
    finally:
        stop.set()
        time.sleep(0.1)
        print()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    status_only = "--status" in sys.argv
    if LOCK.exists() and not status_only:
        pid = LOCK.read_text(encoding="utf-8").strip()
        alive = False
        try:
            import psutil
            alive = psutil.pid_exists(int(pid))
        except Exception:
            alive = False
        if alive:
            raise SystemExit(f"downloader pid {pid} already running")
        LOCK.unlink()
    if not status_only:
        LOCK.write_text(str(os.getpid()), encoding="utf-8")

    try:
        for owner_slug, files in PARTS:
            for name in files:
                dst = DEST / name
                if status_only:
                    sz = dst.stat().st_size if dst.exists() else 0
                    print(f"{name:20s} {sz/1e9:8.3f} GB")
                    continue
                if name.endswith((".json",)) or name in (
                    "all_ids.npy", "all_masks.npy", "window_index.npy",
                    "extra_ids.npy", "extra_masks.npy", "extra_meta.json",
                ):
                    if dst.exists() and dst.stat().st_size > 0:
                        log(f"{name}: present ({dst.stat().st_size:,} B)")
                        continue
                    fetch_whole(name, owner_slug, dst)
                    continue
                total = KNOWN_SIZE.get(name, 0)
                ranged = True
                if not total:
                    last_err = None
                    for attempt in range(12):
                        try:
                            url = signed_url(owner_slug, name)
                            total, ranged = probe_total(url)
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                            log(f"{name}: probe attempt {attempt+1} failed ({e})")
                            time.sleep(min(30, 3 * (attempt + 1)))
                    if last_err is not None:
                        raise last_err
                log(f"{name}: remote {total:,} B ranged={ranged}")
                if dst.exists() and dst.stat().st_size == total:
                    log(f"{name}: already complete")
                    continue
                if total <= SMALL or not ranged:
                    fetch_whole(name, owner_slug, dst)
                else:
                    fetch_resumable(name, owner_slug, dst, total)
    finally:
        if LOCK.exists() and not status_only:
            try:
                LOCK.unlink()
            except OSError:
                pass
    if not status_only:
        log("done")


if __name__ == "__main__":
    main()
