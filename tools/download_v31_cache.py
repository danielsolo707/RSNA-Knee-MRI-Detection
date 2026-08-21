"""Resumable downloader for the private V31 cache kernel output.

Safe to kill and re-run at ANY time: progress = current byte size of each
file on disk. Uses signed Kaggle URLs + HTTP Range requests, so a dropped
connection continues from the exact byte where it stopped.

Usage:
    python tools/download_v31_cache.py            # download/resume + verify
    python tools/download_v31_cache.py --status   # just print progress
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "artifacts" / "v31_cache" / "v31_cache"
LOCK = REPO / "artifacts" / "v31_cache" / ".download.lock"
KERNEL = "danielsolo1770/rsna-knee-v31-cache-280"
CHUNK = 32 * 1024 * 1024          # 32 MB per ranged request
SMALL_LIMIT = 256 * 1024 * 1024   # files below this download whole


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_urls() -> dict[str, str]:
    # NOTE: do NOT set KAGGLE_API_TOKEN here - in kaggle>=2.2 it must contain
    # the token ITSELF, not a path. authenticate() finds ~/.kaggle/access_token.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    owner, slug = KERNEL.split("/")
    with api.build_kaggle_client() as client:
        from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = slug
        resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
    urls = {item.file_name: item.url for item in (resp.files or [])}
    if not urls:
        raise SystemExit("no output files returned - is the kernel COMPLETE?")
    return urls


def probe_total(url: str) -> tuple[int, bool]:
    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    if r.status_code == 206:
        cr = r.headers.get("Content-Range", "")
        return int(cr.split("/")[-1]), True
    if r.status_code == 200:
        return int(r.headers.get("Content-Length", 0)), False
    raise RuntimeError(f"probe failed: HTTP {r.status_code}")


def fetch_whole(name: str, url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(30):
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            tmp.write_bytes(r.content)
            tmp.replace(dst)
            log(f"{name}: done ({len(r.content):,} B)")
            return
        except Exception as e:
            wait = min(60, 5 * (attempt + 1))
            log(f"{name}: attempt {attempt + 1} failed ({e}); retry in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"{name}: giving up after 30 attempts")


def fetch_resumable(name: str, url: str, dst: Path, total: int) -> None:
    import threading

    stop = threading.Event()

    def fmt_eta(sec: float) -> str:
        if sec <= 0 or sec != sec or sec == float("inf"):
            return "--"
        m, s = divmod(int(sec), 60)
        return f"{m}m{s:02d}s"

    def bar_loop() -> None:
        last_bytes = dst.stat().st_size if dst.exists() else 0
        last_t = time.time()
        while not stop.is_set():
            time.sleep(1.0)
            try:
                cur = dst.stat().st_size
            except OSError:
                continue
            now = time.time()
            dt = max(now - last_t, 1e-6)
            speed = (cur - last_bytes) / dt
            last_bytes, last_t = cur, now
            pct = 100.0 * cur / total
            filled = int(pct // 2.5)
            bar = "#" * filled + "-" * (40 - filled)
            eta = (total - cur) / speed if speed > 0 else float("inf")
            print(f"\r{name}: [{bar}] {pct:5.1f}%  {cur/1e9:6.2f}/{total/1e9:.2f} GB"
                  f"  {speed/1e6:5.1f} MB/s  ETA {fmt_eta(eta)}   ", end="", flush=True)

    tick = threading.Thread(target=bar_loop, daemon=True)
    tick.start()
    try:
        while True:
            pos = dst.stat().st_size if dst.exists() else 0
            if pos >= total:
                break
            try:
                end = min(pos + CHUNK, total) - 1
                r = requests.get(url, headers={"Range": f"bytes={pos}-{end}"}, timeout=300)
                if r.status_code == 206:
                    data = r.content
                    if len(data) != end - pos + 1:
                        raise RuntimeError(f"short read {len(data)} != {end - pos + 1}")
                    with open(dst, "ab") as f:
                        f.write(data)
                elif r.status_code == 200 and pos == 0:
                    with open(dst, "wb") as f:
                        f.write(r.content)
                else:
                    raise RuntimeError(f"HTTP {r.status_code} at pos={pos}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                stop.set()
                tick.join(timeout=2)
                print()
                pos_now = dst.stat().st_size if dst.exists() else 0
                log(f"{name}: {e}; pos={pos_now:,}; backoff 10s (safe to close PC - re-run later)")
                time.sleep(10)
                tick = threading.Thread(target=bar_loop, daemon=True)
                stop = threading.Event()
                tick.start()
    finally:
        stop.set()
        time.sleep(0.1)
        final = dst.stat().st_size if dst.exists() else 0
        print(f"\r{name}: [{'#' * 40}] 100.0%  {final:,} B downloaded", flush=True)


def verify(man: dict) -> bool:
    ok = True
    n, shape = man["n"], man["x_shape"]
    checks = {
        "x_f16.dat": 2 * math.prod(int(d) for d in shape),
        "mask.dat": n * int(shape[1]),
    }
    for fname, want in checks.items():
        got = (DEST / fname).stat().st_size if (DEST / fname).exists() else -1
        good = got == want
        ok &= good
        log(f"verify {fname}: {'OK' if good else f'BAD got={got:,} want={want:,}'}")
    try:
        import numpy as np

        y = np.load(DEST / "y.npy")
        good = y.shape == (n, len(man["targets"])) and not np.isnan(y).any()
        ok &= bool(good)
        log(f"verify y.npy: {'OK' if good else 'BAD'} shape={y.shape}")
    except Exception as e:
        ok = False
        log(f"verify y.npy: FAILED ({e})")
    uids = (DEST / "uids.txt").read_text().splitlines()
    good = len(uids) == n and len(set(uids)) == n
    ok &= good
    log(f"verify uids.txt: {'OK' if good else 'BAD'} count={len(uids)}")
    return ok


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)

    # -- single-writer guard: the 2026-08-21 corruption came from two instances
    #    appending to the same file concurrently. Lock + liveness via psutil.
    status_only = "--status" in sys.argv
    if LOCK.exists():
        pid_text = LOCK.read_text().strip()
        alive = False
        try:
            import psutil

            alive = psutil.pid_exists(int(pid_text))
        except Exception:
            alive = False  # cannot check -> assume stale
        if alive and not status_only:
            raise SystemExit(
                f"another downloader (pid {pid_text}) is already running - refusing "
                "to start. Kill it first or delete artifacts/v31_cache/.download.lock"
            )
        if not alive:
            LOCK.unlink()
    if not status_only:
        LOCK.write_text(str(os.getpid()))

    want_x = 0
    man_path = DEST / "manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text())
        want_x = 2 * math.prod(int(d) for d in man["x_shape"])

    if status_only:
        have = (DEST / "x_f16.dat").stat().st_size if (DEST / "x_f16.dat").exists() else 0
        pct = 100.0 * have / want_x if want_x else 0.0
        bar = "#" * int(pct // 4)
        print(f"V31 cache: {have / 1e9:.2f}/{want_x / 1e9:.2f} GB ({pct:.1f}%) |{bar}|")
        return

    # quarantine wrong-sized leftovers automatically (corruption guard)
    x = DEST / "x_f16.dat"
    if x.exists() and want_x and x.stat().st_size != want_x:
        bad = DEST / "x_f16.dat.corrupt"
        sz = x.stat().st_size
        x.replace(bad)
        log(f"quarantined wrong-sized x_f16.dat ({sz:,} B, expected {want_x:,}) -> {bad.name}")

    urls = get_urls()
    log(f"{len(urls)} output files listed")

    man = None
    try:
        for name, url in sorted(urls.items()):
            dst = DEST / Path(name).name
            if name.endswith("manifest.json"):
                fetch_whole(name, url, dst)
                man = json.loads(dst.read_text())
                continue
            if dst.exists() and dst.stat().st_size > 0 and name != "v31_cache/x_f16.dat":
                log(f"{name}: already present ({dst.stat().st_size:,} B), skip")
                continue
            total, ranged = probe_total(url)
            log(f"{name}: remote {total:,} B ranged={ranged}")
            if total <= SMALL_LIMIT or not ranged:
                fetch_whole(name, url, dst)
            else:
                fetch_resumable(name, url, dst, total)
    finally:
        if LOCK.exists():
            try:
                LOCK.unlink()
            except OSError:
                pass

    assert man is not None, "manifest missing"
    if man.get("status") != "complete":
        raise SystemExit(f"kernel manifest says status={man.get('status')} - not complete!")
    log("ALL FILES DOWNLOADED - verifying...")
    ok = verify(man)
    log("VERDICT: " + ("ALL GREEN" if ok else "PROBLEMS FOUND"))


if __name__ == "__main__":
    main()
