"""Live public-leaderboard scan: our rank + score distribution.

Replaces the broken `kagglesdk.KaggleApi` import in _tmp_lb_scan.py.
Correct entry point is `kagglesdk.KaggleClient` -> competitions().get_leaderboard().

Usage:
    python tools/lb_scan.py [--max-pages 60] [--out artifacts/lb_scan/lb.json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest

COMP = "rsna-knee-abnormality-detection"
MY_TEAM = 16697840
ROOT = Path(__file__).resolve().parent.parent


def fetch(max_pages: int = 60):
    client = KaggleClient()
    rows: list[dict] = []
    token = None
    for _ in range(max_pages):
        req = ApiGetLeaderboardRequest()
        req.competition_name = COMP
        req.override_public = True
        req.page_size = 1000
        if token:
            req.page_token = token
        resp = client.competitions.competition_api_client.get_leaderboard(req)
        subs = resp.submissions or []
        if not subs:
            break
        for s in subs:
            rows.append(
                {
                    "team_id": s.team_id,
                    "team_name": s.team_name,
                    "score": float(s.score),
                    "date": str(s.submission_date),
                }
            )
        token = resp.next_page_token or None
        if not token:
            break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--out", default="artifacts/lb_scan/lb.json")
    args = ap.parse_args()

    rows = fetch(args.max_pages)
    print(f"rows fetched: {len(rows)}")

    mine = [(i, r) for i, r in enumerate(rows, 1) if r["team_id"] == MY_TEAM]
    print("MY ENTRIES:", mine)

    hist = Counter(round(r["score"], 3) for r in rows)
    print("score histogram (>= 0.925):")
    for k in sorted([k for k in hist if k >= 0.925], reverse=True):
        print(f"  {k:.3f}  n={hist[k]}")

    srt = sorted(rows, key=lambda r: -r["score"])
    if len(srt) >= 50:
        print(f"rank1={srt[0]['score']}  rank10={srt[9]['score']}  rank50={srt[49]['score']}")
    if mine:
        rank = mine[0][0]
        print(f"MY RANK: {rank} / {len(rows)}  score={mine[0][1]['score']}  date={mine[0][1]['date']}")
        better = sum(1 for r in srt if r["score"] > mine[0][1]["score"])
        print(f"teams strictly above us: {better}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
