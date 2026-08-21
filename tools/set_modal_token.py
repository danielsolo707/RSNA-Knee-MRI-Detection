"""Write ~/.modal.toml from env or stdin. Never prints the secret.

Usage:
  set MODAL_TOKEN_SECRET=as-...
  python tools/set_modal_token.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ID_FILE = Path.home() / ".modal" / "token_id"
TOML = Path.home() / ".modal.toml"


def main() -> int:
    tid = os.environ.get("MODAL_TOKEN_ID", "").strip()
    if not tid and ID_FILE.is_file():
        tid = ID_FILE.read_text(encoding="utf-8").strip()
    sec = os.environ.get("MODAL_TOKEN_SECRET", "").strip()
    if not tid.startswith("ak-"):
        print("missing token id (ak-...). put it in ~/.modal/token_id", file=sys.stderr)
        return 2
    if not sec.startswith("as-"):
        print("set MODAL_TOKEN_SECRET to the as-... value from modal.com/settings/tokens", file=sys.stderr)
        return 2
    TOML.write_text(f'[default]\ntoken_id = "{tid}"\ntoken_secret = "{sec}"\n', encoding="utf-8")
    try:
        os.chmod(TOML, 0o600)
    except OSError:
        pass
    print(f"wrote {TOML} (token_id ends {tid[-4:]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
